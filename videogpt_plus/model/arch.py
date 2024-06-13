from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_vision_tower
from videogpt_plus.constants import *
from .multimodal_projector.builder import build_vision_projector
from einops import rearrange
import math
import torch.nn.functional as F


class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True, image_vision_tower=False)
            self.image_vision_tower = build_vision_tower(config, delay_load=True, image_vision_tower=True)
            self.mm_projector = build_vision_projector(config, image_mm_projector=False)
            self.image_mm_projector = build_vision_projector(config, image_mm_projector=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_image_vision_tower(self):
        image_vision_tower = getattr(self, 'image_vision_tower', None)
        if type(image_vision_tower) is list:
            image_vision_tower = image_vision_tower[0]
        return image_vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        image_vision_tower = model_args.image_vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_image_mm_mlp_adapter = model_args.pretrain_image_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_mm_vision_tower = image_vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.image_mm_projector_type = getattr(model_args, 'image_mm_projector_type', 'linear')

        if model_args.vision_tower is not None:
            vision_tower = build_vision_tower(model_args, image_vision_tower=False)
            self.config.mm_hidden_size = vision_tower.hidden_size
            if not hasattr(self, 'mm_projector'):
                self.mm_projector = build_vision_projector(self.config, image_mm_projector=False)
        if model_args.image_vision_tower is not None:
            image_vision_tower = build_vision_tower(model_args, image_vision_tower=True)
            self.config.image_mm_hidden_size = image_vision_tower.hidden_size
            if not hasattr(self, 'image_mm_projector'):
                self.image_mm_projector = build_vision_projector(self.config, image_mm_projector=True)

        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
            self.image_vision_tower = [image_vision_tower]
        else:
            self.vision_tower = vision_tower
            self.image_vision_tower = image_vision_tower

        if pretrain_mm_mlp_adapter is not None:
            print(f"Initializing projector from {pretrain_mm_mlp_adapter}")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if pretrain_image_mm_mlp_adapter is not None:
            print(f"Initializing projector from {pretrain_image_mm_mlp_adapter}")
            mm_projector_weights = torch.load(pretrain_image_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.image_mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def apply_adaptive_avg_pooling(x, shape=(12, 12)):
    b, num_tokens, c = x.shape
    h = int(math.sqrt(num_tokens))
    assert h * h == num_tokens
    x = x.permute(0, 2, 1).reshape(b, -1, h, h)
    x = F.adaptive_avg_pool2d(x, shape)
    x = x.flatten(2).transpose(1, 2)

    return x


class VideoGPTPlusMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_image_vision_tower(self):
        return self.get_model().get_image_vision_tower()

    def encode_images(self, images):
        image_encoder = self.get_model().get_image_vision_tower()
        video_encoder = self.get_model().get_vision_tower()
        if image_encoder is not None:
            image_features = image_encoder(images, select_feature="patch")
        elif video_encoder is not None:
            image_features = video_encoder(images.unsqueeze(1))  # Adds time dimension (B, T, C, H, W)
            image_features = image_features[:, 1:]

        return image_features

    def encode_videos(self, frames, context_images, batch_size):
        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)
        num_chunks = frames.shape[1] // CHUNK_SIZE
        L = 256  # Number of features per frame from InternVideo2-Stage2_1B-224p-f4
        D = 1408  # Feature dimension of InternVideo2-Stage2_1B-224p-f4

        video_features = torch.zeros(batch_size, num_chunks, 4 * L, D, device=frames.device, dtype=frames.dtype)
        for i in range(batch_size):
            cur_video = frames[i]  # Current video of shape (t, c, h, w)
            chunks = cur_video.chunk(num_chunks, dim=0)
            # New batch dimension for processing all chunks at once
            chunk_batch = torch.stack(chunks, dim=0)  # (num_chunks, 4, c, h, w)
            chunk_features = self.get_model().get_vision_tower()(chunk_batch)  # (num_chunks, 4*L, D)
            # Store the features in the output tensor - Only storing feature - remove cls
            video_features[i] = chunk_features[:, 1:]

        video_features = rearrange(video_features, 'b p (c l) d -> (b p) (c l) d', c=CHUNK_SIZE)
        context_image_features = self.get_model().get_image_vision_tower()(context_images, select_feature="patch")
        context_image_features = rearrange(context_image_features, '(b t) l d -> b t l d', b=batch_size)

        return video_features, context_image_features

    def positional_encoding(self, x, num_features=1024, max_len=64):
        p = torch.zeros((1, max_len, num_features))
        _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_features, 2, dtype=torch.float32) / num_features
        )

        p[:, :, 0::2] = torch.sin(_x)
        p[:, :, 1::2] = torch.cos(_x)
        x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
        return x

    def project(self, video_features, context_features=None, input_type="image"):
        if input_type == "video":
            video_features = self.get_model().mm_projector(video_features)
            video_features = rearrange(video_features, 'b (t l) d -> (b t) l d', t=4)  # t=4 - chunk size
            video_features = apply_adaptive_avg_pooling(video_features, shape=(8, 8))  # Feature pooling from 256 to 64
            video_features = rearrange(video_features, '(b t) l d -> b (t l) d', t=4)  # t=4 - chunk size

            context_image_features = self.get_model().image_mm_projector(context_features)
            context_image_features = apply_adaptive_avg_pooling(context_image_features,
                                                                shape=(12, 12))  # Feature pooling from 576 to 144
            context_image_features = rearrange(context_image_features, '(b t) l d -> b (t l) d',
                                               b=video_features.shape[0])

            merged_features = []
            for i in range(context_image_features.shape[0]):
                merged_features.append(context_image_features[i])

            for i in range(video_features.shape[0]):
                merged_features.append(video_features[i])

            merged_features = torch.cat(merged_features, dim=0).unsqueeze(0)

            return merged_features

        image_encoder = self.get_model().get_image_vision_tower()
        video_encoder = self.get_model().get_vision_tower()

        if image_encoder is not None:
            context_features = self.get_model().image_mm_projector(context_features)
        elif video_encoder is not None:
            context_features = self.get_model().mm_projector(context_features)
        else:
            raise NotImplementedError("Either image_encoder or video_encoder should not be None.")

        return context_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images,
                                             context_images):
        vision_tower = self.get_vision_tower()
        image_vision_tower = self.get_image_vision_tower()
        if (vision_tower is None and image_vision_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if images is not None and context_images is not None:
            video_features, context_features = self.encode_videos(images, context_images, batch_size=input_ids.shape[0])
        elif images is not None:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # Multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if len(image_token_indices) > 1:  # This is a video
                temp = []
                cur, pre = image_token_indices[0], image_token_indices[0]
                for i in image_token_indices:
                    cur = i
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur

                for i in temp:
                    image_token_start = image_token_indices[0]
                    image_token_end = image_token_indices[-1]
                    cur_image_features = []

                    for _ in range(len(i) // CHUNK_SIZE):
                        cur_image_features.append(video_features[cur_image_idx])
                        cur_image_idx += 1

                    if len(i) > 2:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="video")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)
                    else:
                        # This is video but only 1 frame is sampled
                        # This will not happen as video encoder needs at least 4 frames
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                            self.config, 'mm_use_im_start_end', False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                    dtype=labels.dtype
                                )
                            )
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                            cur_labels = cur_labels[image_token_end + 2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                    dtype=labels.dtype
                                )
                            )
                            cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            elif image_token_indices.numel() > 0:  # This is an image
                cur_image_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]

                for _ in image_token_indices:
                    cur_image_features.append(image_features[cur_image_idx])
                    cur_image_idx += 1

                cur_image_features = torch.stack(cur_image_features, dim=0)
                cur_image_features = self.project(video_features=None, context_features=cur_image_features, input_type="image")
                t, l, n = cur_image_features.size()
                cur_image_features = cur_image_features.contiguous().view(t * l, n)

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                        cur_labels = cur_labels[image_token_end + 2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (cur_new_embed, torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device
                    )), dim=0
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (cur_new_label, torch.full(
                            (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype,
                            device=cur_new_label.device
                        )), dim=0
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                        attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN],
                special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                print(f"Initializing projector from {model_args.pretrain_mm_mlp_adapter}")
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
