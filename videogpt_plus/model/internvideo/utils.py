"""
    Modified from https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2

    Copyright 2024 OpenGVLab/InternVideo
    Copyright 2024 MBZUAI ORYX
"""

import numpy as np
import cv2
import os
import torch
from torch import nn
from videogpt_plus.model.internvideo.internvideo2 import pretrain_internvideo2_1b_patch14_224
from videogpt_plus.model.internvideo.pos_embed import interpolate_pos_embed_internvideo2_new

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert (len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_vision(frames, model, topk: int = 5, config: dict = {}, device=torch.device('cuda')):
    vlm = model
    vlm = vlm.to(device)

    fn = config.get('num_frames', 8)
    size_t = config.get('size_t', 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
    vision_embeds, pooled_vision_embeds = vlm.get_vid_feat(frames_tensor)

    return vision_embeds, pooled_vision_embeds


def setup_internvideo2V(config: dict):
    model = InternVideo2_Stage2V(config=config, is_pretrain=True)

    if config.get('compile_model', False):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model = model.to(torch.device(config.device))
    model_without_ddp = model

    if (config.pretrained_path.strip() and (
            os.path.isfile(config.pretrained_path)) or "s3://" in config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        try:
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"]  # This is a deepspeed stage 1 model
        except:
            state_dict = checkpoint

        if config.get('origin_num_frames', None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict, model_without_ddp.vision_encoder, orig_t_size=config.origin_num_frames
            )
            assert a == len(state_dict), state_dict.keys()

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"load_state_dict: {msg}")

    if config.get('use_bf16', False):
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.get('use_half_precision', False):
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return (model_without_ddp)


class VideoTrainProcessor():
    def __init__(self, image_size=(224, 224), mean=None, std=None, num_frames=8):
        super().__init__()

        if mean is None:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        if std is None:
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        self.mean = mean
        self.std = std
        self.num_frames = num_frames

    def normalize(self, data):
        return (data / 255.0 - self.mean) / self.std

    def frames2tensor(self, vid_list, target_size=(224, 224), use_image=False):
        # Ensure we have at least `self.num_frames`
        if not use_image:
            assert (len(vid_list) >= self.num_frames)

        # Process each frame
        vid_list = [cv2.resize(x, target_size) for x in vid_list]
        vid_tube = [normalize(x) for x in vid_list]
        vid_tube = [np.transpose(x, (2, 0, 1)) for x in vid_tube]
        vid_tube = [torch.from_numpy(x) for x in vid_tube]

        return vid_tube

    def preprocess(self, vid_list, use_image=False):
        return {'pixel_values': self.frames2tensor(vid_list, use_image=use_image)}


class InternVideo2_Stage2V(nn.Module):
    """docstring for InternVideo2_Stage2"""

    def __init__(self, config, is_pretrain: bool = True):
        super(InternVideo2_Stage2V, self).__init__()

        self.config = config

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)

        self.image_processor = VideoTrainProcessor(num_frames=self.vision_encoder.num_frames)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint["module"]  # This is a deepspeed stage 1 model
        if self.config.get('origin_num_frames', None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict, self.vision_encoder, orig_t_size=self.config.origin_num_frames
            )
            assert a == len(state_dict), state_dict.keys()
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"load_state_dict: {msg}")

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self, image: torch.Tensor, test: bool = False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                image, mask, use_image
            )
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def forward(self, image):
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds = self.vision_encoder(
            image, None, use_image, x_vis_return_idx=-2, x_vis_only=True
        )

        return vision_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name

        if encoder_name == 'pretrain_internvideo2_1b_patch14_224':
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            vision_embeds, pooled_vision_embeds = self.encode_vision(
                frames, test=True
            )  # vfeat = self.vision_proj(vfeat)  # vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vision_embeds, pooled_vision_embeds

    @property
    def hidden_size(self):
        return self.vision_encoder.embed_dim

    @property
    def num_patches(self):
        return self.config.model.vision_encoder.patch_size
