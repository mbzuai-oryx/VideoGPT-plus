import argparse
from tqdm import tqdm
import shortuuid
from videogpt_plus.conversation import conv_templates
from videogpt_plus.model.builder import load_pretrained_model
from videogpt_plus.mm_utils import tokenizer_image_token, get_model_name_from_path
from eval.mvbench.inference.ddp import *
from torch.utils.data import DataLoader, DistributedSampler
import traceback


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)


mvbench_data_list = {
    "Episodic Reasoning": ("episodic_reasoning.json", "your_data_path/tvqa/frames_fps3_hq/", "frame", True),
    "Action Sequence": ("action_sequence.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "your_data_path/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "your_data_path/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "your_data_path/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "your_data_path/star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "your_data_path/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "your_data_path/sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "your_data_path/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "your_data_path/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "your_data_path/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "your_data_path/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "your_data_path/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "your_data_path/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "your_data_path/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "your_data_path/vlnqa/", "video", False),
    "Counterfactual Inference": (
        "counterfactual_inference.json", "your_data_path/clevrer/video_validation/", "video", False),
}


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    model = model.to("cuda")

    dataset = EvalDatasetMvBench(args.question_dir, args.video_folder, image_processor,
                                 video_processor, mvbench_data_list)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=4, sampler=distributed_sampler)

    for (idx, sample_set, video_frames, context_frames, slice_len) in tqdm(dataloader):
        idx, sample_set, video_frames, context_frames, slice_len = int(idx[0]), sample_set[
            0], video_frames, context_frames, int(slice_len[0])

        sample = sample_set
        qs = sample['Q'][0]

        try:
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop_str = "<|end|>"

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=torch.cat(video_frames, dim=0).half().cuda(),
                    context_images=torch.cat(context_frames, dim=0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("<|end|>", '')
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            video_json_name = sample['video_name'][0].replace('/', '_')
            if len(video_json_name) > 100:
                video_json_name = video_json_name[50:]

            results = {'video_name': sample['video_name'][0],
                       "prompt": cur_prompt,
                       "pred": outputs,
                       "answer_id": ans_id,
                       "Q": sample_set['Q'][0],
                       "task_type": sample['task_type'][0],
                       "A": sample['A'][0]}
            with open(f"{args.output_dir}/{video_json_name}_{idx}.json", "w") as f:
                json.dump(results, f)
        except Exception as e:
            trace = traceback.format_exc()
            print(f"Error processing video file '{sample['video_name'][0]}': {e}")
            print("Detailed traceback:")
            print(trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench")
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--video-folder", type=str, default="OpenGVLab/MVBench/video")
    parser.add_argument("--question-dir", type=str, default="OpenGVLab/MVBench/json")
    parser.add_argument("--output-dir", type=str, default="MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench_eval")
    parser.add_argument("--conv-mode", type=str, default="phi3_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)

    eval_model(args)
