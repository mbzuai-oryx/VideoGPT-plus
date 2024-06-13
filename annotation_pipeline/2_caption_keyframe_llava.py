"""
    Semi-automatic Video Annotation Pipeline - Step # 2: Frame level detailed captioning using LLaVA-v1.6-34b

    Copyright 2024 MBZUAI ORYX
    Copyright 2024 LLaVA https://github.com/haotian-liu/LLaVA

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import argparse
import torch
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER, )
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests
import json
import re
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--key_frame_dir", type=str, required=False, help="Directory containing extracted keyframes.",
                        default="key_frames")
    parser.add_argument("--output_dir", type=str, required=False, default='llava_captions_keyframes',
                        help="Directory to save output files.")
    parser.add_argument("--question", type=str, default="Describe the image in detail.",
                        help="Question to ask about the image.")

    parser.add_argument("--model-path", type=str, required=False, help="Path to the pretrained model.",
                        default="liuhaotian/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None, help="Base model to use.")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode.")
    parser.add_argument("--sep", type=str, default=",", help="Separator.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")

    return parser.parse_args()


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    if "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"

    return model, image_processor, tokenizer, conv_mode


def prepare_conv(qs, model, tokenizer, conv_mode):
    conv = conv_templates[conv_mode].copy()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    return input_ids


def inference(image_files, input_ids, model, image_processor, tokenizer, args):
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def main(args):
    key_frame_dir = args.key_frame_dir
    key_frame_files = os.listdir(key_frame_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model, image_processor, tokenizer, conv_mode = load_model(args)

    question = args.question

    input_ids = prepare_conv(question, model, tokenizer, conv_mode)

    for file in tqdm(key_frame_files):
        file_name = file.split('.')[0]
        output_path = os.path.join(output_dir, f'{file_name}.json')
        if not os.path.exists(output_path):
            image_path = os.path.join(key_frame_dir, file)
            image_files = [image_path]
            result = inference(image_files, input_ids, model, image_processor, tokenizer, args)

            result_dict = {'result': result}
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
