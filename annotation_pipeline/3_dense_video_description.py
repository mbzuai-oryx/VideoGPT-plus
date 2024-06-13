"""
    Semi-automatic Video Annotation Pipeline - Step # 3: Use short ground truth caption along with the frame-level detailed captions to generate a detailed video caption using GPT4-Turbo.

    Copyright 2024 MBZUAI ORYX

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

import openai
import os
import json
import time
import argparse
import warnings
from tqdm import tqdm
from multiprocessing.pool import Pool

# Suppressing all warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Detailed video caption using GPT4-Turbo.")

    parser.add_argument("--ann_video_ids_file", required=True,
                        help="Path to the JSON file with unique video IDs (e.g. path to unique_video_ids.json).")
    parser.add_argument("--output_dir", required=False, help="Directory to save the annotation JSON files.",
                        default="video_descriptions")
    parser.add_argument("--captions_dir", required=False, help="Directory path containing generated video captions.",
                        default="llava_captions_keyframes")
    parser.add_argument("--gt_caption_file", required=True,
                        help="Path to the ground truth captions file (e.g. path to activitynet_gt_captions_train.json).")
    parser.add_argument("--api_keys", required=True, nargs='+', help="List of OpenAI API keys.")
    parser.add_argument("--num_tasks", type=int, default=16, help="Number of splits.")

    return parser.parse_args()


def get_caption_summary_prompt(gt_caption, predicted_captions):
    prompt_prefix_1 = "Generate a detailed and accurate description of a video based on the given ground-truth video caption and multiple frame-level captions. " \
                      "Use the following details to create a clear and complete narrative:\n"
    prompt_prefix_2 = "\nGround-truth Video Caption: "
    prompt_prefix_3 = "\nFrame-level Captions: "
    prompt_suffix = """\n\nInstructions for writing the detailed description:
    1. Focus on describing key visual details such as appearance, motion, sequence of actions, objects involved, and interactions between elements in the video.
    2. Check for consistency between the ground-truth caption and frame-level captions, and prioritize details that match the ground-truth caption. Ignore any conflicting or irrelevant details from the frame-level captions.
    3. Leave out any descriptions about the atmosphere, mood, style, aesthetics, proficiency, or emotional tone of the video.
    4. Make sure the description is no more than 20 sentences.
    5. Combine and organize information from all captions into one clear and detailed description, removing any repeated or conflicting details.
    6. Emphasize important points like the order of events, appearance and actions of people or objects, and any significant changes or movements.
    7. Do not mention that the information comes from ground-truth captions or frame-level captions.
    8. Give a brief yet thorough description, highlighting the key visual and temporal details while keeping it clear and easy to understand.
    Use your intelligence to combine and refine the captions into a brief yet informative description of the entire video."""

    # Create the prompt by iterating over the list_of_elements and formatting the template
    prompt = prompt_prefix_1
    prompt += f"{prompt_prefix_2}{gt_caption}{prompt_prefix_3}{'; '.join(predicted_captions)}"
    prompt += prompt_suffix

    return prompt


def annotate(gt_file, caption_files, output_dir, captions_dir, api_key):
    """
    Generate question-answer pairs using caption and
    dense-captions summarized from off-the-shelf models using OpenAI GPT-3.
    """
    openai.api_key = api_key  # Set the OpenAI API key for this process

    for file in tqdm(caption_files):
        annotated_dit = {}
        key = file.split('.')[0]
        gt_caption = get_gt_caption(gt_file, key)

        # Get pre-computed off-the-shelf predictions
        prediction_captions = get_pseudo_caption(captions_dir, key)

        # Summarize pre-computed off-the-shelf predictions into dense caption
        summary_prompt = get_caption_summary_prompt(gt_caption, prediction_captions)

        dense_caption_summary = openai.ChatCompletion.create(
            model="gpt-4-turbo", messages=[{"role": "user", "content": summary_prompt}]
        )
        dense_caption = ''
        for choice in dense_caption_summary.choices:
            dense_caption += choice.message.content

        annotated_dit['dense_caption'] = dense_caption

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(output_dir, f"{key}.json")
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(annotated_dit, f, ensure_ascii=False, indent=2)

    print(f"Completed, Annotations saved in {output_dir}")


def get_gt_caption(json_data, video_id):
    video_data = json_data[video_id]
    gt_captions = video_data['sentences']
    gt_caption = ''.join(gt_captions)
    return gt_caption


def get_pseudo_caption(pseudo_data_dir, video_id):
    curr_files = [file for file in os.listdir(pseudo_data_dir) if file.startswith(video_id)]
    pred_captions = []
    for file in curr_files:
        pred_caption = json.load(open(f'{pseudo_data_dir}/{file}'))['result']
        pred_captions.append(pred_caption)
    return pred_captions


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.ann_video_ids_file, 'r') as file:
        data = json.load(file)
        video_ids_to_annotate = data['v2_videos']

    # Read ground truth captions file
    gt_file = args.gt_caption_file
    with open(gt_file) as file:
        gt_json_data = json.load(file)

    video_ids_to_annotate = [id for id in video_ids_to_annotate if id in gt_json_data]

    # Prepare list of caption files
    caption_files = [f'{video_id}.json' for video_id in video_ids_to_annotate]

    # List of OpenAI API keys
    api_keys = args.api_keys

    num_tasks = args.num_tasks

    # Main loop: Continues until all question-answer pairs are generated for all captions
    while True:
        try:
            # Files that have already been completed.
            completed_files = os.listdir(args.output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                print("All tasks completed!")
                break

            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            num_tasks = min(len(incomplete_files), num_tasks)
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]

            # Distribute API keys to tasks
            task_args = [(gt_json_data, part, args.output_dir, args.captions_dir, api_keys[i % len(api_keys)]) for
                         i, part
                         in enumerate(all_parts)]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 1 minute...")
            time.sleep(60)  # wait for 1 minute before trying again


if __name__ == "__main__":
    main()
