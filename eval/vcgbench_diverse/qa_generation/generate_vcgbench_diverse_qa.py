"""
    VCGBench-Diverse - Script to Generate VCGBench-Diverse QA from video descriptions using GPT4o

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

    USAGE:
    python prepare_vcgbench_diverse_qa.py --gt_description_path /path/to/vcgbench_diverse_human_annotated_descriptions.json --output_dir /path/to/output  --api_key_list sk-key1 sk-key2 sk-key3 --num_tasks 16

"""

import time
import openai
import os
import json
import ast
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
    parser = argparse.ArgumentParser(
        description="Script to Generate VCGBench-Diverse QA from video descriptions using GPT4o.")
    parser.add_argument("--gt_description_path", required=True,
                        help="Path to gt human annotated video descriptions (i.e. vcgbench_diverse_human_annotated_descriptions.json).")
    parser.add_argument("--output_dir", required=True, help="Path to save the annotation JSON files.")
    parser.add_argument("--api_key_list", required=True, nargs='+', help="List of OpenAI API keys.")
    parser.add_argument("--num_tasks", required=False, type=int, help="Number of splits.", default=16)

    return parser.parse_args()


def get_summary_qa_prompt(dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and answers about video content to create a video instruction tuning dataset. "
        "Your goal is to extract detailed visual and temporal information from the video, ensuring the explanations are comprehensive enough for someone to understand the entire sequence of events in the video."
        "##TASK:"
        "1. Users provide a video description."
        "2. Generate ONE question that effectively prompt a detailed description of the entire video content and sequence of events."
        "------"
        "##INSTRUCTIONS:"
        "- Ensure the question targets the goal of generating a detailed description of the entire video from start to end."
        "- Avoid questions that focus on small parts, less relevant details, or abstract concepts such as logical reasoning, attention to subtle details, overall aesthetic."
        "- The answer must include all the details from the provided video description."
        "- Focus on visual and temporal details."
        "##SAMPLE QUESTIONS:"
        "- Can you describe the entire video in detail from start to finish?"
        "- What happens throughout the entire video, including all key actions and events?"
        "- Could you provide a detailed walkthrough of the entire video?"
    )

    user_prompt = (
        f"The video description is: {dense_caption}. "
        "Generate ONE question and answer about the entire content and sequence of events in the video. "
        "The question should aim to elicit a comprehensive description of the full sequence of events in the video from start to finish. "
        "The answer must include all the details from the provided video description. "
        "Format the output as a dictionary in JSON style containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "{'Q': 'Your question here...', 'A': 'Your answer here...'}. "
        "Most importantly, the answer must provide a full understanding of the video by incorporating ALL the details from the provided video description."
    )

    return system_prompt, user_prompt


def get_combined_qa_prompt(dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers based on a video description. "
        "Your goal is to extract important information from the video content, focusing on temporal events, visual details, and reasoning behind actions."
        "##TASK:"
        "You will receive a video description, and based on it, you must generate a set of questions and answers in three distinct categories: "
        "1. Temporal - These questions should focus on the sequence and timing of events. Use approximate time references where necessary."
        "2. Spatial - These questions should address visual aspects such as appearance, objects, colors, attire, displayed texts, number of objects or people, location, and other significant visual details."
        "3. Reasoning - These questions should delve into the actions, motivations, and consequences as depicted in the video description."
        "##INSTRUCTIONS:"
        "- Each question must directly relate to and be answerable by the provided video description. Avoid assumptions and fabrication of details not present in the description."
        "- Provide clear, unambiguous questions that allow for definitive answers based on the description."
        "- If the video description does not contain enough information to formulate a question in any category, do not include a question for that category."
        "##SAMPLE QUESTIONS:"
        "- Temporal: Describe the entire process the person goes through from start to finish or What happens at the beginning of the video? or What does the person do right after the dog appears?"
        "- Spatial: Can you provide a detailed description of the appearance and activities of all individuals or What is the color of the main character’s shirt? or What is the name of the drink on the bottle? How many people are at the table?"
        "- Reasoning: What action does the coach take after the whistle blows? or Why did the player throw the ball? or Who is John Davis in the video?"
    )

    user_prompt = (
        f"The video description is: {dense_caption}. "
        "Format the output as a dictionary in JSON style, with each key representing a question category and containing a sub-dictionary with 'Q' for the question and 'A' for the answer. "
        "Example output with all three categories filled: "
        "{'temporal': {'Q': 'Temporal question here...', 'A': 'Answer here...'}, "
        "'spatial': {'Q': 'Spatial question here...', 'A': 'Answer here...'}, "
        "'reasoning': {'Q': 'Reasoning question here...', 'A': 'Answer here...'}}. "
        "If a category cannot be filled: "
        "{'temporal': {'Q': 'Describe the sequence of events in the video.', 'A': 'The video starts with...'}, "
        "'spatial': {'Q': 'What is the main character wearing?', 'A': 'The main character is dressed in...'}}"  # reasoning omitted due to lack of information
        "Importantly, the answers MUST extract information DIRECTLY from the given description. Do not include categories that cannot be filled based on the video description alone.")

    return system_prompt, user_prompt


def annotate(caption_files, curr_output_dir, pred_data, api_key):
    """
    Generate question-answer pairs using caption and
    dense-captions summarized from off-the-shelf models using OpenAI GPT-3.
    """
    openai.api_key = api_key  # Set the OpenAI API key for this process
    summary_qa_pairs = True
    combined_qa_pairs = True

    for file in tqdm(caption_files):
        annotated_dict = {}
        key = file.split('.')[0]
        detailed_description = get_video_description(pred_data, key)

        if summary_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3: Summarization
            system_prompt, user_prompt = get_summary_qa_prompt(detailed_description)
            completion_0 = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_0 = completion_0["choices"][0]["message"]["content"]
            response_dict_0 = ast.literal_eval(response_message_0)

            annotated_dict['summary'] = response_dict_0

        if combined_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3
            system_prompt, user_prompt = get_combined_qa_prompt(detailed_description)
            completion_1 = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_1 = completion_1["choices"][0]["message"]["content"]
            response_dict_1 = ast.literal_eval(response_message_1)
            if 'temporal' in response_dict_1:
                annotated_dict['temporal'] = response_dict_1['temporal']
            if 'spatial' in response_dict_1:
                annotated_dict['spatial'] = response_dict_1['spatial']
            if 'reasoning' in response_dict_1:
                annotated_dict['reasoning'] = response_dict_1['reasoning']

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(curr_output_dir, f"{key}.json")
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(annotated_dict, f, ensure_ascii=False, indent=4)

    print(f"Completed, Annotations saved in {curr_output_dir}")


def get_video_description(pred_data, video_id):
    file_name = video_id.split('.')[0]
    dense_caption = pred_data[file_name]['description']
    return dense_caption


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_data = json.load(open(args.gt_description_path))
    video_ids_to_annotate = [ann['id'] for ann in pred_data]
    pred_data = {ann['id']: ann for ann in pred_data}

    # Prepare list of caption files
    caption_files = [f'{video_id}.json' for video_id in video_ids_to_annotate]

    # List of OpenAI API keys
    api_keys = args.api_key_list

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
            task_args = [(part, args.output_dir, pred_data, api_keys[i % len(api_keys)]) for i, part
                         in enumerate(all_parts)]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 1 minute...")
            time.sleep(60)  # Wait for 1 minute before trying again


if __name__ == "__main__":
    main()
