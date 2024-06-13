"""
    Semi-automatic Video Annotation Pipeline - Step # 4: Generate QA pairs using video descriptions generated in Step # 3 using GPT-3.5-Turbo.

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
    parser = argparse.ArgumentParser(description="Generate QA pairs using video descriptions generated in Step # 3")

    parser.add_argument("--ann_video_ids_file", required=True,
                        help="Path to the JSON file with unique video IDs (e.g. path to unique_video_ids.json).")
    parser.add_argument("--output_dir", required=False, help="Directory to save the annotation JSON files.",
                        default="video_qa")
    parser.add_argument("--video_descriptions_path", required=False,
                        help="Directory containing the generated video descriptions.", default="video_descriptions")
    parser.add_argument("--gt_caption_file", required=True,
                        help="Path to the ground truth captions file (e.g. path to activitynet_gt_captions_train.json).")
    parser.add_argument("--api_keys", required=True, nargs='+', help="List of OpenAI API keys.")
    parser.add_argument("--num_tasks", type=int, default=32, help="Number of splits.")

    return parser.parse_args()


def get_summary_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and answers about video content to create a video instruction tuning dataset. "
        "Your goal is to extract detailed visual and temporal information from the video, ensuring the explanations are comprehensive enough for someone to understand the entire sequence of events in the video."
        "##TASK:"
        "1. Users provide a video ground truth caption and a detailed description."
        "2. Generate three questions that effectively prompt a detailed description of the entire video content and sequence of events."
        "------"
        "##INSTRUCTIONS:"
        "- Ensure each question targets the goal of generating a detailed description of the entire video from start to end."
        "- Avoid questions that focus on small parts, less relevant details, or abstract concepts such as logical reasoning, attention to subtle details, overall aesthetic."
        "- Every answer must include all the details from the ground truth caption and integrate additional specifics from the detailed description."
        "- Focus on visual and temporal details."
        "##SAMPLE QUESTIONS:"
        "- Can you describe the entire video in detail from start to finish?"
        "- What happens throughout the entire video, including all key actions and events?"
        "- Could you provide a detailed walkthrough of the entire video?"
    )

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        f"The noisy detailed description is: {dense_caption}. "
        "Generate three questions and answers about the entire content and sequence of events in the video. "
        "Each question should aim to elicit a comprehensive description of the full sequence of events in the video from start to finish. "
        "Each answer must include all the details from the ground truth caption and integrate additional specifics from the detailed description. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, every answer must provide a full understanding of the video by incorporating ALL the details from the ground truth caption and additional specifics from the detailed description."
    )

    return system_prompt, user_prompt


def get_generic_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers based on video descriptions. "
        "Your goal is to extract important information from the video content, ensuring the questions focus on significant aspects and the answers are comprehensive and detailed."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract information directly from the given information, so that the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and detailed answers based on the given information. Each question should focus on a different aspect such as appearance, motion, trajectory, and reasoning."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant aspects."
        "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers."
        "- The answers must be detailed and descriptive."
        "- The answers must include details about the setting, objects involved, and any specific techniques or methods used."
        "- Each question should focus on a different key aspect such as appearance, motion, trajectory, and reasoning."
        "- Avoid asking about irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- Describe the entire process the person goes through from start to finish."
        "- Can you provide a detailed description of the appearance and activities of all individuals."
        "- Explain how the main activity in the video is performed step by step."
        "- What are the different stages of the activity shown in the video, and how does the person's approach change at each stage?"
        "- Outline the key moments and interactions between people, objects, and their environment.")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        f"The detailed noisy description is: {dense_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, the question should focus on a different key aspect such as appearance, action, trajectory, and reasoning."
    )

    return system_prompt, user_prompt


def get_temporal_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers related to the temporal events in a video. "
        "Your goal is to help users understand the sequence and timing of events in the video by asking and answering questions that focus on when events occur."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description  generated from ordered frames of the video in the correct order of events. "
        "You will generate a set of questions and answers related to the events in the video using approximate time references, by closely analyzing the sequence of sentences in the provided information. "
        "Generate THREE different descriptive questions and detailed answers based on the caption and detailed description."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant temporal aspects."
        "- Use approximate time references such as the beginning, middle, and end."
        "- The answers must be based on the information provided in the caption and detailed description."
        "- The answers must be detailed and descriptive."
        "- Do not explicitly mention in the answers that it is based on the caption or frames."
        "##SAMPLE QUESTIONS:"
        "- When does the main character start the primary task, and what leads up to it?"
        "- What actions occur after the initial setup, and how do they progress towards the climax?"
        "- What significant events happen midway, and how do they transition from earlier to later scenes?"
        "- Can you outline the key events from beginning to end, highlighting any turning points?"
        "- How do the events unfold in the final part, and what marks the video's conclusion?"
    )
    user_prompt = (
        f"The ground truth caption is: {gt_caption}. "
        f"The detailed noisy description is: {dense_caption}. "
        "The detailed description provides more detailed explanations of the video content and is in the correct order of events. "
        "Please use the detailed description to extract any relevant additional information, but do not base your questions or answers solely on them. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Emphasize that ALL THREE questions must be designed to extract information DIRECTLY from the given information, focusing on the time and order of events in the video."
    )
    return system_prompt, user_prompt


def get_short_temporal_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers related to the temporal events in a video. "
        "Your goal is to help users understand the sequence and timing of events in the video by asking and answering questions that focus on when events occur."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description generated from ordered frames of the video in the correct order of events. "
        "You will generate a set of questions and answers related to the events in the video using approximate time references, by closely analyzing the sequence of sentences in the provided information. "
        "Generate THREE different descriptive questions and answers based on the provided caption and detailed description."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the events in the video and focus on significant temporal aspects."
        "- Use approximate time references such as the beginning, middle, and end."
        "- The answers must be based on the information provided in the caption and detailed description."
        "- Do not explicitly mention in the answers that it is based on the caption or frames."
        "##SAMPLE QUESTIONS:"
        "- When does event x happen in the video?"
        "- What happens after event x in the video?"
        "- What happens before event x in the video?"
        "- Can you tell me the sequence of events in the video?"
        "- How do the events in the video progress from beginning to end?"
        "- What do the girls do after visiting the park?"
        "- At which part of the video does the dog play with the ball?"
        "- When does the car hit the motorcycle?"
        "- Why is the woman hunched over in the beginning?"
        "- Why does the boy start crying towards the end of the video?"
        "- When does he shoot at the basket?"
        "- What happens before the boys enter the bus?"
    )
    user_prompt = (
        f"The ground truth caption is: {gt_caption}. "
        f"The detailed noisy description is: {dense_caption}. "
        "The provided detailed description has more detailed explanations of the video content and is in the correct order of events. "
        "Please use the detailed description to extract any relevant additional information, but do not base your questions or answers solely on them. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Emphasize that ALL THREE questions must be designed to extract information DIRECTLY from the given information, focusing on the time and order of events in the video."
    )
    return system_prompt, user_prompt


def get_spatial_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and detailed answers based on video descriptions. "
        "Your goal is to extract important spatial information from the video content, ensuring the questions focus on significant visual details."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract spatial information directly from the given information, so that the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and detailed answers focusing on different spatial aspects such as colors, outfits, location, and displayed text."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be based on the visual events in the video and focus on significant spatial details."
        "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers."
        "- The answers must include details about the setting, objects involved, and any specific visual features."
        "- Each question should focus on a different key aspect such as colors, attire, displayed texts, or location."
        "- Avoid asking about irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- What is the color of the woman's shirt?"
        "- What is the name of the drink on the bottle?"
        "- Describe the outfit of the dancers."
        "- Explain the setting of the video and the objects in the scene."
        "- What is the goalkeeper wearing in the video?")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        f"The detailed noisy description is: {dense_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the visual information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, the question should focus on key aspects such as appearance, colors, outfits, location, and displayed text."
    )

    return system_prompt, user_prompt


def get_reasoning_qa_prompt(gt_caption, dense_caption):
    system_prompt = (
        "You are an AI assistant tasked with generating questions and answers based on video descriptions. "
        "Your goal is to extract specific, detailed information from the video content, focusing on observable actions, objects, and settings, ensuring the questions are diverse and cover a range of aspects like the identity of objects, actions of individuals, types or styles of activities, and the reasoning or context for actions."
        "##TASK:"
        "Users will provide a caption of a video and a detailed noisy description, and you will generate a set of questions and answers related to the video. "
        "The questions should be designed to extract specific details directly from the given information, ensuring the provided information or parts of it can serve as the answers. "
        "Generate THREE different questions and concise answers based on the given information. Each question should focus on a different aspect such as actions of individuals, objects involved, and reasoning behind actions."
        "------"
        "##INSTRUCTIONS:"
        "- The questions must be specific and based on significant details visible or inferred from the events in the video."
        "- Ensure the questions cover different types such as what, where, why, and how, focusing on individual actions, object details, and context or reasoning."
        "- Answers should be concise, incorporating brief details about the setting, objects involved, and any specific techniques or methods used."
        "- Avoid asking about generic or irrelevant details."
        "##SAMPLE QUESTIONS:"
        "- What is the man in the red shirt doing?"
        "- Where does the woman look after picking up the object?"
        "- Who is John Davis in the video?"
        "- Why did the player throw the ball?"
        "- What action does the coach take after the whistle blows?")

    user_prompt = (
        f"The video ground truth caption is: {gt_caption}. "
        f"The detailed noisy description is: {dense_caption}. "
        "The detailed description is provided as a supplementary source of information. "
        "It may contain additional details about objects or activities mentioned in the video caption, but the main focus should be on the information provided in the video caption. "
        "Format the output as a list of dictionaries in JSON style, with each dictionary containing a 'Q' key for the question and an 'A' key for the answer. "
        "For example: "
        "[{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, "
        "{'Q': 'Your second question here...', 'A': 'Your second answer here...'}, "
        "{'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
        "Most importantly, each question should explore a different key aspect such as what, where, why, and how, focusing on object identification, specific actions, and contextual or reasoning details."
    )

    return system_prompt, user_prompt


def annotate(gt_file, caption_files, curr_output_dir, curr_video_descriptions_path, api_key):
    """
    Generate question-answer pairs using caption and
    dense-captions summarized from off-the-shelf models using OpenAI GPT-3.
    """
    openai.api_key = api_key  # Set the OpenAI API key for this process
    summary_qa_pairs = False
    generic_qa_pairs = False
    temporal_qa_pairs = False
    spatial_qa_pairs = True
    reasoning_qa_pairs = True
    short_temporal_qa_pairs = True
    model = "gpt-3.5-turbo"

    for file in tqdm(caption_files):
        annotated_dit = {}
        key = file.split('.')[0]
        gt_caption = get_gt_caption(gt_file, key)
        detailed_description = get_video_description(curr_video_descriptions_path, key)

        if summary_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3: Summarization
            system_prompt, user_prompt = get_summary_qa_prompt(gt_caption, detailed_description)
            completion_0 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_0 = completion_0["choices"][0]["message"]["content"]
            response_dict_0 = ast.literal_eval(response_message_0)

            annotated_dit['summary_qa_pairs'] = response_dict_0

        if generic_qa_pairs:
            # Generate QA pairs with OpenAI GPT-3
            system_prompt, user_prompt = get_generic_qa_prompt(gt_caption, detailed_description)
            completion_1 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_1 = completion_1["choices"][0]["message"]["content"]
            response_dict_1 = ast.literal_eval(response_message_1)

            annotated_dit['generic_qa_pairs'] = response_dict_1

        if temporal_qa_pairs:
            system_prompt, user_prompt = get_temporal_qa_prompt(gt_caption, detailed_description)
            completion_2 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_2 = completion_2["choices"][0]["message"]["content"]
            response_dict_2 = ast.literal_eval(response_message_2)

            annotated_dit['temporal_qa_pairs'] = response_dict_2

        if spatial_qa_pairs:
            system_prompt, user_prompt = get_spatial_qa_prompt(gt_caption, detailed_description)
            completion_3 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_3 = completion_3["choices"][0]["message"]["content"]
            response_dict_3 = ast.literal_eval(response_message_3)

            annotated_dit['spatial_qa_pairs'] = response_dict_3

        if reasoning_qa_pairs:
            system_prompt, user_prompt = get_reasoning_qa_prompt(gt_caption, detailed_description)
            completion_4 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_4 = completion_4["choices"][0]["message"]["content"]
            response_dict_4 = ast.literal_eval(response_message_4)

            annotated_dit['reasoning_qa_pairs'] = response_dict_4

        if short_temporal_qa_pairs:
            system_prompt, user_prompt = get_short_temporal_qa_prompt(gt_caption, detailed_description)
            completion_5 = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            response_message_5 = completion_5["choices"][0]["message"]["content"]
            response_dict_5 = ast.literal_eval(response_message_5)

            annotated_dit['short_temporal_qa_pairs'] = response_dict_5

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(curr_output_dir, f"{key}.json")
        with open(json_file_path, "w", encoding='utf-8') as f:
            json.dump(annotated_dit, f, ensure_ascii=False, indent=4)

    print(f"Completed, Annotations saved in {curr_output_dir}")


def get_gt_caption(json_data, video_id):
    video_data = json_data[video_id]
    gt_captions = video_data['sentences']
    gt_caption = ''.join(gt_captions)
    return gt_caption


def get_video_description(video_descriptions_path, video_id):
    file_name = video_id.split('.')[0]
    video_path = os.path.join(video_descriptions_path, f'{file_name}.json')
    data = json.load(open(video_path))
    dense_caption = data['dense_caption']
    return dense_caption


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
            task_args = [
                (gt_json_data, part, args.output_dir, args.video_descriptions_path, api_keys[i % len(api_keys)]) for
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
