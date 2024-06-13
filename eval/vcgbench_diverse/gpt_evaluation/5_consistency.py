"""
    VCGBench-Diverse - Evaluation Script for Consistency (CO) using gpt-3.5-turbo-0125

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
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="VCGBench-Diverse - Evaluation Script for Consistency (CO)")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--gt_json_path", required=True, help="The path to file containing ground_truths.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3 and
    returns a score for consistency.
    """
    for file in tqdm(caption_files):
        key = file.split('.')[0]  # Strip file extension
        qa_set = prediction_set[int(key)]
        question1 = qa_set['q1']
        question2 = qa_set['q2']
        answer = qa_set['a']
        pred1 = qa_set['pred1']
        pred2 = qa_set['pred2']
        try:
            # Compute the consistency score
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an AI assistant tasked with evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                            "You will be given two very similar questions, a common correct answer, and predicted answers for the two questions."
                            "Your task is to compare the predicted answers for two very similar questions with a common correct answer and determine if they are consistent."
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or differences in the conveyed information.\n"
                            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                            "- Provide a single evaluation score that reflects the level of consistency of the two predicted answers compared to the correct answer.\n"
                            "- Assign a consistency score between 0 and 5, where 5 indicates the highest level of consistency.\n"
                            "- Base your evaluation on the following scale:\n"
                            "  5: PERFECT consistency in terms of correctness and conveyed information.\n"
                            "  4: Very little discrepancies, but mostly consistent.\n"
                            "  3: Mostly consistent, though minor elements are inconsistent.\n"
                            "  2: Very little consistent elements.\n"
                            "  1: Mostly incorrect or irrelevant elements, somewhat consistent.\n"
                            "  0: Completely inconsistent with no correct elements."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question 1: {question1}\n"
                            f"Question 2: {question2}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer to Question 1: {pred1}\n"
                            f"Predicted Answer to Question 2: {pred2}\n\n"
                            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'score': 4.8}."
                    }
                ],
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = args.pred_path
    pred_contents = json.load(open(file, 'r'))

    # Read GT file
    gt_contents = json.load(open(args.gt_json_path, 'r'))
    types = ['consistency']
    consistency_ids = [x['id'] for x in gt_contents if x['type'] in types]
    # Generating list of id's and corresponding files
    id_list = [x['ann_id'] for x in pred_contents if x['ann_id'] in consistency_ids]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in pred_contents:
        id = sample['ann_id']
        if id in id_list:
            question1 = sample['prompt_1']
            question2 = sample['prompt_2']
            answer = sample['answer']
            pred1 = sample['text_1']
            pred2 = sample['text_2']
            qa_set = {"q1": question1, "q2": question2, "a": answer, "pred1": pred1, "pred2": pred2}
            prediction_set[id] = qa_set

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                key = file_name.split(".")[0]
                combined_contents[key] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score
    score_sum = 0
    count = 0
    for key, result in combined_contents.items():
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("Average score for consistency:", average_score)


if __name__ == "__main__":
    main()
