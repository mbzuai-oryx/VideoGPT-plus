"""
    VCGBench-Diverse - Evaluation Script for Dense Captioning, Spatial Understanding and Reasoning

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

import argparse
import json
import os
from tqdm import tqdm


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--gt_json_path', type=str, required=True, help="The path to file containing ground_truths.")
    parser.add_argument('--results_dir_path', type=str, required=True,
                        help="The path containing correctness and detail evaluation results (i.e. correctness.json and detail.json files).")

    return parser.parse_args()


def read_jsonl(file_path):
    all_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    return all_data


def main():
    args = parse_args()

    gt_json_contents = json.load(open(args.gt_json_path))
    id_to_type_dict = {}
    for content in gt_json_contents:
        id_to_type_dict[content['id']] = content['type']

    type_to_score_dict = {"summary": [], "spatial": [], "reasoning": []}
    target_jsonl_names = ["correctness.json", "detail.json"]
    for target_jsonl_name in target_jsonl_names:
        target_json_path = os.path.join(args.results_dir_path, target_jsonl_name)
        target_json_data = json.load(open(target_json_path))
        for id_key in tqdm(target_json_data.keys()):
            ann_type = id_to_type_dict[int(id_key)]
            if ann_type in type_to_score_dict.keys():
                type_to_score_dict[ann_type].append(target_json_data[id_key][0]['score'])

    for key in type_to_score_dict.keys():
        type_to_score_dict[key] = sum(type_to_score_dict[key]) / len(type_to_score_dict[key])

    print(f"Dense Caption: {type_to_score_dict['summary']}\n"
          f"Spatial: {type_to_score_dict['spatial']}\n"
          f"Reasoning: {type_to_score_dict['reasoning']}")


if __name__ == '__main__':
    main()
