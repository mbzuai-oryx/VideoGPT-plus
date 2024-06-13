#!/bin/sh

## Path to the VCGBench ground truth path (vcgbench_diverse_qa.json)
GT_PATH=$1
## Path to directory containing predictions (answer-vcgbench-diverse.json)
PRED_PATH=$2
## Path to save the results
OUTPUT_DIR_PATH=$3
## OpenAI API Key
OPENAI_API_KEY=$4


## FORMAT of the PREDICTION FILE (answer-vcgbench-diverse.json) should be as follows.

## List of dictionaries where each dictionary represents one sample.
## For consistency questions, the dictionary must have the keys ann_id, video_name, prompt_1, text_1, prompt_2, text_2, and answer.
## Here ann_id represents the unique annotation id from the ground truth (vcgbench_diverse_qa.json).

## An example of the consistency prediction is,
## {"ann_id": 1715, "video_name": "Mwn9ir0CkF4.mp4", "prompt_1": question_1, "text_1": answer_1, "prompt_2": question_2, "text_2": answer_2, "answer": gt_answer}

## For all other types of question, the prediction will have only one question and answer as follows,
## {"ann_id": 1071, "video_name": "7A3n_hJJjgg.mp4", "prompt": question, "text": answer, "answer": gt_answer}


python 1_correctness_of_information.py --pred_path "$PRED_PATH/answer-vcgbench-diverse.json" --output_dir "$OUTPUT_DIR_PATH/correctness" --output_json "$OUTPUT_DIR_PATH/correctness.json" --gt_json_path "$GT_PATH" --api_key "$OPENAI_API_KEY" --num_tasks 16


python 2_detailed_orientation.py --pred_path "$PRED_PATH/answer-vcgbench-diverse.json" --output_dir "$OUTPUT_DIR_PATH/detail" --output_json "$OUTPUT_DIR_PATH/detail.json" --gt_json_path "$GT_PATH"  --api_key "$OPENAI_API_KEY" --num_tasks 16


python 3_contextual_information.py --pred_path "$PRED_PATH/answer-vcgbench-diverse.json" --output_dir "$OUTPUT_DIR_PATH/context" --output_json "$OUTPUT_DIR_PATH/context.json" --gt_json_path "$GT_PATH"  --api_key "$OPENAI_API_KEY" --num_tasks 16


python 4_temporal_information.py --pred_path "$PRED_PATH/answer-vcgbench-diverse.json" --output_dir "$OUTPUT_DIR_PATH/temporal" --output_json "$OUTPUT_DIR_PATH/temporal.json" --gt_json_path "$GT_PATH"  --api_key "$OPENAI_API_KEY" --num_tasks 16


python 5_consistency.py --pred_path "$PRED_PATH/answer-vcgbench-diverse.json" --output_dir "$OUTPUT_DIR_PATH/consistency" --output_json "$OUTPUT_DIR_PATH/consistency.json" --gt_json_path "$GT_PATH" --api_key "$OPENAI_API_KEY" --num_tasks 16


python dense_captioning_spatial_and_reasoning_scores.py --gt_json_path "$GT_PATH" --results_dir_path "$OUTPUT_DIR_PATH"
