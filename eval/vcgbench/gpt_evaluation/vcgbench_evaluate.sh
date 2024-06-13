#!/bin/sh

## Path to directory containing predictions (answer-video-generic.json, answer-video-temporal.json, answer-video-consistency.json )
PRED_PATH=$1
## Path to save the results
OUTPUT_DIR_PATH=$2
## OpenAI API Key
OPENAI_API_KEY=$3

python evaluate_benchmark_1_correctness.py --pred_path "$PRED_PATH/answer-video-generic.json" --output_dir "$OUTPUT_DIR_PATH/correctness" --output_json "$OUTPUT_DIR_PATH/correctness.json " --api_key "$OPENAI_API_KEY" --num_tasks 16


python evaluate_benchmark_2_detailed_orientation.py --pred_path "$PRED_PATH/answer-video-generic.json" --output_dir "$OUTPUT_DIR_PATH/detail" --output_json "$OUTPUT_DIR_PATH/detail.json" --api_key "$OPENAI_API_KEY" --num_tasks 16


python evaluate_benchmark_3_context.py --pred_path "$PRED_PATH/answer-video-generic.json" --output_dir "$OUTPUT_DIR_PATH/context" --output_json "$OUTPUT_DIR_PATH/context.json" --api_key "$OPENAI_API_KEY" --num_tasks 16


python evaluate_benchmark_4_temporal.py --pred_path "$PRED_PATH/answer-video-temporal.json" --output_dir "$OUTPUT_DIR_PATH/temporal" --output_json "$OUTPUT_DIR_PATH/temporal.json" --api_key "$OPENAI_API_KEY" --num_tasks 16


python evaluate_benchmark_5_consistency.py --pred_path "$PRED_PATH/answer-video-consistency.json" --output_dir "$OUTPUT_DIR_PATH/consistency" --output_json "$OUTPUT_DIR_PATH/consistency.json" --api_key "$OPENAI_API_KEY" --num_tasks 16
