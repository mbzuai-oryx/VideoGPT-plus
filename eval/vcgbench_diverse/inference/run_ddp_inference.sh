#!/bin/sh

# Update the number of gpus as per your configuration
NUM_GPUS=8
MODEL_PATH=MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench
MODEL_BASE=microsoft/Phi-3-mini-4k-instruct
VCGBench_Diverse_PATH=MBZUAI/VCGBench-Diverse

export PYTHONPATH="./:$PYTHONPATH"

torchrun --nproc_per_node="$NUM_GPUS" eval/vcgbench_diverse/inference/infer.py --model-path "$MODEL_PATH" --model-base "$MODEL_BASE" --video-folder "$VCGBench_Diverse_PATH/videos" --question-file "$VCGBench_Diverse_PATH/vcgbench_diverse_qa.json" --output-dir "$MODEL_PATH/vcgbench_diverse_eval/answer-vcgbench-diverse" --conv-mode "phi3_instruct"

python eval/merge.py --input_dir "$MODEL_PATH/vcgbench_diverse_eval/answer-vcgbench-diverse"
