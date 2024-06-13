#!/bin/sh

# Update the number of gpus as per your configuration
NUM_GPUS=8
MODEL_PATH=MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench
MODEL_BASE=microsoft/Phi-3-mini-4k-instruct
VCGBench_PATH=MBZUAI/VCGBench

export PYTHONPATH="./:$PYTHONPATH"

# General
torchrun --nproc_per_node="$NUM_GPUS" eval/vcgbench/inference/infer_general.py --model-path "$MODEL_PATH" --model-base "$MODEL_BASE" --video-folder "$VCGBench_PATH/Test_Videos" --question-file "$VCGBench_PATH/Benchmarking_QA/generic_qa.json" --output-dir "$MODEL_PATH/vcgbench_eval/answer-video-generic" --conv-mode "phi3_instruct"
python eval/merge.py --input_dir "$MODEL_PATH/vcgbench_eval/answer-video-generic"


# Temporal
torchrun --nproc_per_node="$NUM_GPUS" eval/vcgbench/inference/infer_general.py --model-path "$MODEL_PATH" --model-base "$MODEL_BASE" --video-folder "$VCGBench_PATH/Test_Videos" --question-file "$VCGBench_PATH/Benchmarking_QA/temporal_qa.json" --output-dir "$MODEL_PATH/vcgbench_eval/answer-video-temporal" --conv-mode "phi3_instruct"
python eval/merge.py --input_dir "$MODEL_PATH/vcgbench_eval/answer-video-temporal"


# Consistency
torchrun --nproc_per_node="$NUM_GPUS" eval/vcgbench/inference/infer_consistency.py --model-path "$MODEL_PATH" --model-base "$MODEL_BASE" --video-folder "$VCGBench_PATH/Test_Videos" --question-file "$VCGBench_PATH/Benchmarking_QA/consistency_qa.json" --output-dir "$MODEL_PATH/vcgbench_eval/answer-video-consistency" --conv-mode "phi3_instruct"
python eval/merge.py --input_dir "$MODEL_PATH/vcgbench_eval/answer-video-consistency"
