# Quantitative Evaluation 📊

We provide instructions to evaluate VideoGPT+ model on VCGBench, VCGBench-Diverse and MVBench. Please follow the instructions below,

## VCGBench
VCGBench is a commonly used benchmark for video-conversation models, proposed in `Video-ChatGPT` work. It uses GPT-3.5-Turbo to evaluate Correctness of Information (CI), Detail Orientation (DO), 
Contextual Understanding (CU), Temporal Understanding (TU) and Consistency (CO) of a video conversation model. Please follow the steps below to evaluate VideoGPT+ on VCGBench.

### Download VCGBench Dataset
You can download the videos and annotations following the instructions on the official page [https://mbzuai-oryx.github.io/Video-ChatGPT](https://mbzuai-oryx.github.io/Video-ChatGPT).

### Download the VideoGPT+ Model
All the VideoGPT+ models are available on [HuggingFace](https://huggingface.co/collections/MBZUAI/videogpt-665c8643221dda4987a67d8d). Please follow the instructions below to download,

Save the downloaded dataset under `MBZUAI` directory.

```bash

mkdir MBZUAI

git lfs install
git clone https://huggingface.co/MBZUAI/VideoGPT-plus_Phi3-mini-4k
```

### Run Inference
We provide [eval/vcgbench/run_ddp_inference.sh](eval/vcgbench/run_ddp_inference.sh) script to run inference on multiple GPUs,

```bash

bash eval/vcgbench/run_ddp_inference.sh MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench microsoft/Phi-3-mini-4k-instruct MBZUAI/VCGBench

```

Where `MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench` is the path to VideoGPT+ pretrained checkpoints, `microsoft/Phi-3-mini-4k-instruct` is the base model path and `MBZUAI/VCGBench` is the VCGBench dataset path.

### Evaluation
We provide evaluation scripts using GPT-3.5-Turbo. Please use the script [eval/vcgbench/gpt_evaluation/vcgbench_evaluate.sh](eval/vcgbench/gpt_evaluation/vcgbench_evaluate.sh) for evaluation.


## VCGBench-Diverse
VCGBench-Diverse is our proposed benchmarks which effectively addresses the limitations of VCGBench by including videos from 18 broad video categories. We use GPT-3.5-Turbo for the evaluation and report results for 
Correctness of Information (CI), Detail Orientation (DO), 
Contextual Understanding (CU), Temporal Understanding (TU), Consistency (CO), 
Dense Captioning, Spatial Understanding and Reasoning Abilities of video conversation models. Please follow the steps below to evaluate VideoGPT+ on VCGBench-Diverse.


```bash
# Download and extract the VCGBench-Diverse dataset
mkdir MBZUAI
cd MBZUAI
git lfs install
git clone https://huggingface.co/datasets/MBZUAI/VCGBench-Diverse
cd VCGBench-Diverse
tar -xvf videos.tar.gz

# Run inference
bash eval/vcgbench_diverse/inference/run_ddp_inference.sh MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench microsoft/Phi-3-mini-4k-instruct MBZUAI/VCGBench-Diverse

# Run GPT-3.5-Turbo evaluation (replace <OpenAI API Key> with your OpenAI API Key)
bash eval/vcgbench_diverse/gpt_evaluation/vcgbench_diverse_evaluate.sh MBZUAI/VCGBench-Diverse/vcgbench_diverse_qa.json MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench/vcgbench_diverse_eval/answer-vcgbench-diverse.json MBZUAI/VideoGPT-plus_Phi3-mini-4k/vcgbench/vcgbench_diverse_eval/results <OpenAI API Key>
```

## MVBench
MVBench is a comprehensive video understanding benchmark which covers 20 challenging video tasks that cannot be effectively solved with a single frame. It is introduced in the `MVBench: A Comprehensive Multi-modal Video Understanding Benchmark` paper. 
Pleae follow the following steps for evaluation,

```bash
# Download and extract MVBench dataset following the official huggingface link
mkdir OpenGVLab
git lfs install
git clone https://huggingface.co/datasets/OpenGVLab/MVBench

# Extract all the videos in OpenGVLab/MVBench/video

# Run inference
python eval/mvbench/inference/infer.py --model-path MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench --model-base microsoft/Phi-3-mini-4k-instruct

# Evaluate
python eval/mvbench/evaluation/evaluate_mvbench.py
```