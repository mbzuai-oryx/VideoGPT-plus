#!/bin/sh


export DATASET_DIR=playground/data

BASE_LLM_PATH=microsoft/Phi-3-mini-4k-instruct
VISION_TOWER=OpenGVLab/InternVideo2-Stage2_1B-224p-f4
IMAGE_VISION_TOWER=openai/clip-vit-large-patch14-336
PROJECTOR_TYPE=mlp2x_gelu
PRETRAIN_VIDEO_MLP_PATH=MBZUAI/VideoGPT-plus_Phi3-mini-4k_Pretrain/mlp2x_gelu_internvideo2/mm_projector.bin
PRETRAIN_IMAGE_MLP_PATH=MBZUAI/VideoGPT-plus_Phi3-mini-4k_Pretrain/mlp2x_gelu_clip_l14_336px/mm_projector.bin
OUTPUT_DIR_PATH=results/videogpt_plus_finetune

deepspeed videogpt_plus/train/train.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed scripts/zero3.json \
--model_name_or_path "$BASE_LLM_PATH" \
--version phi3_instruct \
--dataset_use FINETUNING \
--vision_tower "$VISION_TOWER" \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--pretrain_mm_mlp_adapter "$PRETRAIN_VIDEO_MLP_PATH" \
--pretrain_image_mm_mlp_adapter "$PRETRAIN_IMAGE_MLP_PATH" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none
