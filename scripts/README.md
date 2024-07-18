# Training VideotGPT+ :train:
We provide scripts for projector pretraining and video fine-tuning of VideoGPT+. Please follow the instructions below.

## Download Training Dataset
You can download all the pretraining and fine-tuning datasets from HuggingFace follow the instructions below,

```bash
mkdir playground
mkdir playground/data
cd playground/data
git lfs install
git clone https://huggingface.co/datasets/MBZUAI/VideoGPT-plus_Training_Dataset
```

## Projector pretraining with CLIP Image Encoder
Use the script [scripts/pretrain_projector_image_encoder.sh](scripts/pretrain_projector_image_encoder.sh) for running MLP projector pretraining with CLIP Image Encoder.

## Projector pretraining with InternVideo2 Video Encoder
Please use the script [scripts/pretrain_projector_video_encoder.sh](scripts/pretrain_projector_video_encoder.sh) for running MLP projector pretraining with InternVideo2 video encoder.

ALTERNATIVELY, you can download the pretrained projector weights provided by us from the HuggingFace,

```bash
git lfs install
git clone https://huggingface.co/MBZUAI/VideoGPT-plus_Phi3-mini-4k_Pretrain
```

## Video Instruction Fine-tuning
Please use the script [scripts/finetune_dual_encoder.sh](finetune_dual_encoder.sh) for video instruction fine-tuning.
