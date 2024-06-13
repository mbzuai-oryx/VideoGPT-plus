import os

DATASET_DIR = os.environ.get("DATASET_DIR", "playground/data")

CC3M_595K = {
    "annotation_path": f"{DATASET_DIR}/pretraining/CC3M-595K/chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/CC3M-595K",
}

COCO_CAP = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_cap_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REG = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_reg_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REC = {
    "annotation_path": f"{DATASET_DIR}/pretraining/COCO/coco_rec_chat.json",
    "data_path": f"{DATASET_DIR}/pretraining/COCO/train2014",
}

CONV_VideoChatGPT = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_HUMAN = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg_human_annotated.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_PLUS_112K = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg-plus_112K.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

CAPTION_VIDEOCHAT = {
    "annotation_path": f"{DATASET_DIR}/annotations/caption_videochat.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}

CLASSIFICATION_K710 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}

CLASSIFICATION_SSV2 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}

CONV_VideoChat1 = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
}

REASONING_NExTQA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}

REASONING_CLEVRER_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

REASONING_CLEVRER_MC = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

VQA_WEBVID_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
