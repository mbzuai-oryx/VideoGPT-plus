#!/bin/sh

## Path containing the videos
VIDEO_DIR_PATH=$1
## Path to unique_video_ids.json file
ANN_VIDEO_IDS_FILE=$2
## Path to ActivityNet GT captions
GT_CAPTION_FILE=$3
## Output directory path to store the intermediate and final outputs
OUTPUT_DIR_PATH=$4


## Step # 1: Detect scenes and extract keyframes
python 1_scenedetect_and_keyframes.py --video_dir "$VIDEO_DIR_PATH" --ann_video_ids_file "$ANN_VIDEO_IDS_FILE" --gt_caption_file "$GT_CAPTION_FILE" --scene_output_dir "$OUTPUT_DIR_PATH/scenes" --frames_output_dir "$OUTPUT_DIR_PATH/key_frames"


## Step # 2: Frame level detailed captioning using LLaVA-v1.6-34b
python 2_caption_keyframe_llava.py --key_frame_dir "$OUTPUT_DIR_PATH/key_frames" --output_dir "$OUTPUT_DIR_PATH/llava_captions_keyframes"


## Step # 3: Use short ground truth caption along with the frame-level detailed captions to generate a detailed video caption using GPT4-Turbo.
python 3_dense_video_description.py --ann_video_ids_file "$ANN_VIDEO_IDS_FILE" --gt_caption_file "$GT_CAPTION_FILE" --captions_dir "$OUTPUT_DIR_PATH/llava_captions_keyframes" --output_dir "$OUTPUT_DIR_PATH/video_descriptions"


## Step # 4: Generate QA pairs using video descriptions generated in Step # 3 using GPT-3.5-Turbo.
python 4_generate_qa.py --ann_video_ids_file "$ANN_VIDEO_IDS_FILE" --gt_caption_file "$GT_CAPTION_FILE" --video_descriptions_path "$OUTPUT_DIR_PATH/video_descriptions" --output_dir "$OUTPUT_DIR_PATH/video_qa"