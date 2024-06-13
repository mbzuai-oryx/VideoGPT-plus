"""
    Semi-automatic Video Annotation Pipeline - Step # 1: Detect scenes and extract keyframes

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
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from scenedetect import detect, ContentDetector, split_video_ffmpeg, open_video, SceneManager
import warnings
import json
from tqdm import tqdm
import sys
import contextlib

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Detect scenes and extract keyframes.")

    parser.add_argument("--video_dir", required=True, help="Directory containing ActivityNet videos.")

    parser.add_argument("--ann_video_ids_file", required=True,
                        help="Path to the unique video ids JSON file (e.g. path to unique_video_ids.json).")
    parser.add_argument("--gt_caption_file", required=True,
                        help="Path to the ground truth captions file (e.g. path to activitynet_gt_captions_train.json).")

    parser.add_argument("--scene_output_dir", required=False, help="Path to save the scene files.", default="scenes")
    parser.add_argument("--frames_output_dir", required=False, help="Path to save the keyframes.", default="key_frames")
    parser.add_argument("--num_keyframes", type=int, default=1, help="Number of keyframes to extract per scene.")

    return parser.parse_args()


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_keyframes(video_path, num_keyframes, output_dir):
    """
    Extracts keyframes using Katna from the video and returns their file paths,
    operating within a temporary directory.
    """
    # Create a temporary directory for extracted frames
    # Initialize video module and disk writer
    vd = Video()
    diskwriter = KeyFrameDiskWriter(location=output_dir)

    # Suppress print output during keyframe extraction
    with suppress_output():
        vd.extract_video_keyframes(no_of_frames=num_keyframes, file_path=video_path, writer=diskwriter)

    return None


def get_scenes(video_path, output_dir):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    # If `start_in_scene` is True, len(scene_list) will always be >= 1
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    split_video_ffmpeg(video_path, scene_list, output_dir)

    return scene_list


def main():
    args = parse_args()
    os.makedirs(args.scene_output_dir, exist_ok=True)
    os.makedirs(args.frames_output_dir, exist_ok=True)
    with open(args.ann_video_ids_file, 'r') as file:
        data = json.load(file)
        video_ids_to_annotate = data['v2_videos']

    # Read ground truth captions file
    gt_file = args.gt_caption_file
    with open(gt_file) as file:
        gt_json_data = json.load(file)

    video_ids_to_annotate = [id for id in video_ids_to_annotate if id in gt_json_data]

    files_to_annotate = [file for file in os.listdir(args.video_dir) if file.split('.')[0] in video_ids_to_annotate]

    for file in tqdm(files_to_annotate):
        try:
            video_id = file.split('.')[0]
            video_path = os.path.join(args.video_dir, file)
            curr_scene_dir = f'{args.scene_output_dir}/{video_id}'
            _ = get_scenes(video_path, curr_scene_dir)  # Extract the scenes and save in the curr_scene_dir
            scenes_to_annotate = os.listdir(curr_scene_dir)
            for scene in tqdm(scenes_to_annotate):
                sce_video_path = os.path.join(curr_scene_dir, scene)
                get_keyframes(sce_video_path, num_keyframes=args.num_keyframes, output_dir=args.frames_output_dir)
        except Exception as e:
            print(f"Error processing video {file}: {e}")


if __name__ == '__main__':
    main()
