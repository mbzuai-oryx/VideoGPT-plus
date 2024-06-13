"""
    Modified from https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2

    Copyright 2024 OpenGVLab/InternVideo
    Copyright 2024 MBZUAI ORYX
"""

from videogpt_plus.model.internvideo.config import Config, eval_dict_leaf
from videogpt_plus.model.internvideo.utils import setup_internvideo2V


def build_internvideo(model_path):
    config = Config.from_file('videogpt_plus/model/internvideo/internvideo2_stage2_config_vision.py')
    config.model.vision_encoder['pretrained'] = model_path
    config = eval_dict_leaf(config)
    config['pretrained_path'] = model_path
    intern_model = setup_internvideo2V(config)

    return intern_model
