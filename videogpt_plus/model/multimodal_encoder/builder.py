from .clip_encoder import CLIPVisionTower
from videogpt_plus.model.internvideo.build_internvideo import build_internvideo

def build_vision_tower(vision_tower_cfg, **kwargs):
    image_vision_tower = kwargs['image_vision_tower']
    if image_vision_tower:
        vision_tower = getattr(vision_tower_cfg, 'image_mm_vision_tower', getattr(vision_tower_cfg, 'image_vision_tower', None))
        kwargs.pop('image_vision_tower', None)
    else:
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    print(f"Building {vision_tower}")
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'InternVideo2' in vision_tower:
        InternVideoTower = build_internvideo(vision_tower)
        InternVideoTower.requires_grad_(False)
        return InternVideoTower

    raise ValueError(f'Unknown vision tower: {vision_tower}')
