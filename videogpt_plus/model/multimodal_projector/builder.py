import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, **kwargs):
    """
        mm_hidden_size = 1408 for InternVideo2-Stage2_1B-224p-f4 (TODO: Update it if you use a different video encoder)
    """
    image_mm_projector = kwargs['image_mm_projector']
    if image_mm_projector:
        projector_type = getattr(config, 'image_mm_projector_type', 'linear')
        config.mm_hidden_size = 1024
    else:
        config.mm_hidden_size = 1408
        projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"Building {projector_type}")

    if projector_type == 'linear':
        projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    print("projector_type:", projector_type)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')
