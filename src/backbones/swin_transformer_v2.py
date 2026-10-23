import logging

import timm
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

SWINV2_CONFIGS = {
    "swinv2_base": {
        "timm_name": "swinv2_base_window8_256",
        "img_size": 256,
        "embed_dim": 128,
        "stage_dims": {0: 128, 1: 256, 2: 512, 3: 1024},
        "stage_resolutions": {0: 64, 1: 32, 2: 16, 3: 8},
    },
    "swinv2_large": {
        "timm_name": "swinv2_large_window12_192",
        "img_size": 192,
        "embed_dim": 192,
        "stage_dims": {0: 192, 1: 384, 2: 768, 3: 1536},
        "stage_resolutions": {0: 48, 1: 24, 2: 12, 3: 6},
    },
}


class SwinTransformerV2Backbone(nn.Module):
    is_dinov3 = False
    is_swin = True

    def __init__(self, name="swinv2_base", pretrained=True, stages_to_extract=(1, 2, 3)):
        super().__init__()
        self.name = name
        self.stages_to_extract = list(stages_to_extract)

        if name not in SWINV2_CONFIGS:
            raise ValueError(
                f"Unknown SwinV2 config: {name}. Available: {list(SWINV2_CONFIGS.keys())}"
            )

        config = SWINV2_CONFIGS[name]
        self.config = config
        self.img_size = config["img_size"]

        LOGGER.info(
            f"Loading SwinV2 ({name}) with {'timm ImageNet pretrained' if pretrained else 'random'} weights"
        )
        self.model = timm.create_model(
            config["timm_name"], pretrained=pretrained, num_classes=0
        )
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def format_to_spatial_grid(feat):
        if feat.dim() == 3:
            B, N, C = feat.shape
            H = W = int(N**0.5)
            return feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        elif feat.dim() == 4:
            return feat.contiguous()
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    def get_stage_resolution(self, stage_idx):
        return self.config["stage_resolutions"].get(stage_idx)

    def get_stage_dim(self, stage_idx):
        return self.config["stage_dims"].get(stage_idx)
