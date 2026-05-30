import os

import torch.nn as nn


class DinoV3Backbone(nn.Module):
    def __init__(self, weights_dir="weights/dinov3"):
        super().__init__()
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(
                f"DINOv3 weights not found at {weights_dir}. Please provide local weights."
            )

        import transformers

        self.model = transformers.AutoModel.from_pretrained(weights_dir)
        self.config = self.model.config
        self.name = "dinov3"
        self.is_dinov3 = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_patch_tokens(feat, num_register_tokens=4):
        special_tokens = 1 + int(num_register_tokens)
        return feat[:, special_tokens:, :]

    @staticmethod
    def format_to_spatial_grid(feat, num_register_tokens=4):
        patch_tokens = DinoV3Backbone.get_patch_tokens(feat, num_register_tokens)
        batch, num_tokens, channels = patch_tokens.shape
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            raise ValueError(
                "Patch token count is not a perfect square after removing special tokens."
            )

        return (
            patch_tokens.reshape(batch, side, side, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
