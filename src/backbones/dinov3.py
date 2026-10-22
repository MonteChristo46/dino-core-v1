"""Canonical DINOv3 backbone wrapper for the whole repository.

Used by both subsystems:
- the PatchCore detector (via the ``patchcore.dinov3`` shim and
  ``patchcore.backbones.load``)
- the synthetic anomaly pipeline (``src.augmentation``, ``scripts/semantic_patching``)
"""
import os

import torch.nn as nn
import transformers

DEFAULT_WEIGHTS_DIR = "weights/dinov3_vits"


class DinoV3Backbone(nn.Module):
    def __init__(self, weights_dir=DEFAULT_WEIGHTS_DIR):
        super().__init__()
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(
                f"DINOv3 weights not found at {weights_dir}. Expected a HuggingFace "
                f"AutoModel directory such as weights/dinov3_vitb or weights/dinov3_vitl."
            )

        self.model = transformers.AutoModel.from_pretrained(weights_dir)
        self.config = self.model.config
        self.name = "dinov3"
        self.is_dinov3 = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_patch_tokens(feat, num_register_tokens=4):
        """Slices off the special tokens from DINOv3's output.

        The token sequence is [CLS, R x Register tokens, patch tokens...].
        Register tokens are additional learned tokens (like the CLS token) added
        to the input sequence. They act as a storage mechanism for global or
        redundant image information, preventing the model from dumping this
        information into the spatial patch tokens. This results in cleaner,
        artifact-free spatial feature maps. Because they do not correspond to
        spatial locations, we discard them for PatchCore.
        """
        special_tokens = 1 + int(num_register_tokens)
        return feat[:, special_tokens:, :]

    @staticmethod
    def format_to_spatial_grid(feat, num_register_tokens=4):
        """Reshapes patch tokens from a 1D sequence into a [B, C, H, W] grid.

        For example, a 224x224 input yields 201 tokens for DINOv3. After
        slicing off the 5 special tokens (CLS + 4 Register), 196 patch tokens
        remain, which are reshaped into a 14x14 spatial grid and permuted to
        channel-first to match the expected PatchCore feature format.
        """
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
