"""ONNX export wrapper for DINOv3 feature extraction pipeline.

Wraps the full PatchCore feature extraction (backbone → spatial grid →
preprocessing → aggregator) into a single nn.Module suitable for ONNX export.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DinoV3FeatureExtractorONNX(nn.Module):
    """Wraps PatchCore feature extraction for ONNX export.

    Input:  (B, 3, H, W) — preprocessed image tensor
    Outputs:
        FEATURES:     (B * num_patches, target_dim) — per-patch feature vectors
        PATCH_SHAPES: (2, 2) — [[grid_h, grid_w], ...] for each extracted layer
    """

    def __init__(
        self,
        patchcore_model,
        image_size: int = 384,
    ):
        super().__init__()
        self.patchcore = patchcore_model
        self.image_size = image_size
        self.layers_to_extract_from = patchcore_model.layers_to_extract_from
        self.patchsize = patchcore_model.patch_maker.patchsize
        self.patchstride = patchcore_model.patch_maker.stride

    def _patchify(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, List[int]]:
        """Replicate PatchMaker.patchify without dynamic padding issues."""
        padding = int((self.patchsize - 1) // 2)
        unfolder = nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.patchstride,
            padding=padding,
            dilation=1,
        )
        unfolded = unfolder(features)

        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) // self.patchstride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded = unfolded.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded = unfolded.permute(0, 4, 1, 2, 3)

        return unfolded, number_of_total_patches

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX tracing.

        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            features: (B * num_patches, target_dim)
            patch_shapes: (num_layers, 2) — [[h, w], ...]
        """
        # Step 1: Extract backbone features
        outputs = self.patchcore.forward_modules["feature_aggregator"](x)
        features = [outputs[layer] for layer in self.layers_to_extract_from]

        # Step 2: Patchify each layer
        patch_shapes: List[List[int]] = []
        patched_features = []
        for feat in features:
            patched, shape = self._patchify(feat)
            patch_shapes.append(shape)
            patched_features.append(patched)

        ref_num_patches = patch_shapes[0]

        # Step 3: Interpolate subsequent layers to match reference grid
        for i in range(1, len(patched_features)):
            _features = patched_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0],
                patch_dims[0],
                patch_dims[1],
                *_features.shape[2:],
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2],
                ref_num_patches[0],
                ref_num_patches[1],
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )
            patched_features[i] = _features

        # Step 4: Flatten and apply preprocessing + aggregator
        features = [x.reshape(-1, *x.shape[-3:]) for x in patched_features]
        features = self.patchcore.forward_modules["preprocessing"](features)
        features = self.patchcore.forward_modules["preadapt_aggregator"](
            features
        )

        # Step 5: Prepare outputs
        patch_shapes_tensor = torch.tensor(
            patch_shapes, dtype=torch.int64, device=x.device
        )

        return features, patch_shapes_tensor


def export_to_onnx(
    patchcore_model,
    export_path: Path,
    image_size: int = 384,
    device: Optional[torch.device] = None,
    opset_version: int = 17,
) -> Path:
    """Export PatchCore feature extractor to ONNX format.

    Args:
        patchcore_model: Trained PatchCore instance with loaded backbone.
        export_path: Destination path for the ONNX file.
        image_size: Input image dimension (assumed square).
        device: Torch device for tracing. Auto-detected if None.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    patchcore_model.to(device)
    patchcore_model.eval()

    wrapper = DinoV3FeatureExtractorONNX(
        patchcore_model, image_size=image_size
    )
    wrapper.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Tracing model for ONNX export...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["IMAGE"],
            output_names=["FEATURES", "PATCH_SHAPES"],
            dynamic_axes={
                "FEATURES": {0: "num_features"},
            },
        )

    logger.info("Exported ONNX model to %s", export_path)
    return export_path
