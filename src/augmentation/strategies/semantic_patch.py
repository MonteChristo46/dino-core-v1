"""Semantic patch augmentation strategy.

Owns the full pipeline: batch DINOv3 feature extraction → per-image clustering
→ semantic composition (with random fallback for images lacking diverse clusters).

This strategy encapsulates the "semantic" approach — placing Perlin islands
between semantically dissimilar DINOv3 clusters within the SAM2 object mask.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.base import BaseClusterer
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.synthesis.patch_composer import PatchComposer
from src.synthesis.random_composer import RandomPatchComposer

LOGGER = logging.getLogger(__name__)


def _resize_mask_to_grid(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.float32),
        (14, 14),
        interpolation=cv2.INTER_AREA,
    )
    return (resized > 0.5).astype(np.uint8)


def _extract_dino_features_batch(
    backbone: DinoV3Backbone,
    images_pil: List[Image.Image],
    device: str,
) -> List[torch.Tensor]:
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_tensor = torch.stack([preprocess(img) for img in images_pil]).to(device)

    with torch.no_grad():
        outputs = backbone(batch_tensor)
        last_hidden = outputs.last_hidden_state

    spatial = DinoV3Backbone.format_to_spatial_grid(last_hidden)
    return [spatial[i] for i in range(len(images_pil))]


class SemanticPatchStrategy:
    """Augmentation strategy that owns the full semantic pipeline.

    Pipeline per batch:
    1. Batch DINOv3 feature extraction (shared forward pass)
    2. Per-image: resize SAM2 mask → cluster DINOv3 features
    3. Per-image: decide semantic vs random based on coverage + cluster diversity
    4. Per-image: compose anomaly via PatchComposer or RandomPatchComposer

    Requires precomputed SAM2 masks passed via per_image_kwargs["sam2_mask"].
    """

    name: str = "semantic_patch"

    def __init__(
        self,
        backbone: DinoV3Backbone,
        clusterer: BaseClusterer,
        composer: PatchComposer,
        random_composer: RandomPatchComposer,
        device: str = "cuda",
        min_coverage: float = 0.05,
        min_clusters: int = 2,
    ):
        self.backbone = backbone
        self.clusterer = clusterer
        self.composer = composer
        self.random_composer = random_composer
        self.device = device
        self.min_coverage = min_coverage
        self.min_clusters = min_clusters

    def can_apply(self, image: np.ndarray, **kwargs) -> bool:
        sam2_mask = kwargs.get("sam2_mask")
        if sam2_mask is None:
            return False
        return (sam2_mask > 0).sum() > 0

    def __call__(
        self,
        image: np.ndarray,
        return_debug: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        results = self.process_batch([image], [kwargs], return_debug=return_debug)
        return results[0]

    def process_batch(
        self,
        images: List[np.ndarray],
        per_image_kwargs: List[dict],
        return_debug: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]]:
        H_list = [img.shape[0] for img in images]
        W_list = [img.shape[1] for img in images]

        images_pil = [Image.fromarray(img) for img in images]
        sam2_masks = [kw.get("sam2_mask") for kw in per_image_kwargs]
        coverages = [kw.get("coverage", 0.0) for kw in per_image_kwargs]

        features_list = _extract_dino_features_batch(self.backbone, images_pil, self.device)

        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]] = []

        profile_fn = getattr(__import__("cuml").accel, "profile", None)
        profile_cm = profile_fn(quiet=True) if callable(profile_fn) else nullcontext(None)

        with profile_cm as profile_result:
            for img, features_2d, sam2_mask, coverage, H, W in zip(
                images, features_list, sam2_masks, coverages, H_list, W_list,
            ):
                fallback_anchor = np.zeros((H, W, 3), dtype=np.uint8)
                fallback_mask = np.zeros((H, W), dtype=np.uint8)

                mask_14 = _resize_mask_to_grid(sam2_mask)
                feat_for_cluster = features_2d.permute(1, 2, 0).reshape(-1, features_2d.shape[0])
                clustering = self.clusterer.cluster(feat_for_cluster, mask_14)

                is_valid = coverage >= self.min_coverage and len(clustering.unique_clusters) >= self.min_clusters

                result = None
                strategy_used = "semantic_patch"

                if is_valid:
                    result = self.composer.compose(
                        image=img,
                        sam2_mask=sam2_mask,
                        clustering=clustering,
                        features_2d=features_2d,
                    )
                else:
                    result = self.random_composer.compose(image=img)
                    strategy_used = "random_patch"

                if result is None or result.anomaly_mask is None or np.sum(result.anomaly_mask) == 0:
                    debug_info = None
                    if return_debug:
                        debug_info = {
                            "success": False,
                            "reason": "composer_failed",
                            "strategy": strategy_used,
                        }
                    results.append((img.copy(), fallback_anchor, fallback_mask, debug_info))
                    continue

                debug_info = None
                if return_debug:
                    debug_info = {
                        "success": True,
                        "strategy": strategy_used,
                        "is_valid": is_valid,
                        "coverage": coverage,
                        "n_clusters": len(clustering.unique_clusters),
                    }
                    if hasattr(result, "source_cluster_id"):
                        debug_info["source_cluster"] = result.source_cluster_id
                        debug_info["target_cluster"] = result.target_cluster_id
                    if hasattr(result, "island"):
                        debug_info["island_area"] = int(np.sum(result.island.mask))
                    if hasattr(result, "placement_center"):
                        debug_info["placement_center"] = result.placement_center
                    if hasattr(result, "debug"):
                        debug_info.update(result.debug)

                anchor_crop = np.zeros((H, W, 3), dtype=np.uint8)
                results.append((result.augmented_image, anchor_crop, result.anomaly_mask, debug_info))

        if profile_result is not None:
            print(profile_result)

        return results
