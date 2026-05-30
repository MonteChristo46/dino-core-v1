"""Semantic patch composer — places Perlin islands between dissimilar DINOv3 clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.augmentation.blenders.base import BlendResult, Blender
from src.augmentation.blenders.feather import FeatherBlender
from src.clustering.base import ClusteringResult
from .fillers import IslandFiller, SourceTextureFiller
from .perlin_island import IslandResult, PerlinIslandGenerator

LOGGER = logging.getLogger(__name__)

GRID_SIZE = 14


@dataclass
class PatchResult:
    augmented_image: np.ndarray
    anomaly_mask: np.ndarray
    source_cluster_id: int
    target_cluster_id: int
    island: IslandResult
    blend_result: BlendResult
    anomaly_patch: np.ndarray
    debug: dict = field(default_factory=dict)


class PatchComposer:
    """Composes synthetic anomalies by pasting Perlin islands between
    semantically dissimilar DINOv3 clusters within the SAM2 object mask.

    Flow:
    1. Upscale 14x14 cluster map to full image resolution
    2. Pick random source cluster from clusters intersecting the SAM2 mask
    3. Generate Perlin island at full image resolution, constrained to
       the SAM2 mask
    4. Fill island with both source texture and source color
    5. Find most dissimilar target cluster via euclidean distance between
       centroids in UMAP space
    6. Scale island down if it would cross SAM2 mask borders
    7. Blend island into target cluster region using a Blender
    """

    def __init__(
        self,
        island_generator: Optional[PerlinIslandGenerator] = None,
        blender: Optional[Blender] = None,
        filler: Optional[IslandFiller] = None,
        seed: Optional[int] = None,
    ):
        self.island_generator = island_generator or PerlinIslandGenerator(seed=seed)
        self.blender = blender or FeatherBlender(feather_radius=12)
        self.filler = filler or SourceTextureFiller()
        self._rng = np.random.default_rng(seed)

    def compose(
        self,
        image: np.ndarray,
        sam2_mask: np.ndarray,
        clustering: ClusteringResult,
        features_2d: torch.Tensor,
    ) -> Optional[PatchResult]:
        """Create a synthetic anomaly by pasting between clusters.

        Args:
            image: RGB image (H, W, 3), uint8
            sam2_mask: Binary mask (H, W), uint8 0/255
            clustering: HDBSCAN clustering result on masked DINO features
            features_2d: DINOv3 patch tokens reshaped to (C, H_grid, W_grid)

        Returns:
            PatchResult or None if composition is not possible
        """
        h, w = image.shape[:2]
        mask_bool = sam2_mask > 0

        if not np.any(mask_bool):
            LOGGER.info("PatchComposer: empty SAM2 mask, skipping")
            return None

        unique = clustering.unique_clusters
        if len(unique) < 2:
            LOGGER.info("PatchComposer: only %d cluster(s), need >= 2", len(unique))
            return None

        cluster_full = self._upscale_cluster_map(clustering.cluster_map_masked, h, w)

        source_id = self._pick_source_cluster(unique, cluster_full, mask_bool)
        if source_id is None:
            LOGGER.info("PatchComposer: no source cluster found within mask")
            return None

        target_id = self._pick_target_cluster(source_id, clustering)
        if target_id is None:
            LOGGER.info("PatchComposer: no dissimilar target cluster found")
            return None

        source_pixels = (cluster_full == source_id) & mask_bool
        if not np.any(source_pixels):
            LOGGER.info("PatchComposer: source cluster %d has no pixels in mask", source_id)
            return None

        island = self.island_generator(h, w, mask=sam2_mask)

        if np.sum(island.mask) == 0:
            LOGGER.info("PatchComposer: island generation produced empty mask")
            return None

        island_masked = island.mask & mask_bool
        if np.sum(island_masked) == 0:
            LOGGER.info("PatchComposer: island entirely outside SAM2 mask")
            return None

        island_scaled, island_center = self._fit_island_to_mask(
            island_masked.astype(np.uint8) * 255, mask_bool
        )

        target_pixels = (cluster_full == target_id) & mask_bool
        if not np.any(target_pixels):
            LOGGER.info("PatchComposer: target cluster %d has no pixels in mask", target_id)
            return None

        target_ys, target_xs = np.where(target_pixels)
        center_y = int(target_ys.mean())
        center_x = int(target_xs.mean())

        island_cropped, island_mask_cropped = self._crop_to_bbox(island_scaled)

        ph, pw = island_cropped.shape[:2]
        i_ys, i_xs = np.where(island_mask_cropped > 0)
        island_cx = int(i_xs.mean()) if len(i_xs) > 0 else pw // 2
        island_cy = int(i_ys.mean()) if len(i_ys) > 0 else ph // 2

        tx1 = max(0, center_x - island_cx)
        ty1 = max(0, center_y - island_cy)
        tx2 = min(w, tx1 + pw)
        ty2 = min(h, ty1 + ph)

        roi_h = ty2 - ty1
        roi_w = tx2 - tx1

        fill = self.filler.fill(image, island_cropped, source_pixels)

        fill_roi = fill[:roi_h, :roi_w]
        intensity_3ch = np.stack([island_cropped[:roi_h, :roi_w]] * 3, axis=-1)
        target_roi = image[ty1:ty2, tx1:tx2]
        anomaly_patch = (target_roi.astype(np.float32) * (1.0 - intensity_3ch / 255.0)
                         + fill_roi.astype(np.float32) * (intensity_3ch / 255.0))
        anomaly_patch = np.clip(anomaly_patch, 0, 255).astype(np.uint8)

        if roi_h < ph or roi_w < pw:
            padded = np.zeros((ph, pw, 3), dtype=np.uint8)
            padded[:roi_h, :roi_w] = anomaly_patch
            anomaly_patch = padded

        mask_for_blend = island_mask_cropped.copy()
        if roi_h < ph or roi_w < pw:
            mask_padded = np.zeros((ph, pw), dtype=np.uint8)
            mask_padded[:roi_h, :roi_w] = island_mask_cropped[:roi_h, :roi_w]
            mask_for_blend = mask_padded

        blend_result = self.blender.blend(
            destination=image.copy(),
            source=anomaly_patch,
            mask=mask_for_blend,
            center=(center_x, center_y),
        )

        return PatchResult(
            augmented_image=blend_result.image,
            anomaly_mask=blend_result.mask,
            source_cluster_id=source_id,
            target_cluster_id=target_id,
            island=island,
            blend_result=blend_result,
            anomaly_patch=anomaly_patch,
            debug={
                "source_cluster": source_id,
                "target_cluster": target_id,
                "island_area": int(np.sum(island.mask)),
                "island_area_after_fit": int(np.sum(island_scaled > 0)),
                "target_center": (center_x, center_y),
            },
        )

    @staticmethod
    def _crop_to_bbox(
        island: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(island > 0)
        if len(ys) == 0:
            return island, island

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        cropped = island[y_min : y_max + 1, x_min : x_max + 1]
        mask = (island > 0).astype(np.uint8) * 255
        mask_cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

        return cropped, mask_cropped

    @staticmethod
    def _upscale_cluster_map(cluster_map_14: np.ndarray, h: int, w: int) -> np.ndarray:
        return cv2.resize(
            cluster_map_14.astype(np.int32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    def _pick_source_cluster(
        self,
        unique_clusters: list[int],
        cluster_full: np.ndarray,
        mask_bool: np.ndarray,
    ) -> Optional[int]:
        candidates = []
        for cid in unique_clusters:
            overlap = np.sum((cluster_full == cid) & mask_bool)
            if overlap > 0:
                candidates.append(cid)

        if not candidates:
            return None

        return int(self._rng.choice(candidates))

    def _pick_target_cluster(
        self,
        source_id: int,
        clustering: ClusteringResult,
    ) -> Optional[int]:
        unique = clustering.unique_clusters
        centroids = clustering.centroids

        if len(unique) < 2 or centroids.size == 0:
            return None

        source_idx = unique.index(source_id)
        if source_idx >= len(centroids):
            return None

        source_centroid = centroids[source_idx]

        best_target = None
        best_dist = -1.0

        for i, cid in enumerate(unique):
            if cid == source_id:
                continue
            if i >= len(centroids):
                continue
            dist = float(np.linalg.norm(centroids[i] - source_centroid))
            if dist > best_dist:
                best_dist = dist
                best_target = cid

        return best_target

    @staticmethod
    def _fit_island_to_mask(
        island: np.ndarray,
        mask_bool: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = mask_bool.shape

        ys, xs = np.where(island > 0)
        if len(ys) == 0:
            return island, (w // 2, h // 2)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        island_h = y_max - y_min + 1
        island_w = x_max - x_min + 1
        island_cy = (y_min + y_max) // 2
        island_cx = (x_min + x_max) // 2

        scale = 1.0
        for dy, dx in [(-island_h // 2, -island_w // 2),
                        (-island_h // 2, island_w // 2),
                        (island_h // 2, -island_w // 2),
                        (island_h // 2, island_w // 2)]:
            cy = island_cy + dy
            cx = island_cx + dx
            if cy < 0 or cy >= h or cx < 0 or cx >= w:
                s_h = max(1, island_h // 2 + abs(dy))
                s_w = max(1, island_w // 2 + abs(dx))
                scale = min(scale, min(h / (2 * s_h), w / (2 * s_w)))

        if scale < 1.0:
            new_h = max(1, int(island_h * scale))
            new_w = max(1, int(island_w * scale))
            island = cv2.resize(island, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(island > 0)
        if len(ys) == 0:
            return island, (w // 2, h // 2)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        island_cy = (y_min + y_max) // 2
        island_cx = (x_min + x_max) // 2

        return island, (island_cx, island_cy)
