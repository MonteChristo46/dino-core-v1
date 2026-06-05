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
from .fillers import DTDTextureFiller, IslandFiller, SourceTextureFiller
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
        dtd_filler: Optional[DTDTextureFiller] = None,
        dtd_mix_prob: float = 0.0,
        target_temperature: float = 0.5,
        target_epsilon: float = 0.05,
        min_target_pixels: int = 32,
        seed: Optional[int] = None,
    ):
        self.island_generator = island_generator or PerlinIslandGenerator(seed=seed)
        self.blender = blender or FeatherBlender(feather_radius=12)
        self.filler = filler or SourceTextureFiller()
        self.dtd_filler = dtd_filler
        self.dtd_mix_prob = float(np.clip(dtd_mix_prob, 0.0, 1.0))
        self.target_temperature = max(1e-6, float(target_temperature))
        self.target_epsilon = float(np.clip(target_epsilon, 0.0, 1.0))
        self.min_target_pixels = max(1, int(min_target_pixels))
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

        cluster_inside_mask_pixels = self._cluster_mask_overlaps(unique, cluster_full, mask_bool)
        target_ids, target_probs, dissimilarity_map = self._sample_target_order(
            source_id=source_id,
            clustering=clustering,
            cluster_inside_mask_pixels=cluster_inside_mask_pixels,
        )
        if not target_ids:
            LOGGER.info("PatchComposer: no eligible target clusters inside SAM mask")
            return None

        source_pixels = (cluster_full == source_id) & mask_bool
        if not np.any(source_pixels):
            LOGGER.info("PatchComposer: source cluster %d has no pixels in mask", source_id)
            return None

        selected_filler = self.filler
        fill_mode = "dtd" if isinstance(self.filler, DTDTextureFiller) else "source"
        if self.dtd_filler is not None and self.dtd_mix_prob > 0.0:
            if float(self._rng.random()) < self.dtd_mix_prob:
                selected_filler = self.dtd_filler
                fill_mode = "dtd"

        sam_distance = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
        clearance_scales = (1.0, 0.85, 0.70, 0.55)
        shrink_factor = 0.90
        max_size_attempts = 2
        max_island_resamples = 2
        sam_mask_area = int(np.sum(mask_bool))
        min_visible_pixels = max(64, int(0.01 * sam_mask_area))

        for target_rank, target_id in enumerate(target_ids):
            target_pixels = (cluster_full == target_id) & mask_bool
            target_inside_mask_pixels = int(np.sum(target_pixels))
            if target_inside_mask_pixels <= 0:
                continue

            max_target_clearance = self._max_distance_for_target(target_pixels, sam_distance)
            if max_target_clearance < 1.0:
                continue

            for island_resample_attempt in range(max_island_resamples):
                island = self.island_generator(h, w, mask=sam2_mask)

                if np.sum(island.mask) == 0:
                    LOGGER.info("PatchComposer: island generation produced empty mask")
                    continue

                island_masked = island.mask & mask_bool
                if np.sum(island_masked) == 0:
                    LOGGER.info("PatchComposer: island entirely outside SAM2 mask")
                    continue

                island_scaled, _ = self._fit_island_to_mask(
                    island_masked.astype(np.uint8) * 255, mask_bool
                )
                island_cropped, island_mask_cropped = self._crop_to_bbox(island_scaled)

                safe_candidate_count = 0
                required_clearance = 0
                clearance_scale_used = 1.0
                placement_attempts = 0
                center_x, center_y = 0, 0
                placement_found = False

                for size_attempt in range(max_size_attempts):
                    base_clearance = self._required_clearance(island_mask_cropped, pad_px=3)
                    (
                        safe_center,
                        safe_candidate_count,
                        required_clearance,
                        clearance_scale_used,
                        attempts_used,
                    ) = self._pick_center_with_relaxed_clearance(
                        target_pixels=target_pixels,
                        sam_distance=sam_distance,
                        base_clearance=base_clearance,
                        clearance_scales=clearance_scales,
                        max_target_clearance=max_target_clearance,
                    )
                    placement_attempts += attempts_used

                    if safe_center is not None:
                        center_x, center_y = safe_center
                        placement_found = True
                        break

                    if size_attempt + 1 >= max_size_attempts:
                        break

                    ph, pw = island_cropped.shape[:2]
                    new_w = max(1, int(pw * shrink_factor))
                    new_h = max(1, int(ph * shrink_factor))
                    if new_w == pw and pw > 1:
                        new_w = pw - 1
                    if new_h == ph and ph > 1:
                        new_h = ph - 1

                    island_cropped = cv2.resize(
                        island_cropped,
                        (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    island_mask_cropped = cv2.resize(
                        island_mask_cropped,
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if not placement_found:
                    LOGGER.info(
                        "PatchComposer: no safe center in target cluster %d (rank %d, resample %d/%d)",
                        target_id,
                        target_rank,
                        island_resample_attempt + 1,
                        max_island_resamples,
                    )
                    continue

                ph, pw = island_cropped.shape[:2]
                island_cropped, island_mask_cropped, ph, pw = self._fit_island_to_image(
                    island_cropped,
                    island_mask_cropped,
                    center_x,
                    center_y,
                    h,
                    w,
                )

                i_ys, i_xs = np.where(island_mask_cropped > 0)
                island_cx = int(i_xs.mean()) if len(i_xs) > 0 else pw // 2
                island_cy = int(i_ys.mean()) if len(i_ys) > 0 else ph // 2

                tx1 = max(0, center_x - island_cx)
                ty1 = max(0, center_y - island_cy)
                tx2 = min(w, tx1 + pw)
                ty2 = min(h, ty1 + ph)

                roi_h = ty2 - ty1
                roi_w = tx2 - tx1

                fill = selected_filler.fill(image, island_cropped, source_pixels)

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

                sam_roi = np.zeros((ph, pw), dtype=np.uint8)
                sam_roi[:roi_h, :roi_w] = (sam2_mask[ty1:ty2, tx1:tx2] > 0).astype(np.uint8) * 255
                mask_for_blend = cv2.bitwise_and(mask_for_blend, sam_roi)

                final_area_inside_sam = int(np.sum(mask_for_blend > 0))
                if final_area_inside_sam == 0:
                    LOGGER.info("PatchComposer: blend mask emptied after SAM border clamp")
                    continue

                if final_area_inside_sam < min_visible_pixels and island_resample_attempt + 1 < max_island_resamples:
                    LOGGER.info(
                        "PatchComposer: anomaly too small (%d px < %d px), resampling island",
                        final_area_inside_sam,
                        min_visible_pixels,
                    )
                    continue

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
                        "target_rank": int(target_rank),
                        "target_probability": float(target_probs.get(target_id, 0.0)),
                        "target_dissimilarity": float(dissimilarity_map.get(target_id, 0.0)),
                        "target_inside_mask_pixels": int(target_inside_mask_pixels),
                        "candidate_count_mask_valid": int(len(target_ids)),
                        "target_max_clearance": float(max_target_clearance),
                        "fill_mode": fill_mode,
                        "dtd_mix_prob": float(self.dtd_mix_prob),
                        "target_temperature": float(self.target_temperature),
                        "target_epsilon": float(self.target_epsilon),
                        "min_target_pixels": int(self.min_target_pixels),
                        "island_area": int(np.sum(island.mask)),
                        "island_area_after_fit": int(np.sum(island_scaled > 0)),
                        "target_center": (center_x, center_y),
                        "required_clearance_px": int(required_clearance),
                        "clearance_scale": float(clearance_scale_used),
                        "safe_candidate_count": int(safe_candidate_count),
                        "placement_attempts": int(placement_attempts),
                        "island_resample_attempt": int(island_resample_attempt + 1),
                        "final_area_inside_sam": final_area_inside_sam,
                        "min_visible_pixels": int(min_visible_pixels),
                    },
                )

        LOGGER.info(
            "PatchComposer: failed to place visible anomaly after trying %d target cluster(s)",
            len(target_ids),
        )
        return None

    def _pick_safe_center(
        self,
        target_pixels: np.ndarray,
        sam_distance: np.ndarray,
        required_clearance: int,
    ) -> tuple[Optional[tuple[int, int]], int]:
        safe_pixels = target_pixels & (sam_distance >= float(required_clearance))
        ys, xs = np.where(safe_pixels)
        count = len(ys)
        if count == 0:
            return None, 0

        idx = int(self._rng.integers(0, count))
        return (int(xs[idx]), int(ys[idx])), count

    def _pick_center_with_relaxed_clearance(
        self,
        target_pixels: np.ndarray,
        sam_distance: np.ndarray,
        base_clearance: int,
        clearance_scales: tuple[float, ...],
        max_target_clearance: float,
    ) -> tuple[Optional[tuple[int, int]], int, int, float, int]:
        attempts_used = 0
        for scale in clearance_scales:
            attempts_used += 1
            requested_clearance = max(1, int(np.ceil(base_clearance * scale)))
            required_clearance = max(1, min(requested_clearance, int(np.floor(max_target_clearance))))
            center, count = self._pick_safe_center(
                target_pixels=target_pixels,
                sam_distance=sam_distance,
                required_clearance=required_clearance,
            )
            if center is not None:
                return center, count, required_clearance, float(scale), attempts_used

        return None, 0, base_clearance, float(clearance_scales[-1]), attempts_used

    @staticmethod
    def _max_distance_for_target(target_pixels: np.ndarray, sam_distance: np.ndarray) -> float:
        ys, xs = np.where(target_pixels)
        if len(ys) == 0:
            return 0.0
        return float(np.max(sam_distance[ys, xs]))

    @staticmethod
    def _required_clearance(island_mask: np.ndarray, pad_px: int = 3) -> int:
        ys, xs = np.where(island_mask > 0)
        if len(ys) == 0:
            return max(1, pad_px)

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        max_dist = float(np.max(np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)))
        return max(1, int(np.ceil(max_dist)) + int(pad_px))

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

    def _rank_target_clusters(
        self,
        source_id: int,
        clustering: ClusteringResult,
    ) -> list[int]:
        unique = clustering.unique_clusters
        centroids = clustering.centroids

        if len(unique) < 2 or centroids.size == 0:
            return []

        source_idx = unique.index(source_id)
        if source_idx >= len(centroids):
            return []

        source_centroid = centroids[source_idx]
        scored_targets: list[tuple[float, int]] = []

        for i, cid in enumerate(unique):
            if cid == source_id:
                continue
            if i >= len(centroids):
                continue
            dist = float(np.linalg.norm(centroids[i] - source_centroid))
            scored_targets.append((dist, cid))

        scored_targets.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored_targets]

    @staticmethod
    def _cluster_mask_overlaps(
        unique_clusters: list[int],
        cluster_full: np.ndarray,
        mask_bool: np.ndarray,
    ) -> dict[int, int]:
        overlaps: dict[int, int] = {}
        for cid in unique_clusters:
            overlaps[cid] = int(np.sum((cluster_full == cid) & mask_bool))
        return overlaps

    def _sample_target_order(
        self,
        source_id: int,
        clustering: ClusteringResult,
        cluster_inside_mask_pixels: dict[int, int],
    ) -> tuple[list[int], dict[int, float], dict[int, float]]:
        unique = clustering.unique_clusters
        centroids = clustering.centroids

        if len(unique) < 2 or centroids.size == 0:
            return [], {}, {}

        source_idx = unique.index(source_id)
        if source_idx >= len(centroids):
            return [], {}, {}

        source_centroid = centroids[source_idx]

        candidates: list[tuple[int, float, int]] = []
        for i, cid in enumerate(unique):
            if cid == source_id:
                continue
            if i >= len(centroids):
                continue
            overlap = int(cluster_inside_mask_pixels.get(cid, 0))
            if overlap < self.min_target_pixels:
                continue
            dist = float(np.linalg.norm(centroids[i] - source_centroid))
            candidates.append((cid, dist, overlap))

        if not candidates:
            for i, cid in enumerate(unique):
                if cid == source_id:
                    continue
                if i >= len(centroids):
                    continue
                overlap = int(cluster_inside_mask_pixels.get(cid, 0))
                if overlap <= 0:
                    continue
                dist = float(np.linalg.norm(centroids[i] - source_centroid))
                candidates.append((cid, dist, overlap))

        if not candidates:
            return [], {}, {}

        cids = [cid for cid, _, _ in candidates]
        dists = np.array([dist for _, dist, _ in candidates], dtype=np.float64)

        scaled = dists / self.target_temperature
        scaled -= float(np.max(scaled))
        exp_scores = np.exp(scaled)
        if float(np.sum(exp_scores)) <= 0.0 or not np.all(np.isfinite(exp_scores)):
            base_probs = np.full(len(cids), 1.0 / float(len(cids)), dtype=np.float64)
        else:
            base_probs = exp_scores / float(np.sum(exp_scores))

        uniform = np.full(len(cids), 1.0 / float(len(cids)), dtype=np.float64)
        mixed_probs = ((1.0 - self.target_epsilon) * base_probs) + (self.target_epsilon * uniform)
        mixed_probs = mixed_probs / float(np.sum(mixed_probs))

        order: list[int] = []
        remaining_idx = list(range(len(cids)))
        while remaining_idx:
            rem_probs = np.array([mixed_probs[i] for i in remaining_idx], dtype=np.float64)
            rem_probs = rem_probs / float(np.sum(rem_probs))
            pick_pos = int(self._rng.choice(len(remaining_idx), p=rem_probs))
            picked_idx = remaining_idx.pop(pick_pos)
            order.append(cids[picked_idx])

        target_probs = {cid: float(mixed_probs[i]) for i, cid in enumerate(cids)}
        dissimilarity_map = {cid: float(dists[i]) for i, cid in enumerate(cids)}
        return order, target_probs, dissimilarity_map

    def _pick_target_cluster(
        self,
        source_id: int,
        clustering: ClusteringResult,
    ) -> Optional[int]:
        ranked = self._rank_target_clusters(source_id, clustering)
        return ranked[0] if ranked else None

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

    @staticmethod
    def _fit_island_to_image(
        island_cropped: np.ndarray,
        island_mask_cropped: np.ndarray,
        center_x: int,
        center_y: int,
        img_h: int,
        img_w: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Scale down island patch so it fits within image bounds when centered at (center_x, center_y).

        cv2.seamlessClone requires the entire source image to fit within the destination.
        """
        ph, pw = island_cropped.shape[:2]

        max_half_w = min(center_x, img_w - center_x)
        max_half_h = min(center_y, img_h - center_y)

        if max_half_w <= 0 or max_half_h <= 0:
            return island_cropped, island_mask_cropped, ph, pw

        scale = min(max_half_w * 2 / pw, max_half_h * 2 / ph)

        if scale < 1.0:
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            island_cropped = cv2.resize(island_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            island_mask_cropped = cv2.resize(island_mask_cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            ph, pw = island_cropped.shape[:2]

        return island_cropped, island_mask_cropped, ph, pw
