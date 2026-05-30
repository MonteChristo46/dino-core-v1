"""Island fill strategies for semantic patch composition.

Encapsulated fillers that determine how Perlin islands are filled
with visual content. Each filler is independent and can be swapped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class IslandFiller(ABC):
    """Abstract base for island fill strategies."""

    @abstractmethod
    def fill(
        self,
        image: np.ndarray,
        island_mask: np.ndarray,
        source_pixels: np.ndarray,
    ) -> np.ndarray:
        """Return filled RGB patch matching island_mask shape.

        Args:
            image: Full RGB image (H, W, 3).
            island_mask: Binary mask of the island (h, w), uint8 0/255.
            source_pixels: Boolean mask of source cluster in full image (H, W).

        Returns:
            RGB patch (h, w, 3) filled with strategy-specific content.
        """


class SourceTextureFiller(IslandFiller):
    """Fill island with actual texture sampled from source region.

    Extracts the source cluster region, scales it to fit the island
    bounding box, and returns it as the fill content.
    """

    def fill(
        self,
        image: np.ndarray,
        island_mask: np.ndarray,
        source_pixels: np.ndarray,
    ) -> np.ndarray:
        ih, iw = island_mask.shape[:2]
        result = np.zeros((ih, iw, 3), dtype=np.uint8)

        if not np.any(source_pixels):
            return result

        ys, xs = np.where(source_pixels)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        src_patch = image[y_min : y_max + 1, x_min : x_max + 1]

        if src_patch.shape[0] == 0 or src_patch.shape[1] == 0:
            return result

        scaled = cv2.resize(src_patch, (iw, ih), interpolation=cv2.INTER_LINEAR)

        result[...] = scaled
        return result


class SourceColorFiller(IslandFiller):
    """Fill island with source region mean color modulated by Perlin intensity.

    Computes the mean RGB of the source cluster, then fills the island
    as: mean_color * (intensity / 255). Produces a smooth color blob
    that matches the source region's overall color.
    """

    def fill(
        self,
        image: np.ndarray,
        island_mask: np.ndarray,
        source_pixels: np.ndarray,
    ) -> np.ndarray:
        ih, iw = island_mask.shape[:2]
        result = np.zeros((ih, iw, 3), dtype=np.uint8)

        if not np.any(source_pixels):
            return result

        mean_color = image[source_pixels].mean(axis=0).astype(np.float32)

        intensity = island_mask.astype(np.float32) / 255.0
        intensity_3ch = np.stack([intensity] * 3, axis=-1)

        filled = (mean_color * intensity_3ch).astype(np.uint8)
        result[...] = filled
        return result
