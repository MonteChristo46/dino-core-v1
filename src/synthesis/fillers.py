"""Island fill strategies for semantic patch composition.

Encapsulated fillers that determine how Perlin islands are filled
with visual content. Each filler is independent and can be swapped.
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


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


#: Default DTD categories that resemble plausible industrial anomalies.
#: Structured patterns (knitted, woven, braided, etc.) are excluded.
DTD_ANOMALY_CATEGORIES: set[str] = {
    "blotchy",
    "bubbly",
    "bumpy",
    "cracked",
    "crystalline",
    "fibrous",
    "flecked",
    "grooved",
    "marbled",
    "matted",
    "perforated",
    "pitted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "spiralled",
    "stained",
    "stratified",
    "studded",
    "swirly",
    "veined",
    "waffled",
    "wrinkled",
}


class DTDTextureFiller(IslandFiller):
    """Fill island with a texture sampled from the Describable Textures Dataset.

    Selects a random DTD texture, tiles and/or scales it to cover the island
    bounding box, and returns it as fill content. Ignores ``source_pixels``
    since the texture comes from an external pool.

    Args:
        dtd_root: Path to the DTD ``images/`` directory containing category
            subdirectories.
        allowed_categories: Set of category names to sample from. Defaults to
            ``DTD_ANOMALY_CATEGORIES``. Pass ``None`` to use all 35 categories.
        seed: Optional seed for reproducible texture selection.
    """

    _VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(
        self,
        dtd_root: str,
        allowed_categories: Optional[set[str]] = None,
        seed: Optional[int] = None,
    ):
        self._rng = random.Random(seed)
        self._texture_paths: list[str] = []

        root = Path(dtd_root)
        if not root.is_dir():
            raise ValueError(f"DTD root not found: {dtd_root}")

        categories = allowed_categories if allowed_categories is not None else None

        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if categories is not None and entry.name not in categories:
                continue
            for file in entry.iterdir():
                if file.suffix.lower() in self._VALID_EXTENSIONS:
                    self._texture_paths.append(str(file))

        if not self._texture_paths:
            raise ValueError(f"No valid texture images found in {dtd_root}")

    def fill(
        self,
        image: np.ndarray,
        island_mask: np.ndarray,
        source_pixels: np.ndarray,
    ) -> np.ndarray:
        ih, iw = island_mask.shape[:2]
        result = np.zeros((ih, iw, 3), dtype=np.uint8)

        if ih == 0 or iw == 0:
            return result

        texture_path = self._rng.choice(self._texture_paths)
        texture = np.array(Image.open(texture_path).convert("RGB"))

        filled = self._tile_and_crop(texture, ih, iw)
        result[...] = filled
        return result

    def _tile_and_crop(self, texture: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        th, tw = texture.shape[:2]

        if th >= target_h and tw >= target_w:
            y0 = self._rng.randint(0, th - target_h)
            x0 = self._rng.randint(0, tw - target_w)
            return texture[y0 : y0 + target_h, x0 : x0 + target_w]

        tiles_y = max(1, (target_h + th - 1) // th)
        tiles_x = max(1, (target_w + tw - 1) // tw)

        tiled = np.tile(texture, (tiles_y, tiles_x, 1))
        y0 = self._rng.randint(0, max(0, tiled.shape[0] - target_h))
        x0 = self._rng.randint(0, max(0, tiled.shape[1] - target_w))
        return tiled[y0 : y0 + target_h, x0 : x0 + target_w]
