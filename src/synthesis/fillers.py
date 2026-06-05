"""Island fill strategies for semantic patch composition.

Encapsulated fillers that determine how Perlin islands are filled
with visual content. Each filler is independent and can be swapped.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

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

    Samples local source windows that match the island patch size,
    then picks the one with the highest source-cluster coverage over
    the island support. This preserves local texture without resizing
    a large source region into the island.

    Args:
        max_tries: Number of random source-centered windows to evaluate.
        min_coverage: Early-stop threshold for accepted coverage in [0, 1].
        strict: If True, return zeros when no candidate reaches
            ``min_coverage``. If False, return the best candidate.
        seed: Optional random seed for reproducible sampling.
    """

    def __init__(
        self,
        max_tries: int = 64,
        min_coverage: float = 0.90,
        strict: bool = False,
        seed: Optional[int] = None,
    ):
        self.max_tries = max(1, int(max_tries))
        self.min_coverage = float(np.clip(min_coverage, 0.0, 1.0))
        self.strict = strict
        self._rng = np.random.default_rng(seed)

    def fill(
        self,
        image: np.ndarray,
        island_mask: np.ndarray,
        source_pixels: np.ndarray,
    ) -> np.ndarray:
        ih, iw = island_mask.shape[:2]
        result = np.zeros((ih, iw, 3), dtype=np.uint8)
        h, w = image.shape[:2]

        if ih == 0 or iw == 0:
            return result
        if h < ih or w < iw:
            return result
        if not np.any(source_pixels):
            return result

        island_support = island_mask > 0
        support_count = int(np.sum(island_support))
        if support_count == 0:
            return result

        ys, xs = np.where(source_pixels)
        if len(ys) == 0:
            return result

        n = len(ys)
        tries = min(self.max_tries, n)
        if tries == n:
            sample_idx = self._rng.permutation(n)
        else:
            sample_idx = self._rng.choice(n, size=tries, replace=False)

        best_patch: Optional[np.ndarray] = None
        best_coverage = -1.0

        for idx in sample_idx:
            cy = int(ys[idx])
            cx = int(xs[idx])

            y1 = cy - ih // 2
            x1 = cx - iw // 2
            y1 = int(np.clip(y1, 0, h - ih))
            x1 = int(np.clip(x1, 0, w - iw))
            y2 = y1 + ih
            x2 = x1 + iw

            window_source = source_pixels[y1:y2, x1:x2]
            coverage = float(np.mean(window_source[island_support]))

            if coverage > best_coverage:
                best_coverage = coverage
                best_patch = image[y1:y2, x1:x2].copy()

            if coverage >= self.min_coverage:
                return image[y1:y2, x1:x2].copy()

        if best_patch is None:
            return result

        if self.strict and best_coverage < self.min_coverage:
            return result

        return best_patch


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
