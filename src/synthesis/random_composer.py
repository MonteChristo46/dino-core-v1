"""Random patch composer — NO_VALID mode for images without valid SAM2 masks.

Generates a random Perlin island, fills it with a random DTD texture,
and places it randomly on the image using seamless cloning.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.augmentation.blenders.base import BlendResult
from src.augmentation.blenders.seamless_clone import SeamlessCloneBlender
from .perlin_island import IslandResult, PerlinIslandGenerator

LOGGER = logging.getLogger(__name__)

DTD_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "dataset", "dtd", "images",
)


@dataclass
class RandomPatchResult:
    augmented_image: np.ndarray
    anomaly_mask: np.ndarray
    island: IslandResult
    dtd_texture: np.ndarray
    dtd_path: str
    blend_result: BlendResult
    placement_center: tuple[int, int]
    debug: dict = field(default_factory=dict)


class RandomPatchComposer:
    """Composes synthetic anomalies without semantic constraints.

    Flow:
    1. Generate random Perlin island (no mask constraint)
    2. Sample random texture from DTD dataset
    3. Tile/scale DTD texture to island bounding box
    4. Place randomly on image with margin
    5. Blend using SeamlessCloneBlender
    """

    def __init__(
        self,
        island_generator: Optional[PerlinIslandGenerator] = None,
        dtd_root: str = DTD_ROOT,
        seed: Optional[int] = None,
    ):
        self.island_generator = island_generator or PerlinIslandGenerator(seed=seed)
        self.dtd_root = dtd_root
        self.blender = SeamlessCloneBlender(mode="mixed")
        self._rng = random.Random(seed)
        self._dtd_cache: list[str] = []

    def _collect_dtd_images(self) -> list[str]:
        if self._dtd_cache:
            return self._dtd_cache

        if not os.path.isdir(self.dtd_root):
            LOGGER.warning("DTD root not found: %s", self.dtd_root)
            return []

        for entry in sorted(os.listdir(self.dtd_root)):
            cat_dir = os.path.join(self.dtd_root, entry)
            if not os.path.isdir(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self._dtd_cache.append(os.path.join(cat_dir, fname))

        return self._dtd_cache

    def _sample_dtd_texture(self, target_h: int, target_w: int) -> tuple[np.ndarray, str]:
        dtd_images = self._collect_dtd_images()
        if not dtd_images:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8), ""

        path = self._rng.choice(dtd_images)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8), path

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ih, iw = img.shape[:2]
        if ih < target_h or iw < target_w:
            scale = max(target_h / ih, target_w / iw)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        ih, iw = img.shape[:2]
        y_off = self._rng.randint(0, max(0, ih - target_h))
        x_off = self._rng.randint(0, max(0, iw - target_w))
        cropped = img[y_off : y_off + target_h, x_off : x_off + target_w]

        if cropped.shape[0] < target_h or cropped.shape[1] < target_w:
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return cropped, path

    def compose(
        self,
        image: np.ndarray,
    ) -> Optional[RandomPatchResult]:
        """Create a random synthetic anomaly.

        Args:
            image: RGB image (H, W, 3), uint8

        Returns:
            RandomPatchResult or None if composition fails
        """
        h, w = image.shape[:2]

        island = self.island_generator(h, w, mask=None)

        if np.sum(island.mask) == 0:
            LOGGER.info("RandomPatchComposer: island generation produced empty mask")
            return None

        island_cropped, island_mask_cropped = self._crop_to_bbox(
            (island.mask * 255).astype(np.uint8)
        )

        ph, pw = island_cropped.shape[:2]

        if ph < 8 or pw < 8:
            LOGGER.info("RandomPatchComposer: island too small (%dx%d)", ph, pw)
            return None

        dtd_texture, dtd_path = self._sample_dtd_texture(ph, pw)

        margin = int(min(h, w) * 0.05)
        min_x = margin + pw // 2
        max_x = w - margin - pw // 2
        min_y = margin + ph // 2
        max_y = h - margin - ph // 2

        if min_x > max_x or min_y > max_y:
            margin = 0
            min_x = pw // 2
            max_x = w - pw // 2
            min_y = ph // 2
            max_y = h - ph // 2

        if min_x > max_x or min_y > max_y:
            LOGGER.info("RandomPatchComposer: island too large for random placement")
            return None

        center_x = self._rng.randint(min_x, max_x)
        center_y = self._rng.randint(min_y, max_y)

        tx1 = max(0, center_x - pw // 2)
        ty1 = max(0, center_y - ph // 2)
        tx2 = min(w, tx1 + pw)
        ty2 = min(h, ty1 + ph)

        roi_h = ty2 - ty1
        roi_w = tx2 - tx1

        texture_roi = dtd_texture[:roi_h, :roi_w]
        intensity_3ch = np.stack([island_cropped[:roi_h, :roi_w]] * 3, axis=-1)
        target_roi = image[ty1:ty2, tx1:tx2]

        anomaly_patch = (
            target_roi.astype(np.float32) * (1.0 - intensity_3ch / 255.0)
            + texture_roi.astype(np.float32) * (intensity_3ch / 255.0)
        )
        anomaly_patch = np.clip(anomaly_patch, 0, 255).astype(np.uint8)

        mask_for_blend = island_mask_cropped.copy()
        if roi_h < ph or roi_w < pw:
            padded = np.zeros((ph, pw, 3), dtype=np.uint8)
            padded[:roi_h, :roi_w] = anomaly_patch
            anomaly_patch = padded
            mask_padded = np.zeros((ph, pw), dtype=np.uint8)
            mask_padded[:roi_h, :roi_w] = island_mask_cropped[:roi_h, :roi_w]
            mask_for_blend = mask_padded

        blend_result = self.blender.blend(
            destination=image.copy(),
            source=anomaly_patch,
            mask=mask_for_blend,
            center=(center_x, center_y),
        )

        # Build anomaly mask directly from the placed island mask.
        # This avoids inverted semantics from clone-side mask handling and
        # guarantees white=anomaly, black=background in NO_VALID mode.
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)
        anomaly_mask[ty1:ty2, tx1:tx2] = island_mask_cropped[:roi_h, :roi_w]

        return RandomPatchResult(
            augmented_image=blend_result.image,
            anomaly_mask=anomaly_mask,
            island=island,
            dtd_texture=dtd_texture,
            dtd_path=dtd_path,
            blend_result=blend_result,
            placement_center=(center_x, center_y),
            debug={
                "island_area": int(np.sum(island.mask)),
                "placement_center": (center_x, center_y),
                "dtd_path": dtd_path,
            },
        )

    @staticmethod
    def _crop_to_bbox(island: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(island > 0)
        if len(ys) == 0:
            return island, island

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        cropped = island[y_min : y_max + 1, x_min : x_max + 1]
        mask = (island > 0).astype(np.uint8) * 255
        mask_cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

        return cropped, mask_cropped
