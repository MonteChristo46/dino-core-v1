"""Random patch augmentation strategy.

Wraps RandomPatchComposer to implement the AugmentationStrategy protocol.
Applies synthetic anomalies without semantic constraints — places random
Perlin islands filled with DTD textures at random positions.

This is the fallback strategy: can_apply() always returns True.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from src.synthesis.random_composer import RandomPatchComposer

LOGGER = logging.getLogger(__name__)


class RandomPatchStrategy:
    """Fallback augmentation strategy with no semantic constraints.

    Always eligible (can_apply returns True). Places random Perlin
    islands filled with DTD textures at random image positions.
    """

    name: str = "random_patch"

    def __init__(self, composer: RandomPatchComposer):
        self.composer = composer

    def can_apply(self, image: np.ndarray, **kwargs) -> bool:
        return True

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
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]] = []

        for img in images:
            H, W = img.shape[:2]
            fallback_anchor = np.zeros((H, W, 3), dtype=np.uint8)

            result = self.composer.compose(image=img)

            if result is None or result.anomaly_mask is None or np.sum(result.anomaly_mask) == 0:
                debug_info = None
                if return_debug:
                    debug_info = {
                        "success": False,
                        "reason": "composer_failed",
                        "strategy": self.name,
                    }
                results.append((img.copy(), fallback_anchor, np.zeros((H, W), dtype=np.uint8), debug_info))
                continue

            debug_info = None
            if return_debug:
                debug_info = {
                    "success": True,
                    "strategy": self.name,
                    "island_area": int(np.sum(result.island.mask)),
                    "placement_center": result.placement_center,
                    "dtd_path": result.dtd_path,
                }
                if hasattr(result, "debug"):
                    debug_info.update(result.debug)

            anchor_crop = np.zeros((H, W, 3), dtype=np.uint8)
            results.append((result.augmented_image, anchor_crop, result.anomaly_mask, debug_info))

        return results
