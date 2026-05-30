import logging
from typing import Tuple

import cv2
import numpy as np

from .base import BlendResult, Blender

LOGGER = logging.getLogger(__name__)

_MODE_MAP = {
    "normal": cv2.NORMAL_CLONE,
    "mixed": cv2.MIXED_CLONE,
}


class SeamlessCloneBlender(Blender):
    name = "seamless_clone"

    def __init__(self, mode: str = "mixed"):
        self.mode = mode
        self._cv_mode = _MODE_MAP.get(mode, cv2.MIXED_CLONE)

    def blend(
        self,
        destination: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int],
    ) -> BlendResult:
        H, W = destination.shape[:2]
        global_cx, global_cy = center

        composited = cv2.seamlessClone(
            source,
            destination.copy(),
            mask,
            (global_cx, global_cy),
            self._cv_mode,
        )

        mh, mw = mask.shape[:2]
        anomaly_mask = np.zeros((H, W), dtype=np.uint8)
        ay1 = max(0, global_cy - mh // 2)
        ax1 = max(0, global_cx - mw // 2)
        ay2 = min(H, ay1 + mh)
        ax2 = min(W, ax1 + mw)
        my1 = ay1 - (global_cy - mh // 2)
        mx1 = ax1 - (global_cx - mw // 2)
        my2 = my1 + (ay2 - ay1)
        mx2 = mx1 + (ax2 - ax1)
        if my1 >= 0 and mx1 >= 0 and my2 <= mh and mx2 <= mw:
            anomaly_mask[ay1:ay2, ax1:ax2] = (mask[my1:my2, mx1:mx2] > 0).astype(np.uint8) * 255

        LOGGER.debug(f"SeamlessClone blend: mode={self.mode}, center=({global_cx},{global_cy})")

        return BlendResult(
            image=composited,
            mask=anomaly_mask,
            debug={
                "bbox": cv2.boundingRect(mask) if (mask > 0).sum() > 0 else (0, 0, 0, 0),
                "patch_shape": source.shape[:2],
                "mask_area": int((mask > 0).sum()),
                "anomaly_area": int((anomaly_mask > 0).sum()),
            },
        )
