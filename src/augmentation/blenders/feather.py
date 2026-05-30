import logging
from typing import Tuple

import cv2
import numpy as np

from .base import BlendResult, Blender

LOGGER = logging.getLogger(__name__)


class FeatherBlender(Blender):
    name = "feather"

    def __init__(self, feather_radius: int = 10, alpha: float = 0.5):
        self.feather_radius = feather_radius
        self.alpha = alpha

    def blend(
        self,
        destination: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int],
    ) -> BlendResult:
        H, W = destination.shape[:2]
        patch_h, patch_w = source.shape[:2]
        global_cx, global_cy = center

        mask_ys, mask_xs = np.where(mask > 0)
        if len(mask_ys) == 0:
            return BlendResult(
                image=destination.copy(),
                mask=np.zeros((H, W), dtype=np.uint8),
                debug={"bbox": (0, 0, 0, 0), "patch_shape": (patch_h, patch_w), "mask_area": 0, "anomaly_area": 0},
            )

        island_cx = int(mask_xs.mean())
        island_cy = int(mask_ys.mean())

        x1 = max(0, global_cx - island_cx)
        y1 = max(0, global_cy - island_cy)
        x2 = min(W, x1 + patch_w)
        y2 = min(H, y1 + patch_h)

        dst_roi = destination[y1:y2, x1:x2].copy()
        patch_roi = source[: y2 - y1, : x2 - x1]
        mask_roi = mask[: y2 - y1, : x2 - x1]

        mask_float = (mask_roi > 0).astype(np.float32)
        if self.feather_radius > 0:
            mask_float = cv2.GaussianBlur(
                mask_float,
                (self.feather_radius * 2 + 1, self.feather_radius * 2 + 1),
                0,
            )
            mask_float = mask_float / max(mask_float.max(), 1e-6)

        mask_3ch = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
        blended_roi = (patch_roi.astype(np.float32) * mask_3ch + dst_roi.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

        composited = destination.copy()
        composited[y1:y2, x1:x2] = blended_roi

        anomaly_mask = np.zeros((H, W), dtype=np.uint8)
        anomaly_mask[y1:y2, x1:x2] = (mask_roi > 0).astype(np.uint8) * 255

        LOGGER.debug(
            f"Feather blend: target=({global_cx},{global_cy}), island_center=({island_cx},{island_cy}), roi=({x1},{y1},{x2-x1},{y2-y1}), radius={self.feather_radius}, alpha={self.alpha}"
        )

        return BlendResult(
            image=composited,
            mask=anomaly_mask,
            debug={
                "bbox": cv2.boundingRect(mask_roi) if (mask_roi > 0).sum() > 0 else (0, 0, 0, 0),
                "patch_shape": (patch_h, patch_w),
                "mask_area": int((mask_roi > 0).sum()),
                "anomaly_area": int((anomaly_mask > 0).sum()),
            },
        )
