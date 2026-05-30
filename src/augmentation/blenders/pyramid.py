import logging
from typing import Tuple

import cv2
import numpy as np

from .base import BlendResult, Blender

LOGGER = logging.getLogger(__name__)


class PyramidBlender(Blender):
    name = "pyramid"

    def __init__(self, levels: int = 5):
        self.levels = levels

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

        x1 = max(0, global_cx - patch_w // 2)
        y1 = max(0, global_cy - patch_h // 2)
        x2 = min(W, x1 + patch_w)
        y2 = min(H, y1 + patch_h)

        dst_roi = destination[y1:y2, x1:x2].copy()
        patch_roi = source[: y2 - y1, : x2 - x1]
        mask_roi = mask[: y2 - y1, : x2 - x1]

        blended_roi = self._pyramid_blend(patch_roi, dst_roi, mask_roi)

        composited = destination.copy()
        composited[y1:y2, x1:x2] = blended_roi

        anomaly_mask = np.zeros((H, W), dtype=np.uint8)
        anomaly_mask[y1:y2, x1:x2] = (mask_roi > 0).astype(np.uint8) * 255

        LOGGER.debug(
            f"Pyramid blend: center=({global_cx},{global_cy}), roi=({x1},{y1},{x2-x1},{y2-y1}), levels={self.levels}"
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

    def _pyramid_blend(self, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
        src_pyramid = self._build_laplacian_pyramid(src, self.levels)
        dst_pyramid = self._build_laplacian_pyramid(dst, self.levels)
        mask_pyramid = self._build_gaussian_pyramid((mask > 0).astype(np.float32), self.levels)

        blended_pyramid = []
        for src_layer, dst_layer, mask_layer in zip(src_pyramid, dst_pyramid, mask_pyramid):
            if mask_layer.ndim == 2:
                mask_layer = np.repeat(mask_layer[:, :, np.newaxis], 3, axis=2)
            if mask_layer.shape[:2] != src_layer.shape[:2]:
                mask_layer = cv2.resize(mask_layer, (src_layer.shape[1], src_layer.shape[0]), interpolation=cv2.INTER_LINEAR)
            blended = src_layer * mask_layer + dst_layer * (1.0 - mask_layer)
            blended_pyramid.append(blended)

        return self._reconstruct_from_laplacian(blended_pyramid)

    @staticmethod
    def _build_gaussian_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
        pyramid = [image.astype(np.float32)]
        for _ in range(num_levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid

    @staticmethod
    def _build_laplacian_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
        gaussian_pyramid = PyramidBlender._build_gaussian_pyramid(image, num_levels)
        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian_pyramid.append(gaussian_pyramid[i] - expanded)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    @staticmethod
    def _reconstruct_from_laplacian(laplacian_pyramid: list[np.ndarray]) -> np.ndarray:
        result = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(result, dstsize=size)
            result = expanded + laplacian_pyramid[i]
        return np.clip(result, 0, 255).astype(np.uint8)
