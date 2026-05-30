import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import BaseMasker

LOGGER = logging.getLogger(__name__)

_CONFIG_NAME_MAP = {
    "tiny": "sam2.1/sam2.1_hiera_t.yaml",
    "small": "sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "large": "sam2.1/sam2.1_hiera_l.yaml",
}

_CHECKPOINT_MAP = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}


class _SAM2Model:
    def __init__(
        self,
        model_size: str,
        weights_dir: str,
        device: str,
        points_per_batch: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        crop_n_layers: int,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
    ):
        self.model_size = model_size
        self.weights_dir = weights_dir
        self.device = device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self._mask_generator = None

    def _load_model(self):
        if self._mask_generator is not None:
            return

        if "mps" in self.device.lower():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        import sam2

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        sam2_path = os.path.dirname(sam2.__file__)
        configs_path = os.path.join(sam2_path, "configs")
        initialize_config_dir(config_dir=configs_path, version_base="1.2")

        size_value = self.model_size if self.model_size in _CONFIG_NAME_MAP else "small"
        config_name = _CONFIG_NAME_MAP[size_value]
        checkpoint_file = _CHECKPOINT_MAP[size_value]
        checkpoint_path = os.path.join(self.weights_dir, checkpoint_file)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint_path}.")

        model = build_sam2(config_name, checkpoint_path, device=self.device)

        self._mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=self.crop_n_layers,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            min_mask_region_area=self.min_mask_region_area,
        )

    @staticmethod
    def _resolve_image(image_input):
        if isinstance(image_input, (str, Path)):
            return np.array(Image.open(image_input).convert("RGB"))
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return np.stack([image_input] * 3, axis=-1)
            return image_input
        if isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        raise TypeError(
            f"Unsupported input type: {type(image_input)}. "
            "Expected str, Path, numpy array, or PIL Image."
        )

    def predict_everything(self, image_input, stability_score_thresh: Optional[float] = None):
        self._load_model()
        image = self._resolve_image(image_input)

        if stability_score_thresh is None:
            with torch.inference_mode():
                return self._mask_generator.generate(image)

        original_thresh = self._mask_generator.stability_score_thresh
        try:
            self._mask_generator.stability_score_thresh = stability_score_thresh
            with torch.inference_mode():
                return self._mask_generator.generate(image)
        finally:
            self._mask_generator.stability_score_thresh = original_thresh


class SAM2Masker(BaseMasker):
    """SAM2-based segmentation masker with full post-processing pipeline."""

    def __init__(
        self,
        model_size: str = "small",
        weights_dir: str = "weights/sam2",
        device: str = "cpu",
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.90,
        stability_score_thresh: float = 0.97,
        crop_n_layers: int = 0,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.4,
        nms_iou_thresh: float = 0.5,
        dedup_iou: float = 0.95,
        fallback_stability_thresholds: Optional[List[float]] = None,
        min_mask_ratio: float = 0.05,
        morph_cleanup: bool = True,
        morph_kernel_size: int = 3,
        remove_background_like: bool = True,
        background_min_area_ratio: float = 0.25,
        background_min_border_touch_ratio: float = 0.15,
        min_mask_region_area: int = 100,
    ):
        self.model_size = model_size
        self.weights_dir = weights_dir
        self.device = device
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.nms_iou_thresh = nms_iou_thresh
        self.dedup_iou = dedup_iou
        self.fallback_stability_thresholds = (
            fallback_stability_thresholds
            if fallback_stability_thresholds is not None
            else [0.9, 0.85]
        )
        self.min_mask_ratio = min_mask_ratio
        self.morph_cleanup = morph_cleanup
        self.morph_kernel_size = morph_kernel_size
        self.remove_background_like = remove_background_like
        self.background_min_area_ratio = background_min_area_ratio
        self.background_min_border_touch_ratio = background_min_border_touch_ratio
        self.min_mask_region_area = min_mask_region_area
        self._model = None
        LOGGER.info("SAM2Masker initialized with device=%s, model_size=%s", self.device, self.model_size)

    def _get_model(self) -> _SAM2Model:
        if self._model is None:
            self._model = _SAM2Model(
                model_size=self.model_size,
                weights_dir=self.weights_dir,
                device=self.device,
                points_per_batch=self.points_per_batch,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                min_mask_region_area=self.min_mask_region_area,
            )
        return self._model

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _area_filter(masks, image_area: int, min_area_ratio: float, max_area_ratio: float):
        filtered = []
        for mask in masks:
            area = np.count_nonzero(mask)
            area_ratio = area / max(image_area, 1)
            if area_ratio < min_area_ratio:
                continue
            if area_ratio > max_area_ratio:
                continue
            filtered.append(mask)
        return filtered

    @staticmethod
    def _nms_merge(masks, iou_threshold: float = 0.9):
        if len(masks) == 0:
            return masks
        to_remove = set()
        for i in range(len(masks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(masks)):
                if j in to_remove:
                    continue
                iou = SAM2Masker._compute_iou(masks[i], masks[j])
                if iou > iou_threshold:
                    if np.count_nonzero(masks[i]) >= np.count_nonzero(masks[j]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        return [masks[i] for i in range(len(masks)) if i not in to_remove]

    @staticmethod
    def _dedup_masks(masks, iou_threshold: float = 0.95):
        if len(masks) == 0:
            return masks
        to_remove = set()
        for i in range(len(masks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(masks)):
                if j in to_remove:
                    continue
                if SAM2Masker._compute_iou(masks[i], masks[j]) > iou_threshold:
                    to_remove.add(j)
        return [masks[i] for i in range(len(masks)) if i not in to_remove]

    @staticmethod
    def _filter_background_like_masks(
        masks,
        image_shape,
        min_area_ratio=0.6,
        min_border_touch_ratio=0.5,
    ):
        h, w = image_shape[:2]
        image_area = h * w
        perimeter = (2 * h) + (2 * w) - 4
        filtered = []

        for mask in masks:
            m = mask > 0
            area_ratio = np.count_nonzero(m) / max(image_area, 1)
            border_count = (
                np.count_nonzero(m[0, :])
                + np.count_nonzero(m[-1, :])
                + np.count_nonzero(m[:, 0])
                + np.count_nonzero(m[:, -1])
            )
            border_touch_ratio = border_count / max(perimeter, 1)
            if area_ratio >= min_area_ratio or border_touch_ratio >= min_border_touch_ratio:
                continue
            filtered.append(m)

        return filtered

    @staticmethod
    def _morph_cleanup_masks(
        masks,
        image_shape,
        min_area_ratio=0.0,
        max_area_ratio=0.95,
        kernel_size=3,
    ):
        import cv2

        h, w = image_shape[:2]
        image_area = h * w
        kernel_size = max(1, int(kernel_size))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned_masks = []

        for mask in masks:
            m = mask > 0
            if m.shape[:2] != (h, w):
                m = cv2.resize(
                    (m.astype(np.uint8) * 255),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

            mask_uint8 = (m.astype(np.uint8)) * 255
            if np.count_nonzero(mask_uint8) == 0:
                continue

            closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(
                closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours or hierarchy is None:
                continue

            filled = np.zeros((h, w), dtype=np.uint8)
            hierarchy = hierarchy[0]
            for idx, _contour in enumerate(contours):
                parent = hierarchy[idx][3]
                if parent == -1:
                    cv2.drawContours(filled, contours, idx, 255, thickness=-1)
                else:
                    cv2.drawContours(filled, contours, idx, 0, thickness=-1)

            cleaned_mask = filled > 0
            cleaned_area_ratio = np.count_nonzero(cleaned_mask) / max(image_area, 1)
            if cleaned_area_ratio < min_area_ratio:
                continue
            if cleaned_area_ratio > max_area_ratio:
                continue
            cleaned_masks.append(cleaned_mask)

        return cleaned_masks

    @staticmethod
    def _merge_masks(final_masks, image_shape) -> np.ndarray:
        h, w = image_shape[:2]
        merged = np.zeros((h, w), dtype=bool)
        for mask in final_masks:
            m = mask > 0
            if m.shape[:2] != (h, w):
                import cv2

                m = cv2.resize(
                    (m.astype(np.uint8) * 255),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            merged = np.logical_or(merged, m)
        return (merged.astype(np.uint8)) * 255

    def _process_masks(self, results, image_shape):
        if not results:
            return None

        raw_all = [(ann["segmentation"] > 0) for ann in results]
        raw_masks = raw_all

        if self.remove_background_like:
            raw_masks = self._filter_background_like_masks(
                raw_masks,
                image_shape,
                min_area_ratio=self.background_min_area_ratio,
                min_border_touch_ratio=self.background_min_border_touch_ratio,
            )

        if not raw_masks:
            return None

        image_area = image_shape[0] * image_shape[1]
        area_filtered = self._area_filter(
            raw_masks, image_area, self.min_area_ratio, self.max_area_ratio
        )
        nms_merged = self._nms_merge(area_filtered, self.nms_iou_thresh)
        final_masks = self._dedup_masks(nms_merged, self.dedup_iou)

        if not final_masks:
            return None

        if self.morph_cleanup:
            final_masks = self._morph_cleanup_masks(
                final_masks,
                image_shape,
                min_area_ratio=self.min_area_ratio,
                max_area_ratio=self.max_area_ratio,
                kernel_size=self.morph_kernel_size,
            )

        if not final_masks:
            return None

        return {
            "raw_all": raw_all,
            "raw_masks": raw_masks,
            "area_filtered": area_filtered,
            "nms_merged": nms_merged,
            "final_masks": final_masks,
        }

    @staticmethod
    def _to_uint8_masks(masks: List[np.ndarray], image_shape) -> List[np.ndarray]:
        h, w = image_shape[:2]
        output = []
        for mask in masks:
            m = mask > 0
            if m.shape[:2] != (h, w):
                import cv2

                m = cv2.resize(
                    (m.astype(np.uint8) * 255),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            output.append((m.astype(np.uint8)) * 255)
        return output

    def get_mask_with_stats(
        self, img_np: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, float]]:
        model = self._get_model()
        image_area = img_np.shape[0] * img_np.shape[1]

        results = model.predict_everything(img_np)
        processed = self._process_masks(results, img_np.shape)

        best_merged = None
        best_masks: List[np.ndarray] = []
        best_stats: Dict[str, float] = {
            "raw_input": 0,
            "raw": 0,
            "area": 0,
            "nms": 0,
            "final": 0,
            "coverage": 0.0,
            "used_fallback": 0.0,
            "fallback_stability": -1.0,
            "raw_masks": [],
            "final_masks": [],
        }

        def build_stats(proc, merged_mask, used_fallback, fallback_stability):
            coverage = np.count_nonzero(merged_mask) / max(image_area, 1)
            return {
                "raw_input": len(proc["raw_all"]),
                "raw": len(proc["raw_masks"]),
                "area": len(proc["area_filtered"]),
                "nms": len(proc["nms_merged"]),
                "final": len(proc["final_masks"]),
                "coverage": coverage,
                "used_fallback": 1.0 if used_fallback else 0.0,
                "fallback_stability": float(fallback_stability),
                "raw_masks": proc["raw_masks"],
                "final_masks": proc["final_masks"],
            }

        if processed is not None:
            merged = self._merge_masks(processed["final_masks"], img_np.shape)
            stats = build_stats(processed, merged, False, -1.0)
            best_merged = merged
            best_masks = self._to_uint8_masks(processed["final_masks"], img_np.shape)
            best_stats = stats
            if stats["coverage"] >= self.min_mask_ratio:
                return best_merged, best_masks, best_stats

        for fallback_thresh in self.fallback_stability_thresholds:
            LOGGER.info(
                "SAM2 retrying with stability_score_thresh=%s", fallback_thresh
            )
            results = model.predict_everything(
                img_np, stability_score_thresh=fallback_thresh
            )
            processed = self._process_masks(results, img_np.shape)
            if processed is None:
                continue

            merged = self._merge_masks(processed["final_masks"], img_np.shape)
            stats = build_stats(processed, merged, True, fallback_thresh)

            if stats["coverage"] >= self.min_mask_ratio:
                return merged, self._to_uint8_masks(processed["final_masks"], img_np.shape), stats

            if best_merged is None or stats["coverage"] > best_stats["coverage"]:
                best_merged = merged
                best_masks = self._to_uint8_masks(processed["final_masks"], img_np.shape)
                best_stats = stats

        if best_merged is not None:
            return best_merged, best_masks, best_stats

        LOGGER.info("SAM2 returned no mask after all fallback thresholds")
        return np.zeros(img_np.shape[:2], dtype=np.uint8), [], best_stats

    def get_mask(self, img_np: np.ndarray) -> np.ndarray:
        merged, _masks, _stats = self.get_mask_with_stats(img_np)
        return merged

    def get_masks(self, img_np: np.ndarray) -> List[np.ndarray]:
        _merged, masks, _stats = self.get_mask_with_stats(img_np)
        return masks
