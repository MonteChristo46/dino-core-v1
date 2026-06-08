"""Configuration dataclasses for SAM2 object masking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAM2Config:
    """SAM2 automatic mask generator configuration.

    Controls both the underlying SAM2 model selection and all post-processing
    stages (area filtering, NMS merging, deduplication, morphological cleanup,
    background rejection, and fallback stability thresholds).

    Usage:
        from config import SAM2Config

        config = SAM2Config(
            model_size="tiny",
            min_area_ratio=0.005,
            device="cuda",
        )
    """

    # -------------------------------------------------------------------------
    # Model selection
    # -------------------------------------------------------------------------

    # SAM2 model variant. One of: "tiny", "small", "base_plus", "large".
    # Maps to SAM2.1 Hiiera configs. Smaller models are faster but less accurate.
    model_size: str = "large"

    # Directory containing SAM2 checkpoint weights (.pt files).
    weights_dir: str = "weights/sam2"

    # Computation device. One of: "cpu", "cuda", "mps".
    device: str = "cuda"

    # -------------------------------------------------------------------------
    # SAM2AutomaticMaskGenerator inference parameters
    # -------------------------------------------------------------------------

    # Number of points sampled per batch during mask generation.
    # Higher values = fewer batches = faster but more memory.
    points_per_batch: int = 64

    # Minimum IoU threshold for predicted mask quality.
    # Masks with predicted IoU below this are discarded.
    # Range: 0.0 (accept all) to 1.0 (only perfect masks).
    pred_iou_thresh: float = 0.86

    # Minimum stability score threshold.
    # Stability measures how sensitive the mask is to perturbations.
    # Higher values = stricter quality filter = fewer masks.
    # Range: 0.0 to 1.0.
    stability_score_thresh: float = 0.80

    # Number of crop layers for multi-scale mask generation.
    # 0 = no cropping (single scale). Higher values = more thorough but slower.
    crop_n_layers: int = 0

    # Downscale factor for the number of points sampled in crop layers.
    # Only relevant when crop_n_layers > 0.
    crop_n_points_downscale_factor: int = 1

    # Minimum connected region area (in pixels) to keep from SAM output.
    # Small regions below this threshold are removed.
    min_mask_region_area: int = 64

    # -------------------------------------------------------------------------
    # Post-processing: area filtering
    # -------------------------------------------------------------------------

    # Minimum mask area as a ratio of total image area.
    # Masks covering less than this fraction of the image are discarded.
    # E.g., 0.01 = masks must cover at least 1% of the image.
    min_area_ratio: float = 0.003

    # Maximum mask area as a ratio of total image area.
    # Masks covering more than this fraction are discarded (likely background).
    # E.g., 0.70 = masks must not cover more than 70% of the image.
    max_area_ratio: float = 0.80

    # -------------------------------------------------------------------------
    # Post-processing: NMS merging
    # -------------------------------------------------------------------------

    # IoU threshold for Non-Maximum Suppression merging.
    # Masks with IoU above this threshold are merged (larger one kept).
    # Lower values = more aggressive merging.
    nms_iou_thresh: float = 0.5

    # -------------------------------------------------------------------------
    # Post-processing: deduplication
    # -------------------------------------------------------------------------

    # IoU threshold for duplicate mask removal.
    # If two masks have IoU above this, the smaller one is removed.
    # Very high values (e.g., 0.95) only remove near-identical duplicates.
    dedup_iou: float = 0.95

    # -------------------------------------------------------------------------
    # Fallback strategy
    # -------------------------------------------------------------------------

    # Stability score thresholds to retry with if the initial pass produces
    # no valid mask (coverage < min_mask_ratio). Each value is tried in order.
    # Lower values produce more masks but with lower quality.
    fallback_stability_thresholds: tuple[float, ...] = (0.78,0.72,0.65)

    # Minimum coverage ratio (mask area / image area) for a result to be
    # considered valid. If the best mask has coverage below this, fallback
    # thresholds are tried.
    min_mask_ratio: float = 0.03

    # -------------------------------------------------------------------------
    # Post-processing: morphological cleanup
    # -------------------------------------------------------------------------

    # Enable morphological closing to fill holes and smooth mask boundaries.
    morph_cleanup: bool = True

    # Kernel size for morphological operations.
    # Larger values = more aggressive hole filling and smoothing.
    morph_kernel_size: int = 3

    # -------------------------------------------------------------------------
    # Post-processing: background rejection
    # -------------------------------------------------------------------------

    # Enable background-like mask filtering.
    # Removes masks that look like background (very large or touching borders).
    remove_background_like: bool = True

    # Minimum area ratio for background rejection.
    # Masks covering more than this fraction of the image are considered
    # background and removed.
    background_min_area_ratio: float = 0.80

    # Minimum border touch ratio for background rejection.
    # Masks touching more than this fraction of the image perimeter are
    # considered background and removed.
    # Range: 0.0 to 1.0.
    background_min_border_touch_ratio: float = 0.95
