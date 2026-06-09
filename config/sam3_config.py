"""Configuration dataclasses for SAM3 exemplar-based masking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAM3Config:
    """SAM3 image model configuration for exemplar-based segmentation.

    Controls model loading, grounding confidence, and exemplar prompt settings
    used in exemplar+box masking workflows.

    Usage:
        from config import SAM3Config

        config = SAM3Config(
            checkpoint_path="weights/sam3/sam3.pt",
            confidence_threshold=0.6,
        )
    """

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    # Path to the SAM3 checkpoint file (.pt).
    checkpoint_path: str = "weights/sam3/sam3.pt"

    # Whether to load weights from HuggingFace Hub instead of local file.
    # When False, uses checkpoint_path directly.
    load_from_hf: bool = False

    # -------------------------------------------------------------------------
    # Inference parameters
    # -------------------------------------------------------------------------

    # Minimum confidence score for keeping detections.
    # Detections with sigmoid(logits) below this threshold are discarded.
    # Range: 0.0 (accept all) to 1.0 (only highest confidence).
    confidence_threshold: float = 0.5

    # -------------------------------------------------------------------------
    # Exemplar prompt settings
    # -------------------------------------------------------------------------

    # Number of exemplar images to use for building the visual prompt.
    # More exemplars = richer visual features but slower.
    n_exemplars: int = 4

    # Number of target images to process per exemplar set.
    n_targets: int = 4

    # Default bounding box (normalized [cx, cy, w, h]) for extracting
    # exemplar features when no explicit box prompt is provided.
    # Format: (center_x, center_y, width, height) in [0, 1] range.
    exemplar_box: tuple[float, float, float, float] = (0.5, 0.5, 0.6, 0.6)
