"""Configuration dataclasses for PatchCore training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchCoreConfig:
    """PatchCore anomaly detection model configuration.

    Controls backbone selection, feature extraction layers, embedding dimensions,
    patch aggregation, and nearest-neighbor scorer settings.

    Usage:
        from config import PatchCoreConfig

        config = PatchCoreConfig(
            backbone_name="dinov3_vitb",
            layers=(7, 8),
            coreset_percentage=0.1,
        )
    """

    # -------------------------------------------------------------------------
    # Backbone selection
    # -------------------------------------------------------------------------

    # Backbone network name. Examples: "dinov3_vitb", "dinov3_vitl",
    # "dinov3_vithplus", "wideresnet50".
    backbone_name: str = "dinov3_vitb"

    # Layer indices to extract features from.
    # For DINOv3 ViT models, common choices are (7, 8) or (4, 12).
    # Earlier layers capture local texture; later layers capture semantics.
    layers: tuple[int, ...] = (7, 8)

    # -------------------------------------------------------------------------
    # Embedding dimensions
    # -------------------------------------------------------------------------

    # Feature dimension from the backbone before projection.
    # Determined by backbone architecture (e.g., 768 for ViT-B, 1024 for ViT-L).
    pretrain_embed_dimension: int = 768

    # Target feature dimension after projection (if different from pretrain).
    # Set equal to pretrain_embed_dimension to skip projection.
    target_embed_dimension: int = 768

    # -------------------------------------------------------------------------
    # Feature processing
    # -------------------------------------------------------------------------

    # Preprocessing method for patch features.
    # "mean" = average pooling, "conv" = convolutional projection.
    preprocessing: str = "mean"

    # Aggregation method for patch features.
    # "mean" = average, "mlp" = MLP-based aggregation.
    aggregation: str = "mean"

    # -------------------------------------------------------------------------
    # Patch parameters
    # -------------------------------------------------------------------------

    # Spatial patch size for feature aggregation.
    # 1 = no patching (use per-patch features directly).
    # Larger values aggregate features over larger spatial regions.
    patchsize: int = 1

    # Scoring method for patch-level anomaly scores.
    # "max" = maximum score in patch, "mean" = average score.
    patchscore: str = "max"

    # Overlap ratio between adjacent patches.
    # 0.0 = no overlap. Higher values = more overlap = smoother scores.
    patchoverlap: float = 0.0

    # -------------------------------------------------------------------------
    # Nearest-neighbor anomaly scorer
    # -------------------------------------------------------------------------

    # Number of nearest neighbors for anomaly scoring.
    # Higher values = smoother scores, more robust to noise.
    anomaly_scorer_num_nn: int = 9

    # Whether to run FAISS on GPU for nearest-neighbor search.
    faiss_on_gpu: bool = True

    # Number of workers for FAISS nearest-neighbor search.
    faiss_num_workers: int = 8

    # -------------------------------------------------------------------------
    # Coreset sampling
    # -------------------------------------------------------------------------

    # Fraction of training features to keep in the coreset memory bank.
    # Lower values = smaller memory bank = faster inference but potentially
    # less accurate. Typical range: 0.01-0.1.
    coreset_percentage: float = 0.1

    # -------------------------------------------------------------------------
    # Dataset loading
    # -------------------------------------------------------------------------

    # Image resize dimension (shorter edge).
    resize: int = 384

    # Final image size after center crop (both dimensions).
    imagesize: int = 384

    # Batch size for data loading.
    batch_size: int = 2

    # Number of data loading workers.
    num_workers: int = 8

    # Train/validation split ratio.
    # 1.0 = all training data, no validation split.
    train_val_split: float = 1.0
