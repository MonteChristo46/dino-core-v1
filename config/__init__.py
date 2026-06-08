"""Centralized configuration dataclasses for dino-core-v1.

All configuration is defined as stdlib dataclasses with sensible defaults.
Edit values directly in this file or import individual configs and override
what you need.

Usage:
    from config import SAM2Config, PerlinConfig, GammaSizeParams

    # Use defaults
    sam2_cfg = SAM2Config()

    # Override specific values
    sam2_cfg = SAM2Config(model_size="tiny", min_area_ratio=0.005)
"""

from __future__ import annotations

# SAM2 masking
from .sam2_config import SAM2Config

# SAM3 exemplar masking
from .sam3_config import SAM3Config

# Perlin noise island generation
from .perlin_config import GammaSizeParams, PerlinConfig

# Feature clustering
from .clustering_config import ClusteringConfig

# PatchCore training
from .patchcore_config import PatchCoreConfig

# Dataset generation pipeline
from .pipeline_config import PipelineConfig

# Augmentation strategies
from .augmentation_config import AugmentationConfig, BlenderConfig, FillerConfig

__all__ = [
    # SAM2
    "SAM2Config",
    # SAM3
    "SAM3Config",
    # Perlin
    "GammaSizeParams",
    "PerlinConfig",
    # Clustering
    "ClusteringConfig",
    # PatchCore
    "PatchCoreConfig",
    # Pipeline
    "PipelineConfig",
    # Augmentation
    "AugmentationConfig",
    "BlenderConfig",
    "FillerConfig",
]
