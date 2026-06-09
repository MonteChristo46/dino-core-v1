"""Configuration dataclasses for augmentation strategies and blending."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlenderConfig:
    """Image blending configuration for synthetic anomaly composition.

    Controls how anomaly patches are blended into the target image.

    Usage:
        from config import BlenderConfig

        config = BlenderConfig(
            method="seamless_clone",
            feather_radius=12,
        )
    """

    # Blending method. One of:
    #   "feather" — simple alpha feathering with configurable radius.
    #   "pyramid" — Laplacian pyramid blending for smooth transitions.
    #   "seamless_clone" — OpenCV seamlessClone (Poisson blending).
    method: str = "feather"

    # Feather radius in pixels (only used when method="feather").
    # Larger values = softer edges.
    feather_radius: int = 20
    feather_alpha: int = 0.8

    # SeamlessClone mode (only used when method="seamless_clone").
    # "normal" = standard Poisson blending.
    # "mixed" = mixes gradients from source and destination.
    # "monochrome_transfer" = preserves destination color distribution.
    seamless_mode: str = "mixed"


@dataclass
class FillerConfig:
    """Fill strategy configuration for Perlin noise islands.

    Controls what texture/content fills the synthetic anomaly region.

    Usage:
        from config import FillerConfig

        config = FillerConfig(
            strategy="dtd",
            dtd_root="dataset/dtd/images",
        )
    """

    # Fill strategy. One of:
    #   "source" — crops texture from a source image in the dataset.
    #   "dtd" — samples a random texture from the Describable Textures Dataset.
    strategy: str = "source"

    # Root directory of DTD images (only used when strategy="dtd").
    dtd_root: str = "dataset/dtd/images"

    # Random seed for texture sampling reproducibility.
    seed: int = 42

    # Probability in [0, 1] of using DTD texture fill instead of source
    # texture fill in semantic PatchComposer mode.
    # 0.0 = always source, 1.0 = always DTD.
    dtd_mix_prob: float = 0.1


@dataclass
class AugmentationConfig:
    """Top-level augmentation configuration.

    Composes blender and filler settings for the augmentation chain.

    Usage:
        from config import AugmentationConfig

        config = AugmentationConfig(
            blender_method="feather",
            fill_strategy="dtd",
        )
    """

    # Blending method — see BlenderConfig.method.
    blender_method: str = "feather"

    # Feather radius — see BlenderConfig.feather_radius.
    feather_radius: int = 20

    # SeamlessClone mode — see BlenderConfig.seamless_mode.
    seamless_mode: str = "mixed"

    # Fill strategy — see FillerConfig.strategy.
    fill_strategy: str = "source"

    # DTD root directory — see FillerConfig.dtd_root.
    dtd_root: str = "dataset/dtd/images"

    # Random seed for augmentation reproducibility.
    seed: int = 42
