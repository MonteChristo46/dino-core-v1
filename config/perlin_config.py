"""Configuration dataclasses for Perlin noise island generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GammaSizeParams:
    """Gamma distribution parameters for sampling synthetic anomaly island area.

    The island area is sampled from a Gamma distribution, then converted to pixel
    count and clamped by min_ratio / max_ratio. The Gamma distribution is ideal
    because it is bounded at zero and right-skewed — most islands are small with
    occasional larger ones — and both the shape and scale are independently tunable.

    Sampling formula (see ``PerlinIslandGenerator._sample_area``)::

        raw_sample = gammavariate(shape, scale)   # mean = shape * scale
        target_pixels = int(raw_sample * total_image_area)
        result = clamp(target_pixels, min_pixels, max_pixels)

    Attributes:
        shape: Gamma shape parameter (alpha). Controls the *consistency* of island
            sizes. At ~1.0 the distribution is exponential (heavy right tail).
            Higher values make it more bell-shaped and centered around the mean.

            - Increase (e.g. 2.0+): islands become more uniform in size, less
              variance, predictable mean.
            - Decrease (e.g. 0.5-0.8): high variance — many tiny islands mixed
              with occasional very large ones. At < 1.0 the distribution is
              heavily skewed toward zero.

        scale: Gamma scale parameter (theta). Controls the *average size* of
            islands as a fraction of total image area (mean = shape * scale).

            - Increase (e.g. 0.20): larger islands on average. Mean raw sample
              rises proportionally.
            - Decrease (e.g. 0.06): smaller islands on average.

        min_ratio: Hard floor — the island will never be smaller than this
            fraction of the total image pixel count, regardless of the Gamma
            sample.

            - Increase (e.g. 0.10): prevents tiny islands entirely.
            - Decrease (e.g. 0.01): allows very small defects like pinpricks.

        max_ratio: Hard ceiling — the island will never exceed this fraction of
            the *mask area* (not total image), regardless of the Gamma sample.

            - Increase (e.g. 0.70): allows large defects covering most of the
              masked object.
            - Decrease (e.g. 0.15): constrains islands to be small relative to
              the object.

    Practical tuning examples:

    - Stains / spots (small localized): shape=0.8, scale=0.06, min_ratio=0.01,
      max_ratio=0.15
    - Scratches / lines (medium elongated): shape=1.0, scale=0.10,
      min_ratio=0.03, max_ratio=0.25
    - Large discolorations (broad regions): shape=1.5, scale=0.20,
      min_ratio=0.08, max_ratio=0.50
    """

    # Gamma shape parameter (alpha). Controls consistency of island sizes.
    shape: float = 1.5

    # Gamma scale parameter (theta). Controls average island size fraction.
    scale: float = 0.01

    # Hard floor — minimum island area as fraction of total image pixels.
    min_ratio: float = 0.005

    # Hard ceiling — maximum island area as fraction of mask area.
    max_ratio: float = 0.025

    # Maximum number of rejection sampling attempts before falling back to
    # the last sampled value (even if out of bounds).
    max_resample_attempts: int = 64


@dataclass
class PerlinConfig:
    """PerlinIslandGenerator configuration for synthetic anomaly islands.

    Controls noise generation parameters, gradient types, sea level thresholding,
    and island size distribution.

    Usage:
        from config import PerlinConfig, GammaSizeParams

        config = PerlinConfig(
            size_params=GammaSizeParams(shape=1.0, scale=0.10),
            noise_octaves=5,
        )
    """

    # -------------------------------------------------------------------------
    # Island size distribution
    # -------------------------------------------------------------------------

    # Gamma distribution parameters controlling island area sampling.
    # See GammaSizeParams for detailed explanation of each field.
    size_params: GammaSizeParams = None  # type: ignore[assignment]  # set in __post_init__

    def __post_init__(self):
        if self.size_params is None:
            object.__setattr__(self, "size_params", GammaSizeParams())

    # -------------------------------------------------------------------------
    # Noise generation
    # -------------------------------------------------------------------------

    # Number of noise octaves (layers of detail).
    # More octaves = finer detail but slower computation.
    # Typical range: 3-6.
    noise_octaves: int = 4

    # Persistence factor — how much each octave contributes relative to the
    # previous one. Lower values = less contribution from higher frequencies.
    # Range: 0.0 to 1.0.
    noise_persistence: float = 0.5

    # Lacunarity factor — frequency multiplier between octaves.
    # Higher values = more separation between detail scales.
    # Typical range: 1.5-3.0.
    noise_lacunarity: float = 2.0

    # Base resolution (width, height) of the first noise octave.
    # Higher resolution = finer base noise pattern.
    noise_base_res: tuple[int, int] = (8, 8)

    # -------------------------------------------------------------------------
    # Shape priors
    # -------------------------------------------------------------------------

    # Tuple of allowed gradient types for island shape priors.
    # Available types: "circular", "elliptical", "linear", "multi_center",
    # "ring", "freeform".
    # The generator randomly picks one from this tuple per island.
    gradient_types: tuple[str, ...] = (
        "circular",
        "elliptical",
        "linear",
        "multi_center",
        "ring",
        "freeform",
    )

    # -------------------------------------------------------------------------
    # Thresholding
    # -------------------------------------------------------------------------

    # Range for sea level threshold as a fraction of max noise value.
    # Controls how much of the noise field becomes part of the island.
    # Lower values = larger islands. Range: (min, max) both in [0.0, 1.0].
    sea_level_range: tuple[float, float] = (0.05, 0.25)
