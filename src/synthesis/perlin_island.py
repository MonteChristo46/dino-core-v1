"""OpenSimplex-based island generator for synthetic anomalies."""

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .mask_ops import dilate_mask, smooth_mask
from .noise import domain_warp_simplex, multi_octave_simplex
from .shape_priors import GRADIENT_TYPES, build_shape_prior


@dataclass
class GammaSizeParams:
    """Gamma distribution parameters for sampling synthetic anomaly island area.

    The island area is sampled from a Gamma distribution, then converted to pixel
    count and clamped by min_ratio / max_ratio. The Gamma distribution is ideal
    because it is bounded at zero and right-skewed — most islands are small with
    occasional larger ones — and both the shape and scale are independently tunable.

    Sampling formula (see ``_sample_area``)::

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

    shape: float = 1.1
    scale: float = 1.5
    min_ratio: float = 0.03
    max_ratio: float = 0.15
    max_resample_attempts: int = 64


@dataclass
class IslandResult:
    """Output of a single island generation call."""

    intensity: np.ndarray
    mask: np.ndarray
    center: tuple[float, float]
    area_ratio: float
    gradient_type: str
    shape_prior: np.ndarray


class PerlinIslandGenerator:
    """Generate synthetic anomaly islands from OpenSimplex noise.

    Despite the historical class name, the implementation is OpenSimplex-only.
    """

    def __init__(
        self,
        size_params: Optional[GammaSizeParams] = None,
        noise_octaves: int = 4,
        noise_persistence: float = 0.5,
        noise_lacunarity: float = 2.0,
        noise_base_res: tuple[int, int] = (8, 8),
        gradient_types: Optional[tuple[str, ...]] = None,
        sea_level_range: tuple[float, float] = (0.05, 0.25),
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        self.size_params = size_params or GammaSizeParams()
        self.noise_octaves = noise_octaves
        self.noise_persistence = noise_persistence
        self.noise_lacunarity = noise_lacunarity
        self.noise_base_res = noise_base_res
        self.gradient_types = gradient_types or GRADIENT_TYPES
        self.sea_level_range = sea_level_range
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = random.Random(seed)

    def _sample_center(
        self,
        height: int,
        width: int,
        mask: Optional[np.ndarray],
    ) -> tuple[float, float]:
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return (height / 2, width / 2)
            idx = self._rng.randint(0, len(ys) - 1)
            return (float(ys[idx]), float(xs[idx]))

        return (
            self._rng.uniform(height * 0.1, height * 0.9),
            self._rng.uniform(width * 0.1, width * 0.9),
        )

    def _sample_area(self, mask_area: int, total_area: int) -> float:
        min_pixels = max(4, int(total_area * self.size_params.min_ratio))
        max_pixels = max(int(mask_area * self.size_params.max_ratio), min_pixels)

        # Rejection sampling avoids a hard pile-up at max_ratio caused by
        # clamping large gamma draws to the same upper bound.
        pixels = min_pixels
        for _ in range(max(1, self.size_params.max_resample_attempts)):
            area = self._rng.gammavariate(self.size_params.shape, self.size_params.scale)
            pixels = int(area * total_area)
            if min_pixels <= pixels <= max_pixels:
                return float(pixels)

        # Fallback keeps behavior bounded even if the truncation range is tight.
        return float(max(min_pixels, min(pixels, max_pixels)))

    def _sample_threshold(self, positive: np.ndarray, target_pixels: int) -> float:
        support_pixels = max(int(positive.size), 1)
        q = np.clip(1.0 - (target_pixels / support_pixels), 0.01, 0.92)
        area_threshold = float(np.quantile(positive, q))
        sea_fraction = self._rng.uniform(*self.sea_level_range)
        sea_threshold = sea_fraction * float(np.maximum(positive.max(), 1e-8))
        return 0.75 * area_threshold + 0.25 * sea_threshold

    def __call__(
        self,
        height: int,
        width: int,
        mask: Optional[np.ndarray] = None,
    ) -> IslandResult:
        constraint = mask if mask is not None else np.ones((height, width), dtype=np.uint8)
        mask_area = int(constraint.sum())
        total_area = height * width

        center = self._sample_center(height, width, constraint)
        island_area = self._sample_area(mask_area, total_area)
        radius = math.sqrt(island_area / math.pi)

        grad_type = self._rng.choice(self.gradient_types)
        shape_prior = build_shape_prior(grad_type, height, width, center, radius, self._rng)

        seed = self._rng.randint(0, 2**31 - 1)
        noise = multi_octave_simplex(
            shape=(height, width),
            base_res=self.noise_base_res,
            octaves=self.noise_octaves,
            persistence=self.noise_persistence,
            lacunarity=self.noise_lacunarity,
            seed=seed,
            device=self.device,
        )

        warp_strength = max(1.0, min(radius * self._rng.uniform(0.18, 0.42), 14.0))
        warped = domain_warp_simplex(noise, strength_px=warp_strength, seed=seed + 97).cpu().numpy()
        shaped = warped * shape_prior

        candidate = shaped if mask is None else np.where(mask > 0, shaped, -np.inf)
        positive = candidate[np.isfinite(candidate) & (candidate > 0)]

        if positive.size == 0:
            binary = np.zeros((height, width), dtype=np.uint8)
        else:
            target_pixels = max(1, int(min(island_area, mask_area)))
            threshold = self._sample_threshold(positive, target_pixels)
            binary = (candidate > threshold).astype(np.uint8)

        if mask is not None:
            binary = binary & mask

        smooth_rounds = 2 if radius > 28 else 1
        binary = smooth_mask(binary, rounds=smooth_rounds)
        binary = dilate_mask(binary, rounds=2 if radius > 24 else 1)

        if mask is not None:
            binary = binary & mask

        island_pixels = int(binary.sum())
        intensity = np.zeros((height, width), dtype=np.float32)
        if island_pixels > 0:
            island_vals = shaped[binary > 0]
            iv_min = float(island_vals.min())
            iv_max = float(island_vals.max())
            if iv_max - iv_min > 1e-8:
                intensity[binary > 0] = ((island_vals - iv_min) / (iv_max - iv_min)).astype(np.float32)
            else:
                intensity[binary > 0] = 1.0

        return IslandResult(
            intensity=intensity,
            mask=binary,
            center=center,
            area_ratio=island_pixels / mask_area if mask_area > 0 else 0.0,
            gradient_type=grad_type,
            shape_prior=shape_prior.astype(np.float32),
        )
