"""Shape priors used to sculpt noise fields into anomaly islands."""

import math
import random

import numpy as np

from . import gradients as grad


GRADIENT_TYPES = (
    "circular",
    "elliptical",
    "linear",
    "multi_center",
    "ring",
    "freeform",
)


def build_shape_prior(
    grad_type: str,
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    rng: random.Random,
) -> np.ndarray:
    """Build a shape prior field for a given gradient type."""
    if grad_type == "circular":
        sharpness = rng.uniform(1.5, 4.0)
        return grad.circular_gradient(height, width, center, radius, sharpness)

    if grad_type == "elliptical":
        aspect = rng.uniform(1.05, 1.6)
        angle = rng.uniform(0, 2 * math.pi)
        sharpness = rng.uniform(1.5, 3.0)
        g0 = grad.elliptical_gradient(
            height, width, center, radius, aspect, angle, sharpness
        )
        g1 = grad.circular_gradient(
            height, width, center, radius * rng.uniform(0.75, 1.2), sharpness=2.0
        )
        return np.maximum(g0 * 0.75, g1 * 0.55)

    if grad_type == "linear":
        angle = rng.uniform(0, 2 * math.pi)
        band = rng.uniform(radius * 0.7, radius * 1.6)
        curvature = rng.uniform(-0.2, 0.2)
        sharpness = rng.uniform(1.5, 3.0)
        line = grad.linear_gradient(
            height, width, center, angle, band, curvature, sharpness
        )
        support = grad.elliptical_gradient(
            height,
            width,
            center,
            radius * rng.uniform(1.1, 1.8),
            aspect_ratio=rng.uniform(1.1, 2.2),
            angle=angle,
            sharpness=2.0,
        )
        return line * support

    if grad_type == "multi_center":
        n_centers = rng.randint(3, 8)
        spread = radius * 1.5
        pts = [
            (
                center[0] + rng.gauss(0, spread * 0.4),
                center[1] + rng.gauss(0, spread * 0.4),
            )
            for _ in range(n_centers)
        ]
        sharpness = rng.uniform(1.5, 3.0)
        g0 = grad.multi_center_gradient(height, width, pts, radius * 0.7, sharpness)
        support = grad.circular_gradient(height, width, center, radius * 1.4, sharpness=2.2)
        return np.maximum(g0 * 0.9, support * 0.35)

    if grad_type == "ring":
        ring_w = rng.uniform(radius * 0.30, radius * 0.60)
        sharpness = rng.uniform(1.5, 3.0)
        ring = grad.ring_gradient(height, width, center, radius * 0.65, ring_w, sharpness)
        fill = grad.circular_gradient(height, width, center, radius * 0.8, sharpness=2.8)
        return np.maximum(ring * 0.85, fill * 0.3)

    if grad_type == "freeform":
        n_pts = rng.randint(5, 12)
        spread = radius * 1.2
        pts = [
            (
                center[0] + rng.gauss(0, spread * 0.3),
                center[1] + rng.gauss(0, spread * 0.3),
            )
            for _ in range(n_pts)
        ]
        sharpness = rng.uniform(1.5, 3.0)
        g0 = grad.freeform_gradient(height, width, pts, radius * 0.6, sharpness)
        support = grad.circular_gradient(height, width, center, radius * 1.3, sharpness=2.4)
        return np.maximum(g0 * 0.9, support * 0.3)

    raise ValueError(f"Unknown gradient type: {grad_type}")
