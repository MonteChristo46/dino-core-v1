"""Gradient field functions for shaping Perlin noise islands.

Each function returns a 2D array of shape (H, W) with values in [0, 1].
Value 1.0 means "on the spine" (no suppression), 0.0 means "far away"
(fully suppressed). The generator multiplies Perlin noise by this field
to constrain the island shape.
"""

import numpy as np


def circular_gradient(
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Radial gradient from a center point.

    Produces blob-like islands. Sharpness controls how quickly the gradient
    falls off — higher values give sharper edges.
    """
    y, x = np.ogrid[:height, :width]
    cy, cx = center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized = dist / radius
    return np.clip(1.0 - normalized**sharpness, 0.0, 1.0)


def elliptical_gradient(
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    aspect_ratio: float,
    angle: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Elliptical gradient with random orientation.

    Produces elongated/smear-like defects.
    """
    y, x = np.ogrid[:height, :width]
    y = y.astype(float)
    x = x.astype(float)
    cy, cx = center

    y -= cy
    x -= cx

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a

    rx = radius * aspect_ratio
    ry = radius
    dist = np.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)
    return np.clip(1.0 - dist**sharpness, 0.0, 1.0)


def linear_gradient(
    height: int,
    width: int,
    center: tuple[float, float],
    angle: float,
    width_band: float,
    curvature: float = 0.0,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Gradient along a line (optionally curved).

    Produces scratch/streak-like defects.
    """
    y, x = np.ogrid[:height, :width]
    y = y.astype(float)
    x = x.astype(float)
    cy, cx = center

    y -= cy
    x -= cx

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a

    if curvature != 0.0:
        parabola = curvature * (x_rot / width) ** 2
        y_rot = y_rot - parabola * height

    dist = np.abs(y_rot) / width_band
    return np.clip(1.0 - dist**sharpness, 0.0, 1.0)


def multi_center_gradient(
    height: int,
    width: int,
    centers: list[tuple[float, float]],
    radius: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Union of multiple circular gradients.

    Produces scattered cluster / pitting defects.
    """
    y, x = np.ogrid[:height, :width]
    combined = np.zeros((height, width), dtype=np.float64)

    for cy, cx in centers:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        normalized = dist / radius
        field = np.clip(1.0 - normalized**sharpness, 0.0, 1.0)
        combined = np.maximum(combined, field)

    return combined


def ring_gradient(
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    ring_width: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Gradient along a ring circumference.

    Produces halo / ring-shaped defects.
    """
    y, x = np.ogrid[:height, :width]
    cy, cx = center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    deviation = np.abs(dist - radius) / ring_width
    return np.clip(1.0 - deviation**sharpness, 0.0, 1.0)


def freeform_gradient(
    height: int,
    width: int,
    control_points: list[tuple[float, float]],
    influence_radius: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Gradient following a freeform path defined by control points.

    Produces irregular organic shapes. Uses min distance to any control point
    with a smoothing kernel.
    """
    y, x = np.ogrid[:height, :width]
    combined = np.zeros((height, width), dtype=np.float64)

    for cy, cx in control_points:
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        normalized = dist / influence_radius
        field = np.clip(1.0 - normalized**sharpness, 0.0, 1.0)
        combined = np.maximum(combined, field)

    return combined
