"""Noise utilities for synthetic anomaly generation."""

import numpy as np
import torch
import torch.nn.functional as F
from opensimplex import OpenSimplex


def multi_octave_simplex(
    shape: tuple[int, int],
    base_res: tuple[int, int],
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate multi-octave OpenSimplex noise as a torch tensor."""
    h, w = shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)

    total = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    max_amplitude = 0.0

    for octave in range(octaves):
        res_h = max(int(base_res[0] * (lacunarity**octave)), 1)
        res_w = max(int(base_res[1] * (lacunarity**octave)), 1)
        simplex = OpenSimplex(seed=seed + 104729 * octave)

        y_coords = yy * res_h
        x_coords = xx * res_w
        layer = simplex.noise2array(x_coords, y_coords).astype(np.float32)

        total += layer * amplitude
        max_amplitude += amplitude
        amplitude *= persistence

    out = total / max(max_amplitude, 1e-8)
    return torch.from_numpy(out).to(device=device)


def domain_warp_simplex(
    field: torch.Tensor,
    strength_px: float,
    seed: int,
) -> torch.Tensor:
    """Apply low-frequency OpenSimplex domain warping."""
    h, w = field.shape
    if strength_px <= 0.0:
        return field

    device = field.device
    warp_x = multi_octave_simplex(
        shape=(h, w),
        base_res=(3, 3),
        octaves=2,
        persistence=0.55,
        lacunarity=2.0,
        seed=seed + 17,
        device=device,
    )
    warp_y = multi_octave_simplex(
        shape=(h, w),
        base_res=(3, 3),
        octaves=2,
        persistence=0.55,
        lacunarity=2.0,
        seed=seed + 43,
        device=device,
    )

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )

    nx = (2.0 * strength_px / max(w - 1, 1)) * warp_x
    ny = (2.0 * strength_px / max(h - 1, 1)) * warp_y
    grid = torch.stack([xx + nx, yy + ny], dim=-1).unsqueeze(0)

    warped = F.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped[0, 0]
