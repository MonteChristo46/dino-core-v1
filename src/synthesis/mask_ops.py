"""Binary mask post-processing utilities."""

import numpy as np
import torch
import torch.nn.functional as F


def smooth_mask(binary: np.ndarray, rounds: int = 1) -> np.ndarray:
    """Close + open operation to remove strands and pinholes."""
    if rounds <= 0:
        return binary.astype(np.uint8)

    x = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    for _ in range(rounds):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        x = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    return (x[0, 0].numpy() > 0.5).astype(np.uint8)


def dilate_mask(binary: np.ndarray, rounds: int = 1) -> np.ndarray:
    """Dilate mask to increase footprint width."""
    if rounds <= 0:
        return binary.astype(np.uint8)

    x = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    for _ in range(rounds):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    return (x[0, 0].numpy() > 0.5).astype(np.uint8)
