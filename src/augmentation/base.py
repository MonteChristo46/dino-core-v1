import abc
import numpy as np
from typing import Tuple


class BaseAugmentor(abc.ABC):
    """Abstract base class for synthetic anomaly augmentors."""

    @abc.abstractmethod
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation. Returns (augmented_image, anomaly_mask)."""
        ...
