import abc
import numpy as np


class BaseMasker(abc.ABC):
    """Abstract base class for object segmentation maskers."""

    @abc.abstractmethod
    def get_mask(self, img_np: np.ndarray) -> np.ndarray:
        """Calculate binary object mask (0-255 uint8)."""
        ...
