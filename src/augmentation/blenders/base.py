from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class BlendResult:
    image: np.ndarray
    mask: np.ndarray
    debug: dict = field(default_factory=dict)


class Blender(ABC):
    name: str = "base"

    @abstractmethod
    def blend(
        self,
        destination: np.ndarray,
        source: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int],
    ) -> BlendResult:
        """Blend source into destination using mask, centered at center.

        Returns:
            BlendResult with composited image, anomaly mask, and optional debug info.
        """
