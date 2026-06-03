"""Chain-of-responsibility orchestrator for augmentation strategies.

Allows multiple augmentors to be registered in priority order. Each strategy
declares whether it can handle a given image, and the chain executes the first
matching strategy.

Usage:
    chain = AugmentationChain([
        IslandPaster(donor_pool, masker, composer, min_mask_ratio=0.15, seed=42),
        TexturePaster(donor_pool, composer, seed=42),
    ])
    result = chain(image, return_debug=True)
"""

import logging
from typing import List, Tuple, Optional, Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


class AugmentationStrategy(Protocol):
    """Interface for augmentation strategies in the chain.

    A strategy must implement:
    - ``can_apply(image)``: returns True if this strategy can handle the image
    - ``__call__(image, return_debug)``: applies augmentation and returns result tuple
    """

    def can_apply(self, image: np.ndarray) -> bool:
        ...

    def __call__(
        self,
        image: np.ndarray,
        return_debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        ...


class AugmentationChain:
    """Chain-of-responsibility orchestrator for augmentation strategies.

    Tries each registered strategy in order. The first strategy whose
    ``can_apply()`` returns True is executed. If no strategy matches,
    returns the original image unchanged with a failure debug dict.

    Args:
        strategies: Ordered list of augmentation strategies. Earlier items
            have higher priority.
    """

    def __init__(self, strategies: List[AugmentationStrategy]):
        self.strategies = strategies

    def __call__(
        self,
        image: np.ndarray,
        return_debug: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        """Execute the first matching augmentation strategy that succeeds.

        Tries each strategy whose ``can_apply()`` returns True. If a strategy
        executes but fails internally (empty anomaly mask or ``success=False``
        in debug info), falls through to the next strategy.

        Args:
            image: RGB image (H, W, 3) to augment.
            return_debug: If True, return additional debug info dict.
            **kwargs: Passed through to the strategy's ``__call__``.

        Returns:
            (augmented_image, anchor_crop, anomaly_mask, debug_info)
        """
        H, W = image.shape[:2]
        fallback = (
            image.copy(),
            np.zeros((H, W, 3), dtype=np.uint8),
            np.zeros((H, W), dtype=np.uint8),
            {"success": False, "reason": "no_matching_strategy"} if return_debug else None,
        )

        for strategy in self.strategies:
            if strategy.can_apply(image):
                LOGGER.debug(f"AugmentationChain: trying {type(strategy).__name__}")
                result = strategy(image, return_debug=return_debug, **kwargs)
                augmented, anchor_crop, anomaly_mask, debug_info = result

                succeeded = False
                if return_debug and debug_info:
                    succeeded = debug_info.get("success", False)
                elif (anomaly_mask > 0).sum() > 0:
                    succeeded = True

                if succeeded:
                    LOGGER.debug(f"AugmentationChain: {type(strategy).__name__} succeeded")
                    return result

                LOGGER.debug(f"AugmentationChain: {type(strategy).__name__} failed, trying next")

        LOGGER.debug("AugmentationChain: all strategies failed, returning original image")
        return fallback

    def __len__(self) -> int:
        return len(self.strategies)

    def __repr__(self) -> str:
        strategy_names = [type(s).__name__ for s in self.strategies]
        return f"AugmentationChain([{', '.join(strategy_names)}])"
