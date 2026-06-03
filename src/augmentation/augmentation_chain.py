"""Chain-of-responsibility orchestrator for augmentation strategies.

Allows multiple augmentors to be registered in priority order. Each strategy
declares whether it can handle a given image, and the chain executes the first
matching strategy.

Usage:
    chain = AugmentationChain([
        SemanticPatchStrategy(backbone, clusterer, composer, random_composer, device),
        RandomPatchStrategy(composer),
    ])

    # Per-image (convenience, delegates to process_batch internally)
    result = chain(image, return_debug=True, sam2_mask=mask, coverage=0.3)

    # Batch (efficient, shared DINOv3 forward pass)
    results = chain.process_batch(images, per_image_kwargs, return_debug=True)
"""

import logging
from typing import List, Tuple, Optional, Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


class AugmentationStrategy(Protocol):
    """Interface for augmentation strategies in the chain.

    A strategy must implement:
    - ``can_apply(image, **kwargs)``: returns True if this strategy can handle the image
    - ``__call__(image, return_debug, **kwargs)``: per-image augmentation
    - ``process_batch(images, per_image_kwargs, return_debug)``: batched augmentation
    """

    def can_apply(self, image: np.ndarray, **kwargs) -> bool:
        ...

    def __call__(
        self,
        image: np.ndarray,
        return_debug: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
        ...

    def process_batch(
        self,
        images: List[np.ndarray],
        per_image_kwargs: List[dict],
        return_debug: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]]:
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
            if strategy.can_apply(image, **kwargs):
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

    def process_batch(
        self,
        images: List[np.ndarray],
        per_image_kwargs: List[dict],
        return_debug: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]]:
        """Process a batch of images, routing each to the first matching strategy.

        Groups images by which strategy's ``can_apply()`` matches, then delegates
        to each strategy's ``process_batch()`` for efficient batched processing
        (e.g., shared DINOv3 forward pass).

        Args:
            images: List of RGB images (H, W, 3).
            per_image_kwargs: List of per-image kwargs dicts (same length as images).
            return_debug: If True, include debug info in each result.

        Returns:
            List of (augmented_image, anchor_crop, anomaly_mask, debug_info) tuples,
            one per input image in the same order.
        """
        assert len(images) == len(per_image_kwargs), "images and per_image_kwargs must match in length"

        n = len(images)
        results: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]]] = [None] * n

        for strategy in self.strategies:
            strategy_indices = [
                i for i in range(n)
                if results[i] is None and strategy.can_apply(images[i], **per_image_kwargs[i])
            ]
            if not strategy_indices:
                continue

            strategy_images = [images[i] for i in strategy_indices]
            strategy_kwargs = [per_image_kwargs[i] for i in strategy_indices]

            LOGGER.debug(
                f"AugmentationChain: delegating {len(strategy_indices)} images to {type(strategy).__name__}"
            )

            strategy_results = strategy.process_batch(
                strategy_images, strategy_kwargs, return_debug=return_debug,
            )

            for idx, result in zip(strategy_indices, strategy_results):
                results[idx] = result

        for i in range(n):
            if results[i] is None:
                H, W = images[i].shape[:2]
                results[i] = (
                    images[i].copy(),
                    np.zeros((H, W, 3), dtype=np.uint8),
                    np.zeros((H, W), dtype=np.uint8),
                    {"success": False, "reason": "no_matching_strategy"} if return_debug else None,
                )

        return results

    def __len__(self) -> int:
        return len(self.strategies)

    def __repr__(self) -> str:
        strategy_names = [type(s).__name__ for s in self.strategies]
        return f"AugmentationChain([{', '.join(strategy_names)}])"
