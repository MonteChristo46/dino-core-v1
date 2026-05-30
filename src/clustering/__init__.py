from __future__ import annotations

from typing import Any

from .base import BaseClusterer, ClusteringResult
from .hdbscan_clusterer import HDBSCANClusterer


def create_clusterer(algorithm: str = "hdbscan", **kwargs: Any) -> BaseClusterer:
    if algorithm == "hdbscan":
        return HDBSCANClusterer(**kwargs)
    raise ValueError(f"Unknown clustering algorithm: {algorithm!r}. Use 'hdbscan' or 'dbscan'.")


__all__ = ["BaseClusterer", "ClusteringResult", "create_clusterer"]
