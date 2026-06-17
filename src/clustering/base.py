from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import umap


_CUML_INSTALLED = False


def init_cuml_accel(log_level: str = "info") -> None:
    global _CUML_INSTALLED
    if _CUML_INSTALLED:
        return
    import os
    os.environ["CUML_ACCEL_LOG_LEVEL"] = log_level
    import cuml
    cuml.accel.install()
    _CUML_INSTALLED = True

GRID_SIZE = 14

LOGGER = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    cluster_map_masked: np.ndarray
    unique_clusters: list[int]
    centroids: np.ndarray
    umap_model: umap.UMAP | None
    scaler: StandardScaler | None
    info: dict[str, Any] = field(default_factory=dict)


class BaseClusterer(ABC):
    def __init__(self, umap_dims: int = 20, umap_n_neighbors: int = 15, umap_min_dist: float = 0.0):
        self.umap_dims = umap_dims
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

    @abstractmethod
    def cluster(
        self,
        features: torch.Tensor,
        mask_14: np.ndarray,
    ) -> ClusteringResult:
        """Run clustering on destination features within the masked region.

        Args:
            features: DINOv3 patch tokens, shape (GRID_SIZE*GRID_SIZE, feature_dim)
            mask_14: Binary mask at 14x14 resolution

        Returns:
            ClusteringResult with labels, centroids, and metadata
        """

    @staticmethod
    def _preprocess_features(
        feat_np: np.ndarray,
        n_components: int,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
    ) -> tuple[np.ndarray, umap.UMAP, StandardScaler]:
        n_components = max(1, min(n_components, feat_np.shape[1], feat_np.shape[0]))

        n_neighbors = min(n_neighbors, feat_np.shape[0] - 1)
        n_neighbors = max(2, n_neighbors)

        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
        )
        reduced = umap_model.fit_transform(feat_np)
        scaler = StandardScaler(copy=True)
        reduced_scaled = scaler.fit_transform(reduced)
        return reduced_scaled, umap_model, scaler

    @staticmethod
    def _compute_centroids(
        labels: np.ndarray,
        mask_flat: np.ndarray,
        reduced_scaled: np.ndarray,
        unique_clusters: list[int],
    ) -> np.ndarray:
        centroids = []
        for cluster_id in unique_clusters:
            cluster_mask = np.logical_and(labels == cluster_id, mask_flat)
            if np.any(cluster_mask):
                centroids.append(reduced_scaled[cluster_mask].mean(axis=0))
            else:
                centroids.append(reduced_scaled[labels == cluster_id].mean(axis=0))
        return np.array(centroids) if centroids else np.array([])

    @staticmethod
    def _build_cluster_maps(
        labels: np.ndarray,
        mask_flat: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        labels_masked = labels.copy()
        labels_masked[~mask_flat] = -1
        cluster_map_masked = labels_masked.reshape((GRID_SIZE, GRID_SIZE))
        unique_clusters = sorted(set(labels_masked.tolist()) - {-1})
        return cluster_map_masked, unique_clusters

    @staticmethod
    def compute_kn(cluster_map: np.ndarray, mask_14: np.ndarray) -> tuple[int, float]:
        valid = mask_14 > 0
        if not np.any(valid):
            return 0, 1.0
        labels = cluster_map[valid]
        k = len(set(labels.tolist()) - {-1})
        n = float(np.mean(labels == -1))
        return int(k), n
