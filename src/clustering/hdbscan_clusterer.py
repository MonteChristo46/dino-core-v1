from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.cluster import HDBSCAN

from .base import BaseClusterer, ClusteringResult

LOGGER = logging.getLogger(__name__)


class HDBSCANClusterer(BaseClusterer):
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        umap_dims: int = 20,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
    ):
        super().__init__(umap_dims=umap_dims, umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

    def cluster(
        self,
        features,
        mask_14: np.ndarray,
    ) -> ClusteringResult:
        if isinstance(features, torch.Tensor):
            feat_np = features.cpu().numpy()
        else:
            feat_np = np.asarray(features)
        if feat_np.shape[0] == 0 or feat_np.shape[1] == 0:
            return ClusteringResult(
                cluster_map_masked=np.full((14, 14), -1, dtype=int),
                unique_clusters=[],
                centroids=np.array([]),
                pca=None,
                scaler=None,
                info=self._empty_info("insufficient_features"),
            )

        reduced_scaled, umap_model, scaler = self._preprocess_features(
            feat_np, self.umap_dims, self.umap_n_neighbors, self.umap_min_dist
        )

        effective_min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=effective_min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            allow_single_cluster=True,
            copy=True,
        )
        labels = clusterer.fit_predict(reduced_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float(np.mean(labels == -1))

        if np.all(labels == -1):
            labels = np.zeros_like(labels)

        mask_flat = mask_14.reshape(-1) > 0
        cluster_map_masked, unique_clusters = self._build_cluster_maps(labels, mask_flat)
        centroids = self._compute_centroids(labels, mask_flat, reduced_scaled, unique_clusters)

        cluster_persistence = []
        if hasattr(clusterer, "cluster_persistence_") and clusterer.cluster_persistence_ is not None:
            cluster_persistence = clusterer.cluster_persistence_.tolist()

        info: dict[str, Any] = {
            "algorithm": "hdbscan",
            "min_cluster_size": int(self.min_cluster_size),
            "min_samples": int(effective_min_samples),
            "cluster_selection_epsilon": float(self.cluster_selection_epsilon),
            "metric": self.metric,
            "umap_dims": int(self.umap_dims),
            "umap_n_neighbors": int(self.umap_n_neighbors),
            "umap_min_dist": float(self.umap_min_dist),
            "num_clusters": int(n_clusters),
            "noise_ratio": float(noise_ratio),
            "cluster_persistence": cluster_persistence,
        }

        return ClusteringResult(
            cluster_map_masked=cluster_map_masked,
            unique_clusters=unique_clusters,
            centroids=centroids,
            umap_model=umap_model,
            scaler=scaler,
            info=info,
        )

    @staticmethod
    def _empty_info(reason: str) -> dict[str, Any]:
        return {
            "algorithm": "hdbscan",
            "min_cluster_size": 0,
            "min_samples": 0,
            "cluster_selection_epsilon": 0.0,
            "metric": "euclidean",
            "umap_dims": 0,
            "umap_n_neighbors": 0,
            "umap_min_dist": 0.0,
            "num_clusters": 0,
            "noise_ratio": 1.0,
            "cluster_persistence": [],
            "error_reason": reason,
        }
