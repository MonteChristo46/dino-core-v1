"""Configuration dataclasses for feature clustering (UMAP + HDBSCAN)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """HDBSCAN clusterer configuration for DINOv3 patch feature clustering.

    Controls both UMAP dimensionality reduction and HDBSCAN clustering parameters.
    Used to identify semantically distinct regions within object masks for
    targeted synthetic anomaly placement.

    Usage:
        from config import ClusteringConfig

        config = ClusteringConfig(
            min_cluster_size=5,
            umap_dims=10,
        )
    """

    # -------------------------------------------------------------------------
    # HDBSCAN clustering parameters
    # -------------------------------------------------------------------------

    # Minimum number of samples to form a cluster.
    # Smaller values = more (smaller) clusters. Larger values = fewer,
    # more robust clusters.
    min_cluster_size: int = 3

    # Minimum number of samples in a neighborhood for a point to be
    # considered a core point. If None, defaults to min_cluster_size.
    min_samples: int | None = None

    # Maximum distance threshold for cluster selection.
    # Clusters with persistence below this are merged.
    # 0.0 = no merging based on distance.
    cluster_selection_epsilon: float = 0.0

    # Distance metric for HDBSCAN.
    # Common choices: "euclidean", "manhattan", "cosine".
    metric: str = "euclidean"

    # -------------------------------------------------------------------------
    # UMAP dimensionality reduction parameters
    # -------------------------------------------------------------------------

    # Target dimensionality after UMAP reduction.
    # Lower values = more aggressive compression, faster clustering,
    # but may lose fine-grained structure.
    umap_dims: int = 20

    # Number of neighbors for UMAP's local manifold approximation.
    # Lower values = focus on local structure. Higher values = focus on
    # global structure.
    umap_n_neighbors: int = 15

    # Minimum distance between embedded points in UMAP space.
    # 0.0 = points can cluster tightly. Higher values = more spread out.
    umap_min_dist: float = 0.0
