"""Configuration dataclasses for the dataset generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Dataset generation pipeline configuration.

    Controls paths, multiprocessing settings, batching, and quality thresholds
    for the synthetic anomaly dataset generation pipeline.

    Usage:
        from config import PipelineConfig

        config = PipelineConfig(
            dataset_root="dataset/VisA_pytorch",
            output_dir="dataset/semantic_synthetic_anomalies_v7",
            num_test=100,
        )
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    # Root directory of the source dataset (MVTec/VisA format with
    # train/good and test/good subdirectories).
    dataset_root: str = "dataset/VisA_pytorch"

    # Output directory for the generated synthetic anomaly dataset.
    # Creates MVTec-format structure:
    #   <output_dir>/<category>/
    #     train/good/
    #     test/good/
    #     test/bad/
    #     ground_truth/bad/
    output_dir: str = "dataset/semantic_synthetic_anomalies_v7"

    # -------------------------------------------------------------------------
    # Dataset size
    # -------------------------------------------------------------------------

    # Total number of test images (good + bad) to generate per category.
    # Split evenly: num_test // 2 good, num_test - num_test // 2 bad.
    num_test: int = 100

    # Minimum SAM2 coverage ratio for an image to be eligible for
    # semantic patch augmentation. Images below this threshold fall
    # back to random patch augmentation.
    min_coverage_threshold: float = 0.05

    # Minimum number of HDBSCAN clusters required for semantic patch
    # strategy to be applicable.
    min_clusters: int = 2

    # -------------------------------------------------------------------------
    # Multiprocessing
    # -------------------------------------------------------------------------

    # Number of reader worker processes (image loading).
    reader_workers: int = 8

    # Number of writer worker processes (image saving).
    writer_workers: int = 8

    # Maximum size of the read queue (pending image paths).
    read_queue_size: int = 32

    # Maximum size of the decoded queue (loaded images waiting for GPU).
    decoded_queue_size: int = 32

    # Maximum size of the write queue (processed images waiting to save).
    write_queue_size: int = 64

    # Interval (seconds) between heartbeat log messages showing pipeline
    # progress and queue statistics.
    heartbeat_sec: float = 5.0

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    # Number of images to process in a single GPU batch.
    # Larger batches = more efficient GPU utilization but more memory.
    batch_size: int = 64

    # Random seed for reproducibility.
    seed: int = 42
