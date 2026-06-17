"""Generate MVTec-format dataset with synthetic anomalies.

Multiprocessing pipeline (single-GPU safe mode):
  reader processes -> GPU main process (SAM2 + DINO/UMAP/HDBSCAN) -> writer processes

Output:
  dataset/semantic_synthetic_anomalies/<category>/
    ground_truth/bad/      # <idx>_mask.png (single-channel, 0/255)
    test/good/             # Real good images (from source test/good)
    test/bad/              # Synthetic anomaly images
    train/good/            # All original train/good images (copied)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from queue import Empty
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import ClusteringConfig, FillerConfig, PerlinConfig, PipelineConfig, SAM2Config, augmentation_config
from src.augmentation import AugmentationChain
from src.augmentation.blenders.feather import FeatherBlender
from src.augmentation.blenders.seamless_clone import SeamlessCloneBlender
from src.augmentation.strategies.random_patch import RandomPatchStrategy
from src.augmentation.strategies.semantic_patch import SemanticPatchStrategy
from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.base import init_cuml_accel
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.masking.sam2_masker import SAM2Masker
from src.synthesis.fillers import DTDTextureFiller, SourceTextureFiller
from src.synthesis.patch_composer import PatchComposer
from src.synthesis.perlin_island import PerlinIslandGenerator
from src.synthesis.random_composer import DTD_ROOT, RandomPatchComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "BigD")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "BigD_synthetic_anomalies_v8")

ALL_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
#ALL_CATEGORIES = [
#    "bottle", "cable", "capsule", "carpet", "grid",
#    "hazelnut", "leather", "metal_nut", "pill", "screw",
#    "tile", "toothbrush", "transistor", "wood", "zipper",
#]


STOP_TOKEN = "__STOP__"


@dataclass
class RollingTimer:
    values: deque[float]

    def add(self, value: float) -> None:
        self.values.append(value)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


def collect_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def safe_qsize(queue_obj: Any) -> int:
    try:
        return int(queue_obj.qsize())
    except (NotImplementedError, OSError, AttributeError):
        return -1


def reader_worker(read_q: mp.Queue, decoded_q: mp.Queue, worker_id: int) -> None:
    while True:
        task = read_q.get()
        if task == STOP_TOKEN:
            return

        job_id = task["job_id"]
        img_path = task["img_path"]
        started = time.time()
        try:
            image_np = np.array(Image.open(img_path).convert("RGB"))
            decoded_q.put(
                {
                    "job_id": job_id,
                    "img_path": img_path,
                    "image_np": image_np,
                    "worker_id": worker_id,
                    "decode_s": time.time() - started,
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover
            decoded_q.put(
                {
                    "job_id": job_id,
                    "img_path": img_path,
                    "image_np": None,
                    "worker_id": worker_id,
                    "decode_s": time.time() - started,
                    "error": str(exc),
                }
            )


def writer_worker(write_q: mp.Queue, ack_q: mp.Queue, worker_id: int) -> None:
    while True:
        task = write_q.get()
        if task == STOP_TOKEN:
            return

        started = time.time()
        idx = task["idx"]
        img_out = task["img_out"]
        mask_out = task["mask_out"]
        augmented_img = task["augmented_img"]
        anomaly_mask = task["anomaly_mask"]

        try:
            Image.fromarray(augmented_img).save(img_out)
            mask_uint8 = (anomaly_mask > 0).astype(np.uint8) * 255
            Image.fromarray(mask_uint8, mode="L").save(mask_out)
            ack_q.put(
                {
                    "idx": idx,
                    "worker_id": worker_id,
                    "write_s": time.time() - started,
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover
            ack_q.put(
                {
                    "idx": idx,
                    "worker_id": worker_id,
                    "write_s": time.time() - started,
                    "error": str(exc),
                }
            )


def process_batch(
    buffer: list[dict[str, Any]],
    chain: AugmentationChain,
    random_strategy: RandomPatchStrategy,
    min_coverage_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float]]:
    batch_start = time.time()
    semantic_eligible = [item for item in buffer if item["coverage"] >= min_coverage_threshold]
    random_only = [item for item in buffer if item["coverage"] < min_coverage_threshold]
    results_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]] = {}

    semantic_s = 0.0
    random_s = 0.0

    if semantic_eligible:
        semantic_start = time.time()
        images_np = [item["image_np"] for item in semantic_eligible]
        per_image_kwargs = [
            {"sam2_mask": item["merged_mask"], "coverage": item["coverage"]}
            for item in semantic_eligible
        ]
        with torch.inference_mode():
            semantic_results = chain.process_batch(images_np, per_image_kwargs, return_debug=True)
        semantic_s = time.time() - semantic_start
        for idx, result in enumerate(semantic_results):
            results_map[id(semantic_eligible[idx])] = result

    if random_only:
        random_start = time.time()
        images_np = [item["image_np"] for item in random_only]
        per_image_kwargs = [{} for _ in random_only]
        with torch.inference_mode():
            random_results = random_strategy.process_batch(images_np, per_image_kwargs, return_debug=True)
        random_s = time.time() - random_start
        for idx, result in enumerate(random_results):
            results_map[id(random_only[idx])] = result

    outputs: list[dict[str, Any]] = []
    counters = {
        "semantic": 0,
        "random": 0,
        "failed": 0,
    }

    for item in buffer:
        result = results_map.get(id(item))
        if result is None:
            counters["failed"] += 1
            continue

        augmented_img, _anchor_crop, anomaly_mask, debug_info = result
        strategy_name = debug_info.get("strategy", "unknown") if debug_info else "unknown"
        succeeded = bool(debug_info and debug_info.get("success", False))

        if not succeeded:
            counters["failed"] += 1
            continue

        if anomaly_mask is None or int(np.sum(anomaly_mask)) == 0:
            counters["failed"] += 1
            continue

        if strategy_name == "semantic_patch":
            counters["semantic"] += 1
        else:
            counters["random"] += 1

        outputs.append(
            {
                "augmented_img": augmented_img,
                "anomaly_mask": anomaly_mask,
                "strategy": strategy_name,
            }
        )

    timings = {
        "batch_s": time.time() - batch_start,
        "semantic_s": semantic_s,
        "random_s": random_s,
    }
    return outputs, counters, timings


def main() -> None:
    overall_start = time.time()

    sam2_defaults = SAM2Config()
    clustering_defaults = ClusteringConfig()
    perlin_defaults = PerlinConfig()
    pipeline_defaults = PipelineConfig()
    filler_defaults = FillerConfig()

    parser = argparse.ArgumentParser(description="Generate MVTec-format synthetic anomaly dataset")
    parser.add_argument("--categories", nargs="*", default=None, help="Specific categories")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-test", type=int, default=pipeline_defaults.num_test)
    parser.add_argument("--device", default=sam2_defaults.device, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sam2-model-size", default=sam2_defaults.model_size, choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--sam2-min-area-ratio", type=float, default=sam2_defaults.min_area_ratio)
    parser.add_argument("--sam2-max-area-ratio", type=float, default=sam2_defaults.max_area_ratio)
    parser.add_argument("--sam2-pred-iou", type=float, default=sam2_defaults.pred_iou_thresh)
    parser.add_argument("--sam2-stability", type=float, default=sam2_defaults.stability_score_thresh)
    parser.add_argument("--dinov3-size", default="vitl", choices=["vits", "vitsplus", "vitb", "vitl", "vithplus"])
    parser.add_argument("--batch-size", type=int, default=pipeline_defaults.batch_size)
    parser.add_argument("--seed", type=int, default=pipeline_defaults.seed)
    parser.add_argument("--fill-strategy", choices=["source", "dtd"], default="source")
    parser.add_argument(
        "--dtd-mix-prob",
        type=float,
        default=filler_defaults.dtd_mix_prob,
        help="Probability of using DTD fill instead of source fill in semantic mode",
    )
    parser.add_argument("--blender", choices=["feather", "seamless_clone"], default="feather")
    parser.add_argument("--dtd-root", default=DTD_ROOT)
    parser.add_argument("--reader-workers", type=int, default=pipeline_defaults.reader_workers, help="Reader process count")
    parser.add_argument("--writer-workers", type=int, default=pipeline_defaults.writer_workers, help="Writer process count")
    parser.add_argument("--read-queue-size", type=int, default=pipeline_defaults.read_queue_size)
    parser.add_argument("--decoded-queue-size", type=int, default=pipeline_defaults.decoded_queue_size)
    parser.add_argument("--write-queue-size", type=int, default=pipeline_defaults.write_queue_size)
    parser.add_argument("--heartbeat-sec", type=float, default=pipeline_defaults.heartbeat_sec)
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES
    num_good = args.num_test // 2
    num_bad = args.num_test - num_good

    LOGGER.info("=" * 64)
    LOGGER.info("Configuration")
    LOGGER.info("=" * 64)
    LOGGER.info("  categories: %s", categories)
    LOGGER.info("  dataset_root: %s", args.dataset_root)
    LOGGER.info("  output_dir: %s", args.output_dir)
    LOGGER.info("  num_test: %d (good=%d, bad=%d)", args.num_test, num_good, num_bad)
    LOGGER.info("  device: %s", args.device)
    LOGGER.info("  seed: %d", args.seed)
    LOGGER.info("  dinov3_size: %s", args.dinov3_size)
    LOGGER.info("  batch_size: %d", args.batch_size)
    LOGGER.info("  min_coverage_threshold: %.4f", pipeline_defaults.min_coverage_threshold)
    LOGGER.info("  min_clusters: %d", pipeline_defaults.min_clusters)
    LOGGER.info("  fill_strategy: %s", args.fill_strategy)
    LOGGER.info("  dtd_mix_prob: %.3f", args.dtd_mix_prob)
    LOGGER.info("  blender: %s", args.blender)
    LOGGER.info("  seamless_mode: mixed")
    LOGGER.info("  reader_workers: %d", args.reader_workers)
    LOGGER.info("  writer_workers: %d", args.writer_workers)
    LOGGER.info("  queue_sizes: read=%d, decoded=%d, write=%d",
                args.read_queue_size, args.decoded_queue_size, args.write_queue_size)
    LOGGER.info("  heartbeat_sec: %.1f", args.heartbeat_sec)
    LOGGER.info("")
    LOGGER.info("  SAM2:")
    LOGGER.info("    model_size: %s", args.sam2_model_size)
    LOGGER.info("    min_area_ratio: %.4f", args.sam2_min_area_ratio)
    LOGGER.info("    max_area_ratio: %.4f", args.sam2_max_area_ratio)
    LOGGER.info("    pred_iou_thresh: %.4f", args.sam2_pred_iou)
    LOGGER.info("    stability_score_thresh: %.4f", args.sam2_stability)
    LOGGER.info("    background_min_area_ratio: %.4f", sam2_defaults.background_min_area_ratio)
    LOGGER.info("    background_min_border_touch_ratio: %.4f", sam2_defaults.background_min_border_touch_ratio)
    LOGGER.info("    fallback_stability_thresholds: %s", sam2_defaults.fallback_stability_thresholds)
    LOGGER.info("")
    LOGGER.info("  Clustering:")
    LOGGER.info("    min_cluster_size: %d", clustering_defaults.min_cluster_size)
    LOGGER.info("    min_samples: %s", clustering_defaults.min_samples)
    LOGGER.info("    cluster_selection_epsilon: %.4f", clustering_defaults.cluster_selection_epsilon)
    LOGGER.info("    metric: %s", clustering_defaults.metric)
    LOGGER.info("    umap_dims: %d", clustering_defaults.umap_dims)
    LOGGER.info("    umap_n_neighbors: %d", clustering_defaults.umap_n_neighbors)
    LOGGER.info("    umap_min_dist: %.4f", clustering_defaults.umap_min_dist)
    LOGGER.info("")
    LOGGER.info("  Perlin:")
    LOGGER.info("    size_params: shape=%.4f, scale=%.4f, min_ratio=%.4f, max_ratio=%.4f",
                perlin_defaults.size_params.shape,
                perlin_defaults.size_params.scale,
                perlin_defaults.size_params.min_ratio,
                perlin_defaults.size_params.max_ratio)
    LOGGER.info("    noise_octaves: %d", perlin_defaults.noise_octaves)
    LOGGER.info("    noise_persistence: %.4f", perlin_defaults.noise_persistence)
    LOGGER.info("    noise_lacunarity: %.4f", perlin_defaults.noise_lacunarity)
    LOGGER.info("    noise_base_res: %s", perlin_defaults.noise_base_res)
    LOGGER.info("    sea_level_range: %s", perlin_defaults.sea_level_range)
    LOGGER.info("=" * 64)

    init_cuml_accel()

    LOGGER.info("Loading models...")
    sam2_weights = os.path.join(PROJECT_ROOT, "weights", "sam2")
    masker = SAM2Masker(
        model_size=args.sam2_model_size,
        weights_dir=sam2_weights,
        device=args.device,
        pred_iou_thresh=args.sam2_pred_iou,
        stability_score_thresh=args.sam2_stability,
        min_area_ratio=args.sam2_min_area_ratio,
        max_area_ratio=args.sam2_max_area_ratio,
        background_min_area_ratio=sam2_defaults.background_min_area_ratio,
        background_min_border_touch_ratio=sam2_defaults.background_min_border_touch_ratio,
        fallback_stability_thresholds=sam2_defaults.fallback_stability_thresholds,
    )

    dinov3_weights = os.path.join(PROJECT_ROOT, "weights", f"dinov3_{args.dinov3_size}")
    if not os.path.exists(dinov3_weights):
        dinov3_weights = os.path.join(PROJECT_ROOT, "weights", "dinov3")
    backbone = DinoV3Backbone(weights_dir=dinov3_weights).to(args.device)
    backbone.eval()

    clusterer = HDBSCANClusterer(
        min_cluster_size=clustering_defaults.min_cluster_size,
        min_samples=clustering_defaults.min_samples,
        cluster_selection_epsilon=clustering_defaults.cluster_selection_epsilon,
        metric=clustering_defaults.metric,
        umap_dims=clustering_defaults.umap_dims,
        umap_n_neighbors=clustering_defaults.umap_n_neighbors,
        umap_min_dist=clustering_defaults.umap_min_dist,
    )

    island_gen = PerlinIslandGenerator(
        size_params=perlin_defaults.size_params,
        noise_octaves=perlin_defaults.noise_octaves,
        noise_persistence=perlin_defaults.noise_persistence,
        noise_lacunarity=perlin_defaults.noise_lacunarity,
        noise_base_res=perlin_defaults.noise_base_res,
        sea_level_range=perlin_defaults.sea_level_range,
        seed=args.seed,
    )

    dtd_mix_prob = float(np.clip(args.dtd_mix_prob, 0.0, 1.0))
    dtd_filler = DTDTextureFiller(args.dtd_root, seed=args.seed)
    filler = dtd_filler if args.fill_strategy == "dtd" else SourceTextureFiller()
    blender = SeamlessCloneBlender(mode=augmentation_config.BlenderConfig.seamless_mode) if args.blender == "seamless_clone" else FeatherBlender(augmentation_config.BlenderConfig.feather_radius, augmentation_config.BlenderConfig.feather_alpha)
    patch_composer = PatchComposer(
        island_generator=island_gen,
        blender=blender,
        filler=filler,
        dtd_filler=(dtd_filler if args.fill_strategy == "source" else None),
        dtd_mix_prob=(dtd_mix_prob if args.fill_strategy == "source" else 0.0),
        seed=args.seed,
    )
    random_composer = RandomPatchComposer(island_generator=island_gen, dtd_root=DTD_ROOT, seed=args.seed)

    semantic_strategy = SemanticPatchStrategy(
        backbone=backbone,
        clusterer=clusterer,
        composer=patch_composer,
        random_composer=random_composer,
        device=args.device,
        min_coverage=pipeline_defaults.min_coverage_threshold,
        min_clusters=2,
    )
    random_strategy = RandomPatchStrategy(composer=random_composer)
    chain = AugmentationChain([semantic_strategy, random_strategy])

    LOGGER.info("=" * 64)
    LOGGER.info("Configuration")
    LOGGER.info("  categories: %s", categories)
    LOGGER.info("  device: %s", args.device)
    LOGGER.info("  DINOv3 weights: %s", dinov3_weights)
    LOGGER.info("  batch_size: %d", args.batch_size)
    LOGGER.info("  min_coverage_threshold: %.3f", pipeline_defaults.min_coverage_threshold)
    LOGGER.info("  reader_workers: %d", args.reader_workers)
    LOGGER.info("  writer_workers: %d", args.writer_workers)
    LOGGER.info("  queue_sizes(read/decoded/write): %d/%d/%d", args.read_queue_size, args.decoded_queue_size, args.write_queue_size)
    LOGGER.info("=" * 64)

    mp_ctx = mp.get_context("spawn")

    for category in categories:
        cat_start = time.time()
        LOGGER.info("Processing category: %s", category)

        cat_src = os.path.join(args.dataset_root, category)
        cat_out = os.path.join(args.output_dir, category)

        train_good_dir = os.path.join(cat_out, "train", "good")
        test_good_dir = os.path.join(cat_out, "test", "good")
        test_bad_dir = os.path.join(cat_out, "test", "bad")
        gt_bad_dir = os.path.join(cat_out, "ground_truth", "bad")

        if os.path.exists(cat_out):
            shutil.rmtree(cat_out)

        os.makedirs(train_good_dir, exist_ok=True)
        os.makedirs(test_good_dir, exist_ok=True)
        os.makedirs(test_bad_dir, exist_ok=True)
        os.makedirs(gt_bad_dir, exist_ok=True)

        train_images = collect_images(os.path.join(cat_src, "train", "good"))
        test_images = collect_images(os.path.join(cat_src, "test", "good"))

        for img_path in train_images:
            shutil.copy2(img_path, os.path.join(train_good_dir, os.path.basename(img_path)))
        LOGGER.info("  Copied %d train/good images", len(train_images))

        test_good_src = test_images[:num_good]
        for img_path in test_good_src:
            shutil.copy2(img_path, os.path.join(test_good_dir, os.path.basename(img_path)))
        LOGGER.info("  Copied %d test/good images", len(test_good_src))

        source_images = test_images if test_images else train_images
        if not source_images:
            LOGGER.warning("  No source images for %s, skipping bad generation", category)
            continue

        max_attempts = len(source_images) * 3
        LOGGER.info("  Source images: %d, target bad: %d, max attempts: %d", len(source_images), num_bad, max_attempts)

        read_q: mp.Queue = mp_ctx.Queue(maxsize=max(1, args.read_queue_size))
        decoded_q: mp.Queue = mp_ctx.Queue(maxsize=max(1, args.decoded_queue_size))
        write_q: mp.Queue = mp_ctx.Queue(maxsize=max(1, args.write_queue_size))
        write_ack_q: mp.Queue = mp_ctx.Queue(maxsize=max(1, args.write_queue_size))

        reader_procs = [
            mp_ctx.Process(target=reader_worker, args=(read_q, decoded_q, idx), daemon=True)
            for idx in range(max(1, args.reader_workers))
        ]
        writer_procs = [
            mp_ctx.Process(target=writer_worker, args=(write_q, write_ack_q, idx), daemon=True)
            for idx in range(max(1, args.writer_workers))
        ]

        for proc in reader_procs + writer_procs:
            proc.start()

        next_submit = 0
        next_job_id = 0
        completed_attempts = 0
        in_flight_reads = 0
        pending_write_acks = 0
        last_heartbeat = time.time()

        bad_idx = 0
        valid_count = 0
        random_count = 0
        fail_count = 0
        sam2_count = 0

        decode_timer = RollingTimer(deque(maxlen=64))
        sam2_timer = RollingTimer(deque(maxlen=64))
        batch_timer = RollingTimer(deque(maxlen=64))
        write_timer = RollingTimer(deque(maxlen=64))
        fail_reason_counter = Counter()

        batch_buffer: list[dict[str, Any]] = []

        def feed_read_queue() -> None:
            nonlocal next_submit, next_job_id, in_flight_reads
            while next_submit < max_attempts and in_flight_reads < args.read_queue_size:
                img_path = source_images[next_submit % len(source_images)]
                read_q.put({"job_id": next_job_id, "img_path": img_path})
                next_submit += 1
                next_job_id += 1
                in_flight_reads += 1

        def flush_write_acks(non_blocking: bool) -> None:
            nonlocal pending_write_acks
            while pending_write_acks > 0:
                timeout = 0.0 if non_blocking else 10.0
                try:
                    ack = write_ack_q.get(timeout=timeout)
                except Empty:
                    break
                pending_write_acks -= 1
                write_timer.add(float(ack.get("write_s", 0.0)))
                if ack.get("error"):
                    fail_reason_counter["write_error"] += 1
                    LOGGER.error("  Writer error for idx=%s: %s", ack.get("idx"), ack["error"])

        try:
            feed_read_queue()

            while (bad_idx < num_bad and completed_attempts < max_attempts) or batch_buffer:
                flush_write_acks(non_blocking=True)

                alive_readers = sum(1 for p in reader_procs if p.is_alive())
                alive_writers = sum(1 for p in writer_procs if p.is_alive())
                if alive_readers == 0:
                    LOGGER.error("All reader workers died! Aborting.")
                    break
                if alive_writers == 0:
                    LOGGER.error("All writer workers died! Aborting.")
                    break

                now = time.time()
                if now - last_heartbeat >= args.heartbeat_sec:
                    LOGGER.info(
                        "  [heartbeat] bad=%d/%d attempts=%d/%d pending=%d q(read=%d decoded=%d write=%d) "
                        "workers(r=%d/%d w=%d/%d) "
                        "mean_s(decode=%.3f sam2=%.3f batch=%.3f write=%.3f) strategy(valid=%d random=%d fail=%d)",
                        bad_idx,
                        num_bad,
                        completed_attempts,
                        max_attempts,
                        len(batch_buffer),
                        safe_qsize(read_q),
                        safe_qsize(decoded_q),
                        safe_qsize(write_q),
                        alive_readers,
                        len(reader_procs),
                        alive_writers,
                        len(writer_procs),
                        decode_timer.mean(),
                        sam2_timer.mean(),
                        batch_timer.mean(),
                        write_timer.mean(),
                        valid_count,
                        random_count,
                        fail_count,
                    )
                    last_heartbeat = now

                if bad_idx >= num_bad and not batch_buffer:
                    break

                if batch_buffer and (len(batch_buffer) >= args.batch_size or completed_attempts >= max_attempts):
                    outputs, counters, timings = process_batch(
                        batch_buffer,
                        chain,
                        random_strategy,
                        min_coverage_threshold=pipeline_defaults.min_coverage_threshold,
                    )
                    batch_timer.add(timings["batch_s"])

                    fail_count += counters["failed"]

                    saved_this_batch = 0
                    saved_semantic = 0
                    saved_random = 0
                    for output in outputs:
                        if bad_idx >= num_bad:
                            break
                        idx_str = f"{bad_idx:03d}"
                        write_q.put(
                            {
                                "idx": bad_idx,
                                "img_out": os.path.join(test_bad_dir, f"{idx_str}.png"),
                                "mask_out": os.path.join(gt_bad_dir, f"{idx_str}_mask.png"),
                                "augmented_img": output["augmented_img"],
                                "anomaly_mask": output["anomaly_mask"],
                            }
                        )
                        pending_write_acks += 1
                        if output["strategy"] == "semantic_patch":
                            saved_semantic += 1
                        else:
                            saved_random += 1
                        bad_idx += 1
                        saved_this_batch += 1

                    valid_count += saved_semantic
                    random_count += saved_random

                    LOGGER.info(
                        "  Batch done: in=%d saved=%d semantic=%d random=%d failed=%d "
                        "time=%.3fs (semantic=%.3fs random=%.3fs)",
                        len(batch_buffer),
                        saved_this_batch,
                        counters["semantic"],
                        counters["random"],
                        counters["failed"],
                        timings["batch_s"],
                        timings["semantic_s"],
                        timings["random_s"],
                    )
                    batch_buffer = []
                    feed_read_queue()
                    continue

                if completed_attempts >= max_attempts:
                    continue

                try:
                    decoded = decoded_q.get(timeout=1.0)
                except Empty:
                    continue

                in_flight_reads = max(0, in_flight_reads - 1)
                completed_attempts += 1
                decode_timer.add(float(decoded.get("decode_s", 0.0)))

                if decoded.get("error"):
                    fail_count += 1
                    fail_reason_counter["decode_error"] += 1
                    LOGGER.warning("  Reader failed job=%s path=%s error=%s", decoded.get("job_id"), decoded.get("img_path"), decoded["error"])
                    feed_read_queue()
                    continue

                image_np = decoded["image_np"]
                sam2_start = time.time()
                merged_mask, _, stats = masker.get_mask_with_stats(image_np)
                sam2_timer.add(time.time() - sam2_start)
                sam2_count += 1

                batch_buffer.append(
                    {
                        "image_np": image_np,
                        "merged_mask": merged_mask,
                        "coverage": float(stats.get("coverage", 0.0)),
                    }
                )

                feed_read_queue()

            if batch_buffer and bad_idx < num_bad:
                outputs, counters, timings = process_batch(
                    batch_buffer,
                    chain,
                    random_strategy,
                    min_coverage_threshold=pipeline_defaults.min_coverage_threshold,
                )
                batch_timer.add(timings["batch_s"])
                fail_count += counters["failed"]

                for output in outputs:
                    if bad_idx >= num_bad:
                        break
                    idx_str = f"{bad_idx:03d}"
                    write_q.put(
                        {
                            "idx": bad_idx,
                            "img_out": os.path.join(test_bad_dir, f"{idx_str}.png"),
                            "mask_out": os.path.join(gt_bad_dir, f"{idx_str}_mask.png"),
                            "augmented_img": output["augmented_img"],
                            "anomaly_mask": output["anomaly_mask"],
                        }
                    )
                    pending_write_acks += 1
                    if output["strategy"] == "semantic_patch":
                        valid_count += 1
                    else:
                        random_count += 1
                    bad_idx += 1

            flush_write_acks(non_blocking=False)

        finally:
            for _ in reader_procs:
                read_q.put(STOP_TOKEN)
            for proc in reader_procs:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()

            for _ in writer_procs:
                write_q.put(STOP_TOKEN)
            for proc in writer_procs:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()

            read_q.close()
            decoded_q.close()
            write_q.close()
            write_ack_q.close()

        LOGGER.info("  SAM2 calls: %d", sam2_count)
        LOGGER.info(
            "  Generated %d test/bad images (VALID=%d, RANDOM=%d, failed=%d)",
            bad_idx,
            valid_count,
            random_count,
            fail_count,
        )
        if fail_reason_counter:
            LOGGER.info("  Failure reasons: %s", dict(fail_reason_counter))
        LOGGER.info("  Category %s completed in %.1fs", category, time.time() - cat_start)
        LOGGER.info("")

    LOGGER.info("Done: processed %d categories in %.1fs", len(categories), time.time() - overall_start)


if __name__ == "__main__":
    main()
