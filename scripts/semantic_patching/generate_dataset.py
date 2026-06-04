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

from src.augmentation import AugmentationChain
from src.augmentation.strategies.random_patch import RandomPatchStrategy
from src.augmentation.strategies.semantic_patch import SemanticPatchStrategy
from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.masking.sam2_masker import SAM2Masker
from src.synthesis.fillers import DTDTextureFiller, SourceTextureFiller
from src.synthesis.patch_composer import PatchComposer
from src.synthesis.perlin_island import GammaSizeParams, PerlinIslandGenerator
from src.synthesis.random_composer import DTD_ROOT, RandomPatchComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "mvtec_small")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "semantic_synthetic_anomalies")

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MIN_COVERAGE_THRESHOLD = 0.05
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
    parser = argparse.ArgumentParser(description="Generate MVTec-format synthetic anomaly dataset")
    parser.add_argument("--categories", nargs="*", default=None, help="Specific categories")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sam2-model-size", default="small", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--sam2-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--sam2-max-area-ratio", type=float, default=0.50)
    parser.add_argument("--sam2-pred-iou", type=float, default=0.70)
    parser.add_argument("--sam2-stability", type=float, default=0.80)
    parser.add_argument("--dinov3-size", default="vitl", choices=["vits", "vitsplus", "vitb", "vitl", "vithplus"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fill-strategy", choices=["source", "dtd"], default="source")
    parser.add_argument("--dtd-root", default=DTD_ROOT)
    parser.add_argument("--reader-workers", type=int, default=2, help="Reader process count")
    parser.add_argument("--writer-workers", type=int, default=2, help="Writer process count")
    parser.add_argument("--read-queue-size", type=int, default=32)
    parser.add_argument("--decoded-queue-size", type=int, default=32)
    parser.add_argument("--write-queue-size", type=int, default=64)
    parser.add_argument("--heartbeat-sec", type=float, default=5.0)
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES
    num_good = args.num_test // 2
    num_bad = args.num_test - num_good

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
        fallback_stability_thresholds=[0.7],
    )

    dinov3_weights = os.path.join(PROJECT_ROOT, "weights", f"dinov3_{args.dinov3_size}")
    if not os.path.exists(dinov3_weights):
        dinov3_weights = os.path.join(PROJECT_ROOT, "weights", "dinov3")
    backbone = DinoV3Backbone(weights_dir=dinov3_weights).to(args.device)
    backbone.eval()

    clusterer = HDBSCANClusterer(
        min_cluster_size=3,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        umap_dims=20,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
    )

    island_gen = PerlinIslandGenerator(
        size_params=GammaSizeParams(),
        noise_octaves=4,
        noise_persistence=0.5,
        noise_lacunarity=2.0,
        noise_base_res=(8, 8),
        sea_level_range=(0.05, 0.25),
        seed=args.seed,
    )

    filler = DTDTextureFiller(args.dtd_root, seed=args.seed) if args.fill_strategy == "dtd" else SourceTextureFiller()
    patch_composer = PatchComposer(island_generator=island_gen, filler=filler, seed=args.seed)
    random_composer = RandomPatchComposer(island_generator=island_gen, dtd_root=DTD_ROOT, seed=args.seed)

    semantic_strategy = SemanticPatchStrategy(
        backbone=backbone,
        clusterer=clusterer,
        composer=patch_composer,
        random_composer=random_composer,
        device=args.device,
        min_coverage=MIN_COVERAGE_THRESHOLD,
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
    LOGGER.info("  min_coverage_threshold: %.3f", MIN_COVERAGE_THRESHOLD)
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
        no_valid_count = 0
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

                now = time.time()
                if now - last_heartbeat >= args.heartbeat_sec:
                    LOGGER.info(
                        "  [heartbeat] bad=%d/%d attempts=%d/%d q(read=%d decoded=%d write=%d) "
                        "mean_s(decode=%.3f sam2=%.3f batch=%.3f write=%.3f) strategy(valid=%d random=%d fail=%d)",
                        bad_idx,
                        num_bad,
                        completed_attempts,
                        max_attempts,
                        safe_qsize(read_q),
                        safe_qsize(decoded_q),
                        safe_qsize(write_q),
                        decode_timer.mean(),
                        sam2_timer.mean(),
                        batch_timer.mean(),
                        write_timer.mean(),
                        valid_count,
                        no_valid_count,
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
                        min_coverage_threshold=MIN_COVERAGE_THRESHOLD,
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
                    no_valid_count += saved_random

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
                    min_coverage_threshold=MIN_COVERAGE_THRESHOLD,
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
                        no_valid_count += 1
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
            "  Generated %d test/bad images (VALID=%d, NO_VALID=%d, failed=%d)",
            bad_idx,
            valid_count,
            no_valid_count,
            fail_count,
        )
        if fail_reason_counter:
            LOGGER.info("  Failure reasons: %s", dict(fail_reason_counter))
        LOGGER.info("  Category %s completed in %.1fs", category, time.time() - cat_start)
        LOGGER.info("")

    LOGGER.info("Done: processed %d categories in %.1fs", len(categories), time.time() - overall_start)


if __name__ == "__main__":
    main()
