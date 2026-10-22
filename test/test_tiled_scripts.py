"""Tests for the tiled_patchcore user scripts (run + load_and_evaluate).

CPU-only, small synthetic MVTec trees; exercised through click's CliRunner
so the exact user-facing flags are what gets tested.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch
from click.testing import CliRunner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "tiled_patchcore")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import shared  # noqa: E402
import run_tiled_patchcore  # noqa: E402
import load_and_evaluate  # noqa: E402

import patchcore  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402


# ---------------------------------------------------------------------------
# fake MVTec-format tree: includes train/bad on purpose (memory-bank leak!)
# ---------------------------------------------------------------------------

def _write_script_tree(root, size=128, classname="bottle"):
    from PIL import Image

    base = np.full((size, size, 3), 40, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    texture = (np.sin(xx / 7.0) * 30 + np.cos(yy / 5.0) * 30).astype(np.int16)[..., None]

    def save(rel, anomaly=False):
        path = root / classname / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.clip(base.astype(np.int16) + texture + (60 if anomaly else 0), 0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(path.with_suffix(".png"))
        return path

    save("train/good/0")
    save("train/good/1")
    save("train/bad/leak")  # must be dropped by the good-only filter
    save("test/good/2")
    save("test/bad/3", anomaly=True)
    save("test/bad/4", anomaly=True)
    mask_dir = root / classname / "ground_truth" / "bad"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[50:80, 40:70] = 255
    for name in ("3", "4"):
        Image.fromarray(mask).save(mask_dir / f"{name}.png")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_resolve_embed_dims_auto_and_override():
    assert shared.resolve_embed_dims("dinov3_vits") == (384, 384)
    assert shared.resolve_embed_dims("dinov3_vitl") == (1024, 1024)
    assert shared.resolve_embed_dims("wideresnet50") == (1024, 1024)
    # explicit overrides always win
    assert shared.resolve_embed_dims("dinov3_vitb", 512, 256) == (512, 256)
    with pytest.raises(ValueError):
        shared.resolve_embed_dims("some_exotic_net")


def test_default_layers():
    assert shared.default_layers("dinov3_vitb") == ["7", "8"]
    assert shared.default_layers("wideresnet50") == ["layer2", "layer3"]
    assert shared.default_layers("mystery") is None


def test_discover_categories_requires_train_dir(tmp_path):
    _write_script_tree(tmp_path)
    (tmp_path / "not_a_category").mkdir()
    assert shared.discover_categories(str(tmp_path)) == ["bottle"]


def test_good_only_filter_drops_train_bad(tmp_path):
    _write_script_tree(tmp_path)
    dataset, dropped = shared.build_dataset(
        str(tmp_path), "bottle", "TRAIN", good_only=True
    )
    assert dropped == 1
    assert all(t[1] == "good" for t in dataset.data_to_iterate)
    assert len(dataset) == 2

    unfiltered, dropped_none = shared.build_dataset(str(tmp_path), "bottle", "TRAIN")
    assert dropped_none == 0
    assert len(unfiltered) == 3


def test_native_geometry_passthrough(tmp_path):
    _write_script_tree(tmp_path)
    dataset, _ = shared.build_dataset(str(tmp_path), "bottle", "TEST", good_only=False)
    assert dataset.imagesize is None
    assert tuple(dataset[0]["image"].shape) == (3, 128, 128)
    assert shared.dataset_image_hw(dataset) == (128, 128)


def test_dataset_geometry_for_models():
    plain = patchcore.patchcore.PatchCore(torch.device("cpu"))
    plain.input_shape = (3, 224, 224)
    assert shared.dataset_geometry_for_model(plain, is_tiled=False) == (None, 224)
    assert shared.dataset_geometry_for_model(None, is_tiled=True) == (None, None)


# ---------------------------------------------------------------------------
# end-to-end: train -> analytics pack -> reload via load_and_evaluate
# ---------------------------------------------------------------------------

def test_run_then_load_and_evaluate_e2e(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_script_tree(data_root)
    results_root = tmp_path / "results"

    runner = CliRunner()
    train_result = runner.invoke(
        run_tiled_patchcore.main,
        [
            "--data_path", str(data_root),
            "--results_path", str(results_root),
            "--backbone", "wideresnet50",
            "--tile_size", "64",
            "--coreset_percentage", "0.5",
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    run_dir = results_root / "tiled"
    for artifact in (
        "results.csv",
        "confusion_matrix.csv",
        "per_image_scores.csv",
        "memory_bank.json",
        "timing.json",
        "curves/mvtec_bottle_curves.npz",
        "plots/mvtec_bottle_tile_grid.png",
        "plots/mvtec_bottle_score_dist.png",
        "plots/mvtec_bottle_roc.png",
        "plots/mvtec_bottle_top_anomalous.png",
        "models/mvtec_bottle/patchcore_params.pkl",
        "models/mvtec_bottle/nnscorer_search_index.faiss",
        "models/mvtec_bottle/tiling_params.pkl",
    ):
        assert (run_dir / artifact).exists(), f"missing {artifact}"

    # good-only discipline must be visible in the pooled feature count:
    # 2 train images x 9 tiles x 64 patches (WR50 layer2 @64px = 8x8)
    with open(run_dir / "memory_bank.json") as handle:
        bank = json.load(handle)["mvtec_bottle"]
    assert bank["tiles_per_image"] == 9
    assert bank["pooled_feature_rows"] == 2 * 9 * 64
    assert bank["coreset_rows"] == bank["pooled_feature_rows"] // 2

    with open(run_dir / "per_image_scores.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # 1 good + 2 bad test images
    assert all(r["category"] == "bottle" for r in rows)
    # confusion matrix must be populated WITHOUT the segmentation-images flag
    with open(run_dir / "confusion_matrix.csv") as handle:
        cm_rows = [r for r in csv.DictReader(handle) if r["Category"] == "mvtec_bottle"]
    counts = cm_rows[0]
    assert int(counts["TP"]) + int(counts["FP"]) + int(counts["TN"]) + int(counts["FN"]) == 3

    # ---- reload path ------------------------------------------------------
    eval_result = runner.invoke(
        load_and_evaluate.main,
        [
            "--model_root", str(run_dir / "models"),
            "--data_path", str(data_root),
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output

    eval_dir = run_dir / "eval"
    assert (eval_dir / "results.csv").exists()
    assert (eval_dir / "per_image_scores.csv").exists()
    assert (eval_dir / "plots/mvtec_bottle_tile_grid.png").exists()

    def _instance_auroc(path):
        with open(path) as handle:
            row = [r for r in csv.DictReader(handle) if r["Category"] == "mvtec_bottle"][0]
        return float(row["instance_auroc"])

    # a reloaded checkpoint must reproduce the fresh-run metric
    assert np.isclose(
        _instance_auroc(run_dir / "results.csv"),
        _instance_auroc(eval_dir / "results.csv"),
        rtol=1e-6,
    )
