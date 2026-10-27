"""Tests for the tiled_patchcore user scripts (run + load_and_evaluate).

CPU-only, small synthetic MVTec trees; exercised through click's CliRunner
so the exact user-facing flags are what gets tested.
"""
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import click
import numpy as np
import pytest
import torch
from click.testing import CliRunner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts", "tiled_patchcore")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import shared  # noqa: E402
import run_tiled_patchcore  # noqa: E402
import load_and_evaluate  # noqa: E402
import cross_evaluate  # noqa: E402
import compare_cross_eval_runs  # noqa: E402

import patchcore  # noqa: E402
import patchcore.backbones  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402
import patchcore.sampler  # noqa: E402


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
# good-only category: single-class splits — AUROC/threshold undefined there
# (BigD has 17 of 27 such categories; scripts must not crash on them)
# ---------------------------------------------------------------------------

def _write_good_only_tree(root, size=128, classname="caps"):
    from PIL import Image

    pixels = np.full((size, size, 3), 90, dtype=np.uint8)
    for rel in ("train/good", "test/good"):
        path = root / classname / rel
        path.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            Image.fromarray(pixels).save(path / f"{idx}.png")


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
# anomaly-map rendering: the segmentation_images pack + top-N overlay
# ---------------------------------------------------------------------------

def test_map_cutoff_converts_score_scale_into_map_scale():
    """Scores are min-max normalized over *image scores*, maps over *all pixels*.

    Same top, different bottom — so the decision threshold cannot be read off a
    map directly; using the raw number over-flags the picture badly.
    """
    evaluation = {"norm_stats": {
        "score_min": np.array([8.0]), "score_max": np.array([40.0]),
        "map_min": np.array([3.0]), "map_max": np.array([40.0]),
    }}
    cutoff = shared.map_cutoff_for_threshold(evaluation, 0.195)
    assert math.isclose(cutoff, (0.195 * 32 + 8 - 3) / 37.0, rel_tol=1e-9)
    assert 0.195 < cutoff < 1.0  # strictly above the naive reading
    # no decision boundary exists without a two-class split
    assert shared.map_cutoff_for_threshold(evaluation, float("nan")) is None
    assert shared.map_cutoff_for_threshold(evaluation, None) is None
    assert shared.map_cutoff_for_threshold({}, 0.5) == 0.5  # stats-less callers


def _render_flat_overlay(map_value, cutoff=0.4):
    """Pixels of a uniform-gray image with an overlay of constant map_value."""
    import matplotlib.pyplot as plt

    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    heatmap = np.full((64, 64), map_value, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(1, 1), dpi=50)
    axis.axis("off")
    if map_value is None:
        axis.imshow(rgb)
    else:
        shared.overlay_anomaly_map(axis, rgb, heatmap, vmax=1.0, cutoff=cutoff)
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[20:30, 20:30, :3].copy()
    plt.close(figure)
    return pixels


def test_overlay_context_layer_leaves_sub_threshold_image_untouched():
    """The faint context layer must not dim the image where nothing is hot.

    Otherwise every render looks washed out and the red that *does* appear stops
    being trustworthy.
    """
    plain = _render_flat_overlay(None)
    faint = _render_flat_overlay(0.05)   # whole map far below the 0.4 cutoff
    hot = _render_flat_overlay(0.9)      # whole map above it
    assert np.array_equal(plain, faint)
    assert float(np.abs(hot.astype(int) - plain.astype(int)).mean()) > 20.0


def test_segmentation_pack_renders_on_one_shared_scale(tmp_path, monkeypatch):
    """Regression for the old vendored panel: imshow(segmentation) with no
    vmin/vmax, which stretched *each image's own* min/max to the full colormap —
    so a flat, low-scoring image looked exactly as alarming as a real defect.
    """
    import matplotlib

    _write_script_tree(tmp_path)
    dataset, _ = shared.build_dataset(str(tmp_path), "bottle", "TEST", good_only=False)
    count = len(dataset)
    segmentations = np.zeros((count, 128, 128), dtype=np.float32)
    segmentations[0] = 1.0                      # one saturated "defect" image
    scores = segmentations.reshape(count, -1).max(axis=1)
    records = [("TP", True)] + [("TN", False)] * (count - 1)

    shown = []
    real_imshow = matplotlib.axes.Axes.imshow

    def recording_imshow(self, image, **kwargs):
        shown.append((np.asarray(image), kwargs))
        return real_imshow(self, image, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", recording_imshow)
    savefolder = str(tmp_path / "segmentation_images" / "mvtec_bottle")
    shared.save_segmentation_images_for_category(
        savefolder, dataset, scores, segmentations, records, 0.5, cutoff=0.4,
    )

    jpgs = sorted(Path(savefolder).rglob("*.jpg"))
    assert len(jpgs) == count                       # one render per test image
    assert {p.parent.name for p in jpgs} == {"TP", "TN"}

    map_panels = [kwargs for array, kwargs in shown if array.ndim == 2]
    assert map_panels, "no anomaly-map panels were drawn"
    for kwargs in map_panels:                       # heatmap, context, flagged
        assert kwargs.get("vmin") == 0.0
        assert math.isclose(float(kwargs.get("vmax")), 1.0)
    # Original | Ground Truth | Overlay: each needs its own RGB base
    assert sum(1 for array, _ in shown if array.ndim == 3) == 3 * count


# ---------------------------------------------------------------------------
# end-to-end: train -> analytics pack -> reload via load_and_evaluate
# ---------------------------------------------------------------------------

def test_resolve_names_explicit_win():
    assert cross_evaluate._resolve_names(["/a/x", "/b/y"], ("real", "synth")) == [
        "real", "synth",
    ]


def test_resolve_names_explicit_count_mismatch():
    with pytest.raises(click.UsageError):
        cross_evaluate._resolve_names(["/a/x", "/b/y"], ("only_one",))


def test_resolve_names_unique_basenames_kept():
    assert cross_evaluate._resolve_names(["/a/runA", "/b/runB"]) == ["runA", "runB"]


def test_resolve_names_collision_gets_parent_prefix():
    names = cross_evaluate._resolve_names([
        "/x/results_tiled_bigd/bigd_macaroni2",
        "/y/results_tiled_bigd_synthetic/bigd_macaroni2",
    ])
    assert names == [
        "results_tiled_bigd__bigd_macaroni2",
        "results_tiled_bigd_synthetic__bigd_macaroni2",
    ]


def test_resolve_names_sanitized():
    assert cross_evaluate._resolve_names(["/a/my run", "/b/my run"]) == [
        "a__my_run", "b__my_run",
    ]


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


# ---------------------------------------------------------------------------
# end-to-end: cross-evaluate a tiled run and a plain run on the same test set
# ---------------------------------------------------------------------------

def _make_plain_run(data_root, plain_run):
    """Fit a tiny plain PatchCore and store it in the training-run contract
    (models/mvtec_<cat>/ + curves/mvtec_<cat>_curves.npz)."""
    model_dir = plain_run / "models" / "mvtec_bottle"
    curves_dir = plain_run / "curves"
    model_dir.mkdir(parents=True)
    curves_dir.mkdir()

    device = torch.device("cpu")
    train_set, _ = shared.build_dataset(
        str(data_root), "bottle", "TRAIN", resize=None, imagesize=64,
        good_only=True, seed=0,
    )
    train_loader = shared.build_dataloader(train_set, 1, 0)
    torch.manual_seed(0)
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 64, 64),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        anomaly_score_num_nn=3,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(0.5, device),
        nn_method=patchcore.common.FaissNN(False, 4),
    )
    model.fit(train_loader)
    model.save_to_path(str(model_dir))

    test_set, _ = shared.build_dataset(
        str(data_root), "bottle", "TEST", resize=None, imagesize=64, seed=0
    )
    test_loader = shared.build_dataloader(test_set, 1, 0)
    evaluation = shared.evaluate_models([(model, False)], test_loader)
    np.savez(
        curves_dir / "mvtec_bottle_curves.npz",
        image_fpr=evaluation["image_metrics"]["fpr"],
        image_tpr=evaluation["image_metrics"]["tpr"],
        image_thresholds=evaluation["image_metrics"]["threshold"],
        optimal_threshold=float(evaluation["image_metrics"]["optimal_threshold"]),
    )


def test_cross_evaluate_rejects_run_without_models(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_script_tree(data_root)
    bogus_run = tmp_path / "bogus_run"
    bogus_run.mkdir()  # no models/ inside

    runner = CliRunner()
    result = runner.invoke(
        cross_evaluate.main,
        [
            "--model_run", str(bogus_run),
            "--data_path", str(data_root),
            "--output_root", str(tmp_path / "out"),
            "--cpu",
        ],
    )
    assert result.exit_code != 0
    assert "models/" in result.output


def test_cross_evaluate_tiled_and_plain_e2e(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_script_tree(data_root)

    # tiled run via the real training CLI (writes models/ + curves/)
    runner = CliRunner()
    train_result = runner.invoke(
        run_tiled_patchcore.main,
        [
            "--data_path", str(data_root),
            "--results_path", str(tmp_path / "results"),
            "--backbone", "wideresnet50",
            "--tile_size", "64",
            "--coreset_percentage", "0.5",
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    tiled_run = tmp_path / "results" / "tiled"

    # plain run with the same training-run contract
    plain_run = tmp_path / "plain_run"
    _make_plain_run(data_root, plain_run)

    # cross-evaluate both on the same real test set, with explicit names
    output_root = tmp_path / "cross_eval"
    eval_result = runner.invoke(
        cross_evaluate.main,
        [
            "--model_run", str(tiled_run),
            "--model_run", str(plain_run),
            "--model_name", "tiled",
            "--model_name", "plain",
            "--data_path", str(data_root),
            "--output_root", str(output_root),
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output

    # one CrossEval folder per run, named by --model_name
    run_dirs = sorted(output_root.glob("CrossEval_*"))
    assert len(run_dirs) == 2
    names = {
        re.match(r"^CrossEval_\d{8}_\d+_(.+)$", d.name).group(1) for d in run_dirs
    }
    assert names == {"tiled", "plain"}

    infos = {}
    for run_dir in run_dirs:
        # schema compatibility with the legacy comparison charts is the contract
        run_data = compare_cross_eval_runs.load_run_data(run_dir, run_dir.name)
        assert run_data is not None, f"results.csv not parseable in {run_dir}"
        assert run_data["categories"] == ["mvtec_bottle"]
        for key in ("saved_threshold", "computed_threshold", "instance_auroc",
                    "precision", "recall", "f1"):
            assert np.isfinite(run_data[key]).all(), key
        # saved threshold must come from the training run's own curves
        with np.load(run_dir / "curves" / "mvtec_bottle_curves.npz") as curves:
            assert np.isclose(float(curves["optimal_threshold"]),
                              run_data["saved_threshold"][0])
        assert (run_dir / "confusion_matrix.csv").exists()
        assert (run_dir / "per_image_scores.csv").exists()
        # 1 good + 2 bad test images classified at the saved threshold
        with open(run_dir / "confusion_matrix.csv") as handle:
            row = [r for r in csv.DictReader(handle) if r["Category"] != "Mean"][0]
        total = sum(int(row[k]) for k in ("TP", "FP", "FN", "TN"))
        assert total == 3
        # provenance file
        with open(run_dir / "run_info.json") as handle:
            info = json.load(handle)
        infos[info["model_name"]] = info

    assert infos["tiled"]["tiled"] is True
    assert infos["tiled"]["tile_size"] == 64
    assert infos["plain"]["tiled"] is False
    assert "tile_size" not in infos["plain"]
    for info in infos.values():
        assert info["backbone"] == "wideresnet50"
        assert info["categories"] == ["bottle"]
        assert set(info["saved_thresholds"]) == {"bottle"}
        assert os.path.isabs(info["source_run_path"])

    # identical chart pack from the legacy comparison script
    comparison = output_root / "cross_eval_comparison"
    for plot in ("threshold_delta_grouped", "delta_lines", "auroc_lines",
                 "threshold_scatter", "precision_heatmap", "recall_heatmap",
                 "f1_heatmap", "delta_boxplot"):
        assert (comparison / f"{plot}.jpg").exists(), f"missing {plot}.jpg"

    # paired per-image scatter for the single run pair
    scatters = list((output_root / "paired_scatter").glob("paired_scatter_*.png"))
    assert len(scatters) == 1


# ---------------------------------------------------------------------------
# single-class test splits: train saves the model, metrics skipped; eval
# scripts skip the category instead of crashing on roc_curve
# ---------------------------------------------------------------------------

def _train_two_categories_run(tmp_path, data_root):
    """Train tiled runs over data_root (bottle + caps), return run dir."""
    runner = CliRunner()
    result = runner.invoke(
        run_tiled_patchcore.main,
        [
            "--data_path", str(data_root),
            "--results_path", str(tmp_path / "results"),
            "--backbone", "wideresnet50",
            "--tile_size", "64",
            "--coreset_percentage", "0.5",
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "results" / "tiled"


def test_run_tiled_patchcore_saves_model_for_single_class_test(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_script_tree(data_root)        # bottle: both classes
    _write_good_only_tree(data_root)     # caps: good-only test split

    run_dir = _train_two_categories_run(tmp_path, data_root)

    # the good-only category must still yield a trained, saved checkpoint …
    assert (run_dir / "models/mvtec_caps/patchcore_params.pkl").exists()
    assert (run_dir / "models/mvtec_caps/tiling_params.pkl").exists()
    # … but no curves (no threshold definable) -> cross-eval will skip it
    assert (run_dir / "curves/mvtec_bottle_curves.npz").exists()
    assert not (run_dir / "curves/mvtec_caps_curves.npz").exists()

    # results.csv carries a NaN row for caps and a real row for bottle
    with open(run_dir / "results.csv") as handle:
        rows = {r["Category"]: r for r in csv.DictReader(handle) if r["Category"]}
    assert set(rows) == {"mvtec_bottle", "mvtec_caps"}
    assert math.isnan(float(rows["mvtec_caps"]["instance_auroc"]))
    assert math.isnan(float(rows["mvtec_caps"]["optimal_threshold"]))
    assert math.isfinite(float(rows["mvtec_bottle"]["instance_auroc"]))

    # confusion CSV only contains genuinely evaluated categories
    with open(run_dir / "confusion_matrix.csv") as handle:
        cm_cats = {r["Category"] for r in csv.DictReader(handle)}
    assert cm_cats == {"mvtec_bottle", "Mean"}

    with open(run_dir / "timing.json") as handle:
        timings = json.load(handle)
    assert timings["mvtec_caps"]["predict_seconds"] is None


def test_cross_evaluate_and_load_skip_single_class_category(tmp_path):
    # trained on a root where BOTH categories have anomalies
    data_train = tmp_path / "data_train"
    data_train.mkdir()
    _write_script_tree(data_train)
    _write_script_tree(data_train, classname="caps")
    run_dir = _train_two_categories_run(tmp_path, data_train)
    assert (run_dir / "curves/mvtec_caps_curves.npz").exists()

    # eval root where caps lost its anomalies (single-class test split)
    data_eval = tmp_path / "data_eval"
    data_eval.mkdir()
    _write_script_tree(data_eval)
    _write_good_only_tree(data_eval)

    runner = CliRunner()
    output_root = tmp_path / "cross_eval"
    result = runner.invoke(
        cross_evaluate.main,
        [
            "--model_run", str(run_dir),
            "--data_path", str(data_eval),
            "--output_root", str(output_root),
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
            "--skip_comparison",
        ],
    )
    assert result.exit_code == 0, result.output

    (run_out_dir,) = list(output_root.glob("CrossEval_*"))
    run_data = compare_cross_eval_runs.load_run_data(run_out_dir, run_out_dir.name)
    assert run_data is not None
    assert run_data["categories"] == ["mvtec_bottle"]  # caps skipped, not crashed
    with open(run_out_dir / "per_image_scores.csv") as handle:
        assert {r["category"] for r in csv.DictReader(handle)} == {"bottle"}
    with open(run_out_dir / "run_info.json") as handle:
        info = json.load(handle)
    assert info["categories"] == ["bottle"]
    assert set(info["saved_thresholds"]) == {"bottle"}

    # load_and_evaluate must skip the same category gracefully
    eval_result = runner.invoke(
        load_and_evaluate.main,
        [
            "--model_root", str(run_dir / "models"),
            "--data_path", str(data_eval),
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output
    with open(run_dir / "eval" / "results.csv") as handle:
        cats = {r["Category"] for r in csv.DictReader(handle) if r["Category"]}
    assert cats == {"mvtec_bottle"}  # caps skipped with a warning, no crash


# ---------------------------------------------------------------------------
# evaluate_models: memmap (maps_dir) vs in-RAM single-model equivalence
# ---------------------------------------------------------------------------

def _tiny_tiled_model():
    from torchvision import models as tv_models

    import patchcore.tiled_patchcore

    backbone = tv_models.wide_resnet50_2(weights=None)
    backbone.name, backbone.seed = "wideresnet50", 0
    return patchcore.tiled_patchcore.TiledPatchCore.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        tile_size=64,
        overlap=32,
        tile_batch_size=4,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        anomaly_scorer_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 2),
    )


def test_evaluate_models_memmap_matches_in_ram(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_script_tree(data_root)
    train_set, _ = shared.build_dataset(
        str(data_root), "bottle", "TRAIN", resize=None, imagesize=None,
        good_only=True, seed=0,
    )
    test_set, _ = shared.build_dataset(
        str(data_root), "bottle", "TEST", resize=None, imagesize=None, seed=0
    )

    torch.manual_seed(0)
    model = _tiny_tiled_model()
    model.fit(shared.build_dataloader(train_set, 1, 0))

    eval_ram = shared.evaluate_models(
        [(model, True)], shared.build_dataloader(test_set, 1, 0)
    )
    eval_mm = shared.evaluate_models(
        [(model, True)], shared.build_dataloader(test_set, 1, 0),
        maps_dir=str(tmp_path / "maps"),
    )

    assert isinstance(eval_mm["segmentations"], np.memmap)
    assert eval_mm["masks_gt"].dtype == np.uint8
    np.testing.assert_allclose(
        np.asarray(eval_mm["segmentations"]),
        np.asarray(eval_ram["segmentations"]),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(eval_mm["masks_gt"]),
        np.asarray(eval_ram["masks_gt"]).astype(np.uint8),
    )
    np.testing.assert_allclose(eval_mm["scores"], eval_ram["scores"], rtol=1e-6)
    for key in ("auroc", "optimal_threshold"):
        assert np.isclose(eval_mm["image_metrics"][key], eval_ram["image_metrics"][key])
        assert np.isclose(eval_mm["pixel_metrics"][key], eval_ram["pixel_metrics"][key])
        assert np.isclose(
            eval_mm["anomaly_pixel_metrics"][key],
            eval_ram["anomaly_pixel_metrics"][key],
        )
    # norm_stats must keep the [n_models] shape contract (map_cutoff math)
    for key in ("score_min", "score_max", "map_min", "map_max"):
        assert eval_mm["norm_stats"][key].shape == (1,)
        np.testing.assert_allclose(
            eval_mm["norm_stats"][key], eval_ram["norm_stats"][key], rtol=1e-6
        )
