"""Tests for MVTec-style datasets WITHOUT ground_truth masks (e.g. 40mp test
data used to exercise the pipeline).

Guards three things:
- dataset construction tolerates a missing ground_truth/<defect> folder;
- evaluation stays honest: instance metrics valid, pixel metrics -> NaN;
- every plot (notably top_anomalous.png) renders without touching masks.
"""
import csv
import logging
import os
import sys

import numpy as np
import pytest
from click.testing import CliRunner

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts", "tiled_patchcore")
for _p in (_SCRIPTS_DIR, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shared  # noqa: E402
import run_tiled_patchcore  # noqa: E402
import load_and_evaluate  # noqa: E402

from patchcore.datasets import mvtec  # noqa: E402


def _write_maskless_tree(root, size=128, classname="mosque"):
    """train/good + test/good + test/bad — and NO ground_truth folder."""
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
    save("test/good/2")
    save("test/bad/3", anomaly=True)
    save("test/bad/4", anomaly=True)
    assert not (root / classname / "ground_truth").exists()


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------

def test_test_split_loads_without_ground_truth(tmp_path):
    _write_maskless_tree(tmp_path)
    dataset = mvtec.MVTecDataset(
        str(tmp_path), classname="mosque", resize=None, imagesize=None,
        split=mvtec.DatasetSplit.TEST,
    )
    assert len(dataset) == 3
    bad_entries = [t for t in dataset.data_to_iterate if t[1] != "good"]
    assert len(bad_entries) == 2
    assert all(t[3] is None for t in bad_entries)  # "no mask" marker

    for entry in dataset.data_to_iterate:
        sample = dataset[dataset.data_to_iterate.index(entry)]
        assert sample["mask"].sum() == 0.0
        assert tuple(sample["mask"].shape) == (1, 128, 128)


def test_missing_ground_truth_warns(tmp_path, caplog):
    _write_maskless_tree(tmp_path)
    with caplog.at_level(logging.WARNING, logger="patchcore.datasets.mvtec"):
        mvtec.MVTecDataset(
            str(tmp_path), classname="mosque", split=mvtec.DatasetSplit.TEST
        )
    warnings = [r for r in caplog.records if "No ground_truth masks" in r.getMessage()]
    assert len(warnings) == 1
    assert "bad" in warnings[0].getMessage()


def test_train_split_never_needs_ground_truth(tmp_path):
    _write_maskless_tree(tmp_path)
    dataset = mvtec.MVTecDataset(str(tmp_path), classname="mosque",
                                 split=mvtec.DatasetSplit.TRAIN)
    assert len(dataset) == 2


# ---------------------------------------------------------------------------
# end-to-end: train + eval on a maskless tree (the 40mp use case)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def maskless_run(tmp_path_factory):
    """One shared train->eval run (slowest part), reused by the checks below."""
    tmp = tmp_path_factory.mktemp("maskless")
    data_root = tmp / "data"
    data_root.mkdir()
    _write_maskless_tree(data_root)
    results_root = tmp / "results"

    runner = CliRunner()
    result = runner.invoke(
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
            "--save_segmentation_images",
        ],
    )
    assert result.exit_code == 0, result.output
    return results_root / "tiled", data_root


def _category_row(path):
    with open(path) as handle:
        return [r for r in csv.DictReader(handle) if r["Category"] == "mvtec_mosque"][0]


def test_results_have_valid_instance_metrics_nan_pixel(maskless_run):
    run_dir, _ = maskless_run
    row = _category_row(run_dir / "results.csv")
    assert 0.0 <= float(row["instance_auroc"]) <= 1.0
    assert np.isnan(float(row["full_pixel_auroc"]))
    assert np.isnan(float(row["anomaly_pixel_auroc"]))
    assert np.isfinite(float(row["optimal_threshold"]))
    # Mean row must survive an all-NaN pixel column (no NaN-poisoned mean)
    mean_row = [r for r in csv.DictReader(open(run_dir / "results.csv"))
                if r["Model"] == "Mean"][0]
    assert np.isnan(float(mean_row["full_pixel_auroc"]))
    assert 0.0 <= float(mean_row["instance_auroc"]) <= 1.0


def test_all_plots_and_analytics_generated_without_masks(maskless_run):
    run_dir, _ = maskless_run
    # top_anomalous/tile_grid/score_dist/roc are mask-free by design
    for plot in ("tile_grid", "score_dist", "roc", "top_anomalous"):
        assert (run_dir / "plots" / f"mvtec_mosque_{plot}.png").exists(), plot
    with open(run_dir / "per_image_scores.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert sum(int(r["is_anomaly"]) for r in rows) == 2
    cm = _category_row(run_dir / "confusion_matrix.csv")
    assert sum(int(cm[k]) for k in ("TP", "FP", "TN", "FN")) == 3
    curves = np.load(run_dir / "curves" / "mvtec_mosque_curves.npz")
    assert curves["pixel_fpr"].size == 0
    assert np.isnan(float(curves["pixel_auroc"]))
    assert np.isfinite(float(curves["image_auroc"]))
    # segmentation images fall back to 2-panel rendering (no fake GT column)
    seg_dir = run_dir / "segmentation_images" / "mvtec_mosque"
    assert seg_dir.is_dir()
    jpgs = [f for _root, _dirs, files in os.walk(seg_dir) for f in files
            if f.endswith(".jpg")]
    assert len(jpgs) == 3  # one render per test image, grouped under TP/TN dirs


def test_reload_eval_on_maskless_tree(maskless_run):
    run_dir, data_root = maskless_run
    runner = CliRunner()
    result = runner.invoke(
        load_and_evaluate.main,
        [
            "--model_root", str(run_dir / "models"),
            "--data_path", str(data_root),
            "--batch_size", "1",
            "--num_workers", "0",
            "--cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    row = _category_row(run_dir / "eval" / "results.csv")
    assert np.isnan(float(row["full_pixel_auroc"]))
    assert 0.0 <= float(row["instance_auroc"]) <= 1.0
    assert (run_dir / "eval" / "plots" / "mvtec_mosque_top_anomalous.png").exists()
