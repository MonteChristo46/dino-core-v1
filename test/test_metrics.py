"""Tests for patchcore.metrics — the pixel-budget subsampling path.

CPU-only; run via ``./local_run_test.sh`` or
``PYTHONPATH=src pytest test/test_metrics.py``.
"""
import numpy as np

import patchcore.metrics


def _toy_pixel_data(n=6, h=40, w=40, seed=0):
    """Perfectly separable toy pixel data: positives shifted out of the
    background score range, so any reasonable subsample still gives AUROC 1."""
    rng = np.random.default_rng(seed)
    segs = rng.random((n, h, w)).astype(np.float32)
    masks = np.zeros((n, 1, h, w), dtype=np.float32)
    for i in range(n):
        if i % 2 == 0:
            masks[i, 0, 5:15, 5:15] = 1.0
            segs[i, 5:15, 5:15] += 2.0
    return segs, masks


def test_below_budget_is_full_fidelity():
    segs, masks = _toy_pixel_data()
    full = patchcore.metrics.compute_pixelwise_retrieval_metrics(segs, masks)
    capped = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs, masks, max_pixels=10**9
    )
    assert full["pixel_metric_samples"] == segs.size
    assert capped["pixel_metric_samples"] == segs.size
    assert capped["auroc"] == full["auroc"]
    np.testing.assert_array_equal(capped["fpr"], full["fpr"])
    np.testing.assert_array_equal(capped["tpr"], full["tpr"])


def test_above_budget_subsamples_deterministically():
    segs, masks = _toy_pixel_data()
    budget = segs.size // 10
    first = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs, masks, max_pixels=budget, seed=3
    )
    second = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs, masks, max_pixels=budget, seed=3
    )
    # seeded => bit-reproducible, and clearly below the full pixel count
    assert first["pixel_metric_samples"] == second["pixel_metric_samples"]
    assert 0 < first["pixel_metric_samples"] < segs.size // 2
    assert first["auroc"] == second["auroc"] == 1.0  # separable toy data


def test_budget_respects_list_inputs_and_gt_shape_mismatch():
    # callers pass GT as [N, 1, H, W] while scores are [N, H, W] — the
    # subsample must flatten both to the same pixel order.
    segs, masks = _toy_pixel_data()
    segs_list = [s for s in segs]
    masks_list = [m for m in masks]
    budget = segs.size // 10
    result = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs_list, masks_list, max_pixels=budget, seed=3
    )
    reference = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs, masks, max_pixels=budget, seed=3
    )
    assert result["pixel_metric_samples"] == reference["pixel_metric_samples"]
    assert result["auroc"] == reference["auroc"]


def test_budget_on_memmap_inputs(tmp_path):
    # the native-resolution path hands in memmaps; results must match arrays
    segs, masks = _toy_pixel_data()
    seg_mm = np.memmap(tmp_path / "seg.dat", dtype=np.float32, mode="w+",
                       shape=segs.shape)
    seg_mm[:] = segs
    gt_mm = np.memmap(tmp_path / "gt.dat", dtype=np.uint8, mode="w+",
                      shape=masks.shape)
    gt_mm[:] = masks.astype(np.uint8)
    budget = segs.size // 10
    from_memmap = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        seg_mm, gt_mm, max_pixels=budget, seed=3
    )
    from_array = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segs, masks, max_pixels=budget, seed=3
    )
    assert from_memmap["pixel_metric_samples"] == from_array["pixel_metric_samples"]
    assert from_memmap["auroc"] == from_array["auroc"] == 1.0
