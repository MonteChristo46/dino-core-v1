"""Tests for patchcore.feature_store.MemmapFeatureStore.

CPU-only; run via ``./local_run_test.sh`` or
``PYTHONPATH=src pytest test/test_feature_store.py``.
"""
import os
import tempfile

import numpy as np
import pytest

from patchcore.feature_store import MemmapFeatureStore


def test_append_finalize_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    blocks = [rng.random((n, 8)).astype(np.float32) for n in (3, 1, 5)]
    with MemmapFeatureStore(cache_dir=str(tmp_path)) as store:
        for block in blocks:
            store.append(block)
        bank = store.finalize()
        assert isinstance(bank, np.memmap)
        assert bank.shape == (9, 8)
        assert bank.dtype == np.float32
        # identical content to the list-concatenation it replaces
        np.testing.assert_array_equal(np.asarray(bank), np.concatenate(blocks, axis=0))
        # writable: in-place ops (chunked normalization) must write through
        bank[0, 0] = -1.0
        assert bank[0, 0] == -1.0


def test_rows_tracked_incrementally(tmp_path):
    with MemmapFeatureStore(cache_dir=str(tmp_path)) as store:
        assert store.rows == 0
        store.append(np.zeros((4, 3), dtype=np.float32))
        store.append(np.zeros((2, 3), dtype=np.float32))
        assert store.rows == 6


def test_dim_mismatch_rejected(tmp_path):
    with MemmapFeatureStore(cache_dir=str(tmp_path)) as store:
        store.append(np.zeros((2, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="dim"):
            store.append(np.zeros((2, 5), dtype=np.float32))


def test_non_2d_rejected(tmp_path):
    with MemmapFeatureStore(cache_dir=str(tmp_path)) as store:
        with pytest.raises(ValueError):
            store.append(np.zeros((2, 3, 4), dtype=np.float32))


def test_1d_block_becomes_single_row(tmp_path):
    with MemmapFeatureStore(np.uint8, cache_dir=str(tmp_path)) as store:
        store.append(np.array([1.0, 0.0, 1.0]))  # 1D -> [1, 3], cast to uint8
        store.append(np.zeros((2, 3)))
        bank = store.finalize()
        assert bank.dtype == np.uint8
        assert bank.shape == (3, 3)
        assert bank[0].tolist() == [1, 0, 1]


def test_finalize_without_rows_rejected(tmp_path):
    with MemmapFeatureStore(cache_dir=str(tmp_path)) as store:
        with pytest.raises(ValueError, match="no rows"):
            store.finalize()


def test_close_deletes_files(tmp_path):
    store = MemmapFeatureStore(cache_dir=str(tmp_path))
    store.append(np.ones((2, 3), dtype=np.float32))
    path = store.path
    store.close()
    assert not os.path.exists(path)
    assert not os.path.exists(os.path.dirname(path))  # whole tmpdir removed


def test_default_cache_dir_is_system_temp():
    with MemmapFeatureStore() as store:
        parent = os.path.dirname(os.path.dirname(store.path))
        assert os.path.realpath(parent) == os.path.realpath(tempfile.gettempdir())


def test_cache_dir_created_if_missing(tmp_path):
    missing = tmp_path / "not" / "there" / "yet"
    with MemmapFeatureStore(cache_dir=str(missing)) as store:
        store.append(np.ones((1, 2), dtype=np.float32))
        assert os.path.dirname(store.path).startswith(str(missing))
