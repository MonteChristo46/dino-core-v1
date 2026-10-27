"""Memory-mapped row store for out-of-RAM PatchCore feature matrices.

Tiled training pools the patch features of *all tiles of all training
images* before the single global coreset — at native resolution that bank
is tens of GB, far beyond host RAM. ``MemmapFeatureStore`` appends feature
blocks as raw bytes to a temp file (no pre-sizing, so the row count need
not be known up front and per-image row counts may vary) and exposes the
result as one ``np.memmap`` — an ``np.ndarray`` subclass, so samplers and
metric code work on it unchanged, reading through the OS page cache
instead of resident RAM.

Caveat: if the temp directory is tmpfs, the file *is* RAM and nothing is
gained — point ``cache_dir`` at a real (local SSD) disk, never NFS.
"""
import os
import shutil
import tempfile

import numpy as np


class MemmapFeatureStore:
    """Append-only [N, D] row store finalized as a single np.memmap."""

    def __init__(self, dtype=np.float32, cache_dir=None, name="features"):
        self._dtype = np.dtype(dtype)
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        self._tmpdir = tempfile.mkdtemp(prefix="patchcore_bank_", dir=cache_dir)
        self._path = os.path.join(self._tmpdir, name + ".dat")
        self._file = open(self._path, "wb")
        self._rows = 0
        self._dim = None

    @property
    def rows(self):
        """Number of rows appended so far."""
        return self._rows

    @property
    def path(self):
        return self._path

    def append(self, block):
        """Append a [rows, D] block; D is fixed by the first append."""
        block = np.ascontiguousarray(block, dtype=self._dtype)
        if block.ndim == 1:
            block = block.reshape(1, -1)
        if block.ndim != 2:
            raise ValueError(
                f"MemmapFeatureStore stores 2D [rows, D] blocks, got shape "
                f"{block.shape}."
            )
        if self._dim is None:
            self._dim = int(block.shape[1])
        elif block.shape[1] != self._dim:
            raise ValueError(
                f"Feature dim changed mid-stream: first block had D={self._dim}, "
                f"got D={block.shape[1]}."
            )
        self._file.write(block.tobytes(order="C"))
        self._rows += block.shape[0]

    def finalize(self):
        """Close for writing and return the whole store as np.memmap [N, D].

        The memmap is writable (mode r+); the file it maps is removed by
        :meth:`close` / the context manager, so callers that need the data
        beyond the store's lifetime must copy it first.
        """
        if self._rows == 0:
            raise ValueError("MemmapFeatureStore.finalize() with no rows appended.")
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        return np.memmap(
            self._path, dtype=self._dtype, mode="r+", shape=(self._rows, self._dim)
        )

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
