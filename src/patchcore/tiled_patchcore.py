"""Tiled PatchCore: PatchCore over grids of backbone-sized image tiles.

Very large images cannot be fed to the backbone without destructive
downscaling. ``TiledPatchCore`` decomposes each image into a grid of
overlapping tiles (:class:`patchcore.tiling.TileMaker`), treats every tile
as an independent image through the *unmodified* PatchCore feature
pipeline, and reassembles per-tile score grids into a full-resolution
anomaly map.

Two disciplines this wrapper enforces (see docs/tiling_plan.md):

1. **Global coreset, once.** The memory bank is built by pooling the patch
   features of *all tiles of all training images* and calling
   ``featuresampler.run()`` exactly once — never per tile.
2. **No coreset at inference.** The predict path only does
   extract -> embed -> kNN -> stitch; the sampler is never touched.

The wrapper composes (never modifies) ``PatchCore``, configured with
``input_shape = (3, tile_size, tile_size)``, so the inner model never knows
tiling exists. Attribute access falls through to the inner instance, giving
the same public surface (``fit``, ``predict``, ``save_to_path``,
``load_from_path``, plus ``backbone``/``device``/``featuresampler``/...).
"""
import logging
import os
import pickle

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.utils.data
import tqdm

import patchcore.backbones
import patchcore.common
import patchcore.feature_store
import patchcore.patchcore
import patchcore.sampler
import patchcore.tiling

LOGGER = logging.getLogger(__name__)

DEFAULT_SMOOTHING = 4  # same sigma as patchcore.common.RescaleSegmentor


class TiledPatchCore:
    """PatchCore wrapper that tiles large images at native resolution.

    Args:
        patchcore: configured inner ``PatchCore`` whose ``input_shape`` is
            ``(3, tile_size, tile_size)``.
        tile_maker: the :class:`patchcore.tiling.TileMaker` defining the grid.
        tile_batch_size: max number of tiles per backbone forward pass —
            bounds GPU peak independently of the dataloader batch of images.
        smoothing: Gaussian sigma applied once to the stitched full-resolution
            map (never per tile — that would create seams).
        bank_cache_dir: directory for the memmap feature bank written during
            ``fit`` (default: system temp). Must be a real disk — on tmpfs the
            "disk" bank is RAM and the whole point is defeated. Local SSD, not
            NFS. Ignored when the model never fits (eval-only).
    """

    def __init__(self, patchcore, tile_maker, tile_batch_size=8, smoothing=DEFAULT_SMOOTHING,
                 bank_cache_dir=None):
        if tile_batch_size < 1:
            raise ValueError(f"tile_batch_size must be >= 1, got {tile_batch_size}.")
        # Private names so that attribute delegation via __getattr__ exposes the
        # *inner* featuresampler/anomaly_scorer/backbone etc. transparently.
        self._patchcore = patchcore
        self._tile_maker = tile_maker
        self._tile_batch_size = int(tile_batch_size)
        self._smoothing = smoothing
        self._bank_cache_dir = bank_cache_dir

    # ------------------------------------------------------------------
    # Delegation: the wrapper behaves like the inner PatchCore.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        # Only reached when normal lookup fails. Guard against recursion if
        # the wrapped instance is not (yet) set.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, "_patchcore")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    @property
    def patchcore(self):
        return self._patchcore

    @property
    def tile_maker(self):
        return self._tile_maker

    @property
    def tile_batch_size(self):
        return self._tile_batch_size

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        backbone,
        layers_to_extract_from,
        device,
        tile_size=384,
        overlap=None,
        stride=None,
        pad_value=0.0,
        tile_batch_size=8,
        smoothing=DEFAULT_SMOOTHING,
        bank_cache_dir=None,
        **patchcore_kwargs,
    ):
        """Build a ``TiledPatchCore`` wrapping a fresh tile-sized ``PatchCore``.

        ``patchcore_kwargs`` are forwarded to ``PatchCore.load``
        (pretrain_embed_dimension, target_embed_dimension, patchsize,
        featuresampler, anomaly_scorer_num_nn, nn_method, ...).
        ``input_shape`` is fixed to the tile size — the inner model only
        ever sees tiles.
        """
        inner = patchcore.patchcore.PatchCore(device)
        inner.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=(3, tile_size, tile_size),
            **patchcore_kwargs,
        )
        tile_maker = patchcore.tiling.TileMaker(
            tile_size=tile_size, overlap=overlap, stride=stride, pad_value=pad_value
        )
        return cls(inner, tile_maker, tile_batch_size=tile_batch_size,
                   smoothing=smoothing, bank_cache_dir=bank_cache_dir)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _embed_tiled(self, images):
        """Embed one image batch as tiles.

        Args:
            images: [B, 3, H, W] tensor at full/native resolution.

        Returns:
            (flat tile features [B*T*s*s, D] numpy, TileLayout,
             patch_shapes of one tile batch)
        """
        tiles, layout = self._tile_maker.extract(images)
        num_tiles_total = tiles.shape[0]
        features = []
        patch_shapes = None
        for start in range(0, num_tiles_total, self._tile_batch_size):
            chunk = tiles[start : start + self._tile_batch_size]
            chunk_features, patch_shapes = self._patchcore._embed(
                chunk, provide_patch_shapes=True
            )
            features.append(chunk_features)
        return np.concatenate(features, axis=0), layout, patch_shapes

    # ------------------------------------------------------------------
    # Training: global coreset over all pooled tile features, exactly once.
    # ------------------------------------------------------------------
    def fit(self, training_data):
        """Fill the memory bank: pool features of ALL tiles of ALL training
        images, then run the coreset sampler exactly once over the pooled
        matrix. Per-tile subsampling would destroy the coverage guarantee.

        The pooled [N, D] bank is written to a memory-mapped temp file
        (``MemmapFeatureStore``) instead of a Python list: at native
        resolution N*D floats are tens of GB, past host RAM. The sampler
        works on the memmap unchanged (np.memmap is an np.ndarray); its
        chunked projection pages blocks in from disk, so host RAM stays
        bounded by one chunk. Note the sampler's *projected* [N, 128] bank
        still lives fully on the GPU — that wall is hit around N ≈ 25M rows
        (~13 GB); everything below that is handled here.
        """
        _ = self._patchcore.forward_modules.eval()

        with patchcore.feature_store.MemmapFeatureStore(
            cache_dir=self._bank_cache_dir
        ) as store:
            with tqdm.tqdm(
                training_data, desc="Computing tiled support features...", position=1, leave=False
            ) as data_iterator:
                for image in data_iterator:
                    if isinstance(image, dict):
                        image = image["image"]
                    if image.ndim == 3:
                        image = image.unsqueeze(0)
                    image = image.to(torch.float).to(self.device)
                    with torch.no_grad():
                        image_features, _, _ = self._embed_tiled(image)
                    store.append(image_features)

            features = store.finalize()
            LOGGER.info(
                "Pooled %d tile patch features (memmap-backed); taking ONE global coreset.",
                features.shape[0],
            )
            # The single, global coreset call. Inference paths never do this.
            features = self.featuresampler.run(features)
            if isinstance(features, np.memmap):
                # Pass-through sampler (IdentitySampler / percentage==1): the
                # returned bank IS the mapped file, which close() deletes.
                features = np.array(features)
        if self.device.type == "cuda":
            # Hand PyTorch-cached blocks back to the driver so FAISS' raw
            # cudaMalloc for the GPU index does not hit torch's reserve.
            torch.cuda.empty_cache()
        self._patchcore.anomaly_scorer.fit(detection_features=[features])

    # ------------------------------------------------------------------
    # Inference: kNN only, then overlap-averaged stitching.
    # ------------------------------------------------------------------
    def predict(self, data, maps_dir=None):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, maps_dir=maps_dir)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, maps_dir=None):
        """Infer a full dataloader: (scores, segmentations, labels_gt, masks_gt).

        With ``maps_dir=None`` the per-image maps accumulate in RAM as lists
        of ndarrays (historic behavior — fine at 1024px MVTec scale). With
        ``maps_dir`` set, the full-resolution fp32 segmentation maps and the
        uint8 GT masks are streamed into memory-mapped files under
        ``maps_dir`` and returned as np.memmap arrays shaped [N, H, W] and
        [N, 1, H, W] — at native resolution the list form is tens of GB.
        np.memmap is an np.ndarray subclass, so per-image indexing and all
        downstream consumers keep working unchanged. The caller owns the
        lifetime of ``maps_dir`` (memmaps must not be used after its
        deletion); requires uniform image sizes per dataloader.
        """
        _ = self._patchcore.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        seg_store = gt_store = None
        mask_hw = gt_shape = None
        if maps_dir is not None:
            seg_store = patchcore.feature_store.MemmapFeatureStore(
                np.float32, cache_dir=maps_dir, name="segmentations"
            )
            gt_store = patchcore.feature_store.MemmapFeatureStore(
                np.uint8, cache_dir=maps_dir, name="masks_gt"
            )
        with tqdm.tqdm(dataloader, desc="Inferring (tiled)...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    gt = image["mask"].numpy()
                    if gt_store is not None:
                        # GT masks are binary 0./1. floats (metrics truncate to
                        # int anyway); uint8 quarters the disk footprint.
                        gt_shape = gt.shape[1:]
                        gt_store.append(gt.reshape(gt.shape[0], -1).astype(np.uint8))
                    else:
                        # Keep masks as ndarrays ([1,H,W] per image). The former
                        # .tolist() materialized H*W Python floats per image —
                        # seconds and ~100MB each at native resolution.
                        masks_gt.extend(gt)
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    if seg_store is not None:
                        mask_hw = mask.shape
                        seg_store.append(mask.reshape(1, -1))
                    else:
                        masks.append(mask)
        if seg_store is None:
            return scores, masks, labels_gt, masks_gt
        segmentations = (
            seg_store.finalize().reshape(-1, *mask_hw) if seg_store.rows else []
        )
        ground_truth = (
            gt_store.finalize().reshape(-1, *gt_shape) if gt_store.rows else []
        )
        return scores, segmentations, labels_gt, ground_truth

    def _predict(self, images):
        """Infer one image-level score and a full-resolution map per image."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.to(torch.float).to(self.device)
        _ = self._patchcore.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, layout, patch_shapes = self._embed_tiled(images)

            patch_scores = self._patchcore.anomaly_scorer.predict([features])[0]
            patch_scores = np.asarray(patch_scores)

            # Image score: max over every patch of every tile — identical
            # semantics to PatchCore's PatchMaker.score.
            image_scores = patch_scores.reshape(batchsize, -1).max(axis=1)

            # Per-tile score grids [B*T, s, s] -> stitched [B, H, W] with
            # overlaps averaged, then ONE Gaussian smoothing over the whole
            # stitched map (never per-tile: seams).
            grid = int(patch_shapes[0][0])
            expected_rows = batchsize * layout.num_tiles * grid * grid
            if patch_scores.shape[0] != expected_rows:
                raise RuntimeError(
                    f"Expected {expected_rows} tile patch scores "
                    f"(B={batchsize}, T={layout.num_tiles}, s={grid}), got "
                    f"{patch_scores.shape[0]}."
                )
            tile_maps = patch_scores.reshape(-1, grid, grid)
            stitched = self._tile_maker.stitch(tile_maps, layout, upsample=True)
            stitched = ndimage.gaussian_filter(stitched, sigma=(0, self._smoothing, self._smoothing))

        return [float(s) for s in image_scores], [mask for mask in stitched]

    # ------------------------------------------------------------------
    # Persistence (self-describing checkpoints).
    # ------------------------------------------------------------------
    @staticmethod
    def _tiling_params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "tiling_params.pkl")

    @staticmethod
    def has_tiling_params(load_path, prepend=""):
        """True if a tiled checkpoint was saved at ``load_path``."""
        return os.path.exists(TiledPatchCore._tiling_params_file(load_path, prepend))

    def save_to_path(self, save_path, prepend=""):
        LOGGER.info("Saving TiledPatchCore data.")
        os.makedirs(save_path, exist_ok=True)
        self._patchcore.save_to_path(save_path, prepend)
        tiling_params = {
            "tile_size": self._tile_maker.tile_size,
            "stride": self._tile_maker.stride,
            "pad_value": self._tile_maker.pad_value,
            "tile_batch_size": self._tile_batch_size,
            "smoothing": self._smoothing,
        }
        os.makedirs(save_path, exist_ok=True)
        with open(self._tiling_params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(tiling_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, load_path, device, nn_method, prepend=""):
        LOGGER.info("Loading and initializing TiledPatchCore.")
        if self._patchcore is None:
            self._patchcore = patchcore.patchcore.PatchCore(device)
        self._patchcore.load_from_path(
            load_path=load_path, device=device, nn_method=nn_method, prepend=prepend
        )
        with open(self._tiling_params_file(load_path, prepend), "rb") as load_file:
            tiling_params = pickle.load(load_file)
        self._tile_maker = patchcore.tiling.TileMaker(
            tile_size=tiling_params["tile_size"],
            stride=tiling_params["stride"],
            pad_value=tiling_params.get("pad_value", 0.0),
        )
        self._tile_batch_size = tiling_params.get("tile_batch_size", 8)
        self._smoothing = tiling_params.get("smoothing", DEFAULT_SMOOTHING)


def tiled_from_path(load_path, device, nn_method, prepend=""):
    """Convenience loader: build a ``TiledPatchCore`` from a saved tiled model."""
    instance = TiledPatchCore(None, None)
    instance.load_from_path(load_path, device, nn_method, prepend=prepend)
    return instance
