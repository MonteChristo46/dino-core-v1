"""Tests for tiled PatchCore (docs/tiling_plan.md).

CPU-friendly; run via ``./local_run_test.sh`` or
``PYTHONPATH=src pytest test/test_tiling.py``.
"""
import numpy as np
import pytest
import torch
import torch.utils.data
from torchvision import models

import patchcore
from patchcore import patchcore as patchcore_model
from patchcore.tiled_patchcore import TiledPatchCore, tiled_from_path
from patchcore.tiling import TileMaker, TileLayout

# ---------------------------------------------------------------------------
# helpers (style follows test/test_patchcore.py)
# ---------------------------------------------------------------------------

def _dummy_images(number_of_examples, image_shape):
    torch.manual_seed(0)
    return torch.rand([number_of_examples, *image_shape])


def _wr50_backbone():
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name, backbone.seed = "wideresnet50", 0
    return backbone


def _tiled_wr50(tile_size, overlap=None, stride=None, sampler=None, image_dim=224):
    """TiledPatchCore on CPU with an untrained WR50 (mini end-to-end style)."""
    return TiledPatchCore.load(
        backbone=_wr50_backbone(),
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        tile_size=tile_size,
        overlap=overlap,
        stride=stride,
        tile_batch_size=4,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        featuresampler=sampler or patchcore.sampler.IdentitySampler(),
        anomaly_scorer_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 2),
    )


def _plain_wr50(image_dim):
    instance = patchcore_model.PatchCore(torch.device("cpu"))
    instance.load(
        backbone=_wr50_backbone(),
        layers_to_extract_from=["layer2", "layer3"],
        device=torch.device("cpu"),
        input_shape=[3, image_dim, image_dim],
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        patchstride=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        anomaly_scorer_num_nn=1,
        nn_method=patchcore.common.FaissNN(False, 2),
    )
    return instance


class RecordingSampler:
    """Mock featuresampler that records every run() call (shape of input)."""

    def __init__(self):
        self.calls = []

    def run(self, features):
        self.calls.append(tuple(features.shape))
        return features


class _DictDataset(torch.utils.data.Dataset):
    """MVTec-style dict items at native resolution (image/mask/is_anomaly)."""

    def __init__(self, images, masks, labels):
        self.images = images
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "mask": self.masks[idx],
            "is_anomaly": torch.tensor(self.labels[idx]),
        }


# ---------------------------------------------------------------------------
# 1. grid math
# ---------------------------------------------------------------------------

def test_layout_exact_fit():
    tm = TileMaker(tile_size=384, stride=192)
    layout = tm.compute_layout(768, 768)
    assert layout.padded_hw == (768, 768)
    assert layout.origins == [
        (y, x) for y in (0, 192, 384) for x in (0, 192, 384)
    ]
    assert layout.num_tiles == 9


def test_layout_partial_fit_pads_right_bottom():
    tm = TileMaker(tile_size=384, overlap=192)
    layout = tm.compute_layout(500, 500)
    assert layout.padded_hw == (576, 576)
    assert layout.origins == [(0, 0), (0, 192), (192, 0), (192, 192)]
    assert layout.num_tiles == 4


def test_layout_smaller_than_tile_never_crops():
    tm = TileMaker(tile_size=384, stride=192)
    layout = tm.compute_layout(200, 300)
    assert layout.padded_hw == (384, 384)
    assert layout.num_tiles == 1
    assert layout.origins == [(0, 0)]


def test_layout_non_square():
    tm = TileMaker(tile_size=384, stride=192)
    layout = tm.compute_layout(500, 700)
    # h: 384 + ceil(116/192)*192 = 576 ; w: 384 + ceil(316/192)*192 = 768
    assert layout.padded_hw == (576, 768)
    assert layout.num_tiles == 2 * 3
    # every pixel of the padded canvas must be covered by >= 1 tile
    covered = np.zeros(layout.padded_hw, dtype=bool)
    for y, x in layout.origins:
        covered[y : y + 384, x : x + 384] = True
    assert covered.all()


# ---------------------------------------------------------------------------
# 2. validation
# ---------------------------------------------------------------------------

def test_tile_size_must_be_multiple_of_patch_size():
    with pytest.raises(ValueError):
        TileMaker(tile_size=300)  # 300 % 16 != 0


def test_overlap_and_stride_are_mutually_exclusive():
    with pytest.raises(ValueError):
        TileMaker(tile_size=384, overlap=192, stride=192)


def test_invalid_overlap_and_stride():
    with pytest.raises(ValueError):
        TileMaker(tile_size=384, overlap=384)
    with pytest.raises(ValueError):
        TileMaker(tile_size=384, stride=0)


def test_default_overlap_is_half_tile():
    tm = TileMaker(tile_size=384)
    assert tm.overlap == 192
    assert tm.stride == 192
    assert TileMaker(tile_size=384, stride=192).stride == tm.stride


# ---------------------------------------------------------------------------
# 3. extract / stitch roundtrips
# ---------------------------------------------------------------------------

def test_extract_shapes_and_content():
    tm = TileMaker(tile_size=32, stride=16)
    images = _dummy_images(2, [3, 48, 48])
    tiles, layout = tm.extract(images)
    assert tiles.shape == (2 * 4, 3, 32, 32)
    assert layout.num_tiles == 4
    # batch-then-row-major tile ordering: tile idx = b*T + iy*nx + ix
    assert torch.equal(tiles[0], images[0, :, 0:32, 0:32])
    assert torch.equal(tiles[1], images[0, :, 0:32, 16:48])
    assert torch.equal(tiles[6], images[1, :, 16:48, 0:32])


def test_extract_pads_with_pad_value():
    tm = TileMaker(tile_size=32, stride=32, pad_value=-0.5)
    images = torch.ones(1, 3, 48, 48)
    tiles, layout = tm.extract(images)
    assert layout.padded_hw == (64, 64)
    last_tile = tiles[3]  # origin (32, 32): content rows 32..47, rest padding
    assert torch.allclose(last_tile[:, 16:, 16:], torch.full((3, 16, 16), -0.5))
    assert torch.allclose(last_tile[:, :16, :16], torch.ones(3, 16, 16))


def test_stitch_constant_maps_survive():
    tm = TileMaker(tile_size=32, stride=16)
    images = torch.full((2, 3, 48, 48), 0.7)
    _, layout = tm.extract(images)
    tile_maps = torch.full((2 * layout.num_tiles, 32, 32), 0.7)
    stitched = tm.stitch(tile_maps, layout, upsample=False)
    assert stitched.shape == (2, 48, 48)
    assert np.allclose(stitched, 0.7)


def test_stitch_overlap_pixels_are_means_of_contributors():
    tm = TileMaker(tile_size=32, stride=16)
    images = torch.zeros(1, 3, 48, 48)
    _, layout = tm.extract(images)
    # tile constants v0..v3 in origin order (0,0),(0,16),(16,0),(16,16)
    tile_maps = torch.stack(
        [torch.full((32, 32), float(v)) for v in (0.0, 1.0, 2.0, 3.0)]
    )
    stitched = tm.stitch(tile_maps, layout, upsample=False)
    assert stitched.shape == (1, 48, 48)
    assert stitched[0, 8, 8] == 0.0          # only tile 0
    assert np.isclose(stitched[0, 8, 24], 0.5)   # tiles 0+1
    assert np.isclose(stitched[0, 24, 8], 1.0)   # tiles 0+2
    assert np.isclose(stitched[0, 24, 24], 1.5)  # all four tiles -> mean
    assert np.isclose(stitched[0, 47, 47], 3.0)  # only tile 3


def test_stitch_roundtrip_is_lossless_on_content():
    tm = TileMaker(tile_size=384, stride=192, pad_value=0.0)
    images = _dummy_images(2, [3, 500, 500])
    tiles, layout = tm.extract(images)
    channel_mean = tiles.mean(dim=1)  # [B*T, 384, 384]
    stitched = tm.stitch(channel_mean, layout, upsample=False)
    expected = images.mean(dim=1).numpy()  # [2, 500, 500]
    assert stitched.shape == (2, 500, 500)
    assert np.allclose(stitched, expected, atol=1e-6)


def test_stitch_upsamples_coarse_tile_grids():
    tm = TileMaker(tile_size=32, stride=16)
    _, layout = tm.extract(torch.zeros(1, 3, 48, 48))
    coarse = torch.full((layout.num_tiles, 4, 4), 0.9)
    stitched = tm.stitch(coarse, layout, upsample=True)
    assert stitched.shape == (1, 48, 48)
    assert np.allclose(stitched, 0.9)


def test_stitch_rejects_undersized_maps_without_upsample():
    tm = TileMaker(tile_size=32, stride=16)
    _, layout = tm.extract(torch.zeros(1, 3, 48, 48))
    with pytest.raises(ValueError):
        tm.stitch(torch.zeros(layout.num_tiles, 4, 4), layout, upsample=False)


# ---------------------------------------------------------------------------
# 4. coreset invariant (the important one)
# ---------------------------------------------------------------------------

def test_coreset_is_global_and_called_once():
    tile, stride, image_dim = 112, 56, 224
    sampler = RecordingSampler()
    model = _tiled_wr50(tile, stride=stride, sampler=sampler)

    training = _dummy_images(2, [3, image_dim, image_dim])
    loader = torch.utils.data.DataLoader(training, batch_size=2)
    model.fit(loader)

    # 224 -> padded 224 -> origins [0,56,112] -> 3x3 = 9 tiles;
    # WR50 @112 -> layer2 grid 14x14 -> 196 patches/tile, D=1024.
    n_tiles, s_sq = 9, 14 * 14
    assert sampler.calls == [(2 * n_tiles * s_sq, 1024)], (
        "coreset sampler must be called EXACTLY ONCE on ALL pooled tile features"
    )

    sampler.calls.clear()
    scores, masks = model.predict(training[:1])
    assert sampler.calls == [], "inference must never run the coreset sampler"
    assert len(scores) == 1
    assert np.all(masks[0].shape == (image_dim, image_dim))


def test_featuresampler_and_backbone_delegate_to_inner():
    model = _tiled_wr50(112, stride=56)
    assert isinstance(model.featuresampler, patchcore.sampler.IdentitySampler)
    assert model.featuresampler is model.patchcore.featuresampler
    assert model.backbone.seed == 0  # run_patchcore.py accesses .backbone.seed
    assert model.device.type == "cpu"
    assert model.input_shape == (3, 112, 112)


# ---------------------------------------------------------------------------
# 5. predict contract
# ---------------------------------------------------------------------------

def test_predict_contract_dict_dataloader():
    image_dim, tile = 128, 112
    model = _tiled_wr50(tile, stride=tile)  # non-overlapping; 2x2 tiles @128
    training = _dummy_images(3, [3, image_dim, image_dim])
    model.fit(torch.utils.data.DataLoader(training, batch_size=3))

    masks = torch.zeros(3, 1, image_dim, image_dim)
    dataset = _DictDataset(training, masks, [0, 1, 0])
    scores, segs, labels_gt, masks_gt = model.predict(
        torch.utils.data.DataLoader(dataset, batch_size=3)
    )
    assert len(scores) == 3
    for seg in segs:
        assert seg.shape == (image_dim, image_dim)  # native full-res, not tile size
    assert labels_gt == [0, 1, 0]
    assert len(masks_gt) == 3


# ---------------------------------------------------------------------------
# 5b. degenerate equivalence: one tile == plain PatchCore
# ---------------------------------------------------------------------------

def test_single_tile_matches_untiled_patchcore():
    image_dim = 112
    training = _dummy_images(2, [3, image_dim, image_dim])
    loader = torch.utils.data.DataLoader(training, batch_size=2)
    test_batch = _dummy_images(2, [3, image_dim, image_dim]) * 1.3

    # Identical (untrained) backbone weights for both models: seeding makes
    # wide_resnet50_2(weights=None) deterministic, so both get the same net.
    torch.manual_seed(1234)
    tiled = _tiled_wr50(image_dim, stride=image_dim)  # exactly 1 tile per image
    torch.manual_seed(1234)
    plain = _plain_wr50(image_dim)
    tiled.fit(loader)
    plain.fit(loader)

    t_scores, t_masks = tiled.predict(test_batch)
    p_scores, p_masks = plain.predict(test_batch)

    assert np.allclose(t_scores, p_scores, rtol=1e-4, atol=1e-6)
    for t_mask, p_mask in zip(t_masks, p_masks):
        assert t_mask.shape == p_mask.shape
        # same bilinear upsample + same single Gaussian -> same map
        assert np.allclose(t_mask, p_mask, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    image_dim, tile, stride = 224, 112, 56
    # Seed identically before both constructions: the untrained WR50 is
    # re-created from scratch inside load_from_path (real pretrained /
    # DINOv3 weights are deterministic, this random init is not).
    torch.manual_seed(42)
    model = _tiled_wr50(tile, stride=stride)
    training = _dummy_images(2, [3, image_dim, image_dim])
    loader = torch.utils.data.DataLoader(training, batch_size=2)
    model.fit(loader)

    save_path = str(tmp_path / "model")
    model.save_to_path(save_path, "tp_")
    assert TiledPatchCore.has_tiling_params(save_path, "tp_")
    assert not TiledPatchCore.has_tiling_params(save_path, "other_")

    torch.manual_seed(42)
    loaded = tiled_from_path(
        save_path, torch.device("cpu"), patchcore.common.FaissNN(False, 2), prepend="tp_"
    )
    assert loaded.tile_maker.tile_size == tile
    assert loaded.tile_maker.stride == stride
    assert loaded.tile_batch_size == model.tile_batch_size

    test_batch = _dummy_images(1, [3, image_dim, image_dim]) * 1.7
    a_scores, a_masks = model.predict(test_batch)
    b_scores, b_masks = loaded.predict(test_batch)
    assert np.allclose(a_scores, b_scores, rtol=1e-4, atol=1e-6)
    assert np.allclose(a_masks, b_masks, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# mvtec native-resolution passthrough
# ---------------------------------------------------------------------------

def _write_fake_mvtec(root, classname="bottle", size=(64, 48)):
    from PIL import Image

    h, w = size
    for rel in (
        f"{classname}/train/good",
        f"{classname}/test/good",
        f"{classname}/test/broken",
        f"{classname}/ground_truth/broken",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), (10, 20, 30))
    img.save(root / f"{classname}/train/good/0.png")
    img.save(root / f"{classname}/test/good/1.png")
    img.save(root / f"{classname}/test/broken/2.png")
    Image.new("L", (w, h), 0).save(root / f"{classname}/ground_truth/broken/2.png")


def test_mvtec_native_resolution_passthrough(tmp_path):
    from patchcore.datasets import mvtec

    _write_fake_mvtec(tmp_path)
    ds = mvtec.MVTecDataset(
        str(tmp_path), "bottle", resize=None, imagesize=None,
        split=mvtec.DatasetSplit.TEST,
    )
    assert ds.imagesize is None
    item = ds[0]
    assert tuple(item["image"].shape) == (3, 64, 48)      # native, unresized
    mask = ds[1]  # broken sample has a GT mask
    assert tuple(mask["mask"].shape) == (1, 64, 48)


def test_mvtec_default_transforms_unchanged(tmp_path):
    from patchcore.datasets import mvtec

    _write_fake_mvtec(tmp_path)
    ds = mvtec.MVTecDataset(str(tmp_path), "bottle", split=mvtec.DatasetSplit.TRAIN)
    assert ds.imagesize == (3, 224, 224)
    assert tuple(ds[0]["image"].shape) == (3, 224, 224)
