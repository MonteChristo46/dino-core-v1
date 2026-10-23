"""Visualize what the tiled PatchCore backbone sees — per-patch DINOv3
features reduced to 3D (UMAP/PCA), painted as RGB, stitched back to one
full-resolution map with the tile grid drawn in.

Each image is cut into overlapping backbone-sized tiles exactly like at
training/inference time. For EVERY extracted layer the per-patch features
are reduced to 3 components, mapped to RGB (percentile-clipped), and
stitched with the same overlap-averaged machinery as the anomaly maps —
plus one column for the aggregated multi-layer features (MeanMapper ->
layer stack -> pooled), which is bit-identical to what the kNN scorer
operates on. The result per image is a 2-row panel:

    row 1: original + tile grid | layer A (tiled) | ... | patchcore avg (tiled)
    row 2: squashed global input | layer A (full) | ... | patchcore avg (full)

Row 2 is the NON-tiled baseline: the whole image squashed to one backbone
input of --full_image_size² (default 384²), embedded in a single pass. The
squash resolution is decoupled from the tiling geometry, and every row-2
series is colored through its OWN reducer fit (not the tiled column's), so
row 2 is a fixed reference: identical features and identical colors no
matter what --tile_size is set to. The price: hues are only comparable
within one row, never between the tiled row and the global row.

Two fit scopes:
- ``image``:    reducer fits per image per series. Colors are only
                comparable *within* one image column (default).
- ``category``: reducer fits once per series on a subsample pooled over all
                selected images. Colors are comparable across images.

Examples
--------
# Fresh backbone (no checkpoint needed — the aggregator is parameter-free):
uv run python scripts/tiled_patchcore/visualize_features.py \
    --data_path dataset --category mosque --split test --n_images 2

# Pick the DINO layers (each gets a panel, plus the PatchCore average):
uv run python scripts/tiled_patchcore/visualize_features.py \
    --data_path dataset --category mosque --layers 5,7,8

# Exact checkpoint pipeline (layers + tiling geometry from the checkpoint):
uv run python scripts/tiled_patchcore/visualize_features.py \
    --model_dir results_tiled/.../models/mvtec_mosque \
    --data_path dataset --category mosque --reducer pca --fit_scope category
"""
import logging
import os
import sys

import click
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared  # noqa: F401,E402  — must come first: puts repo src/ on sys.path

import matplotlib.patches  # noqa: E402  (matplotlib Agg set by shared)
import matplotlib.pyplot as plt  # noqa: E402

import patchcore.backbones  # noqa: E402
import patchcore.common  # noqa: E402
import patchcore.utils  # noqa: E402
from patchcore.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from patchcore.tiled_patchcore import TiledPatchCore  # noqa: E402

LOGGER = logging.getLogger("visualize_features")


# ----------------------------------------------------------------------
# Reducers
# ----------------------------------------------------------------------
def build_reducer(name, seed, n_neighbors, min_dist, n_fit_rows):
    """3-component reducer: umap (default) | pca | cuml_umap (RAPIDS)."""
    if name == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=3, random_state=seed)
    n_neighbors = max(2, min(n_neighbors, n_fit_rows - 1))
    if name == "umap":
        import umap

        return umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,  # deterministic, single-threaded
        )
    if name == "cuml_umap":
        try:
            from cuml.manifold import UMAP as CuMLUMAP
        except ImportError as exc:
            raise click.ClickException(
                "--reducer cuml_umap needs NVIDIA RAPIDS cuML installed "
                "(not part of this env). Use --reducer umap for CPU."
            ) from exc
        return CuMLUMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
    raise click.ClickException(f"Unknown reducer '{name}'.")


def fit_reducer(reducer, rows, max_fit_rows, seed):
    """Fit on a seeded subsample; return (reducer, clip_lo, clip_hi)."""
    if rows.shape[0] > max_fit_rows:
        rng = np.random.default_rng(seed)
        fit_rows = rows[rng.choice(rows.shape[0], max_fit_rows, replace=False)]
        LOGGER.info(
            "Fitting %s on %d/%d rows (subsampled).",
            type(reducer).__name__, max_fit_rows, rows.shape[0],
        )
    else:
        fit_rows = rows
        LOGGER.info("Fitting %s on %d rows.", type(reducer).__name__, rows.shape[0])
    fit_coords = reducer.fit_transform(np.ascontiguousarray(fit_rows, dtype=np.float32))
    lo, hi = np.percentile(fit_coords, [1.0, 99.0], axis=0)
    return reducer, lo, hi


def transform_rows(reducer, rows):
    return reducer.transform(np.ascontiguousarray(rows, dtype=np.float32))


def coords_to_rgb(coords, lo, hi):
    rgb = (coords - lo) / np.maximum(hi - lo, 1e-12)
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------------------------------------------------
# Feature extraction + stitching
# ----------------------------------------------------------------------
def extract_tile_features(model, image_tensor):
    """One image -> aggregated (multi-layer, PatchCore-pipeline) features.

    Args:
        image_tensor: [3, H, W] normalized tensor (native resolution).

    Returns:
        (tile_grids [T, s, s, D] float32, TileLayout)
    """
    images = image_tensor.unsqueeze(0).to(torch.float).to(model.device)
    _ = model.forward_modules.eval()
    with torch.no_grad():
        features, layout, patch_shapes = model._embed_tiled(images)
    patch_h, patch_w = int(patch_shapes[0][0]), int(patch_shapes[0][1])
    tile_grids = features.reshape(layout.num_tiles, patch_h, patch_w, -1)
    return tile_grids.astype(np.float32), layout


def extract_layer_features(model, image_tensor):
    """One image -> per-layer patch features, before PatchCore aggregation.

    Taps ``feature_aggregator`` directly (same tiles, same chunking as
    ``_embed_tiled``) and patchifies each requested layer individually.
    For patchsize > 1 the patch window is mean-pooled to the channel dim.

    Returns:
        ({layer_name: tile_grids [T, s_l, s_l, C_l] float32}, TileLayout)
    """
    images = image_tensor.unsqueeze(0).to(torch.float).to(model.device)
    tiles, layout = model.tile_maker.extract(images)
    aggregator = model.forward_modules["feature_aggregator"]
    aggregator.eval()
    layers = list(model.layers_to_extract_from)
    per_layer = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in range(0, tiles.shape[0], model.tile_batch_size):
            chunk = tiles[start : start + model.tile_batch_size]
            outputs = aggregator(chunk)  # {layer: [B, C, h, w]}
            for layer in layers:
                patches, (ph, pw) = model.patch_maker.patchify(
                    outputs[layer], return_spatial_info=True
                )
                # patches: [B, N, C, ps, ps], N == ph*pw
                if patches.shape[-1] > 1 or patches.shape[-2] > 1:
                    patches = patches.mean(dim=(-1, -2))
                else:
                    patches = patches[:, :, :, 0, 0]
                grids = patches.reshape(len(patches), ph, pw, -1)
                per_layer[layer].append(grids.detach().cpu().numpy())
    per_layer = {
        layer: np.concatenate(chunks, axis=0).astype(np.float32)
        for layer, chunks in per_layer.items()
    }
    return per_layer, layout


def extract_full_image_series(model, image_tensor, full_size):
    """One image -> per-series features of a single NON-tiled global pass.

    The whole native-resolution image is squashed to ``full_size``²
    (``--full_image_size``) — deliberately decoupled from the tiling
    geometry, so this row is a stable reference across tile sizes.
    Returns {series_key: [s, s, C]} with one entry per extracted layer plus
    the aggregated PatchCore features — the exact non-tiled counterparts of
    the tiled series from extract_layer_features / extract_tile_features
    (same keys).
    """
    images = (
        F.interpolate(image_tensor.unsqueeze(0), size=(full_size, full_size), mode="area")
        .to(torch.float)
        .to(model.device)
    )
    _ = model.forward_modules.eval()
    series = {}
    with torch.no_grad():
        outputs = model.forward_modules["feature_aggregator"](images)
        for layer in model.layers_to_extract_from:
            patches, (ph, pw) = model.patch_maker.patchify(
                outputs[layer], return_spatial_info=True
            )
            if patches.shape[-1] > 1 or patches.shape[-2] > 1:
                patches = patches.mean(dim=(-1, -2))
            else:
                patches = patches[:, :, :, 0, 0]
            series[f"layer {layer}"] = (
                patches.reshape(len(patches), ph, pw, -1)[0]
                .detach().cpu().numpy().astype(np.float32)
            )
        # Inner PatchCore: TiledPatchCore.__getattr__ blocks _-names.
        features, patch_shapes = model.patchcore._embed(images, provide_patch_shapes=True)
    features = np.asarray(features, dtype=np.float32)
    patch_h, patch_w = int(patch_shapes[0][0]), int(patch_shapes[0][1])
    avg_key = f"patchcore avg ({','.join(model.layers_to_extract_from)})"
    series[avg_key] = features.reshape(patch_h, patch_w, -1)
    return series


def stitch_rgb(tile_maker, tile_rgb, layout):
    """[T, s, s, 3] tile RGB grids -> [H, W, 3] full-res overlap-averaged map."""
    channels = [
        tile_maker.stitch(tile_rgb[..., c], layout, upsample=True)[0]
        for c in range(3)
    ]
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def upsample_grid_to_image(grid_rgb, image_hw):
    """[s, s, 3] -> [H, W, 3] bilinear (same interpolation as tile stitching)."""
    tensor = torch.from_numpy(np.ascontiguousarray(grid_rgb)).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(tensor, size=image_hw, mode="bilinear", align_corners=False)
    return np.clip(up.squeeze(0).permute(1, 2, 0).numpy(), 0.0, 1.0)


def _npz_key(series_key):
    """'layer 7' -> 'coords_layer_7'; 'patchcore avg (7,8)' -> 'coords_avg_7-8'."""
    cleaned = (
        series_key.replace("patchcore avg", "avg")
        .replace(" ", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "-")
    )
    return f"coords_{cleaned}"


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _draw_grid(axis, layout):
    for idx, (y, x) in enumerate(layout.origins):
        color = "tab:red" if idx % 2 else "tab:blue"
        axis.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), layout.tile_size, layout.tile_size,
                fill=False, edgecolor=color, linewidth=1.0,
            )
        )


def plot_feature_panel(rows, row_titles, layout, save_path, title):
    """Grid of cells: row 1 = original | tiled series, (optional) row 2 =
    squashed full-image input | non-tiled counterparts. The tile grid is
    drawn on row 1 only — row 2 computes no tiles, so it stays clean.

    Args:
        rows: list of rows; each row a list of HxWx3 arrays (same length).
        row_titles: parallel list of per-cell titles.
    """
    height, width = rows[0][0].shape[:2]
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 6.5 * (height / width) * n_rows),
        dpi=110, squeeze=False,
    )
    for r, (row, titles) in enumerate(zip(rows, row_titles)):
        for c, (cell, cell_title) in enumerate(zip(row, titles)):
            axis = axes[r][c]
            axis.imshow(cell)
            axis.set_title(cell_title)
            if r == 0:
                _draw_grid(axis, layout)
            axis.set_xlim(-0.5, width - 0.5)
            axis.set_ylim(height - 0.5, -0.5)
            axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------
def load_dataset_items(data_path, category, split, n_images, seed):
    """First n_images of a dataset split -> [(name, image_path, tensor)]."""
    dataset, _ = shared.build_dataset(
        data_path, category, split, resize=None, imagesize=None, seed=seed
    )
    items = []
    for idx in range(min(n_images, len(dataset))):
        sample = dataset[idx]
        items.append((sample["image_name"], sample["image_path"], sample["image"]))
    return items, f"mvtec_{category}"


def load_image_path_items(image_paths):
    """Explicit image paths -> [(name, image_path, tensor)] (ImageNet norm)."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    items = []
    for path in image_paths:
        with PIL.Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        items.append((os.path.basename(path), path, tensor))
    return items, "custom"


def build_model(opts, device, faiss_on_gpu):
    """Checkpoint (--model_dir) or fresh backbone; returns (model, source_tag)."""
    if opts["model_dir"]:
        models = shared.load_models_from_dir(
            opts["model_dir"],
            device,
            patchcore.common.FaissNN(faiss_on_gpu, opts["faiss_num_workers"]),
            seed=opts["seed"],
        )
        model, is_tiled = models[0]
        if not is_tiled:
            raise click.ClickException(
                f"{opts['model_dir']} holds a plain (non-tiled) PatchCore — "
                "this script needs a TiledPatchCore checkpoint."
            )
        maker = model.tile_maker
        LOGGER.info(
            "Checkpoint pipeline: tile=%d stride=%d pad=%.2f (tiling flags ignored).",
            maker.tile_size, maker.stride, maker.pad_value,
        )
        return model, "checkpoint"

    layers = opts["layers"]
    layers = layers.split(",") if layers else shared.default_layers(opts["backbone"])
    if not layers:
        raise click.UsageError(
            f"--layers is required for backbone '{opts['backbone']}'."
        )
    pretrain_dim, target_dim = shared.resolve_embed_dims(
        opts["backbone"], opts["pretrain_embed_dimension"], opts["target_embed_dimension"]
    )
    torch.manual_seed(opts["seed"])  # match run_tiled_patchcore seeding
    backbone = patchcore.backbones.load(opts["backbone"])
    backbone.name, backbone.seed = opts["backbone"], None
    model = TiledPatchCore.load(
        backbone=backbone,
        layers_to_extract_from=layers,
        device=device,
        tile_size=opts["tile_size"],
        overlap=opts["tile_overlap"],
        pad_value=opts["pad_value"],
        tile_batch_size=opts["tile_batch_size"],
        pretrain_embed_dimension=pretrain_dim,
        target_embed_dimension=target_dim,
        patchsize=opts["patchsize"],
    )
    return model, f"fresh {opts['backbone']}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
# Input
@click.option("--data_path", type=click.Path(exists=True, file_okay=False), default=None,
              help="Dataset root with MVTec-format categories.")
@click.option("--category", default=None, help="Category under --data_path.")
@click.option("--split", default="test", show_default=True,
              type=click.Choice(["train", "test", "val"], case_sensitive=False))
@click.option("--n_images", type=int, default=4, show_default=True,
              help="First N images of the split (dataset mode).")
@click.option("--image_path", multiple=True,
              help="Explicit image path(s); overrides --data_path/--category.")
# Model
@click.option("--model_dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Tiled checkpoint folder; tiling geometry is read from it.")
@click.option("--backbone", default="dinov3_vitb", show_default=True)
@click.option("--layers", default=None,
              help="Comma-separated layers (default: per family, e.g. '7,8' DINOv3). "
              "Each layer gets its own panel, plus the PatchCore average.")
@click.option("--tile_size", type=int, default=384, show_default=True)
@click.option("--tile_overlap", type=int, default=None,
              help="Default tile_size // 2 (50%).")
@click.option("--tile_batch_size", type=int, default=8, show_default=True)
@click.option("--pad_value", type=float, default=0.0, show_default=True)
@click.option("--patchsize", type=int, default=1, show_default=True)
@click.option("--pretrain_embed_dimension", type=int, default=None)
@click.option("--target_embed_dimension", type=int, default=None)
# Reduction
@click.option("--reducer", default="umap", show_default=True,
              type=click.Choice(["umap", "pca", "cuml_umap"], case_sensitive=False))
@click.option("--fit_scope", default="image", show_default=True,
              type=click.Choice(["image", "category"], case_sensitive=False),
              help="'image': colors comparable within one image. "
              "'category': one fit over all images, comparable colors.")
@click.option("--max_fit_rows", type=int, default=50000, show_default=True,
              help="Subsample cap for reducer fitting.")
@click.option("--umap_neighbors", type=int, default=15, show_default=True)
@click.option("--umap_min_dist", type=float, default=0.1, show_default=True)
@click.option("--no_full_image", is_flag=True,
              help="Skip the non-tiled full-image row (one global pass).")
@click.option("--full_image_size", type=int, default=384, show_default=True,
              help="Squash resolution for the non-tiled row-2 global pass. "
              "Independent of --tile_size, so row 2 is a fixed reference.")
# Compute / output
@click.option("--gpu", type=int, default=None,
              help="CUDA device id. Default: auto — CUDA, then Apple MPS, then CPU.")
@click.option("--cpu", is_flag=True, help="Force CPU.")
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out_dir", default="feature_maps", show_default=True)
@click.option("--save_npz", is_flag=True,
              help="Also save the 3D reduced coords + tile layout per image.")
def main(**kwargs):
    """Render per-patch backbone features as one stitched RGB map per image.

    \b
    Examples:
      uv run python scripts/tiled_patchcore/visualize_features.py \
          --data_path dataset --category mosque --n_images 2
      uv run python scripts/tiled_patchcore/visualize_features.py \
          --model_dir results_tiled/.../models/mvtec_mosque \
          --data_path dataset --category mosque --reducer pca
    """
    opts = kwargs
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not opts["image_path"] and not (opts["data_path"] and opts["category"]):
        raise click.UsageError(
            "Provide either --image_path, or --data_path together with --category."
        )

    device, faiss_on_gpu = shared.resolve_device(gpu=opts["gpu"], cpu=opts["cpu"])
    patchcore.utils.fix_seeds(opts["seed"], device)
    os.makedirs(opts["out_dir"], exist_ok=True)

    if opts["image_path"]:
        items, tag = load_image_path_items(list(opts["image_path"]))
    else:
        items, tag = load_dataset_items(
            opts["data_path"], opts["category"], opts["split"],
            opts["n_images"], opts["seed"],
        )
    model, model_source = build_model(opts, device, faiss_on_gpu)
    LOGGER.info(
        "Visualizing %d image(s) | model=%s reducer=%s fit_scope=%s device=%s",
        len(items), model_source, opts["reducer"], opts["fit_scope"], device,
    )

    # ---- pass 1: features for every image --------------------------------
    layer_names = list(model.layers_to_extract_from)
    avg_key = f"patchcore avg ({','.join(layer_names)})"
    full_size = opts["full_image_size"]
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    per_image = []  # (name, image_path, series, full_series, squashed_input, layout)
    for name, image_path, tensor in items:
        layer_grids, layout = extract_layer_features(model, tensor)
        series = {f"layer {layer}": grids for layer, grids in layer_grids.items()}
        series[avg_key], _ = extract_tile_features(model, tensor)
        full_series, squashed_input = None, None
        if not opts["no_full_image"]:
            full_series = extract_full_image_series(model, tensor, full_size)
            small = F.interpolate(
                tensor.unsqueeze(0), size=(full_size, full_size), mode="area"
            ).squeeze(0).numpy()
            squashed_input = np.transpose(
                np.clip((small * std + mean) * 255, 0, 255).astype(np.uint8), (1, 2, 0)
            )
        total_bytes = sum(grids.nbytes for grids in series.values())
        for key, grids in series.items():
            LOGGER.info(
                "%s | %s: %d tiles x %dx%d patches x %d dims",
                name, key, grids.shape[0], grids.shape[1], grids.shape[2], grids.shape[3],
            )
        if full_series is not None:
            LOGGER.info(
                "%s | full image (no tiling): %d series x %dx%d patches",
                name, len(full_series),
                full_series[avg_key].shape[0], full_series[avg_key].shape[1],
            )
        LOGGER.info("%s | feature tensors held: %.1f GB", name, total_bytes / 1e9)
        per_image.append((name, image_path, series, full_series, squashed_input, layout))

    # ---- reducer fits ------------------------------------------------------
    def _fit(rows):
        reducer = build_reducer(
            opts["reducer"].lower(), opts["seed"],
            opts["umap_neighbors"], opts["umap_min_dist"], rows.shape[0],
        )
        return fit_reducer(reducer, rows, opts["max_fit_rows"], opts["seed"])

    shared_fits = None
    shared_full_fits = None
    if opts["fit_scope"] == "category":
        # One fit per series over pooled rows -> colors comparable across images.
        shared_fits = {}
        for key in per_image[0][2]:
            all_rows = np.concatenate(
                [series[key].reshape(-1, series[key].shape[-1])
                 for _n, _p, series, _f, _s, _l in per_image],
                axis=0,
            )
            shared_fits[key] = _fit(all_rows)
            del all_rows
        if per_image[0][3] is not None:
            # Row-2 series get their own pooled fits (never the tiled ones).
            shared_full_fits = {}
            for key in per_image[0][3]:
                all_rows = np.concatenate(
                    [full[key].reshape(-1, full[key].shape[-1])
                     for _n, _p, _s, full, _sq, _l in per_image],
                    axis=0,
                )
                shared_full_fits[key] = _fit(all_rows)
                del all_rows

    # ---- pass 2: reduce -> RGB -> stitch -> panel -------------------------
    backbone_name = getattr(model.backbone, "name", model_source)
    for name, image_path, series, full_series, squashed_input, layout in per_image:
        # Row 1: original + one column per TILED series.
        tiled_rgbs, tiled_titles, npz_coords = [], [], {}
        for key, grids in series.items():
            t, ph, pw, dim = grids.shape
            rows = grids.reshape(-1, dim)
            reducer, lo, hi = shared_fits[key] if shared_fits else _fit(rows)
            coords = transform_rows(reducer, rows)
            tile_rgb = coords_to_rgb(coords, lo, hi).reshape(t, ph, pw, 3)
            tiled_rgbs.append(stitch_rgb(model.tile_maker, tile_rgb, layout))
            tiled_titles.append(f"{key} ({dim}d)\n{opts['reducer']}→RGB, tiled")
            npz_coords[_npz_key(key)] = coords
            npz_coords[_npz_key(key) + "_grid_shape"] = np.array([t, ph, pw, dim])

        with PIL.Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        panel_rows = [[rgb] + tiled_rgbs]
        panel_titles = [["original + tile grid"] + tiled_titles]

        # Row 2 (optional): squashed backbone input + NON-tiled counterparts.
        # Each row-2 series transforms through its OWN fit (not the tiled
        # column's): colors stay stable across tile sizes, at the price of
        # hue comparability between the two rows.
        if full_series is not None:
            full_rgbs = [upsample_grid_to_image(
                squashed_input.astype(np.float32) / 255.0, layout.image_hw
            )]
            full_titles = [f"full image→{full_size}² (squashed input)"]
            for key, grids in series.items():
                full_grid = full_series[key]
                sh, sw, dim = full_grid.shape
                rows = full_grid.reshape(-1, dim)
                reducer, lo, hi = (
                    shared_full_fits[key] if shared_full_fits else _fit(rows)
                )
                coords = transform_rows(reducer, rows)
                full_rgb = coords_to_rgb(coords, lo, hi).reshape(sh, sw, 3)
                full_rgbs.append(upsample_grid_to_image(full_rgb, layout.image_hw))
                full_titles.append(
                    f"{key} ({dim}d)\nfull→{full_size}², own-fit colors"
                )
                npz_key = _npz_key(key).replace("coords_", "coords_full_", 1)
                npz_coords[npz_key] = coords
                npz_coords[npz_key + "_grid_shape"] = np.array([sh, sw, dim])
            panel_rows.append(full_rgbs)
            panel_titles.append(full_titles)

        reducer_desc = opts["reducer"]
        if opts["reducer"] in ("umap", "cuml_umap"):
            reducer_desc += f"(n={opts['umap_neighbors']}, d={opts['umap_min_dist']})"
        geom = (
            f"{layout.num_tiles} tiles (tile={layout.tile_size}, "
            f"stride={layout.stride}, pad={model.tile_maker.pad_value:g})"
        )
        if full_series is not None:
            geom += f" | full→{full_size}²"
        title = (
            f"{tag}: {name}\n"
            f"backbone={backbone_name} ({model_source}) | layers={layer_names} "
            f"patchsize={model.patch_maker.patchsize} | dims "
            f"{model.forward_modules['preprocessing'].output_dim}→"
            f"{model.forward_modules['preadapt_aggregator'].target_dim}\n"
            f"{geom} | {reducer_desc}→RGB, {opts['fit_scope']}-fit, "
            f"seed={opts['seed']}"
        )

        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(opts["out_dir"], f"{tag}_{stem}_features.png")
        plot_feature_panel(panel_rows, panel_titles, layout, save_path, title=title)
        LOGGER.info("Saved %s", save_path)

        if opts["save_npz"]:
            npz_path = os.path.splitext(save_path)[0] + ".npz"
            np.savez(
                npz_path,
                origins=np.asarray(layout.origins),
                tile_size=layout.tile_size,
                stride=layout.stride,
                image_path=image_path,
                **npz_coords,
            )
            LOGGER.info("Saved %s", npz_path)

    LOGGER.info("All artifacts in %s", opts["out_dir"])


if __name__ == "__main__":
    main()
