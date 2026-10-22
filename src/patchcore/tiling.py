"""Tiling utilities for very large images (Tiled PatchCore).

A tile is treated exactly like an independent image: the dataloader batch
dimension simply becomes the tile dimension ``B images -> B*T tiles``.
This module only adds grid construction, padding and (overlap-averaged)
map stitching — the PatchCore core algorithm is unaware of tiling.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# DINOv3 ViT patch size. Tiles must be a multiple of this so that the
# patch-token grid is a perfect square (see DinoV3Backbone.format_to_spatial_grid).
_PATCH_SIZE = 16


@dataclass
class TileLayout:
    """Geometry of the tile grid for one image size (all batch entries share it)."""

    image_hw: tuple
    padded_hw: tuple
    tile_size: int
    stride: int
    origins: list = field(default_factory=list)  # row-major (y, x) tile corners

    @property
    def num_tiles(self) -> int:
        return len(self.origins)


class TileMaker:
    """Cuts images into a fixed grid of (overlapping) backbone-sized tiles.

    Args:
        tile_size: square tile edge in pixels. Must be a multiple of 16
            (DINOv3 square-grid constraint).
        overlap: pixel overlap between adjacent tiles. Default ``tile_size // 2``.
        stride: alternative to ``overlap`` (stride = tile_size - overlap).
            Provide exactly one of overlap/stride.
        pad_value: constant value used to pad right/bottom borders. Dataset
            images are ImageNet-normalized, so 0.0 pads with the dataset mean
            (neutral grey). Training and inference must pad identically.
    """

    def __init__(self, tile_size=384, overlap=None, stride=None, pad_value=0.0):
        if tile_size % _PATCH_SIZE != 0:
            raise ValueError(
                f"tile_size must be a multiple of {_PATCH_SIZE} (DINOv3 square "
                f"patch-grid constraint), got {tile_size}."
            )
        if overlap is not None and stride is not None:
            raise ValueError("Provide only one of overlap/stride, not both.")
        if overlap is None and stride is None:
            overlap = tile_size // 2
        if stride is not None:
            if not 1 <= stride <= tile_size:
                raise ValueError(f"stride must be in [1, tile_size], got {stride}.")
            self.stride = stride
        else:
            if not 0 <= overlap < tile_size:
                raise ValueError(f"overlap must be in [0, tile_size), got {overlap}.")
            self.overlap = overlap
            self.stride = tile_size - overlap
        self.tile_size = tile_size
        self.overlap = self.tile_size - self.stride
        self.pad_value = float(pad_value)

    def _padded_extent(self, extent):
        """Smallest ``tile + k*stride`` >= extent (k >= 0)."""
        if extent <= self.tile_size:
            return self.tile_size
        k = math.ceil((extent - self.tile_size) / self.stride)
        return self.tile_size + k * self.stride

    def compute_layout(self, height, width) -> TileLayout:
        padded_h = self._padded_extent(height)
        padded_w = self._padded_extent(width)
        ys = range(0, padded_h - self.tile_size + 1, self.stride)
        xs = range(0, padded_w - self.tile_size + 1, self.stride)
        origins = [(y, x) for y in ys for x in xs]  # row-major
        return TileLayout(
            image_hw=(height, width),
            padded_hw=(padded_h, padded_w),
            tile_size=self.tile_size,
            stride=self.stride,
            origins=origins,
        )

    def extract(self, images):
        """Split a batch of images into a flat batch of tiles.

        Args:
            images: [B, 3, H, W] tensor (any device).

        Returns:
            (tiles [B*T, 3, tile, tile], TileLayout for this batch)
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected [B,3,H,W] images, got shape {tuple(images.shape)}.")
        batch, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3 image channels, got {channels}.")

        layout = self.compute_layout(height, width)
        pad_h = layout.padded_hw[0] - height
        pad_w = layout.padded_hw[1] - width
        if pad_h or pad_w:
            # F.pad pads last dim first: (left, right, top, bottom) -> right/bottom only.
            images = F.pad(
                images, (0, pad_w, 0, pad_h), mode="constant", value=self.pad_value
            )

        tiles = images.unfold(2, self.tile_size, self.stride).unfold(
            3, self.tile_size, self.stride
        )
        # [B, C, ny, nx, tile, tile] -> [B, ny, nx, C, tile, tile] ->
        # [B*ny*nx, C, tile, tile]; row-major tile order matches TileLayout.origins.
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
        tiles = tiles.reshape(batch * layout.num_tiles, channels, self.tile_size, self.tile_size)
        return tiles.contiguous(), layout

    def stitch(self, tile_maps, layout, upsample=True):
        """Reassemble per-tile score maps into one full-resolution map per image.

        Overlapping tile regions are averaged (overlap-add divided by the
        count of contributing tiles). Optionally each ``m x m`` tile score
        grid is bilinearly upsampled to ``tile_size x tile_size`` first.

        Args:
            tile_maps: [B*T, m, m] torch tensor or numpy array.
            layout: TileLayout returned by :meth:`extract`.
            upsample: interpolate tile maps to tile_size before stitching.

        Returns:
            [B, H, W] numpy array cropped to the original image size.
        """
        if not isinstance(tile_maps, torch.Tensor):
            tile_maps = torch.from_numpy(np.ascontiguousarray(tile_maps, dtype=np.float32))
        else:
            tile_maps = tile_maps.detach().cpu().to(torch.float)
        if tile_maps.ndim == 2:
            tile_maps = tile_maps.unsqueeze(0)
        n_tiles_total, m, mm = tile_maps.shape
        if m != mm:
            raise ValueError(f"Tile maps must be square, got ({m}, {mm}).")
        if n_tiles_total % layout.num_tiles != 0:
            raise ValueError(
                f"Got {n_tiles_total} tile maps, expected a multiple of "
                f"{layout.num_tiles} tiles per image."
            )
        batch = n_tiles_total // layout.num_tiles

        tile = layout.tile_size
        if m != tile:
            if not upsample:
                raise ValueError(
                    f"Tile maps of size ({m}, {m}) cannot be stitched onto a "
                    f"{tile}x{tile} tile canvas without upsampling."
                )
            tile_maps = F.interpolate(
                tile_maps.unsqueeze(1),
                size=(tile, tile),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        padded_h, padded_w = layout.padded_hw
        sums = torch.zeros((batch, padded_h, padded_w), dtype=torch.float)
        weights = torch.zeros((batch, padded_h, padded_w), dtype=torch.float)
        reshaped = tile_maps.reshape(batch, layout.num_tiles, tile, tile)
        for idx, (y, x) in enumerate(layout.origins):
            patch = reshaped[:, idx]
            sums[:, y : y + tile, x : x + tile] += patch
            weights[:, y : y + tile, x : x + tile] += 1.0

        if (weights < 1).any():
            raise RuntimeError("Tile layout left uncovered pixels (division by zero).")
        stitched = sums / weights
        height, width = layout.image_hw
        stitched = stitched[:, :height, :width]
        return stitched.numpy()
