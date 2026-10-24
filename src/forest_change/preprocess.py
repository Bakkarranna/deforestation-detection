"""Band scaling, NDVI baseline forest masks and tiling helpers."""

from __future__ import annotations

import numpy as np

from . import config

# Sentinel-2 Blue, Green, Red, NIR (B8) indices inside config.BANDS
B02, B03, B04, B08 = 1, 2, 3, 7


def band_stats(cities) -> dict:
    """Per-band 2nd/98th percentiles across a list of City, for scaling.

    Sentinel-2 reflectances are stored as uint16 with a x10000 scale factor, so
    raw magnitudes span 0-10000 while most usable signal sits in the low
    thousands. Percentile clipping is more robust than a fixed divisor because
    bright deserts and dark water would otherwise dominate one axis.
    """
    lo, hi = {b: [] for b in range(len(config.BANDS))}, {b: [] for b in range(len(config.BANDS))}
    sample = []
    for city in cities:
        sample.append(city.pre.reshape(-1, len(config.BANDS)))
        sample.append(city.post.reshape(-1, len(config.BANDS)))
    if not sample:
        return {"lo": np.zeros(len(config.BANDS), np.float32),
                "hi": np.ones(len(config.BANDS), np.float32)}
    flat = np.concatenate(sample, axis=0)
    # Sub-sample for speed: percentiles are stable at a few million pixels.
    if len(flat) > 4_000_000:
        rng = np.random.default_rng(config.SEED)
        flat = flat[rng.choice(len(flat), 4_000_000, replace=False)]
    for b in range(len(config.BANDS)):
        lo[b] = np.percentile(flat[:, b], 2)
        hi[b] = np.percentile(flat[:, b], 98)
    return {"lo": np.asarray([lo[b] for b in range(len(config.BANDS))], np.float32),
            "hi": np.asarray([hi[b] for b in range(len(config.BANDS))], np.float32)}


def scale(stack: np.ndarray, stats: dict) -> np.ndarray:
    """Clip to the percentile range and scale each band to [0, 1]."""
    lo, hi = stats["lo"], stats["hi"]
    out = (stack.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0)


def ndvi(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Normalized difference vegetation index for two (H, W, 12) stacks.

    (NIR - Red) / (NIR + Red): healthy vegetation scores high, bare soil and
    built-up land score low. This is the classic remote-sensing forest test.
    """
    out = []
    for stack in (pre, post):
        nir = stack[..., B08].astype(np.float32)
        red = stack[..., B04].astype(np.float32)
        out.append((nir - red) / (nir + red + 1e-6))
    return out[0], out[1]


def ndvi_forest(ndvi_img: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """Forest mask from an NDVI image and threshold."""
    return (ndvi_img > threshold).astype(np.uint8)


def forest_t1(pre: np.ndarray) -> np.ndarray:
    """Forest mask for the first date (before), used to define cleared areas."""
    ndvi_pre, _ = ndvi(pre, pre)
    return ndvi_forest(ndvi_pre)


def tile_pre(city, stats: dict, size: int = config.TILE, stride: int = config.STRIDE):
    """Yield scaled pre/post tiles plus the mask, skipping empty tiles."""
    for row, col in city.tiles(size, stride):
        pre = scale(city.pre[row:row + size, col:col + size], stats)
        post = scale(city.post[row:row + size, col:col + size], stats)
        mask = city.mask[row:row + size, col:col + size]
        if mask.sum() == 0 and pre.mean() < 0.05:
            continue
        yield pre, post, mask, row, col


def tile_mask_only(city, size: int = config.TILE, stride: int = config.STRIDE):
    """Yield (row, col, mask) tile positions without scaling (for baselines)."""
    for row, col in city.tiles(size, stride):
        yield row, col, city.mask[row:row + size, col:col + size]
