"""Change maps between two dates: model, NDVI baseline and area accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import config

PIXEL_M2 = 100  # Sentinel-2 10 m x 10 m

# 50 px at 10 m = 0.05 ha. Anything smaller is co-registration noise or a thin
# cloud edge, not a clearing.
MIN_PIXELS = 50


@dataclass
class ChangeMap:
    pred: np.ndarray  # (H, W) bool, changed
    forest_before: np.ndarray  # (H, W) bool
    forest_after: np.ndarray  # (H, W) bool


def min_patch(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected components smaller than min_pixels (speckle filter).

    At 10 m, 50 pixels = 0.05 ha. Without this, single-pixel noise from
    co-registration error and thin clouds shows up as cleared forest.
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(mask, np.float32), labels, range(1, n + 1))
    keep = np.where(sizes >= min_pixels)[0] + 1
    return np.isin(labels, keep)


def forest_change(forest_before: np.ndarray, forest_after: np.ndarray,
                  min_pixels: int = 50) -> np.ndarray:
    """Pixels forest at t1 and not forest at t2: candidate deforestation."""
    cleared = forest_before & ~forest_after
    return min_patch(cleared, min_pixels)


def hectares(mask: np.ndarray, pixel_m2: int = PIXEL_M2) -> float:
    return float(np.asarray(mask).sum()) * pixel_m2 / 10_000.0


def compare_areas(model_change: np.ndarray, baseline_change: np.ndarray,
                  truth: np.ndarray) -> dict:
    """Hectares for model, NDVI baseline and the reference mask."""
    return {
        "model_ha": round(hectares(model_change), 2),
        "baseline_ha": round(hectares(baseline_change), 2),
        "reference_ha": round(hectares(truth), 2),
    }


def predict_full_city(model, city, stats, tile: int = config.TILE,
                      stride: int = config.STRIDE, device: str = "cpu") -> np.ndarray:
    """Run the model over a whole city by stitching overlapping tiles.

    Predictions are averaged in the overlap so tile boundaries do not leave
    visible seams in the change map.
    """
    import torch

    from .preprocess import scale

    h, w = city.mask.shape
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    model.eval()
    with torch.no_grad():
        for row, col in city.tiles(tile, stride):
            pre = scale(city.pre[row:row + tile, col:col + tile], stats)
            post = scale(city.post[row:row + tile, col:col + tile], stats)
            x = torch.from_numpy(
                np.concatenate([pre, post], -1).transpose(2, 0, 1)[None].astype(np.float32)
            ).to(device)
            p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            acc[row:row + tile, col:col + tile] += p
            cnt[row:row + tile, col:col + tile] += 1
    cnt[cnt == 0] = 1
    return acc / cnt
