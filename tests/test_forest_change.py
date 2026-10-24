"""Unit tests for the forest change pipeline (pure numpy/scipy - no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forest_change import baseline, change, config, data, evaluate, preprocess  # noqa: E402


def make_city(h: int = 256, w: int = 256, bands: int = 12) -> data.City:
    """A synthetic city: forest everywhere except a cleared rectangle.

    Forest needs NIR (B08, index 7) far above Red (B04, index 3), so the
    cleared block is the NIR drop, exactly as on real Sentinel-2 pixels.
    """
    rng = np.random.default_rng(config.SEED)
    ones = np.full((h, w, bands), 400, np.uint16)
    ones[..., 7] = 2200  # B08 NIR
    noise = rng.integers(0, 30, size=(h, w, bands)).astype(np.uint16)
    pre = np.clip(ones + noise, 0, None).astype(np.uint16)
    post = pre.copy()
    cleared = np.zeros((h, w), bool)
    cleared[64:192, 64:192] = True
    # Clearing drops NIR to bare-soil level, so NDVI collapses there.
    post[..., 7][cleared] = 350
    return data.City(name="toy", pre=pre, post=post, mask=cleared.astype(np.uint8))


def test_ndvi_forest_mask_matches_ground_truth():
    city = make_city()
    # NDVI of the *second* date: the cleared block has lost its NIR there.
    ndvi_post = preprocess.ndvi(city.pre, city.post)[1]
    forest = preprocess.ndvi_forest(ndvi_post)
    assert forest[10, 10] == 1          # untouched forest
    assert forest[120, 120] == 0        # cleared block


def test_ndvi_loss_baseline_recovers_the_cleared_block():
    city = make_city()
    pred = baseline.predict_city(city).astype(bool)
    truth = city.mask.astype(bool)
    inter = float((pred & truth).sum())
    union = float((pred | truth).sum())
    assert inter / union > 0.9


def test_decode_mask_handles_both_oscd_encodings():
    """The shipped rasters are 1/2 even though the README says 0/1."""
    shipped = np.array([[1, 2, 2], [1, 1, 2]])
    assert float(data.decode_mask(shipped).mean()) == 3 / 6
    documented = np.array([[0, 1, 1], [0, 0, 1]])
    assert float(data.decode_mask(documented).mean()) == 3 / 6
    png_style = np.array([[0, 255], [255, 0]])
    assert float(data.decode_mask(png_style).mean()) == 0.5


def test_min_patch_drops_speckle():
    mask = np.zeros((64, 64), np.uint8)
    mask[0, 0] = 1                       # single pixel, well below MIN_PIXELS
    mask[10:60, 10:60] = 1               # 2500 px block, kept
    out = change.min_patch(mask, change.MIN_PIXELS)
    assert out[0, 0] == 0 and out[30, 30] == 1


def test_hectares_and_area_metrics():
    mask = np.zeros((10, 10), np.uint8)
    mask[:5] = 1
    m = evaluate.area_metrics(mask, mask)
    assert m["true_ha"] == pytest.approx(50 * 100 / 10_000)
    assert m["abs_ha_error"] == 0.0


def test_pixel_metrics_perfect_and_empty():
    y = np.zeros((8, 8), np.uint8)
    y[2:4, 2:4] = 1
    perfect = evaluate.pixel_metrics(y, y.copy())
    assert perfect["iou_change"] == 1.0 and perfect["f1_change"] == 1.0
    empty = evaluate.pixel_metrics(y, np.zeros_like(y))
    assert empty["iou_change"] == 0.0 and empty["f1_change"] == 0.0


def test_threshold_tuning_uses_the_grid():
    city = make_city()
    best, rows = baseline.tune_threshold([city], [0.4, 0.5, 0.6, 0.7])
    assert best in (0.4, 0.5, 0.6, 0.7)
    assert len(rows) == 4 and all("iou_change" in r for r in rows)


def test_scale_clips_to_unit_range():
    stats = {"lo": np.zeros(12, np.float32), "hi": np.full(12, 4000.0, np.float32)}
    dark = preprocess.scale(np.zeros((4, 4, 12), np.uint16), stats)
    bright = preprocess.scale(np.full((2, 2, 12), 8000, np.uint16), stats)
    assert dark.dtype == np.float32 and float(dark.max()) == 0.0
    assert float(bright.max()) == 1.0


def test_band_stats_lengths_and_order():
    stats = preprocess.band_stats([make_city()])
    assert len(stats["lo"]) == len(config.BANDS) == 12
    assert (stats["hi"] >= stats["lo"]).all()
