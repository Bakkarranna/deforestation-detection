"""Train-free NDVI-loss baseline for cleared-forest detection.

The baseline is the classic remote-sensing rule: healthy vegetation scores high
on NDVI, so a pixel that was forest before and is not forest after is a
candidate clearing. It needs no training, no checkpoint and no GPU, which makes
it the honest floor for the U-Net: any learned model that does not beat it has
learned nothing.

The threshold is the *only* knob, and it is chosen on the training cities only.
The held-out test cities are scored with the number picked on train, so the
comparison stays honest in the same way the U-Net's does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import change, data, evaluate, preprocess

# NDVI > 0.6 is the usual "dense vegetation" cut for 10 m Sentinel-2 L2A. Above
# it, the NIR reflectance dominates the red by a wide margin; below it, bare
# soil, built-up land and water all sit.
DEFAULT_FOREST_THRESHOLD = 0.6
# The grid extends down to 0.05 so the optimum is never pinned to a grid edge:
# on the first pass it landed exactly on the lowest value offered, which says the
# true optimum is somewhere below it and the reported baseline was not tuned.
THRESHOLD_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


def forest_mask(stack: np.ndarray, threshold: float = DEFAULT_FOREST_THRESHOLD) -> np.ndarray:
    """Boolean forest mask for one (H, W, 12) band stack."""
    ndvi_image, _ = preprocess.ndvi(stack, stack)
    return ndvi_image > threshold


def cleared_mask(pre: np.ndarray, post: np.ndarray,
                 threshold: float = DEFAULT_FOREST_THRESHOLD,
                 min_pixels: int = change.MIN_PIXELS) -> np.ndarray:
    """Forest at t1 and not forest at t2, speckle-filtered."""
    before = forest_mask(pre, threshold)
    after = forest_mask(post, threshold)
    return change.min_patch(before & ~after, min_pixels).astype(np.uint8)


def predict_city(city: data.City, threshold: float = DEFAULT_FOREST_THRESHOLD) -> np.ndarray:
    """Full-city binary change prediction for one threshold."""
    return cleared_mask(city.pre, city.post, threshold)


def score_cities(cities, threshold: float) -> dict:
    """Mean pixel metrics and hectares over a list of cities at one threshold."""
    per_city, ious, f1s = {}, [], []
    for city in cities:
        pred = predict_city(city, threshold)
        m = evaluate.evaluate_all(city.mask, pred)
        m["city"] = city.name
        per_city[city.name] = m
        ious.append(m["iou_change"])
        f1s.append(m["f1_change"])
    mean = {
        "iou_change": round(float(np.mean(ious)), 4) if ious else 0.0,
        "f1_change": round(float(np.mean(f1s)), 4) if f1s else 0.0,
    }
    return {"per_city": per_city, "mean": mean}


def tune_threshold(cities, grid=None) -> tuple[float, list[dict]]:
    """Pick the forest threshold with the best mean IoU on the given cities."""
    grid = grid or THRESHOLD_GRID
    rows, best_threshold, best_iou = [], grid[0], -1.0
    for threshold in grid:
        mean = score_cities(cities, threshold)["mean"]
        rows.append({"threshold": threshold, **mean})
        if mean["iou_change"] > best_iou:
            best_threshold, best_iou = threshold, mean["iou_change"]
    return best_threshold, rows


def run_baseline(raw_dir: Path, out_dir: Path, grid=None) -> dict:
    """Tune on train cities, then score the held-out test cities once."""
    train_cities, test_cities = data.city_split(raw_dir)
    train_objs = data.load_cities(raw_dir, train_cities)
    test_objs = data.load_cities(raw_dir, test_cities)

    threshold, tuning = tune_threshold(train_objs, grid)
    print(f"baseline: NDVI forest threshold tuned on train -> {threshold}", flush=True)

    test = score_cities(test_objs, threshold)
    for name, m in test["per_city"].items():
        print(f"  {name}: IoU(change) {m['iou_change']:.4f} F1 {m['f1_change']:.4f} "
              f"pred {m['pred_ha']:.0f} ha / true {m['true_ha']:.0f} ha", flush=True)
    print(f"  mean: IoU {test['mean']['iou_change']:.4f} "
          f"F1 {test['mean']['f1_change']:.4f}", flush=True)

    payload = {
        "model": "NDVI-loss baseline",
        "rule": f"cleared = NDVI(t1) > {threshold} AND NDVI(t2) <= {threshold}, "
                f"speckle filter >= {change.MIN_PIXELS} px",
        "forest_threshold": threshold,
        "threshold_grid": grid or THRESHOLD_GRID,
        "tuning": tuning,
        "test": test,
        "train_cities": train_cities,
        "test_cities": test_cities,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    import sys

    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    run_baseline(raw, out)


if __name__ == "__main__":
    main()
