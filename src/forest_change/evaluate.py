"""IoU, F1 and per-tile metrics for the change-detection model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def _flat(y_true: np.ndarray, y_pred: np.ndarray):
    return y_true.ravel().astype(int), y_pred.ravel().astype(int)


def pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Class-level and pixel-level metrics, all with the "change" class as 1."""
    t, p = _flat(y_true, y_pred)
    iou_0 = jaccard_score(t, p, pos_label=0, zero_division=0)
    iou_1 = jaccard_score(t, p, pos_label=1, zero_division=0)
    return {
        "iou_change": round(float(iou_1), 4),
        "iou_no_change": round(float(iou_0), 4),
        "mean_iou": round(float((iou_0 + iou_1) / 2), 4),
        "f1_change": round(float(f1_score(t, p, pos_label=1, zero_division=0)), 4),
        "precision_change": round(float(precision_score(t, p, pos_label=1, zero_division=0)), 4),
        "recall_change": round(float(recall_score(t, p, pos_label=1, zero_division=0)), 4),
    }


def tile_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tile-level detection: does the tile contain any predicted change?"""
    t = np.asarray(y_true).reshape(len(y_true), -1).max(axis=1).astype(int)
    p = np.asarray(y_pred).reshape(len(y_pred), -1).max(axis=1).astype(int)
    return {
        "tile_detect_f1": round(float(f1_score(t, p, pos_label=1, zero_division=0)), 4),
        "tile_detect_precision": round(float(precision_score(t, p, pos_label=1, zero_division=0)), 4),
        "tile_detect_recall": round(float(recall_score(t, p, pos_label=1, zero_division=0)), 4),
    }


def area_metrics(mask: np.ndarray, pred: np.ndarray, pixel_m2: int = 100) -> dict:
    """Hectares of true and predicted changed area, plus the absolute error."""
    def ha(a: np.ndarray) -> float:
        return float(a.sum()) * pixel_m2 / 10_000.0
    t_ha, p_ha = ha(mask), ha(pred)
    return {"true_ha": round(t_ha, 2), "pred_ha": round(p_ha, 2),
            "abs_ha_error": round(abs(t_ha - p_ha), 2)}


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, pixel_m2: int = 100) -> dict:
    out = pixel_metrics(y_true, y_pred)
    out.update(tile_metrics(y_true, y_pred))
    out.update(area_metrics(y_true, y_pred, pixel_m2))
    return out


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
