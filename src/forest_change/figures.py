"""Charts and qualitative figures for the results report.

Everything here is generated from artefacts the run already wrote: the metrics
JSON files, the per-epoch history and the ``prob_<city>.npy`` prediction maps.
No figure retrains the model or re-runs inference, so the pictures and the
reported numbers cannot disagree.

Run from the project root:

    python -m forest_change.figures data/raw results/

PNGs land in ``results/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import baseline, change, data, visualize

FIG_SUBDIR = "figures"
UNET_COLOR = "#2563eb"
BASELINE_COLOR = "#f59e0b"
CITIES_FOR_QUALITATIVE = ["montpellier", "lasvegas", "valencia"]


def _style():
    """Matplotlib in headless mode with a shared, report-friendly look."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _load(out_dir: Path, name: str) -> dict:
    path = out_dir / name
    return json.loads(path.read_text()) if path.exists() else {}


def unet_means(payload: dict) -> dict:
    """Mean of the per-city metrics; ``metrics.json`` only stores per-city."""
    per_city = payload.get("test", payload).get("per_city", {})
    if not per_city:
        return {}
    keys = ["iou_change", "f1_change", "precision_change", "recall_change",
            "iou_no_change", "mean_iou"]
    out = {}
    for key in keys:
        vals = [c[key] for c in per_city.values() if key in c]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    return out


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def chart_training(out_dir: Path, fig_dir: Path) -> Path:
    """Validation loss, IoU/F1 and precision/recall across the 8 epochs."""
    plt = _style()
    run = _load(out_dir, "metrics.json").get("run", {})
    hist = run.get("history", [])
    if not hist:
        raise SystemExit("no epoch history in metrics.json")
    ep = [h["epoch"] for h in hist]
    best = run.get("best_epoch") or max(hist, key=lambda h: h["iou_change"])["epoch"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (key, label, color) in zip(axes, [
            ("loss", "Training loss (Dice+BCE)", "#475569"),
            ("iou_change", "Val IoU (change)", UNET_COLOR),
            ("f1_change", "Val F1 (change)", UNET_COLOR),
            ("precision_change", "Val precision / recall", None)]):
        if color:
            ax.plot(ep, [h[key] for h in hist], "o-", color=color, label=label)
        else:
            ax.plot(ep, [h["precision_change"] for h in hist], "s-",
                    color=UNET_COLOR, label="precision")
            ax.plot(ep, [h["recall_change"] for h in hist], "^--",
                    color="#dc2626", label="recall")
            ax.legend(frameon=False, loc="lower right")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        if key == "iou_change" or key == "f1_change":
            ax.axvline(best, color="#16a34a", lw=1, ls=":")
            ax.annotate(f"best (epoch {best})", (best, 0.02),
                        xytext=(4, 0), textcoords="offset points",
                        color="#16a34a", fontsize=8)
    fig.suptitle(f"Validation over {len(ep)} epochs ({run.get('device', 'cpu')}, "
                 f"{run.get('n_train_tiles', '?')} train / "
                 f"{run.get('n_val_tiles', '?')} val tiles)", fontsize=11)
    fig.tight_layout()
    return _save(fig, fig_dir / "training_curves.png")


def chart_per_city(out_dir: Path, fig_dir: Path) -> Path:
    """IoU and F1 per held-out city: U-Net against the NDVI baseline."""
    plt = _style()
    unet = _load(out_dir, "metrics.json").get("test", {}).get("per_city", {})
    base = _load(out_dir, "baseline_metrics.json").get("test", {}).get("per_city", {})
    cities = sorted(unet | base)
    x = np.arange(len(cities))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharex=True)
    for ax, key, title in zip(axes, ("iou_change", "f1_change"),
                              ("IoU (change)", "F1 (change)")):
        ax.bar(x - 0.2, [unet.get(c, {}).get(key, 0) for c in cities], 0.4,
               label="U-Net", color=UNET_COLOR)
        ax.bar(x + 0.2, [base.get(c, {}).get(key, 0) for c in cities], 0.4,
               label="NDVI baseline", color=BASELINE_COLOR)
        um, bm = unet_means({"test": {"per_city": unet}}), unet_means({"test": {"per_city": base}})
        ax.axhline(um.get(key, 0), color=UNET_COLOR, lw=1, ls=":")
        ax.axhline(bm.get(key, 0), color=BASELINE_COLOR, lw=1, ls=":")
        ax.set_title(f"{title} per held-out city (dotted = mean)")
        ax.set_xticks(x)
        ax.set_xticklabels(cities, rotation=35, ha="right")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return _save(fig, fig_dir / "per_city_metrics.png")


def chart_area(out_dir: Path, fig_dir: Path) -> Path:
    """True vs predicted cleared area per city, in hectares."""
    plt = _style()
    unet = _load(out_dir, "metrics.json").get("test", {}).get("per_city", {})
    base = _load(out_dir, "baseline_metrics.json").get("test", {}).get("per_city", {})
    cities = sorted(unet | base)
    x = np.arange(len(cities))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    axes[0].bar(x - 0.25, [unet.get(c, {}).get("true_ha", 0) for c in cities], 0.25,
                label="Ground truth ha", color="#334155")
    axes[0].bar(x, [unet.get(c, {}).get("pred_ha", 0) for c in cities], 0.25,
                label="U-Net ha", color=UNET_COLOR)
    axes[0].bar(x + 0.25, [base.get(c, {}).get("pred_ha", 0) for c in cities], 0.25,
                label="Baseline ha", color=BASELINE_COLOR)
    axes[0].set_title("Cleared area per city (ha)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cities, rotation=35, ha="right")
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("hectares")

    axes[1].bar(x - 0.2, [unet.get(c, {}).get("abs_ha_error", 0) for c in cities], 0.4,
                label="U-Net", color=UNET_COLOR)
    axes[1].bar(x + 0.2, [base.get(c, {}).get("abs_ha_error", 0) for c in cities], 0.4,
                label="Baseline", color=BASELINE_COLOR)
    axes[1].set_title("Absolute area error (ha)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cities, rotation=35, ha="right")
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("|predicted - true| ha")
    fig.tight_layout()
    return _save(fig, fig_dir / "area_error.png")


def _error_map(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Green = correct hit, red = false alarm, blue = missed clearing."""
    t, p = truth.astype(bool), pred.astype(bool)
    out = np.zeros((*t.shape, 3), np.float32)
    out[t & p] = (0.13, 0.65, 0.35)
    out[p & ~t] = (0.86, 0.20, 0.24)
    out[t & ~p] = (0.20, 0.40, 0.90)
    return out


def figure_city(raw_dir: Path, out_dir: Path, city_name: str, fig_dir: Path) -> list[Path]:
    """Six-panel qualitative figure plus a cleared-forest overlay for one city.

    Reads the saved probability map rather than re-running the model, so the
    picture shows exactly the prediction the metrics were computed from.
    """
    plt = _style()
    city = data.load_city(raw_dir, city_name)
    prob_path = out_dir / f"prob_{city_name}.npy"
    if not prob_path.exists():
        raise SystemExit(f"missing {prob_path}; run forest_change.train first")
    pred = np.load(prob_path) > 0.5
    truth = city.mask.astype(bool)
    base_pred = baseline.predict_city(city).astype(bool)
    m = _load(out_dir, "metrics.json")["test"]["per_city"][city_name]
    bm = _load(out_dir, "baseline_metrics.json")["test"]["per_city"].get(city_name, {})

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    panels = [
        (visualize.truecolor(city.pre), "t1 - before (RGB)"),
        (visualize.truecolor(city.post), "t2 - after (RGB)"),
        (truth, "Ground-truth change"),
        (pred, "U-Net prediction"),
        (base_pred, "NDVI-loss baseline"),
        (_error_map(truth, pred),
         "U-Net errors: green hit / red false alarm / blue miss"),
    ]
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        ax.set_aspect("equal")
    b_iou, b_ha = bm.get("iou_change", 0.0), bm.get("pred_ha", 0.0)
    fig.suptitle(
        f"{city_name}  -  U-Net IoU {m['iou_change']:.3f} | F1 {m['f1_change']:.3f} | "
        f"{m['pred_ha']:.0f} ha of {m['true_ha']:.0f} ha true"
        f"   vs   baseline IoU {b_iou:.3f} | {b_ha:.0f} ha", fontsize=10)
    fig.tight_layout()
    panel = _save(fig, fig_dir / f"qual_{city_name}.png")

    cleared = change.min_patch(pred, change.MIN_PIXELS)
    overlay = fig_dir / f"change_map_{city_name}.png"
    visualize.save_change_map(overlay, city.pre, city.post, cleared, city_name.title())
    return [panel, overlay]


def chart_threshold(out_dir: Path, fig_dir: Path) -> Path:
    """Baseline threshold tuning on the train cities, with the chosen value."""
    plt = _style()
    payload = _load(out_dir, "baseline_metrics.json")
    tuning = payload.get("tuning", [])
    if not tuning:
        raise SystemExit("no threshold tuning rows in baseline_metrics.json")
    thr = [r["threshold"] for r in tuning]
    chosen = payload.get("forest_threshold", thr[0])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(thr, [r["iou_change"] for r in tuning], "o-", color=BASELINE_COLOR,
            label="mean IoU (change)")
    ax.plot(thr, [r["f1_change"] for r in tuning], "s--", color="#b45309",
            label="mean F1 (change)")
    ax.axvline(chosen, color="#16a34a", lw=1.2, ls=":")
    ax.annotate(f"chosen on train: {chosen}", (chosen, 0.1),
                xytext=(6, 0), textcoords="offset points", color="#16a34a")
    ax.set_title("NDVI baseline: forest threshold tuned on train cities only")
    ax.set_xlabel("NDVI forest threshold")
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, fig_dir / "baseline_threshold.png")


def write_all(raw_dir: Path, out_dir: Path) -> list[Path]:
    """Every figure for the report, in a stable order."""
    fig_dir = out_dir / FIG_SUBDIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = [
        chart_training(out_dir, fig_dir),
        chart_per_city(out_dir, fig_dir),
        chart_area(out_dir, fig_dir),
        chart_threshold(out_dir, fig_dir),
    ]
    for city in CITIES_FOR_QUALITATIVE:
        written += figure_city(raw_dir, out_dir, city, fig_dir)
    return written


def main() -> None:
    import sys

    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    for path in write_all(raw, out):
        print(path.as_posix())


if __name__ == "__main__":
    main()
