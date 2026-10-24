"""Combine the U-Net and the NDVI baseline into one README-ready results table.

Reads ``results/metrics.json`` (U-Net, written by ``forest_change.train``) and
``results/baseline_metrics.json`` (NDVI baseline, written by
``forest_change.baseline``) and writes ``results/results_table.md`` plus a CSV,
so every number in the README is generated from the actual runs rather than
typed by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config

MODEL_KEYS = ["iou_change", "f1_change", "precision_change", "recall_change"]
BASELINE_LABEL = "NDVI-loss baseline"


def load(path: Path) -> dict:
    """Read a metrics JSON into {'per_city': {...}, 'mean': {...}, ...}."""
    if not path.exists():
        return {"per_city": {}, "mean": {}}
    payload = json.loads(path.read_text())
    block = payload.get("test", payload)
    return {"per_city": block.get("per_city", {}), "mean": block.get("mean", {}),
            "forest_threshold": payload.get("forest_threshold"),
            "run": payload.get("run", {})}


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _unet_label(run: dict) -> str:
    """Label the encoder that was actually trained, from the run record.

    `forest_change.train` writes the encoder name and epoch count into
    `metrics.json`, so the table follows the real run instead of a label that
    goes stale when someone switches the encoder (as happened when the default
    moved from ResNet-34 to MobileNetV2).
    """
    if not run:
        return "U-Net"
    encoder = run.get("encoder") or "mobilenet_v2"
    channels = run.get("in_channels", 24)
    epochs = run.get("epochs")
    tail = f", {epochs} epochs" if epochs else ""
    return f"U-Net ({encoder}, {channels} ch{tail})"


def _mean(block: dict) -> dict:
    """Per-metric mean over the held-out cities.

    `baseline_metrics.json` carries a `mean` block, `metrics.json` does not, so
    for the U-Net run the mean is computed from the same per-city numbers the
    table prints. Only metrics that every city reports are averaged, so a
    missing precision/recall stays `-` instead of being read as 0.
    """
    mean = dict(block.get("mean") or {})
    if mean:
        return mean
    per_city = block.get("per_city", {})
    if not per_city:
        return {}
    return {key: sum(m[key] for m in per_city.values()) / len(per_city)
            for key in MODEL_KEYS if all(key in m for m in per_city.values())}


def render(unet: dict, baseline: dict) -> str:
    lines = [
        "## Mean over the held-out test cities",
        "",
        "| Model | IoU (change) | F1 (change) | Precision | Recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in (("unet", _unet_label(unet.get("run", {}))),
                       ("baseline", BASELINE_LABEL)):
        mean = _mean(unet if key == "unet" else baseline)
        if not mean:
            continue
        lines.append("| " + label + " | " + " | ".join(
            _fmt(mean.get(k)) for k in MODEL_KEYS) + " |")

    lines += ["", "## Per city (held-out)", "",
              "| City | U-Net IoU | U-Net F1 | Baseline IoU | Baseline F1 | "
              "True ha | U-Net ha | Baseline ha |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    up, bp = unet.get("per_city", {}), baseline.get("per_city", {})
    for city in sorted(set(up) | set(bp)):
        u, b = up.get(city, {}), bp.get(city, {})
        lines.append(
            f"| {city} | {_fmt(u.get('iou_change'))} | {_fmt(u.get('f1_change'))} "
            f"| {_fmt(b.get('iou_change'))} | {_fmt(b.get('f1_change'))} "
            f"| {u.get('true_ha', 0):.1f} | {u.get('pred_ha', 0):.1f} "
            f"| {b.get('pred_ha', 0):.1f} |")
    return "\n".join(lines) + "\n"


def write_table(results_dir: Path = config.RESULTS_DIR) -> str:
    """Write results_table.md/.csv from the two metric files; returns the md."""
    unet = load(results_dir / "metrics.json")
    baseline = load(results_dir / "baseline_metrics.json")
    md = render(unet, baseline)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results_table.md").write_text(md)
    cols = ["city", "source", *MODEL_KEYS, "true_ha", "pred_ha"]
    with open(results_dir / "results_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for source, block in (("unet", unet), ("baseline", baseline)):
            for city, m in block.get("per_city", {}).items():
                w.writerow({"city": city, "source": source, **m,
                            "true_ha": m.get("true_ha"),
                            "pred_ha": m.get("pred_ha")})
        for source, block in (("unet", unet), ("baseline", baseline)):
            mean = _mean(block)
            if mean:
                w.writerow({"city": "MEAN", "source": source, **mean})
    return md


def main() -> None:
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR)
    print(write_table(target), end="")


if __name__ == "__main__":
    main()
