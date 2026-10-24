"""Figures: before/after RGB, ground-truth mask, prediction, error map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

# OSCD bands come as 16-bit reflectance scaled by 1e4, so a raw multiplication
# by any brighten factor saturates a whole city to white.
REFLECTANCE_SCALE = 10_000.0
RED, GREEN, BLUE = (config.BAND_INDICES[k] for k in ("B04", "B03", "B02"))


def truecolor(stack: np.ndarray, brighten: float = 2.7) -> np.ndarray:
    """(H, W, 12) uint16 stack -> (H, W, 3) RGB float in [0, 1]."""
    rgb = np.stack([stack[..., RED], stack[..., GREEN], stack[..., BLUE]], axis=-1)
    rgb = rgb.astype(np.float32) / REFLECTANCE_SCALE
    return np.clip(rgb * brighten, 0, 1)


def save_figure(path: Path, title: str, pre: np.ndarray, post: np.ndarray,
                mask: np.ndarray, pred: np.ndarray, nir: np.ndarray | None = None) -> None:
    """Save a 1x4 panel: before, after, ground truth, prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        (truecolor(pre), "Before (RGB)"),
        (truecolor(post), "After (RGB)"),
        (mask, "Ground truth change"),
        (pred, "Predicted change"),
    ]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    if nir is not None:
        err = (pred.astype(int) - mask.astype(int)).astype(int)
        axes[2].imshow(np.ma.masked_where(err == 0, err), cmap="coolwarm", alpha=0.5,
                       vmin=-1, vmax=1)
    for ax in axes:
        ax.set_anchor("N")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_change_map(path: Path, city_pre: np.ndarray, city_post: np.ndarray,
                    cleared: np.ndarray, title: str) -> None:
    """Red overlay of cleared forest on the after image (the selling figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(truecolor(city_post))
    ax.imshow(np.ma.masked_where(cleared == 0, cleared), cmap="autumn",
              alpha=0.65, vmin=0, vmax=1)
    ha = float(cleared.sum()) * 100 / 10_000
    ax.set_title(f"{title}\ncleared: {ha:.2f} ha", fontsize=11)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
