"""Train the U-Net change model on OSCD train cities.

Run from the project root:

    python -m forest_change.train data/raw results/

Writes `results/model.pt` (best validation IoU checkpoint), `results/metrics.json`
and a per-run entry in `results/runs.csv`.

Run order is `forest_change.preprocess` (band statistics) then `train`, then
`evaluate` on the held-out test cities; see `README.md` for the full commands.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import change, config, data, dataset, evaluate, models, preprocess


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(raw_dir: Path, out_dir: Path, epochs: int = config.EPOCHS,
          batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
          quick: bool = False) -> dict:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _device()
    print(f"device: {dev}", flush=True)

    train_cities, test_cities = data.city_split(raw_dir)
    print(f"train cities ({len(train_cities)}): {train_cities}")
    print(f"test cities  ({len(test_cities)}): {test_cities}")
    print("loading cities (12 GeoTIFF bands + mask per city, two dates):", flush=True)
    train_objs = data.load_cities(raw_dir, train_cities)
    stats = preprocess.band_stats(train_objs)
    np.save(out_dir / "band_stats.npy", np.stack([stats["lo"], stats["hi"]]))

    # Validation = a held-out block of train cities, so the test cities stay
    # untouched until final evaluation.
    val_cities = [c for c in train_cities if c in config.VAL_CITIES]
    fit_cities = [c for c in train_cities if c not in val_cities]
    fit = [c for c in train_objs if c.name in fit_cities]
    val = [c for c in train_objs if c.name in val_cities] or fit[:1]

    train_ds = dataset.TileDataset(fit, stats, augment=True)
    val_ds = dataset.TileDataset(val, stats, augment=False)
    print(f"tiles: train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_dl = dataset.make_loader(train_ds, batch_size, True)
    val_dl = dataset.make_loader(val_ds, batch_size, False)

    model = models.build_unet().to(dev)
    encoder_name = model.encoder_name
    n_ch = 24 if len(config.BANDS) == 12 else len(config.BANDS) * 2
    loss_fn = models.DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best, history = 0.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(x)
            n += len(x)
        train_loss = tot / max(n, 1)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                p = torch.sigmoid(model(x.to(dev))).cpu().numpy()
                ys.append(y.numpy())
                ps.append((p > 0.5).astype(np.float32))
        m = evaluate.pixel_metrics(np.concatenate(ys), np.concatenate(ps))
        sched.step(m["iou_change"])
        history.append({"epoch": epoch, "loss": round(train_loss, 4), **m})
        print(f"epoch {epoch:02d} loss {train_loss:.4f} val IoU(change) "
              f"{m['iou_change']:.4f} F1 {m['f1_change']:.4f}", flush=True)

        if m["iou_change"] > best:
            best = m["iou_change"]
            torch.save({"state_dict": model.state_dict(), "iou": best,
                        "epoch": epoch, "in_channels": n_ch,
                        "encoder": encoder_name},
                       out_dir / "model.pt")
        if quick and epoch >= 2:
            print("quick mode: stopping after 2 epochs")
            break

    return {"history": history, "best_val_iou_change": round(best, 4),
            "train_cities": train_cities, "test_cities": test_cities,
            "val_cities": val_cities, "n_train_tiles": len(train_ds),
            "n_val_tiles": len(val_ds), "device": dev,
            "encoder": encoder_name,
            "in_channels": n_ch,
            "epochs": len(history),
            "best_epoch": max(history, key=lambda h: h["iou_change"])["epoch"]
            if history else 0}


def evaluate_cities(raw_dir: Path, out_dir: Path, batch_size: int = config.BATCH_SIZE) -> dict:
    """Full-city evaluation of the best checkpoint on held-out test cities."""
    dev = _device()
    model = models.load_model(out_dir / "model.pt", device=dev)

    stats = np.load(out_dir / "band_stats.npy")
    st = {"lo": stats[0], "hi": stats[1]}
    _, test_cities = data.city_split(raw_dir)
    per_city = {}
    for name in test_cities:
        city = data.load_city(raw_dir, name)
        prob = change.predict_full_city(model, city, st, device=dev)
        pred = prob > 0.5
        pm = evaluate.pixel_metrics(city.mask, pred.astype(np.uint8))
        pm.update(evaluate.area_metrics(city.mask, pred.astype(np.uint8)))
        per_city[name] = pm
        print(f"  {name}: IoU(change) {pm['iou_change']:.4f} "
              f"F1 {pm['f1_change']:.4f} pred {pm['pred_ha']:.0f} ha / "
              f"true {pm['true_ha']:.0f} ha", flush=True)
        np.save(out_dir / f"prob_{name}.npy", prob)
    return {"per_city": per_city}


def main() -> None:
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    t0 = time.time()
    summary = train(raw, out)
    test = evaluate_cities(raw, out)
    payload = {"run": summary, "test": test, "minutes": round((time.time() - t0) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    with open(out / "runs.csv", "a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["timestamp", "epochs", "best_val_iou_change",
                        "test_iou_mean", "test_f1_mean", "minutes"])
        ious = [c["iou_change"] for c in test["per_city"].values()]
        f1s = [c["f1_change"] for c in test["per_city"].values()]
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), summary["epochs"],
                    summary["best_val_iou_change"],
                    round(float(np.mean(ious)), 4), round(float(np.mean(f1s)), 4),
                    payload["minutes"]])
    print(json.dumps({k: v for k, v in payload.items() if k != "run"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
