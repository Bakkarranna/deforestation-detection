# Forest Change Detection (Sentinel-2 / OSCD)

End-to-end pipeline for detecting **deforestation and land-cover change** from
multi-date Sentinel-2 imagery, built on the [OSCD (Onera Satellite Change
Detection)](https://rcapek.github.io/oscd-dataset/) benchmark. It downloads and
extracts the dataset, normalises and tiles it, trains a U-Net change
segmentation model, computes an NDVI-loss baseline for comparison, and evaluates
**per-city** on the official held-out test split with pixel metrics and
hectare-scale area accounting.

![Per-city IoU and F1, U-Net vs NDVI baseline](results/figures/per_city_metrics.png)

## Results

Mean over the 10 held-out test cities (full-city inference, best checkpoint by
validation IoU):

| Model | IoU (change) | F1 (change) | Precision | Recall |
| --- | --- | --- | --- | --- |
| U-Net (mobilenet_v2, 24 ch, 8 epochs) | **0.2586** | **0.3862** | 0.3928 | 0.4642 |
| NDVI-loss baseline | 0.0793 | 0.1345 | - | - |

The full table with per-city breakdown and area accounting lives in
[results/results_table.md](results/results_table.md).

### What the numbers say

- The U-Net wins on **mean** IoU (0.26 vs 0.08) and F1 (0.39 vs 0.13), and is
  best on the cities with subtle or fragmented change: `montpellier` 0.51,
  `lasvegas` 0.51, `chongqing` 0.32.
- The baseline wins only where the change is a **pure, large-scale greenness
  drop**: on `brasilia` it reaches 0.30 IoU, largely because it puts almost all
  of its predictions where the U-Net over-predicts area.
- `montpellier` — sparse ground truth and clean imagery — produces sensible,
  well-localised maps:

![Predicted change map, montpellier](results/figures/change_map_montpellier.png)

![Qualitative prediction montpellier — green hit / red false alarm / blue miss](results/figures/qual_montpellier.png)

- Pixel scores and hectare accounting disagree honestly. The model
  over-predicts hectares on `valencia` (114 ha vs 9.7 ha true) and `saclay_w`
  (211 ha vs 50 ha), under-predicts on `norcia` (23.9 ha vs 24.2 ha with only
  0.0998 IoU — it finds the right area in the wrong places), and on bare
  `lasvegas`/`dubai` it drifts toward labelling all non-forest land as change.

![Predicted vs true cleared area, and absolute area error](results/figures/area_error.png)

- The NDVI threshold used by the baseline was tuned on the 14 labelled train
  cities and then frozen for the held-out test evaluation, so the test numbers
  above are uncontaminated:

![Baseline threshold tuning](results/figures/baseline_threshold.png)

- Validation converges to 0.2857 IoU (change) by epoch 08, with the best
  checkpoint at epoch 05 (0.3187) — the three extra epochs overfit slightly:

![Validation curves](results/figures/training_curves.png)

Analysis of the failure modes is in [notes.md](notes.md).

## Pipeline

```
forest_change/
  config.py         paths, dataset constants, hyperparameters
  download.py       resumable parallel download + extraction of the OSCD archives
  data.py           City loading, mask decoding, tiling, train/val/test split
  preprocess.py     NDVI, cloud/shadow masking, band normalisation
  models.py         U-Net (smp) with a recorded encoder + DiceBCELoss
  train.py          training loop, checkpointing, per-city evaluation
  evaluate.py       pixel metrics (IoU, F1, precision, recall) and area accounting
  baseline.py       NDVI-loss baseline used as a reference model
  change.py         full-city change maps and the connected-component speckle filter
  visualize.py      true-colour composites and cleared-forest overlay maps
  figures.py        all result figures
  results_table.py  markdown + CSV + JSON results table generation
```

Two decisions drive the result:

- **City-level split** — tiles never cross cities, so neighbouring tiles of a
  test city can't leak into training.
- **Dice + BCE loss** — changed pixels are 1-5% of every tile, so plain BCE
  collapses to "no change everywhere". Dice (0.5 weight) keeps the gradient
  alive on the minority class.

## Repository layout

```
src/forest_change/   pipeline source
tests/               pure numpy/scipy unit tests (no torch, no data needed)
results/             metrics, tables, logs and the figures above
data/                dataset, downloaded on demand (git-ignored)
.github/workflows/   CI: install, ruff lint, pytest
```

## Getting started

Requires **Python 3.10+**.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"         # numpy, scipy, rasterio, torch, smp, sklearn, matplotlib, pandas
```

### Download the data

The ~38 GB OSCD archive is fetched from Hugging Face in 16 parallel 8 MB chunks
into `data/raw/`, and resumes from `.part` files if interrupted:

```bash
python -m forest_change.download data/raw
```

Each interrupted chunk is refetched with retries, and the downloader reports any
archive it could not complete rather than letting you train on truncated data.

## Training and evaluation

Train and evaluate in one command — it writes `results/metrics.json`, the
per-city probability maps, and appends a row per run to `results/runs.csv`:

```bash
python -m forest_change.train                        # data/raw -> results/
python -m forest_change.train data/raw results       # explicit input/output dirs
python -m forest_change.baseline                     # NDVI-loss baseline + threshold tuning
```

Hyperparameters (epochs, batch size, learning rate) live in
[src/forest_change/config.py](src/forest_change/config.py); `train()` also takes
`epochs` / `batch_size` / `lr` / `quick` arguments for notebook use.

Artifacts per run: `results/model.pt` (best val-IoU checkpoint, with the encoder
name and channel count recorded so it reloads exactly), `results/band_stats.npy`
(normalisation constants), and one `results/prob_<city>.npy` prediction map per
test city.

A full 8-epoch run takes ~108 min on a 4-thread CPU with batch 8 and 128×128
tiles; the measured per-step cost of MobileNetV2 vs ResNet-34 is documented in
[src/forest_change/config.py](src/forest_change/config.py).

## Reproducing the tables and figures

```bash
python -m forest_change.results_table            # results_table.md, .csv and .json
python -m forest_change.figures data/raw results  # needs data/raw and the prob_*.npy maps
```

The tables are regenerated from `results/metrics.json` plus
`results/baseline_metrics.json` alone. The qualitative figures need the raw data
and the `prob_*.npy` probability maps that `train` writes — the per-city charts
are read from those maps rather than re-running the model, so they show exactly
the prediction the metrics were computed from.

## Tests

```bash
pytest                # 9 unit tests, pure numpy/scipy
ruff check src tests
```

CI runs both on every push and pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Implementation notes

- **Dataset quirk**: the shipped OSCD change masks are encoded `1/2`, never
  `0/1`, so a naive `> 0` read marks every pixel as changed.
  [data.decode_mask](src/forest_change/data.py) detects the encoding by value
  range and still decodes a repackaged `0/1` copy correctly.
- **No cloud mask**: OSCD ships only the 12 L2 spectral bands, with no SCL or
  QA60 scene-classification band, so no explicit cloud layer can be built. Thin
  cloud edges are instead handled by the speckle filter below; residual cloud
  artefacts are one of the known failure modes noted in [notes.md](notes.md).
- **Speckle filter**: connected components below 50 px (0.05 ha at 10 m) are
  dropped — co-registration noise and thin cloud edges are not clearings.
- **Baseline tuning**: the NDVI forest threshold is tuned on train cities only
  and frozen for the test evaluation.
