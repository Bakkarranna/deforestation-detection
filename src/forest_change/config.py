"""Paths, dataset constants and hyperparameters for the OSCD change pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# The 13 Sentinel-2 L2A bands stored in the OSCD `imgs_*_rect` folders, in the
# order the dataset uses. B8A is the narrow NIR band (20 m, resampled to 10 m).
# OSCD ships 12 of them per scene (B10 is absent, atmospheric window omitted).
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
         "B09", "B11", "B12"]
BAND_INDICES = {"B02": 1, "B03": 2, "B04": 3, "B08": 7, "B8A": 8, "B11": 10, "B12": 11}

# Official OSCD split: 14 cities with labels for training, 10 held out for test.
# Splitting by city (not by tile) keeps neighbouring tiles out of the test set.
TRAIN_CITIES = [
    "aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes",
    "saclay_e", "abudhabi", "cupertino", "pisa", "beihai", "hongkong",
    "beirut", "mumbai",
]
TEST_CITIES = [
    "brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia",
    "dubai", "lasvegas", "milano", "chongqing",
]
# Held-out block of the train cities, used only for model selection (early
# stopping and best-checkpoint). The official test cities stay untouched.
VAL_CITIES = ["bercy", "nantes", "rennes", "beirut"]

TILE = 128  # tile size in pixels (1.28 km at 10 m)
STRIDE = 64  # overlapping tiles give more training samples per city
# A tile with a mean raw reflectance below this and no change pixels carries no
# signal (open water / no-data) and is dropped from the training set.
MIN_TILE_BRIGHTNESS = 200.0
SEED = 42
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
# Measured on this machine (4 threads, batch 8, 128x128 tiles, 107 steps/epoch
# over the 10 fit cities): one forward+backward step costs 3.91 s with the
# default MobileNetV2 encoder and 9.64 s with ResNet-34. A full 8-epoch run is
# therefore about 56 min, which is the budget that fits the laptop while still
# letting ReduceLROnPlateau and the best-val-IoU checkpoint do their job.
# Epochs are cheap to change in a notebook; the cost is minutes, not hours.
THREADS = 4


@dataclass(frozen=True)
class Sample:
    """One tile: the two dates' band stacks and the binary change mask."""

    pre: object
    post: object
    mask: object
    city: str
    row: int
    col: int
