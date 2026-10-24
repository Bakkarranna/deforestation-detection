"""Read OSCD cities into memory and cut them into fixed-size tiles.

OSCD stores, per city, two rectified Sentinel-2 scenes (`imgs_1_rect`,
`imgs_2_rect`) as one GeoTIFF per band, plus a pixel-level change mask under
`cm/<city>-cm.tif`. Bands are read in `config.BANDS` order into a single
(H, W, 12) uint16 stack; percentiles computed on the training tiles are used to
scale every band to a comparable range.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from . import config

IMG_DIRS = {1: "imgs_1_rect", 2: "imgs_2_rect"}
MASK_DIR = "cm"

# Root folder names differ slightly from the archive-internal top level.
ARCHIVE_ROOTS = [
    "Onera Satellite Change Detection dataset - Images",
    "Onera Satellite Change Detection dataset - Train Labels",
    "Onera Satellite Change Detection dataset - Test Labels",
]


def _city_dirs(raw_dir: Path) -> dict[str, dict[int, Path]]:
    """Map city name -> date -> image folder, for every city we can find."""
    out: dict[str, dict[int, Path]] = {}
    for root in ARCHIVE_ROOTS:
        base = raw_dir / root
        if not base.exists():
            continue
        for city_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            dates = {}
            for date, name in IMG_DIRS.items():
                d = city_dir / name
                if d.is_dir():
                    dates[date] = d
            if len(dates) == 2:
                out.setdefault(city_dir.name, dates)
    return out


def city_split(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Train and test city lists, intersected with what is on disk."""
    cities = _city_dirs(raw_dir)
    train = [c for c in config.TRAIN_CITIES if c in cities]
    test = [c for c in config.TEST_CITIES if c in cities]
    return train, test


def read_stack(img_dir: Path) -> np.ndarray:
    """Read the twelve bands of one date as (H, W, 12) uint16."""
    bands = []
    for name in config.BANDS:
        path = img_dir / f"{name}.tif"
        with rasterio.open(path) as src:
            bands.append(src.read(1))
    return np.stack(bands, axis=-1)


def decode_mask(raw: np.ndarray) -> np.ndarray:
    """Decode an OSCD change-mask raster into 1 = changed, 0 = unchanged.

    The archive README says the raster is 0/1, but the shipped GeoTIFFs are
    1/2 - and never contain a zero, so a `> 0` read marks the whole city as
    changed. The two encodings are told apart by value range: a raster whose
    only values are 1 and 2 uses the 1/2 encoding, anything else falls back to
    the documented `> 0` rule so a repackaged 0/1 copy still decodes.
    """
    values = set(np.unique(raw).tolist())
    if values and values <= {1, 2}:
        return (raw == 2).astype(np.uint8)
    return (raw > 0).astype(np.uint8)


def read_mask(raw_dir: Path, city: str) -> np.ndarray:
    """Read <city>-cm.tif into a (H, W) uint8 mask of changed pixels."""
    for root in ARCHIVE_ROOTS:
        candidate = raw_dir / root / city / MASK_DIR / f"{city}-cm.tif"
        if candidate.exists():
            with rasterio.open(candidate) as src:
                return decode_mask(src.read(1))
    raise FileNotFoundError(f"change mask missing for city {city}")


@dataclass
class City:
    """A city's two registered scenes and its change mask."""

    name: str
    pre: np.ndarray  # (H, W, 12) uint16
    post: np.ndarray  # (H, W, 12) uint16
    mask: np.ndarray  # (H, W) uint8

    def tiles(self, tile: int = config.TILE, stride: int = config.STRIDE):
        """Yield (row, col) top-left corners of non-overlapping tiles."""
        h, w = self.mask.shape
        for row in range(0, h - tile + 1, stride):
            for col in range(0, w - tile + 1, stride):
                yield row, col


def load_city(raw_dir: Path, city: str) -> City:
    dirs = _city_dirs(raw_dir)[city]
    pre = read_stack(dirs[1])
    post = read_stack(dirs[2])
    mask = read_mask(raw_dir, city)
    if pre.shape[:2] != mask.shape:
        raise ValueError(f"{city}: scene {pre.shape[:2]} does not match mask {mask.shape}")
    return City(name=city, pre=pre, post=post, mask=mask)


def load_cities(raw_dir: Path, names: list[str]) -> list[City]:
    """Load each city, reporting progress: GeoTIFF reads are the slow part."""
    out = []
    for i, n in enumerate(names, 1):
        t0 = time.time()
        out.append(load_city(raw_dir, n))
        print(f"  loaded {n} ({i}/{len(names)}) in {time.time() - t0:.0f}s",
              flush=True)
    return out
