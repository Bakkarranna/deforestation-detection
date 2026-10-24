"""Tile-level Dataset for training and evaluating the change model."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .preprocess import scale


class TileDataset(Dataset):
    """Fixed-size tiles cut from registered OSCD image pairs.

    Cities stay in memory as raw uint16 stacks (~90 MB per city per date) and
    every tile is scaled on demand in ``__getitem__``. Materialising one float32
    copy of each city would need 4-5 GB of RAM for the fourteen training cities
    on this 8 GB laptop; scaling a single 128x128x12 tile costs a fraction of a
    training step instead, and nothing else ever touches the whole stack.
    """

    def __init__(self, cities, stats: dict, tile: int = config.TILE,
                 stride: int = config.STRIDE, augment: bool = False):
        self.cities = list(cities)
        self.stats = stats
        self.tile = tile
        self.augment = augment
        # Index only: (city position, row, col). No pixel data is held here.
        self.items: list[tuple[int, int, int]] = []
        for ci, city in enumerate(self.cities):
            for row, col in city.tiles(tile, stride):
                m_t = city.mask[row:row + tile, col:col + tile]
                pre_t = city.pre[row:row + tile, col:col + tile]
                # Drop tiles with almost no signal at all (open water and
                # no-data): they only dilute the loss. Raw uint16 brightness
                # is a good enough emptiness test for this one filter.
                if int(m_t.sum()) == 0 and float(pre_t.mean()) < config.MIN_TILE_BRIGHTNESS:
                    continue
                self.items.append((ci, row, col))

    def __len__(self) -> int:
        return len(self.items)

    def tile_arrays(self, idx: int):
        """Raw ``(pre, post, mask)`` arrays of one tile index."""
        ci, row, col = self.items[idx]
        city = self.cities[ci]
        t = self.tile
        return (city.pre[row:row + t, col:col + t],
                city.post[row:row + t, col:col + t],
                city.mask[row:row + t, col:col + t])

    def __getitem__(self, idx: int):
        pre, post, mask = self.tile_arrays(idx)
        pre = scale(pre, self.stats)
        post = scale(post, self.stats)
        mask = mask.astype(np.float32)
        if self.augment:
            # Flips and 90-degree rotations are label-preserving for both the
            # scenes and the mask, so this is safe augmentation for change
            # detection.
            k = int(torch.randint(0, 4, (1,)))
            flip = bool(torch.randint(0, 2, (1,)))
            pre = np.rot90(pre, k)
            post = np.rot90(post, k)
            mask = np.rot90(mask, k)
            if flip:
                pre, post, mask = pre[:, ::-1], post[:, ::-1], mask[:, ::-1]
            pre, post, mask = (np.ascontiguousarray(a) for a in (pre, post, mask))
        x = np.concatenate([pre, post], axis=-1).transpose(2, 0, 1)
        return torch.from_numpy(x.copy()), torch.from_numpy(mask[None].copy())


def make_loader(dataset: TileDataset, batch_size: int, shuffle: bool,
                workers: int = 0) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=False, drop_last=False)
