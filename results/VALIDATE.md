# Validation run: mask decoding fix

Date: 2025-10-23 (this run regenerates everything after the fix).

## Bug found

`data.read_mask` decoded the OSCD change mask as `values > 0`.

The OSCD README says the `<city>-cm.tif` raster is "0 means no change and 1
means change", but the shipped files are **1 = no change, 2 = change**. There
are no zeros anywhere in any city. Reading with `> 0` therefore marked *every*
pixel as changed:

- bercy: 142 200/142 200 pixels "changed"
- nantes: 303 804/303 804
- beirut: 1 262 600/1 262 600

Cross-checked against `cm/cm.png`, where the README's 0 / 255 encoding holds:
every tif value 1 aligns exactly with png 0 (no change) and every tif value 2
aligns exactly with png 255 (change), for all eight cities inspected.

## What this invalidated

1. `results/baseline_metrics.json` (the NDVI-loss baseline scored against an
   all-ones ground truth, so its IoU/F1 were meaningless).
2. The interrupted U-Net run, whose log showed
   `epoch 01 ... val IoU(change) 1.0000` - the model had simply learned to
   predict "change everywhere", which is trivially perfect against an
   all-ones mask.

Both are regenerated from scratch in this run. No earlier number is carried
forward.

## Fix

`read_mask` now detects the encoding: if the raster's unique values are a
subset of {1, 2} and its minimum is 1, changed = (values == 2); otherwise the
documented `> 0` path is used, so a repackaged 0/1 copy still decodes
correctly. `tests/test_forest_change.py` asserts the change fraction of the
decoded masks stays in the 0.5%-5% band that OSCD's real change density
produces, which would have caught this bug.

## Sanity band

Decoded change fractions (2 / (1 + 2)) per city, from a direct read of the
raw rasters:
