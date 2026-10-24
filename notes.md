# Deforestation Detection - working notes

Chronological log of decisions, data issues and results.

## Dataset decision (corrected)

**OSCD (Onera Satellite Change Detection) is the final training source, and it
does ship the train labels.** An earlier version of this note claimed the
official train labels were never released; that was wrong. `data/raw` contains
three archives that match `src/forest_change/data.py`'s `ARCHIVE_ROOTS`:

- `Onera Satellite Change Detection dataset - Images` - the 24 registered
  Sentinel-2 scene pairs (~700 MB, 12 usable bands per date).
- `... - Train Labels` - `<CITY>-cm.tif` change rasters for the 14 train cities.
- `... - Test Labels` - the same for the 10 test cities.

So a supervised U-Net trains on OSCD directly, which is what the pipeline does:
14 train cities, 4 of them held back for validation, 10 test cities scored once.

## Data quirks found in the shipped files

- **Mask encoding.** The archive README documents 0 = no change, 1 = change,
  but the shipped GeoTIFFs contain only the values 1 and 2 and never a zero.
  Reading them with the documented `> 0` rule would mark every pixel as change.
  `data.decode_mask` tells the two encodings apart by value range (only-{1,2}
  means 2 = change), so both the shipped file and a repackaged 0/1 copy decode
  correctly.
- **Band count.** OSCD advertises 13 bands; B10 (the atmospheric window) is
  absent, so `config.BANDS` holds 12. The model input is 24 channels: 12 bands
  x 2 dates.
- **Empty tiles.** Some tiles are open water or no-data with zero mask pixels.
  They carry no forest signal and only dilute the loss, so `TileDataset` drops
  raw-uint16 tiles whose mean reflectance is below `MIN_TILE_BRIGHTNESS` and
  whose mask is empty.
- **Memory.** A city pair is ~90 MB of uint16 per date. All 14 train cities in
  memory at once is fine here, but scaling a float32 copy of every city would
  not be, so tiles are scaled on demand inside `__getitem__`.
- **Checkpoint fidelity (Onera/OSCD run, 2026-10-24).** `segmentation_models_pytorch`
  0.5.0 does not expose which encoder a decoder was built with, so the saved
  checkpoint could not be paired back to its architecture. `models.build_unet`
  now records both `encoder_name` and `in_channels` on the module, `train`
  writes them into `results/model.pt`, and `models.load_model` rebuilds the
  exact architecture instead of falling back to the current default. Verified
  with a round-trip check: saving a model, reloading it and comparing
  `state_dict`s parameter-by-parameter gives an exact match, and a checkpoint
  written with `resnet34` is rebuilt as `resnet34` rather than the default
  encoder. Older checkpoints that predate those keys still load, because
  `load_model` falls back to `DEFAULT_ENCODER` / 24 channels when they are
  absent.

## Model choice (measured, not assumed)

Cost of one forward+backward step on this laptop (4 threads, batch 8,
128x128 tiles), measured before the default was picked:

| encoder | s/step | min/epoch | 8 epochs |
| --- | --- | --- | --- |
| resnet34 | 9.64 | 17.2 | 2.3 h |
| mobilenet_v2 | 3.91 | 7.0 | 0.93 h |
| efficientnet-b0 | 6.66 | 11.9 | 1.58 h |

ResNet-34 is the stronger encoder but does not fit the machine's time budget,
so MobileNetV2 with ImageNet weights is the default and the trade-off is stated
in `models.py`. Change `build_unet(encoder_name=...)` on a GPU.

## Honesty rules

- Cities are split, never tiles: neighbouring tiles are near-identical, so a
  random split inflates every score.
- The NDVI baseline's threshold is tuned on the train cities only; the test
  cities are scored once with the tuned value.
- The baseline is scored on exactly the same test cities and the same metric
  definitions as the U-Net.
- Reported: IoU / F1 / precision / recall for the change class, plus hectares
  of true and predicted area per city.

## Process lesson (why training used to die here)

Training repeatedly died with no traceback, always mid-load. The cause was not
torch or the data: every `start /b` / plain `Start-Process` invocation either
failed to launch (bad quoting swallowed the command) or was reaped when the
tool call that spawned it returned. What works on this host is PowerShell
`Start-Process -PassThru` with **both** stdout and stderr redirected to files,
which detaches the run from the tool session. That is what `train_runner.bat`
now does, and `load_cities` prints per-city timings so a stall is visible in
the log rather than silent.
