# Results figures

Every PNG in this folder is written by `src/forest_change/figures.py` from the
run's own artefacts — `metrics.json`, `baseline_metrics.json`, the saved
`prob_<city>.npy` prediction maps and `data/raw`. No figure re-runs training or
inference, so the pictures and the numbers in the README cannot drift apart.

Regenerate with:

```
python -m forest_change.figures data/raw results
```

## Quantitative

| File | Contents |
| --- | --- |
| `training_curves.png` | Training loss, validation IoU/F1 and precision/recall over the 8 epochs, with the best-val-IoU checkpoint marked. |
| `per_city_metrics.png` | IoU and F1 for the change class on each of the 10 held-out cities, U-Net against the NDVI-loss baseline, with the per-city mean as a dotted line. |
| `area_error.png` | Ground-truth, U-Net and baseline cleared area per city in hectares, plus the absolute area error. |
| `baseline_threshold.png` | The NDVI baseline's forest threshold swept over a grid and tuned on the train cities only; the value used for the reported test numbers is marked. |

## Qualitative

`qual_<city>.png` is a six-panel figure for one held-out city: the two
Sentinel-2 dates as true-colour RGB, the ground-truth change mask, the U-Net
prediction, the NDVI-loss baseline prediction, and an error map where green is
a hit, red a false alarm and blue a missed clearing.

`change_map_<city>.png` overlays the U-Net prediction on the after image, which
is the clearest view of where the model puts cleared forest.

Cities shown: `montpellier`, `lasvegas` and `valencia` — a large and a small
clearing, and the city where the model fails. Any city can be added by editing
`CITIES_FOR_QUALITATIVE` in `figures.py`.

## Displaying the rasters

OSCD bands arrive as 16-bit reflectance scaled by 1e4, so `visualize.truecolor`
divides by that scale before brightening; without it every pixel clips to white
and the RGB panels come out blank. The composite is B04/B03/B02 as R/G/B.
