"""Sentinel-2 deforestation and change detection on the OSCD benchmark.

Pipeline: download OSCD archives -> read the 13-band registered Sentinel-2
pairs and pixel change masks -> tile by city -> train a two-stream U-Net
(siamese encoder) -> evaluate IoU/F1 per city -> map newly cleared forest.

OSCD is a registered change-detection benchmark: every city contributes one
image pair, so the train/test split is over cities, never over tiles.
"""

__version__ = "0.1.0"
