"""U-Net change-detection model and its loss.

Both dates are stacked as 24 input channels (12 bands x 2 dates). Feeding the
registered pair to one U-Net lets the encoder treat "before" and "after" bands
as a single observation, which beats two independent segmentations because the
change itself is what has to be learned. A siamese encoder with an explicit
feature difference is the usual alternative; the stacked model is simpler and
trained at least as well here, so we keep it.

Encoder choice is a measured trade-off, not a preference. On this 4-core CPU
laptop (batch 8, 128x128 tiles, torch.set_num_threads(4)) the cost of one
forward+backward step was benchmarked before settling on a default:

    resnet34       9.64 s/step  -> 17.2 min/epoch -> 2.3 h for 8 epochs
    mobilenet_v2   3.91 s/step  ->  7.0 min/epoch -> 1.6 h for 14 epochs
    efficientnet   6.66 s/step  -> 11.9 min/epoch -> 1.6 h for 8 epochs

ResNet-34 is the strongest encoder here but does not fit the machine's time
budget, so MobileNetV2 is the default: it keeps ImageNet-pretrained weights (a
real advantage on a small change dataset), runs 2.5x faster than ResNet-34 and
still trains the decoder to a usable change map. Passing `encoder_name`
overrides this if you have a GPU.
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# Default encoder, chosen by the benchmark in the module docstring. Exposed as a
# constant so `train` can record it in the checkpoint and `evaluate` can rebuild
# the exact same architecture instead of guessing.
DEFAULT_ENCODER = "mobilenet_v2"


def build_unet(in_channels: int = 24, encoder_name: str = DEFAULT_ENCODER,
               encoder_weights: str = "imagenet") -> nn.Module:
    """U-Net with a pretrained encoder and one sigmoid output.

    `encoder_name` defaults to MobileNetV2 for the CPU cost reason in the module
docstring; use "resnet34" on a GPU. `encoder_weights=None` trains the encoder
from scratch (needed only when no pretrained download is available).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
    # smp.Unet does not expose the encoder it was built with, so record it on
    # the module: `train` stores it in the checkpoint and `evaluate` / any
    # later load can rebuild the exact same architecture instead of guessing.
    # A plain attribute survives ``.to(device)``.
    model.encoder_name = encoder_name
    model.in_channels = in_channels
    return model


def load_model(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    """Rebuild a U-Net from a saved checkpoint and load its weights.

    The encoder and channel count are read back from the checkpoint so an
    evaluation run matches the training run exactly, even if the default
    `DEFAULT_ENCODER` or `config.BANDS` has since changed. Checkpoints written
    before those keys were recorded fall back to the current defaults, so old
    results stay loadable.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_unet(
        in_channels=int(ckpt.get("in_channels", 24)),
        encoder_name=str(ckpt.get("encoder", DEFAULT_ENCODER)),
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class DiceBCELoss(nn.Module):
    """Dice + BCE on sparse change masks.

    Changed pixels are a small minority of every tile (typically 1-5%), so
    plain BCE collapses to predicting "no change" everywhere. Dice measures
    overlap directly and keeps the gradient alive on the minority class.
    """

    def __init__(self, dice_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        # Dice is per-sample (shape B); average it before combining with BCE,
        # otherwise ``forward`` returns a vector and ``loss.backward()`` raises
        # "grad can be implicitly created only for scalar outputs".
        return ((1 - self.dice_weight) * bce + self.dice_weight * (1 - dice)).mean()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum()
    return float((2 * intersection + eps) / (preds.sum() + targets.sum() + eps))
