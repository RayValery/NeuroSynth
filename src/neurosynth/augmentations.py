"""
Standard EEG augmentation transforms for the baseline comparison (spec RR-2).

All transforms accept and return a tensor of shape (C, T) — one epoch.
They are applied at training time only (val/test loaders use no augmentor).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_noise(x: Tensor, std: float = 0.1) -> Tensor:
    """Add zero-mean Gaussian noise scaled to 10% of the epoch's std."""
    return x + torch.randn_like(x) * std * x.std()


def time_shift(x: Tensor, max_shift: int = 50) -> Tensor:
    """Roll the signal along the time axis by a random amount up to max_shift samples."""
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shift, dims=-1)


def channel_dropout(x: Tensor, p: float = 0.1) -> Tensor:
    """Zero out each channel independently with probability p."""
    mask = (torch.rand(x.shape[0], 1, device=x.device) > p).float()
    return x * mask


def sliding_window_crop(x: Tensor, crop_frac: float = 0.8) -> Tensor:
    """
    Crop a random contiguous time window of length floor(T * crop_frac),
    then pad back to T with reflection so the shape is preserved.
    """
    T = x.shape[-1]
    crop_len = int(T * crop_frac)
    start = int(torch.randint(0, T - crop_len + 1, (1,)).item())
    cropped = x[:, start : start + crop_len]
    return F.pad(cropped.unsqueeze(0), (0, T - crop_len), mode="reflect").squeeze(0)


class Augmentor:
    """Randomly applies each transform in the list with probability p_apply."""

    def __init__(self, transforms: list, p_apply: float = 0.5):
        self.transforms = transforms
        self.p_apply = p_apply

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            if torch.rand(1).item() < self.p_apply:
                x = t(x)
        return x


STANDARD_AUGMENTOR = Augmentor(
    [gaussian_noise, time_shift, channel_dropout, sliding_window_crop],
    p_apply=0.5,
)
