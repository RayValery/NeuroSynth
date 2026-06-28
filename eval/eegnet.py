"""
EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces.
Lawhern et al. 2018, DOI 10.1088/1741-2552/aace8c

Input shape: (batch, 1, n_channels, n_times)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018). Three-block depthwise separable CNN for EEG.

    Args:
        n_channels: Number of EEG channels (spatial dim). Default 64.
        n_times:    Number of time samples per epoch. Default 801 (160 Hz, 5 s).
        n_classes:  Output classes. Default 2.
        F1:         Number of temporal filters. Default 8.
        D:          Depth multiplier for spatial filters. Default 2.
        F2:         Number of pointwise filters (= F1 * D). Default 16.
        dropout:    Dropout probability. Default 0.5.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 801,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Block 1 — temporal filter across time
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — depthwise spatial filter across channels
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3 — separable convolution (depthwise + pointwise)
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened size via a dummy forward pass — robust to any n_times.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_size = self.block3(self.block2(self.block1(dummy))).numel()

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.flatten(1))

    def apply_weight_constraints(self, max_norm_conv: float = 1.0, max_norm_fc: float = 0.25) -> None:
        """Clip conv and linear weights to max_norm after each optimizer step (per paper)."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    norm = m.weight.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                    m.weight.data *= (norm.clamp(max=max_norm_conv) / norm.clamp(min=1e-8))
                elif isinstance(m, nn.Linear):
                    norm = m.weight.norm(p=2, dim=1, keepdim=True)
                    m.weight.data *= (norm.clamp(max=max_norm_fc) / norm.clamp(min=1e-8))

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"EEGNet(params={n_params:,})"
