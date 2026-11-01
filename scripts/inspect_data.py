"""
Quick sanity check for data/processed.npz.

Usage:
    python scripts/inspect_data.py --data data/processed.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from neurosynth.data import EEGDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/inspect_psd.png"))
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f"Not found: {args.data}. Run `python scripts/build_dataset.py` first.")

    ds = EEGDataset.load(args.data)
    X, y, sfreq = ds.X, ds.y, ds.sfreq

    print(f"X shape        : {X.shape}  (epochs, channels, timepoints)")
    print(f"y shape        : {y.shape}")
    print(f"sfreq          : {sfreq} Hz")
    print(f"n_channels     : {len(ds.ch_names)}")
    print(f"Duration/epoch : {X.shape[2] / sfreq:.2f} s")
    print(f"Unique subjects: {len(set(ds.subject_ids.tolist()))}")
    print(f"Class balance  : {np.bincount(y)}  (class 0 / class 1)")
    print(f"X dtype        : {X.dtype}, min={X.min():.3f}, max={X.max():.3f}")

    # PSD of channel Fz (index 0) per class, averaged over epochs
    ch_idx = 0
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls in [0, 1]:
        epochs_cls = X[y == cls, ch_idx, :]  # (n_cls_epochs, T)
        psds = []
        for epoch in epochs_cls:
            f, psd = welch(epoch, fs=sfreq, nperseg=256)
            psds.append(psd)
        mean_psd = np.mean(psds, axis=0)
        ax.semilogy(f, mean_psd, label=f"class {cls} (n={len(epochs_cls)})")

    ax.set_xlim(0, 50)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"Mean PSD — channel {ds.ch_names[ch_idx]}")
    ax.legend()
    ax.axvspan(7, 30, alpha=0.1, color="green", label="mu/beta band")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPSD plot saved to {args.out}")


if __name__ == "__main__":
    main()
