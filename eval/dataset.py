from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGTorchDataset(Dataset):
    """Wraps (X, y) arrays as a PyTorch Dataset. X: float32 (N, C, T), y: int64 (N,)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augmentor=None):
        self.X = torch.from_numpy(X)  # (N, C, T) float32
        self.y = torch.from_numpy(y)  # (N,) int64
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        if self.augmentor is not None:
            x = self.augmentor(x)
        return x, self.y[idx]


def subject_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[tuple, tuple, tuple]:
    """
    Assign whole subjects to train/val/test — no subject appears in more than one split.
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(set(subject_ids.tolist())))

    if len(subjects) < 4:
        raise ValueError(
            f"Only {len(subjects)} subject(s) found — need at least 4 for a "
            "train/val/test split. Re-run the pipeline with more subjects "
            "(e.g. --subjects $(seq 1 109))."
        )

    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test

    train_subjs = set(subjects[:n_train].tolist())
    val_subjs = set(subjects[n_train : n_train + n_val].tolist())
    test_subjs = set(subjects[n_train + n_val :].tolist())

    def _select(subj_set: set) -> tuple[np.ndarray, np.ndarray]:
        idx = np.where(np.isin(subject_ids, list(subj_set)))[0]
        return X[idx], y[idx]

    (X_train, y_train) = _select(train_subjs)
    (X_val, y_val) = _select(val_subjs)
    (X_test, y_test) = _select(test_subjs)

    print(
        f"Split — train: {len(train_subjs)} subjects / {len(y_train)} epochs | "
        f"val: {len(val_subjs)} subjects / {len(y_val)} epochs | "
        f"test: {len(test_subjs)} subjects / {len(y_test)} epochs"
    )
    print(f"  Train class balance : {np.bincount(y_train)}")
    print(f"  Val   class balance : {np.bincount(y_val)}")
    print(f"  Test  class balance : {np.bincount(y_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def within_subject_loro_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    run_ids: np.ndarray,
) -> list[dict]:
    """
    Leave-One-Run-Out (LORO) splits for within-subject EEGNet evaluation.

    For each (subject, held-out-run) pair, returns train epochs from the
    remaining runs and test epochs from the held-out run.

    Returns a list of dicts, ordered by (subject, test_run):
        {
            "subject":  int,
            "test_run": int,
            "X_train":  ndarray (n_train, C, T),
            "y_train":  ndarray (n_train,),
            "X_test":   ndarray (n_test,  C, T),
            "y_test":   ndarray (n_test,),
        }
    """
    splits = []
    for subject in np.unique(subject_ids):
        subj_mask = subject_ids == subject
        X_subj = X[subj_mask]
        y_subj = y[subj_mask]
        runs_subj = run_ids[subj_mask]

        for test_run in np.unique(runs_subj):
            test_mask = runs_subj == test_run
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            splits.append({
                "subject":  int(subject),
                "test_run": int(test_run),
                "X_train":  X_subj[train_mask],
                "y_train":  y_subj[train_mask],
                "X_test":   X_subj[test_mask],
                "y_test":   y_subj[test_mask],
            })
    return splits
