"""
EEGBCI (PhysioNet Motor Movement/Imagery) data pipeline.

Covers spec Section 7 / First Steps #2:
    "Load EEGBCI via MNE; build preprocessing + epoching pipeline."

Dataset notes (PhysioNet EEGBCI, 109 subjects, 64ch, 160Hz):
  Each subject has 14 runs. We use the *motor imagery* runs only:
    - Runs 4, 8, 12:  imagine left fist (T1) vs. right fist (T2)
    - Runs 6, 10, 14: imagine both fists (T1) vs. both feet (T2)
  T0 in all runs = rest, and is dropped here (binary task discrimination
  is the standard first benchmark; rest can be added back later).

Pipeline: download -> load raw -> set montage -> band-pass filter
  (mu/beta band, 7-30 Hz, standard for motor imagery) -> epoch per run
  -> per-epoch z-score normalization -> save as .npz
  (X, y, subject_ids, run_ids, ch_names, sfreq).

Usage:
    python eegbci_pipeline.py --subjects $(seq 1 109) --task imagery_lr --out data/processed.npz
    python eegbci_pipeline.py --subjects 1 2 3 --task imagery_ff --out ./processed.npz
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MNE is imported lazily inside functions so this module can be imported
# (e.g. for the dataclasses/constants) without requiring MNE to be present.

# Run groups per task type (see PhysioNet EEGBCI documentation).
RUNS_LEFT_RIGHT_FIST = [4, 8, 12]
RUNS_FISTS_FEET = [6, 10, 14]

TASK_RUNS = {
    "imagery_lr": RUNS_LEFT_RIGHT_FIST,   # left fist vs right fist
    "imagery_ff": RUNS_FISTS_FEET,        # both fists vs both feet
}

# Standard motor-imagery filter band (mu + beta rhythms).
FILTER_LOW_HZ = 7.0
FILTER_HIGH_HZ = 30.0

# Epoch window relative to event onset (seconds).
EPOCH_TMIN = -1.0
EPOCH_TMAX = 4.0


@dataclass
class EEGDataset:
    X: np.ndarray            # (n_epochs, n_channels, n_times)
    y: np.ndarray            # (n_epochs,) int labels, 0/1
    subject_ids: np.ndarray  # (n_epochs,) which subject each epoch came from
    run_ids: np.ndarray      # (n_epochs,) which run each epoch came from
    ch_names: list[str]
    sfreq: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            subject_ids=self.subject_ids,
            run_ids=self.run_ids,
            ch_names=np.array(self.ch_names),
            sfreq=self.sfreq,
        )

    @classmethod
    def load(cls, path: Path) -> "EEGDataset":
        data = np.load(path, allow_pickle=False)
        return cls(
            X=data["X"],
            y=data["y"],
            subject_ids=data["subject_ids"],
            run_ids=data["run_ids"],
            ch_names=list(data["ch_names"]),
            sfreq=float(data["sfreq"]),
        )


def download_subject_runs(subject: int, runs: list[int]):
    """Download (if needed) and return MNE Raw objects for one subject's runs."""
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    edf_paths = eegbci.load_data(subject, runs, update_path=True)
    raws = [read_raw_edf(p, preload=True, verbose="ERROR") for p in edf_paths]
    raw = concatenate_raws(raws)

    # EEGBCI channel names have trailing dots/inconsistent case; standardize
    # so we can map onto the standard 10-05 montage.
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")
    return raw


def preprocess_raw(raw):
    """Band-pass filter to the mu/beta motor-imagery band."""
    raw = raw.copy()
    raw.filter(FILTER_LOW_HZ, FILTER_HIGH_HZ, fir_design="firwin", verbose="ERROR")
    return raw


def epoch_subject(raw, subject: int) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Extract T1/T2 epochs as a binary classification problem."""
    import mne

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # event_id is typically {'T0': 1, 'T1': 2, 'T2': 3}; keep only T1/T2.
    wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if not wanted:
        raise RuntimeError(f"Subject {subject}: no T1/T2 annotations found")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    # Map T1 -> 0, T2 -> 1 regardless of the underlying numeric codes.
    label_map = {wanted["T1"]: 0, wanted["T2"]: 1}
    y = np.array([label_map[code] for code in epochs.events[:, 2]])

    return X, y, epochs.ch_names, float(epochs.info["sfreq"])


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment (He & Wu, 2019, DOI: 10.1109/TNNLS.2019.2913946).

    Whitens one subject's epochs by their mean spatial covariance so that
    inter-subject spatial variability is removed before cross-subject training.

    NOTE: Not used in the standard within-subject pipeline (V05+). Kept for
    future cross-subject experiments. If re-enabled, apply AFTER z-score
    normalization — applying before causes z-score to re-introduce the
    correlations EA removed (verified empirically in V04).

    Args:
        X: (n_epochs, n_channels, n_times) float32 — already z-scored epochs
    Returns:
        X_aligned: same shape, spatially whitened
    """
    n, C, T = X.shape
    cov = np.mean([x @ x.T / T for x in X], axis=0)  # (C, C)
    vals, vecs = np.linalg.eigh(cov + 1e-6 * np.eye(C))
    R_inv_sqrt = (vecs * (vals ** -0.5)) @ vecs.T
    return np.einsum("ij,kjt->kit", R_inv_sqrt, X)


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel z-score normalization."""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def build_dataset(subjects: list[int], task: str) -> EEGDataset:
    if task not in TASK_RUNS:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(TASK_RUNS)}")
    runs = TASK_RUNS[task]

    all_X, all_y, all_subj, all_run = [], [], [], []
    ch_names_ref: list[str] | None = None
    sfreq_ref: float | None = None

    for subject in subjects:
        for run in runs:
            raw = download_subject_runs(subject, [run])
            raw = preprocess_raw(raw)
            try:
                X, y, ch_names, sfreq = epoch_subject(raw, subject)
            except RuntimeError:
                continue  # no T1/T2 events in this run, skip

            X = zscore_normalize(X)

            all_X.append(X)
            all_y.append(y)
            all_subj.append(np.full(len(y), subject, dtype=np.int64))
            all_run.append(np.full(len(y), run, dtype=np.int64))

            if ch_names_ref is None:
                ch_names_ref, sfreq_ref = ch_names, sfreq

    # Drop subjects/runs whose epoch length differs from the majority.
    majority_times = Counter(X.shape[2] for X in all_X).most_common(1)[0][0]
    filtered = [
        (X, y, s, r)
        for X, y, s, r in zip(all_X, all_y, all_subj, all_run)
        if X.shape[2] == majority_times
    ]
    dropped_subjs = sorted({
        int(s[0]) for X, y, s, r in zip(all_X, all_y, all_subj, all_run)
        if X.shape[2] != majority_times
    })
    if dropped_subjs:
        print(
            f"Skipped runs with non-standard epoch length "
            f"(expected {majority_times} samples) from subjects: {dropped_subjs}"
        )
    all_X, all_y, all_subj, all_run = zip(*filtered)

    return EEGDataset(
        X=np.concatenate(all_X, axis=0).astype(np.float32),
        y=np.concatenate(all_y, axis=0).astype(np.int64),
        subject_ids=np.concatenate(all_subj, axis=0),
        run_ids=np.concatenate(all_run, axis=0),
        ch_names=ch_names_ref or [],
        sfreq=sfreq_ref or 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3],
                         help="PhysioNet EEGBCI subject IDs (1-109)")
    parser.add_argument("--task", choices=list(TASK_RUNS), default="imagery_lr")
    parser.add_argument("--out", type=Path, default=Path("data/processed.npz"))
    args = parser.parse_args()

    print(f"Building dataset for subjects={args.subjects}, task={args.task}")
    dataset = build_dataset(args.subjects, args.task)
    print(f"X shape: {dataset.X.shape}, y shape: {dataset.y.shape}, "
          f"sfreq: {dataset.sfreq}, n_channels: {len(dataset.ch_names)}")
    print(f"Unique subjects: {len(np.unique(dataset.subject_ids))}, "
          f"unique runs: {sorted(np.unique(dataset.run_ids).tolist())}")
    print(f"Class balance: {np.bincount(dataset.y)}")

    dataset.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
