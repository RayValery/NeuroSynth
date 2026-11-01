# Phase 0 — Step 3: EEGNet Classifier & Standard-Augmentation Baseline

**Status:** Draft  
**Created:** 2026-06-28  
**Precondition:** `data/processed.npz` exists (output of `data/eegbci_pipeline.py`).  
**Next step after this:** Phase 0 Step 4 — 1D U-Net diffusion model training.

---

## Executive Summary

This spec covers the third task in Phase 0: implementing the **downstream EEG classifier** (EEGNet) and the **standard-augmentation baseline** that the generative model will later need to beat. These two pieces together answer the go/no-go question: if training on synthetic data cannot beat naive signal augmentation, there is no product. This step establishes the baseline accuracy number that all future comparisons will be measured against.

---

## Goals

- Train an EEGNet classifier on real EEGBCI data and record a reproducible baseline accuracy.
- Implement every standard-augmentation technique (Gaussian noise, time-shift, channel dropout, sliding-window crops) as a reusable `Augmentor` class.
- Run an augmentation-augmented training and record its accuracy as the **baseline to beat** (RR-2).
- Log all runs to MLflow so results are reproducible and citable in the preprint (RR-5).

## Non-Goals

- No synthetic data generation in this step (that is Step 4).
- No TSTR harness yet (that is Step 5).
- No hyperparameter search — use literature defaults for EEGNet.
- No deployment or API integration.

---

## Background & Context

The critical go/no-go question (Section 3.3 of the project spec): does training with synthetic EEG outperform training with standard augmentation? To answer it we need:

1. A fixed, well-understood **classifier** used as the downstream evaluator — EEGNet is the standard choice for motor-imagery EEG (Lawhern et al. 2018, DOI 10.1088/1741-2552/aace8c). It is compact (~2,500 parameters), robust across paradigms, and ubiquitous in BCI benchmarks, making it the right choice for both research credibility and fair comparison.
2. A strong **augmentation baseline** that represents the current state-of-the-art without generative models. If the diffusion model cannot beat this, it provides no added value.

Data available: `data/processed.npz` — shape `(n_epochs, 64, n_times)`, labels `{0, 1}`, per-epoch z-score normalized, 160 Hz, motor-imagery left/right fist (task `imagery_lr`).

---

## Requirements

### Must Have
- EEGNet implementation in PyTorch matching the original paper architecture (F1, D, F2, T1, T2 as configurable params with paper defaults).
- Subject-stratified train/val/test split (no subject leakage across splits).
- Training loop with early stopping (patience = 15 epochs) and Adam optimizer.
- Four augmentation transforms: Gaussian noise, time-shift, channel dropout, sliding-window crop.
- `Augmentor` class that randomly applies one or more transforms per batch.
- Augmentation-augmented training run producing `aug_baseline_accuracy` logged to MLflow.
- Vanilla (no augmentation) training run producing `real_only_accuracy` as a sanity check.
- Saved best-checkpoint `.pt` files for both runs.
- Reproducible: all runs seeded (`torch.manual_seed`, `numpy.random.seed`).

### Should Have
- Per-class F1 score alongside accuracy (binary, but F1 is more appropriate for imbalanced folds).
- Confusion matrix logged as an MLflow artifact.
- Short CLI entrypoint: `python eval/train_baseline.py --data data/processed.npz --run vanilla` and `--run augmented`.

### Nice to Have
- `eval/inspect_data.py` quick sanity script — prints dataset shape, class balance, plots one epoch's PSD (non-blocking matplotlib).
- Docstring citing EEGNet paper on the model class.

---

## Architecture Overview

```
eval/
  train_baseline.py    # CLI entrypoint — calls trainer, logs to MLflow
  eegnet.py            # EEGNet PyTorch module
  augmentations.py     # Augmentor class + 4 transforms
  dataset.py           # EEGDataset → PyTorch Dataset + stratified splitter
  inspect_data.py      # (optional) quick data sanity check
models/
  checkpoints/         # best_vanilla.pt, best_augmented.pt
```

MLflow experiment: `phase0-baseline`. Two runs: `vanilla` and `augmented`. Artifacts: `confusion_matrix.png`, training curves.

---

## Data Models

### EEGTorchDataset (`eval/dataset.py`)

```python
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGTorchDataset(Dataset):
    """Wraps (X, y) arrays as a PyTorch Dataset. X: float32 (N, C, T), y: int64 (N,)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, augmentor=None):
        self.X = torch.from_numpy(X)   # (N, C, T) float32
        self.y = torch.from_numpy(y)   # (N,) int64
        self.augmentor = augmentor

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (C, T)
        if self.augmentor is not None:
            x = self.augmentor(x)
        return x, self.y[idx]
```

### Splitting strategy

Subject-stratified 70/15/15 split: group epochs by `subject_ids`, assign whole subjects to train/val/test (never split a subject across sets). This is the standard BCI evaluation protocol and prevents the classifier from learning subject-specific artifacts instead of task-related signal.

```python
def subject_stratified_split(
    X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
    val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[tuple, tuple, tuple]:
    """Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)."""
    ...
```

---

## EEGNet Architecture (`eval/eegnet.py`)

EEGNet (Lawhern et al. 2018) — compact depthwise separable CNN for EEG.

```
Input: (batch, 1, n_channels, n_times)   ← unsqueeze channel dim before passing

Block 1 — Temporal convolution:
  Conv2d(1, F1, kernel=(1, 64), padding=(0,32), bias=False)
  BatchNorm2d(F1)

Block 2 — Depthwise spatial convolution:
  Conv2d(F1, F1*D, kernel=(n_channels,1), groups=F1, bias=False)
  BatchNorm2d(F1*D)
  ELU
  AvgPool2d(kernel=(1,4))
  Dropout(p=0.5)

Block 3 — Separable convolution:
  Conv2d(F1*D, F1*D, kernel=(1,16), padding=(0,8), groups=F1*D, bias=False)
  Conv2d(F1*D, F2, kernel=(1,1), bias=False)
  BatchNorm2d(F2)
  ELU
  AvgPool2d(kernel=(1,8))
  Dropout(p=0.5)

Classifier:
  Flatten → Linear(F2 * temporal_out, n_classes)
```

Default hyperparameters (from paper, motor imagery):

| Param | Value | Meaning |
|-------|-------|---------|
| F1 | 8 | Number of temporal filters |
| D | 2 | Depth multiplier |
| F2 | 16 | Number of pointwise filters (= F1*D) |
| n_channels | 64 | EEG channels |
| n_times | 801 | Time points at 160 Hz for 5s epoch (tmin=-1, tmax=4) |
| n_classes | 2 | Binary motor imagery |
| dropout | 0.5 | — |

```python
class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=801, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.5):
        ...
```

---

## Augmentation Interface (`eval/augmentations.py`)

All transforms operate on a single epoch tensor `x: torch.Tensor` of shape `(C, T)` and return the same shape. They are applied at training time only, per-sample, inside `__getitem__`.

### Transforms

```python
def gaussian_noise(x: Tensor, std: float = 0.1) -> Tensor:
    """Add zero-mean Gaussian noise. std relative to signal std (0.1 = 10%)."""
    return x + torch.randn_like(x) * std * x.std()

def time_shift(x: Tensor, max_shift: int = 50) -> Tensor:
    """Randomly roll signal along time axis by up to max_shift samples."""
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(x, shift, dims=-1)

def channel_dropout(x: Tensor, p: float = 0.1) -> Tensor:
    """Zero out each channel independently with probability p."""
    mask = (torch.rand(x.shape[0], 1) > p).float()
    return x * mask

def sliding_window_crop(x: Tensor, crop_frac: float = 0.8) -> Tensor:
    """Randomly crop a contiguous time window of length floor(T * crop_frac),
    then pad back to T with reflection to preserve shape."""
    T = x.shape[-1]
    crop_len = int(T * crop_frac)
    start = torch.randint(0, T - crop_len + 1, (1,)).item()
    cropped = x[:, start:start + crop_len]
    return torch.nn.functional.pad(
        cropped.unsqueeze(0), (0, T - crop_len), mode="reflect"
    ).squeeze(0)
```

### Augmentor class

```python
class Augmentor:
    """Randomly applies a subset of transforms each call."""
    def __init__(self, transforms: list, p_apply: float = 0.5):
        self.transforms = transforms
        self.p_apply = p_apply  # probability of applying each transform

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            if torch.rand(1).item() < self.p_apply:
                x = t(x)
        return x

STANDARD_AUGMENTOR = Augmentor([
    gaussian_noise,
    time_shift,
    channel_dropout,
    sliding_window_crop,
], p_apply=0.5)
```

---

## Training Loop (`eval/train_baseline.py`)

### Hyperparameters

| Param | Value |
|-------|-------|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Loss | CrossEntropyLoss |
| Epochs max | 200 |
| Early stopping patience | 15 (on val loss) |
| Batch size | 64 |
| Seed | 42 |
| Device | auto (MPS > CUDA > CPU) |

### Training procedure

```python
def train(
    data_path: Path,
    run_name: str,           # "vanilla" or "augmented"
    checkpoint_dir: Path,
    mlflow_experiment: str = "phase0-baseline",
    seed: int = 42,
) -> dict:
    """Returns dict with test_accuracy, test_f1, run_id."""
```

1. Load `EEGDataset.load(data_path)`.
2. Subject-stratified split (train 70% subjects / val 15% / test 15%).
3. Build `EEGTorchDataset` — augmentor only on train split; val and test always clean (no augmentation on eval).
4. Instantiate `EEGNet(n_channels, n_times, n_classes=2)`.
5. Training loop with early stopping — save best checkpoint on lowest val loss to `checkpoint_dir/best_{run_name}.pt`.
6. Load best checkpoint, evaluate on test set → `test_accuracy`, `test_f1`.
7. Log to MLflow: all hyperparams, `test_accuracy`, `test_f1`, training curve PNG, confusion matrix PNG.
8. Return metrics dict.

### CLI

```
python eval/train_baseline.py --data data/processed.npz --run vanilla
python eval/train_baseline.py --data data/processed.npz --run augmented
```

Both runs must be logged under the same MLflow experiment `phase0-baseline` so they appear side-by-side in the UI.

---

## Implementation Plan

### Phase A: Data layer
- [ ] Create `eval/__init__.py`, `eval/dataset.py`
- [ ] Implement `EEGTorchDataset` with optional augmentor hook
- [ ] Implement `subject_stratified_split` — shuffle subjects with `seed`, assign to train/val/test by subject count fractions
- [ ] Write a quick sanity print: number of subjects per split, class balance per split, epoch shape

### Phase B: Model
- [ ] Create `eval/eegnet.py`
- [ ] Implement `EEGNet` exactly matching paper architecture (Block 1, 2, 3 + classifier)
- [ ] Add `__repr__` that prints parameter count
- [ ] Smoke test: `model(torch.randn(4, 1, 64, 801))` returns `(4, 2)` without error

### Phase C: Augmentations
- [ ] Create `eval/augmentations.py`
- [ ] Implement the four transforms as standalone functions
- [ ] Implement `Augmentor` class
- [ ] Export `STANDARD_AUGMENTOR` constant

### Phase D: Training loop
- [ ] Create `eval/train_baseline.py`
- [ ] Implement `train()` function with early stopping, MLflow logging, checkpoint saving
- [ ] Add device auto-detection (MPS > CUDA > CPU)
- [ ] Add CLI argument parsing (`--data`, `--run`, `--seed`, `--checkpoint-dir`)
- [ ] Input tensor: add `unsqueeze(1)` inside model or DataLoader collate to get shape `(B, 1, C, T)`

### Phase E: Run both baselines
- [ ] Run `python eval/train_baseline.py --data data/processed.npz --run vanilla`
- [ ] Run `python eval/train_baseline.py --data data/processed.npz --run augmented`
- [ ] Confirm both runs appear in `mlflow ui` under experiment `phase0-baseline`
- [ ] Record `vanilla_test_accuracy`, `augmented_test_accuracy`, `vanilla_test_f1`, `augmented_test_f1` — these numbers are the go/no-go comparison target

### Phase F: (Optional) Sanity visualization
- [ ] Create `eval/inspect_data.py` — load `.npz`, print shape/balance, plot one epoch's PSD per class

---

## Error Handling

- If `data/processed.npz` is missing, `train_baseline.py` must print a helpful message: "Run `python data/eegbci_pipeline.py` first to generate the dataset."
- If subject count is too low to split (< 4 subjects), raise `ValueError` with a clear message.
- If no GPU is found, log a warning and continue on CPU — full training will be slow (~hours) but correct. Document expected CPU runtime in the CLI output.
- If MLflow tracking fails (e.g. no server), fall back to local `mlruns/` directory (this is the default MLflow behavior — no explicit handling needed).

---

## Testing Strategy

- **Smoke test (run immediately after implementation):** `model(torch.randn(4, 1, 64, 801))` → shape `(4, 2)`.
- **Split sanity:** assert no subject appears in more than one split.
- **Augmentor shape invariance:** for each transform, assert `output.shape == input.shape`.
- **Augmentor train-only:** val/test DataLoaders must have `augmentor=None`; assert this in the split construction code.
- **Overfitting sanity:** train for 5 epochs on 10 epochs of data — loss should decrease. If it doesn't, model or data pipeline is broken.
- **Reproducibility:** two runs with the same seed must produce identical val loss curves.

No formal pytest suite is required at this stage — the above are manual checks to run before committing results to MLflow.

---

## Success Criteria

This step is done when:
1. `eval/eegnet.py`, `eval/augmentations.py`, `eval/dataset.py`, `eval/train_baseline.py` all exist and are runnable.
2. MLflow experiment `phase0-baseline` contains two completed runs: `vanilla` and `augmented`.
3. Both runs have logged `test_accuracy` and `test_f1`.
4. `models/checkpoints/best_vanilla.pt` and `models/checkpoints/best_augmented.pt` exist.
5. The `augmented` run accuracy is recorded and becomes the **threshold the diffusion model must beat in Step 5 (TSTR)**.

Expected accuracy range (literature baseline, motor imagery L/R on EEGBCI): 65–80% for a clean EEGNet with limited subjects. If accuracy is below 55% (near chance), re-check the data pipeline for label leakage or normalization issues before proceeding.

---

## Open Questions

- [ ] How many subjects is `processed.npz` currently built from? (Check `data/test.py` output or re-run pipeline — this determines realistic subject split sizes.) If only 3 subjects, the split is too small; re-run pipeline with `--subjects 1 2 3 ... 109` before proceeding.
- [ ] Should `imagery_ff` (fists vs feet) be included alongside `imagery_lr` (left vs right) for a richer dataset? Recommend yes for the preprint — increases epoch count ~2× and tests cross-paradigm generalization. Can be done now or deferred to Step 4.
