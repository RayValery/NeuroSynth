# Phase 0 — Euclidean Alignment for Cross-Subject EEG

**Status:** Draft  
**Created:** 2026-06-28  
**Prerequisite:** `28-06-2026_V03_phase0-cross-subject-training-fix.md` implemented  
**Problem:** EEGNet val loss decreasing at ~0.000047/epoch. At this rate the model reaches ~52% accuracy by epoch 300 — below the 55% minimum threshold, and far below literature cross-subject baselines of 57–65%.

---

## Why Cross-Subject EEG Is Hard: The Actual Mechanism

To understand why Euclidean Alignment fixes this, we need to understand precisely what makes cross-subject EEG classification hard.

### What the EEGNet spatial filter does

Block 2 of EEGNet is a depthwise spatial filter: a learned matrix **W** ∈ ℝ^(D×C) (D=16 filters, C=64 channels) applied across all electrode channels. For each epoch **X** ∈ ℝ^(C×T), it computes:

```
output = W @ X    →   (16, T)
```

This is equivalent to a learned spatial beamformer: it combines channels in a weighted sum to extract task-relevant sources (e.g., the contralateral motor cortex during left/right fist imagery). The whole model's ability to discriminate between classes depends on **W** correctly isolating ERD (event-related desynchronization) signal from background noise.

### The subject-variability problem

Every person's EEG looks fundamentally different in the spatial domain because:

1. **Brain anatomy**: the motor cortex sits at different positions relative to the scalp across people — the C3/C4 electrodes capture different amounts of motor cortex signal per subject.
2. **Volume conduction**: the degree of signal spreading across electrodes depends on individual skull thickness, scalp conductivity, and CSF volume.
3. **Electrode impedances**: contact quality differs per electrode per session, scaling channel amplitudes non-uniformly.
4. **Individual ERD topographies**: some subjects show strongly lateralized motor ERD; others show diffuse, bilateral patterns.

These factors manifest as different **inter-channel covariance structures** per subject. Subject A's covariance matrix **R_A** and subject B's **R_B** can differ substantially — meaning the spatial geometry of the EEG signal is different between them.

### Why this breaks cross-subject training

The optimal spatial filter **W*** for subject A is:

```
W*_A = argmin_W  Loss(W @ X_A, y_A)
```

This optimum depends on **R_A** (the spatial structure of subject A's data). For subject B with a different **R_B**, the optimal **W*_B** points in a different direction in weight space.

When we train cross-subject, the gradient from subject A's batch pulls **W** toward **W*_A**, and the gradient from subject B's batch pulls it toward **W*_B**. If these optima are far apart (high inter-subject variability), the gradients partially cancel and the net update is tiny. The model makes progress on no subject in particular.

This is exactly what we observe: training loss barely moves despite correct model architecture and optimizer settings. The per-epoch gradients are noisy and pointing in opposing directions.

---

## Why Our Current Z-Score Normalization Does Not Fix This

Our pipeline applies per-epoch, per-channel z-score normalization (mean=0, std=1 over time):

```python
mean = X.mean(axis=-1, keepdims=True)   # shape: (N, C, 1)
std  = X.std(axis=-1, keepdims=True)    # shape: (N, C, 1)
X_normalized = (X - mean) / std
```

This operates on the **temporal axis** of each channel independently. It removes:
- DC offsets per channel (inter-session baseline drift)
- Per-channel amplitude differences

What it does NOT remove: **spatial covariance structure**. The inter-channel correlations — how much C3 co-activates with C4, how much Fz correlates with Cz — remain intact and remain different between subjects. Z-scoring each channel to unit variance does not decorrelate channels or equalize their spatial geometry. Subject A's spatially-biased covariance structure persists after z-scoring.

---

## The Fix: Euclidean Alignment

**Euclidean Alignment (EA)** is a per-subject preprocessing step proposed by He & Wu (2019, DOI: 10.1109/TNNLS.2019.2913946). It whitens each subject's data by their mean spatial covariance matrix, mapping all subjects to the same reference geometry.

### Algorithm

For subject *s* with epochs **X₁, X₂, ..., Xₙ** ∈ ℝ^(C×T):

**Step 1 — Compute the mean covariance matrix:**
```
R_s = (1/n) * Σᵢ (Xᵢ @ Xᵢᵀ) / T       →   shape (C, C)
```
This is the average of the per-epoch sample covariance matrices. It captures the typical spatial structure of this subject's EEG — which channels tend to co-activate, which are decorrelated.

**Step 2 — Compute the inverse square root:**
```
R_s^(-1/2)    via eigendecomposition:
    R_s = V Λ Vᵀ
    R_s^(-1/2) = V Λ^(-1/2) Vᵀ
```

**Step 3 — Align every epoch:**
```
X̃ᵢ = R_s^(-1/2) @ Xᵢ       →   shape (C, T)
```

**What this achieves:** after alignment, the mean covariance of subject *s* becomes:

```
(1/n) * Σᵢ (X̃ᵢ @ X̃ᵢᵀ) / T
= R_s^(-1/2) @ R_s @ R_s^(-1/2)
= I     (identity matrix)
```

Every subject's mean covariance is now the identity. All subjects are in the same spatial reference frame. The EEGNet spatial filter **W** now sees data with a consistent geometry across subjects — gradients from different subjects no longer point in opposing directions.

### What EA preserves

EA does NOT destroy the task signal:
- ERD/ERS patterns are **relative changes** in power during motor imagery vs. rest. The mean covariance captures the AVERAGE structure; task-modulated deviations from that average survive whitening.
- Temporal dynamics (the shape of the EEG over time) are unchanged — EA is a linear spatial transform applied identically to each time point.
- The classification signal (which class produces more or less power in which spatial pattern) is preserved and made more comparable across subjects.

---

## Implementation Plan

### What changes

**Only `data/eegbci_pipeline.py`** — EA is a data preprocessing step, not an architectural or training change. The result is saved into `data/processed.npz` and all downstream code (`eval/`, `models/`) is unchanged.

### Where in the pipeline

EA must be applied **after** z-score normalization. This is counter-intuitive but empirically verified: applying EA before z-score completely fails.

**Why EA → z-score fails:**  
After EA on raw data, the per-channel variances are very small (the whitened signal has diagonal ≈ 0.0004 in raw μV² units). Z-score then divides each channel by its per-epoch standard deviation — a tiny, epoch-specific scalar that is correlated with cross-channel activity. This epoch-specific rescaling re-introduces the exact inter-channel correlations EA just removed. Verified empirically: off-diagonal mean covariance after EA = 0.000, after EA + z-score = 0.604 — fully restored.

**Why z-score → EA works:**  
After z-score, all channels already have unit per-epoch variance (diagonal = 1). The covariance matrix is now a pure **correlation matrix**, and the off-diagonal correlations (mean abs ≈ 0.607) encode the subject-specific spatial geometry. EA applied to this representation whitens those correlations to zero while leaving the diagonal at 1. Result: off-diagonal = 0.000, diagonal = 1.000. This is stable because there is no per-epoch rescaling after EA that could undo the alignment.

New pipeline order per subject:
```
download → bandpass filter → epoch → z-score → EA → save
```

vs. previous order:
```
download → bandpass filter → epoch → z-score → save
```

### Where EA is computed: full dataset vs. per-split

EA must be computed per-subject using **all of that subject's epochs** (not just training epochs). This is not leakage because:
- EA uses only the subject's own data to compute their own covariance
- Test subjects' EA is computed from test subjects' data only — no train information flows in
- The train/val/test split is by subject, so alignment is fully self-contained per subject

This means EA is applied in `build_dataset()` after all subjects are collected but before saving — or equivalently, as a per-subject step inside the subject loop.

### New function in `data/eegbci_pipeline.py`

```python
def euclidean_align(X: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment (He & Wu, 2019) for one subject's epochs.
    Whitens X by its mean covariance so that inter-subject spatial
    variability is removed before cross-subject training.
    Must be applied AFTER z-score normalization (see spec for reasoning).

    Args:
        X: (n_epochs, n_channels, n_times) float32 — already z-scored
    Returns:
        X_aligned: same shape, spatially whitened
    """
```

- Compute mean covariance R = mean of (xᵢ @ xᵢᵀ / T) over epochs (this is the correlation matrix when input is z-scored)
- Add a small ridge `ε·I` (ε=1e-6) for numerical stability before inversion
- Compute R^(-1/2) via `np.linalg.eigh` (symmetric matrix, numerically stable)
- Return `(R^(-1/2) @ X_i)` for each epoch i

### Call site in `build_dataset()`

After `X, y, ch_names, sfreq = epoch_subject(raw, subject)`, z-score first then EA:

```python
X = zscore_normalize(X)     # existing: temporal normalization first
X = euclidean_align(X)      # NEW: spatial alignment on z-scored data
```

### Re-generate `data/processed.npz`

After implementing, re-run:
```
python data/eegbci_pipeline.py --subjects $(seq 1 109) --out data/processed.npz
```

The `.npz` file is overwritten in-place. No changes needed to `eval/` or `models/`.

---

## Expected Improvement

From the EA paper and replications on EEGBCI motor imagery:
- Without EA: 52–58% cross-subject accuracy (what we're seeing)
- With EA: 57–65% cross-subject accuracy (+5–10% typical)

A clean EEGNet implementation with EA has been reported at ~60–63% on left/right fist imagery cross-subject (e.g., Kostas et al. 2020, "Thinker invariance"). This would satisfy our ≥55% threshold and provide a legitimate baseline for the TSTR comparison.

The mechanism: after EA, all subject gradients point in similar directions, and the spatial filter can converge to a meaningful cross-subject beamformer within normal training timescales (100–200 epochs).

---

## Success Criteria

- Training loss decreases visibly below 0.680 by epoch 50 (previously stuck at 0.694)
- `vanilla` test accuracy ≥ 55%
- If accuracy is still <55% after a full run with EA: the problem is confirmed architectural (subject-adaptive layers, domain adaptation) and out of scope for Phase 0 — proceed with the best available baseline and note the limitation in the preprint

---

## Non-Goals

- No changes to EEGNet architecture
- No changes to training loop or hyperparameters
- No changes to split strategy
- No new dependencies — `numpy.linalg.eigh` is sufficient; `scipy` not required
