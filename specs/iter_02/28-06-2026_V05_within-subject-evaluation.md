# Phase 0 — Switch to Within-Subject Evaluation

**Status:** Draft  
**Created:** 2026-06-28  
**Supersedes:** V01–V04 evaluation strategy (cross-subject EEGNet baseline)  
**Scope:** `data/eegbci_pipeline.py`, `eval/dataset.py`, `eval/train_baseline.py`

---

## Why We Are Switching

### The cross-subject approach failed — and for fundamental reasons

After six rounds of hyperparameter tuning and preprocessing changes (V01–V04), the cross-subject EEGNet is stuck at ~52% accuracy, barely above chance. This is not a tuning failure. It is the expected behaviour of a model that must learn one fixed spatial filter across 74 subjects whose motor cortices sit in different positions, whose skull thicknesses differ, and whose ERD topographies are idiosyncratic. The gradients from different subjects point in opposing directions in weight space; the net update per batch is near-zero. No learning rate, patience, or preprocessing fix can overcome this — it requires architectural changes (domain adaptation, meta-learning, per-subject batch normalisation) that are explicitly out of scope for Phase 0.

### The within-subject protocol is scientifically and commercially correct

**Scientifically:** The EEGNet paper reports within-subject results (65–80%). All standard EEGBCI benchmarks are within-subject. A cross-subject baseline cannot be cited against the literature; a within-subject baseline can be. The preprint needs comparable numbers.

**Commercially:** The product use case is: *a BCI company has a new user with 5 minutes of calibration data; they buy synthetic EEG to augment it and get a better personal classifier.* This is a within-subject augmentation problem. The generative model generates population-level data; the classifier is trained per-user. The within-subject TSTR evaluation maps directly onto this scenario.

**For TSTR (the go/no-go gate):** The question becomes "does adding synthetic EEG from the generative model to a subject's small real training set improve their within-subject classifier?" This is cleaner, higher-accuracy, and publishable.

### The generative model is unchanged

The diffusion model (Phase 0 Step 4) still trains on all subjects' data. It still generates label-conditioned (T1/T2) synthetic epochs from the population distribution. What changes is only how EEGNet is evaluated. The synthetic data is not subject-specific — it is added to each subject's real training set as augmentation. The evaluation loop iterates over subjects and averages.

---

## EA Decision: Remove It

### Why EA was added

EA was introduced to reduce inter-subject spatial variability for cross-subject training — to map all subjects' covariance structures to a common identity reference so EEGNet's spatial filter could learn a single beamformer that works across subjects.

### Why EA is harmful for within-subject training

For within-subject evaluation, EEGNet trains exclusively on one subject's data. Its spatial filter learns **this subject's specific motor cortex topology** — which electrodes over C3 and C4 show ERD for this person's anatomy. The inter-channel covariance structure *is* the task signal: C3 correlates with its neighbours because they all pick up the same motor cortex source. This correlation pattern differs between left and right fist imagery.

EA whitens all inter-channel correlations to zero. For within-subject, this removes the primary spatial information that EEGNet's spatial filter is designed to exploit. Applying EA before training within-subject is equivalent to deliberately blinding the model to the subject's spatial brain geometry.

### Empirical confirmation

We observed this directly: EA (applied after z-score) reduced off-diagonal covariance from 0.607 to 0.000005. The subsequent training run (V04) achieved *worse* performance than the pre-EA run — val loss started higher (0.6957 vs 0.6940) and improved at the same glacially slow rate. Removing the spatial information hurt, not helped.

### What to do

Remove the `euclidean_align()` call from `build_dataset()`. Keep the function definition in the file for future reference (it may be useful for cross-subject experiments later). Revert the pipeline to:

```
epoch → zscore_normalize → save
```

Regenerate `data/processed.npz`.

---

## Evaluation Protocol: Leave-One-Run-Out

### Why not k-fold on epochs

Simple k-fold on shuffled epochs has a subtle data leakage issue: epochs within the same experimental run are temporally correlated (they share the same EEG session, the same electrode impedances, the same subject state). Shuffling and splitting within-run can place temporally adjacent epochs on opposite sides of the train/test boundary, inflating accuracy.

### Leave-one-run-out (LORO)

EEGBCI `imagery_lr` uses exactly 3 runs per subject (runs 4, 8, 12). LORO uses 2 runs for training and holds out 1 for testing, cycling through all 3 combinations:

```
Fold 1: train on runs {8, 12}, test on run {4}
Fold 2: train on runs {4, 12}, test on run {8}
Fold 3: train on runs {4,  8}, test on run {12}
```

Each fold gives ~30 training epochs and ~15 test epochs per subject. Per-fold accuracy is averaged across 3 folds, then per-subject accuracy is averaged across all subjects. This is the standard split for EEGBCI motor imagery in the literature.

### Why run_ids must be tracked

LORO requires knowing which run each epoch came from. Currently `processed.npz` stores `subject_ids` but not `run_ids`. We need to add `run_ids` to the pipeline.

---

## Implementation Plan

### Files changed

| File | Change |
|---|---|
| `data/eegbci_pipeline.py` | Remove EA call; add `run_ids` tracking; update `EEGDataset` |
| `eval/dataset.py` | Add `within_subject_loro_split()` function; keep old `subject_stratified_split` for reference |
| `eval/train_baseline.py` | Replace global train/test loop with per-subject LORO evaluation loop; new hyperparameters |
| `data/processed.npz` | Regenerated (no EA, with run_ids) |

`eval/eegnet.py` and `eval/augmentations.py` are unchanged.

---

### 1. `data/eegbci_pipeline.py`

#### 1a. Remove EA call from `build_dataset()`

```python
# Before (V04):
X = zscore_normalize(X)
X = euclidean_align(X)    # ← remove this line

# After (V05):
X = zscore_normalize(X)
```

Keep the `euclidean_align()` function definition — do not delete it. Update its docstring to note it is unused in the standard pipeline and intended for cross-subject experiments only.

#### 1b. Track run_ids in `EEGDataset`

Add `run_ids` field to the dataclass, `save()`, and `load()`:

```python
@dataclass
class EEGDataset:
    X: np.ndarray           # (n_epochs, n_channels, n_times)
    y: np.ndarray           # (n_epochs,) int labels
    subject_ids: np.ndarray # (n_epochs,) subject per epoch
    run_ids: np.ndarray     # (n_epochs,) run number per epoch  ← NEW
    ch_names: list[str]
    sfreq: float
```

#### 1c. Track run_ids in `build_dataset()`

`epoch_subject()` already processes a concatenated raw that spans multiple runs. To know which epoch came from which run, process each run separately inside `build_dataset()` and tag accordingly:

```python
for subject in subjects:
    for run in runs:                            # iterate runs individually
        raw = download_subject_runs(subject, [run])
        raw = preprocess_raw(raw)
        X_run, y_run, ch_names, sfreq = epoch_subject(raw, subject)
        X_run = zscore_normalize(X_run)
        all_X.append(X_run)
        all_y.append(y_run)
        all_subj.append(np.full(len(y_run), subject, dtype=int))
        all_run.append(np.full(len(y_run), run, dtype=int))   # ← NEW
```

This replaces the current single `download_subject_runs(subject, runs)` call with a per-run loop. The epoch count and values are identical — only the run tagging is new.

---

### 2. `eval/dataset.py`

Add a new function. Keep `subject_stratified_split` unchanged.

```python
def within_subject_loro_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    run_ids: np.ndarray,
) -> list[dict]:
    """
    Leave-One-Run-Out splits for within-subject evaluation.

    Returns a list of dicts, one per (subject, held-out-run):
        {
            "subject": int,
            "test_run": int,
            "X_train": ndarray, "y_train": ndarray,
            "X_test":  ndarray, "y_test":  ndarray,
        }
    The caller iterates this list, trains one EEGNet per entry, and averages.
    """
```

Logic: for each subject, for each unique run of that subject, hold that run out as test and use the rest as train.

---

### 3. `eval/train_baseline.py`

#### 3a. New evaluation loop

Replace the current single split → single model → single test with:

```python
results = []
for split in within_subject_loro_splits:
    model = EEGNet(...)
    # train on split["X_train"], split["y_train"]
    # evaluate on split["X_test"], split["y_test"]
    results.append({"subject": split["subject"], "test_run": split["test_run"], "accuracy": acc, "f1": f1})

# Aggregate: mean per subject, then mean across subjects
per_subject = {s: mean(acc for r in results if r["subject"]==s) for s in subjects}
overall_accuracy = mean(per_subject.values())
overall_f1 = mean(f1 for r in results)
```

#### 3b. Hyperparameters for within-subject

Within-subject trains on ~30 epochs per fold. Appropriate hyperparameters differ from cross-subject:

| Param | Cross-subject (V03) | Within-subject (V05) | Reason |
|---|---|---|---|
| `lr` | 1e-4 | **1e-3** | Small dataset converges fast; high LR is fine |
| `batch_size` | 64 | **16** | Only 30 training samples; batch=64 > dataset size |
| `max_epochs` | 300 | **500** | More epochs needed for small data with high LR decay |
| `patience` | 50 | **100** | Within-subject loss is noisy on tiny batches |
| `dropout` | 0.25 | **0.5** | Back to paper default; small within-subject data needs regularisation |
| `max_norm_fc` | 0.5 | **0.25** | Back to paper default; within-subject is the intended regime |
| Scheduler | CosineAnnealingLR | **CosineAnnealingLR** | Unchanged |
| `min_lr` | 1e-5 | **1e-5** | Unchanged |

#### 3c. MLflow logging

Per-subject results logged as a JSON artifact. Summary metrics logged as scalars:

```
test_accuracy_mean    ← primary metric (mean across all subjects)
test_accuracy_std     ← inter-subject variability
test_f1_mean
n_subjects
n_folds_total
```

Per-subject breakdown in a `per_subject_results.json` artifact.

#### 3d. CLI

No new flags required. The `--run vanilla` / `--run augmented` distinction is preserved. All hyperparameter changes are new defaults.

---

## Data Pipeline After This Change

```
download → bandpass filter → epoch (per run) → z-score → save
                                                           ↓
                                              X, y, subject_ids, run_ids, ch_names, sfreq
```

EA is gone. run_ids are added. Everything else is identical to the original V01 pipeline.

---

## Expected Results

Literature within-subject EEGNet on EEGBCI `imagery_lr`, LORO evaluation:

| Run | Expected accuracy |
|---|---|
| Vanilla (real only) | 65–75% mean across subjects |
| Augmented (real + noise/shift/crop) | 68–78% mean |
| TSTR (real + synthetic, Phase 0 Step 5) | Target: beat augmented by ≥5% |

High inter-subject variance is expected (σ ≈ 10–15%). Some subjects achieve 90%+; others are near chance (genuine non-responders to motor imagery BCI). The mean is the reported number.

---

## Success Criteria

- `vanilla` mean test accuracy ≥ 60% across subjects (conservative floor given literature)
- Loss curve per subject shows visible convergence within 200 epochs for at least 80% of subjects
- `augmented` mean accuracy ≥ `vanilla` mean accuracy (confirms augmentation is not hurting)
- Per-subject JSON artifact present in MLflow

---

## Non-Goals

- No architecture changes to EEGNet
- No subject-conditional generation (deferred to Phase 1)
- No fine-tuning or transfer learning — EEGNet trains from scratch per fold
- No removal of the `euclidean_align()` function from the codebase (keep for future use)
- No changes to `eval/augmentations.py` or `eval/eegnet.py`
