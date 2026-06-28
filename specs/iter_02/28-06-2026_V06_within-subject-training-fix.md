# Phase 0 — Within-Subject Training Fix: Remove Val Split, Fixed Epochs, Full Batch

**Status:** Draft  
**Created:** 2026-06-28  
**Supersedes:** V05 training loop (within-subject LORO with early stopping)  
**Scope:** `eval/train_baseline.py` only — data pipeline and model unchanged

---

## Observed Results (V05)

```
Run        : vanilla
Subjects   : 106
Folds      : 318
Accuracy   : 0.5411 ± 0.0832  (54.1%)
F1 (macro) : 0.5413
```

This is barely above the 52% we had cross-subject. Within-subject should be 65–75%.

---

## Diagnosis

### The fold accuracy distribution is strongly bimodal

| Fold accuracy range | Count | Share |
|---|---|---|
| <40% (below chance) | 20 | 6.3% |
| 40–56% (dead zone) | 200 | 62.9% |
| 60–70% (working) | 63 | 19.8% |
| 70–80% | 17 | 5.3% |
| >80% | 16 | 5.0% |

- **62.9% of all folds are in the dead zone** — model never learned anything useful
- **Folds that did work average 73.3%** — exactly the literature range for EEGNet within-subject
- **40 subjects have all 3 folds in dead zone** (model consistently failed)
- **4 subjects have all 3 folds above 60%** (model consistently worked)

The model is correct and CAN learn (73.3% when it works). The problem is that 63% of folds silently fail at the training loop level, not at the architecture level.

### Root cause: 6-sample validation set for early stopping

In V05 we carved 20% of the 30-sample training fold as a validation set for early stopping:

```
30 training epochs → 6 val + 24 actual train
```

CrossEntropyLoss on 6 binary samples has extremely high variance. On any given epoch, the loss on 6 samples is not a reliable signal about model quality — it is dominated by random fluctuation. Two concrete failure modes result:

**Failure Mode 1 — Premature early stopping**

The best checkpoint is saved whenever val loss makes a new minimum. On 6 samples, this minimum can occur by chance at epoch 10–30 (when the model has barely seen the data). With patience=100, early stopping then fires at epoch 110–130. At that point, CosineAnnealingLR(T_max=500) has decayed the learning rate from 1e-3 to only ~9.9e-4 (barely moved). The model has made only 110–130 × 1.5 ≈ 165–195 gradient steps and has not converged. The saved checkpoint is from an early underfitted state.

**Failure Mode 2 — Class-biased checkpoint selection**

Some early checkpoints have the model predicting strongly toward one class (a natural intermediate state of gradient descent before the model distinguishes classes). If the 6 val samples happen to be skewed toward that class, the checkpoint looks good on val and gets saved. On the balanced 15-sample test fold, a model predicting one class scores 33–40% — below chance. This explains the 20 folds with accuracy <40% (6.3%).

**Supporting evidence:**
- Fold accuracies of 0.267 and 0.333 are consistent with a model that predicts one class for all 15 test samples where the minority class is 4–5 of 15
- S019 achieves [0.267, 0.533, 0.333] — folds 1 and 3 have the model locked on the wrong class, fold 2 happened to get a less-bad checkpoint
- When the model does work, it works well: 73.3% average for the 62 working folds (>60%)

### Secondary cause: noisy BatchNorm from small last batch

With batch_size=16 and 24 actual training samples:
- Batch 1: 16 samples
- Batch 2: 8 samples

BatchNorm running statistics computed over 8 samples in every other epoch are very noisy. This compounds with the early stopping problem: if the best checkpoint is saved at an early epoch where the running stats are poorly estimated, inference on the test fold uses those poor stats.

---

## What Does NOT Need to Change

- **Architecture (EEGNet)**: correct, achieves 73.3% on working folds
- **Data pipeline**: correct, no EA, run_ids present, z-score normalization
- **LORO split**: correct, 30 train / 15 test per fold is fine
- **Hyperparameters lr, dropout, max_norm_fc**: these are appropriate for within-subject, not the problem
- **CosineAnnealingLR**: correct scheduler, just needs to run to completion
- **Augmentations**: unchanged

---

## Fix

Three closely related changes, all in `_train_fold` and `train()`.

### Fix 1 — Remove the 20% val split

**Before:**
```python
val_size = max(2, round(len(y_tr_fold) * 0.2))   # → 6 samples
train_idx = idx[val_size:]                          # → 24 samples
# pass X_vl, y_vl to _train_fold for early stopping
```

**After:**
```python
# No split. Use all 30 training samples.
X_tr, y_tr = X_tr_fold, y_tr_fold
# _train_fold takes no val arguments
```

**Why this is safe without early stopping:** EEGNet's dropout (0.5), max-norm constraints, and the CosineAnnealingLR decay to min_lr=1e-5 are sufficient regularization. The model is not going to memorize 30 samples in 500 epochs — it will learn the stable spatial filter and temporal structure shared across trials of the same subject. Within-subject, some overfitting is actually desirable: the test fold is from the same subject, so subject-specific features learned from training runs do transfer.

### Fix 2 — Remove early stopping, train for fixed epochs

**Before (V05):** train up to max_epochs with patience=100 early stopping based on val loss.

**After (V06):** train for exactly max_epochs. Save the checkpoint at best training loss (which will be near the final epoch for a convergent run and earlier for unstable runs).

**Why fixed epochs works:**
- The CosineAnnealingLR schedule is designed to run from lr=1e-3 to min_lr=1e-5 over exactly max_epochs. When early stopping fires at epoch 130/500, the LR never decays below 9.9e-4 — the model is stuck at the initial learning rate and never fine-tunes.
- Saving at best training loss gives a simple, noise-free signal: the model that best fits the training data. Since val and train are from the same subject, this checkpoint generalizes to the held-out run.
- max_epochs=500 with CosineAnnealingLR gives the model enough steps to converge and fine-tune.

**Why "best train loss" and not "final epoch":**
- With dropout, training loss is not monotonically decreasing even on full batch (dropout introduces stochasticity). Best train loss is slightly more robust.
- In practice for converging runs, best train loss ≈ final epoch checkpoint. The difference is small.

### Fix 3 — Full-batch gradient descent

**Before:** `batch_size=min(16, len(y_train))` — with 24 train samples: batch 1 = 16, batch 2 = 8.

**After:** `batch_size=len(y_train)` — single batch with all 30 training samples per epoch.

**Why full batch is better for within-subject with 30 samples:**
- **Clean BatchNorm**: batch statistics estimated from 30 samples at every epoch are stable. With mini-batch (last batch = 8 samples), BatchNorm statistics are noisy, especially at early epochs.
- **Deterministic gradient**: the gradient is the exact gradient of the full training loss, not a noisy mini-batch estimate. For small datasets, this means faster convergence per epoch.
- **No last-batch pathology**: with batch_size=16 and 24 samples, the "last batch of 8" gets a noisier BatchNorm estimate AND a noisier gradient. This could bias the model toward features that appear in those 8 samples.

**Trade-off acknowledged:** full batch loses mini-batch noise that helps escape local minima for large datasets. For 30 samples and a relatively simple optimization landscape (EEGNet on within-subject EEG), this is not a concern.

---

## Implementation Plan

### Changes to `eval/train_baseline.py`

| Part | Change |
|---|---|
| `_train_fold` signature | Remove `X_val`, `y_val`, `patience` params; remove val loop |
| `_train_fold` internals | `batch_size=len(y_train)` (full batch); save at best train loss |
| `train()` signature | Remove `patience` param |
| `train()` body | Remove val split logic; remove `patience` from `_train_fold` call |
| `train()` body | Remove `patience` from MLflow params; add `fixed_epochs=True` |
| `train()` print | Remove `n_val=X` from per-fold output line |
| `main()` | Remove `--patience` CLI argument |

### `_train_fold` before/after (pseudocode)

```python
# BEFORE
def _train_fold(X_train, y_train, X_val, y_val, ..., patience, checkpoint_path):
    train_loader = DataLoader(..., batch_size=min(batch_size, len(y_train)))
    val_loader = DataLoader(X_val, y_val, batch_size=len(y_val))
    best_val_loss = inf
    no_improve = 0
    for epoch in range(max_epochs):
        # train
        # compute val_loss
        if val_loss < best_val_loss:
            save checkpoint
        else:
            no_improve += 1
            if no_improve >= patience: break
    return best_val_loss, stopped_epoch

# AFTER
def _train_fold(X_train, y_train, ..., checkpoint_path):
    train_loader = DataLoader(..., batch_size=len(y_train))   # full batch
    best_train_loss = inf
    for epoch in range(max_epochs):
        loss = train_step()
        if loss < best_train_loss:
            best_train_loss = loss
            save checkpoint
    return best_train_loss, max_epochs
```

### `train()` before/after (val split section)

```python
# BEFORE
rng = np.random.default_rng(seed + subj * 100 + i)
idx = rng.permutation(len(y_tr_fold))
val_size = max(2, round(len(y_tr_fold) * 0.2))
val_idx, train_idx = idx[:val_size], idx[val_size:]
X_tr = X_tr_fold[train_idx]
y_tr = y_tr_fold[train_idx]
X_vl = X_tr_fold[val_idx]
y_vl = y_tr_fold[val_idx]

# AFTER
X_tr, y_tr = X_tr_fold, y_tr_fold   # use all training fold samples
```

---

## Hyperparameters After This Change

| Param | V05 | V06 | Note |
|---|---|---|---|
| `lr` | 1e-3 | 1e-3 | unchanged |
| `batch_size` | 16 | **len(train)** (~30) | full batch |
| `max_epochs` | 500 | 500 | unchanged |
| `patience` | 100 | **removed** | no early stopping |
| `dropout` | 0.5 | 0.5 | unchanged |
| `max_norm_fc` | 0.25 | 0.25 | unchanged |
| Scheduler | CosineAnnealingLR(T_max=500) | CosineAnnealingLR(T_max=500) | now runs to completion |
| `min_lr` | 1e-5 | 1e-5 | unchanged |

---

## Expected Results

| Metric | V05 (broken) | V06 (expected) |
|---|---|---|
| Mean test accuracy | 54.1% | **65–75%** |
| Dead-zone folds (40–56%) | 62.9% | <10% |
| Working folds (>60%) | 19.5% | >80% |
| Subjects with all 3 folds in dead zone | 40 | <5 |

The folds that worked in V05 already averaged 73.3%. V06 just extends that learning to the folds that previously failed due to noisy early stopping.

---

## Success Criteria

- `vanilla` mean test accuracy ≥ 62% across subjects
- Dead-zone folds (<56%) drop from 62.9% to below 15%
- No fold accuracy at 0.267 (which indicates class collapse from a bad checkpoint)
- `augmented` mean accuracy ≥ `vanilla` mean accuracy after both runs complete

---

## Non-Goals

- No architecture changes
- No data pipeline changes
- No change to epoch window (−1 to 4 s) or run selection (imagery only: 4, 8, 12)
- Not adding execution runs (3, 7, 11) — that changes the task from pure imagery
- No subject filtering (non-responders are expected; they drag the mean down but are real data)
