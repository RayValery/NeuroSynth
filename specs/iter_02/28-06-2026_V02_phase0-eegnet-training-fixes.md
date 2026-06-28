# Phase 0 — EEGNet Training Fixes

**Status:** Draft  
**Created:** 2026-06-28  
**Supersedes:** Nothing — addendum to `phase0-eegnet-classifier-and-augmentation-baseline.md`  
**Problem:** `vanilla` run achieved 50.1% accuracy (chance) with early stopping at epoch 29.

---

## Diagnosis

| Root cause | Evidence | Impact |
|---|---|---|
| `max_norm_fc=0.25` fights Adam every batch | FC norms clipped from 0.58 → 0.25 at init; logit abs-mean stays ~0.15 throughout training | Model never produces confident predictions; loss never escapes log(2) ≈ 0.693 |
| `patience=15` too short | Best val_loss hit ~epoch 14, stopped at epoch 29; 15 epochs of non-improvement is insufficient noise budget for 16-subject val set | Premature stopping before any useful basin is found |
| No LR scheduler | No mechanism to escape plateau | Model spins in place once constraints push it to a flat region |
| `dropout=0.5` tuned for within-subject | EEGNet paper trains one model per subject on ~45 epochs; we train cross-subject on 3310 epochs — much harder generalization regime | Too aggressive regularization suppresses cross-subject signal |

---

## Changes

### `eval/train_baseline.py`

1. **Add `ReduceLROnPlateau` scheduler** — decays LR by `factor=0.5` when val loss has not improved for `lr_patience=10` epochs, floor at `min_lr=1e-5`. Step it after every epoch's val pass.

2. **Increase default `patience`** from 15 → 30. The early-stopping counter is now reset by *any* val loss improvement (unchanged), but gives the model twice as much headroom.

3. **Expose `--dropout` CLI arg** (default `0.25`) and pass it to `EEGNet(dropout=...)`. Old default was 0.5.

4. **Expose `--max-norm-fc` CLI arg** (default `0.5`) and pass it to `model.apply_weight_constraints(max_norm_fc=...)` each batch. Old hardcoded value was 0.25.

5. **Log new hyperparams to MLflow** — `dropout`, `max_norm_fc`, `lr_patience`, scheduler `factor`.

### `eval/eegnet.py`

No changes — `dropout` is already a constructor parameter. `apply_weight_constraints` already accepts `max_norm_fc` as an argument.

---

## Hyperparameter summary (before → after)

| Param | Before | After | Why |
|---|---|---|---|
| `patience` | 15 | 30 | More headroom for noisy val loss |
| `dropout` | 0.5 | 0.25 | Less regularization for cross-subject regime |
| `max_norm_fc` | 0.25 | 0.5 | Allows FC logits to grow; stops fighting Adam |
| LR scheduler | none | ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5) | Escape plateaus |

All other hyperparams (lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=200, max_norm_conv=1.0) unchanged.

---

## Success Criteria

- Training loss decreases visibly beyond epoch 30 (not stuck at 0.693).
- `vanilla` test accuracy ≥ 55% (spec threshold from baseline spec, section "Success Criteria").
- MLflow run logs `dropout`, `max_norm_fc`, `lr_patience`, `lr_factor`.

---

## Non-Goals

- No hyperparameter search — these are principled one-shot adjustments.
- No architecture changes to EEGNet.
- No changes to data pipeline or splits.
