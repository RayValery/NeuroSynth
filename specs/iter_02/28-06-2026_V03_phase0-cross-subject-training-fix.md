# Phase 0 — Cross-Subject Training Fix

**Status:** Draft  
**Created:** 2026-06-28  
**Supersedes:** `28-06-2026_V02_phase0-eegnet-training-fixes.md`  
**Problem:** After two rounds of hyperparameter adjustment, vanilla EEGNet still achieves chance-level accuracy (49.9%) on cross-subject motor imagery. Training loss oscillates around log(2) ≈ 0.693 and never converges.

---

## Diagnosis Summary

| Finding | Evidence |
|---|---|
| Model can learn | Overfits 50 samples to 100% accuracy in 20 epochs (no constraints, no dropout) |
| Constraints don't block learning on small data | Same overfit test with max_norm_fc=0.5 still reaches 100% — FC norms pin at 0.5, loss still drops |
| `lr=1e-3` causes oscillation at scale | Train loss bounces 0.6911→0.6929 between epochs 30–40; never trends down monotonically across 3310 cross-subject samples |
| `ReduceLROnPlateau` timing conflicts with early stopping | LR decays at epoch best+10, early stopping fires at epoch best+30 — model gets only 20 epochs at the decayed LR before terminating |
| Root cause | High gradient variance across 74 subjects with partial cancellation; `lr=1e-3` step size is too large, causing the optimizer to overshoot and oscillate rather than converge |

---

## What We Are NOT Changing

- Architecture (EEGNet F1=8, D=2, F2=16) — model is correct and can learn
- Evaluation protocol — cross-subject remains correct for unconditional generative model validation
- Data pipeline — preprocessing and splits are correct
- Dropout=0.25, max_norm_fc=0.5, weight_decay=1e-4 — these are fine

---

## Changes

All changes are in `eval/train_baseline.py` only. `eval/eegnet.py` is untouched.

### 1. Lower `lr` default: `1e-3` → `1e-4`

The single most impactful change. Smaller gradient steps prevent overshooting on noisy cross-subject gradients. Convergence will be slower but monotonic.

### 2. Replace `ReduceLROnPlateau` with `CosineAnnealingLR(T_max=max_epochs)`

`ReduceLROnPlateau` is reactive — it decays only after a plateau is detected, which conflicts with early stopping timing. `CosineAnnealingLR` decays smoothly from `lr` to `min_lr` over `max_epochs` regardless of val loss. No timing interaction with early stopping.

Remove params: `lr_patience`, `lr_factor`.  
Keep: `min_lr` (used as `eta_min` in cosine schedule).

### 3. Increase `patience` default: `30` → `50`

Lower LR converges more slowly. 50 epochs of non-improvement gives the cosine schedule enough room to decay before stopping.

### 4. Increase `max_epochs` default (CLI): `200` → `300`

Matched to the slower lr. With cosine decay over 300 epochs, the LR reaches `min_lr` at the end of training rather than mid-run.

### 5. Update MLflow param logging

Remove `lr_patience`, `lr_factor`. Add `scheduler=cosine`.

---

## Hyperparameter table (final state after this spec)

| Param | V01 (original) | V02 (prev fix) | V03 (this spec) |
|---|---|---|---|
| `lr` | 1e-3 | 1e-3 | **1e-4** |
| `patience` | 15 | 30 | **50** |
| `max_epochs` | 200 | 200 | **300** |
| `dropout` | 0.5 | 0.25 | 0.25 |
| `max_norm_fc` | 0.25 | 0.5 | 0.5 |
| Scheduler | none | ReduceLROnPlateau(patience=10, factor=0.5) | **CosineAnnealingLR(T_max=max_epochs, eta_min=min_lr)** |
| `min_lr` | — | 1e-5 | 1e-5 |

---

## Success Criteria

- Training loss decreases monotonically (no oscillation) for at least 50 epochs
- `vanilla` test accuracy ≥ 55% (cross-subject lower bound from literature)
- If accuracy is still <55% after full run: escalate to architecture-level investigation (subject-adaptive BN, MAML-style meta-learning) — out of scope for Phase 0 baseline

---

## CLI after this change

```
python eval/train_baseline.py --data data/processed.npz --run vanilla
python eval/train_baseline.py --data data/processed.npz --run augmented
```

No new flags required. All changed values are new defaults.
