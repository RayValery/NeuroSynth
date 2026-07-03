# Phase 0 Step 4 (Probe) — Pooled-Real Augmentation: does population data dilute the within-subject spatial filter?

**Status:** Draft  
**Created:** 2026-07-03  
**Depends on:** Phase 0 Step 3b (vanilla 56.7%, augmented-2× 59.5%) — complete  
**Feeds into:** Phase 0 Step 4 (EEG diffusion model) — this probe decides whether the *population-first* rung is worth building, or whether to jump straight to subject-conditional generation  
**Related spec:** `specs/iter_04/03-07-2026_step4_eeg-diffusion-model.md`

---

## Executive Summary

Before spending compute on a population-level diffusion model, this probe answers a cheaper, prior question with **real data and the existing baseline harness**: *does adding label-consistent EEG epochs from other subjects to a subject's within-subject training set help or hurt classification of that subject's held-out run?*

The concern (well-grounded in this project's own history — cross-subject training died at ~52%, Euclidean Alignment made it worse, both because per-subject ERD spatial patterns don't align) is that population data pulls EEGNet's depthwise **spatial filter** toward a population-average pattern that is wrong for the target subject — the same spatial-cancellation that forced the pivot to within-subject LORO. If that mechanism dominates, a *population* diffusion model (which samples the same cross-subject distribution) is unlikely to help either.

Real foreign-subject epochs are the **best case** for population data — sharper than any generative approximation. So the logic is a one-way upper bound: **if real pooled data doesn't beat plain augmentation, population synthetic won't** → skip the population diffusion rung and go straight to subject-conditional generation. If it *does* help, the population diffusion model is worth building. The probe adds one run mode (`pooled_real`) to `train_baseline.py`, reuses the entire LORO evaluation and comparison pipeline, and needs no GPU.

---

## Goals

- Add a `pooled_real` run mode to `eval/train_baseline.py` that augments each within-subject LORO training fold with **N real, label-consistent epochs sampled from other subjects**, at N matched to the existing augmented baselines (30 / 60 / 90)
- Produce per-subject result JSONs directly comparable to `vanilla` and `augmented_offline`
- Extend `eval/compare_baselines.py` to plot `pooled_real` alongside the augmented curve and to run the decisive paired test: **`pooled_real` vs `augmented_offline` at matched N** (isolates the cost of spatial mismatch)
- Emit a clear go/no-go **decision** for the population-diffusion rung of Step 4

## Non-Goals

- **No diffusion model** — this probe uses only real data and the existing EEGNet/LORO harness
- **Not a leakage-controlled TSTR** — this is a *mechanism* probe, not the RR-3 gate; foreign epochs are extra labeled data, not synthetic
- **Not subject-conditional** — this does not test whether subject-*matched* synthetic helps (that is the diffusion spec's rung 3)
- No changes to EEGNet, the LORO split, the training schedule, or aggregation

---

## Background & Context

### The question this settles

EEGNet's discriminative power for left vs. right fist comes from its **depthwise spatial convolution** — a learned spatial filter that isolates the contralateral mu/beta ERD (left fist → right hemisphere ~C4; right fist → left ~C3). That topography differs subject-to-subject (neuroanatomy, electrode placement, BCI literacy). Pooling subjects forces one spatial filter to serve many incompatible topographies → the ~52% cross-subject failure recorded in this project's V01–V04.

The population-first diffusion plan (see related spec) generates *generic, subject-agnostic* epochs sampled from exactly that cross-subject distribution and mixes N of them into each subject's within-subject training set. The open worry: those generic samples drag the spatial filter back toward the population average and re-introduce cross-subject cancellation — just diluted by the presence of the subject's real data.

### Why this probe is a valid upper bound

We can test the *mechanism* — "does foreign-distribution data dilute the within-subject spatial filter?" — with **real** foreign epochs, no generative model required. Real individual patterns are *sharper and more valid* than anything a diffusion model would produce. Therefore:

- **If real pooled data ≤ plain augmentation → population synthetic will not do better.** Decisive: skip the population-diffusion rung, prioritize subject-conditional generation.
- **If real pooled data ≥ plain augmentation → population data genuinely helps** → the population diffusion model is worth building.

### The comparison that isolates spatial mismatch

At matched N, three arms differ only in *where the extra samples come from*:

| Arm | Extra samples per fold | Spatial pattern of extras |
|---|---|---|
| `vanilla` (N=0) | none | — |
| `augmented_offline` (n_copies=k) | k×|X_train| jittered copies | **same subject** (correct pattern + jitter) |
| `pooled_real` (matched) | k×|X_train| foreign epochs | **other subjects** (foreign patterns) |

`augmented_offline` and `pooled_real` add the *same number* of extra samples and train identical EEGNets for identical epochs; the only difference is same-subject vs. foreign spatial content. **The `augmented − pooled_real` gap at matched N is the price of spatial mismatch** — the exact quantity the diffusion worry is about.

### Predicted signatures (if the dilution hypothesis is true)

- `pooled_real` accuracy is flat or **declines as N grows** (more foreign data → more dilution)
- `pooled_real` sits significantly **below** `augmented_offline` at matched N
- Per-tier: harm concentrated in **Learning / Strong** responders (crisp patterns to dilute); **Data-limited** subjects may be neutral or slightly helped (variance-limited, not pattern-limited)

---

## Requirements

### Must Have

- `_create_pooled_real_set(X_train, y_train, foreign_X, foreign_y, n_copies, seed)` in `eval/train_baseline.py`:
  - Adds `n_copies × len(X_train)` foreign epochs, **class-balanced**, sampled without replacement from `foreign_*`
  - Real samples prepended unchanged (same convention as `_create_offline_augmented_set`)
  - Returns `(X_combined, y_combined)` with the **same total size** as the matched `augmented_offline` run
- `pooled_real` added to `valid_runs`, argparse `--run` choices, and the per-fold branch in `train()`
- Foreign pool built per fold from **all epochs of subjects other than the current subject** (`ds.subject_ids != subj`)
- Per-subject results written to `models/checkpoints/per_subject_results_pooled_real_{n_copies}x.json` (same schema as existing baselines)
- MLflow logging under experiment `phase0-baseline`, with `augmentation_mode="pooled_real"`, `n_copies`, `n_train_samples`, `pool_source="other_subjects"`

### Should Have

- `eval/compare_baselines.py` extended to:
  - Load `per_subject_results_pooled_real_{1,2,3}x.json` (N = 30/60/90)
  - Plot `pooled_real` as a distinct curve on `fig10_accuracy_vs_n.png`
  - Add a **paired `pooled_real` vs `augmented_offline` Wilcoxon at matched N** to the summary table (the spatial-mismatch test)
  - Include `pooled_real` columns and a `pooled_real − aug` delta in `subject_deltas.csv`
- Tier-breakdown figure (`fig11`) includes the `pooled_real` curve per tier

### Nice to Have

- `reports/phase0_pooled_real_probe_report.md` — results, per-tier signature, and the explicit go/no-go decision for the population-diffusion rung
- `--pool-source {other_subjects, all}` flag (default `other_subjects`) for a future ablation

---

## Architecture Overview

Reuses the entire existing pipeline. Only the per-fold training-set construction changes.

```
EEGDataset.load(processed.npz)          ds.X, ds.y, ds.subject_ids, ds.run_ids
        │
        ▼
within_subject_loro_split(...)          318 folds (106 subjects × runs {4,8,12})
        │
        │  for fold (subject S, held-out run R):
        │     X_tr, y_tr = S's other two runs   (~30 epochs)
        │     foreign_X  = ds.X[ds.subject_ids != S]   ← NEW: cross-subject pool
        │     foreign_y  = ds.y[ds.subject_ids != S]
        │           │
        │           ▼
        │     _create_pooled_real_set(X_tr, y_tr, foreign_X, foreign_y,
        │                             n_copies, seed)      ← NEW
        │           │  X_combined = [ X_tr(real) | N balanced foreign epochs ]
        │           ▼
        │     _train_fold(...)  →  _eval_on_test(... S's run R ...)   ← unchanged
        ▼
per_subject_results_pooled_real_{k}x.json      ← same schema as baselines
        │
        ▼
compare_baselines.py  →  fig10 / fig11 / subject_deltas.csv + Wilcoxon(pooled vs aug)
```

**Reused unchanged:** `EEGNet`, `_train_fold`, `_eval_on_test`, `within_subject_loro_split`, aggregation, MLflow experiment, JSON/confusion-matrix artifacts.
**New:** one sampling function + one run-mode branch + foreign-pool construction; comparison-plot additions.

---

## Data Models

### New function signature

```python
def _create_pooled_real_set(
    X_train: np.ndarray,     # (n_real, C, T) this fold's real training epochs (subject S)
    y_train: np.ndarray,     # (n_real,)
    foreign_X: np.ndarray,   # (n_foreign, C, T) all epochs of subjects != S
    foreign_y: np.ndarray,   # (n_foreign,)
    n_copies: int,           # match augmented_offline: adds n_copies * n_real foreign epochs
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Additive pooled-real augmentation. Adds N = n_copies * len(X_train) real
    epochs sampled (without replacement, class-balanced) from other subjects.
    Real samples are prepended unchanged so the combined set matches the size
    and class balance of the corresponding augmented_offline run.
    Returns (X_combined, y_combined) of length (1 + n_copies) * len(X_train).
    """
```

**Sampling rule (deterministic):**
- `rng = np.random.default_rng(seed)` with `seed = base_seed + subject*100 + fold_index` (identical convention to `_create_offline_augmented_set`)
- `N = n_copies * len(X_train)`; draw `ceil(N/2)` from `foreign` class-0 indices and `floor(N/2)` from class-1 indices, without replacement
- Concatenate `[X_train, foreign_sampled]`; labels likewise

### Result artifact (unchanged schema)

`models/checkpoints/per_subject_results_pooled_real_{n_copies}x.json`:
```json
[{"subject": 1, "mean_accuracy": 0.57, "fold_accuracies": [0.53, 0.60, 0.58]}, ...]
```

### MLflow parameters (added to the existing `base_params`)

```python
{
  "run_name": "pooled_real",
  "augmentation_mode": "pooled_real",
  "pool_source": "other_subjects",
  "n_copies": n_copies,                     # 1, 2, or 3
  "n_train_samples": 30 * (1 + n_copies),   # nominal, matches augmented_offline
  # ... all existing base_params (eval_protocol, max_epochs, lr, etc.) unchanged
}
# Metrics: test_accuracy_mean, test_accuracy_std, test_f1_mean (unchanged)
```

---

## Interface Definitions

### `eval/train_baseline.py` — CLI

```
python eval/train_baseline.py --data data/processed.npz --run pooled_real --n-copies 1   # +30
python eval/train_baseline.py --data data/processed.npz --run pooled_real --n-copies 2   # +60 (matches aug-2×)
python eval/train_baseline.py --data data/processed.npz --run pooled_real --n-copies 3   # +90
```

Changes:
- `--run` choices: add `pooled_real`
- `--n-copies` help text: note it also controls pooled-real N (`N = n_copies × per-fold real count ≈ 30/60/90`)

### `eval/compare_baselines.py` — CLI

Unchanged CLI; automatically picks up the new JSONs if present:
```
python eval/compare_baselines.py
```

---

## Implementation Plan

### Phase 1 — Run mode in `eval/train_baseline.py`

- [ ] Add `_create_pooled_real_set(...)` (signature and sampling rule above)
- [ ] Add `"pooled_real"` to `valid_runs` and to the argparse `--run` choices
- [ ] In `train()`, before the fold loop, keep `ds` in scope (already available) so the foreign pool can be built per fold
- [ ] In the per-fold loop, add a branch:
  ```python
  elif run_name == "pooled_real":
      foreign_mask = ds.subject_ids != subj
      X_tr, y_tr = _create_pooled_real_set(
          X_tr, y_tr,
          foreign_X=ds.X[foreign_mask], foreign_y=ds.y[foreign_mask],
          n_copies=n_copies, seed=seed + subj * 100 + i,
      )
      fold_augmentor = None
  ```
- [ ] Extend `base_params` block: when `run_name == "pooled_real"`, set `augmentation_mode="pooled_real"`, `pool_source="other_subjects"`, `n_copies`, `n_train_samples`
- [ ] JSON stem: `per_subject_results_pooled_real_{n_copies}x`; confusion-matrix stem `confusion_matrix_pooled_real_{n_copies}x`
- [ ] Assertion inside `_create_pooled_real_set`: `subject S never appears in the sampled foreign set` is guaranteed by construction (pool masked on `!= subj`); assert no NaN, correct shapes, and combined length == `(1+n_copies)*len(X_train)`

### Phase 2 — Comparison in `eval/compare_baselines.py`

- [ ] Add candidates to `load_all_results`:
  `("pooled_1x", ".../per_subject_results_pooled_real_1x.json", 30)`, `_2x → 60`, `_3x → 90`
- [ ] Add a `_wilcoxon_paired(a: RunResult, b: RunResult, alternative)` helper (paired signed-rank on shared subjects) for **pooled_real vs augmented at matched N**
- [ ] Extend `print_summary_table`: add pooled rows with Δ-vs-vanilla and a new "Δ vs aug (p)" column at matched N
- [ ] Extend `plot_accuracy_vs_n`: add the pooled_real curve (color `C_PURPLE`) at N=30/60/90, anchored at vanilla N=0
- [ ] Extend `plot_tier_breakdown_vs_n`: add pooled_real curve per tier
- [ ] Extend `save_delta_csv`: add `pooled_{k}x_acc`, `pooled_{k}x_delta` (vs vanilla) and `pooled_{k}x_minus_aug` columns

### Phase 3 — Run & decide

- [ ] Run all three: `--run pooled_real --n-copies {1,2,3}` (reuse existing device/seed defaults)
- [ ] `python eval/compare_baselines.py` → refreshed fig10/fig11/CSV + Wilcoxon table
- [ ] Write `reports/phase0_pooled_real_probe_report.md` with the decision (below)

---

## Error Handling

- **`processed.npz` missing** — existing `sys.exit` with rebuild command (unchanged path)
- **Foreign pool too small for a class** — cannot occur (≈2,350 epochs/class available vs. ≤45 needed); still, if `N_per_class > available`, fall back to sampling **with** replacement and log a warning
- **Class imbalance in a fold's real data** — irrelevant: foreign additions are balanced independently; combined balance stays ≈ vanilla
- **Comparison script run before JSONs exist** — `load_all_results` already skips missing files silently

---

## Testing Strategy

Manual verification (research code — consistent with the baseline scripts, no automated suite):

**Sampling correctness:**
```python
import numpy as np
from eval.train_baseline import _create_pooled_real_set
X_tr = np.random.randn(30, 64, 801).astype("float32"); y_tr = np.array([0,1]*15)
fX  = np.random.randn(4000, 64, 801).astype("float32"); fY = np.array([0,1]*2000)
Xc, yc = _create_pooled_real_set(X_tr, y_tr, fX, fY, n_copies=2, seed=142)
assert Xc.shape == (90, 64, 801)              # (1+2)*30
assert np.array_equal(Xc[:30], X_tr)          # real prepended unchanged
assert np.bincount(yc[30:]).tolist() == [30, 30]  # added block class-balanced
Xc2, _ = _create_pooled_real_set(X_tr, y_tr, fX, fY, n_copies=2, seed=142)
assert np.allclose(Xc, Xc2)                   # deterministic under seed
```

**Size parity with augmented_offline:** for the same `n_copies`, `len(pooled_combined) == len(augmented_combined)` (both `(1+n_copies)*len(X_train)`), so the two arms train on identically sized sets.

**End-to-end smoke:** `--run pooled_real --n-copies 1 --max-epochs 20` completes, writes `per_subject_results_pooled_real_1x.json` with 106 entries.

---

## Success Criteria & Decision Rule

**Probe complete when:** all three `pooled_real` runs finish, JSONs written, `compare_baselines.py` produces the pooled curve + the pooled-vs-augmented Wilcoxon table, and the report records the decision.

**Decision (drives Step 4 direction):**

- **Dilution CONFIRMED → skip the population-diffusion rung; go subject-conditional.**
  Triggered if any of:
  - `pooled_real` mean ≤ `vanilla` (56.7%) at every N, or **declines** as N grows, **or**
  - `pooled_real` is significantly **below** `augmented_offline` at matched N (one-sided paired Wilcoxon, p < 0.05, meaningful absolute gap)

  Interpretation: adding foreign-distribution data hurts the within-subject spatial filter — the same mechanism as the ~52% cross-subject result. A *population* diffusion model would inherit this. Rewrite `specs/iter_04/03-07-2026_step4_eeg-diffusion-model.md` to lead with subject-conditional generation (and its subject-holdout evaluation), and demote/skip the population rung.

- **Dilution REFUTED → build the population diffusion model (rung 2 as specced).**
  Triggered if `pooled_real` > `vanilla` and is statistically **indistinguishable from or above** `augmented_offline` at matched N.
  Interpretation: label-consistent cross-subject diversity is not harmful within-subject; a population generator is worth training as the clean go/no-go gate.

- **Ambiguous (small, non-significant differences):** proceed with the population diffusion model but treat the population rung explicitly as a *control* whose most likely value is a clean negative that motivates rung 3.

**Report must include:** the accuracy-vs-N trend for all three arms, the pooled-vs-augmented paired test at N=60, and the per-tier signature (is harm concentrated in Learning/Strong as predicted?).

---

## Open Questions

- [ ] Should foreign epochs be drawn from *all* runs of other subjects (current default) or run-matched to the fold's training runs? Default `other_subjects`/all-runs keeps it simplest and maximal-diversity; a run-matched ablation is a Nice-to-Have, not required for the decision.
- [ ] Report as a standalone probe report, or fold the result into `reports/phase0_baseline_report.md`? (Leaning standalone, to keep the decision auditable.)
