# Phase 0 Step 5 — TSTR Harness & RR-3 Go/No-Go Gate

**Status:** Draft
**Created:** 2026-07-07
**Author:** Claude
**Depends on:** Phase 0 Step 3b (additive augmented baseline, 59.5% at 2×) — complete; Phase 0 Step 4 (three run-holdout diffusion models) — complete and validated
**Feeds into:** Phase 0 exit decision (build vs. pivot to subject-conditional generation) and the preprint

---

## Executive Summary

Phase 0 Step 5 runs the central experiment of the entire project: **Train-on-Synthetic, Test-on-Real (TSTR)** — does augmenting a subject's tiny real training fold with *synthetic* epochs from the population diffusion models beat augmenting it with naive *signal augmentation*? Every LORO fold's real training set is extended with N synthetic samples drawn from the run-holdout diffusion model that never saw that fold's test run, EEGNet is trained on real+synthetic, and accuracy is measured on the real held-out run. The result is compared, per subject, against the two established reference points: the **vanilla baseline (56.7% ± 10.8%)** and the **additive augmented baseline (59.5% ± 11.2% at N=60)**.

The go/no-go gate (**RR-3**) passes if synthetic-augmented training at N=60 exceeds the 59.5% augmented ceiling by a margin that is both statistically significant (one-sided Wilcoxon signed-rank across the 106 subjects, p < 0.05) and practically meaningful (> +1% absolute). This spec adds one function to `eval/train_baseline.py` (`_load_synthetic_set`, the drop-in replacement for `_create_offline_augmented_set`), a Modal launcher, and the comparison/verdict logic in `eval/compare_baselines.py`. It changes nothing about the EEGNet model, the LORO protocol, or the diffusion models.

---

## Goals

- Implement the `tstr` run mode in `eval/train_baseline.py`: additive synthetic augmentation (real + N synthetic) at N = {30, 60, 90}, routed per fold to the diffusion model whose `held_out_run` equals that fold's test run.
- Produce per-subject result JSONs (`per_subject_results_tstr_{1,2,3}x.json`) in the **identical format** as the existing baselines, so they slot directly into `compare_baselines.py`.
- Extend `eval/compare_baselines.py` to load the TSTR results, plot the real TSTR curve alongside the augmented curve, run the **paired TSTR-vs-augmented Wilcoxon test**, and print the RR-3 go/no-go verdict.
- Run the three TSTR passes on Modal (cloud GPU), mirroring the Step 4 deployment, and pull the result JSONs back to `models/checkpoints/`.
- Deliver a Step 5 report (`reports/phase0_tstr_report.md`) documenting the result, whichever way the gate lands.

## Non-Goals

- **Retraining or modifying the diffusion models or EEGNet.** Step 5 consumes the frozen Step 4 checkpoints and the existing classifier unchanged.
- **Re-running the vanilla / augmented baselines.** Their per-subject JSONs already exist and were produced by this same harness at `seed=42`; they are reused directly as paired comparators.
- **Pure-synthetic TSTR (train on synthetic only).** The gate is the additive comparison; a standalone-quality diagnostic is out of scope (may be added later).
- **Subject-conditional generation (Phase 1).** The population models are subject-agnostic by design.
- **The `pooled_real` probe.** It was a pre-Step-4 decision gate; the population models are now built and validated, so it is retrospective/optional.

---

## Background & Context

### The TSTR test and why it is the whole point

NeuroSynth's commercial hypothesis is that a BCI vendor with only ~5 minutes of calibration data per new user can buy synthetic EEG to augment that data and get a better personal classifier without collecting more real data. Phase 0 validates that hypothesis with a strict, leakage-free experiment:

```
vanilla   : EEGNet on real only               → 56.7%   (established, Step 3a)
augmented : EEGNet on real + N augmented copies→ 59.5%@60 (established, Step 3b)
TSTR      : EEGNet on real + N synthetic       → ???      (this spec)

Go criterion (RR-3): TSTR@60 > 59.5%, significantly and meaningfully.
```

If synthetic data cannot beat naive signal augmentation (Gaussian noise, time-shift, channel dropout, crop), the population generator provides no practical value at this rung, and the project pivots directly to subject-conditional generation. If it wins, the population rung is validated and becomes the first result of the preprint.

### What Step 4 delivered (the inputs to this spec)

Three population DDPMs are trained and validated (`reports/phase0_baseline_report.md` §9):

| Checkpoint | `held_out_run` | Trained on | Serves folds testing on |
|---|---|---|---|
| `models/checkpoints/diffusion_holdout_run4.pt`  | 4  | runs 8 & 12 | run 4  |
| `models/checkpoints/diffusion_holdout_run8.pt`  | 8  | runs 4 & 12 | run 8  |
| `models/checkpoints/diffusion_holdout_run12.pt` | 12 | runs 4 & 8  | run 12 |

Each passed the TSTR-readiness gates: samples overlap real PSD across the 8–30 Hz mu/beta band, `nn_distance ≈ nn_distance_real ≈ 290` (clean, non-memorizing), and the three are mutually comparable (`psd_divergence` spread 0.02). The public interface is `EEGDiffusionModel` (`eval/diffusion.py`):

```python
model.held_out_run                              # int in {4, 8, 12}
model.generate(n, subject_id, class_label=None, seed=None) -> np.ndarray  # (n, 64, 801) f32, z-scored
model.labels(n) -> np.ndarray                   # (n,) int64, balanced [0,1,0,1,...]
```

`subject_id` is accepted but ignored (population model). Generation returns per-epoch z-scored output in the same space as `data/processed.npz`.

### Why leave-one-run-out cross-fitting keeps the gate clean

Step 5's LORO tests each of a subject's three runs (4, 8, 12) in turn. For any fold testing on run *R*, the synthetic augmentation is drawn from `diffusion_holdout_run{R}` — the model that **never saw run *R* of any subject**. So the synthetic data provably carries no test-run information, the identical cleanliness standard as the augmented baseline (which only ever jitters a fold's *training* runs). Because every subject is evaluated leakage-free, the comparison is a **direct, full-cohort, per-subject paired test** against the existing 56.7% / 59.5% numbers — no sub-selection, no loss of statistical power. A routing assertion guarantees a wiring mistake cannot silently reintroduce leakage.

### Why pre-generated per-model banks (compute decision)

The models are subject-agnostic, so a fold's synthetic samples do not depend on the fold's subject — only on the fold's *test run* (which selects the model). Generating fresh samples inside each of the 318 folds would repeat 50-step DDIM sampling ~318× for no correctness gain. Instead, each of the three models generates a large bank **once** (e.g. 1024 samples/class), and each fold draws its N samples from the bank keyed by its test run. This is valid (population, leakage-free by run), deterministic, and reduces generation to minutes.

---

## Requirements

### Must Have

- `tstr` run mode in `eval/train_baseline.py`, additive design: for each fold, `X_combined = X_real ⊕ X_synth`, `y_combined = y_real ⊕ y_synth`, where `X_synth` is drawn from `bank[test_run]` and matches the real fold's per-class counts × `n_copies`.
- `_load_synthetic_set(...)` — the drop-in replacement for `_create_offline_augmented_set(...)`, with a routing assertion `bank.held_out_run == test_run`.
- `_build_synthetic_banks(...)` — load the three checkpoints once, generate a class-balanced bank per model, return a dict keyed by held-out run.
- N parity with the augmented curve: `n_copies ∈ {1,2,3}` → N = `n_copies × len(y_train_fold)` extra synthetic samples (same count the augmented arm added at the same `n_copies`).
- Per-subject JSON output `per_subject_results_tstr_{n_copies}x.json` in the existing schema (`[{subject, mean_accuracy, fold_accuracies}]`).
- `eval/compare_baselines.py` extended to load `tstr_{1,2,3}x`, plot the TSTR curve on fig10, run the paired **TSTR-vs-augmented-2×** one-sided Wilcoxon, and print the RR-3 verdict (PASS/FAIL with the numbers).
- Modal launcher `modal_train_tstr.py`: mount the data and checkpoints volumes, build banks, run the three TSTR passes on GPU, write the JSONs back to the volume.
- Leakage guard: assert no fold ever draws from a model whose `held_out_run != test_run`.

### Should Have

- MLflow logging under a `phase0-tstr` experiment (one run per N), mirroring the baseline params/metrics so results are traceable.
- Paired per-subject delta figure (`fig12_tstr_vs_aug_delta.png`): TSTR@60 − augmented@60 per subject, sorted, colored by tier.
- Extended `subject_deltas.csv` with `tstr_{1,2,3}x_acc` / `_delta` columns.
- `reports/phase0_tstr_report.md` — result write-up (methodology, gate outcome, tier analysis), regardless of pass/fail.
- Deterministic bank draws: a fold-specific seed (`seed + subject*100 + fold_idx`) mirroring the augmented seeding, so reruns are identical.

### Nice to Have

- A `--bank-size` knob and a sensitivity check that the verdict is stable across two bank seeds (guards against a lucky/unlucky draw).
- Secondary reporting of TSTR-vs-vanilla at each N (not the gate, but informative).
- A combined fig10 showing both the augmented and TSTR curves with the 59.5% gate line annotated.

---

## Architecture Overview

### System context

```
models/checkpoints/diffusion_holdout_run{4,8,12}.pt   data/processed.npz
        │  (frozen Step 4 weights, EEGDiffusionModel)          │  X,y,subject_ids,run_ids
        ▼                                                      │
┌──────────────────────────────┐                              │
│ _build_synthetic_banks()     │  generate ONCE per model     │
│  bank[4], bank[8], bank[12]  │  (1024/class, class-balanced)│
└──────────────┬───────────────┘                              │
               │ banks keyed by held-out run                  │
               ▼                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ eval/train_baseline.py  --run tstr  --n-copies {1,2,3}            │
│                                                                    │
│  for each of 318 LORO folds (subject × test_run):                 │
│     X_tr,y_tr,test_run = fold                                      │
│     X_comb,y_comb = _load_synthetic_set(                          │
│         X_tr,y_tr, test_run, n_copies, banks[test_run], seed)     │
│         └─ assert banks[test_run].held_out_run == test_run        │
│     train EEGNet on X_comb ; evaluate on real held-out run        │
│                                                                    │
│  aggregate per subject → per_subject_results_tstr_{n}x.json       │
└──────────────┬───────────────────────────────────────────────────┘
               │ 3 JSONs (N=30/60/90)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ eval/compare_baselines.py                                         │
│  load {vanilla, aug_1x/2x/3x, tstr_1x/2x/3x}                      │
│  paired Wilcoxon: tstr_2x  >  aug_2x   (RR-3 gate)                │
│  paired Wilcoxon: tstr_Nx  >  vanilla  (secondary)               │
│  fig10 (both curves + gate line), fig12 (paired delta), CSV       │
│  print PASS/FAIL verdict                                           │
└──────────────────────────────────────────────────────────────────┘
```

### What's new vs. reused

| Component | Status |
|---|---|
| EEGNet, `_train_fold`, `_eval_on_test`, LORO split, per-subject aggregation | **reused, unchanged** |
| `EEGDiffusionModel`, the three checkpoints | **reused, frozen** |
| `_create_offline_augmented_set` | **reused for the augmented baseline; `_load_synthetic_set` mirrors it** |
| `_build_synthetic_banks`, `_load_synthetic_set`, `tstr` run mode | **new (train_baseline.py)** |
| `modal_train_tstr.py` | **new** |
| TSTR loading, paired gate test, verdict, fig12 | **new (compare_baselines.py)** |

### Cloud deployment (Modal)

Both required inputs already live on Modal volumes from Step 4: `neurosynth-data` (`processed.npz`) and `neurosynth-checkpoints` (the three `diffusion_holdout_run{R}.pt`). The launcher mounts both, uploads the current `eval/` + `data/` source, builds the banks on-GPU, runs the three TSTR passes, and writes `per_subject_results_tstr_{1,2,3}x.json` back to the checkpoints volume for download. To guarantee the corrected sampler is used, the launcher ships the current `eval/diffusion.py` (with `clip_denoised` + z-score); the checkpoint *weights* on the volume are identical to the local copies, and `generate()` reads weights, not the stored scalar metadata.

---

## Data Models

### Synthetic bank

```python
@dataclass
class SyntheticBank:
    held_out_run: int          # 4, 8, or 12 — the run the source model never saw
    X: np.ndarray              # (bank_size, 64, 801) float32, z-scored
    y: np.ndarray              # (bank_size,) int64 in {0,1}, class-balanced

# Banks are keyed by held-out run == the fold test_run they serve:
banks: dict[int, SyntheticBank]   # {4: ..., 8: ..., 12: ...}
```

`bank_size` default 2048 (1024/class). Max draw per fold at 3× is `3 × ~30 = 90` (≤45/class), far below 1024/class, so per-fold draws are without replacement.

### Per-subject result JSON (unchanged schema, new filename)

```jsonc
// models/checkpoints/per_subject_results_tstr_2x.json
[
  {"subject": 1, "mean_accuracy": 0.611, "fold_accuracies": [0.60, 0.667, 0.567]},
  ...
]
```

### MLflow parameters logged per TSTR run

```python
{
    "experiment": "phase0-tstr",
    "run_name": "tstr",
    "augmentation_mode": "synthetic",
    "n_copies": n_copies,                # 1, 2, or 3
    "n_extra_per_fold": "n_copies * len(train)",
    "eval_protocol": "within_subject_loro",
    "bank_size": 2048,
    "guidance_scale": 3.0, "ddim_steps": 50, "clip_denoised": 4.0,
    "diffusion_checkpoints": "diffusion_holdout_run{4,8,12}.pt",
    "n_subjects": 106, "n_folds_total": 318, "seed": 42,
    # + the shared EEGNet params (lr, dropout, max_epochs, ...)
}
# Metrics: test_accuracy_mean, test_accuracy_std, test_f1_mean
```

---

## Interface Definitions

### `_build_synthetic_banks` — generate one bank per model (new, train_baseline.py)

```python
def _build_synthetic_banks(
    checkpoint_dir: Path,
    device: torch.device,
    bank_size: int = 2048,
    seed: int = 42,
    runs: tuple[int, ...] = (4, 8, 12),
) -> dict[int, SyntheticBank]:
    """
    Load each diffusion_holdout_run{R}.pt once and generate a class-balanced bank
    of `bank_size` epochs. Returns {R: SyntheticBank(held_out_run=R, X, y)}.
    Asserts each loaded model's held_out_run matches its key.
    Missing checkpoint -> sys.exit with the training command to produce it.
    """
```

### `_load_synthetic_set` — drop-in replacement for `_create_offline_augmented_set` (new)

```python
def _load_synthetic_set(
    X_real: np.ndarray,          # (n_real, 64, 801)
    y_real: np.ndarray,          # (n_real,)
    test_run: int,               # this fold's held-out run
    n_copies: int,               # 1/2/3 → N = n_copies * n_real extra synthetic
    bank: SyntheticBank,         # banks[test_run]
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Append n_copies synthetic epochs per real sample, matched by class, drawn
    (without replacement, fold-seeded) from `bank`. Real samples are prepended
    unchanged so Block 2 anchors on real signal — identical structure to
    _create_offline_augmented_set.

    Guard (must not be removed): asserts bank.held_out_run == test_run, so a fold
    can never be augmented by a model that saw its test run.
    Returns (X_combined, y_combined) with len = n_real * (1 + n_copies).
    """
```

Per-class matching: for each class `c`, add `n_copies × count(y_real == c)` synthetic epochs of class `c`. This reproduces the exact per-fold sample count and class distribution the augmented arm produced at the same `n_copies`, making the two curves comparable sample-for-sample.

### `eval/train_baseline.py` — CLI additions

```
python eval/train_baseline.py --run tstr --n-copies 1   # N = +30
python eval/train_baseline.py --run tstr --n-copies 2   # N = +60  (gate)
python eval/train_baseline.py --run tstr --n-copies 3   # N = +90

New / reused arguments:
    --run tstr                 New run mode (adds to vanilla|augmented|augmented_offline)
    --n-copies {1,2,3}         Reused: synthetic copies per real sample
    --diffusion-dir PATH       Where the 3 checkpoints live (default: models/checkpoints)
    --bank-size INT            Bank per model (default: 2048)
    --experiment STR           MLflow experiment (default: phase0-tstr for --run tstr)
```

Routing: `train()` calls `_build_synthetic_banks(...)` once before the LORO loop; inside the loop, `tstr` mode calls `_load_synthetic_set(X_tr, y_tr, split["test_run"], n_copies, banks[split["test_run"]], fold_seed)`. JSON stem becomes `per_subject_results_tstr_{n_copies}x`.

### `eval/compare_baselines.py` — additions

```python
# load_all_results(): add candidates
("tstr_1x", checkpoint_dir/"per_subject_results_tstr_1x.json", 30),
("tstr_2x", checkpoint_dir/"per_subject_results_tstr_2x.json", 60),
("tstr_3x", checkpoint_dir/"per_subject_results_tstr_3x.json", 90),

def _wilcoxon_paired(a: RunResult, b: RunResult) -> tuple[float, float, int]:
    """One-sided Wilcoxon: is a > b across shared subjects?
       Returns (mean_delta_pct, p_value, n_subjects)."""

def print_gate_verdict(results) -> None:
    """RR-3 gate: tstr_2x vs aug_2x.
       PASS iff mean_delta > +1.0% AND one-sided Wilcoxon p < 0.05.
       Prints the numbers and PASS/FAIL. Also prints tstr_Nx vs vanilla."""
```

`plot_accuracy_vs_n` replaces the TSTR placeholder lane (currently `ax.plot([], [], ...)`, lines ~180–182) with the real `tstr_1x/2x/3x` curve; a horizontal annotation marks the 59.5% augmented gate line.

---

## Implementation Plan

### Phase 1 — Harness: `eval/train_baseline.py`

- [ ] Add `SyntheticBank` dataclass (or a small local structure).
- [ ] Implement `_build_synthetic_banks(checkpoint_dir, device, bank_size, seed)`:
  - `from eval.diffusion import EEGDiffusionModel`
  - For `R in (4,8,12)`: load `diffusion_holdout_run{R}.pt`; assert `model.held_out_run == R`; `X = model.generate(bank_size, subject_id=0, seed=seed+R)`; `y = model.labels(bank_size)`; store `SyntheticBank(R, X, y)`.
  - `sys.exit` with the Step 4 train command if a checkpoint is missing.
- [ ] Implement `_load_synthetic_set(X_real, y_real, test_run, n_copies, bank, seed)`:
  - `assert bank.held_out_run == test_run` (leakage guard).
  - Per class `c`: need `n_copies * count(y_real==c)` synthetic of class `c`; select those indices from `bank` via a fold-seeded RNG, without replacement.
  - Prepend real unchanged; return `(X_combined, y_combined)`.
- [ ] Extend `valid_runs` and the CLI `--run` choices with `tstr`; add `--diffusion-dir`, `--bank-size`.
- [ ] In `train()`: if `run_name == "tstr"`, build banks once before the subject loop; in the fold loop select `_load_synthetic_set(...)` (mirroring the `augmented_offline` branch), set `fold_augmentor=None`.
- [ ] JSON/CM stems: `per_subject_results_tstr_{n_copies}x`, `confusion_matrix_tstr_{n_copies}x`.
- [ ] MLflow: default experiment `phase0-tstr` for `--run tstr`; log the params in Data Models.
- [ ] Default `--experiment` handling so `tstr` logs separately from `phase0-baseline`.

### Phase 2 — Modal launcher: `modal_train_tstr.py`

- [ ] App + image (torch/numpy/scipy/scikit-learn/matplotlib/mlflow; `GIT_PYTHON_REFRESH=quiet`; `NEUROSYNTH_GPU` env; add local `eval/` and `data/` code, exclude `*.npz`).
- [ ] Mount volumes: `neurosynth-data` at `/vol/data`, `neurosynth-checkpoints` at `/vol/checkpoints`.
- [ ] (Optional, for cleanliness) `modal volume put` the three corrected local checkpoints before running.
- [ ] `train_tstr(n_copies)` remote function: `chdir` into the output volume; call `train(data_path=/vol/data/processed.npz, run_name="tstr", checkpoint_dir=/vol/checkpoints, n_copies=..., ...)`; `ckpt_volume.commit()`.
- [ ] `local_entrypoint`: spawn `train_tstr` for `n_copies in (1,2,3)` in parallel; print the three result means; document the `modal volume get` pull commands for the JSONs.
- [ ] Timeout 6 h; `--detach` documented (EEGNet is fast, but bank generation + 3×318 folds should still run detached).

### Phase 3 — Comparison & gate: `eval/compare_baselines.py`

- [ ] Add `tstr_1x/2x/3x` to `load_all_results` candidates.
- [ ] Implement `_wilcoxon_paired(a, b)` (one-sided `a > b`, shared subjects) and `print_gate_verdict(results)` (RR-3: `tstr_2x` vs `aug_2x`; PASS iff Δ>+1% and p<0.05; plus `tstr_Nx` vs vanilla).
- [ ] Replace the fig10 TSTR placeholder with the real `tstr` curve; annotate the 59.5% gate line; keep the augmented curve.
- [ ] Add `plot_tstr_vs_aug_delta(results, out)` → `fig12_tstr_vs_aug_delta.png` (per-subject Δ at N=60, sorted, tier-colored).
- [ ] Extend `save_delta_csv` with `tstr_{1,2,3}x_acc`/`_delta` columns.
- [ ] `main()`: call `print_gate_verdict` after the summary table.

### Phase 4 — Run & collect (Modal)

- [ ] Ensure the three checkpoints are on `neurosynth-checkpoints` (present from Step 4; optionally refresh with corrected local copies).
- [ ] `modal run --detach modal_train_tstr.py`.
- [ ] `modal volume get neurosynth-checkpoints per_subject_results_tstr_{1,2,3}x.json models/checkpoints/`.
- [ ] `python eval/compare_baselines.py` → summary table, gate verdict, fig10/fig11/fig12, CSV.

### Phase 5 — Report & records

- [ ] `reports/phase0_tstr_report.md`: methodology, per-N results table (TSTR vs vanilla vs augmented), the RR-3 verdict, tier breakdown, and the resulting decision (validate population rung vs pivot to subject-conditional).
- [ ] Update `reports/phase0_baseline_report.md` §12.3 status (Step 5 complete) and the header.
- [ ] Update project memory with the gate outcome.

---

## Error Handling

- **Missing diffusion checkpoint**: `_build_synthetic_banks` `sys.exit`s with the exact Step 4 train/pull command for the missing run.
- **Routing mismatch** (`bank.held_out_run != test_run`): `AssertionError` — hard stop; this can only mean a wiring bug and must never be silently tolerated (it would reintroduce leakage).
- **Bank too small for the requested draw** (`n_copies × per-class count > bank per-class size`): raise a clear `ValueError` telling the user to raise `--bank-size` (should not happen at default 1024/class vs max 45).
- **Test run not in {4,8,12}**: `KeyError` on `banks[test_run]` — surfaces immediately; the LORO split should only ever yield these runs.
- **Missing baseline JSONs in `compare_baselines`**: skip gracefully (existing behavior); the gate verdict prints "cannot evaluate — need vanilla, aug_2x, and tstr_2x" if any is absent.
- **NaN Wilcoxon** (fewer than ~10 shared subjects, or identical arrays): report `n/a` rather than a spurious p-value (existing `_wilcoxon_vs_vanilla` pattern).
- **Modal**: reuse the Step 4 hardening — `NEUROSYNTH_GPU` env (no `with_options`), no `volume.reload()` with cwd inside the volume, 6 h timeout, background-commit safety.

---

## Testing Strategy

Manual verification (research code — no automated suite required), plus a few cheap asserts.

**Leakage / routing (the critical correctness check):**
```python
banks = _build_synthetic_banks(Path("models/checkpoints"), torch.device("cpu"), bank_size=64)
for R in (4, 8, 12):
    assert banks[R].held_out_run == R
# every fold must route to its own held-out model
for split in within_subject_loro_split(...):
    assert banks[split["test_run"]].held_out_run == split["test_run"]
```

**`_load_synthetic_set` shape / balance / determinism:**
```python
Xc, yc = _load_synthetic_set(X_tr, y_tr, test_run=8, n_copies=2, bank=banks[8], seed=123)
assert len(yc) == len(y_tr) * 3                       # real + 2×
assert np.array_equal(Xc[:len(y_tr)], X_tr)           # real prepended unchanged
# per-class parity with augmented structure
for c in (0, 1):
    assert (yc == c).sum() == (y_tr == c).sum() * 3
Xc2, yc2 = _load_synthetic_set(X_tr, y_tr, 8, 2, banks[8], seed=123)
assert np.allclose(Xc, Xc2)                           # fold-seeded determinism
```

**Smoke run (few subjects, few epochs):**
```bash
python eval/train_baseline.py --run tstr --n-copies 2 --max-epochs 20   # subset for a fast check
# expect: banks built for 4/8/12, folds route correctly, JSON written
```

**Gate logic unit check:** feed `print_gate_verdict` synthetic `RunResult`s where `tstr_2x` is (a) +2% and clearly significant → PASS, (b) +0.3% → FAIL (margin), (c) +2% but p=0.2 → FAIL (significance); confirm the verdict text.

**Edge cases to cover:** a subject/fold with an imbalanced real fold (16/14) draws synthetic in the same 16:14 × n_copies proportion; `n_copies=3` at the largest fold stays within bank capacity.

---

## Security Considerations

Research code, local/'cloud-compute only, no untrusted input. Two integrity concerns rather than security per se:

- **Leakage integrity** is enforced by the `bank.held_out_run == test_run` assertion — the single most important invariant in this spec.
- **Checkpoint trust**: `torch.load(..., weights_only=False)` is used on our own checkpoints only (same as Step 4). Modal runs on the user's authenticated account and writes only to the project's own volumes.

---

## Success Criteria

**Implementation complete when:**
- `--run tstr --n-copies {1,2,3}` runs end-to-end, builds the three banks, routes every fold to its held-out model (assertion active), and writes `per_subject_results_tstr_{1,2,3}x.json` in the standard schema.
- `_load_synthetic_set` passes the shape/balance/determinism/routing checks above.
- `compare_baselines.py` loads all nine runs (vanilla, aug ×3, tstr ×3, + online reference), prints the summary table, the RR-3 verdict, and saves fig10 (both curves), fig11, fig12, and the extended CSV.

**Experiment complete when:**
- All three TSTR passes finish on Modal; the three JSONs cover all 106 subjects × 3 folds (318 total), leakage-free.
- The RR-3 verdict is computed: **PASS** iff `mean(tstr_2x) − mean(aug_2x) > +1.0%` **and** one-sided Wilcoxon (`tstr_2x > aug_2x`, paired across the 106 subjects) `p < 0.05`; otherwise **FAIL**.
- `reports/phase0_tstr_report.md` documents the outcome and the resulting decision either way.

**Decision output (the actual deliverable):**
- **PASS** → the population synthetic rung is validated; it becomes the first preprint result, and Phase 1 (subject-conditional) is pursued as an *improvement*, not a *rescue*.
- **FAIL** → population synthetic does not beat augmentation at this rung; pivot directly to subject-conditional generation (addresses the Block 2 spatial-filter bottleneck, `phase0_baseline_report.md` §10.5), and report the negative result honestly.

---

## Compute & Time Estimates

Per TSTR pass = the cost of one existing baseline (318 folds × 500 full-batch epochs of a 2,930-param EEGNet) plus one-time bank generation (3 models × 2048 samples × 50 DDIM steps).

| Hardware | Bank gen (once) | Per TSTR pass | All three passes |
|---|---|---|---|
| NVIDIA A10G (Modal) | ~2–4 min | ~15–30 min | ~1 h (parallel) |
| Apple M-series (MPS) | ~5–10 min | ~2.5 h | ~7–8 h |

Modal is the chosen path: one `modal run --detach modal_train_tstr.py` spawns the three passes in parallel on A10G, ~1 h wall-clock, ~$1–2. Cost is dominated by the many short EEGNet fold-trainings, not the (tiny) diffusion sampling.

---

## Context for Handoff

### Repository state at time of writing

```
eval/
  eegnet.py              ← unchanged
  augmentations.py       ← unchanged
  dataset.py             ← within_subject_loro_split (unchanged)
  diffusion.py           ← EEGDiffusionModel (frozen; clip_denoised + z-scored generate)
  train_diffusion.py     ← Step 4 trainer (unchanged here)
  train_baseline.py      ← ADD: _build_synthetic_banks, _load_synthetic_set, tstr mode
  compare_baselines.py   ← ADD: tstr loading, paired gate, verdict, fig12
modal_train_diffusion.py ← Step 4 launcher (reference for modal_train_tstr.py)
modal_train_tstr.py      ← DOES NOT EXIST (Phase 5 creates this)
models/checkpoints/
  diffusion_holdout_run{4,8,12}.pt          ← frozen inputs
  per_subject_results_vanilla.json          ← 56.7% (paired comparator)
  per_subject_results_augmented_2x.json     ← 59.5% (RR-3 target)
  per_subject_results_tstr_{1,2,3}x.json    ← DOES NOT EXIST (Phase 5 produces)
data/processed.npz       ← 4748×64×801, run_ids ∈ {4,8,12}
```

### Constants

```python
SAMPLING_RATE=160; N_CHANNELS=64; N_TIMES=801; N_CLASSES=2
N_SUBJECTS=106; N_FOLDS=318; RUNS=(4,8,12); SEED=42
VANILLA=56.7; AUG_2X=59.5     # reference means (percent)
GATE_MARGIN=1.0; GATE_ALPHA=0.05
```

### The one invariant that must never break

```python
assert bank.held_out_run == test_run   # in _load_synthetic_set
# A fold testing on run R is augmented ONLY by the model that never saw run R.
# Removing/weakening this reintroduces the exact leakage the whole cross-fitting
# scheme exists to prevent, and would invalidate the RR-3 result.
```
