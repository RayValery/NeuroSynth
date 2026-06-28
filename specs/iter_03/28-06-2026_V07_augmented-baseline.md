# Phase 0 — Augmented Baseline Evaluation (Step 3b)

**Status:** Approved  
**Created:** 2026-06-28  
**Author:** Claude  
**Follows:** V06 — within-subject training fix (specs/iter_02/28-06-2026_V06_within-subject-training-fix.md)  
**Scope:** Run `--run augmented`, produce comparison analysis vs vanilla, update report

---

## Executive Summary

The vanilla within-subject LORO baseline is complete at 56.7% ± 10.8% (V06, 106 subjects, 318 folds). The augmented baseline is the second half of Phase 0 Step 3 — it establishes the accuracy ceiling achievable with standard signal augmentation on real data alone. This number is the **threshold synthetic data must beat** to justify building the diffusion model (RR-3). The augmented training code is already implemented; the remaining work is running it, comparing results per-subject, and documenting findings.

---

## Goals

- Establish the augmented baseline accuracy (mean ± std across 106 subjects) under identical evaluation conditions as vanilla
- Quantify per-subject augmentation benefit, segmented by the four subject tiers from V06
- Determine the TSTR delta: the gap between augmented and vanilla accuracy that synthetic data generation must surpass
- Update `reports/phase0_baseline_report.md` with augmented results and comparative analysis

## Non-Goals

- Tuning augmentation parameters (transforms are locked per RR-2; the baseline must be fixed to compare against)
- Adding new augmentation types (out of scope for the comparison baseline)
- Changing the evaluation protocol (LORO splits, 500 epochs, full-batch — identical to vanilla)
- Architecture changes to EEGNet

---

## Background & Context

### What's already done

V06 completed the vanilla run: EEGNet trained with no augmentation, within-subject LORO, full-batch, 500 fixed epochs, CosineAnnealingLR. Result: **56.7% ± 10.8%** (F1: 0.567), MLflow run `fe46e741342f4cf1b7ae2645d224e0c1`.

Subject tier distribution from V06:
| Tier | Range | Count | % | Interpretation |
|---|---|---|---|---|
| Non-responders | <45% | 14 | 13% | Genuine BCI illiteracy — no reliable ERD |
| Data-limited | 45–56% | 41 | 39% | Correct formulation, too few samples |
| Learning | 56–70% | 42 | 40% | Model works, room to grow |
| Strong | >70% | 9 | 8% | Clean ERD, consistent signal |

The **data-limited tier** (39%) is the primary augmentation target. These are subjects where the model is learning but doesn't have enough signal variation to generalize. Augmentation should help by artificially diversifying the 30-sample training fold.

### Why augmentation may or may not help

**May help:** At 30 training samples, EEGNet (2,930 params) is severely data-starved. Block 2 (spatial filter, 128 params) in particular needs enough variation to distinguish ERD topography from noise. Augmentation that preserves spatial structure (channel dropout, time-shift) while adding variation (gaussian noise, crop) gives the model more effective training signal.

**May not help significantly:** Block 2 is subject-specific — it needs variation *of the same subject's signal*, not noise. Gaussian noise and crop preserve the relative spatial structure across channels. However, if the core problem is genuine ERD absence (non-responders, 13%) or signal instability across runs, augmentation cannot fix those.

**Expected outcome:** The augmented mean should be 2–8% higher than vanilla (58–65%), driven mainly by the data-limited tier. The non-responder tier should be unaffected. The strong-responder tier may slightly regress if augmentation distorts already-clean signal.

### The augmentation implementation

`eval/augmentations.py` — `STANDARD_AUGMENTOR`:

```python
STANDARD_AUGMENTOR = Augmentor(
    transforms=[gaussian_noise, time_shift, channel_dropout, sliding_window_crop],
    p_apply=0.5,
)
```

Each transform is applied independently per sample per epoch with probability 0.5. Over 500 training epochs, each of the ~30 training samples is seen 500 times, augmented in a different random configuration each time. Expected augmentations per epoch per sample: 2 of the 4 transforms.

| Transform | Parameters | Effect |
|---|---|---|
| `gaussian_noise` | `std=0.1` | Adds zero-mean noise at 10% of the epoch's own std — adaptive, mild |
| `time_shift` | `max_shift=50` | Rolls signal ±312ms (at 160Hz) — preserves spatial structure, introduces temporal variation |
| `channel_dropout` | `p=0.1` | Zeros ~6/64 channels — forces robustness to channel absence |
| `sliding_window_crop` | `crop_frac=0.8` | Keeps a random contiguous 4-second window, reflects-pads back to 5s — temporal invariance |

All transforms preserve the `(C, T)` shape. Augmentation is applied **training-only** — test inference is always on clean data.

---

## Requirements

### Must Have

- Run `eval/train_baseline.py --run augmented` under identical hyperparameters as vanilla (seed=42, max_epochs=500, lr=1e-3, dropout=0.5, full-batch)
- Log to MLflow experiment `phase0-baseline` with run name `augmented`
- Produce `models/checkpoints/per_subject_results_augmented.json` (per-subject fold accuracies)
- Produce a comparison analysis script at `eval/compare_baselines.py` that loads both JSON files and outputs:
  - Overall: vanilla mean/std vs augmented mean/std, delta, paired Wilcoxon p-value
  - Per-tier: mean delta for each subject tier
  - Per-subject: sorted delta bar chart (positive = augmentation helped, negative = regressed)
- Update `reports/phase0_baseline_report.md` section 6 (results) with augmented numbers and comparative analysis
- Save comparison figures to `reports/figures/` and regenerate the figures script

### Should Have

- Histogram overlaying vanilla vs augmented subject accuracy distributions
- Subject-level scatter plot: vanilla accuracy (x) vs augmented accuracy (y), colored by tier, with y=x reference line
- Statistical significance annotation on the comparison (paired Wilcoxon signed-rank test, α=0.05)

### Nice to Have

- Per-fold (not per-subject) comparison to understand fold-level variance reduction
- McNemar test on pooled predictions to check if augmentation improves classification at the prediction level

---

## Architecture Overview

No new code components. The augmented run reuses the entire existing pipeline:

```
data/processed.npz
       │
       ▼
within_subject_loro_split()          ← same splits as vanilla (deterministic)
       │
       ▼
EEGTorchDataset + STANDARD_AUGMENTOR ← augmentation added here, training-time only
       │
       ▼
_train_fold() × 318 folds            ← identical hyperparameters
       │
       ▼
_eval_on_test() × 318 folds          ← no augmentation at inference
       │
       ▼
per_subject_results_augmented.json   ← input to compare_baselines.py
MLflow run "augmented"
```

The comparison script is a pure analysis utility — it reads the two JSON outputs and produces figures:

```
models/checkpoints/per_subject_results_vanilla.json    ─┐
models/checkpoints/per_subject_results_augmented.json  ─┤─→ eval/compare_baselines.py → reports/figures/
```

---

## Data Models

### `per_subject_results_augmented.json` (already produced by `train_baseline.py`)

```python
[
    {
        "subject": 1,
        "mean_accuracy": 0.578,
        "fold_accuracies": [0.533, 0.600, 0.600]
    },
    ...  # 106 entries total
]
```

### Internal comparison data structure in `compare_baselines.py`

```python
@dataclass
class SubjectComparison:
    subject: int
    vanilla_acc: float     # mean of 3 folds
    augmented_acc: float   # mean of 3 folds
    delta: float           # augmented - vanilla
    tier: str              # "non-responder" | "data-limited" | "learning" | "strong"
```

---

## Implementation Plan

### Phase 1 — Run augmented baseline

- [ ] Run: `source .venv/bin/activate && python eval/train_baseline.py --data data/processed.npz --run augmented`
  - Runtime: ~1–2 hours on CPU (same as vanilla); ~15 min on MPS/CUDA
  - Produces: `models/checkpoints/per_subject_results_augmented.json`, `models/checkpoints/confusion_matrix_augmented.png`, MLflow metrics
- [ ] Verify MLflow logged: `test_accuracy_mean`, `test_accuracy_std`, `test_f1_mean` for the augmented run
- [ ] Record the final summary output:
  ```
  Run        : augmented
  Accuracy   : X.XXXX ± X.XXXX  (XX.X%)
  F1 (macro) : X.XXXX
  ```

### Phase 2 — Comparison analysis script

Create `eval/compare_baselines.py`:

- [ ] Load `per_subject_results_vanilla.json` and `per_subject_results_augmented.json`
- [ ] Build per-subject comparison table: subject id, vanilla_acc, augmented_acc, delta, tier assignment
- [ ] Compute tier assignments using the V06 thresholds:
  ```python
  def assign_tier(acc: float) -> str:
      if acc < 0.45: return "non-responder"
      if acc < 0.56: return "data-limited"
      if acc < 0.70: return "learning"
      return "strong"
  ```
  Note: tier is assigned based on the **vanilla** accuracy (it describes the subject's baseline difficulty, not the augmented result)
- [ ] Compute overall statistics:
  ```
  vanilla:   mean ± std, median
  augmented: mean ± std, median
  delta:     mean ± std
  paired Wilcoxon signed-rank p-value
  ```
- [ ] Compute per-tier mean delta and per-tier mean augmented accuracy
- [ ] Generate figures (save to `reports/figures/`):
  - `fig10_vanilla_vs_augmented_distribution.png` — overlapping histograms or KDE of both distributions
  - `fig11_augmentation_delta_by_tier.png` — boxplot of per-subject delta by tier
  - `fig12_vanilla_vs_augmented_scatter.png` — scatter plot, x=vanilla acc, y=augmented acc, colored by tier, y=x reference line
- [ ] Print summary table to stdout

### Phase 3 — Report update

- [ ] Update `reports/phase0_baseline_report.md`:
  - Section on results: add augmented row to the comparison table
  - Add a new section: "Step 3b — Augmented Baseline" with the same level of detail as Step 3a
  - Add analysis: per-tier delta, statistical significance, interpretation
  - Embed `fig10`, `fig11`, `fig12`
  - Update the abstract/executive summary with both baseline numbers
- [ ] Update `reports/generate_figures.py` to include the three new figures (or import them from `compare_baselines.py` if extracted to functions)
- [ ] Update `README.md` status table: mark Step 3b done with the augmented accuracy number

---

## `compare_baselines.py` — Detailed Specification

```python
"""
Compare vanilla vs augmented within-subject LORO baselines.

Loads per-subject result JSONs from models/checkpoints/ and produces:
- Console summary table
- Three comparison figures saved to reports/figures/

Usage:
    python eval/compare_baselines.py
    python eval/compare_baselines.py --vanilla path/to/vanilla.json --augmented path/to/augmented.json
"""
```

**Key functions:**

```python
def load_results(path: Path) -> dict[int, float]:
    """Load per_subject_results JSON → {subject_id: mean_accuracy}."""

def assign_tier(vanilla_acc: float) -> str:
    """Classify subject by vanilla accuracy into the four V06 tiers."""

def compute_statistics(
    vanilla: dict[int, float],
    augmented: dict[int, float],
) -> dict:
    """
    Returns:
        {
            "vanilla_mean": float, "vanilla_std": float, "vanilla_median": float,
            "augmented_mean": float, "augmented_std": float, "augmented_median": float,
            "delta_mean": float, "delta_std": float,
            "wilcoxon_stat": float, "wilcoxon_p": float,
            "per_tier": {tier_name: {"n": int, "vanilla_mean": float, "augmented_mean": float, "delta_mean": float}}
        }
    """

def plot_distributions(subjects: list[SubjectComparison], out: Path) -> None:
    """fig10: overlapping KDE + rug plots for vanilla and augmented."""

def plot_delta_by_tier(subjects: list[SubjectComparison], out: Path) -> None:
    """fig11: boxplot of (augmented - vanilla) delta, one box per tier."""

def plot_scatter(subjects: list[SubjectComparison], out: Path) -> None:
    """fig12: scatter x=vanilla, y=augmented, colored by tier, y=x diagonal."""
```

---

## Error Handling

- If `per_subject_results_augmented.json` doesn't exist: print a clear message and exit with the run command
- If subject counts differ between vanilla and augmented JSONs: warn and use the intersection
- If Wilcoxon test fails (e.g., all deltas are 0): catch and report N/A for p-value

---

## Interpreting Results

### Decision tree after seeing augmented accuracy

| Augmented result | Interpretation | Action |
|---|---|---|
| < 56.7% (below vanilla) | Augmentation hurts — transforms distort the within-subject signal more than they help | Investigate which subjects regressed; consider reducing p_apply; do not change spec thresholds — this is a valid finding |
| 56.7%–60% (≈ vanilla) | Augmentation provides negligible benefit — data-limited subjects don't respond to generic transforms | Valid finding; TSTR delta is ≤3.3%; synthetic data target is still meaningful |
| 60%–65% | Augmentation modestly helps data-limited subjects | Expected outcome; TSTR needs to beat this with subject-conditioned generation |
| > 65% | Augmentation works well | TSTR bar is higher; stronger motivation for subject-conditioned generation |

In all cases, record the number and proceed. The augmented baseline is not a gate — it is a fixed reference point. Whether the result is 57% or 65%, the TSTR comparison is valid.

### What to look for in per-tier analysis

- **Non-responder tier (<45% vanilla):** expect minimal change — these subjects don't produce reliable ERD; augmentation cannot create signal that isn't there
- **Data-limited tier (45–56% vanilla):** expect the most improvement — augmentation provides training variation they lack
- **Learning tier (56–70% vanilla):** expect moderate improvement or neutral — model already learning, augmentation may push some over 70%
- **Strong tier (>70% vanilla):** expect neutral or slight regression — model already at ceiling; aggressive augmentation may occasionally hurt

---

## Testing Strategy

No automated tests required for a research evaluation run. Manual verification:

- [ ] MLflow run appears in `phase0-baseline` experiment with correct params (`augmented=True`, `eval_protocol=within_subject_loro`)
- [ ] Subject count in JSON is 106 (not 109 — the 3 dropped subjects)
- [ ] Each subject entry has exactly 3 fold accuracies
- [ ] All fold accuracies are in [0, 1]
- [ ] `compare_baselines.py` runs without error on the two JSONs
- [ ] Wilcoxon p-value is meaningful (not NaN)
- [ ] All three figures saved to `reports/figures/`

---

## Success Criteria

- `eval/train_baseline.py --run augmented` completes across all 318 folds without error
- `per_subject_results_augmented.json` contains 106 subjects × 3 folds
- `compare_baselines.py` produces the summary table and three figures
- `reports/phase0_baseline_report.md` updated with augmented results and per-tier analysis
- `README.md` status table updated: Step 3b marked done with the augmented accuracy number
- MLflow run logged with `augmented=True` params

---

## Context for Handoff

### Repository state at time of writing

```
NeuroSynth/
├── data/processed.npz              ← 4748 epochs, 106 subjects, shape (4748, 64, 801)
├── eval/
│   ├── eegnet.py                   ← EEGNet, 2,930 params, unchanged
│   ├── augmentations.py            ← STANDARD_AUGMENTOR ready, unchanged
│   ├── dataset.py                  ← within_subject_loro_split, unchanged
│   └── train_baseline.py           ← supports --run augmented, ready to run
├── models/checkpoints/
│   ├── per_subject_results_vanilla.json  ← V06 results, all 106 subjects
│   └── confusion_matrix_vanilla.png
├── reports/
│   ├── phase0_baseline_report.md   ← covers V01–V06, needs augmented section
│   └── figures/                    ← fig1–fig9 generated
└── mlruns/                         ← MLflow tracking, vanilla run recorded
```

### V06 vanilla result (the reference)

```
Run        : vanilla
Subjects   : 106
Folds      : 318
Accuracy   : 0.5671 ± 0.1080  (56.7%)
F1 (macro) : 0.5674
Run ID     : fe46e741342f4cf1b7ae2645d224e0c1
```

### Running environment

```bash
source .venv/bin/activate
# Python 3.x, PyTorch, MNE, MLflow, scikit-learn, matplotlib, scipy all installed
# Device: check with python -c "import torch; print(torch.backends.mps.is_available())"
```

### Key design decisions to preserve

- **Full-batch training** (`batch_size=len(y_train)` ≈ 30): preserves clean BatchNorm statistics on small within-subject fold
- **Fixed 500 epochs, no early stopping**: CosineAnnealingLR must run to completion
- **LORO splits are deterministic**: same (subject, test_run) fold structure as vanilla — enabling paired comparison
- **Augmentation at training time only**: `EEGTorchDataset` applies augmentor in `__getitem__`; test loader has no augmentor
- **Seed 42**: torch and numpy seeds set identically to vanilla for reproducibility of weight initialization

### What comes after this spec

Once augmented baseline is recorded:
- **Phase 0 Step 4**: 1D U-Net diffusion model training on `data/processed.npz` (class-conditional generation of motor imagery epochs)
- **Phase 0 Step 5**: TSTR harness — for each LORO fold, augment the 30-sample training set with N synthetic epochs, compare accuracy vs augmented baseline
- **Go/no-go decision (RR-3)**: does TSTR accuracy beat augmented by a meaningful margin?

The augmented accuracy number from this spec directly sets the TSTR target.
