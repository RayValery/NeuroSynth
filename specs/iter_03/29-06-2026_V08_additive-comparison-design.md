# Phase 0 — Additive Comparison Design: Offline Augmentation + TSTR Framework

**Status:** Approved  
**Created:** 2026-06-29  
**Author:** Claude  
**Supersedes:** V07 augmented baseline methodology (specs/iter_03/28-06-2026_V07_augmented-baseline.md)  
**Scope:** Redefine the augmented baseline as an additive experiment; establish the N-variation framework that TSTR will reuse

---

## Executive Summary

The V07 augmented baseline used online augmentation — transforming the 30 training samples in-place each epoch. This is not the correct comparison for TSTR, which *adds* synthetic samples on top of real data. A fair comparison requires both baselines to use the same additive structure: `X_real + N_extra`, where the only variable is the source of extra samples (augmented copies vs. synthetic). This spec defines the corrected methodology, the N-variation experiment, and the full report update plan. The online augmented run (V07) is preserved as a reference data point, not the primary baseline.

---

## Goals

- Define a scientifically defensible comparison framework that directly answers RR-3
- Establish the offline augmented baseline: train on `X_real_30 + N × aug_copies` for N = {1×, 2×, 3×}
- Produce the accuracy-vs-N scaling curve that becomes the central figure of the paper
- Design the framework so TSTR (Phase 0 Step 5) slots in with zero structural changes
- Document everything in the Phase 0 report so the methodology is reproducible and audit-ready

## Non-Goals

- Training the diffusion model (Phase 0 Step 4 — separate spec)
- Running TSTR (Phase 0 Step 5 — separate spec, depends on Step 4)
- Changing EEGNet architecture or LORO evaluation protocol
- Changing augmentation transforms (locked per RR-2)
- Optimising augmentation hyperparameters

---

## Background & Context

### Why the current V07 approach is wrong for TSTR

TSTR (Train on Synthetic, Test on Real) works by taking a real training fold and *adding* synthetic samples to it:

```
TSTR training set = X_real_30 + X_synthetic_N
```

To compare fairly, the augmented baseline must have the same structure:

```
Augmented baseline = X_real_30 + X_aug_N
```

What V07 does instead:

```python
# V07 — online augmentation (WRONG for TSTR comparison)
DataLoader(EEGTorchDataset(X_train_30, y_train, augmentor=STANDARD_AUGMENTOR), batch_size=30)
# Each epoch: 30 augmented versions of the real samples. Real samples are never seen clean.
# Dataset size: always 30.
```

The V07 approach conflates two different things:
1. **Regularisation** — preventing the model from memorising the exact noise pattern of 30 samples (this is what online augmentation achieves)
2. **Data augmentation** — genuinely increasing the number of training samples the model learns from (this is what the TSTR comparison requires)

A reviewer would immediately flag this: "You compare adding N synthetic samples to a baseline that adds zero extra samples. Of course synthetic data wins — you just gave it more training data." That objection would sink the paper.

### The correct experimental variable

The research question is: **given the same number of additional training samples, which source produces better classifiers — augmented copies or synthetic data?**

```
Independent variable:  source of extra samples (augmented | synthetic)
Controlled variable:   number of extra samples N
Controlled variable:   real training data (always the same 30 samples)
Outcome:               LORO test accuracy
```

This isolation is only possible with the additive design.

### Why N matters — the scaling argument

Augmentation creates copies derived from 30 base signals. Each additional copy adds diminishing information — the 10th copy of a time-shifted signal teaches the model very little that the 3rd copy didn't. Augmentation *plateaus*.

Synthetic data from a well-trained diffusion model generates from the learned population distribution, conditioned on class. Each synthetic sample is an independent draw — it adds genuinely new information about the signal manifold. Synthetic data *scales*.

Plotting accuracy vs N on the same graph is the core figure of the paper:

```
Accuracy (%)
  70 │                        · · ·  TSTR (synthetic)
     │                   · ·
  65 │              · ·
     │         · ·           · · · · Augmented (offline copies)
  60 │    · ·      · ·
     │  ·
  57 │· ─────────────────────────── Vanilla (N=0)
     └──────────────────────────────
        0   30   60   90  120  150   N extra samples
```

If this shape appears — synthetic keeps climbing while augmented plateaus — the result is publishable and the business case is clear.

### What to do with the V07 online augmented run

Let V07 finish. It gives a useful data point: "online augmentation (standard PyTorch training-time transforms) achieves X%." This contextualises the corrected result and shows why the additive design matters. Keep it in the report as a reference, not the primary comparison.

---

## Requirements

### Must Have

- Extend `eval/train_baseline.py` with a new run mode `augmented_offline` and parameter `--n-copies`
- Implement `_create_offline_augmented_set()` that pre-generates N copies per sample before training
- Run for `--n-copies 1`, `--n-copies 2`, `--n-copies 3` (N = 30, 60, 90 extra samples)
- Class balance must be preserved: generate equal copies of class 0 and class 1
- Log each run to MLflow with `n_copies` as a tracked parameter
- Produce `per_subject_results_augmented_offline_Nx.json` for each N
- Produce accuracy-vs-N comparison figure saved to `reports/figures/`
- Update `reports/phase0_baseline_report.md` with corrected methodology and results

### Should Have

- Paired Wilcoxon signed-rank test comparing each offline augmented N vs vanilla
- Per-tier analysis: which subject tier benefits most from additional augmented samples?
- `eval/compare_baselines.py` updated to handle multiple N values

### Nice to Have

- N values extended to {5×, 10×} to show the plateau more clearly
- 95% CI bands on the accuracy-vs-N curves

---

## Architecture Overview

```
                       ┌─────────────────────────────┐
                       │  data/processed.npz          │
                       └──────────────┬───────────────┘
                                      │ within_subject_loro_split()
                                      ▼
                       ┌─────────────────────────────┐
                       │   X_real_train (30 samples)  │
                       └──────────┬──────────────────┘
                                  │
               ┌──────────────────┼──────────────────────┐
               ▼                  ▼                       ▼
    ┌───────────────────┐ ┌──────────────────┐  ┌─────────────────────┐
    │ vanilla           │ │ augmented_offline │  │ TSTR (Phase 5)      │
    │ X_real_30         │ │ X_real + N×aug   │  │ X_real + N×synthetic│
    └────────┬──────────┘ └───────┬──────────┘  └──────────┬──────────┘
             │                    │                         │
             └────────────────────┴─────────────────────────┘
                                  │
                            _train_fold()          ← identical for all modes
                                  │
                            _eval_on_test()
                                  │
                          MLflow + per_subject JSON
                                  │
                          compare_baselines.py     ← accuracy vs N figure
```

TSTR (Phase 5) is structurally identical to `augmented_offline` — it only replaces `_create_offline_augmented_set()` with `_load_synthetic_set()`. Everything downstream (training loop, evaluation, logging) is unchanged.

---

## Data Models

### Per-subject results JSON (one file per N)

```
models/checkpoints/per_subject_results_vanilla.json
models/checkpoints/per_subject_results_augmented_online.json  ← V07 reference
models/checkpoints/per_subject_results_augmented_1x.json      ← N=30
models/checkpoints/per_subject_results_augmented_2x.json      ← N=60
models/checkpoints/per_subject_results_augmented_3x.json      ← N=90
```

Each file has the same structure (already defined by `train_baseline.py`):

```python
[
    {
        "subject": 1,
        "mean_accuracy": 0.623,
        "fold_accuracies": [0.600, 0.600, 0.667]
    },
    ...  # 106 entries
]
```

### MLflow parameters for offline augmented runs

```python
{
    "run_name": "augmented_offline",
    "augmented": True,
    "augmentation_mode": "offline",  # new
    "n_copies": 2,                   # new
    "n_train_samples": 90,           # 30 real + 60 aug copies
    "eval_protocol": "within_subject_loro",
    "fixed_epochs": True,
    "seed": 42,
    "max_epochs": 500,
    # ... rest identical to vanilla
}
```

### Internal structure for compare_baselines.py

```python
@dataclass
class RunResult:
    run_name: str          # "vanilla", "aug_online", "aug_1x", "aug_2x", "aug_3x"
    n_extra: int           # 0, 0, 30, 60, 90
    subject_accs: dict     # {subject_id: mean_accuracy}
    overall_mean: float
    overall_std: float

@dataclass
class SubjectRecord:
    subject: int
    vanilla_acc: float
    tier: str              # assigned from vanilla_acc
    aug_online: float
    aug_1x: float
    aug_2x: float
    aug_3x: float
```

---

## Interface Definitions

### `_create_offline_augmented_set()` — new function in `train_baseline.py`

```python
def _create_offline_augmented_set(
    X_train: np.ndarray,      # (30, C, T) real training samples
    y_train: np.ndarray,      # (30,) labels
    n_copies: int,            # how many augmented copies per real sample
    augmentor,                # STANDARD_AUGMENTOR
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-generate n_copies augmented versions of each training sample.
    Returns X_combined (30 + 30*n_copies, C, T) and y_combined (30 + 30*n_copies,).

    Real samples are always included unchanged. Augmented copies are appended.
    Class balance is preserved: each class gets the same number of copies.
    """
```

**Why include real samples unchanged:**  
The model must anchor on the clean signal. Including originals ensures the model sees at least some unperturbed examples. Without them, Block 2 (spatial filter) might over-fit to augmentation artefacts (e.g., the boundary effect of `time_shift` roll-over).

**Why pre-generate instead of on-the-fly for this experiment:**  
The augmented samples are fixed at the start of training. This makes runs deterministic and reproducible — essential for the N-variation comparison. Online augmentation generates new variants every epoch, which confounds the "how many samples" variable with "how many unique variants". Offline augmentation holds N constant.

### New CLI parameters in `train_baseline.py`

```
--run {vanilla, augmented, augmented_offline}
      augmented          = V07 online (keep for backward compat)
      augmented_offline  = new: uses _create_offline_augmented_set()

--n-copies INT           (default: 2, only used when --run augmented_offline)
      Number of augmented copies added per real sample.
      n-copies=1 → 30+30=60 total   (1× real data added)
      n-copies=2 → 30+60=90 total   (2× real data added)
      n-copies=3 → 30+90=120 total  (3× real data added)
```

### Updated `compare_baselines.py`

```python
def load_all_results(checkpoint_dir: Path) -> dict[str, RunResult]:
    """Load all available per_subject JSON files. Keys: run names."""

def plot_accuracy_vs_n(results: dict[str, RunResult], out: Path) -> None:
    """
    Main paper figure: accuracy vs N_extra_samples.
    Lines: vanilla (N=0), aug_online (N=0*), aug_1x, aug_2x, aug_3x.
    Error bars: ± std across subjects.
    (* aug_online plotted at N=0 because it does not add samples)
    """

def plot_tier_breakdown_vs_n(results: dict, out: Path) -> None:
    """
    4-panel plot (one per tier): accuracy vs N for that tier's subjects.
    Shows whether data-limited subjects benefit more than strong subjects.
    """

def print_summary_table(results: dict) -> None:
    """Console table: run_name | n_extra | mean_acc | std | Δ_vs_vanilla | p-value"""
```

---

## Implementation Plan

### Phase 1 — Let V07 finish, document it

- [x] Wait for current `--run augmented` (V07) to complete
- [x] Record the final result: `Run: augmented_online | Accuracy: X.X% ± X.X%`
- [x] Note in report: "V07 online augmented = reference only, not the TSTR comparison baseline"

### Phase 2 — Implement offline augmented mode

**File: `eval/train_baseline.py`**

- [ ] Add `_create_offline_augmented_set(X_train, y_train, n_copies, augmentor, seed)`:
  ```python
  def _create_offline_augmented_set(X_train, y_train, n_copies, augmentor, seed=42):
      rng = np.random.default_rng(seed)
      aug_X, aug_y = [], []
      for _ in range(n_copies):
          perm = rng.permutation(len(X_train))
          for idx in perm:
              x = torch.tensor(X_train[idx], dtype=torch.float32)
              aug_X.append(augmentor(x).numpy())
              aug_y.append(y_train[idx])
      X_combined = np.concatenate([X_train, np.array(aug_X, dtype=np.float32)])
      y_combined = np.concatenate([y_train, np.array(aug_y, dtype=np.int64)])
      return X_combined, y_combined
  ```

- [ ] Update `train()` to handle `run_name == "augmented_offline"`:
  ```python
  if run_name == "augmented_offline":
      X_tr, y_tr = _create_offline_augmented_set(
          X_tr_fold, y_tr_fold, n_copies=n_copies,
          augmentor=STANDARD_AUGMENTOR,
          seed=seed + subj * 100 + fold_idx,
      )
      augmentor_for_loader = None  # no further online augmentation
  ```

- [ ] Add `n_copies: int = 2` parameter to `train()` signature

- [ ] Update MLflow logging to include `augmentation_mode`, `n_copies`, `n_train_samples`

- [ ] Update `main()` with new CLI args:
  ```
  --run choices: ["vanilla", "augmented", "augmented_offline"]
  --n-copies INT (default 2)
  ```

- [ ] Update output JSON filename: `per_subject_results_augmented_{n_copies}x.json`

### Phase 3 — Run offline augmented baselines

Run sequentially (each takes ~1–2h on CPU, ~15 min on MPS):

```bash
source .venv/bin/activate

python eval/train_baseline.py --data data/processed.npz \
    --run augmented_offline --n-copies 1
# → per_subject_results_augmented_1x.json

python eval/train_baseline.py --data data/processed.npz \
    --run augmented_offline --n-copies 2
# → per_subject_results_augmented_2x.json

python eval/train_baseline.py --data data/processed.npz \
    --run augmented_offline --n-copies 3
# → per_subject_results_augmented_3x.json
```

### Phase 4 — Update compare_baselines.py

- [ ] Extend `compare_baselines.py` to load all result files (vanilla, aug_online, aug_1x, aug_2x, aug_3x)
- [ ] Implement `plot_accuracy_vs_n()` — the main paper figure:
  - x-axis: N extra samples (0, 0, 30, 60, 90)
  - y-axis: mean accuracy ± std
  - Separate line per run type
  - Horizontal reference line at 50% (chance)
  - Save to `reports/figures/fig10_accuracy_vs_n.png`
- [ ] Implement `plot_tier_breakdown_vs_n()` — 4-panel by tier:
  - Save to `reports/figures/fig11_tier_breakdown_vs_n.png`
- [ ] Implement `print_summary_table()` with Wilcoxon p-values vs vanilla
- [ ] Save per-subject delta CSV for each N to `reports/figures/`

### Phase 5 — Update the Phase 0 report

**File: `reports/phase0_baseline_report.md`**

- [ ] Add new section: **"Methodology Note: Corrected Comparison Design"**

  Content:
  - Explain why online augmentation (V07) is not the correct TSTR baseline
  - Explain the additive design and why it isolates the right variable
  - State that V07 result is kept as a reference data point
  - Forward-reference to TSTR section (to be filled in Phase 0 Step 5)

- [ ] Add new section: **"Step 3b — Offline Augmented Baseline"**

  Content:
  - Results table:

    | Run | N extra | N total | Accuracy | Δ vs vanilla | p-value |
    |---|---|---|---|---|---|
    | Vanilla | 0 | 30 | 56.7% ± 10.8% | — | — |
    | Aug online (V07) | 0* | 30 | X% | +X% | X |
    | Aug offline 1× | 30 | 60 | X% | +X% | X |
    | Aug offline 2× | 60 | 90 | X% | +X% | X |
    | Aug offline 3× | 90 | 120 | X% | +X% | X |

  - Embed `fig10_accuracy_vs_n.png`
  - Embed `fig11_tier_breakdown_vs_n.png`
  - Analysis: where does augmentation plateau? Which tier benefits most?
  - Conclusion: the offline augmented result at the best N becomes the **primary baseline for TSTR comparison**

- [ ] Update `reports/generate_figures.py` to include `fig10` and `fig11`

- [ ] Update `README.md` status table:
  - Step 3b: mark done once all three N runs complete, record best offline augmented accuracy

---

## Error Handling

- If `n_copies` would cause `X_combined` to exceed memory on the target machine (unlikely at N≤3 with 30 samples, but possible at N=10+): log a warning and cap at available memory
- If augmented set has class imbalance after generation (can happen if original set is imbalanced): log per-class counts, but do not resample — preserve the original imbalance for scientific integrity
- If a JSON file is missing when `compare_baselines.py` runs: skip that run with a warning, do not error out — partial results are still valuable

---

## Testing Strategy

Manual verification (research code, no automated tests required):

- [ ] After implementing `_create_offline_augmented_set`:
  ```python
  X_c, y_c = _create_offline_augmented_set(X_train_30, y_train_30, n_copies=2, augmentor=STANDARD_AUGMENTOR)
  assert len(X_c) == 90      # 30 real + 60 aug
  assert len(y_c) == 90
  assert np.array_equal(X_c[:30], X_train_30)   # real samples unchanged at front
  assert not np.array_equal(X_c[30:60], X_train_30)  # copies are different
  ```
- [ ] Verify class balance: `np.bincount(y_c)` should be roughly equal (2:1 of original balance)
- [ ] MLflow: each N run logged with `n_copies`, `n_train_samples`, `augmentation_mode=offline`
- [ ] `compare_baselines.py` runs without error when only a subset of N files exist
- [ ] `fig10_accuracy_vs_n.png` has correct x-axis values (0, 30, 60, 90) and all expected lines

---

## Success Criteria

**Implementation complete when:**
- `train_baseline.py --run augmented_offline --n-copies N` runs for any N without error
- All three N runs complete: 1×, 2×, 3× (318 folds each)
- Three JSON files present in `models/checkpoints/`
- `fig10_accuracy_vs_n.png` and `fig11_tier_breakdown_vs_n.png` generated
- Report updated with corrected methodology section and results table

**Scientific success when:**
- Offline augmented at best N establishes a clear ceiling (plateaus between N=60 and N=90)
- TSTR (Phase 5) exceeds this ceiling — the gap is the publishable result
- If augmented keeps climbing without plateau: increase N to 5×, 10× to find the ceiling

**TSTR readiness gate:**
The offline augmented result at N=2× (90 total samples) becomes the **primary comparison baseline for TSTR**. TSTR will be run at the same N values (adding 30, 60, 90 synthetic samples) so the accuracy-vs-N graph has matched points on both curves.

---

## Relationship to TSTR (Phase 0 Step 5)

This spec creates the infrastructure that TSTR reuses unchanged:

```python
# Phase 5 TSTR — only this function changes:
def _load_synthetic_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    subject_id: int,
    n_synthetic: int,
    diffusion_model,
) -> tuple[np.ndarray, np.ndarray]:
    X_synth = diffusion_model.generate(n=n_synthetic, subject_id=subject_id)
    y_synth = diffusion_model.labels(n=n_synthetic)
    return np.concatenate([X_real, X_synth]), np.concatenate([y_real, y_synth])
```

Training loop, evaluation, MLflow logging, report structure — all identical to `augmented_offline`. The only question Phase 5 answers is: "does `_load_synthetic_set` outperform `_create_offline_augmented_set` at the same N?"

---

## Context for Handoff

### Repository state at time of writing

```
eval/train_baseline.py          ← needs _create_offline_augmented_set() + new run mode
eval/augmentations.py           ← STANDARD_AUGMENTOR unchanged
eval/compare_baselines.py       ← does not yet exist (to be created)
models/checkpoints/
  per_subject_results_vanilla.json          ← V06 done, 56.7%
  per_subject_results_augmented.json        ← V07 in progress (online, ~25% done)
reports/phase0_baseline_report.md           ← covers V01–V06, needs update
```

### Key numbers so far

| Run | Accuracy | Mode | Status |
|---|---|---|---|
| Vanilla (V06) | **56.7% ± 10.8%** | N/A | Done |
| Aug online (V07) | tracking ~57% | online (reference) | In progress |
| Aug offline 1× | — | additive, N=30 | Not started |
| Aug offline 2× | — | additive, N=60 | Not started |
| Aug offline 3× | — | additive, N=90 | Not started |

### Why this matters for the paper

The accuracy-vs-N figure is the central result of the Phase 0 paper:
- If TSTR > augmented at all N: strong result, clear business case
- If TSTR > augmented only at large N: suggests synthetic data is higher quality per sample
- If TSTR ≤ augmented: go/no-go fails — reassess diffusion model quality before Phase 1

The corrected design ensures this figure can withstand peer review. A reviewer cannot object to "you added more data to TSTR than to augmented" — both curves will be plotted at the same N values.
