# NeuroSynth: Phase 0 Step 5 — TSTR Harness & RR-3 Go/No-Go Gate

**Project:** NeuroSynth — Synthetic EEG Data Platform
**Phase:** 0 — Baseline Establishment
**Step:** 5 — Train-on-Synthetic, Test-on-Real (TSTR) go/no-go gate (RR-3)
**Date:** 2026-07-08
> **Review note (2026-10-31).** Two statements in this report were corrected after a final review: (1) "significantly worse than doing nothing" — the paired difference vs real-only is −2.0 pts with a 95% bootstrap CI of −4.3 to +0.2 (two-sided Wilcoxon p = 0.09), i.e. a trend, not a significant harm; (2) the tier-vs-vanilla deltas in §5.3 are inflated by regression to the mean (tiers are defined on vanilla accuracy). The tier comparison against augmentation, which is not affected, is now shown alongside. The gate verdict itself is unchanged. Consolidated analysis: [the README](../README.md).

**Status:** COMPLETE. Three TSTR passes ran on Modal (A10G, 500 epochs, bank_size=2048). **RR-3 gate: FAIL** — synthetic-augmented training scores 54.7% at N=60, below both the augmented ceiling (59.5%) and the vanilla real-only baseline (56.7%). Decision: pivot to subject-conditional generation.

---

## Abstract

Phase 0 Step 5 runs the central experiment of the project: does augmenting a subject's tiny real training fold with **synthetic** epochs from the population diffusion models beat augmenting it with naive **signal augmentation**? Every LORO fold's real training set is extended with N synthetic epochs drawn from the run-holdout diffusion model that never saw that fold's test run, EEGNet is trained on real + synthetic, and accuracy is measured on the real held-out run. The result is compared, per subject, against the two established reference points — the vanilla baseline (**56.7% ± 10.8%**) and the additive augmented ceiling (**59.5% ± 11.2%** at N=60). The go/no-go gate (**RR-3**) passes iff synthetic-augmented training at N=60 exceeds the 59.5% augmented ceiling by a margin that is both statistically significant (one-sided Wilcoxon signed-rank across the 106 subjects, p < 0.05) and practically meaningful (> +1% absolute).

This step added no new model and did not touch EEGNet, the LORO protocol, or the diffusion models. It added one function pair to `src/neurosynth/train_baseline.py` (`_build_synthetic_banks` + `_load_synthetic_set`, the drop-in replacement for `_create_offline_augmented_set`), a Modal launcher (`cloud/modal_train_tstr.py`), and the comparison/verdict logic in `scripts/compare_baselines.py`.

---

## 1. What the Gate Tests, and Why It Is the Whole Point

NeuroSynth's hypothesis is that when a new BCI user provides only ~5 minutes of calibration data, synthetic EEG can augment that data and yield a better personal classifier without collecting more real data. Phase 0 validates that hypothesis with a strict, leakage-free experiment:

```
vanilla   : EEGNet on real only                → 56.7%       (established, Step 3a)
augmented : EEGNet on real + N augmented copies → 59.5% @ 60  (established, Step 3b)
TSTR      : EEGNet on real + N synthetic        → ???         (this step)

Go criterion (RR-3): TSTR@60 > 59.5%, significantly (p < 0.05) and meaningfully (> +1%).
```

If synthetic data cannot beat naive signal augmentation (Gaussian noise, time-shift, channel dropout, crop), the population generator provides no practical value at this rung, and the project pivots directly to subject-conditional generation. If it wins, the population rung is validated and becomes the first result of the preprint.

## 2. Method

### 2.1 Additive, sample-for-sample comparable to the augmented arm

For each LORO fold the real training set (~30 samples) is extended to `len(y_real) × (1 + n_copies)` by appending, **per class** `c`, exactly `n_copies × count(y_real == c)` synthetic epochs of class `c`. This reproduces the exact per-fold sample count and class distribution the augmented arm produced at the same `n_copies`, so the TSTR and augmented curves are comparable point-for-point at N = {30, 60, 90}. Real samples are prepended unchanged so the classifier still anchors on real signal — identical structure to `_create_offline_augmented_set`.

### 2.2 Leakage-free routing (the one invariant that must never break)

Step 4 produced three population DDPMs under leave-one-run-out cross-fitting:

| Checkpoint | `held_out_run` | Trained on | Serves folds testing on |
|---|---|---|---|
| `diffusion_holdout_run4.pt`  | 4  | runs 8 & 12 | run 4  |
| `diffusion_holdout_run8.pt`  | 8  | runs 4 & 12 | run 8  |
| `diffusion_holdout_run12.pt` | 12 | runs 4 & 8  | run 12 |

For any fold testing on run *R*, the synthetic augmentation is drawn from `diffusion_holdout_run{R}` — the model that never saw run *R* of any subject. The synthetic data therefore provably carries no test-run information, the same cleanliness standard as the augmented baseline (which only jitters a fold's *training* runs). A hard assertion enforces this:

```python
assert bank.held_out_run == test_run   # in _load_synthetic_set
```

Because every subject is evaluated leakage-free, the comparison is a **direct, full-cohort, per-subject paired test** against 56.7% / 59.5% — no sub-selection, no loss of statistical power.

### 2.3 Pre-generated per-model banks (compute decision)

The models are subject-agnostic, so a fold's synthetic samples depend only on its *test run* (which selects the model), not on its subject. Rather than repeat 50-step DDIM sampling inside each of the 318 folds, each model generates a class-balanced bank **once** (default 2048 = 1024/class); each fold draws its N samples from `bank[test_run]` without replacement via a fold-seeded RNG (`seed + subject*100 + fold_idx`), mirroring the augmented seeding so reruns are identical. Max draw per fold at 3× is ~45/class, far below 1024/class.

### 2.4 Sampler-health guard (protects against the `clip_denoised` fragility)

`EEGDiffusionModel.load()` rebuilds the config from checkpoint metadata, but `clip_denoised` was added after Step 4 and is **not serialized** in the three checkpoints — on load it falls back to the current `DiffusionConfig` default (4.0). Correct sampling therefore depends on the shipped code's default, not stored metadata. Rather than trust that, `_build_synthetic_banks` asserts the generated `0.8 < X.std() < 1.3` (real z-scored data has std = 1.0). A std of a few hundred means the DDIM sampler regressed to the pre-fix explosion; the guard aborts loudly *before* any experiment compute is spent. `clip_denoised` is also logged as an MLflow param for traceability.

**Local verification (2026-07-08):** building banks from the three real checkpoints produced `std = 1.000` for all of runs 4/8/12, class balance `[32 32]` at `bank_size=64` — the corrected sampler is confirmed healthy.

## 3. Implementation Summary

| Component | Change |
|---|---|
| `src/neurosynth/train_baseline.py` | **New:** `SyntheticBank` dataclass, `_build_synthetic_banks`, `_load_synthetic_set`, `tstr` run mode, `--bank-size`, `phase0-tstr` experiment default. EEGNet / `_train_fold` / `_eval_on_test` / LORO / aggregation unchanged. |
| `scripts/compare_baselines.py` | **New:** load `tstr_{1,2,3}x`, `_wilcoxon_paired`, `print_gate_verdict` (RR-3), real TSTR curve + gate line on fig10, `fig12_tstr_vs_aug_delta`, `tstr_*` CSV columns. |
| `cloud/modal_train_tstr.py` | **New:** cloud-GPU launcher; mounts the data + checkpoint volumes, builds banks on-GPU, runs the three TSTR passes in parallel, writes the JSONs back. |
| `EEGDiffusionModel`, the three checkpoints, EEGNet | **Reused, frozen — unchanged.** |

The TSTR JSONs land in `models/checkpoints/per_subject_results_tstr_{1,2,3}x.json` in the identical schema as the existing baselines, so they slot directly into `compare_baselines.py`.

## 4. Reproduction

```bash
# 1. Run the three TSTR passes on Modal (cloud GPU), in parallel, ~1 h, ~$1-2:
modal run --detach cloud/modal_train_tstr.py

# 2. Pull the result JSONs back:
modal volume get neurosynth-checkpoints per_subject_results_tstr_1x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_2x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_3x.json models/checkpoints/

# 3. Compute the RR-3 verdict + figures locally:
python scripts/compare_baselines.py
```

A local run is also possible (`python scripts/train_classifier.py --run tstr --n-copies 2`) but is slow on Apple MPS (~2.5 h/pass); Modal is the chosen path.

---

## 5. Results

Three TSTR passes completed on Modal (A10G, 500 epochs, bank_size=2048, seed=42), all 106 subjects × 3 folds = 318 folds each, leakage-free. Numbers from `python scripts/compare_baselines.py`.

### 5.1 Per-N results

| Run | N extra | Mean acc | Std | Δ vs vanilla | Δ vs aug@same-N | Wilcoxon p (> vanilla) |
|---|---|---|---|---|---|---|
| vanilla | 0 | 56.7% | 10.8% | — | — | — |
| aug_1x | 30 | 58.6% | 9.1% | +1.9% | — | 0.034 |
| aug_2x (ceiling) | 60 | 59.5% | 11.2% | +2.8% | — | 0.013 |
| aug_3x | 90 | 59.0% | 10.0% | +2.3% | — | 0.034 |
| **tstr_1x** | 30 | **55.2%** | 10.0% | −1.6% | −3.4% | 0.920 |
| **tstr_2x** | 60 | **54.7%** | 8.9% | −2.0% | −4.8% | 0.954 |
| **tstr_3x** | 90 | **55.5%** | 9.9% | −1.2% | −3.5% | 0.866 |

Every synthetic point lands **below the vanilla real-only baseline** and well below the augmented curve. Adding more synthetic (1×→3×) does not help — the curve is flat around 55%. TSTR is not merely failing to beat augmentation; it also trends below real-only training (paired Δ −2.0 pts, 95% bootstrap CI −4.3 to +0.2, two-sided Wilcoxon p = 0.09). A high p-value on a one-sided "better than vanilla" test does not by itself establish "worse".

### 5.2 RR-3 gate

- **Margin test:** `mean(tstr_2x) − mean(aug_2x)` = **−4.80%** (paired, 106 subjects) vs +1.0% threshold → **FAIL**
- **Significance test:** one-sided paired Wilcoxon (`tstr_2x > aug_2x`) p = **1.0000** vs 0.05 → **FAIL**
- Only **27 / 106** subjects scored higher with synthetic than with augmentation; only **44 / 106** beat their own vanilla score.
- **VERDICT: FAIL.**

### 5.3 Tier breakdown (tstr_2x, tiers assigned from vanilla accuracy)

| Tier | n | vanilla | aug_2x | tstr_2x | tstr_2x − vanilla † | tstr_2x − aug_2x [95% CI] |
|---|---|---|---|---|---|---|
| Non-responders (<45%) | 14 | 41.0% | 51.9% | 51.4% | +10.4 | −0.5 [−4.5, +4.0] |
| Data-limited (45–56%) | 41 | 51.1% | 57.4% | 52.8% | +1.7 | **−4.6** [−7.8, −1.3] |
| Learning (56–70%) | 42 | 62.5% | 61.2% | 56.1% | −6.4 | **−5.2** [−7.9, −2.4] |
| Strong (>70%) | 9 | 79.5% | 72.5% | 61.9% | −17.6 | **−10.6** [−16.5, −4.9] |

† Biased by regression to the mean: tiers are selected on vanilla accuracy, so the vanilla score of the top tier is optimistically high and that of the bottom tier pessimistically low. Augmentation shows the same pattern (+10.9 / −7.0), and so does vanilla itself when tiers are defined on another run ([README, results §4](../README.md#4-which-subjects-lose-most-and-a-selection-bias-check)). The last column compares two arms that were not used for tiering and is not affected.

Read against augmentation (the unbiased column), the population synthetic is no worse than augmentation for Non-responders but falls behind it in every tier where the classifier has a usable signal, and the gap grows with tier: −4.6 (data-limited), −5.2 (learning), −10.6 (strong, n = 9). This is consistent with the dilution hypothesis in `phase0_baseline_report.md` §10.5: population-averaged synthetic epochs wash out the individual ERD/ERS topography that the good subjects' classifiers depend on, pulling everyone toward a blurred population mean. Augmentation, which only perturbs the subject's own real epochs, preserves that topography and therefore wins.

### 5.4 Figures

- `reports/figures/fig10_accuracy_vs_n.png` — augmented (rising to 59.5%) vs TSTR (flat ~55%) curves, with the 59.5% gate line; the two curves visibly diverge.
- `reports/figures/fig11_tier_breakdown_vs_n.png` — accuracy vs N by tier.
- `reports/figures/fig12_tstr_vs_aug_delta.png` — per-subject TSTR@60 − augmented@60, sorted, tier-colored; the deficit is dominated by the Learning/Strong tiers.
- `results/subject_deltas.csv` — full per-subject accuracies and deltas for all six arms.

---

## 6. Decision

**FAIL → pivot to subject-conditional generation.**

Population synthetic EEG does not beat naive signal augmentation at this rung — it trends below real-only data, and relative to augmentation the deficit is largest for the subjects whose neural signal is most learnable. The population generator, however spectrally faithful (Step 4 confirmed PSD overlap and non-memorization), is *subject-agnostic* by construction, and that is the flaw: a good BCI classifier is a per-subject spatial filter, and averaging across 106 subjects erases the very structure it needs.

The next research direction (Phase 1) is **subject-conditional generation** — conditioning the diffusion model on a subject-identity embedding so it produces subject-specific synthetic epochs that preserve individual topography. This directly targets the Block 2 spatial-filter bottleneck (`phase0_baseline_report.md` §10.5). (At the time this was expected to require a stricter subject-holdout leakage scheme. The Pivot 2 report §2 shows it does not: run-holdout already guarantees the generator never sees the test run of any subject.)

This is a legitimate, publishable Phase 0 result: the experiment was designed so that "population synthetic doesn't help" is as informative as the alternative, and it cleanly motivates the subject-conditional rung as the correct next step rather than a rescue.

---

**Follow-up (Pivot 1).** Before committing to subject-conditional generation, the leading alternative explanation — "the imagery data was too weak a teacher" — was tested by enriching the generator with real, high-SNR motor-execution data (`reports/pivot_iter01_exec_conditioned_tstr_report.md`). It **also failed, with zero lift** over the imagery-only synthetic here (54.7% → 54.7%, paired Δ +0.01%, p=0.62), confirming that the bottleneck is subject-agnostic population averaging rather than data quality/domain, and reinforcing subject-conditional generation as the next step.
