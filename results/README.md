# Results

Every number in the top-level README can be traced to a file in this folder.

## Summary tables

Written by `scripts/analyze_results.py` (run via `make figures`) from the per-subject files below.

| File | Contents |
|---|---|
| `summary.csv` | One row per arm: mean accuracy over 106 subjects, and the paired difference vs real-only training and vs signal augmentation (2×) with 95% bootstrap CIs |
| `paired_comparisons.csv` | All paired comparisons between arms at 60 extra samples per fold: mean difference, 95% CI, two-sided Wilcoxon p, subjects better / worse |
| `tier_comparisons.csv` | The same comparisons within subject tiers (tiers defined on real-only accuracy) |
| `tier_selection_check.csv` | Regression-to-the-mean check behind `figures/tier_selection_check.png` |
| `subject_deltas.csv` | Per-subject accuracy and difference vs real-only, every arm (written by `scripts/compare_baselines.py`) |

Statistics: differences are in percentage points, paired over subjects; CIs use 10,000 bootstrap
resamples of subjects (seed 0).

## Per-subject results (`per_subject/`)

The raw outcome of every training run: one JSON per arm, 106 subjects × 3 leave-one-run-out folds.

| File | Arm | Extra samples per fold |
|---|---|---|
| `per_subject_results_vanilla.json` | real data only | 0 |
| `per_subject_results_augmented_{1,2,3}x.json` | additive signal augmentation | 30 / 60 / 90 |
| `per_subject_results_tstr_{1,2,3}x.json` | population generator, imagery | 30 / 60 / 90 |
| `per_subject_results_tstr_exec_{1,2,3}x.json` | population generator, + execution data | 30 / 60 / 90 |
| `per_subject_results_tstr_subj_{1,2,3}x.json` | subject-conditional generator | 30 / 60 / 90 |
| `per_subject_results_tstr_subj_g1.5_{1,2,3}x.json` | subject-conditional + subject guidance 1.5 | 30 / 60 / 90 |
| `per_subject_results_augmented.json` | online augmentation (reference only, not sample-matched) | 0 |
| `per_subject_results_vanilla_first_loro_attempt.json` | real data only, first (superseded) training setup with early stopping; used for research-log fig5/fig8 | 0 |

Schema: a list of `{"subject": int, "mean_accuracy": float, "fold_accuracies": [float, float, float]}`
(accuracies in 0–1). All runs use seed 42.

Training writes these files to `models/checkpoints/`; `make collect` copies them here.
