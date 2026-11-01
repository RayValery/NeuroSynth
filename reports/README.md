# Research log

The project's results and conclusions are in the [top-level README](../README.md). This folder is
the record of how it got there: one report per stage, written as each experiment finished, in
chronological order. Each design was specified in advance in [`specs/`](../specs/README.md).

| # | Report | Stage | Outcome |
|---|---|---|---|
| 1 | [`phase0_baseline_report.md`](phase0_baseline_report.md) | Data pipeline, EEGNet, evaluation protocol (six iterations), real-only and augmentation baselines, population diffusion model and the sampler fix | Real-only 56.7%, augmentation 59.5%; healthy, non-memorizing generators |
| 2 | [`phase0_augmented_online_report.md`](phase0_augmented_online_report.md) | Online augmentation run | Not sample-matched, so rejected as the comparison baseline |
| 3 | [`phase0_tstr_report.md`](phase0_tstr_report.md) | Population generator vs. augmentation (the pre-registered test) | Fail: 54.7% |
| 4 | [`pivot_iter01_exec_conditioned_tstr_report.md`](pivot_iter01_exec_conditioned_tstr_report.md) | Generator also trained on motor-execution data | Fail, no change: 54.7% |
| 5 | [`pivot_iter02_subject_conditional_tstr_report.md`](pivot_iter02_subject_conditional_tstr_report.md) | Subject-conditional generator, embedding probe, subject guidance | Fail, best synthetic arm: 56.1% |

**Review notes.** Each report carries a dated note (2026-10-31) from a final review. It corrected
tier-level claims inflated by regression to the mean, one-sided p-values described as significant,
an EEGNet benchmark claim and a parameter table. The reports are otherwise unchanged, so their
wording reflects what was known at the time. Where they disagree with the README, the README is current.

**Naming.** "TSTR" (train on synthetic, test on real) is used for every arm that adds generated data.
"Phase 0", "pivot" and "RR-3" (the success criterion) come from the original plan in
`specs/iter_01`.

**Figures.** `figures/` holds the figures these reports reference. All except the three
`diffusion_fidelity_run*.png` plots (written by generator training) are rebuilt from
`results/per_subject/` with `make figures`.
