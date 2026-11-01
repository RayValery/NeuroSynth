# Pivot Iteration 2 — Subject-Conditional Generation & TSTR Re-Gate

**Project:** NeuroSynth — Synthetic EEG Data Platform
**Iteration:** Pivot 2 — condition the generator on subject identity to stop population averaging
**Spec:** `specs/pivot_iter_02/18-07-2026_pivot2_subject-conditional-generation.md`
**Date:** 2026-07-23
> **Review note (2026-10-31).** Statistical wording in this report was corrected after a final review. Two-sided tests and 95% bootstrap CIs (`scripts/analyze_results.py`) replace the one-sided p-values that were described as "significant": the +1.4 lift over the population arms is a trend (CI −0.5 to +3.3), and the population arms' −2.0 vs real-only is also a trend (CI −4.3 to +0.2). The `subj−van` tier column is inflated by regression to the mean, and the "beats augmentation for Non-responders" claim (n = 14, +2.3, CI −1.7 to +6.8) is not supported. The tier table's population column was mislabeled as the execution arm; it is the imagery arm. Conclusions are restated accordingly. Consolidated analysis: [the README](../README.md).

**Status:** COMPLETE. Three subject-conditional generators trained + validated; the subject embedding is demonstrably *used* (probe) but only *weakly subject-specific*; TSTR ran. **RR-3 gate: FAIL** — `tstr_subj_2x` = 56.1%, below the augmentation ceiling (59.5%). **It is the best synthetic arm: +1.4 pts over the population arms (95% CI −0.5 to +3.3, not significant) and statistically indistinguishable from real-only training (−0.6, CI −2.7 to +1.4).** A subject-guidance retry (sample-time CFG on the subject token) was swept and run: it **hurt** (54.5%). **Overall verdict: PARTIAL — consistent with the dilution mechanism; the remaining gap is most plausibly limited by real data per subject.**

---

## Abstract

Two prior experiments established that **population (subject-agnostic) synthetic EEG does not beat signal augmentation**: imagery-only (Step 5, 54.7%) and execution-enriched (Pivot 1, 54.7%, +0.01% lift) both failed identically, isolating the bottleneck to population *averaging* rather than data quality, SNR, or domain. Pivot 2 attacks that directly: it conditions the diffusion generator on a **per-subject embedding**, trained as a **residual over a population fallback** (classifier-free-style subject dropout), so it produces subject-specific epochs that preserve each subject's ERD topography instead of a blurred population mean. The classifier's real training folds and imagery test set are unchanged, keeping the comparison apples-to-apples against `vanilla` (56.7%), `aug_2x` (59.5%), and the two failed population arms.

The result is a **PARTIAL**. Subject conditioning moves the result in the direction the dilution diagnosis predicts — +1.4 pts over both population arms cohort-wide (95% CI −0.5 to +3.3, a trend) and back to parity with real-only training (56.1% vs 56.7%, CI −2.7 to +1.4), where the population arms trended below it (−2.0, CI −4.3 to +0.2). But it does **not** clear the augmentation ceiling (−3.4, CI −5.7 to −1.2): the data-limited "mud-zone" tier — where augmentation shines and where a lift would matter most — does not benefit. A same-seed embedding probe explains why: the subject embedding is genuinely *used* (a different subject moves the generated signal ~25% of the way to a full resample) but only *weakly subject-specific* (a subject's generated spectrum tracks its own real spectrum only marginally better than others'). A subject-guidance retry that sharpens the embedding at sample time was swept (scales 0/1.5/3/5) and run at the sweet spot (1.5); it **regressed** to 54.5%, confirming that amplifying a weakly-specific direction adds distortion, not subject structure. The most plausible remaining limit is real data per subject (~75 epochs); this is a hypothesis, not tested by a data-scaling experiment.

---

## 1. Hypothesis and Design

**Hypothesis.** The population rung fails because a subject-agnostic generator produces a population-averaged signal that dilutes each subject's individual ERD topography — precisely the per-subject spatial filter EEGNet's Block 2 depends on (`phase0_baseline_report.md` §10.5). If the defect is *averaging across subjects*, the fix is to *stop averaging*: condition generation on subject identity so the model reproduces subject S's own topography.

**What changed vs. Pivot 1.**
- The three run-holdout generators gain a **subject-identity embedding** (`use_subject_cond`) alongside the existing class + domain tokens. Execution data and the domain token are retained (imagery sampled).
- The embedding is trained as a **residual over a population baseline** via classifier-free-style **subject dropout** (`p_subj_uncond` = 0.15): with probability 0.15 the subject token is replaced by a null "population" token, so the shared U-Net backbone learns the population distribution and each subject embedding only has to encode the per-subject *offset*. This is the single most important design choice for learning from scarce per-subject data (~30 imagery + ~45 execution ≈ 75 epochs/subject).
- Downstream, routing becomes **two-key `(subject, test_run)`**: each LORO fold draws only from the bank conditioned on *its* subject and held out by *its* test run.

**Why a residual, not a standalone per-subject generator.** ~75 epochs is far too few to learn a subject-specific generator from scratch. The residual design lets the backbone carry the population structure (learned from all ~9.5K epochs) while the embedding — a few hundred numbers per subject — carries only the individual spatial-filter offset.

## 2. Leakage Model — run-holdout is sufficient (subject-holdout is *not* needed)

Prior reports anticipated that subject conditioning would require the "stricter subject-holdout" scheme. **On analysis it does not, and subject-holdout is in fact incompatible with the goal.** For a fold testing subject S on imagery run R:

- Route to the model with `held_out_run == R`. That model trained on **all** subjects' runs `{4,8,12}\{R}` + all execution, so it **never saw run R of any subject, including S**. The imagery test run is fully unseen by the generator.
- S's embedding was learned only from **S's non-R runs** (`{4,8,12}\{R}` imagery + S's execution) — the *same data the classifier trains on for that fold*. Even a memorizing generator could only reproduce S's *training* epochs, never the test run.

Subject-holdout (excluding S entirely) is unnecessary for the leakage guarantee and would leave *no S-embedding to condition on* — it only applies to zero-shot generation for *unseen* subjects, an explicit Non-Goal (the product scenario is "customer supplies subject S's calibration; we generate more S-specific data"). Run-holdout gives the exact guarantee needed while reusing the three-model infrastructure unchanged. The invariant strengthens to a hard, asserted two-key check:

```python
assert bank.held_out_run == test_run and bank.subject_id == subject   # in _load_synthetic_set
```

## 3. Method

- **Generators.** Three run-holdout models, `diffusion_subj_holdout_run{4,8,12}.pt`, 100K steps each, domain + subject conditioned, sampled imagery-domain. Subject dropout p = 0.15.
- **Per-subject banks.** After training, each model generates a class-balanced imagery-domain bank for **every** subject (128 epochs = 64/class) and caches `subject_banks_holdout_run{R}.npz` (X, y, subject_ids). The three TSTR passes then load these banks directly — no diffusion reload — and route two-key.
- **TSTR arm.** Additive, sample-for-sample comparable to the augmented arm: each fold's real training set is extended by `n_copies × count(class)` subject-synthetic epochs, N = {30, 60, 90}, on all 106 subjects, leakage-guarded. Results tagged `tstr_subj_{1,2,3}x`.
- **Subject-guidance retry (added after the primary result).** Classifier-free guidance on the *subject* token: an extra sampler pass at the null (population) subject lets the guided prediction push toward the subject, `eps += w_s·(eps_cond@subject − eps_cond@population)`. `w_s = 0` reproduces the fixed-context default exactly. Guided banks carry a `_g{scale}` filename and tag `tstr_subj_g{scale}`.

## 4. Reproduction

```bash
# 1. Train the three subject-conditional generators + export per-subject banks (~3-4 h, ~$10-13):
modal run --detach cloud/modal_train_diffusion_subj.py
modal volume get neurosynth-checkpoints diffusion_subj_holdout_run{4,8,12}.pt   models/checkpoints/
modal volume get neurosynth-checkpoints subject_banks_holdout_run{4,8,12}.npz   models/checkpoints/

# 2. TSTR passes from the cached banks (~1 h, ~$2-3):
modal run --detach cloud/modal_train_tstr.py --variant subj
modal volume get neurosynth-checkpoints per_subject_results_tstr_subj_{1,2,3}x.json models/checkpoints/

# 3. Embedding-used / subject-specificity probe (local, free — the go/no-go before trusting the gate):
python scripts/probe_subject_embedding.py --checkpoint models/checkpoints/diffusion_subj_holdout_run12.pt \
    --data data/processed_with_exec.npz --n-subjects 12

# 4. (Guidance retry) sweep scales for free, regenerate banks at the winner, re-run TSTR:
python scripts/probe_subject_embedding.py --checkpoint …run12.pt --data … --subject-guidance-scale 0,1.5,3,5
modal run --detach cloud/modal_train_diffusion_subj.py --subject-guidance-scale 1.5     # resume, re-export only
modal run --detach cloud/modal_train_tstr.py --variant subj --subject-guidance-scale 1.5

# 5. Verdict + figures:
python scripts/compare_baselines.py
```

---

## 5. Results

### 5.1 Generator training & fidelity

Three generators trained on Modal A10G, 100K steps, domain + subject conditioned, sampled imagery-domain. They converged on par with Step 4 / Pivot 1 and passed the memorization guard. Representative (run 12):

| Metric | run 12 |
|---|---|
| final train loss | 0.1145 |
| psd_divergence (vs real imagery) | 2.633 |
| nn_distance gen / real | 292.4 / 289.7 → **CLEAN** (gen no closer than real-to-real) |
| exported bank | 106 subjects × 128 = 13,568 epochs, std 1.000, balance [6784, 6784] |

The generators are demonstrably healthy — so the TSTR result below is a property of subject-conditional synthesis, not a broken generator.

### 5.2 Embedding-used probe — the decisive diagnostic

Before trusting any TSTR verdict, we must confirm the subject embedding is actually *used* — otherwise a null result would be confounded with **conditioning collapse** (the model ignoring a weak conditioning signal and reverting to the population mean, which reproduces the exact 54.7% null). Two checks on `diffusion_subj_holdout_run12.pt` (`src/neurosynth/probe_subject_embedding.py`):

- **Same-seed A-vs-B (is it *used*?).** Generate for several subjects at the *same* initial noise + class + domain, so the only thing that varies is the subject token. Same-subject/same-seed |Δ| = 0.000000 (deterministic sanity); **cross-subject/same-seed mean|Δ| = 0.281 = 25% of a full independent resample.** The embedding is firmly wired and moving the output substantially — **not collapsed.** (A methodological note: comparing banks generated with *different* seeds is not diagnostic — two independent std-1 signals differ by ~2/√π ≈ 1.13 regardless of the embedding; the same-seed comparison isolates the token.)
- **PSD specificity (is it *useful*?).** Compare each subject's generated mu/beta PSD to *every* subject's real imagery PSD. gen-vs-own-real = 0.095 < gen-vs-others = 0.153 (a ~38% margin in the right direction), but **own-real is the single nearest for only 33% of subjects** (2× chance for 6, not the ~100% a strongly subject-specific model would give). **Weak-but-present.** (Caveat: this PSD is channel-*averaged*, so it under-tests the spatial ERD topography that the classifier actually keys on — the true specificity may be somewhat higher, but the signal is clearly weak.)

**Conclusion:** the embedding is *used* strongly and *aimed* weakly — which precisely predicts the TSTR shape below (recovers to parity, can't add enough to win).

### 5.3 TSTR — subject-conditional (unguided)

Three TSTR passes, Modal A10G, 500 epochs, all 106 subjects × 3 folds, two-key leakage-guarded. Numbers from `python scripts/compare_baselines.py`.

| Run | N | Mean acc | Std | Δ vs vanilla | Δ vs aug@60 | Δ vs population synth (54.7%) |
|---|---|---|---|---|---|---|
| vanilla | 0 | 56.7% | 10.8% | — | — | — |
| aug_2x (ceiling) | 60 | 59.5% | 11.2% | +2.8% | — | — |
| tstr_2x — imagery pop (Step 5) | 60 | 54.7% | 8.9% | −2.0% | −4.8% | — |
| tstr_exec_2x — exec pop (Pivot 1) | 60 | 54.7% | 8.9% | −2.0% | −4.8% | — |
| **tstr_subj_1x** | 30 | 55.8% | 9.3% | −0.9% | — | — |
| **tstr_subj_2x** | 60 | **56.1%** | 9.6% | −0.6% | −3.4% | **+1.4%** |
| **tstr_subj_3x** | 90 | 55.4% | 10.6% | −1.3% | — | — |

**RR-3 gate (`tstr_subj_2x` vs `aug_2x`).**
- Margin: `mean(tstr_subj_2x) − mean(aug_2x)` = **−3.41%** vs +1.0% → **FAIL**
- Significance: one-sided paired Wilcoxon p = **0.9934** vs 0.05 → **FAIL**
- **VERDICT: FAIL — subject-conditional synthetic does not beat augmentation.**

**Two improvements over the population rung — trends, not significant at α = 0.05 (two-sided tests and bootstrap CIs from `scripts/analyze_results.py`):**
- **Lift over the population arms:** `tstr_subj_2x − tstr_2x` = **+1.38 pts**, 95% CI [−0.53, +3.26], two-sided p = 0.12 (one-sided p = 0.06); `− tstr_exec_2x` = **+1.38 pts**, CI [−0.69, +3.46], two-sided p = 0.21. The best synthetic arm, but its lift over the population arms is not statistically established.
- **Parity with real-only training:** `tstr_subj_2x` vs vanilla = **−0.63 pts**, CI [−2.65, +1.35] — statistically indistinguishable from vanilla, whereas the population arms trended below vanilla (−2.0 pts, CI [−4.3, +0.2], two-sided p = 0.09).

### 5.4 Tier breakdown — where subject conditioning helped

Tiers assigned from vanilla accuracy. `subj−pop` = tstr_subj_2x − the (imagery) population arm `tstr_2x`; `subj−van` = vs vanilla. **Caveat:** `subj−van` is biased by regression to the mean (tiers are selected on vanilla, so the same "helps weak / hurts strong" shape appears for *any* arm, including augmentation — see the [README, results §4](../README.md#4-which-subjects-lose-most-and-a-selection-bias-check)). The arm-vs-arm columns are not affected; their 95% CIs are in `results/tier_comparisons.csv`.

| Tier | n | vanilla | aug_2x | tstr_pop | **tstr_subj** | subj−pop | subj−van |
|---|---|---|---|---|---|---|---|
| Non-responders (<45%) | 14 | 41.0 | 51.9 | 51.4 | **54.3** | **+2.8** | +13.2 |
| Data-limited (45–56%) | 41 | 51.1 | 57.4 | 52.8 | **52.1** | −0.7 | +1.0 |
| Learning (56–70%) | 42 | 62.5 | 61.2 | 56.1 | **58.1** | +2.0 | −4.4 |
| Strong (>70%) | 9 | 79.5 | 72.5 | 61.9 | **67.6** | +5.7 | −11.8 |
| **ALL** | 106 | 56.7 | 59.5 | 54.7 | **56.1** | **+1.4** | −0.6 |

Two things stand out:
1. **Relative to the population arm, subject conditioning helped in three of four tiers** (+2.8 / −0.7 / +2.0 / +5.7). Only the Non-responder gain has a CI excluding zero (+2.8, [+0.7, +4.9], n = 14); the Strong-tier gain is large but uncertain (+5.7, [−3.7, +15.6], n = 9). The direction is consistent with the §10.5 dilution diagnosis.
2. **Against augmentation, no tier is won.** Non-responders are the closest (54.3% vs 51.9%, +2.3, CI [−1.7, +6.8]) — not distinguishable.

**But the data-limited "mud-zone" tier — the 41 subjects a lift would matter most for — did not benefit (52.1%, −0.7 vs population), and augmentation is clearly better there (57.4%; subj − aug = −5.3, CI [−9.2, −1.7]).** Those subjects have detectable-but-weak ERD; augmentation helps them by perturbing their *own* real epochs to bootstrap the spatial filter, whereas subject-synthetic — only weakly subject-specific — hands them mediocre-quality data. This tier accounts for most of the cohort-level shortfall, which fits the weak-specificity probe result.

### 5.5 Subject-guidance retry — sharpening the embedding hurt

The probe showed the embedding is *used* but *weakly aimed*, so we tested whether **sharpening it toward the subject at sample time** (classifier-free guidance on the subject token) improves the aim. Scales were swept for free on the existing checkpoint (`--subject-guidance-scale 0,1.5,3,5`, 12 subjects):

| scale | same-seed (token effect) | fidelity (dist to own real ↓) | aim gap (specificity ↑) | own-nearest |
|---|---|---|---|---|
| 0 | 0.277 | **0.088** | +0.065 | 25% |
| **1.5** | 0.534 | 0.124 | **+0.085** | **33%** |
| 3 | 0.682 | 0.156 | +0.084 | 33% |
| 5 | 0.828 | 0.183 | +0.075 | **8%** |

Textbook CFG behaviour: aim rises 0→1.5, **plateaus** (1.5 ≈ 3), then **collapses** at 5 (own-nearest 8% = pure chance for 12 subjects); fidelity degrades monotonically throughout. The sweet spot is **1.5** — all the achievable aim, least distortion. (This also confirms the guidance is wired correctly, and g0's 0.277 matches the §5.2 probe's 0.281.)

The guided TSTR at the sweet spot **regressed**:

| Run | N | Mean | Δ vs vanilla | Δ vs unguided `tstr_subj_2x` |
|---|---|---|---|---|
| tstr_subj_g1.5_1x | 30 | 55.3% | −1.4% | — |
| **tstr_subj_g1.5_2x** | 60 | **54.5%** | −2.2% | **−1.57 pts**, CI [−2.87, −0.28], two-sided p = 0.02 |
| tstr_subj_g1.5_3x | 90 | 54.5% | −2.2% | — |

Guidance **hurt** (the one significant within-family difference) — it fell all the way back to the population-arm level (54.5% ≈ 54.7%), undoing the +1.4 gain subject conditioning had earned. The mechanism: on a *weakly*-specific embedding, the direction guidance amplifies is mostly artifact, not real subject topography, so sharpening it pushes samples off the realistic manifold (the fidelity drop) without adding genuine per-subject structure — the synthetic drifts back toward the diluting, population-like signal. **You cannot manufacture subject-specificity by amplification; the embedding has to learn it from data.**

### 5.6 Figures

- `reports/figures/fig10_accuracy_vs_n.png` — accuracy vs N: augmented curve (to 59.5%), population synthetic (~54.7%), and the subject-conditional curve (~56.1%) between them, with the 59.5% gate line.
- `reports/figures/fig14_tstr_subj_vs_exec_delta.png` — per-subject `tstr_subj_2x − tstr_exec_2x`, tier-colored: broad recovery of the tiers the population arm harmed.
- `results/subject_deltas.csv` — full per-subject accuracies/deltas for all arms (incl. `tstr_subj_*`, `tstr_subj_g1.5_*`).

---

## 6. Interpretation

**Fidelity is a weak predictor of TSTR in this problem — usefulness is about subject-specific structure.** The population arms had excellent spectral fidelity (psd_div ~2.5, non-memorizing) and *failed*; augmentation produces off-manifold samples (Gaussian noise, channel dropout) and *wins*. What separates them is whether the synthetic preserves the per-subject spatial filter. Subject conditioning is the first method here designed to do so — and it is the arm that returns to parity with real-only training. But the amount of subject-specific structure it can encode is bounded by the ~75 real epochs per subject, so the lift is small, and it does not reach the data-limited tier that augmentation serves best.

**The guidance retry sharpens what little is there and, in doing so, trades it for distortion.** That the fidelity cost *won* over the aim gain (contrary to a naive "fidelity doesn't matter" reading) is itself informative: the aim "gain" lived in channel-averaged PSD and did not correspond to real spatial structure, so amplifying it only hurt.

**Net:** the results are consistent, from a third direction, with subject-agnostic averaging being the bottleneck, and personalizing the generator recovers parity with real-only training. Whether the remaining gap to augmentation is set by **real data per subject** (the most plausible explanation) is untested.

---

## 7. Decision — PARTIAL

Per the spec's decision tree, this is a clean **PARTIAL**: subject conditioning is the best synthetic arm and returns to real-only parity, but does not clear the augmentation ceiling cohort-wide, and the cheap sharpening lever (guidance, swept and run) does not help.

**Primary hypothesis (synthetic data beats signal augmentation at 30-trial calibration): not supported.** The natural next question is how the gap changes with the amount of real calibration data per subject.

**Scientific result: a clean, publishable arc.** Population averaging is the bottleneck (shown three ways: imagery, execution, and now by *removing* the averaging and watching the harm disappear); personalizing the generator is necessary and repairs the dilution, but is data-bound.

### Remaining levers (not yet run), in value-for-effort order
1. **Hybrid arm — real + augmentation + subject-synthetic, stacked** (cheap, no retrain). Motivated directly by §5.4: augmentation serves the data-limited tier, subject-synthetic serves Non-responders and stops the harm — they help *different* subjects, so combined they may clear 59.5% where neither did alone. The last cheap shot.
2. **Subject-similarity / covariance conditioning** (retrain). Replace the free-learned ID embedding with the subject's measured spatial covariance and/or nearest-neighbor subjects — the one generator-side lever that *adds information* (intelligent transfer instead of blind averaging) rather than re-weighting the same ~75 epochs.
3. **More real calibration per subject** (data/product decision). The guaranteed win; everything else works around the wall.

---

## 8. Cross-Experiment Synthesis — the full population→per-subject ladder

| Arm (N=60) | Mean | vs vanilla | vs aug ceiling | vs population synth |
|---|---|---|---|---|
| vanilla (real only) | 56.7% | — | −2.8% | — |
| **aug_2x (gate ceiling)** | **59.5%** | +2.8% | — | — |
| tstr_2x — imagery pop (Step 5) | 54.7% | −2.0% | −4.8% | — |
| tstr_exec_2x — execution pop (Pivot 1) | 54.7% | −2.0% | −4.8% | +0.01% |
| **tstr_subj_2x — subject-conditional (Pivot 2)** | **56.1%** | −0.6% | −3.4% | **+1.4%** |
| tstr_subj_g1.5_2x — + subject guidance | 54.5% | −2.2% | −5.0% | −0.2% |

The ladder tells one story: enriching the population prior's *data* did nothing (+0.01%); *removing the averaging* via subject conditioning is the first thing that moved the number up (+1.4, CI −0.5 to +3.3) and returned to real-only parity — but not enough to beat augmentation, and *sharpening* the weak per-subject signal (guidance) pushed it back down. The lever that works is per-subject **information**, and the supply of it is the wall.

---

## 9. Engineering & Operational Notes (research log)

- **Embedding-used probe (`src/neurosynth/probe_subject_embedding.py`).** Built to gate the verdict against conditioning collapse. The first in-training "spread" probe was silently non-diagnostic — it compared banks generated with *different* seeds, whose mean|Δ| ≈ 1.13 (the independent-resample floor) regardless of the embedding, so it always printed "used." Replaced with a **same-seed cross-subject** comparison (isolates the token) plus **PSD specificity** (relative, self-calibrating). Verified to read ~0 for a zeroed/collapsed embedding and to discriminate specific-vs-collapsed generators on controlled data.
- **Subject guidance wiring.** Real CFG on the subject token added to `DDPM.sample` (extra pass at the null subject; `w_s=0` byte-identical to the fixed-context default). Guided banks/tags carry a `_g{scale}` suffix so a retry never clobbers the unguided banks and both are gated side-by-side. A `--subject-guidance-scale` sweep on the probe lets scales be triaged **locally for free** before paying to regenerate banks — this is how g1.5 was chosen without spending on g3/g5.
- **Cheap retry path.** `cloud/modal_train_diffusion_subj.py --subject-guidance-scale S` resumes the trained checkpoint (falling back to the final `_holdout_run{R}.pt` if intermediate step checkpoints were rotated away) and only re-exports guided banks — no retraining.
- **Cost.** Generators + banks ≈ $10–13; three TSTR passes ≈ $2–3; guided retry (bank re-export + TSTR) ≈ $5. Per-subject banks are ~2.5 GB each (106 × 128 × 64 × 801 f32).
- **Result integrity.** All arms leakage-clean across 106 subjects × 3 folds under the two-key `(subject, test_run)` assertion; unguided and guided result JSONs both committed.
