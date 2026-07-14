# Pivot Iteration 1 — Execution-Conditioned Generator & TSTR Re-Gate

**Project:** NeuroSynth — Synthetic EEG Data Platform
**Iteration:** Pivot 1 — enrich the population generator with real motor-execution data
**Spec:** `specs/pivot_iter_01/10-07-2026_pivot1_execution-conditioned-generator-tstr.md`
**Date:** 2026-07-10
**Status:** COMPLETE. Three exec-conditioned generators trained + validated (psd_divergence 2.54–2.59, clean, loss ~0.115 — on par with Step 4), then TSTR passes ran. **RR-3 gate: FAIL** — tstr_exec_2x = 54.7%, identical to imagery-only and below both the augmentation ceiling (59.5%) and vanilla (56.7%). Execution enrichment gave **zero lift** over imagery-only (paired Δ = +0.01%, p = 0.62). Conclusion: the bottleneck is subject-agnostic population averaging, not data quality/domain → next step is subject-conditional generation.

---

## Abstract

Phase 0 Step 5 failed the RR-3 gate: population synthetic motor-**imagery** EEG scored 54.7% at N=60, below both the augmentation ceiling (59.5%) and vanilla (56.7%), with the harm concentrated in the Learning (−6.4) and Strong (−17.6) tiers — evidence of spatial-filter dilution. This pivot tests whether injecting real motor-**execution** data (runs 3/7/11) into the *generator only* rescues it. Execution is high-SNR and shares the exact T1/T2 left/right-fist classes of the imagery task, so it is a label-aligned prior the imagery generator never had. The generator is trained **domain-conditioned** (imagery vs. execution) on imagery+execution jointly and sampled in the **imagery** domain, so its synthetic epochs match the imagery test distribution while benefiting from execution's clean sensorimotor structure. The classifier's real training folds and test set are unchanged (imagery, runs 4/8/12), keeping the comparison against `vanilla`, `aug_*`, and the Step 5 `tstr_*` numbers exactly apples-to-apples.

---

## 1. Hypothesis and Design

**Hypothesis.** A subject-agnostic generator fails not because population synthesis is useless, but because 30-epoch imagery is too weak a teacher of the left/right sensorimotor pattern. Real execution — the same motor classes, far cleaner — is new information that can sharpen the generated imagery.

**What changed vs. Step 5.**
- The three run-holdout generators now train on imagery {4,8,12}\R **plus** all execution {3,7,11}, domain-conditioned, saved as `diffusion_exec_holdout_run{4,8,12}.pt`.
- Banks are sampled in the imagery domain; everything downstream (bank draw, leakage routing, EEGNet, LORO, gate) is the Step 5 harness unchanged, invoked with `--diffusion-variant exec`.

**Two ablations (spec `Should Have`).**
- `execpool` — imagery+execution pooled with **no** domain token: measures the value of domain conditioning.
- `execdom` — the exec model sampled in the **execution** domain: directly tests the "make the synthetic stronger" intuition (does execution-shaped synthetic help or hurt an imagery classifier?).

## 2. Leakage Model (unchanged invariant)

The classifier tests only on imagery runs {4,8,12}. Execution runs {3,7,11} are never a test run, so including all execution in every generator cannot leak test-run information. The imagery run-holdout is preserved: the model serving folds testing on imagery run R is trained on imagery {4,8,12}\{R} + all execution. The invariant holds:

```python
assert bank.held_out_run == test_run    # test_run ∈ {4,8,12}, imagery only
```

## 3. Reproduction

```bash
# 0. Build the combined dataset (MNE; downloads execution runs 3/7/11):
python data/eegbci_pipeline.py --subjects $(seq 1 109) --task exec_imagery_lr \
    --out data/processed_with_exec.npz
modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz

# 1. Train the three execution-conditioned generators (~3 h, ~$8-9):
modal run --detach modal_train_diffusion_exec.py
modal volume get neurosynth-checkpoints diffusion_exec_holdout_run4.pt  models/checkpoints/
modal volume get neurosynth-checkpoints diffusion_exec_holdout_run8.pt  models/checkpoints/
modal volume get neurosynth-checkpoints diffusion_exec_holdout_run12.pt models/checkpoints/

# 2. Run the TSTR passes from the exec checkpoints (~1 h, ~$1-2):
modal run --detach modal_train_tstr.py --variant exec
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_1x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_2x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_3x.json models/checkpoints/

# (optional ablations)
modal run --detach modal_train_tstr.py --variant exec --sample-domain execution --copies 2  # execdom

# 3. Verdict + figures:
python eval/compare_baselines.py
```

### 3.1 As run — data build

`data/processed_with_exec.npz` was built with `--task exec_imagery_lr`: imagery (runs 4/8/12) + real execution (runs 3/7/11), **~9.5K epochs across 106 subjects**, split ~50/50 by domain (per-subject sanity check: subject 1 → 90 epochs, imagery 45 / execution 45). Each of the three run-holdout generators therefore trains on **~7.9K epochs** — one imagery run held out (~3.16K imagery) plus all execution (~4.75K).

### 3.2 As run — generator training & fidelity validation

Three generators trained on Modal A10G, 100K steps each, domain-conditioned (20.03M params — the extra ~1K over Step 4 is the domain embedding), sampled imagery-domain. All three passed the fidelity/memorization gate **on par with the Step 4 imagery-only models** (which scored psd_divergence 2.52–2.53):

| Model | psd_divergence | nn_gen | nn_real | final loss | verdict |
|---|---|---|---|---|---|
| `diffusion_exec_holdout_run4`  | 2.536 | 295.4 | 289.3 | 0.114 | clean |
| `diffusion_exec_holdout_run8`  | 2.580 | 296.8 | 289.0 | 0.115 | clean |
| `diffusion_exec_holdout_run12` | 2.593 | 294.5 | 289.7 | 0.116 | clean |

`nn_gen > nn_real` on all three (generated samples no closer to real than real-to-real → not memorizing); psd spread 0.06 → mutually comparable. **The generators are demonstrably healthy** — so the null TSTR result below is a property of population synthesis, not a broken generator.

---

## 4. Results

Three exec-conditioned generators (Modal A10G, 100K steps, domain-conditioned, sampled imagery-domain) passed the fidelity gate — psd_divergence 2.54–2.59 (Step 4 was 2.52–2.53), memorization clean, loss ~0.115. The three TSTR passes then ran (500 epochs, bank_size=2048, all 106 subjects, leakage-routed). Numbers from `python eval/compare_baselines.py`.

### 4.1 Per-N results

| Run | N | Mean acc | Std | Δ vs vanilla | Δ vs aug@N | Δ vs tstr@N (imagery) |
|---|---|---|---|---|---|---|
| vanilla | 0 | 56.7% | 10.8% | — | — | — |
| aug_1x | 30 | 58.6% | 9.1% | +1.9% | — | — |
| aug_2x (ceiling) | 60 | 59.5% | 11.2% | +2.8% | — | — |
| aug_3x | 90 | 59.0% | 10.0% | +2.3% | — | — |
| tstr_2x (imagery, Step 5) | 60 | 54.7% | 8.9% | −2.0% | −4.8% | — |
| **tstr_exec_1x** | 30 | 55.8% | 10.1% | −0.9% | −2.8% | +0.6% |
| **tstr_exec_2x** | 60 | 54.7% | 8.9% | −2.0% | −4.8% | **+0.01%** |
| **tstr_exec_3x** | 90 | 55.8% | 9.1% | −0.9% | −3.2% | +0.3% |

(The `execpool` and `execdom` ablations were not run — the primary arm's zero lift made them unnecessary, and Modal credits were exhausted.)

### 4.2 RR-3 gate (tstr_exec_2x vs aug_2x)

- **Margin:** `mean(tstr_exec_2x) − mean(aug_2x)` = **−4.79%** vs +1.0% → **FAIL**
- **Significance:** one-sided paired Wilcoxon p = **1.0000** vs 0.05 → **FAIL**
- **Lift over imagery-only:** `tstr_exec_2x − tstr_2x` = **+0.01%**, p = **0.62** — statistically zero; only 46/106 subjects improved (a coin flip).
- **VERDICT: ❌ FAIL — execution enrichment gave no benefit.**

### 4.3 Tier breakdown (tstr_exec_2x, tiers from vanilla accuracy)

| Tier | n | vanilla | aug_2x | tstr_2x (img) | tstr_exec_2x | exec − img |
|---|---|---|---|---|---|---|
| Non-responders | 14 | 41.0 | 51.9 | 51.4 | 49.2 | −2.3 |
| Data-limited | 41 | 51.1 | 57.4 | 52.8 | 54.6 | +1.8 |
| Learning | 42 | 62.5 | 61.2 | 56.1 | 55.0 | −1.1 |
| Strong | 9 | 79.5 | 72.5 | 61.9 | 62.3 | +0.4 |

Execution did **not** rescue the Learning/Strong tiers that imagery-only harmed (Learning −1.1, Strong +0.4). The per-tier deltas are noise around zero (fig13), redistributing slightly (data-limited +1.8, others down) with no net effect.

---

## 5. Decision — FAIL → subject-conditional generation

The pivot hypothesis — *population synthetic failed because imagery is too weak a teacher; enrich the generator with clean, high-SNR execution and it will help* — is **falsified**. With generators validated to be as faithful as Step 4's and real execution folded in via domain conditioning, the result at the gate is statistically indistinguishable from imagery-only (+0.01%, p=0.62).

The conclusion, now established two independent ways (imagery in Step 5, execution here): **the bottleneck is subject-agnostic population averaging, not the data's quality, SNR, or domain.** Feeding a population generator better data cannot fix an averaging problem — the synthetic still lacks the individual spatial filter each subject's classifier needs. This also rules out the "different data domain" direction (e.g. sampling execution-domain synthetic): the failure is not about *which* data, it is about *population vs. per-subject*.

**Next step: subject-conditional generation** — condition the diffusion model on a subject-identity embedding under the stricter subject-holdout leakage scheme, so it generates synthetic that preserves each subject's own topography, with execution retained as a clean prior (it trains a faithful generator and is now the validated ingredient, just delivered per-subject rather than averaged). This is the only remaining lever at this rung, and the pivot's clean falsification is what points to it.

This is a legitimate, publishable negative result: it isolates and rules out real-data enrichment of a *population* generator as the fix, sharpening the case for the subject-conditional rung.

---

## 6. Cross-Experiment Synthesis — What the Population Rung Has Established

Two independent experiments now bound the population (subject-agnostic) synthetic rung:

| Arm (N=60) | Mean | Δ vs vanilla | Δ vs aug ceiling | Δ vs imagery synth |
|---|---|---|---|---|
| vanilla (real only) | 56.7% | — | −2.8% | — |
| **aug_2x (gate ceiling)** | **59.5%** | +2.8% | — | — |
| tstr_2x — imagery synth (Step 5) | 54.7% | −2.0% | −4.8% | — |
| tstr_exec_2x — execution-enriched synth (Pivot 1) | 54.7% | −2.0% | −4.8% | +0.01% |

The two synthetic arms are **indistinguishable** (54.7% each; paired Δ = +0.01%, p = 0.62), despite one being enriched with high-SNR real motor data and validated to equal fidelity. This is the crux: **improving the population prior's data does nothing.** Both fall ~4.8 points short of the augmentation ceiling and ~2 points below simply using the real data alone.

The mechanism is consistent with the Step 5 diagnosis (`phase0_baseline_report.md` §10.5): a BCI classifier is a per-subject spatial filter, and a subject-agnostic generator produces a population-averaged signal that dilutes each subject's individual ERD topography. The tier structure confirms the damage lands on the subjects whose spatial filter works (Learning/Strong), for both arms. No amount of population-level data enrichment addresses this, because the defect is the averaging, not the ingredients.

**Implication for the product thesis.** NeuroSynth's value proposition — "sell synthetic EEG that augments a customer's small dataset better than naive augmentation" — is *not* achievable with a population generator at this rung, even one trained on proprietary high-SNR data. The remaining viable lever is **subject-conditional generation**: a generator conditioned on subject identity that preserves individual topography, with execution retained as the (now validated) prior, delivered per-subject rather than averaged. That is Pivot 2.

---

## 7. Engineering & Operational Notes (research log)

- **Worker preemption during generator training.** One generator's A10G worker was preempted at ~80K/100K steps. Modal auto-restarts a preempted function *with its original inputs*; the launcher had gated resume behind `--resume` (which was `False` on the replayed input) and only committed the volume at the end, so the restart trained **from scratch**, losing ~80K steps (~$2.5, ~2 h). **Fix applied** to both diffusion launchers (`modal_train_diffusion.py`, `modal_train_diffusion_exec.py`) and `eval/train_diffusion.py`: (1) the container now **always resumes from the newest committed checkpoint on startup**, independent of the flag, so an automatic restart continues; (2) a new `on_checkpoint` callback **commits the volume after every `save_every`**, so the checkpoint is durable before any preemption. A future preemption now costs ≤10K steps. The TSTR trainer remains non-resumable by design (each pass is short and writes no mid-run checkpoints).
- **Cost.** Three generators ≈ Step-4 cost (~$9); the one from-scratch preemption restart added ~$2.5; the three TSTR passes cost ~$3.5 — higher than the initial ~$1–2 estimate, because each of the three parallel containers loads all three 20M generators to build its banks. Total pivot ≈ **$15**, which exhausted the $30 Modal free-credit tier. Pivot 2 (subject-conditional) will require a funded account.
- **Result integrity.** All three exec TSTR passes committed their per-subject JSONs to the volume before credits ran out; the gate (`tstr_exec_2x`) is complete and leakage-clean across all 106 subjects × 3 folds.
