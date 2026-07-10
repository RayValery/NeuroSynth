# Pivot Iteration 1 — Execution-Conditioned Generator & TSTR Re-Gate

**Status:** Draft
**Created:** 2026-07-10
**Depends on:** Phase 0 Step 4 (population DDPM, `eval/diffusion.py` + `eval/train_diffusion.py`) and Step 5 (TSTR harness in `eval/train_baseline.py` + `eval/compare_baselines.py`) — both complete.
**Feeds into:** The decision on whether real motor-execution data, injected into the *generator* only, lets population synthetic finally beat signal augmentation (RR-3), and whether to pursue subject-conditional generation next.

---

## Executive Summary

Phase 0 Step 5 failed the RR-3 gate: population synthetic motor-**imagery** EEG scored 54.7% at N=60 — below both the augmentation ceiling (59.5%) and vanilla (56.7%) — with the harm concentrated in the Learning (−6.4) and Strong (−17.6) tiers. The diagnosis was **spatial-filter dilution**: a subject-agnostic generator trained only on 30-epoch-per-subject imagery cannot manufacture discriminative structure it never saw, so it blurs the individual ERD topography the good subjects depend on.

This pivot tests a second hypothesis: **enrich the generator's learned manifold with real motor-execution data (runs 3/7/11)** — high-SNR recordings of the *same* left/right-fist motor classes — added to the diffusion model's training set only. The classifier's real training folds and its test set stay exactly as they were (motor imagery, runs 4/8/12), so the comparison against the existing `vanilla`, `aug_*`, and `tstr_*` numbers is apples-to-apples. The generator is trained **domain-conditioned** (imagery vs. execution) on imagery+execution jointly, then sampled in the *imagery* domain, so its synthetic epochs match the imagery test distribution while benefiting from the clean sensorimotor structure execution teaches the shared backbone. We regenerate the three per-run banks, re-run the additive TSTR arm at N = {30, 60, 90}, and re-evaluate RR-3.

The entire downstream harness (bank draw, leakage routing, EEGNet, LORO, gate verdict) is reused unchanged. New work is confined to: extracting execution runs in the pipeline, adding a domain-conditioning token to the diffusion model, a variant of the diffusion trainer/launcher, a `--diffusion-variant` switch in the TSTR harness, and the comparison/report plumbing for the new arm.

---

## Goals

- Produce a generator whose training manifold includes real motor-execution data, without changing the classifier's real training data or test data.
- Sample **imagery-domain** synthetic epochs from that generator (domain-conditioned), matching the imagery test distribution.
- Re-run the additive TSTR arm (`tstr_exec_{1,2,3}x`) at N = {30, 60, 90}, leakage-free, on all 106 subjects.
- Re-evaluate the RR-3 gate for this arm (`tstr_exec_2x` vs `aug_2x` = 59.5%) and quantify the lift over the Step 5 imagery-only arm (`tstr_2x` = 54.7%).
- Keep the comparison strictly apples-to-apples: identical LORO folds, identical N, identical seeds, identical EEGNet.

## Non-Goals

- **Changing the classifier's real data.** EEGNet still trains on real imagery folds and tests on the held-out real imagery run. Execution data never enters the classifier's real training set or its test set. (Adding execution to the *classifier* is a separate "reference ceiling" experiment, out of scope here.)
- **Subject-conditional generation.** This generator remains population-level / subject-agnostic. Subject conditioning is the next pivot if this one clears (or partially clears) the gate.
- **Retraining EEGNet or changing the LORO protocol, augmentation baseline, or gate statistics.** Those are frozen comparators.
- **Both-fists / feet tasks (runs 5/6/9/10/13/14).** Only left/right-fist execution (runs 3/7/11) is used, because it shares the exact T1/T2 class semantics of the imagery task (runs 4/8/12).
- **Beating the literature 65–75% ceiling.** That regime needs more *real* classifier data; this experiment tests whether execution as a generative prior beats augmentation, nothing more.

---

## Background & Context

### What Step 5 established (the thing we are trying to fix)

`reports/phase0_tstr_report.md`: three population DDPMs (`diffusion_holdout_run{4,8,12}.pt`), each trained on two of three imagery runs, generated class-balanced banks; each LORO fold drew N synthetic epochs from the model that never saw its test run. Result — a flat ~55% TSTR curve, below vanilla, RR-3 **FAIL** (paired Δ = −4.80%, p = 1.0000). The tier split was the key signal:

| Tier | n | vanilla | aug_2x | tstr_2x (imagery) |
|---|---|---|---|---|
| Non-responders | 14 | 41.0 | 51.9 | 51.4 |
| Data-limited | 41 | 51.1 | 57.4 | 52.8 |
| Learning | 42 | 62.5 | 61.2 | **56.1** |
| Strong | 9 | 79.5 | 72.5 | **61.9** |

Synthetic helped the weak tiers slightly but degraded the subjects whose within-subject spatial filter works. A population-averaged imagery generator, starved of per-subject structure, pulls everyone toward a blurred mean.

### Why execution data, and why it *can* escape the Step 5 wall

The Step 5 post-mortem's core argument was information-theoretic: synthetic data cannot add discriminative structure that is not already present in the generator's training set, and the imagery generator's training set was the same scarce, low-SNR imagery. **Motor-execution data breaks that constraint** — it is genuinely new information the generator did not previously have:

- **Same motor classes.** In EEGBCI, runs 3/7/11 are *real* open/close of the left (T1) or right (T2) fist; runs 4/8/12 are *imagining* the same. The class labels map identically (T1→0, T2→1), so execution is a label-aligned prior, not a different problem.
- **Much stronger, cleaner ERD/ERS.** Actual movement produces robust contralateral mu/beta desynchronization; imagery is a weak echo of the same spatial pattern. Execution therefore teaches the generator's backbone the *shape* of the sensorimotor manifold with far higher SNR.
- **It is data NeuroSynth can plausibly own.** Under the corrected product framing (a data-augmentation platform for research/BCI customers, not an assistive device), a generator whose prior is enriched by proprietary/extra data that a customer lacks is exactly the moat — provided the comparison is reported honestly (see Success Criteria).

### Pooled vs. domain-conditioned: the design choice (DECIDED — domain-conditioned)

Both options start the same way: mix imagery (4/8/12) and execution (3/7/11) into **one** training set and train a single generator. The only difference is whether each training epoch carries a **domain tag** (imagery vs. execution).

- **Pooled / no-domain (no tag).** The model learns the left/right-fist pattern from the mixed pile but cannot tell the two sources apart. Class-conditional sampling then yields a *blend* you cannot steer — and because execution's ERD is the more consistent, higher-SNR mode, the blend leans toward execution. Those epochs may not match the imagery test distribution, which risks re-introducing the very mismatch we are trying to avoid.
- **Domain-conditioned (tag = imagery/execution).** We add a domain token alongside the existing class token. The U-Net backbone still learns the shared sensorimotor manifold from *both* sources — this is where execution's clean signal helps — but the domain token lets the model separate imagery-specific from execution-specific detail. At sample time we condition on **domain = imagery**, so synthetic epochs live in the imagery manifold, *sharpened by* execution but not *shaped like* it. The same model can also be sampled with domain = execution at zero extra cost — a free, direct probe of the domain gap.

**Decision (2026-07-10, owner):** domain-conditioned joint training with imagery-domain sampling is the **primary arm**. The literal pooled/no-domain reading is retained as the `execpool` ablation (`Should Have`) — it is one config flag away (`use_domain_cond=False`) and its gap to the primary arm quantifies the value of the domain token.

### Why not simply generate a "stronger" signal?

A natural intuition is: execution is a stronger, cleaner signal, so make the synthetic strong and EEGNet will learn the left/right distinction more easily. It is a reasonable instinct, and the `execdom` ablation (below) tests it directly — but two things argue against making it the default.

- **The classifier is graded on the weak signal.** EEGNet is *tested* on real imagery, so it tunes its filters to whatever it trains on. Train it on execution-shaped epochs and it keys on features that are prominent in execution but faint or differently-shaped in imagery; at test time those features aren't really there, and the decision boundary is mis-calibrated. This train/test **distribution mismatch** is the most common way extra data backfires. Step 5 already showed a mild version: population-imagery synthetic was only *slightly* the wrong shape and still degraded the Strong tier (−17.6). Pushing the synthetic *further* from imagery (toward execution) would likely worsen that for exactly the subjects whose real imagery is already informative.
- **"Stronger" cannot mean "bigger" here — everything is z-scored.** The pipeline z-scores every epoch per-channel to mean 0 / std 1 (real imagery, real execution, and all synthetic alike), so absolute amplitude is normalized away; you literally cannot make the synthetic "louder." Execution's real advantage is therefore a **cleaner, more trial-consistent ERD pattern (higher SNR)**, not larger values. The question reduces to *"is cleaner / more-prototypical synthetic better?"* — a genuine trade-off: a crisp exemplar gives a clearer anchor (plausibly helping the weak/non-responder tiers, whose own imagery is barely usable), but training on only tidy prototypes can leave the model brittle to real imagery noise and drag well-served subjects toward the wrong shape (the Step 5 tier pattern).

The resolution is that domain-conditioning captures the *useful half* of the "make it stronger" intuition without its risk: sampling **domain = imagery** yields imagery-shaped epochs that are nonetheless **sharpened** because execution taught the shared backbone what a clean left/right ERD looks like. We do not have to choose by argument — the `execdom` arm samples the *same* model in execution style and runs it through the identical gate. If execution-style wins, the intuition was right and we learn the classifier prefers clean strong templates; if it loses, it confirms distribution-matching dominates and imagery-style is correct.

### Why the test data stays fixed

The whole value of RR-3 is that it compares *two ways of expanding the same limited real set* against the *same* held-out real target. Changing the test set (or the real training folds) would break comparability with `vanilla` (56.7%), `aug_2x` (59.5%), and `tstr_2x` (54.7%). Keeping folds and test identical means the new arm slots directly onto the existing figures and paired Wilcoxon.

### Leakage model (unchanged invariant, execution is always safe)

The classifier tests only on imagery runs {4,8,12}. Execution runs {3,7,11} are therefore **never a test run**, so including *all* execution in *every* generator can never leak test-run information. The imagery run-holdout is preserved exactly as in Step 5: the model serving folds that test on imagery run R is trained on imagery {4,8,12}\{R} **plus** all execution {3,7,11}. The one invariant that must never break remains:

```python
assert bank.held_out_run == test_run    # test_run ∈ {4,8,12}, imagery only
```

Because the generator is population (subject-agnostic), the samples carry no subject- or test-run-specific information — the same cleanliness standard Step 5 met.

---

## Requirements

### Must Have

- Pipeline can extract left/right-fist **execution** runs (3/7/11) alongside imagery (4/8/12) into a combined dataset `data/processed_with_exec.npz`, with a `domains` array (0=imagery, 1=execution) and `run_ids` preserved. The imagery-only `data/processed.npz` is unchanged and remains the classifier's data.
- Diffusion model gains an optional **domain-conditioning** token (`use_domain_cond`, `n_domains=2`), backward-compatible: existing Step 4/5 imagery-only checkpoints still load and sample identically.
- A diffusion trainer variant that trains three run-holdout models on imagery+execution jointly, domain-conditioned, saved as `models/checkpoints/diffusion_exec_holdout_run{4,8,12}.pt`, each excluding exactly one imagery run and including all execution.
- The trained models sample **imagery-domain** epochs by default; the sampler-health guard (`0.8 < std < 1.3`) and the memorization guard are evaluated against real **imagery** epochs.
- TSTR harness runs the additive arm from these checkpoints via a `--diffusion-variant exec` switch, writing `per_subject_results_tstr_exec_{1,2,3}x.json` in the existing schema, leakage-routed identically to Step 5.
- `compare_baselines.py` loads the new arm, prints it in the summary table, runs the **RR-3 gate for `tstr_exec_2x` vs `aug_2x`**, and reports the lift vs `tstr_2x` (imagery-only).
- Modal launchers for both stages (train the three exec generators; run the three TSTR passes with `--variant exec`).
- Leakage guard active and asserted end-to-end; execution runs verified never to be a test run.

### Should Have

- **Pooled/no-domain ablation:** a second set of generators trained on imagery+execution *without* the domain token (`use_domain_cond=False`), sampled as-is, producing `tstr_execpool_{1,2,3}x` — quantifies the value of domain conditioning.
- **Execution-domain sampling ablation:** sample the domain-conditioned models with domain=execution (`tstr_execdom_2x`) to measure the domain gap directly (does execution-*like* synthetic hurt an imagery classifier?).
- MLflow logging under a `pivot1-diffusion` (generators) and `pivot1-tstr` (classifier) experiment, mirroring Step 4/5 params plus `include_execution`, `use_domain_cond`, `sample_domain`.
- Fidelity figures (PSD overlap, nn_distance) for the exec generators vs real imagery, mirroring Step 4 §9, so we can confirm the imagery-domain samples are still spectrally faithful and non-memorizing.
- A comparison figure overlaying `aug`, `tstr` (imagery), and `tstr_exec` curves with the 59.5% gate line.

### Nice to Have

- Tier-resolved delta figure `tstr_exec_2x − tstr_2x` per subject, to see whether execution specifically rescues the Learning/Strong tiers the imagery generator harmed.
- A `--sample-domain` CLI knob on the bank builder so imagery/execution sampling is switchable without code edits.
- Sensitivity check that the verdict is stable across two bank seeds.

---

## Architecture Overview

### Data flow

```
                data/eegbci_pipeline.py  --include-execution
                          │
                          ▼
  data/processed.npz (imagery 4/8/12)      data/processed_with_exec.npz
   X,y,subject_ids,run_ids                  X,y,subject_ids,run_ids,domains
        │  (classifier — UNCHANGED)              │  (generator only)
        │                                        ▼
        │                       ┌────────────────────────────────────────┐
        │                       │ eval/train_diffusion.py (exec variant)  │
        │                       │  holdout imagery run R; train on         │
        │                       │  imagery{4,8,12}\R + ALL execution{3,7,11}│
        │                       │  domain-conditioned (imagery=0/exec=1)   │
        │                       └───────────────┬────────────────────────┘
        │                                       │ 3 checkpoints
        │                        diffusion_exec_holdout_run{4,8,12}.pt
        │                                       │
        ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ eval/train_baseline.py  --run tstr  --diffusion-variant exec           │
│   _build_synthetic_banks(variant=exec, sample_domain=imagery)          │
│     → bank[4],bank[8],bank[12]  (imagery-domain, z-scored, balanced)    │
│   for each LORO fold (test imagery run R):                             │
│     X_comb = X_real_imagery ⊕ N synthetic ;  assert bank.held_out==R   │
│     train EEGNet ; test on real held-out imagery run                   │
│   → per_subject_results_tstr_exec_{1,2,3}x.json                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ eval/compare_baselines.py                                             │
│   load vanilla, aug_*, tstr_* (imagery), tstr_exec_*                   │
│   RR-3 gate: tstr_exec_2x vs aug_2x ;  lift vs tstr_2x                 │
│   figures + verdict                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### What's new vs. reused

| Component | Status |
|---|---|
| EEGNet, `_train_fold`, `_eval_on_test`, LORO split, per-subject aggregation | **reused, unchanged** |
| `SyntheticBank`, `_load_synthetic_set`, `tstr` run mode, leakage routing | **reused, unchanged** |
| `vanilla`, `aug_1x/2x/3x` per-subject JSONs, gate statistics | **reused as frozen comparators** |
| `data/eegbci_pipeline.py` | **modified: extract execution runs + domains** |
| `eval/diffusion.py` (`DiffusionConfig`, `UNet1D`, `DDPM`, `EEGDiffusionModel`) | **modified: optional domain conditioning (backward-compatible)** |
| `eval/train_diffusion.py` | **modified: combined dataset, domain, exec checkpoint naming, fidelity vs imagery** |
| `_build_synthetic_banks`, `train_baseline.py` CLI | **modified: `--diffusion-variant`, `--sample-domain`** |
| `compare_baselines.py` | **modified: load/plot/gate the exec arm** |
| `modal_train_diffusion_exec.py`, `modal_train_tstr.py --variant` | **new / modified launchers** |
| `reports/pivot_iter01_exec_conditioned_tstr_report.md` | **new** |

---

## Data Models

### Combined dataset (`data/processed_with_exec.npz`)

```python
X:           np.ndarray  # (n_epochs, 64, 801) float32, per-epoch z-scored (same space as processed.npz)
y:           np.ndarray  # (n_epochs,) int64 ∈ {0,1}  (T1=left→0, T2=right→1)
subject_ids: np.ndarray  # (n_epochs,) int
run_ids:     np.ndarray  # (n_epochs,) int ∈ {3,4,7,8,11,12}
domains:     np.ndarray  # (n_epochs,) int ∈ {0,1}  (0=imagery run∈{4,8,12}, 1=execution run∈{3,7,11})
ch_names:    np.ndarray  # (64,) str
sfreq:       float       # 160.0
```

`domains` is fully derivable from `run_ids` (`1 if run in {3,7,11} else 0`) and is stored for convenience + an integrity assert. The classifier never loads this file; it continues to use `data/processed.npz`.

### `DiffusionConfig` additions (`eval/diffusion.py`)

```python
@dataclass
class DiffusionConfig:
    # ... all existing fields unchanged ...
    use_domain_cond: bool = False   # False → identical to Step 4/5 (backward compatible)
    n_domains: int = 2              # imagery, execution
    sample_domain: int = 0          # default generation domain: 0=imagery, 1=execution
```

`DOMAIN_IMAGERY = 0`, `DOMAIN_EXECUTION = 1` module constants. When `use_domain_cond=False`, no domain embedding is created and `forward()`/`sample()` ignore any domain argument — so the three Step 4/5 checkpoints load and behave exactly as before.

### Checkpoints & result JSONs

```
models/checkpoints/
  diffusion_exec_holdout_run{4,8,12}.pt          ← new (domain-conditioned, imagery+execution)
  per_subject_results_tstr_exec_{1,2,3}x.json    ← new (schema identical to Step 5)
  # optional ablations:
  diffusion_execpool_holdout_run{4,8,12}.pt      ← no domain token
  per_subject_results_tstr_execpool_{1,2,3}x.json
  per_subject_results_tstr_execdom_2x.json       ← domain=execution sampling
```

### MLflow params (per exec generator)

```python
{ "experiment": "pivot1-diffusion", "holdout_run": R,
  "train_runs_imagery": [imagery \ R], "train_runs_execution": [3,7,11],
  "include_execution": True, "use_domain_cond": True, "n_domains": 2,
  "n_steps": 100_000, "guidance_scale": 3.0, "ddim_steps": 50, ... }   # + Step 4 params
```

---

## Interface Definitions

### Pipeline — `data/eegbci_pipeline.py`

```
python data/eegbci_pipeline.py --subjects $(seq 1 109) \
    --runs 3,4,7,8,11,12 --out data/processed_with_exec.npz
```

- Add a `--runs` option (comma-separated) replacing/generalising the hardcoded imagery run list; default keeps `4,8,12` so existing behaviour and `data/processed.npz` regeneration are unchanged.
- Emit the `domains` array (derived from `run_ids`).
- Reuse the exact epoching (T1/T2, tmin=−1, tmax=4s), band-pass (7–30 Hz), and per-epoch z-score. Execution runs use the same annotation→label mapping as imagery.
- Report per-run epoch counts and flag any subject whose execution runs epoch to a non-standard length (same guard that dropped 88/92/100 for imagery); such subjects are simply omitted from the execution set — the population generator does not require per-subject completeness, and the classifier cohort (imagery, 106 subjects) is unaffected.

### `eval/diffusion.py` — domain conditioning

```python
# UNet1D.__init__ (when config.use_domain_cond):
self.domain_emb = nn.Embedding(config.n_domains, cond_dim)

# UNet1D.forward(x, t, class_label, domain=None):
cond = self.time_emb(t) + self.class_emb(class_label)
if self.config.use_domain_cond:
    cond = cond + self.domain_emb(domain)     # domain: (B,) long

# DDPM.train_step(x0, y, domain):  pass domain through both model() calls;
#   CFG still drops only the CLASS label to null_token; domain is always provided
#   (a fixed context, not guided).

# DDPM.sample(n, class_label, domain, ddim_steps, guidance_scale, seed):
#   build domain_batch = full((n,), domain); pass to model in uncond AND cond passes.

# EEGDiffusionModel.generate(n, subject_id, class_label=None, domain=None, seed=None):
#   domain = self.config.sample_domain if domain is None else domain
#   (ignored when use_domain_cond=False)
```

**Rationale for not guiding the domain:** classifier-free guidance sharpens the conditioning we are uncertain about at sample time (the class). The domain is *chosen* (we always want imagery), so it is a fixed context in both passes; guiding it would push samples toward an exaggerated imagery/execution caricature. Keeping guidance on class only preserves the validated Step 4 sampler behaviour.

### `eval/train_diffusion.py` — exec variant

```python
train(
    data_path = Path("data/processed_with_exec.npz"),
    holdout_run = R,                 # imagery run to exclude (4/8/12)
    include_execution = True,        # keep run_ids {3,7,11}
    use_domain_cond = True,
    checkpoint_name = f"diffusion_exec_holdout_run{R}.pt",
    fidelity_domain = DOMAIN_IMAGERY,  # PSD/nn metrics vs real imagery
    ...                              # all Step 4 params unchanged
)
```

- Training-set selection is simply `mask = run_ids != holdout_run` on the combined dataset → keeps imagery{4,8,12}\R + all execution{3,7,11}. Domain per epoch = `domains[mask]`.
- Fidelity/memorization (`psd_divergence`, `nn_distance`, `nn_distance_real`) computed against real **imagery** epochs of the held-out run (the target distribution), sampling `domain=imagery`.
- Checkpoint embeds `use_domain_cond`, `n_domains`, `sample_domain` in its config so `load()` reconstructs the domain head. **Note:** unlike `clip_denoised` (which was not serialized in Step 4), these new fields ARE in `asdict(config)` and therefore round-trip; the Step 5 sampler-health assert remains the runtime safety net.

### `eval/train_baseline.py` — variant switch

```
python eval/train_baseline.py --run tstr --diffusion-variant exec --n-copies 2
```

```python
--diffusion-variant {imagery,exec,execpool}   # default: imagery (Step 5 behaviour)
--sample-domain {imagery,execution}           # default: imagery (only meaningful for exec)
```

- `_build_synthetic_banks(checkpoint_dir, device, bank_size, seed, variant, sample_domain)`:
  - `variant` selects the checkpoint filename pattern (`diffusion_holdout_run{R}.pt` vs `diffusion_exec_holdout_run{R}.pt` vs `diffusion_execpool_holdout_run{R}.pt`).
  - generation call becomes `model.generate(bank_size, subject_id=0, domain=<sample_domain>, seed=seed+R)`.
  - Same `held_out_run == R` assert, same sampler-health guard.
- JSON/CM stem becomes `per_subject_results_tstr_exec_{n}x` (or `tstr_execpool`, `tstr_execdom`) — thread `variant`/`sample_domain` into the stem so arms never collide.
- Everything else in `tstr` mode (routing, `_load_synthetic_set`, fold seeds) is untouched.

### `eval/compare_baselines.py` — additions

```python
# load_all_results(): add candidates
("tstr_exec_1x", ".../per_subject_results_tstr_exec_1x.json", 30),
("tstr_exec_2x", ".../per_subject_results_tstr_exec_2x.json", 60),
("tstr_exec_3x", ".../per_subject_results_tstr_exec_3x.json", 90),

# reuse _wilcoxon_paired + print_gate_verdict, parametrised on the TSTR arm:
print_gate_verdict(results, tstr_key="tstr_exec_2x")   # RR-3 for the exec arm
# plus a lift line: _wilcoxon_paired(tstr_exec_2x, tstr_2x)  → did execution help?
```

- Add the exec arm to `print_summary_table`, the fig10 curve set (a third line + gate annotation), a `plot_tstr_exec_vs_imagery_delta` figure, and the `subject_deltas.csv` columns.
- `print_gate_verdict` generalised to accept which TSTR arm is the candidate (default `tstr_2x`), so both Step 5 and this pivot use one code path.

### Modal launchers

```
# Stage 1 — train the three exec generators (mirrors modal_train_diffusion.py):
modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz
modal run --detach modal_train_diffusion_exec.py            # runs 4,8,12 in parallel

# Stage 2 — TSTR passes from the exec checkpoints (modal_train_tstr.py + --variant):
modal run --detach modal_train_tstr.py --variant exec       # n_copies 1,2,3 in parallel
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_1x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_2x.json models/checkpoints/
modal volume get neurosynth-checkpoints per_subject_results_tstr_exec_3x.json models/checkpoints/
python eval/compare_baselines.py
```

`modal_train_tstr.py` gains a `--variant {imagery,exec,execpool}` and `--sample-domain` passthrough to `train(..., diffusion_variant=..., sample_domain=...)`.

---

## Implementation Plan

### Phase 1 — Data: execution runs

- [ ] Generalise `data/eegbci_pipeline.py` run selection to a `--runs` list (default `4,8,12`); add the `domains` output array and an integrity assert (`domains == (run_ids ∈ {3,7,11})`).
- [ ] Verify runs 3/7/11 exist and epoch cleanly for the cohort; log per-run counts; omit malformed execution subjects.
- [ ] Produce `data/processed_with_exec.npz`; confirm `data/processed.npz` (imagery) is byte-identical to before (regenerate with default `--runs` and diff, or leave untouched).

### Phase 2 — Diffusion: domain conditioning (backward-compatible)

- [ ] `DiffusionConfig`: add `use_domain_cond`, `n_domains`, `sample_domain`; module constants `DOMAIN_IMAGERY/EXECUTION`.
- [ ] `UNet1D`: conditional `domain_emb`; thread `domain` through `forward`.
- [ ] `DDPM.train_step` / `sample`: accept and pass `domain`; keep CFG on class only.
- [ ] `EEGDiffusionModel.generate`: `domain` param defaulting to `config.sample_domain`; ignored when `use_domain_cond=False`.
- [ ] Regression check: load a Step 4 imagery-only checkpoint, generate — output identical to pre-change (numerically, given the same seed).

### Phase 3 — Diffusion trainer + launcher (exec variant)

- [ ] `eval/train_diffusion.py`: accept `include_execution`, `use_domain_cond`, `checkpoint_name`, `fidelity_domain`; derive/pass domain; train-set mask `run_ids != holdout_run`; fidelity vs real imagery.
- [ ] `modal_train_diffusion_exec.py`: thin copy of `modal_train_diffusion.py` that mounts `processed_with_exec.npz` and trains the three exec models (100K steps, A10G, `--detach`, `--resume`).

### Phase 4 — TSTR harness variant

- [ ] `_build_synthetic_banks`: add `variant` + `sample_domain`; select checkpoint pattern; pass `domain` to `generate`.
- [ ] `train_baseline.py`: `--diffusion-variant`, `--sample-domain`; thread into bank build and JSON/CM stems; resolve MLflow experiment `pivot1-tstr` for exec variants.
- [ ] `modal_train_tstr.py`: `--variant` / `--sample-domain` passthrough.

### Phase 5 — Comparison & gate

- [ ] `compare_baselines.py`: load `tstr_exec_*`; generalise `print_gate_verdict(tstr_key=...)`; add exec RR-3 verdict + lift-vs-`tstr_2x`; overlay curve on fig10; `plot_tstr_exec_vs_imagery_delta`; CSV columns.

### Phase 6 — Run, collect, report

- [ ] Stage 1 on Modal (three exec generators); validate fidelity (PSD overlap vs imagery, `nn_gen ≳ nn_real`, sampler std≈1.0).
- [ ] Stage 2 on Modal (three TSTR passes, `--variant exec`); pull JSONs.
- [ ] `python eval/compare_baselines.py`; record verdict + tier lift.
- [ ] `reports/pivot_iter01_exec_conditioned_tstr_report.md`: methodology, per-N table, RR-3 verdict, lift over Step 5, tier analysis, decision.
- [ ] (If run) ablations: `execpool` (no domain) and `execdom` (domain=execution) arms + their gate lines.
- [ ] Update project memory + fix the misleading `phase0_baseline_report.md` §12.4 "motor disability" line.

---

## Error Handling

- **Missing `processed_with_exec.npz`** → `train_diffusion` exits with the pipeline command to build it.
- **Execution run absent/malformed for a subject** → omit that subject's execution epochs, log it; do not abort (population generator).
- **Old checkpoint without domain fields** → `use_domain_cond` defaults `False`; loads and samples as Step 4/5 (regression-tested in Phase 2).
- **Sampler-health guard** (`0.8 < std < 1.3` on imagery-domain samples) → hard exit before any TSTR compute, exactly as Step 5. This is the primary safety net.
- **Routing mismatch** (`bank.held_out_run != test_run`) → `AssertionError`, hard stop (leakage).
- **Domain out of range** (`domain ∉ {0,1}` with `use_domain_cond`) → `ValueError` at generate.
- **Modal** → reuse Step 4/5 hardening (`NEUROSYNTH_GPU` env, no `volume.reload()` with cwd in-volume, 6 h timeout, per-container commit).

---

## Security Considerations

Research code, own data, cloud-compute only. Integrity, not security:

- **Leakage integrity** — enforced by `bank.held_out_run == test_run` and the structural fact that execution runs are never test runs. This is the single most important invariant; the exec arm must not weaken it.
- **Comparison honesty** — the exec arm's generator has information the augmentation arm lacks (execution). This must be reported as *"customer imagery data + a generator enriched with execution"* vs *"customer imagery data augmented"* — never as an equal-information comparison. It reflects the real product (proprietary prior as moat), but the asymmetry must be stated plainly in the report.
- **Checkpoint trust** — `torch.load(weights_only=False)` on our own checkpoints only (as Step 4/5).

---

## Testing Strategy

Manual verification (research code) plus cheap asserts.

- **Pipeline:** `domains` matches `run_ids`; imagery cohort unchanged (106 subjects, runs {4,8,12} identical to `processed.npz`); execution epoch counts reported.
- **Domain backward-compat (unit):** a Step 4 checkpoint (`use_domain_cond=False`) generates numerically identical output pre/post the diffusion changes at fixed seed.
- **Domain conditioning (unit):** with `use_domain_cond=True`, `generate(domain=imagery)` and `generate(domain=execution)` differ; both pass the std≈1.0 guard against their respective real domains.
- **Leakage/routing:** every LORO fold routes to `diffusion_exec_holdout_run{test_run}`; execution never appears as a `test_run`.
- **`_load_synthetic_set` unchanged:** shape/balance/determinism asserts from Step 5 still pass with the exec banks.
- **Fidelity (Stage 1 gate):** imagery-domain samples overlap real imagery PSD in 8–30 Hz; `nn_gen ≳ nn_real` (non-memorizing); three models mutually comparable.
- **Gate logic:** `print_gate_verdict(tstr_key="tstr_exec_2x")` reproduces PASS/FAIL correctly on synthetic `RunResult`s (reuse Step 5 gate unit check).

---

## Success Criteria

**Implementation complete when:**
- `processed_with_exec.npz` builds with correct `domains`; imagery classifier data unchanged.
- Three `diffusion_exec_holdout_run{4,8,12}.pt` train, pass fidelity + sampler-health, and load with domain head intact; Step 4/5 checkpoints still load unchanged.
- `--run tstr --diffusion-variant exec` produces `per_subject_results_tstr_exec_{1,2,3}x.json` (106 subjects × 3 folds), leakage-routed, imagery-domain.
- `compare_baselines.py` prints the exec arm, its RR-3 verdict, and the lift vs `tstr_2x`, and saves the overlay + delta figures.

**Experiment complete when:**
- The exec RR-3 verdict is computed: **PASS** iff `mean(tstr_exec_2x) − mean(aug_2x) > +1.0%` AND one-sided paired Wilcoxon (`tstr_exec_2x > aug_2x`, 106 subjects) `p < 0.05`; else **FAIL**.
- The lift over Step 5 (`tstr_exec_2x − tstr_2x`, paired) is reported with its tier breakdown (did execution rescue Learning/Strong?).
- `reports/pivot_iter01_exec_conditioned_tstr_report.md` documents the outcome and decision.

**Decision output:**
- **PASS** → execution-as-generative-prior validated; population synthetic beats augmentation. This becomes the first positive result and motivates the funding narrative (more proprietary data → stronger generator). Subject-conditional generation becomes an *improvement*.
- **Partial** (beats `tstr_2x` and/or vanilla but not `aug_2x`) → execution helps but a subject-agnostic prior still isn't enough; proceed to subject-conditional generation with execution included (the two levers combined).
- **FAIL / no lift** → execution as a population prior does not transfer to imagery classification; the domain gap dominates. Report honestly; the `execdom` ablation should show execution-domain samples actively hurting, confirming the mechanism.

---

## Compute & Time Estimates

| Stage | Work | Hardware | Time |
|---|---|---|---|
| Stage 1 (generators) | 3 models × 100K steps, ~2.5× the Step 4 data (imagery 2 runs + all execution) | A10G ×3 parallel | ~2.5–3 h/model, ~3 h wall-clock, ~$8–9 |
| Stage 2 (TSTR) | 3 passes × (bank gen + 318 folds × 500 epochs) | A10G ×3 parallel | ~1 h wall-clock, ~$1–2 |

Bank generation and the classifier cost are unchanged from Step 5. The added cost is Stage-1 retraining of the three generators (comparable to Step 4). Total ≈ $10–11, ~4 h wall-clock across two `--detach` launches.

---

## Context for Handoff

### Frozen comparators (do not recompute)

```
models/checkpoints/
  per_subject_results_vanilla.json        56.7%   (real imagery only)
  per_subject_results_augmented_2x.json   59.5%   (RR-3 ceiling; the gate target)
  per_subject_results_tstr_2x.json        54.7%   (Step 5 imagery-only synthetic; the lift baseline)
  diffusion_holdout_run{4,8,12}.pt        Step 4 imagery-only generators (kept; --diffusion-variant imagery)
```

### The invariants that must never break

```python
assert bank.held_out_run == test_run          # test_run ∈ {4,8,12}; execution never a test run
assert 0.8 < bank.X.std() < 1.3               # imagery-domain sampler health
# classifier real train/test data = imagery only, identical folds/seeds to Step 5
```

### Constants

```python
IMAGERY_RUNS=(4,8,12); EXECUTION_RUNS=(3,7,11); RUNS=(4,8,12)   # classifier test runs
DOMAIN_IMAGERY=0; DOMAIN_EXECUTION=1
N_SUBJECTS=106; N_FOLDS=318; SEED=42; BANK_SIZE=2048
VANILLA=56.7; AUG_2X=59.5; TSTR_2X=54.7; GATE_MARGIN=1.0; GATE_ALPHA=0.05
```

---

## Open Questions

- [x] **Primary design:** RESOLVED (2026-07-10, owner) — domain-conditioned joint training with imagery-domain sampling is the primary arm; pooled/no-domain is the `execpool` ablation. Reasoning captured in Background → "Pooled vs. domain-conditioned" and "Why not simply generate a stronger signal?".
- [ ] Run the two ablations (`execpool`, `execdom`) in the first pass, or only if the primary arm is ambiguous? (Cheap to include; recommended if budget allows. `execdom` directly tests the "make the synthetic stronger" hypothesis.)
- [ ] Do execution runs 3/7/11 epoch cleanly for the same 106 subjects, or is the execution cohort smaller? (Resolved empirically in Phase 1; does not affect the population generator's validity.)
