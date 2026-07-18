# Pivot Iteration 2 — Subject-Conditional Generation & TSTR Re-Gate

**Status:** Draft
**Created:** 2026-07-18
**Depends on:** Step 4 (population DDPM), Step 5 (TSTR harness), Pivot 1 (domain conditioning + execution data, `processed_with_exec.npz`) — all complete.
**Feeds into:** The decision on whether *subject-specific* synthetic EEG beats signal augmentation (RR-3) — the last remaining lever at the population/per-subject rung before conceding that synthetic augmentation of tiny motor-imagery calibration sets does not work.

---

## Executive Summary

Two experiments have now shown that **population (subject-agnostic) synthetic EEG does not beat signal augmentation** — imagery-only (Step 5: 54.7%) and execution-enriched (Pivot 1: 54.7%, +0.01% lift) both fail identically, proving the bottleneck is population *averaging* rather than data quality or domain. This pivot addresses that directly: condition the diffusion model on a **subject-identity embedding** so it generates *subject-specific* synthetic epochs that preserve each subject's individual ERD topography, instead of a blurred population mean. The subject embedding is trained as a **residual on a population fallback** (classifier-free-style subject dropout), and real motor-execution data is retained as the per-subject prior. The classifier's real training folds and imagery test set are unchanged, so the result compares apples-to-apples against `vanilla` (56.7%), `aug_2x` (59.5%), and the two failed synthetic arms. This reuses the run-holdout cross-fitting and the TSTR harness essentially intact; the new work is the subject-embedding head, per-subject bank generation, and two-key `(subject, test_run)` routing.

---

## Goals

- Condition the diffusion generator on subject identity (plus the existing class + domain tokens) so it produces subject-specific imagery-domain synthetic epochs.
- Train the subject embedding as a residual over a population baseline (subject dropout), so it learns per-subject structure from limited data instead of memorizing or collapsing to the population.
- Generate **per-subject** synthetic banks and route each LORO fold to the bank for *its* subject and *its* held-out run, leakage-free.
- Re-run the additive TSTR arm (`tstr_subj_{1,2,3}x`) at N = {30, 60, 90} on all 106 subjects and re-evaluate RR-3 (`tstr_subj_2x` vs `aug_2x`), plus the lift over the population arms (`tstr_2x`, `tstr_exec_2x`).
- Keep folds, seeds, EEGNet, and the imagery test set identical to Step 5 / Pivot 1.

## Non-Goals

- **Changing the classifier's real data.** EEGNet still trains on real imagery folds and tests on the held-out real imagery run. Execution and subject conditioning affect only the *generator*.
- **Generating for unseen subjects.** The product scenario is "customer provides subject S's calibration; we generate more S-specific data," so S is always in the generator's training set. Zero-shot generation for held-out *subjects* is explicitly out of scope (and would require a different, subject-holdout eval).
- **Retraining EEGNet, changing the LORO protocol, the augmentation baseline, or the gate statistics.**
- **Abandoning the domain token.** Domain conditioning (imagery/execution) from Pivot 1 is retained; we still sample the imagery domain.
- **A new preprocessing pipeline.** Reuses `data/processed_with_exec.npz` (imagery + execution, with `subject_ids`).

---

## Background & Context

### The two negatives that motivate this

| Arm (N=60) | Mean | vs vanilla | vs aug ceiling | vs imagery synth |
|---|---|---|---|---|
| vanilla | 56.7% | — | −2.8% | — |
| aug_2x (gate ceiling) | 59.5% | +2.8% | — | — |
| tstr_2x — population imagery (Step 5) | 54.7% | −2.0% | −4.8% | — |
| tstr_exec_2x — population + execution (Pivot 1) | 54.7% | −2.0% | −4.8% | +0.01% |

Both synthetic arms are statistically identical and both lose to augmentation. Pivot 1 was decisive: enriching the *population* prior with clean, high-SNR execution data — validated to equal fidelity — changed nothing (+0.01%, p=0.62). The diagnosis (`phase0_baseline_report.md` §10.5, §12.4) is that a BCI classifier is a **per-subject spatial filter**, and a subject-agnostic generator produces a population-averaged signal that dilutes each subject's individual ERD topography — harming precisely the Learning/Strong tiers whose filter works. No population-level data change can fix an averaging defect.

### Why subject conditioning is the remaining lever

If the defect is averaging across subjects, the fix is to *stop averaging*: condition generation on subject identity so the model reproduces subject S's own topography. This is the only untried lever at this rung. The honest caveat (see Compute & Honest Prognosis) is that each subject has very little data to learn from, so the design must extract a per-subject signal from ~75 epochs — which is why execution is retained (it nearly doubles per-subject data) and why the subject embedding is a *residual*, not a standalone generator.

### Why run-holdout + subject conditioning is leakage-safe (and subject-holdout is unnecessary)

`phase0_baseline_report.md` §9.2 anticipated that subject conditioning would need "the stricter subject-holdout scheme." On analysis, **it does not** — the existing run-holdout scheme is sufficient and cleaner. For a fold testing subject S on imagery run R:

- Route to the model with `held_out_run == R`. That model trained on **all** subjects' runs `{4,8,12}\{R}` + all execution — so it **never saw run R of any subject, including S**. The imagery test run is therefore fully unseen by the generator. ✓
- The model learned S's embedding only from **S's non-R runs** (`{4,8,12}\{R}` imagery + S's execution) — the *same data the classifier trains on for that fold*. So even a memorizing generator could only reproduce S's *training* epochs, never the test run. No new test-run information is introduced. ✓

Subject-holdout (excluding subject S entirely) would only be needed to test generation for *unseen* subjects, which is a Non-Goal. Run-holdout gives us the exact leakage guarantee we need while reusing the three-model infrastructure unchanged.

> **Design decision (leakage):** keep the three run-holdout models; add subject conditioning. Routing becomes two-key `(subject, test_run)`; the invariant strengthens to `bank.held_out_run == test_run AND bank.subject_id == subject`.

### Why a residual subject embedding (subject dropout)

Each subject contributes only ~30 imagery + ~45 execution ≈ **75 epochs** to learn its embedding — far too few to learn a full subject-specific generator from scratch. So we train the subject token with **classifier-free-style dropout to a null-subject token**: with probability `p_subj_uncond`, the subject is replaced by "null" (population) during training. The U-Net backbone thus learns a **population baseline** shared across all subjects (from all ~9.5K epochs), and each subject embedding only has to learn the **per-subject residual** (the individual spatial-filter offset). At sample time we condition on S (no dropout), getting population structure adjusted toward S. This is the single most important design choice for making the approach viable on scarce per-subject data.

> **Design decision (conditioning):** subject embedding with dropout-to-null (residual over population). Guidance (CFG) stays on the class only; subject and domain are fixed contexts at sample time.

---

## Requirements

### Must Have

- Diffusion model gains an optional **subject-conditioning** token (`use_subject_cond`, `n_subjects`, null-subject index = `n_subjects`) alongside the existing class + domain tokens. Backward-compatible: Step 4/5 and Pivot 1 checkpoints load and sample unchanged.
- Training-time **subject dropout** (`p_subj_uncond`, default 0.15) so the embedding is a residual over a population fallback.
- A **subject-id → contiguous-index mapping** built from the dataset and stored in the checkpoint, so `generate(subject_id=<original id>)` maps to the right embedding.
- Three subject-conditional run-holdout generators, trained on imagery `{4,8,12}\R` + all execution, subject-conditioned, saved as `diffusion_subj_holdout_run{4,8,12}.pt`.
- **Per-subject bank generation** on the training GPU: after training, each model generates an imagery-domain bank for every subject and saves `subject_banks_holdout_run{R}.npz` (X, y, subject_ids). (Generation is cheap once the model is loaded; caching avoids regenerating banks in each of the three TSTR passes.)
- TSTR harness loads the cached per-subject banks (`--diffusion-variant subj`), routes each fold `(S, R)` to `banks[R][S]`, and asserts the two-key invariant. Writes `per_subject_results_tstr_subj_{1,2,3}x.json` in the existing schema.
- `compare_baselines.py` loads the new arm, runs the RR-3 gate (`tstr_subj_2x` vs `aug_2x`) and the lift vs `tstr_2x` and `tstr_exec_2x`, plots the curve, and prints the verdict.
- Modal launchers for training+bank-gen and for the TSTR passes.
- Leakage guard active end-to-end; the two-key routing asserted.

### Should Have

- **Fidelity validation per subject-model**: sampler-health (`std≈1.0`), PSD overlap vs that subject's real imagery, and a per-subject memorization guard (generated-for-S vs S's real training epochs must stay ≥ S's real-to-real NN) — the memorization check matters more here than in the population case, since the model now conditions on the subject it could copy.
- **Subject-embedding sanity probe**: confirm the embedding is used, e.g. samples conditioned on two different subjects differ, and per-subject-conditioned PSDs track each subject's real PSD better than the population model's do.
- MLflow logging under `pivot2-diffusion` / `pivot2-tstr`, with `use_subject_cond`, `p_subj_uncond`, `n_subjects`.
- Tier-resolved delta figure `tstr_subj_2x − tstr_exec_2x` (did per-subject generation rescue the Learning/Strong tiers?).
- Secondary reporting on the **data-limited / mud-zone tier** specifically (the subjects most likely to benefit).

### Nice to Have

- A `p_subj_uncond` sweep (e.g. {0.1, 0.15, 0.3}) to find the residual/population balance.
- Optional subject-CFG guidance knob (`subject_guidance_scale`) to sharpen toward the subject — off by default.
- A per-subject "amount of real data" vs "synthetic lift" scatter, to see whether subjects with more calibration benefit more.

---

## Architecture Overview

### Conditioning stack

```
cond = time_emb(t) + class_emb(c) + domain_emb(d) + subject_emb(s)
                       │ CFG-guided      │ fixed          │ fixed context,
                       │ (class only)    │ (imagery=0)    │ dropout→null in training
```

### Data / leakage flow

```
data/processed_with_exec.npz  (imagery 4/8/12 + execution 3/7/11, subject_ids, domains)
        │
        ▼  (holdout imagery run R; keep {4,8,12}\R + all execution; subject-conditioned)
┌───────────────────────────────────────────────────────────────┐
│ eval/train_diffusion.py  --use-subject-cond  --ckpt-prefix diffusion_subj │
│   train 3 models (R=4,8,12); subject dropout p=0.15                       │
│   after training: for each subject S → generate imagery-domain bank       │
│     save subject_banks_holdout_run{R}.npz  (X, y, subject_ids)            │
└───────────────┬───────────────────────────────────────────────┘
                │ diffusion_subj_holdout_run{4,8,12}.pt  +  subject_banks_holdout_run{4,8,12}.npz
                ▼
┌───────────────────────────────────────────────────────────────┐
│ eval/train_baseline.py  --run tstr  --diffusion-variant subj    │
│   load cached per-subject banks → banks[R][S]                    │
│   for each LORO fold (subject S, test imagery run R):           │
│     assert banks[R][S].held_out_run==R and .subject_id==S        │
│     X_comb = X_real ⊕ N synthetic(from banks[R][S])             │
│     train EEGNet; test on real held-out imagery run             │
│   → per_subject_results_tstr_subj_{1,2,3}x.json                 │
└───────────────┬───────────────────────────────────────────────┘
                ▼
        eval/compare_baselines.py → RR-3 gate + lift vs population arms
```

### What's new vs. reused

| Component | Status |
|---|---|
| EEGNet, `_train_fold`, `_eval_on_test`, LORO, aggregation, gate stats | **reused, unchanged** |
| Run-holdout 3-model cross-fitting, domain conditioning, execution data, `processed_with_exec.npz` | **reused from Pivot 1** |
| `_load_synthetic_set`, `tstr` run mode, `compare_baselines` scaffolding | **reused; extended for 2-key banks + new arm** |
| Subject embedding + dropout in `eval/diffusion.py` | **new (backward-compatible)** |
| Per-subject bank generation + `subject_banks_holdout_run{R}.npz` | **new** |
| Two-key `(subject, run)` bank routing in `train_baseline.py` | **new** |
| `modal_train_diffusion_subj.py` | **new** |
| `reports/pivot_iter02_...md` | **new** |

---

## Data Models

### `DiffusionConfig` additions (`eval/diffusion.py`)

```python
@dataclass
class DiffusionConfig:
    # ... existing (incl. use_domain_cond, n_domains, sample_domain from Pivot 1) ...
    use_subject_cond: bool = False       # False → identical to Step 4/5/Pivot 1
    n_subjects: int = 106                 # embedding table size (null index = n_subjects)
    p_subj_uncond: float = 0.15           # training dropout of the subject token → residual
    subject_ids: tuple[int, ...] = ()     # original ids in embedding-index order (metadata)
    subject_guidance_scale: float = 0.0   # optional; 0 = subject is unguided fixed context

    @property
    def null_subject(self) -> int:
        return self.n_subjects
```

`subject_ids` maps embedding index → original dataset subject id; the inverse is built at load. Stored in the checkpoint config (round-trips via `asdict`).

### `SyntheticBank` extension (`eval/train_baseline.py`)

```python
@dataclass
class SyntheticBank:
    held_out_run: int
    X: np.ndarray                # (bank_size, 64, 801) f32, z-scored, subject-conditioned
    y: np.ndarray                # (bank_size,) int64 balanced
    subject_id: int | None = None  # NEW: the subject this bank was conditioned on (None = population)

# Population arms: banks[run] (unchanged). Subject arm: banks[run][subject].
banks_subj: dict[int, dict[int, SyntheticBank]]   # banks_subj[R][S]
```

### Cached per-subject bank file (`subject_banks_holdout_run{R}.npz`)

```python
X:           (n_subjects * bank_size_per_subject, 64, 801) float32
y:           (…,) int64
subject_ids: (…,) int64     # original id per row; grouped into per-subject banks on load
held_out_run: int           # scalar, for the routing assert
bank_size_per_subject: int  # scalar
```

`bank_size_per_subject` default **128** (64/class) — covers the max per-fold draw (3× ≈ 45/class) without replacement.

### Result JSONs / checkpoints

```
models/checkpoints/
  diffusion_subj_holdout_run{4,8,12}.pt              ← new generators
  subject_banks_holdout_run{4,8,12}.npz              ← new cached per-subject banks
  per_subject_results_tstr_subj_{1,2,3}x.json        ← new (schema identical to Step 5)
```

---

## Interface Definitions

### `eval/diffusion.py` — subject conditioning

```python
# UNet1D.__init__ (when config.use_subject_cond):
self.subject_emb = nn.Embedding(config.n_subjects + 1, cond_dim)   # +1 null subject

# UNet1D.forward(x, t, class_label, domain=None, subject=None):
cond = self.time_emb(t) + self.class_emb(class_label)
if self.domain_emb is not None:  cond = cond + self.domain_emb(domain)
if self.subject_emb is not None:
    if subject is None: raise ValueError("use_subject_cond=True but subject not provided")
    cond = cond + self.subject_emb(subject)

# DDPM.train_step(x0, y, domain=None, subject=None):
#   after class CFG dropout, also drop subject → null_subject with prob p_subj_uncond;
#   domain is NOT dropped. Pass both to model.

# DDPM.sample(..., domain=None, subject=None):
#   build subject batch = full((n,), subject_index) when use_subject_cond; pass to model
#   in both uncond and cond passes (subject is a fixed context; guidance stays on class).

# EEGDiffusionModel.generate(n, subject_id, class_label=None, domain=None, seed=None):
#   NOTE: subject_id is now USED when use_subject_cond — mapped via config.subject_ids to an
#   embedding index. For population/Pivot-1 models (use_subject_cond=False) it is ignored,
#   exactly as today. Backward-compatible signature.
```

**Subject-id mapping:** `EEGDiffusionModel` builds `self._subj_to_idx = {orig: i for i, orig in enumerate(config.subject_ids)}` at construction/load. `generate(subject_id=S)` uses `self._subj_to_idx[S]`; a missing S raises `KeyError` (the model was not trained on that subject).

### `eval/train_diffusion.py` — subject-conditional variant + bank export

```python
train(
    data_path=Path("data/processed_with_exec.npz"),
    holdout_run=R,
    use_domain_cond=True, use_subject_cond=True,
    p_subj_uncond=0.15,
    ckpt_prefix="diffusion_subj",
    export_subject_banks=True, bank_size_per_subject=128,  # save subject_banks_holdout_run{R}.npz
    fidelity_domain=DOMAIN_IMAGERY,
    ... # all Step 4 params unchanged
)
```

- Build `subject_ids` (sorted unique of the training cohort) → set `config.n_subjects` and `config.subject_ids`. Per epoch, map its `subject_id` → index for `train_step`.
- Training-set mask unchanged (`run_ids != holdout_run`); domain + subject arrays are masked alongside.
- After training + fidelity, if `export_subject_banks`: for each subject S, `generate(bank_size_per_subject, subject_id=S, domain=imagery, seed=seed+R+S)`; concatenate and save `subject_banks_holdout_run{R}.npz`. Per-subject memorization/PSD spot-checks logged.

### `eval/train_baseline.py` — subject variant

```
python eval/train_baseline.py --run tstr --diffusion-variant subj --n-copies 2
```

- `_VARIANT_CKPT_PREFIX["subj"] = "diffusion_subj"`.
- New `_load_subject_banks(checkpoint_dir, runs=(4,8,12)) -> dict[int, dict[int, SyntheticBank]]`: loads `subject_banks_holdout_run{R}.npz`, groups rows by `subject_id` into per-subject `SyntheticBank(held_out_run=R, X, y, subject_id=S)`. (No GPU / no diffusion load needed — banks are precomputed.)
- `_load_synthetic_set(...)` extended: when `variant == "subj"`, select `bank = banks[test_run][subject]`; assert `bank.held_out_run == test_run and bank.subject_id == subject`; per-class matched draw exactly as today.
- `train()`: for `diffusion_variant == "subj"`, call `_load_subject_banks` once before the loop and index `banks[split["test_run"]][subj]` per fold. JSON stem `per_subject_results_tstr_subj_{n}x`; MLflow experiment `pivot2-tstr`.

### `eval/compare_baselines.py` — additions

```python
# candidates: tstr_subj_1x/2x/3x
# print_gate_verdict(results, tstr_key="tstr_subj_2x",
#                    secondary_keys=("tstr_subj_1x","tstr_subj_2x","tstr_subj_3x"),
#                    lift_key="tstr_exec_2x")   # + a second call with lift_key="tstr_2x"
# fig10: add the subj curve; fig14: per-subject (tstr_subj_2x − tstr_exec_2x) tier-colored
```

### Modal launchers

```
# Train 3 subject-conditional generators + export per-subject banks (~3-4 h):
modal run --detach modal_train_diffusion_subj.py
modal volume get neurosynth-checkpoints diffusion_subj_holdout_run{4,8,12}.pt      models/checkpoints/
modal volume get neurosynth-checkpoints subject_banks_holdout_run{4,8,12}.npz      models/checkpoints/

# TSTR passes from the cached banks (~1 h; pure classifier work, no diffusion load):
modal run --detach modal_train_tstr.py --variant subj
modal volume get neurosynth-checkpoints per_subject_results_tstr_subj_{1,2,3}x.json models/checkpoints/
python eval/compare_baselines.py
```

`modal_train_diffusion_subj.py` mirrors `modal_train_diffusion_exec.py` (mounts `processed_with_exec.npz`, `use_subject_cond=True`, `export_subject_banks=True`) and inherits the preemption hardening (auto-resume from newest checkpoint + `on_checkpoint` volume commit). `modal_train_tstr.py --variant subj` uploads the cached bank npz mount and loads them (no per-container generation).

---

## Implementation Plan

### Phase 1 — Diffusion: subject conditioning (backward-compatible)
- [ ] `DiffusionConfig`: add `use_subject_cond`, `n_subjects`, `p_subj_uncond`, `subject_ids`, `subject_guidance_scale`, `null_subject`.
- [ ] `UNet1D`: optional `subject_emb`; thread `subject` through `forward`.
- [ ] `DDPM.train_step`/`sample`: accept/pass `subject`; subject dropout → null in training; subject unguided at sample time.
- [ ] `EEGDiffusionModel`: build `_subj_to_idx` from `config.subject_ids`; `generate(subject_id=…)` maps and uses it when `use_subject_cond`, ignores it otherwise.
- [ ] Regression check: a Pivot 1 (`use_subject_cond=False`) checkpoint generates identical output pre/post.

### Phase 2 — Diffusion trainer + bank export + launcher
- [ ] `train_diffusion.py`: `use_subject_cond`, `p_subj_uncond`, `export_subject_banks`, `bank_size_per_subject`; build subject index map; feed subject per batch; after training export `subject_banks_holdout_run{R}.npz` (+ per-subject memorization/PSD spot-check).
- [ ] `modal_train_diffusion_subj.py`: thin copy of the exec launcher with subject conditioning + bank export; inherits preemption hardening.

### Phase 3 — TSTR harness (two-key banks)
- [ ] `SyntheticBank.subject_id`; `_VARIANT_CKPT_PREFIX["subj"]`.
- [ ] `_load_subject_banks(...)` (loads npz, groups by subject).
- [ ] `_load_synthetic_set(...)`: two-key select + assert for `subj`.
- [ ] `train_baseline.py`: `--diffusion-variant subj` routing; JSON stem; `pivot2-tstr` experiment.
- [ ] `modal_train_tstr.py --variant subj`: mount + load cached banks (no diffusion generation).

### Phase 4 — Comparison & gate
- [ ] `compare_baselines.py`: load `tstr_subj_*`; gate (`tstr_subj_2x` vs `aug_2x`); lift vs `tstr_exec_2x` and `tstr_2x`; fig10 subj curve; fig14 subj-vs-exec delta; CSV columns.

### Phase 5 — Run, collect, report
- [ ] Train the three generators + export banks on Modal; validate fidelity + subject-embedding-used probe + per-subject memorization.
- [ ] TSTR passes; pull JSONs; `python eval/compare_baselines.py`.
- [ ] `reports/pivot_iter02_subject_conditional_tstr_report.md`; update `phase0_baseline_report.md` §12.5 and memory.

---

## Error Handling

- **Missing `subject_banks_holdout_run{R}.npz`** → harness `sys.exit`s with the `modal_train_diffusion_subj.py` command to produce it.
- **Unknown subject at generate** (`subject_id ∉ config.subject_ids`) → `KeyError` — the model was not trained on that subject (a Non-Goal path).
- **Two-key routing mismatch** (`bank.held_out_run != test_run` or `bank.subject_id != subject`) → `AssertionError`, hard stop (leakage guard).
- **Missing subject in a bank file** (a fold's subject absent from `subject_banks_holdout_run{R}.npz`) → `KeyError` with the subject id; should not happen (banks are generated for the full cohort).
- **Per-subject bank too small for the draw** → `ValueError` telling the user to raise `bank_size_per_subject`.
- **Sampler-health / per-subject memorization failure** → abort before the TSTR passes (loud, before compute).
- **Backward-compat**: any checkpoint without `use_subject_cond` loads with it `False` (no subject head), behaving exactly as Step 4/5/Pivot 1.
- **Modal**: reuse the Pivot-1 preemption hardening (auto-resume + per-save commit).

---

## Security Considerations

Research code, own data, cloud-compute only. Integrity concerns:

- **Leakage integrity** — the two-key `(subject, run)` assert plus the structural guarantee (the model never saw the test run of anyone; the subject embedding is learned only from that subject's non-test runs) is the single most important invariant.
- **Per-subject memorization** — because the generator now conditions on the subject it could copy, the memorization guard is elevated to a *per-subject* check (generated-for-S must stay no closer to S's real epochs than S's real-to-real NN). A memorizing subject-conditional generator would smuggle a subject's *training* epochs into the synthetic set; that is not test-run leakage (the classifier has those epochs already) but it collapses "synthetic" into "duplicated real" and must be reported.
- **Checkpoint trust** — `torch.load(weights_only=False)` on our own checkpoints only.

---

## Testing Strategy

- **Subject conditioning (unit):** with `use_subject_cond=True`, `generate(subject_id=A)` ≠ `generate(subject_id=B)` at a fixed seed; both pass the `std≈1.0` guard. With `use_subject_cond=False`, the `subject_id` arg is inert (numerically identical output).
- **Subject-id mapping (unit):** `config.subject_ids` round-trips through save/load; `_subj_to_idx` inverts it; unknown subject → `KeyError`.
- **Subject dropout (unit):** over many `train_step` calls, ~`p_subj_uncond` fraction of the subject batch equals `null_subject`.
- **Two-key routing:** every LORO fold `(S, R)` selects `banks[R][S]` with `held_out_run==R` and `subject_id==S`; execution never a test run; an intentional mis-key raises `AssertionError`.
- **Bank export/load round-trip:** `subject_banks_holdout_run{R}.npz` groups back into per-subject banks with correct `subject_id`, class balance, and `std≈1.0`.
- **`_load_synthetic_set` for subj:** shape/balance/determinism identical to the population path, on a per-subject bank.
- **Per-subject memorization (Phase 5 gate):** for a sample of subjects, generated-for-S NN-distance to S's real ≥ S's real-to-real NN.
- **Subject-embedding-used probe (Phase 5):** per-subject-conditioned PSDs track each subject's real PSD better than the population model's; embeddings are not all-equal.
- **Gate logic (unit):** `print_gate_verdict(tstr_key="tstr_subj_2x", lift_key="tstr_exec_2x")` reproduces PASS/FAIL on synthetic `RunResult`s.
- **Smoke run:** tiny cohort, few steps, `bank_size_per_subject` small → banks export, `--diffusion-variant subj` routes and writes `tstr_subj_2x`.

---

## Success Criteria

**Implementation complete when:**
- `--use-subject-cond` trains three `diffusion_subj_holdout_run{4,8,12}.pt` and exports `subject_banks_holdout_run{4,8,12}.npz`; Pivot 1 / Step 4 checkpoints still load unchanged.
- Subject conditioning passes the unit checks (A≠B, inert when off, dropout rate, mapping).
- `--run tstr --diffusion-variant subj` routes every fold two-key, leakage-guarded, and writes `per_subject_results_tstr_subj_{1,2,3}x.json`.
- `compare_baselines.py` prints the subj arm, its RR-3 verdict, and the lift over `tstr_exec_2x` / `tstr_2x`, and saves the figures.

**Experiment complete when:**
- Generators pass fidelity + per-subject memorization + the subject-embedding-used probe.
- The RR-3 verdict is computed: **PASS** iff `mean(tstr_subj_2x) − mean(aug_2x) > +1.0%` AND one-sided paired Wilcoxon (`tstr_subj_2x > aug_2x`, 106 subjects) `p < 0.05`.
- The lift over the population arms (`tstr_subj_2x − tstr_exec_2x`, `− tstr_2x`) is reported with tier breakdown.
- `reports/pivot_iter02_...md` documents the outcome and decision.

**Decision output:**
- **PASS** → subject-conditional synthetic beats augmentation; the core NeuroSynth thesis is validated. First positive result; funds scaling the generator with more per-subject/real data.
- **PARTIAL** (beats the population arms and/or vanilla but not `aug_2x`) → per-subject conditioning helps but not enough at 30-epoch calibration; the lever is real but data-bound → the honest next question is whether *more real calibration per subject* (the one thing that reliably moves this) is the product, not more synthesis.
- **FAIL / no lift over population** → subject conditioning cannot extract a useful per-subject signal from ~75 epochs; synthetic augmentation of tiny imagery calibration does not work at this rung. Report the negative honestly and reconsider the task/benchmark (e.g. execution-task BCI, or more calibration data) rather than the generator.

---

## Compute & Honest Prognosis

| Stage | Work | Hardware | Time |
|---|---|---|---|
| Generators + bank export | 3 × 100K steps (subject-conditioned) + per-model 106-subject bank gen (~13.6K samples/model) | A10G ×3 | ~3–4 h, ~$10–13 |
| TSTR (cached banks) | 3 passes × 318 folds (no diffusion load) | A10G ×3 | ~1 h, ~$2–3 |

Total ≈ **$12–16**, ~4–5 h across two `--detach` launches. Exceeds the free Modal tier — a funded account is a prerequisite.

**Honest prognosis (unchanged from prior analysis).** This is the right experiment but not a likely home run. The wall is data: each subject offers ~75 epochs (imagery + execution) to learn its embedding, and the augmentation curve already plateaus at ~59.5%, which bounds what expanding a subject's own data — by any method — can achieve. Most-likely outcome is **PARTIAL**: land at or modestly above the population arms, with the gains concentrated in the **data-limited / mud-zone** subjects (whose own imagery is weakest, so a subject-anchored prior helps most) and, critically, **without the Learning/Strong-tier harm** the population arms caused (subject conditioning should at least stop degrading the good subjects). Clearing +1% over `aug_2x=59.5%` cohort-wide is possible but would be a strong result. The residual-embedding design and the execution prior are the two things that make even a PARTIAL win plausible; if it fails, it fails cleanly and tells us the bottleneck is real-data volume, not generation — which is itself a decisive, publishable conclusion for the whole population/per-subject rung.

---

## Future Directions (post-Pivot 2)

Two levers surfaced during review that add *information* rather than variance (unlike augmenting a subject's own epochs, which only re-expresses what those ~75 epochs already contain and is bounded by the `aug_2x`=59.5% ceiling). Both are **conditional follow-ons, out of this pivot's implementation scope** — the natural next steps if Pivot 2 lands PARTIAL (item 1 is also cheap enough to stack on a PASS).

### 1. Hybrid arm — real + augmentation + subject-synthetic (stacked)

- **Rationale:** the two levers are complementary — augmentation injects the variance that regularizes weak/data-limited-tier spatial filters, while subject-synthetic injects topography-preserving on-manifold diversity. Neither beats `aug_2x`=59.5% alone; stacked they might. This is the cheapest untested shot at the ceiling (no new generator).
- **Sketch:** extend `_load_synthetic_set` to compose a fold's extra-N budget from both sources (e.g. N/2 augmented via the existing `_create_offline_augmented_set` path + N/2 subject-synthetic via the two-key banks), per class, leakage-routed. New arm `tstr_subj_hybrid_{1,2,3}x`.
- **Cost / gate:** ~1 h TSTR, no training. Same RR-3 gate (`> aug_2x` by +1%, one-sided paired Wilcoxon p<0.05); also report vs `tstr_subj_2x` to isolate the stacking benefit.

### 2. Subject-similarity conditioning (cross-subject transfer)

- **Rationale:** the only way past the per-subject data wall is to add information from *other* subjects. Instead of (or alongside) a per-subject identity embedding learned from S's ~75 epochs, condition on S's k nearest-neighbor subjects so S borrows structure from physiologically similar people — the one lever that genuinely adds information rather than re-expressing S's own epochs.
- **Sketch:** precompute a subject-similarity matrix (Riemannian distance between mean spatial covariances, or the Euclidean-Alignment covariance distance from `phase0_baseline_report.md` §322); condition the generator on a similarity-weighted neighbor-embedding combination (or a soft-neighbor auxiliary token). Leakage unchanged (still run-holdout; neighbors computed only from training runs).
- **Cost / risk:** a new generator training pass (similarity head) + TSTR — comparable to Pivot 2. Most likely of the two to clear the wall, but the most involved — the natural next *pivot* if Pivot 2 lands PARTIAL.

---

## Open Questions

- [ ] `p_subj_uncond` value: 0.15 assumed. Sweep {0.1, 0.15, 0.3} if the primary arm is ambiguous (Nice to Have).
- [ ] `bank_size_per_subject`: 128 (64/class) assumed — enough for 3× draws; raise only if a fold needs more.
- [ ] Optional subject-CFG guidance (`subject_guidance_scale`): off by default; consider only if per-subject PSDs under-track and the embedding looks under-used.
- [ ] Whether to also sweep more training steps for the subject models (learning a per-subject residual may need longer than the population models' 100K) — decide from the loss/fidelity curves after the first run.
