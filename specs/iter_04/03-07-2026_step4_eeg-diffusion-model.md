# Phase 0 Step 4 — EEG Diffusion Model: 1D U-Net DDPM for Class-Conditional Generation

**Status:** Draft — revised 2026-07-03 (population-first, leave-one-run-out cross-fitting; memorization guard)  
**Created:** 2026-07-03  
**Depends on:** Phase 0 Step 3b (additive augmented baseline, 59.5% at 2×) — complete  
**Feeds into:** Phase 0 Step 5 (TSTR harness + go/no-go gate) — separate spec  

---

## Executive Summary

Phase 0 Step 4 trains **population-level** denoising diffusion probabilistic models (DDPMs) to generate class-conditional synthetic EEG epochs. To keep the downstream go/no-go gate leakage-free *while still evaluating every subject*, training mirrors the classifier's own Leave-One-Run-Out (LORO) protocol: **three** population models are trained, each excluding one of the three motor-imagery runs (4, 8, 12) across all 106 subjects. In Phase 5, each LORO fold (which tests on run *R*) draws synthetic data from the model that never saw run *R* — so the synthetic augmentation for any fold provably carries no information from that fold's test run. This is the identical cleanliness standard as the existing offline augmentation baseline (which only ever touches a fold's *training* runs), so the synthetic arm compares **directly against the full-cohort 56.7% / 59.5% baselines** with no reduction in subjects or statistical power.

Each model is class-conditional (left/right fist) and **population-level (subject-agnostic)**: it generates generic motor-imagery epochs, not per-subject data. `generate(n, subject_id)` accepts `subject_id` for Phase 1 compatibility but ignores it. No architectural changes to EEGNet or the LORO evaluation protocol are required.

This run-holdout structure exists because Phase 5's within-subject LORO tests on each run in turn. A single generator trained on all data would have seen every fold's test run before generating for it, and a DDPM at this parameter-to-data ratio can memorise training samples — either would inflate TSTR accuracy and produce a false positive on the RR-3 gate. See *Leave-one-run-out cross-fitting* below.

---

## Goals

- Train **three** population DDPMs (1D U-Net backbone), each excluding one motor-imagery run (4, 8, 12) across all 106 subjects — leave-one-run-out cross-fitting (~3160 epochs each of `data/processed.npz`, 64 channels, 801 time points)
- Implement classifier-free guidance (CFG) for binary class conditioning (left fist = 0, right fist = 1)
- Produce a `EEGDiffusionModel` wrapper with a fixed interface, exposing `held_out_run` so Phase 5 routes each LORO fold to the model that never saw that fold's test run
- Log training to MLflow (`phase0-diffusion` experiment) with loss curves, fidelity snapshots, and a nearest-neighbor memorization metric
- Save three checkpoints (`models/checkpoints/diffusion_holdout_run{4,8,12}.pt`) with model weights, config, and `held_out_run`
- Verify generation quality with PSD comparison, basic distributional checks, and a memorization guard before Phase 5

## Non-Goals

- Subject-conditional generation (Phase 1 — subject identity conditioning added in the next research phase)
- TSTR evaluation (Phase 0 Step 5 — this spec only trains and validates the generative model)
- Format export (EDF+, BIDS) — Phase 1 product infrastructure
- Hyperparameter search / NAS — single architecture; three run-holdout training runs, no per-model tuning
- Real-time generation latency optimisation — offline research code only

---

## Background & Context

### Why DDPM?

DDPM (Ho et al. 2020) is the standard choice for high-quality time-series generation. Compared to GANs:
- Training is stable (no mode collapse, no adversarial game)
- Quality scales with compute, not GAN tricks
- Well-understood variance decomposition makes the loss function interpretable
- Literature precedent: multiple papers use DDPM variants for EEG/ECG generation (DiffuSSM, SSSD, EEG-specific DDPM papers)

Compared to VAEs, DDPM produces sharper, more realistic samples because it models the full data distribution without the blurriness that comes from optimising reconstruction loss.

### Why 1D U-Net?

The EEG signal has shape (C=64, T=801): 64 spatial channels, 801 time samples. A 1D U-Net with Conv1d treats time as the sequence axis and channels as the feature dimension. This:
- Preserves temporal structure through the convolutional receptive field
- Shares weights across time (no positional overfit)
- Is parameter-efficient compared to treating the signal as a 2D image
- Allows skip connections to carry fine temporal detail from encoder to decoder

The "1D" refers to the convolution dimension, not the model depth. The model operates on tensors of shape (B, C_feat, T) throughout.

### Why class-conditional only (no subject conditioning)?

The research question — does synthetic data beat augmentation for motor imagery classification? — can be answered at the population level. A population model generates plausible left/right fist signals drawn from the full distribution of all 106 subjects (minus one held-out run). Subject-to-subject variability within a class is captured by the model's learned variance. This is the simplest testable hypothesis, and — crucially — being subject-agnostic is what lets the run-holdout cross-fitting stay leakage-free (see below).

Subject-conditional generation (conditioning on subject ID) would require the model to learn 106 different per-subject distributions, which is harder with 4748 total training samples (~45 per subject) and raises overfitting risk. That is Phase 1 scope.

The `generate(subject_id=...)` parameter is accepted at the interface level but ignored in Phase 0 (population model), ensuring Phase 5 and Phase 1 code is identical.

### Leave-one-run-out cross-fitting (leakage prevention)

**Problem.** Phase 5 evaluates TSTR with within-subject Leave-One-Run-Out (LORO): each of a subject's three motor-imagery runs (4, 8, 12) is used as the held-out test run in turn (`within_subject_loro_split` in `eval/dataset.py`). If a *single* diffusion model were trained on the full 4748-epoch dataset, it would have already seen every fold's test-run epochs *before* generating augmentation data for that fold. Worse, with ~5–15M parameters over only ~4700 samples, DDPMs are known to memorise individual training examples (Carlini et al. 2023, "Extracting Training Data from Diffusion Models"). Either way the synthetic augmentation would carry test-run information, inflating TSTR accuracy for reasons unrelated to synthetic-data quality — a false positive on the RR-3 go/no-go gate, the single most important result of Phase 0. The PSD/mean-std fidelity checks would **not** catch this: a memorised generator passes them trivially.

**Why the augmentation baseline is already clean.** The offline augmentation baseline (`_create_offline_augmented_set`) only ever jitters a fold's *training* runs; the test run never touches the augmented data. The synthetic arm must be held to the same standard.

**Fix — mirror LORO at the generator level.** The classifier's test unit is the *run*, so we cross-fit at the run level. Train **three** population diffusion models:

| Model | Trained on | Never sees | Used for folds testing on |
|---|---|---|---|
| `D_run4`  | runs 8 & 12 (all 106 subjects) | run 4  | run 4  |
| `D_run8`  | runs 4 & 12 (all 106 subjects) | run 8  | run 8  |
| `D_run12` | runs 4 & 8  (all 106 subjects) | run 12 | run 12 |

For any LORO fold, Phase 5 pulls synthetic data from the model whose `held_out_run` equals that fold's test run. Because that model never saw that run (of *any* subject), the synthetic data provably contains no test-run information — clean by the exact standard as augmentation. All 106 subjects are evaluated (three folds each, 318 total), so the synthetic arm compares **directly** against the existing full-cohort 56.7% / 59.5% baselines — no reduction in cohort or statistical power, no baseline re-training.

**Why population, not subject-conditional (Phase 0).** These are subject-agnostic generators — generic left/right-fist epochs, not subject-specific data. This is what keeps run-holdout airtight: since generation is not targeted at any subject, a model can safely have seen a subject's *other* runs (those runs are that fold's legitimate training data anyway, exactly as augmentation uses them). Subject-conditional generation (Phase 1) would break this — a model conditioned on subject S and trained on S's runs 4 & 8 could echo S's run 12 — and would require the stricter *subject-holdout* scheme. That is deliberately out of scope here.

**Memorization guard.** Alongside PSD divergence, each model logs `nn_distance`: the mean L2 distance from a batch of generated samples to their nearest real *training* sample, compared against the real-to-real nearest-neighbor distance. If generated samples sit systematically closer to real samples than real samples sit to each other, the model is memorising and the checkpoint must be rejected (or an earlier, less-converged checkpoint chosen).

**Cost & caveat.** Three training runs instead of one (still cheap — see *Compute*). Each model trains on ~2/3 of the data (~3160 epochs), so it is mildly more data-starved than a full-data model; this biases the gate *conservatively* (a pass is trustworthy; a marginal fail could be data-starvation rather than a bad method). Before pooling fold results, verify the three models have **comparable fidelity** (PSD divergence, `nn_distance`), so fold-to-fold differences reflect the held-out run rather than generator quality.

### Relationship to TSTR (Phase 5)

The V08 spec established the exact integration point. Phase 5 replaces one function call:

```python
# Current (augmented_offline in train_baseline.py):
X_combined, y_combined = _create_offline_augmented_set(
    X_train, y_train, n_copies=N, augmentor=STANDARD_AUGMENTOR
)

# Phase 5 replacement (routes to the model that never saw this fold's test run):
X_combined, y_combined = _load_synthetic_set(
    X_train, y_train,
    subject_id=subject_id,
    n_synthetic=N,
    diffusion_model=diffusion_models[test_run],   # dict keyed by held-out run
)
```

The LORO training loop, evaluation, MLflow logging, `compare_baselines.py`, and reporting are all unchanged. The one behavioural difference is that Phase 5 selects, per fold, the diffusion model whose `held_out_run` matches that fold's test run (a dict of three models keyed by run), and compares against the existing full-cohort baselines directly. Phase 4's obligation is to deliver three `EEGDiffusionModel` checkpoints, each reporting its `held_out_run`.

---

## Requirements

### Must Have

- Train **three** population DDPMs, each excluding one motor-imagery run (4 / 8 / 12) across all 106 subjects — leave-one-run-out cross-fitting
- Embed `held_out_run` (the run this model never saw) in every checkpoint and expose it on `EEGDiffusionModel`, so Phase 5 routes each fold to the correct model
- Implement `UNet1D`: 1D U-Net denoising backbone with sinusoidal time embedding and class conditioning
- Implement `DDPM`: training-step logic (forward diffusion + noise prediction loss) and inference sampler
- Implement `EEGDiffusionModel`: public interface wrapper (`generate`, `labels`, `save`, `load`)
- Cosine noise schedule (Nichol & Dhariwal 2021): smoother than linear, prevents over-noising at t≈T
- Classifier-free guidance with p_uncond=0.1 (10% of batches drop the class label during training)
- DDIM sampler at inference (50 steps), replacing the full T=1000 reverse chain for tractable generation
- Training script `eval/train_diffusion.py` with MLflow logging (experiment: `phase0-diffusion`)
- Save checkpoint every 10K steps; keep last 3 checkpoints + final
- After training: generate 100 synthetic samples per class and save PSD comparison figure

### Should Have

- `nn_distance` memorization metric (per model): mean L2 from a probe batch of generated samples to the nearest real training sample, logged periodically to MLflow and compared against the real-to-real nearest-neighbor baseline
- EMA (exponential moving average) of model weights, decay=0.9999: standard for DDPM quality
- Gradient clipping (max_norm=1.0): prevents instability in early training
- `--resume` flag on training script to continue from a checkpoint
- PSD divergence scalar logged to MLflow as a fidelity proxy metric
- Per-channel mean/std comparison plot saved to `reports/figures/diffusion_fidelity_run{R}.png` (one per model)

### Nice to Have

- DDIM sampler with configurable η (η=0 deterministic, η=1 stochastic DDPM): useful for ablation
- FID-style feature-space distance using EEGNet features rather than Inception (EEG-appropriate)
- `--n-samples N` flag to generate N samples at training end for inspection

---

## Architecture Overview

### System context

```
data/processed.npz
  X=(4748, 64, 801) float32
  y=(4748,) int64 {0=left, 1=right}
  run_ids=(4748,)  ∈ {4, 8, 12}
        │
        │  filter: exclude ONE run (4, 8, or 12) — all 106 subjects kept
        ▼
  X_train ≈ (3160, 64, 801)   ← this model trains on the other two runs
        │
        ▼
┌────────────────────────┐
│  eval/train_diffusion  │  ← run 3× (--holdout-run 4 / 8 / 12)
│  DDPM.train_step()     │
│  EMA weight averaging  │
│  MLflow logging        │
│  nn_distance guard     │
└────────────┬───────────┘
             │ saves 3 checkpoints (weights + config + held_out_run)
             ▼
models/checkpoints/diffusion_holdout_run{4,8,12}.pt
             │
             │ each loaded by
             ▼
┌────────────────────────────────┐
│  EEGDiffusionModel             │  ← public interface (new)
│  .generate(n, subj_id)         │
│  .labels(n)   .held_out_run    │
└────────────┬───────────────────┘
             │ Phase 5 builds {4: D_run4, 8: D_run8, 12: D_run12}
             │ and routes fold(test_run=R) → D_run{R}
             ▼
eval/train_baseline.py   ← unchanged until Phase 5 adds _load_synthetic_set()
```

### Diffusion model internals

```
x_noisy: (B, 64, 801)
   │
   │ pad to T=1024
   ▼
x_pad: (B, 64, 1024)

   ┌─── time embedding ──────────────────────────────────┐
   │  t ∈ {0..999} → sin/cos → Linear(256→512) → SiLU   │
   │  → Linear(512→512)                       emb_t: 512  │
   └─────────────────────────────────────────────────────┘
   ┌─── class embedding ─────────────────────────────────┐
   │  c ∈ {0,1,null} → Embedding(3, 512)     emb_c: 512  │
   └─────────────────────────────────────────────────────┘
   cond = emb_t + emb_c   ← combined conditioning signal

   ┌─────────── ENCODER ────────────────────────────────────────┐
   │  Stem: Conv1d(64→128, k=7, p=3)  →  h_stem: (B,128,1024)  │
   │  Enc1: 2×ResBlock(128, cond) → h_1: (B,128,1024)           │
   │        ↓ Downsample: Conv1d(128→256, k=4, s=2, p=1)         │
   │  Enc2: 2×ResBlock(256, cond) → h_2: (B,256,512)             │
   │        ↓ Downsample: Conv1d(256→512, k=4, s=2, p=1)         │
   │  Enc3: 2×ResBlock(512, cond) → h_3: (B,512,256)             │
   │        ↓ Downsample: Conv1d(512→512, k=4, s=2, p=1)         │
   └──────────────────────────────────────────────────────────────┘

   ┌─────────── BOTTLENECK ──────────────────────────────────────┐
   │  ResBlock(512, cond)                                         │
   │  SelfAttention1D(512, heads=8)                               │
   │  ResBlock(512, cond)            →  (B,512,128)               │
   └──────────────────────────────────────────────────────────────┘

   ┌─────────── DECODER ────────────────────────────────────────┐
   │        ↑ Upsample×2 + Conv1d(512→512, k=3, p=1)            │
   │  Dec3: Cat(h_3): (B,1024,256) → 2×ResBlock(1024→512,cond)  │
   │        ↑ Upsample×2 + Conv1d(512→256, k=3, p=1)            │
   │  Dec2: Cat(h_2): (B,512,512)  → 2×ResBlock(512→256,cond)   │
   │        ↑ Upsample×2 + Conv1d(256→128, k=3, p=1)            │
   │  Dec1: Cat(h_1): (B,256,1024) → 2×ResBlock(256→128,cond)   │
   │  Head: GroupNorm(128) → SiLU → Conv1d(128→64, k=3, p=1)    │
   │        crop [:,:,:801]          →  (B,64,801)                │
   └──────────────────────────────────────────────────────────────┘

Output: ε_pred: (B, 64, 801)   ← predicted noise for MSE loss
```

### ResBlock1D (used throughout encoder/decoder)

```
ResBlock1D(C_in, C_out, cond_dim=512):
  GroupNorm(C_in) → SiLU → Conv1d(C_in, C_out, k=3, p=1)
  + Linear(cond_dim, C_out) [applied element-wise along T axis, broadcast]
  GroupNorm(C_out) → SiLU → Conv1d(C_out, C_out, k=3, p=1)
  + residual connection (Conv1d(C_in, C_out, k=1) if C_in ≠ C_out else identity)
```

### SelfAttention1D (bottleneck only)

```
SelfAttention1D(C, heads=8):
  GroupNorm(C)
  → reshape: (B, C, T) → (B, T, C)           ← sequence of T tokens, each dim C
  → MultiheadAttention(C, heads, batch_first=True)
  → reshape back: (B, C, T)
  + residual
```

### Noise schedule

Cosine schedule (Nichol & Dhariwal 2021), T=1000 timesteps:

```python
s = 0.008
t_arr = torch.arange(T + 1, dtype=torch.float64)
alpha_bar = torch.cos(((t_arr / T + s) / (1 + s)) * (math.pi / 2)) ** 2
alpha_bar = alpha_bar / alpha_bar[0]         # normalise so α̅_0 = 1
beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
beta = beta.clamp(0, 0.999)
```

### Classifier-free guidance (inference)

At inference time, run two forward passes per denoising step:
1. Unconditional: class label set to null token (embedding index 2)
2. Conditional: actual class label (0 or 1)

```python
eps_uncond = model(x_t, t, class_label=null_token)
eps_cond   = model(x_t, t, class_label=c)
eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

Default guidance_scale = 3.0. Higher values produce more class-discriminative samples at the cost of diversity.

---

## Data Models

### ModelConfig

```python
@dataclass
class DiffusionConfig:
    # Signal
    n_channels: int = 64
    n_times: int = 801
    n_pad: int = 1024         # internal working length (next power of 2 above 801)
    n_classes: int = 2

    # U-Net
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 4)   # → 128, 256, 512
    n_res_blocks: int = 2
    attn_resolution: int = 128         # apply self-attention at T=128 only (bottleneck)

    # Diffusion
    T: int = 1000
    schedule: str = "cosine"
    p_uncond: float = 0.1              # CFG dropout probability during training

    # Conditioning
    cond_dim: int = 512                # time + class embedding dimension

    # Cross-fitting (leakage prevention — see Background)
    holdout_run: int = 12              # motor-imagery run excluded from THIS model's training
                                       # (4, 8, or 12); one model trained per run → 3 models total

    # Training
    lr: float = 1e-4
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    batch_size: int = 32
    n_steps: int = 100_000             # ~1000 epochs over ~3160 samples; monitor nn_distance
                                       # for memorization rather than training to a fixed count

    # Inference
    ddim_steps: int = 50
    guidance_scale: float = 3.0
```

### Checkpoint format

```python
# Saved with torch.save() to models/checkpoints/diffusion_holdout_run{R}.pt
{
    "model_state_dict":     OrderedDict,    # UNet1D weights
    "ema_state_dict":       OrderedDict,    # EMA shadow weights (used for inference)
    "config":               dict,           # DiffusionConfig.__dict__
    "held_out_run":         int,            # run this model never saw (4, 8, or 12)
    "train_runs":           list[int],      # the two runs it trained on
    "step":                 int,            # training step at save
    "train_loss":           float,          # running mean loss at save
    "nn_distance":          float,          # memorization metric at save (gen→nearest-real L2)
    "mlflow_run_id":        str,            # MLflow run for traceability
}
```

Intermediate checkpoints: `models/checkpoints/diffusion_run{R}_step_{N}.pt`

### MLflow parameters logged per run

```python
{
    "experiment": "phase0-diffusion",
    "model": "ddpm_unet1d",
    "n_channels": 64,
    "n_times": 801,
    "T": 1000,
    "schedule": "cosine",
    "base_channels": 128,
    "channel_mults": "1,2,4",
    "n_res_blocks": 2,
    "holdout_run": 12,                  # run excluded from THIS model (4, 8, or 12)
    "train_runs": "4,8",                # the two runs trained on
    "n_subjects": 106,                  # all subjects kept; only one run per subject excluded
    "n_train_samples": 3160,            # ~epochs after excluding the held-out run
    "batch_size": 32,
    "lr": 1e-4,
    "ema_decay": 0.9999,
    "p_uncond": 0.1,
    "guidance_scale": 3.0,
    "ddim_steps": 50,
}
# One MLflow run per model (3 total); "holdout_run" distinguishes them.
# Metrics (logged every 500 steps):
# "train_loss", "psd_divergence"
# Metrics (logged every save_every steps):
# "nn_distance"  (mean gen→nearest-real L2; memorization guard)
```

---

## Interface Definitions

### `EEGDiffusionModel` — public interface consumed by Phase 5

```python
class EEGDiffusionModel:
    """
    Population-level class-conditional EEG generator.
    Phase 0: subject_id is accepted but ignored (population model).
    Trained on all subjects but excluding one motor-imagery run; leakage-free
    for LORO folds whose test run == held_out_run.
    """

    held_out_run: int    # motor-imagery run this model never saw (4, 8, or 12).
                         # Phase 5 routes each LORO fold to the model whose held_out_run == test_run.

    def generate(
        self,
        n: int,
        subject_id: int,          # ignored in Phase 0; accepted for Phase 1 compatibility
        class_label: int | None = None,   # None → balanced generation
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate n synthetic EEG epochs using DDIM sampling.
        
        Returns:
            X_synth: float32 ndarray, shape (n, 64, 801)
                     In the same z-scored space as data/processed.npz.
        
        If class_label is None, generates ceil(n/2) class-0 and floor(n/2) class-1
        epochs interleaved [0,1,0,1,...], consistent with self.labels(n).
        """

    def labels(self, n: int) -> np.ndarray:
        """
        Return the class labels corresponding to self.generate(n, ..., class_label=None).
        
        Returns:
            y_synth: int64 ndarray, shape (n,), values in {0, 1}
                     Balanced: [0, 1, 0, 1, ...], ceil(n/2) zeros.
        """

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device | None = None) -> "EEGDiffusionModel":
        """Load a trained model from checkpoint."""

    def save(self, checkpoint_path: Path) -> None:
        """Save current EMA weights and config to checkpoint."""
```

### `UNet1D` — model backbone

```python
class UNet1D(nn.Module):
    def __init__(self, config: DiffusionConfig): ...

    def forward(
        self,
        x: Tensor,           # (B, 64, 801) noisy signal
        t: Tensor,           # (B,) int, diffusion timestep in [0, T-1]
        class_label: Tensor, # (B,) int, class in {0, 1, 2=null}
    ) -> Tensor:             # (B, 64, 801) predicted noise ε
```

### `DDPM` — training and sampling logic

```python
class DDPM:
    def __init__(self, model: UNet1D, config: DiffusionConfig, device: torch.device): ...

    def train_step(
        self,
        x0: Tensor,      # (B, 64, 801) clean batch
        y: Tensor,       # (B,) class labels
    ) -> float:          # scalar loss value

    @torch.no_grad()
    def sample(
        self,
        n: int,
        class_label: int,
        ddim_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int | None = None,
    ) -> Tensor:         # (n, 64, 801) generated samples
```

### `eval/train_diffusion.py` — CLI

```
Usage:
    # Train one model per held-out run — run three times:
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 4
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 8
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 --n-steps 100000
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 --resume models/checkpoints/diffusion_run12_step_50000.pt

Arguments:
    --data PATH               Path to processed.npz (default: data/processed.npz)
    --holdout-run {4,8,12}    Motor-imagery run excluded from THIS model's training (REQUIRED)
    --checkpoint-dir PATH     Output directory (default: models/checkpoints)
    --n-steps INT             Training steps (default: 100000)
    --batch-size INT          Mini-batch size (default: 32)
    --lr FLOAT                Learning rate (default: 1e-4)
    --guidance-scale FLOAT    CFG scale for fidelity evaluation (default: 3.0)
    --ddim-steps INT          DDIM inference steps (default: 50)
    --seed INT                RNG seed (default: 42)
    --resume PATH             Resume from checkpoint
    --save-every INT          Save checkpoint every N steps (default: 10000)
    --mlflow-experiment STR   MLflow experiment name (default: phase0-diffusion)
```

---

## Implementation Plan

### Phase 1 — Core model: `eval/diffusion.py`

**File: `eval/diffusion.py`** (new file, ~350 lines)

- [ ] Implement `DiffusionConfig` dataclass with all hyperparameters and defaults
- [ ] Implement `SinusoidalTimeEmbedding(dim)`: sinusoidal → 2-layer MLP → cond vector
- [ ] Implement `ResBlock1D(C_in, C_out, cond_dim)`:
  - GroupNorm → SiLU → Conv1d → add projected cond → GroupNorm → SiLU → Conv1d + residual
  - Residual: Conv1d(C_in→C_out, k=1) if C_in ≠ C_out else identity
- [ ] Implement `SelfAttention1D(C, heads)`:
  - GroupNorm → reshape (B,C,T)→(B,T,C) → MHA → reshape back → residual
- [ ] Implement `UNet1D(config)`:
  - Stem: Conv1d(64→128, k=7, p=3)
  - 3 encoder levels, channel multipliers [1,2,4] → channels [128,256,512]
  - Each level: n_res_blocks=2 ResBlock1D + Downsample
  - Downsample: Conv1d(C→C_next, k=4, s=2, p=1)
  - Bottleneck: ResBlock1D + SelfAttention1D + ResBlock1D
  - 3 decoder levels (mirrored), skip connections via channel concatenation
  - Upsample: F.interpolate(scale_factor=2, mode='nearest') + Conv1d(C→C_prev, k=3, p=1)
  - After concat with skip: ResBlock1D(C_skip+C_up → C_prev)
  - Head: GroupNorm → SiLU → Conv1d(128→64, k=3, p=1)
  - Pad input to T=1024 at forward entry; crop output to T=801 before returning
- [ ] Implement class conditioning in `UNet1D.__init__`:
  - `self.class_emb = nn.Embedding(n_classes + 1, cond_dim)` where index n_classes = null token
  - `cond = time_emb + class_emb(c)` before passing to ResBlocks
- [ ] Implement `DDPM.__init__`:
  - Pre-compute cosine schedule arrays: `beta`, `alpha`, `alpha_bar`, `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar`
  - Register as buffers (not parameters) on the device
  - EMA: copy of model weights updated as `shadow = decay*shadow + (1-decay)*param`
- [ ] Implement `DDPM.train_step(x0, y)`:
  ```python
  t = torch.randint(0, T, (B,))
  eps = torch.randn_like(x0)
  x_t = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * eps
  # CFG: drop class label with probability p_uncond
  c = y.clone()
  mask = torch.rand(B) < p_uncond
  c[mask] = null_token
  eps_pred = model(x_t, t, c)
  loss = F.mse_loss(eps_pred, eps)
  return loss
  ```
- [ ] Implement `DDPM._update_ema()`: called after each optimizer step
- [ ] Implement `DDPM.sample(n, class_label, ddim_steps, guidance_scale, seed)`:
  - Select DDIM timestep subsequence of length ddim_steps evenly from [0, T]
  - Start from z ~ N(0,I) shape (n, 64, 1024) [internally padded]
  - For each timestep (reversed):
    ```python
    eps_uncond = ema_model(x_t, t, null_token)
    eps_cond   = ema_model(x_t, t, class_label)
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    # DDIM deterministic update (η=0):
    x0_pred = (x_t - sqrt_one_minus_alpha_bar[t] * eps_guided) / sqrt_alpha_bar[t]
    x_prev  = sqrt_alpha_bar[t_prev] * x0_pred + sqrt_one_minus_alpha_bar[t_prev] * eps_guided
    ```
  - Return x_prev[:,:,:801] (cropped) as float32 numpy array
- [ ] Implement `EEGDiffusionModel`:
  - Wraps `DDPM` + `UNet1D`; holds `held_out_run`
  - `generate(n, subject_id, class_label=None, seed=None)`: calls `DDPM.sample` for each class, interleaves results (subject_id ignored — population model)
  - `labels(n)`: returns np.array([0,1,0,1,...], dtype=int64) of length n
  - `save(path)`: saves EMA state dict + config + `held_out_run` + `train_runs`
  - `load(path, device)`: reconstructs from checkpoint, restoring `held_out_run`

### Phase 2 — Training script: `eval/train_diffusion.py`

**File: `eval/train_diffusion.py`** (new file, ~200 lines)

- [ ] Load `data/processed.npz`, extract X (float32), y (int64), and `run_ids` (int64)
- [ ] Verify expected shape: (4748, 64, 801); warn and continue if shape differs
- [ ] Apply the run-holdout filter (`--holdout-run R`, required):
  - Drop all epochs with `run_id == R` → training set is the other two runs (~3160 epochs, all 106 subjects)
  - Print n_subjects, n_train_epochs, `held_out_run`, per-class balance; assert no epoch with `run_id == R` remains
- [ ] Build PyTorch `TensorDataset(X_tensor, y_tensor)` on the **filtered** train set and `DataLoader(batch_size, shuffle=True, num_workers=0)`
- [ ] Instantiate `DiffusionConfig` (from CLI args), `UNet1D(config)`, `DDPM(model, config, device)`
- [ ] Set up MLflow run: `mlflow.set_experiment(args.mlflow_experiment)`, log all config params
- [ ] Training loop (steps, not epochs):
  ```python
  step = 0
  while step < n_steps:
      for x_batch, y_batch in dataloader:
          loss = ddpm.train_step(x_batch.to(device), y_batch.to(device))
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
          optimizer.step()
          optimizer.zero_grad()
          ddpm._update_ema()
          step += 1
          if step % 500 == 0:
              mlflow.log_metric("train_loss", loss.item(), step=step)
          if step % save_every == 0:
              nn_d = _nn_distance(model_wrapper, X_train, class_probe=32)
              mlflow.log_metric("nn_distance", nn_d, step=step)   # memorization guard
              checkpoint_path = checkpoint_dir / f"diffusion_run{R}_step_{step}.pt"
              model_wrapper.save(checkpoint_path)
              _rotate_checkpoints(checkpoint_dir, keep=3)
          if step >= n_steps:
              break
  ```
- [ ] Resume logic: if `--resume` provided, load checkpoint and set step counter
- [ ] After training completes:
  - Generate 50 samples per class (100 total) using EMA model
  - Compute and log `psd_divergence` scalar to MLflow
  - Compute and log final `nn_distance`, plus the real-to-real nearest-neighbor baseline
    (`nn_distance_real`), so memorization is visible as `nn_distance < nn_distance_real`
  - Save `reports/figures/diffusion_fidelity_run{R}.png` (PSD comparison + channel mean/std plots)
  - Save final checkpoint to `models/checkpoints/diffusion_holdout_run{R}.pt` (embedding `held_out_run`)
- [ ] Run the whole script **three times** (`--holdout-run 4`, `8`, `12`) to produce all three models
- [ ] Implement `_nn_distance(model, X_real_train, class_probe)`:
  - Generate a probe batch, compute mean L2 to the nearest real training sample
    (flatten channels×time; chunk the pairwise distance to stay within memory)
- [ ] Implement `_plot_fidelity(X_real, X_synth, y_real, out_path)`:
  - 2 panels: PSD (real vs. synthetic per class), per-channel mean/std scatter
  - Mark class 0 and class 1 separately in both panels

### Phase 3 — Train and verify

- [ ] Smoke test on CPU (50 steps, batch_size=4): confirm loss decreases, shapes correct
  ```bash
  python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 --n-steps 50 --batch-size 4
  ```
- [ ] Full training run on GPU (Modal / RunPod / local MPS) — one per held-out run:
  ```bash
  python eval/train_diffusion.py --data data/processed.npz --holdout-run 4  --n-steps 100000 --batch-size 32
  python eval/train_diffusion.py --data data/processed.npz --holdout-run 8  --n-steps 100000 --batch-size 32
  python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 --n-steps 100000 --batch-size 32
  ```
- [ ] Verify no leakage (each model): the filtered training set contains **no** epoch with `run_id == held_out_run` (of any subject)
- [ ] Verify memorization guard (each model): final `nn_distance` is not below `nn_distance_real`; if it is, reject the checkpoint or select an earlier one
- [ ] Verify comparable fidelity across the three models (PSD divergence, `nn_distance`) before pooling their fold results
- [ ] Verify `diffusion_fidelity_run{R}.png`: generated PSD should overlap with real PSD in mu/beta band (8–30 Hz)
- [ ] Verify each `diffusion_holdout_run{R}.pt` loads in a fresh Python session and restores the correct `held_out_run`
- [ ] Verify `EEGDiffusionModel.generate(60, subject_id=1)` returns shape (60, 64, 801), dtype float32

---

## Error Handling

- **OOM on GPU**: reduce `batch_size` (default 32 → try 16 → 8); document in training output
- **`processed.npz` not found**: `sys.exit()` with explicit message and rebuild command
- **Training divergence** (loss → NaN): stop immediately, log to MLflow, print last valid step; caused by LR too high or missing gradient clipping. Fix: verify grad_clip=1.0 is active, reduce LR
- **DDIM degenerate output** (uniform/zero signal): most likely guidance_scale too high. Reduce from 3.0 → 1.5
- **Resume from wrong config**: checkpoint config vs. CLI config mismatch → log warning, use checkpoint config (checkpoint is authoritative for architecture parameters)
- **Missing MLflow server**: use local file-based tracking (`mlruns/`) which is already the default; no special handling needed

---

## Testing Strategy

Manual verification (research code — no automated test suite required):

**Architecture sanity (before training):**
```python
from eval.diffusion import UNet1D, DiffusionConfig
config = DiffusionConfig()
model = UNet1D(config)
x = torch.randn(2, 64, 801)
t = torch.randint(0, 1000, (2,))
c = torch.tensor([0, 1])
out = model(x, t, c)
assert out.shape == (2, 64, 801)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")  # expect 5–15M
```

**Forward diffusion correctness:**
```python
from eval.diffusion import DDPM
ddpm = DDPM(model, config, device=torch.device("cpu"))
x0 = torch.randn(4, 64, 801)
# At t=0: x_t ≈ x0 (very little noise)
t_early = torch.zeros(4, dtype=torch.long)
xt_early = ddpm.q_sample(x0, t_early)
assert torch.allclose(xt_early, x0, atol=0.1)
# At t=999: x_t ≈ pure noise
t_late = torch.full((4,), 999, dtype=torch.long)
xt_late = ddpm.q_sample(x0, t_late)
assert xt_late.std().item() > 0.8  # should be close to N(0,1)
```

**Generation interface:**
```python
from eval.diffusion import EEGDiffusionModel
from pathlib import Path
model = EEGDiffusionModel.load(Path("models/checkpoints/diffusion_holdout_run12.pt"))
assert model.held_out_run == 12
X = model.generate(n=60, subject_id=1, seed=42)
y = model.labels(n=60)
assert X.shape == (60, 64, 801)
assert X.dtype == np.float32
assert y.shape == (60,)
assert np.bincount(y).tolist() == [30, 30]  # balanced
# Same seed → same output
X2 = model.generate(n=60, subject_id=1, seed=42)
assert np.allclose(X, X2)
```

**PSD sanity:**
```python
from scipy.signal import welch
X_real, _ = np.load("data/processed.npz").values()
X_synth = model.generate(n=200, subject_id=0)
# PSD of synthetic should peak in mu/beta band (8–30 Hz at 160 Hz sampling rate)
_, psd_real = welch(X_real[:200, :, :].mean(axis=(0,1)), fs=160, nperseg=256)
_, psd_synth = welch(X_synth.mean(axis=(0,1)), fs=160, nperseg=256)
# Not a hard assertion — visual check in diffusion_fidelity.png
```

---

## Success Criteria

**Implementation complete when:**
- `eval/diffusion.py` exists with `UNet1D`, `DDPM`, `EEGDiffusionModel` passing all sanity checks above
- `eval/train_diffusion.py` runs smoke test to completion with `--holdout-run` (50 steps, no errors)
- Checkpoint loads cleanly in a fresh Python session and restores `held_out_run`
- `generate(n=60, subject_id=1)` returns (60, 64, 801) float32

**Training complete when:**
- All **three** run-holdout models trained (~100K steps each); three checkpoints saved, each embedding its `held_out_run`
- `train_loss` in MLflow has converged for each (stable below 0.1 for at least 10K steps)
- `diffusion_fidelity_run{R}.png` shows synthetic PSD overlapping real PSD in 8–30 Hz band, and the three models are comparable

**TSTR-ready when:**
- No leakage: each model's training set excludes its `held_out_run` (no epoch of that run, from any subject)
- No memorization: final `nn_distance ≥ nn_distance_real` for each model
- PSD divergence between real and synthetic (per class) is lower than the within-class PSD divergence between subjects (i.e., synthetic is indistinguishable from real at the population level)
- The three checkpoints load as `{4: D_run4, 8: D_run8, 12: D_run12}`, each reporting its `held_out_run`
- EEGDiffusionModel passes the generation interface checks above
- Phase 5 can `from eval.diffusion import EEGDiffusionModel`, load the three models, and route folds by test run in `_load_synthetic_set()`

**Go-ahead to Phase 5 when all of the above are satisfied.**

---

## Compute & Time Estimates

| Hardware | Estimated wall time (100K steps, batch=32) |
|---|---|
| NVIDIA A100 (80GB) | ~12–20 minutes |
| NVIDIA RTX 4090 | ~22–35 minutes |
| Apple M3 Max (MPS) | ~2.5–4.5 hours |
| Apple M1/M2 (MPS) | ~5–8 hours |
| CPU only | 1.5–2.5 days (impractical) |

Times are **per model**. Three models (one per held-out run) → multiply by 3: on an A100 the full set is ~40–60 minutes end-to-end; on M3 Max ~8–14 hours (run overnight or offload to Modal).

Recommendations:
- **Development**: run 1K–5K steps locally on MPS with `--holdout-run 12` to verify loss curve shape
- **Full training**: use Modal (A10G on-demand, ~$0.50–$1.50 per model, ~$1.50–$4.50 for all three) or Colab A100

For Modal deployment, the training script is CPU/GPU agnostic (uses `get_device()` already in `train_baseline.py`). The script needs only `requirements.txt` + access to `data/processed.npz`.

---

## Context for Handoff

### Repository state at time of writing

```
eval/
  eegnet.py                  ← EEGNet classifier (unchanged)
  augmentations.py           ← augmentation transforms (unchanged)
  dataset.py                 ← EEGTorchDataset, LORO split (unchanged)
  train_baseline.py          ← LORO training loop, all baselines (unchanged)
  compare_baselines.py       ← accuracy-vs-N comparison plots (unchanged)
  diffusion.py               ← DOES NOT EXIST (Phase 4 creates this)
  train_diffusion.py         ← DOES NOT EXIST (Phase 4 creates this)
data/
  processed.npz                           ← full 4748-epoch dataset (unchanged; has run_ids ∈ {4,8,12})
models/checkpoints/
  per_subject_results_vanilla.json        ← 56.7% ± 10.8% (all 106 subjects) ← direct comparator
  per_subject_results_augmented_1x.json   ← 58.6% ± 9.1%  (all 106 subjects)
  per_subject_results_augmented_2x.json   ← 59.5% ± 11.2% (all 106) ← primary gate (direct)
  per_subject_results_augmented_3x.json   ← 59.0% ± 10.0% (all 106 subjects)
  diffusion_holdout_run4.pt               ← DOES NOT EXIST (Phase 4 trains: excludes run 4)
  diffusion_holdout_run8.pt               ← DOES NOT EXIST (Phase 4 trains: excludes run 8)
  diffusion_holdout_run12.pt              ← DOES NOT EXIST (Phase 4 trains: excludes run 12)
```

### Key constants to hard-code correctly

```python
SAMPLING_RATE = 160    # Hz — needed for PSD x-axis
N_CHANNELS    = 64
N_TIMES       = 801    # 5.006 seconds at 160 Hz
N_CLASSES     = 2      # 0=left fist, 1=right fist
N_SUBJECTS    = 106    # full cohort; every model trains on all 106 but excludes one run
N_EPOCHS      = 4748   # full dataset; each run-holdout model trains on ~3160 (two of three runs)
RUNS          = (4, 8, 12)   # motor-imagery runs; one diffusion model per held-out run
```

### Integration point for Phase 5 (NOT implemented here)

The only change Phase 5 makes to `eval/train_baseline.py` is:

```python
from eval.diffusion import EEGDiffusionModel

# Phase 5 loads the three run-holdout models once, keyed by held-out run:
#   diffusion_models = {
#       r: EEGDiffusionModel.load(ckpt_dir / f"diffusion_holdout_run{r}.pt")
#       for r in (4, 8, 12)
#   }

def _load_synthetic_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    subject_id: int,
    test_run: int,
    n_synthetic: int,
    diffusion_models: dict[int, EEGDiffusionModel],
) -> tuple[np.ndarray, np.ndarray]:
    model = diffusion_models[test_run]
    assert model.held_out_run == test_run, "routing bug: model saw this fold's test run"
    X_synth = model.generate(n=n_synthetic, subject_id=subject_id)
    y_synth = model.labels(n=n_synthetic)
    return np.concatenate([X_real, X_synth]), np.concatenate([y_real, y_synth])
```

This function is a drop-in replacement for `_create_offline_augmented_set()` and is out of scope for this spec. It is documented here only to show the minimal surface area Phase 4 must satisfy. **The one addition the cross-fitting imposes:** Phase 5 routes each fold to the model whose `held_out_run` matches the fold's `test_run` — guarded by the `assert` above so a wiring mistake can't silently reintroduce leakage.

### TSTR comparison baseline (the target to beat)

Phase 5 runs TSTR at N = {30, 60, 90} extra synthetic samples and compares — on **all 106 subjects, per subject** — against:
- Vanilla baseline: **56.7% ± 10.8%** (N=0)
- Augmented offline 2×: **59.5% ± 11.2%** (N=60) ← primary gate

Because every subject is evaluated leakage-free (via run cross-fitting — each fold uses the model that never saw its test run), the comparison is a **direct, full-cohort, per-subject paired test** against these existing numbers. No sub-selection, no re-training, full statistical power.

The go/no-go gate (RR-3) passes if synthetic-augmented training at N=60 exceeds **59.5%** by a margin that is statistically significant (one-sided Wilcoxon signed-rank across the 106 subjects, p < 0.05) and practically meaningful (>+1% absolute improvement across subjects).
