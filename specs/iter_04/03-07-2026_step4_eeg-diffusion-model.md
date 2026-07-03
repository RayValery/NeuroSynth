# Phase 0 Step 4 — EEG Diffusion Model: 1D U-Net DDPM for Class-Conditional Generation

**Status:** Draft  
**Created:** 2026-07-03  
**Author:** Claude  
**Depends on:** Phase 0 Step 3b (additive augmented baseline, 59.5% at 2×) — complete  
**Feeds into:** Phase 0 Step 5 (TSTR harness + go/no-go gate) — separate spec  

---

## Executive Summary

Phase 0 Step 4 trains a population-level denoising diffusion probabilistic model (DDPM) on the full EEGBCI dataset to generate class-conditional synthetic EEG epochs. The model learns the joint distribution of left-fist vs. right-fist motor imagery signals across all 106 subjects, then exposes a `generate(n, subject_id)` interface that Phase 0 Step 5 drops into the existing LORO evaluation loop as a direct replacement for the additive augmentation function. No architectural changes to EEGNet or the LORO evaluation protocol are required.

---

## Goals

- Train a DDPM with a 1D U-Net backbone on `data/processed.npz` (4748 epochs, 64 channels, 801 time points)
- Implement classifier-free guidance (CFG) for binary class conditioning (left fist = 0, right fist = 1)
- Produce a `EEGDiffusionModel` wrapper with a fixed interface that Phase 5 can call without modification
- Log training to MLflow (`phase0-diffusion` experiment) with loss curves and fidelity snapshots
- Save a checkpoint to `models/checkpoints/diffusion_model.pt` with model weights + config
- Verify generation quality with PSD comparison and basic distributional checks before Phase 5

## Non-Goals

- Subject-conditional generation (Phase 1 — subject identity conditioning added in the next research phase)
- TSTR evaluation (Phase 0 Step 5 — this spec only trains and validates the generative model)
- Format export (EDF+, BIDS) — Phase 1 product infrastructure
- Hyperparameter search / NAS — single architecture, one training run
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

The research question — does synthetic data beat augmentation for motor imagery classification? — can be answered at the population level. A population model generates plausible left/right fist signals drawn from the full distribution of all 106 subjects. Subject-to-subject variability within a class is captured by the model's learned variance. This is the simplest testable hypothesis.

Subject-conditional generation (conditioning on subject ID) would require the model to learn 106 different per-subject distributions, which is harder with 4748 total training samples (~45 per subject) and raises overfitting risk. That is Phase 1 scope.

The `generate(subject_id=...)` parameter is accepted at the interface level but ignored in Phase 0 (population model), ensuring Phase 5 and Phase 1 code is identical.

### Relationship to TSTR (Phase 5)

The V08 spec established the exact integration point. Phase 5 replaces one function call:

```python
# Current (augmented_offline in train_baseline.py):
X_combined, y_combined = _create_offline_augmented_set(
    X_train, y_train, n_copies=N, augmentor=STANDARD_AUGMENTOR
)

# Phase 5 replacement:
X_combined, y_combined = _load_synthetic_set(
    X_train, y_train,
    subject_id=subject_id,
    n_synthetic=N,
    diffusion_model=model,
)
```

The LORO training loop, evaluation, MLflow logging, `compare_baselines.py`, and reporting are all unchanged. Phase 4's only obligation is to deliver a model object that satisfies the `EEGDiffusionModel` interface below.

---

## Requirements

### Must Have

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

- EMA (exponential moving average) of model weights, decay=0.9999: standard for DDPM quality
- Gradient clipping (max_norm=1.0): prevents instability in early training
- `--resume` flag on training script to continue from a checkpoint
- PSD divergence scalar logged to MLflow as a fidelity proxy metric
- Per-channel mean/std comparison plot saved to `reports/figures/diffusion_fidelity.png`

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
        │
        ▼
┌────────────────────────┐
│  eval/train_diffusion  │  ← training script (new)
│  DDPM.train_step()     │
│  EMA weight averaging  │
│  MLflow logging        │
└────────────┬───────────┘
             │ saves
             ▼
models/checkpoints/diffusion_model.pt
             │
             │ loaded by
             ▼
┌────────────────────────┐
│  EEGDiffusionModel     │  ← public interface (new)
│  .generate(n, subj_id) │
│  .labels(n)            │
└────────────┬───────────┘
             │ called by
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

    # Training
    lr: float = 1e-4
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    batch_size: int = 32
    n_steps: int = 200_000

    # Inference
    ddim_steps: int = 50
    guidance_scale: float = 3.0
```

### Checkpoint format

```python
# Saved with torch.save() to models/checkpoints/diffusion_model.pt
{
    "model_state_dict":     OrderedDict,    # UNet1D weights
    "ema_state_dict":       OrderedDict,    # EMA shadow weights (used for inference)
    "config":               dict,           # DiffusionConfig.__dict__
    "step":                 int,            # training step at save
    "train_loss":           float,          # running mean loss at save
    "mlflow_run_id":        str,            # MLflow run for traceability
}
```

Intermediate checkpoints: `models/checkpoints/diffusion_step_{N}.pt`

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
    "n_train_samples": 4748,
    "batch_size": 32,
    "lr": 1e-4,
    "ema_decay": 0.9999,
    "p_uncond": 0.1,
    "guidance_scale": 3.0,
    "ddim_steps": 50,
}
# Metrics (logged every 500 steps):
# "train_loss", "psd_divergence"
```

---

## Interface Definitions

### `EEGDiffusionModel` — public interface consumed by Phase 5

```python
class EEGDiffusionModel:
    """
    Population-level class-conditional EEG generator.
    Phase 0: subject_id is accepted but ignored (population model).
    """

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
    python eval/train_diffusion.py --data data/processed.npz
    python eval/train_diffusion.py --data data/processed.npz --n-steps 200000
    python eval/train_diffusion.py --data data/processed.npz --resume models/checkpoints/diffusion_step_100000.pt
    python eval/train_diffusion.py --data data/processed.npz --batch-size 64 --lr 2e-4

Arguments:
    --data PATH               Path to processed.npz (default: data/processed.npz)
    --checkpoint-dir PATH     Output directory (default: models/checkpoints)
    --n-steps INT             Training steps (default: 200000)
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
  - Wraps `DDPM` + `UNet1D`
  - `generate(n, subject_id, class_label=None, seed=None)`: calls `DDPM.sample` for each class, interleaves results
  - `labels(n)`: returns np.array([0,1,0,1,...], dtype=int64) of length n
  - `save(path)`: saves EMA state dict + config
  - `load(path, device)`: reconstructs from checkpoint

### Phase 2 — Training script: `eval/train_diffusion.py`

**File: `eval/train_diffusion.py`** (new file, ~200 lines)

- [ ] Load `data/processed.npz`, extract X (float32) and y (int64)
- [ ] Verify expected shape: (4748, 64, 801); warn and continue if shape differs
- [ ] Build PyTorch `TensorDataset(X_tensor, y_tensor)` and `DataLoader(batch_size, shuffle=True, num_workers=0)`
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
              checkpoint_path = checkpoint_dir / f"diffusion_step_{step}.pt"
              model_wrapper.save(checkpoint_path)
              _rotate_checkpoints(checkpoint_dir, keep=3)
          if step >= n_steps:
              break
  ```
- [ ] Resume logic: if `--resume` provided, load checkpoint and set step counter
- [ ] After training completes:
  - Generate 50 samples per class (100 total) using EMA model
  - Compute and log `psd_divergence` scalar to MLflow
  - Save `reports/figures/diffusion_fidelity.png` (PSD comparison + channel mean/std plots)
  - Save final checkpoint to `models/checkpoints/diffusion_model.pt`
- [ ] Implement `_plot_fidelity(X_real, X_synth, y_real, out_path)`:
  - 2 panels: PSD (real vs. synthetic per class), per-channel mean/std scatter
  - Mark class 0 and class 1 separately in both panels

### Phase 3 — Train and verify

- [ ] Smoke test on CPU (50 steps, batch_size=4): confirm loss decreases, shapes correct
  ```bash
  python eval/train_diffusion.py --data data/processed.npz --n-steps 50 --batch-size 4
  ```
- [ ] Full training run on GPU (Modal / RunPod / local MPS):
  ```bash
  python eval/train_diffusion.py --data data/processed.npz --n-steps 200000 --batch-size 32
  ```
- [ ] Verify `diffusion_fidelity.png`: generated PSD should overlap with real PSD in mu/beta band (8–30 Hz)
- [ ] Verify `diffusion_model.pt` loads correctly in a fresh Python session
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
model = EEGDiffusionModel.load(Path("models/checkpoints/diffusion_model.pt"))
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
- `eval/train_diffusion.py` runs smoke test to completion (50 steps, no errors)
- Checkpoint loads cleanly in a fresh Python session
- `generate(n=60, subject_id=1)` returns (60, 64, 801) float32

**Training complete when:**
- Full training run finishes (200K steps) and final checkpoint saved
- `train_loss` in MLflow has converged (stable below 0.1 for at least 10K steps)
- `diffusion_fidelity.png` shows synthetic PSD overlapping real PSD in 8–30 Hz band

**TSTR-ready when:**
- PSD divergence between real and synthetic (per class) is lower than the within-class PSD divergence between subjects (i.e., synthetic is indistinguishable from real at the population level)
- EEGDiffusionModel passes the generation interface checks above
- Phase 5 can `from eval.diffusion import EEGDiffusionModel` and immediately write `_load_synthetic_set()`

**Go-ahead to Phase 5 when all of the above are satisfied.**

---

## Compute & Time Estimates

| Hardware | Estimated wall time (200K steps, batch=32) |
|---|---|
| NVIDIA A100 (80GB) | ~25–40 minutes |
| NVIDIA RTX 4090 | ~45–70 minutes |
| Apple M3 Max (MPS) | ~5–9 hours |
| Apple M1/M2 (MPS) | ~10–16 hours |
| CPU only | 3–5 days (impractical) |

Recommendations:
- **Development**: run 1K–5K steps locally on MPS to verify loss curve shape
- **Full training**: use Modal (A10G on-demand, ~$0.50–$1.50 per run) or Colab A100

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
models/checkpoints/
  per_subject_results_vanilla.json        ← 56.7% ± 10.8%
  per_subject_results_augmented_1x.json   ← 58.6% ± 9.1%
  per_subject_results_augmented_2x.json   ← 59.5% ± 11.2% ← TSTR target
  per_subject_results_augmented_3x.json   ← 59.0% ± 10.0%
  diffusion_model.pt                      ← DOES NOT EXIST (Phase 4 trains this)
```

### Key constants to hard-code correctly

```python
SAMPLING_RATE = 160    # Hz — needed for PSD x-axis
N_CHANNELS    = 64
N_TIMES       = 801    # 5.006 seconds at 160 Hz
N_CLASSES     = 2      # 0=left fist, 1=right fist
N_SUBJECTS    = 106
N_EPOCHS      = 4748   # total training samples for diffusion model
```

### Integration point for Phase 5 (NOT implemented here)

The only change Phase 5 makes to `eval/train_baseline.py` is:

```python
from eval.diffusion import EEGDiffusionModel

def _load_synthetic_set(
    X_real: np.ndarray,
    y_real: np.ndarray,
    subject_id: int,
    n_synthetic: int,
    diffusion_model: EEGDiffusionModel,
) -> tuple[np.ndarray, np.ndarray]:
    X_synth = diffusion_model.generate(n=n_synthetic, subject_id=subject_id)
    y_synth = diffusion_model.labels(n=n_synthetic)
    return np.concatenate([X_real, X_synth]), np.concatenate([y_real, y_synth])
```

This function is a drop-in replacement for `_create_offline_augmented_set()` and is out of scope for this spec. It is documented here only to show the minimal surface area Phase 4 must satisfy.

### TSTR comparison baseline (the target to beat)

Phase 5 must run TSTR at N = {30, 60, 90} extra synthetic samples and compare against:
- Vanilla baseline: **56.7% ± 10.8%** (N=0)
- Augmented offline 2×: **59.5% ± 11.2%** (N=60) ← primary gate

The go/no-go gate (RR-3) passes if synthetic-augmented training at N=60 exceeds 59.5% by a margin that is statistically significant (one-sided Wilcoxon p < 0.05) and practically meaningful (>+1% absolute improvement across subjects).
