"""
DDPM 1D U-Net for class-conditional EEG generation (Phase 0 Step 4).

Population-level denoising diffusion model that generates class-conditional
(left/right fist) motor-imagery epochs matching the z-scored space of
``data/processed.npz`` (64 channels, 801 time points).

Public surface consumed by Phase 5:
    EEGDiffusionModel  — .generate(n, subject_id), .labels(n), .save/.load,
                         and the .held_out_run attribute used to route each
                         LORO fold to the model that never saw its test run.

Internals:
    DiffusionConfig    — hyperparameters (see spec §Data Models)
    UNet1D             — 1D U-Net noise-prediction backbone (time + class cond)
    DDPM               — cosine forward diffusion, EMA, DDIM sampler w/ CFG

The model is subject-agnostic: ``generate(subject_id=...)`` accepts the argument
for Phase 1 compatibility but ignores it. Leakage prevention is handled by
training one model per held-out run (see ``train_diffusion.py``); this module
only stores/exposes ``held_out_run`` so Phase 5 can route folds correctly.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Motor-imagery runs in the EEGBCI left/right-fist task; one model per held-out run.
MOTOR_IMAGERY_RUNS = (4, 8, 12)

# Domain labels for pivot_iter_01's execution-conditioned generator. When
# DiffusionConfig.use_domain_cond is True the model is trained jointly on imagery
# and real motor-execution epochs, tagged by this domain, and sampled in the
# imagery domain by default (see specs/pivot_iter_01).
DOMAIN_IMAGERY = 0
DOMAIN_EXECUTION = 1


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU (mirrors eval/train_baseline.py)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DiffusionConfig:
    # Signal
    n_channels: int = 64
    n_times: int = 801
    n_pad: int = 1024          # internal working length (next power of 2 above 801)
    n_classes: int = 2

    # U-Net
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 4)   # -> 128, 256, 512
    n_res_blocks: int = 2
    attn_resolution: int = 128         # self-attention at the bottleneck (T=128) only

    # Diffusion
    T: int = 1000
    schedule: str = "cosine"
    p_uncond: float = 0.1              # CFG dropout probability during training

    # Conditioning
    cond_dim: int = 512                # time + class embedding dimension

    # Cross-fitting (leakage prevention — see spec §Background)
    holdout_run: int = 12              # motor-imagery run excluded from THIS model

    # Training
    lr: float = 1e-4
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    batch_size: int = 32
    n_steps: int = 100_000

    # Inference
    ddim_steps: int = 50
    guidance_scale: float = 3.0
    clip_denoised: float = 4.0          # clamp predicted x0 each DDIM step; without it the
                                        # 1/sqrt(alpha_bar_t) factor at high t blows samples up

    # Domain conditioning (pivot_iter_01). When enabled, an extra domain token
    # (imagery=0 / execution=1) is added to the conditioning so one generator can be
    # trained on imagery+execution jointly and sampled in a chosen domain. Default
    # False → architecture and behaviour identical to Step 4/5 (backward compatible).
    use_domain_cond: bool = False
    n_domains: int = 2
    sample_domain: int = 0              # default generation domain: 0=imagery, 1=execution

    @property
    def null_token(self) -> int:
        """Class-embedding index used for the unconditional (dropped-label) pass."""
        return self.n_classes


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides ``channels``."""
    return nn.GroupNorm(num_groups=math.gcd(channels, max_groups), num_channels=channels)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep features -> 2-layer MLP -> conditioning vector."""

    def __init__(self, cond_dim: int):
        super().__init__()
        self.sin_dim = cond_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(self.sin_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: Tensor) -> Tensor:  # t: (B,) long -> (B, cond_dim)
        half = self.sin_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, sin_dim)
        return self.mlp(emb)


class ResBlock1D(nn.Module):
    """
    GroupNorm -> SiLU -> Conv1d, inject conditioning, GroupNorm -> SiLU -> Conv1d,
    plus a residual connection (1x1 conv when the channel count changes).
    """

    def __init__(self, c_in: int, c_out: int, cond_dim: int):
        super().__init__()
        self.norm1 = _group_norm(c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, c_out)
        self.norm2 = _group_norm(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None]   # broadcast over the time axis
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Multi-head self-attention over the time axis (bottleneck only)."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.norm = _group_norm(channels)
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, T)
        h = self.norm(x).transpose(1, 2)     # (B, T, C) — T tokens, each dim C
        h, _ = self.mha(h, h, h)
        return x + h.transpose(1, 2)


class UNet1D(nn.Module):
    """1D U-Net noise-prediction backbone with sinusoidal time + class conditioning."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        chs = [config.base_channels * m for m in config.channel_mults]  # [128, 256, 512]
        n_levels = len(chs)
        cond_dim = config.cond_dim

        # Conditioning: cond = time_emb + class_emb(c) [+ domain_emb(d)]; null token = n_classes.
        self.time_emb = SinusoidalTimeEmbedding(cond_dim)
        self.class_emb = nn.Embedding(config.n_classes + 1, cond_dim)
        # Optional domain token (imagery/execution). Absent unless use_domain_cond,
        # so Step 4/5 imagery-only checkpoints load with an identical parameter set.
        self.domain_emb = (
            nn.Embedding(config.n_domains, cond_dim) if config.use_domain_cond else None
        )

        # Encoder
        self.stem = nn.Conv1d(config.n_channels, chs[0], kernel_size=7, padding=3)
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_levels):
            blocks = nn.ModuleList(
                ResBlock1D(chs[i], chs[i], cond_dim) for _ in range(config.n_res_blocks)
            )
            self.enc_blocks.append(blocks)
            c_next = chs[min(i + 1, n_levels - 1)]
            self.downsamples.append(nn.Conv1d(chs[i], c_next, kernel_size=4, stride=2, padding=1))

        # Bottleneck
        self.mid_block1 = ResBlock1D(chs[-1], chs[-1], cond_dim)
        self.mid_attn = SelfAttention1D(chs[-1], heads=8)
        self.mid_block2 = ResBlock1D(chs[-1], chs[-1], cond_dim)

        # Decoder (mirrored). Lists are stored top-down: index 0 == deepest level.
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(n_levels)):
            c_in = chs[min(i + 1, n_levels - 1)]           # channels flowing in from below
            self.up_convs.append(nn.Conv1d(c_in, chs[i], kernel_size=3, padding=1))
            blocks = nn.ModuleList()
            for j in range(config.n_res_blocks):
                bc_in = 2 * chs[i] if j == 0 else chs[i]   # first block absorbs the skip concat
                blocks.append(ResBlock1D(bc_in, chs[i], cond_dim))
            self.dec_blocks.append(blocks)

        self.head = nn.Sequential(
            _group_norm(chs[0]),
            nn.SiLU(),
            nn.Conv1d(chs[0], config.n_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, x: Tensor, t: Tensor, class_label: Tensor, domain: Tensor | None = None
    ) -> Tensor:
        # x: (B, 64, 801) -> pad to (B, 64, 1024) -> predict noise -> crop to 801.
        pad = self.config.n_pad - self.config.n_times
        h = F.pad(x, (0, pad))
        cond = self.time_emb(t) + self.class_emb(class_label)
        if self.domain_emb is not None:
            if domain is None:
                raise ValueError("use_domain_cond=True but no domain passed to UNet1D.forward()")
            cond = cond + self.domain_emb(domain)

        h = self.stem(h)
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.enc_blocks):
            for block in blocks:
                h = block(h, cond)
            skips.append(h)
            h = self.downsamples[level](h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for idx, up_conv in enumerate(self.up_convs):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = up_conv(h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in self.dec_blocks[idx]:
                h = block(h, cond)

        h = self.head(h)
        return h[:, :, : self.config.n_times]


# --------------------------------------------------------------------------- #
# Diffusion process
# --------------------------------------------------------------------------- #
class DDPM:
    """Cosine forward diffusion, EMA weight tracking, and a CFG-guided DDIM sampler."""

    def __init__(self, model: UNet1D, config: DiffusionConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

        betas = self._cosine_beta_schedule(config.T).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # Registered as plain tensors (not nn buffers) — DDPM is not an nn.Module.
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        # EMA shadow model used for all sampling.
        self.ema_model = copy.deepcopy(model).to(device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008) -> Tensor:
        """Nichol & Dhariwal (2021) cosine schedule."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / T + s) / (1 + s)) * (math.pi / 2)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(0, 0.999).float()

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor | None = None) -> Tensor:
        """Forward diffusion: x_t = sqrt(a_bar_t) x0 + sqrt(1 - a_bar_t) eps."""
        if eps is None:
            eps = torch.randn_like(x0)
        sab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        somab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sab * x0 + somab * eps

    def train_step(self, x0: Tensor, y: Tensor, domain: Tensor | None = None) -> Tensor:
        """One noise-prediction step with CFG label dropout. Returns the loss tensor.

        ``domain`` (imagery/execution) is a fixed context — it is NOT dropped for CFG,
        so guidance sharpens only the class. Ignored when use_domain_cond=False.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.config.T, (b,), device=self.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)

        # Classifier-free guidance: drop the class label with prob p_uncond.
        c = y.clone()
        drop = torch.rand(b, device=self.device) < self.config.p_uncond
        c[drop] = self.config.null_token

        eps_pred = self.model(x_t, t, c, domain)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def _update_ema(self) -> None:
        d = self.config.ema_decay
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        class_label: int,
        ddim_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int | None = None,
        domain: int | None = None,
    ) -> Tensor:
        """Deterministic (eta=0) DDIM sampling with classifier-free guidance.

        ``domain`` selects the generation domain when use_domain_cond=True
        (defaults to config.sample_domain); ignored otherwise.
        """
        if seed is not None:
            torch.manual_seed(seed)
        model = self.ema_model
        model.eval()

        c, t_len = self.config.n_channels, self.config.n_times
        x = torch.randn(n, c, t_len, device=self.device)

        ts = torch.linspace(0, self.config.T - 1, ddim_steps).round().long().tolist()
        null = torch.full((n,), self.config.null_token, device=self.device, dtype=torch.long)
        cond = torch.full((n,), int(class_label), device=self.device, dtype=torch.long)

        dom_batch = None
        if self.config.use_domain_cond:
            d = self.config.sample_domain if domain is None else int(domain)
            dom_batch = torch.full((n,), d, device=self.device, dtype=torch.long)

        for i in reversed(range(len(ts))):
            t = ts[i]
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps_uncond = model(x, t_batch, null, dom_batch)
            eps_cond = model(x, t_batch, cond, dom_batch)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i - 1]] if i > 0 else torch.ones((), device=self.device)
            x0_pred = (x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
            # Clamp the predicted clean signal — essential for numerical stability, since
            # 1/sqrt(alpha_bar_t) is ~2e4 at t≈T and amplifies any eps error without this.
            clip = self.config.clip_denoised
            if clip and clip > 0:
                x0_pred = x0_pred.clamp(-clip, clip)
            x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps

        return x


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def _zscore_epochs(x: np.ndarray) -> np.ndarray:
    """
    Per-epoch, per-channel z-score over the time axis — identical to the real
    pipeline's ``zscore_normalize`` (data/eegbci_pipeline.py), so synthetic epochs
    land in the same z-scored space as data/processed.npz.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32)


class EEGDiffusionModel:
    """
    Population-level class-conditional EEG generator (Phase 0).

    ``subject_id`` is accepted but ignored (population model). Trained on all
    subjects but excluding one motor-imagery run; leakage-free for LORO folds
    whose test run == ``held_out_run``.
    """

    def __init__(self, config: DiffusionConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or get_device()
        self.model = UNet1D(config).to(self.device)
        self.ddpm = DDPM(self.model, config, self.device)

        self.held_out_run: int = config.holdout_run
        self.train_runs: list[int] = [r for r in MOTOR_IMAGERY_RUNS if r != config.holdout_run]

        # Metadata populated by the training script before each save.
        self.step: int = 0
        self.train_loss: float = float("nan")
        self.nn_distance: float = float("nan")
        self.mlflow_run_id: str = ""

    def generate(
        self,
        n: int,
        subject_id: int,
        class_label: int | None = None,
        seed: int | None = None,
        domain: int | None = None,
    ) -> np.ndarray:
        """
        Generate ``n`` synthetic epochs via DDIM sampling. ``subject_id`` is ignored
        (population model). ``class_label=None`` -> balanced, interleaved [0,1,0,1,...].
        ``domain`` selects imagery (0) / execution (1) when the model is domain-conditioned
        (defaults to config.sample_domain); ignored for imagery-only Step 4/5 models.
        Returns float32 array of shape (n, 64, 801) in the z-scored data space.
        """
        _ = subject_id  # population model — accepted for Phase 1 compatibility

        if class_label is not None:
            x = self.ddpm.sample(
                n, class_label, self.config.ddim_steps, self.config.guidance_scale, seed,
                domain=domain,
            )
            return _zscore_epochs(x.detach().cpu().numpy().astype(np.float32))

        n0 = (n + 1) // 2   # ceil -> class 0
        n1 = n // 2         # floor -> class 1
        x0 = self.ddpm.sample(n0, 0, self.config.ddim_steps, self.config.guidance_scale, seed,
                              domain=domain)
        seed1 = None if seed is None else seed + 1
        x1 = self.ddpm.sample(n1, 1, self.config.ddim_steps, self.config.guidance_scale, seed1,
                              domain=domain)

        out = np.empty((n, self.config.n_channels, self.config.n_times), dtype=np.float32)
        out[0::2] = x0.detach().cpu().numpy().astype(np.float32)
        out[1::2] = x1.detach().cpu().numpy().astype(np.float32)
        return _zscore_epochs(out)

    def labels(self, n: int) -> np.ndarray:
        """Labels matching ``generate(n, ..., class_label=None)``: [0,1,0,1,...]."""
        return (np.arange(n) % 2).astype(np.int64)

    def save(self, checkpoint_path: Path) -> None:
        """Save model + EMA weights, config, and cross-fitting metadata."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ddpm.ema_model.state_dict(),
                "config": asdict(self.config),
                "held_out_run": self.held_out_run,
                "train_runs": list(self.train_runs),
                "step": self.step,
                "train_loss": self.train_loss,
                "nn_distance": self.nn_distance,
                "mlflow_run_id": self.mlflow_run_id,
            },
            checkpoint_path,
        )

    @classmethod
    def load(
        cls, checkpoint_path: Path, device: torch.device | None = None
    ) -> "EEGDiffusionModel":
        """Load a trained model from checkpoint (checkpoint config is authoritative)."""
        device = device or get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        cfg_dict = dict(ckpt["config"])
        cfg_dict["channel_mults"] = tuple(cfg_dict["channel_mults"])
        config = DiffusionConfig(**cfg_dict)

        model = cls(config, device)
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.ddpm.ema_model.load_state_dict(ckpt["ema_state_dict"])
        model.held_out_run = int(ckpt["held_out_run"])
        model.train_runs = list(
            ckpt.get("train_runs", [r for r in MOTOR_IMAGERY_RUNS if r != model.held_out_run])
        )
        model.step = int(ckpt.get("step", 0))
        model.train_loss = float(ckpt.get("train_loss", float("nan")))
        model.nn_distance = float(ckpt.get("nn_distance", float("nan")))
        model.mlflow_run_id = str(ckpt.get("mlflow_run_id", ""))
        model.model.eval()
        model.ddpm.ema_model.eval()
        return model
