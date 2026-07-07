"""
Train a population DDPM (1D U-Net) for class-conditional EEG generation.

Leave-one-run-out cross-fitting (leakage prevention): run this once per
motor-imagery run. Each model trains on all 106 subjects but excludes ONE run
(4, 8, or 12); Phase 5 then routes each LORO fold to the model that never saw
that fold's test run.

Usage:
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 4
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 8
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 --n-steps 100000
    python eval/train_diffusion.py --data data/processed.npz --holdout-run 12 \
        --resume models/checkpoints/diffusion_run12_step_50000.pt

Produces (per held-out run R):
    models/checkpoints/diffusion_holdout_run{R}.pt   final model (EMA weights)
    models/checkpoints/diffusion_run{R}_step_{N}.pt  intermediate checkpoints
    reports/figures/diffusion_fidelity_run{R}.png    PSD + channel mean/std plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from torch.utils.data import DataLoader, TensorDataset

from data.eegbci_pipeline import EEGDataset
from eval.diffusion import (
    MOTOR_IMAGERY_RUNS,
    DiffusionConfig,
    EEGDiffusionModel,
    get_device,
)

SAMPLING_RATE = 160  # Hz — EEGBCI


# --------------------------------------------------------------------------- #
# Fidelity / memorization metrics
# --------------------------------------------------------------------------- #
def _pairwise_nn(probe: np.ndarray, db: np.ndarray, chunk: int = 512) -> float:
    """Mean L2 distance from each probe row to its nearest row in ``db`` (chunked)."""
    probe = probe.astype(np.float32)
    p2 = (probe ** 2).sum(axis=1)
    min_d2 = np.full(len(probe), np.inf, dtype=np.float64)
    for i in range(0, len(db), chunk):
        block = db[i : i + chunk].astype(np.float32)
        r2 = (block ** 2).sum(axis=1)
        cross = probe @ block.T                         # (P, b)
        d2 = p2[:, None] + r2[None, :] - 2.0 * cross    # ||p - r||^2
        min_d2 = np.minimum(min_d2, d2.min(axis=1))
    return float(np.sqrt(np.clip(min_d2, 0, None)).mean())


def _nn_distance_gen(
    model: EEGDiffusionModel, X_real_flat: np.ndarray, n_probe: int, seed: int
) -> float:
    """Mean L2 from a probe batch of generated samples to the nearest real sample."""
    X_gen = model.generate(n_probe, subject_id=0, seed=seed)
    return _pairwise_nn(X_gen.reshape(len(X_gen), -1), X_real_flat)


def _nn_distance_real(X_real_flat: np.ndarray, n_probe: int, rng: np.random.Generator) -> float:
    """Real-to-real nearest-neighbor baseline (probe held out of the database)."""
    n_probe = min(n_probe, len(X_real_flat) - 1)
    idx = rng.choice(len(X_real_flat), n_probe, replace=False)
    mask = np.ones(len(X_real_flat), dtype=bool)
    mask[idx] = False
    return _pairwise_nn(X_real_flat[idx], X_real_flat[mask])


def _mean_psd(X: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per (epoch, channel), averaged. X: (N, C, T) -> (freqs, mean_psd)."""
    nperseg = min(256, X.shape[-1])
    f, pxx = welch(X, fs=fs, nperseg=nperseg, axis=-1)  # pxx: (N, C, F)
    return f, pxx.mean(axis=(0, 1))


def _psd_divergence(X_real: np.ndarray, X_synth: np.ndarray, fs: float) -> float:
    """Mean absolute difference of log10 PSD between real and synthetic (scalar)."""
    _, pr = _mean_psd(X_real, fs)
    _, ps = _mean_psd(X_synth, fs)
    eps = 1e-12
    return float(np.mean(np.abs(np.log10(pr + eps) - np.log10(ps + eps))))


def _plot_fidelity(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    fs: float,
    holdout_run: int,
    out_path: Path,
) -> None:
    """PSD (real vs synthetic per class) and per-channel mean/std scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 — PSD per class
    ax = axes[0]
    for cls, color in ((0, "tab:blue"), (1, "tab:red")):
        f, pr = _mean_psd(X_real[y_real == cls], fs)
        _, ps = _mean_psd(X_synth[y_synth == cls], fs)
        ax.semilogy(f, pr, color=color, lw=1.8, label=f"real class {cls}")
        ax.semilogy(f, ps, color=color, lw=1.4, ls="--", label=f"synth class {cls}")
    ax.axvspan(8, 30, color="gray", alpha=0.12, label="mu/beta 8-30 Hz")
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"PSD real vs synthetic (holdout run {holdout_run})")
    ax.legend(fontsize=8)

    # Panel 2 — per-channel mean/std scatter
    ax = axes[1]
    rm, sm = X_real.mean(axis=(0, 2)), X_synth.mean(axis=(0, 2))
    rs, ss = X_real.std(axis=(0, 2)), X_synth.std(axis=(0, 2))
    ax.scatter(rm, sm, s=12, alpha=0.6, label="per-channel mean")
    ax.scatter(rs, ss, s=12, alpha=0.6, label="per-channel std")
    lo = float(min(rm.min(), sm.min(), rs.min(), ss.min()))
    hi = float(max(rm.max(), sm.max(), rs.max(), ss.max()))
    ax.plot([lo, hi], [lo, hi], color="k", lw=1, ls=":", label="y = x")
    ax.set_xlabel("real")
    ax.set_ylabel("synthetic")
    ax.set_title("Per-channel mean / std")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _rotate_checkpoints(checkpoint_dir: Path, holdout_run: int, keep: int = 3) -> None:
    """Keep only the newest ``keep`` intermediate checkpoints for this run."""
    pattern = f"diffusion_run{holdout_run}_step_*.pt"
    ckpts = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    data_path: Path,
    holdout_run: int,
    checkpoint_dir: Path,
    n_steps: int,
    batch_size: int,
    lr: float,
    guidance_scale: float,
    ddim_steps: int,
    seed: int,
    resume: Path | None,
    save_every: int,
    mlflow_experiment: str,
    n_eval_samples: int,
    nn_probe: int,
) -> dict:
    if holdout_run not in MOTOR_IMAGERY_RUNS:
        raise ValueError(f"--holdout-run must be one of {MOTOR_IMAGERY_RUNS}, got {holdout_run}")
    if not data_path.exists():
        sys.exit(
            f"Dataset not found: {data_path}\n"
            "Run `python data/eegbci_pipeline.py --subjects $(seq 1 109) "
            "--out data/processed.npz` first."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ds = EEGDataset.load(data_path)
    if ds.X.shape[1:] != (64, 801):
        print(f"WARNING: expected (*, 64, 801), got {ds.X.shape} — continuing anyway.")

    # -- Run-holdout filter: drop the held-out run, keep all subjects --
    keep = ds.run_ids != holdout_run
    X_train = np.ascontiguousarray(ds.X[keep].astype(np.float32))
    y_train = ds.y[keep].astype(np.int64)
    runs_train = ds.run_ids[keep]
    assert (runs_train == holdout_run).sum() == 0, "leakage: held-out run present in train set"

    train_runs = [r for r in MOTOR_IMAGERY_RUNS if r != holdout_run]
    n_subjects = len(np.unique(ds.subject_ids[keep]))
    device = get_device()

    print(f"Device        : {device}")
    print(f"Holdout run   : {holdout_run}  (train runs: {train_runs})")
    print(f"Subjects      : {n_subjects}  (all kept)")
    print(f"Train epochs  : {len(y_train)}  | class balance: {np.bincount(y_train).tolist()}")
    print(f"Signal        : {X_train.shape[1]} channels x {X_train.shape[2]} times")

    # -- Config / model --
    config = DiffusionConfig(
        n_channels=X_train.shape[1],
        n_times=X_train.shape[2],
        holdout_run=holdout_run,
        lr=lr,
        batch_size=batch_size,
        n_steps=n_steps,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
    )

    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["channel_mults"] = tuple(cfg_dict["channel_mults"])
        config = DiffusionConfig(**cfg_dict)   # checkpoint config is authoritative
        start_step = int(ckpt.get("step", 0))
        print(f"Resuming from : {resume}  (step {start_step})")

    model = EEGDiffusionModel(config, device)
    if resume is not None:
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.ddpm.ema_model.load_state_dict(ckpt["ema_state_dict"])

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"UNet1D params : {n_params:,}")

    optimizer = torch.optim.Adam(model.model.parameters(), lr=config.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    X_real_flat = X_train.reshape(len(X_train), -1)  # view — for nn_distance

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("reports/figures")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"ddpm_holdout_run{holdout_run}"):
        mlflow.log_params({
            "model": "ddpm_unet1d",
            "n_channels": config.n_channels,
            "n_times": config.n_times,
            "T": config.T,
            "schedule": config.schedule,
            "base_channels": config.base_channels,
            "channel_mults": ",".join(map(str, config.channel_mults)),
            "n_res_blocks": config.n_res_blocks,
            "holdout_run": holdout_run,
            "train_runs": ",".join(map(str, train_runs)),
            "n_subjects": n_subjects,
            "n_train_samples": len(y_train),
            "batch_size": config.batch_size,
            "lr": config.lr,
            "ema_decay": config.ema_decay,
            "p_uncond": config.p_uncond,
            "guidance_scale": config.guidance_scale,
            "ddim_steps": config.ddim_steps,
            "n_params": n_params,
            "seed": seed,
        })
        run_id = mlflow.active_run().info.run_id

        # -- Training loop (step-based) --
        step = start_step
        recent_losses: list[float] = []
        model.model.train()
        done = False
        while not done:
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                loss = model.ddpm.train_step(x_batch, y_batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.model.parameters(), config.grad_clip)
                optimizer.step()
                model.ddpm._update_ema()

                step += 1
                loss_val = loss.item()
                recent_losses.append(loss_val)
                if len(recent_losses) > 500:
                    recent_losses.pop(0)

                if not np.isfinite(loss_val):
                    mlflow.log_metric("train_loss", loss_val, step=step)
                    sys.exit(
                        f"Training diverged (loss={loss_val}) at step {step}. "
                        "Reduce --lr or verify grad clipping."
                    )

                if step % 500 == 0:
                    mean_loss = float(np.mean(recent_losses))
                    mlflow.log_metric("train_loss", mean_loss, step=step)
                    print(f"step {step:>7d} | train_loss={mean_loss:.4f}", flush=True)

                if step % save_every == 0:
                    model.step = step
                    model.train_loss = float(np.mean(recent_losses))
                    model.nn_distance = _nn_distance_gen(model, X_real_flat, nn_probe, seed=seed)
                    mlflow.log_metric("nn_distance", model.nn_distance, step=step)
                    model.model.train()
                    ckpt_path = checkpoint_dir / f"diffusion_run{holdout_run}_step_{step}.pt"
                    model.save(ckpt_path)
                    _rotate_checkpoints(checkpoint_dir, holdout_run, keep=3)
                    print(f"  saved {ckpt_path.name} | nn_distance={model.nn_distance:.3f}")

                if step >= n_steps:
                    done = True
                    break

        # -- Post-training fidelity + memorization checks --
        print("Generating evaluation samples for fidelity/memorization checks ...")
        n_per = max(1, n_eval_samples // 2)
        X_s0 = model.generate(n_per, subject_id=0, class_label=0, seed=seed)
        X_s1 = model.generate(n_per, subject_id=0, class_label=1, seed=seed + 1)
        X_synth = np.concatenate([X_s0, X_s1])
        y_synth = np.concatenate([np.zeros(len(X_s0), np.int64), np.ones(len(X_s1), np.int64)])

        psd_div = _psd_divergence(X_train, X_synth, SAMPLING_RATE)
        nn_gen = _nn_distance_gen(model, X_real_flat, nn_probe, seed=seed)
        nn_real = _nn_distance_real(X_real_flat, nn_probe, rng)
        mlflow.log_metrics({
            "psd_divergence": psd_div,
            "nn_distance": nn_gen,
            "nn_distance_real": nn_real,
        }, step=step)

        memorizing = nn_gen < nn_real
        print(f"psd_divergence     : {psd_div:.4f}")
        print(f"nn_distance (gen)  : {nn_gen:.3f}")
        print(f"nn_distance (real) : {nn_real:.3f}")
        print(
            "MEMORIZATION GUARD : "
            + ("FAIL — generated closer to real than real-to-real (reject checkpoint)"
               if memorizing else "OK  — generated no closer than real-to-real")
        )

        fig_path = figures_dir / f"diffusion_fidelity_run{holdout_run}.png"
        _plot_fidelity(X_train, y_train, X_synth, y_synth, SAMPLING_RATE, holdout_run, fig_path)
        mlflow.log_artifact(str(fig_path))
        print(f"Saved fidelity figure: {fig_path}")

        # -- Final checkpoint --
        model.step = step
        model.train_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
        model.nn_distance = nn_gen
        model.mlflow_run_id = run_id
        final_path = checkpoint_dir / f"diffusion_holdout_run{holdout_run}.pt"
        model.save(final_path)
        print(f"Saved final model: {final_path}  (held_out_run={model.held_out_run})")

    print(f"\n{'='*60}")
    print(f"  Holdout run    : {holdout_run}")
    print(f"  Steps          : {step}")
    print(f"  Final loss     : {model.train_loss:.4f}")
    print(f"  psd_divergence : {psd_div:.4f}")
    print(f"  nn_distance    : {nn_gen:.3f} (real-to-real {nn_real:.3f})")
    print(f"  MLflow run id  : {run_id}")
    print(f"{'='*60}\n")

    return {
        "holdout_run": holdout_run,
        "step": step,
        "train_loss": model.train_loss,
        "psd_divergence": psd_div,
        "nn_distance": nn_gen,
        "nn_distance_real": nn_real,
        "memorizing": memorizing,
        "run_id": run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed.npz"))
    parser.add_argument("--holdout-run", type=int, choices=list(MOTOR_IMAGERY_RUNS), required=True,
                        help="Motor-imagery run excluded from THIS model's training.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--n-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--mlflow-experiment", default="phase0-diffusion")
    parser.add_argument("--n-eval-samples", type=int, default=100,
                        help="Total samples generated at training end for fidelity checks.")
    parser.add_argument("--nn-probe", type=int, default=32,
                        help="Probe batch size for the nn_distance memorization metric.")
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu":
        print(
            "WARNING: No GPU/MPS detected — full training on CPU is impractical.\n"
            "Tip: run a smoke test with --n-steps 50 --batch-size 4 --ddim-steps 10 "
            "--n-eval-samples 8 --nn-probe 8 first."
        )

    result = train(
        data_path=args.data,
        holdout_run=args.holdout_run,
        checkpoint_dir=args.checkpoint_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        resume=args.resume,
        save_every=args.save_every,
        mlflow_experiment=args.mlflow_experiment,
        n_eval_samples=args.n_eval_samples,
        nn_probe=args.nn_probe,
    )
    print(f"Run ID  : {result['run_id']}")
    print("Results logged. Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    main()
