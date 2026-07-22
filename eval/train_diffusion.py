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
from typing import Callable

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
    DOMAIN_IMAGERY,
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
    model: EEGDiffusionModel,
    X_real_flat: np.ndarray,
    n_probe: int,
    seed: int,
    subject_id: int = 0,
) -> float:
    """Mean L2 from a probe batch of generated samples to the nearest real sample.

    ``subject_id`` is ignored by population models but must be a real cohort id for
    subject-conditional models (subject_id=0 would raise KeyError there).
    """
    X_gen = model.generate(n_probe, subject_id=subject_id, seed=seed)
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


def _rotate_checkpoints(
    checkpoint_dir: Path, holdout_run: int, keep: int = 3, prefix: str = "diffusion"
) -> None:
    """Keep only the newest ``keep`` intermediate checkpoints for this run."""
    pattern = f"{prefix}_run{holdout_run}_step_*.pt"
    ckpts = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Subject-conditional bank export (pivot_iter_02)
# --------------------------------------------------------------------------- #
def _export_subject_banks(
    model: EEGDiffusionModel,
    subjects: list[int],
    bank_size_per_subject: int,
    domain: int,
    seed: int,
    holdout_run: int,
    checkpoint_dir: Path,
) -> tuple[Path, dict[int, np.ndarray]]:
    """Generate a class-balanced imagery-domain bank for every subject and cache them.

    Saves ``subject_banks_holdout_run{R}.npz`` (X, y, subject_ids grouped per subject,
    plus scalar held_out_run / bank_size_per_subject). The TSTR harness later loads this
    file directly — no diffusion model reload — and routes each fold (S, R) to its bank.
    Returns the file path and a {subject_id: generated_epochs} dict for the spot-checks.
    """
    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_s: list[np.ndarray] = []
    per_subject: dict[int, np.ndarray] = {}
    for S in subjects:
        # Distinct, deterministic per-(run, subject) seed (no cross-(R,S) collision).
        s_seed = seed + holdout_run * 100_000 + int(S)
        Xs = model.generate(bank_size_per_subject, subject_id=int(S), domain=domain, seed=s_seed)
        ys = model.labels(bank_size_per_subject)
        parts_X.append(Xs.astype(np.float32))
        parts_y.append(ys.astype(np.int64))
        parts_s.append(np.full(len(ys), int(S), dtype=np.int64))
        per_subject[int(S)] = Xs

    X_all = np.concatenate(parts_X)
    y_all = np.concatenate(parts_y)
    s_all = np.concatenate(parts_s)
    path = checkpoint_dir / f"subject_banks_holdout_run{holdout_run}.npz"
    np.savez_compressed(
        path,
        X=X_all,
        y=y_all,
        subject_ids=s_all,
        held_out_run=np.int64(holdout_run),
        bank_size_per_subject=np.int64(bank_size_per_subject),
    )
    std = float(X_all.std())
    print(f"Exported {path.name}: {len(subjects)} subjects x {bank_size_per_subject} "
          f"= {len(X_all)} epochs | std={std:.3f} | class balance={np.bincount(y_all).tolist()}")
    return path, per_subject


def _subject_bank_checks(
    per_subject: dict[int, np.ndarray],
    X_fid: np.ndarray,
    subj_fid: np.ndarray,
    nn_probe: int,
    rng: np.random.Generator,
    max_subjects: int = 8,
) -> dict:
    """Per-subject memorization guard + subject-embedding-used probe (Should Have).

    * Memorization: for a sample of subjects, generated-for-S must stay no closer to
      S's real imagery than S's real-to-real NN (nn_gen_S >= nn_real_S).
    * Embedding-used: banks conditioned on different subjects must differ (spread > 0).
    Returns a small metrics dict; prints a summary. Never raises — the caller decides.
    """
    subjects = sorted(per_subject)
    sample = subjects[:: max(1, len(subjects) // max_subjects)][:max_subjects]

    memo_fail: list[int] = []
    min_ratio = float("inf")
    for S in sample:
        real_S = X_fid[subj_fid == S]
        if len(real_S) < 3:
            continue
        real_flat = real_S.reshape(len(real_S), -1)
        gen_flat = per_subject[S].reshape(len(per_subject[S]), -1)
        n_pr = min(nn_probe, len(gen_flat))
        nn_gen_S = _pairwise_nn(gen_flat[:n_pr], real_flat)
        nn_real_S = _nn_distance_real(real_flat, min(nn_probe, len(real_flat) - 1), rng)
        ratio = nn_gen_S / nn_real_S if nn_real_S > 0 else float("inf")
        min_ratio = min(min_ratio, ratio)
        if nn_gen_S < nn_real_S:
            memo_fail.append(int(S))

    # Embedding-used: mean pairwise difference between the first few subjects' banks.
    spread = float("nan")
    if len(sample) >= 2:
        diffs = []
        for i in range(len(sample) - 1):
            a = per_subject[sample[i]]
            b = per_subject[sample[i + 1]]
            m = min(len(a), len(b))
            diffs.append(float(np.abs(a[:m] - b[:m]).mean()))
        spread = float(np.mean(diffs))

    print(f"Per-subject memorization  : checked {len(sample)} subjects | "
          f"min(nn_gen/nn_real)={min_ratio:.3f} | "
          + (f"FAIL for subjects {memo_fail}" if memo_fail else "OK (all >= real-to-real)"))
    print(f"Subject-embedding spread  : mean|Δ| across neighbouring subjects={spread:.4f} "
          + ("(embedding is used)" if spread and spread > 0 else "(WARNING: near-zero — embedding may be ignored)"))
    return {
        "subject_memo_min_ratio": min_ratio,
        "subject_memo_fail_count": len(memo_fail),
        "subject_embedding_spread": spread,
    }


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
    use_domain_cond: bool = False,
    ckpt_prefix: str = "diffusion",
    fidelity_domain: int = DOMAIN_IMAGERY,
    on_checkpoint: Callable[[], None] | None = None,
    use_subject_cond: bool = False,
    p_subj_uncond: float = 0.15,
    export_subject_banks: bool = False,
    bank_size_per_subject: int = 128,
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

    # -- Run-holdout filter: drop the held-out (imagery) run, keep everything else --
    # With the combined dataset this keeps imagery {4,8,12}\R (domain 0) AND all
    # execution runs {3,7,11} (domain 1); execution is never a held-out/test run.
    keep = ds.run_ids != holdout_run
    X_train = np.ascontiguousarray(ds.X[keep].astype(np.float32))
    y_train = ds.y[keep].astype(np.int64)
    runs_train = ds.run_ids[keep]
    assert (runs_train == holdout_run).sum() == 0, "leakage: held-out run present in train set"

    # Domain per training epoch (0=imagery, 1=execution). Prefer the stored domains
    # array; legacy imagery-only files have none → all-imagery.
    if ds.domains is not None:
        domain_train = ds.domains[keep].astype(np.int64)
    else:
        domain_train = np.zeros(len(y_train), dtype=np.int64)
    if use_domain_cond and domain_train.max() == 0:
        sys.exit(
            "use_domain_cond=True but the dataset has no execution (domain=1) epochs. "
            "Point --data at data/processed_with_exec.npz (built with --task exec_imagery_lr)."
        )

    train_runs = sorted(np.unique(runs_train).tolist())
    # Subject cohort in a stable order → contiguous embedding index. subjects_train is
    # the per-epoch original id; unique_subjects[i] is the id at embedding index i.
    subjects_train = ds.subject_ids[keep].astype(np.int64)
    unique_subjects = sorted(int(s) for s in np.unique(subjects_train))
    n_subjects = len(unique_subjects)
    device = get_device()

    print(f"Device        : {device}")
    print(f"Holdout run   : {holdout_run}  (train runs: {train_runs})")
    print(f"Subjects      : {n_subjects}  (all kept)  | subject_cond={use_subject_cond}")
    print(f"Train epochs  : {len(y_train)}  | class balance: {np.bincount(y_train).tolist()}")
    print(f"Domain balance: imagery={int((domain_train == 0).sum())} "
          f"execution={int((domain_train == 1).sum())}  | domain_cond={use_domain_cond}")
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
        use_domain_cond=use_domain_cond,
        sample_domain=fidelity_domain,   # sample the domain we evaluate against (imagery)
        use_subject_cond=use_subject_cond,
        n_subjects=n_subjects if use_subject_cond else DiffusionConfig.n_subjects,
        p_subj_uncond=p_subj_uncond,
        subject_ids=tuple(unique_subjects) if use_subject_cond else (),
    )

    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["channel_mults"] = tuple(cfg_dict["channel_mults"])
        cfg_dict["subject_ids"] = tuple(cfg_dict.get("subject_ids", ()))
        config = DiffusionConfig(**cfg_dict)   # checkpoint config is authoritative
        start_step = int(ckpt.get("step", 0))
        print(f"Resuming from : {resume}  (step {start_step})")

    model = EEGDiffusionModel(config, device)
    if resume is not None:
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.ddpm.ema_model.load_state_dict(ckpt["ema_state_dict"])

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"UNet1D params : {n_params:,}")

    # Per-epoch subject embedding index (pivot_iter_02). Built from the FINAL config's
    # subject_ids (checkpoint-authoritative on resume) so a resumed run maps ids the same
    # way it was first trained. Zeros (unused) when subject conditioning is off.
    if config.use_subject_cond:
        subj_to_idx = {int(o): i for i, o in enumerate(config.subject_ids)}
        missing = sorted(set(int(s) for s in subjects_train) - set(subj_to_idx))
        if missing:
            sys.exit(
                f"Subjects {missing} present in data but absent from the checkpoint's "
                "subject_ids — cannot map them to an embedding index. Retrain from scratch "
                "on this dataset instead of resuming a model trained on a different cohort."
            )
        subject_idx_train = np.array([subj_to_idx[int(s)] for s in subjects_train], dtype=np.int64)
    else:
        subject_idx_train = np.zeros(len(y_train), dtype=np.int64)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=config.lr)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(domain_train),
            torch.from_numpy(subject_idx_train),
        ),
        batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True,
    )

    # Fidelity / memorization are judged against the SAMPLING domain's real epochs
    # (imagery) — the distribution the classifier is tested on — not against the
    # execution epochs that only enrich the generator. For imagery-only training this
    # is the whole training set, so behaviour is unchanged.
    fid_mask = domain_train == fidelity_domain
    X_fid = X_train[fid_mask]
    y_fid = y_train[fid_mask]
    subj_fid = subjects_train[fid_mask]              # original id per imagery-real epoch
    X_real_flat = X_fid.reshape(len(X_fid), -1)  # for nn_distance (imagery real)
    # Representative cohort subject for population-level fidelity generate() calls —
    # subject-conditional models reject subject_id=0 (not a real id).
    fid_subject = int(config.subject_ids[0]) if config.use_subject_cond else 0

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
            "use_domain_cond": use_domain_cond,
            "n_domains": config.n_domains,
            "sample_domain": config.sample_domain,
            "use_subject_cond": use_subject_cond,
            "p_subj_uncond": config.p_subj_uncond,
            "n_imagery_epochs": int((domain_train == 0).sum()),
            "n_execution_epochs": int((domain_train == 1).sum()),
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
            for x_batch, y_batch, d_batch, s_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                d_batch = d_batch.to(device) if config.use_domain_cond else None
                s_batch = s_batch.to(device) if config.use_subject_cond else None

                loss = model.ddpm.train_step(x_batch, y_batch, d_batch, s_batch)
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
                    model.nn_distance = _nn_distance_gen(
                        model, X_real_flat, nn_probe, seed=seed, subject_id=fid_subject
                    )
                    mlflow.log_metric("nn_distance", model.nn_distance, step=step)
                    model.model.train()
                    ckpt_path = checkpoint_dir / f"{ckpt_prefix}_run{holdout_run}_step_{step}.pt"
                    model.save(ckpt_path)
                    _rotate_checkpoints(checkpoint_dir, holdout_run, keep=3, prefix=ckpt_prefix)
                    # Durably persist this checkpoint before continuing. On Modal this
                    # commits the volume so a preempted worker can resume from here
                    # rather than restarting from step 0.
                    if on_checkpoint is not None:
                        on_checkpoint()
                    print(f"  saved {ckpt_path.name} | nn_distance={model.nn_distance:.3f}")

                if step >= n_steps:
                    done = True
                    break

        # -- Post-training fidelity + memorization checks --
        print("Generating evaluation samples for fidelity/memorization checks ...")
        n_per = max(1, n_eval_samples // 2)
        X_s0 = model.generate(n_per, subject_id=fid_subject, class_label=0, seed=seed)
        X_s1 = model.generate(n_per, subject_id=fid_subject, class_label=1, seed=seed + 1)
        X_synth = np.concatenate([X_s0, X_s1])
        y_synth = np.concatenate([np.zeros(len(X_s0), np.int64), np.ones(len(X_s1), np.int64)])

        psd_div = _psd_divergence(X_fid, X_synth, SAMPLING_RATE)
        nn_gen = _nn_distance_gen(model, X_real_flat, nn_probe, seed=seed, subject_id=fid_subject)
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

        fig_path = figures_dir / f"{ckpt_prefix}_fidelity_run{holdout_run}.png"
        _plot_fidelity(X_fid, y_fid, X_synth, y_synth, SAMPLING_RATE, holdout_run, fig_path)
        mlflow.log_artifact(str(fig_path))
        print(f"Saved fidelity figure: {fig_path}")

        # -- Final checkpoint --
        model.step = step
        model.train_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
        model.nn_distance = nn_gen
        model.mlflow_run_id = run_id
        final_path = checkpoint_dir / f"{ckpt_prefix}_holdout_run{holdout_run}.pt"
        model.save(final_path)
        print(f"Saved final model: {final_path}  (held_out_run={model.held_out_run})")

        # -- Per-subject bank export (pivot_iter_02) --
        subject_bank_path = None
        subj_checks: dict = {}
        if export_subject_banks:
            if not config.use_subject_cond:
                sys.exit("export_subject_banks=True requires use_subject_cond=True.")
            print(f"Exporting per-subject banks ({bank_size_per_subject}/subject, "
                  f"domain={fidelity_domain}) ...")
            model.model.eval()
            bank_path, per_subject = _export_subject_banks(
                model,
                subjects=list(config.subject_ids),
                bank_size_per_subject=bank_size_per_subject,
                domain=fidelity_domain,
                seed=seed,
                holdout_run=holdout_run,
                checkpoint_dir=checkpoint_dir,
            )
            subject_bank_path = str(bank_path)
            subj_checks = _subject_bank_checks(
                per_subject, X_fid, subj_fid, nn_probe, rng,
            )
            mlflow.log_metrics({k: float(v) for k, v in subj_checks.items()
                                if np.isfinite(v)}, step=step)
            if on_checkpoint is not None:
                on_checkpoint()   # commit the bank npz to the volume (Modal durability)

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
        "subject_bank_path": subject_bank_path,
        **subj_checks,
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
    parser.add_argument("--use-domain-cond", action="store_true",
                        help="pivot_iter_01: add an imagery/execution domain token; train jointly "
                             "on imagery+execution (requires --data processed_with_exec.npz).")
    parser.add_argument("--ckpt-prefix", default="diffusion",
                        help="Checkpoint filename prefix (use 'diffusion_exec' for the "
                             "execution-conditioned models so they don't clash with Step 4).")
    parser.add_argument("--use-subject-cond", action="store_true",
                        help="pivot_iter_02: add a per-subject embedding (residual over a "
                             "population fallback via subject dropout). Use with --ckpt-prefix "
                             "diffusion_subj and --export-subject-banks.")
    parser.add_argument("--p-subj-uncond", type=float, default=0.15,
                        help="Subject-token dropout probability (residual strength). Default 0.15.")
    parser.add_argument("--export-subject-banks", action="store_true",
                        help="After training, generate a per-subject imagery-domain bank for "
                             "every subject and save subject_banks_holdout_run{R}.npz.")
    parser.add_argument("--bank-size-per-subject", type=int, default=128,
                        help="Epochs generated per subject for the exported bank (64/class at "
                             "128). Default 128.")
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
        use_domain_cond=args.use_domain_cond,
        ckpt_prefix=args.ckpt_prefix,
        use_subject_cond=args.use_subject_cond,
        p_subj_uncond=args.p_subj_uncond,
        export_subject_banks=args.export_subject_banks,
        bank_size_per_subject=args.bank_size_per_subject,
    )
    print(f"Run ID  : {result['run_id']}")
    print("Results logged. Run `mlflow ui` to view the experiment.")


if __name__ == "__main__":
    main()
