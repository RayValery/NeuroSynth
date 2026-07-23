"""
Modal launcher for pivot_iter_02 — subject-conditional diffusion training (cloud GPU).

Same as modal_train_diffusion_exec.py, but each of the three run-holdout models adds a
per-subject embedding (residual over a population fallback via subject dropout) on top of
the domain token, and — after training — exports a per-subject imagery-domain bank. It calls
`eval/train_diffusion.py::train(use_domain_cond=True, use_subject_cond=True,
export_subject_banks=True, ckpt_prefix="diffusion_subj")`; nothing in eval/ changes here.

Produces, per held-out run R:
  diffusion_subj_holdout_run{R}.pt        subject-conditional generator (EMA weights)
  subject_banks_holdout_run{R}.npz        cached per-subject banks (X, y, subject_ids)

The combined dataset `processed_with_exec.npz` (imagery + execution + subject_ids) must be
built and uploaded first — same file used by Pivot 1. The imagery test data used by the
classifier is unchanged.

────────────────────────────────────────────────────────────────────────────
RUNBOOK (from the repo root)
────────────────────────────────────────────────────────────────────────────
1. The combined dataset must already be on the data volume (from Pivot 1):
       modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz

2. Train the three subject-conditional models + export banks (~3-4 h on A10G):
       modal run --detach modal_train_diffusion_subj.py
   Options:
       modal run --detach modal_train_diffusion_subj.py --resume     # finish from newest checkpoints
       modal run --detach modal_train_diffusion_subj.py --runs 12     # just one model
       NEUROSYNTH_GPU=A100 modal run --detach modal_train_diffusion_subj.py
       modal run modal_train_diffusion_subj.py --n-steps 300 --bank-size-per-subject 8  # cloud smoke

3. Pull the checkpoints and cached banks down:
       modal volume get neurosynth-checkpoints diffusion_subj_holdout_run4.pt   models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_subj_holdout_run8.pt   models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_subj_holdout_run12.pt  models/checkpoints/
       modal volume get neurosynth-checkpoints subject_banks_holdout_run4.npz   models/checkpoints/
       modal volume get neurosynth-checkpoints subject_banks_holdout_run8.npz   models/checkpoints/
       modal volume get neurosynth-checkpoints subject_banks_holdout_run12.npz  models/checkpoints/

Then run the TSTR passes (no diffusion reload — pure classifier work):
       modal run --detach modal_train_tstr.py --variant subj
"""
from __future__ import annotations

import os

import modal

app = modal.App("neurosynth-diffusion-subj")

GPU = os.environ.get("NEUROSYNTH_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"GIT_PYTHON_REFRESH": "quiet"})
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "mlflow>=2.10",
    )
    .add_local_dir("eval", "/root/eval", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("data", "/root/data", ignore=["*.npz", "__pycache__", "*.pyc"])
)

data_volume = modal.Volume.from_name("neurosynth-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("neurosynth-checkpoints", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/checkpoints"
CKPT_PREFIX = "diffusion_subj"


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_MOUNT: data_volume, OUT_MOUNT: ckpt_volume},
    timeout=60 * 60 * 8,   # 100K steps (~3-4h) + per-subject bank export, with slow-end headroom
)
def train_one(
    holdout_run: int,
    n_steps: int = 100_000,
    batch_size: int = 32,
    seed: int = 42,
    resume: bool = False,
    p_subj_uncond: float = 0.15,
    bank_size_per_subject: int = 128,
    subject_guidance_scale: float = 0.0,
) -> dict:
    """Train one subject-conditional run-holdout model, export its banks, persist all.

    With ``subject_guidance_scale>0`` and an already-trained checkpoint on the volume, this
    resumes at step 100K (no real training) and only RE-EXPORTS subject-guided banks to a
    ``subject_banks_g{scale}_holdout_run{R}.npz`` file — the cheap pivot_iter_02 retry."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    os.makedirs(OUT_MOUNT, exist_ok=True)
    os.chdir(OUT_MOUNT)

    # Always resume from the newest committed checkpoint for this run if one exists —
    # independent of the --resume flag (Modal restarts a preempted function with the same
    # inputs, i.e. resume=False, so gating on the flag would restart from step 0). A fresh
    # run finds nothing and trains from scratch; a preempted run continues from its save.
    _ = resume
    ckpts = sorted(
        Path(OUT_MOUNT).glob(f"{CKPT_PREFIX}_run{holdout_run}_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    resume_path = ckpts[-1] if ckpts else None
    # Fallback to the FINAL checkpoint if intermediate step checkpoints were rotated away.
    # Critical for the subject-guidance retry: it must resume a completed model (start_step
    # == n_steps → skips training, only re-exports guided banks), never retrain from scratch.
    if resume_path is None:
        final_ckpt = Path(OUT_MOUNT) / f"{CKPT_PREFIX}_holdout_run{holdout_run}.pt"
        if final_ckpt.exists():
            resume_path = final_ckpt
    if resume_path is not None:
        print(f"[subj run {holdout_run}] resuming from {resume_path.name}")
    else:
        print(f"[subj run {holdout_run}] no checkpoint found — training from scratch")

    import torch

    from eval.train_diffusion import train

    data_path = Path(DATA_MOUNT) / "processed_with_exec.npz"
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} not found on the data volume. Upload it first:\n"
            "  modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz"
        )

    print(f"[subj run {holdout_run}] CUDA available: {torch.cuda.is_available()} "
          f"| device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    result = train(
        data_path=data_path,
        holdout_run=holdout_run,
        checkpoint_dir=Path(OUT_MOUNT),
        n_steps=n_steps,
        batch_size=batch_size,
        lr=1e-4,
        guidance_scale=3.0,
        ddim_steps=50,
        seed=seed,
        resume=resume_path,
        save_every=10_000,
        mlflow_experiment="pivot2-diffusion",
        n_eval_samples=100,
        nn_probe=32,
        use_domain_cond=True,        # keep the imagery/execution domain token (Pivot 1)
        ckpt_prefix=CKPT_PREFIX,
        use_subject_cond=True,       # pivot_iter_02: per-subject residual embedding
        p_subj_uncond=p_subj_uncond,
        export_subject_banks=True,   # write subject_banks_holdout_run{R}.npz after training
        bank_size_per_subject=bank_size_per_subject,
        subject_guidance_scale=subject_guidance_scale,  # >0 → guided-bank retry (no retrain)
        on_checkpoint=ckpt_volume.commit,  # commit each save (+ the bank npz) so preemption resumes
    )

    ckpt_volume.commit()
    return result


@app.local_entrypoint()
def main(
    runs: str = "4,8,12",
    n_steps: int = 100_000,
    batch_size: int = 32,
    seed: int = 42,
    resume: bool = False,
    p_subj_uncond: float = 0.15,
    bank_size_per_subject: int = 128,
    subject_guidance_scale: float = 0.0,
) -> None:
    """Launch one GPU container per held-out run (in parallel) and print results.

    Cheap pivot_iter_02 retry (no retraining): with the trained checkpoints already on the
    volume, `modal run --detach modal_train_diffusion_subj.py --subject-guidance-scale 3`
    resumes each at 100K and only re-exports subject-guided banks (subject_banks_g3_*.npz).
    """
    holdout_runs = [int(r) for r in runs.split(",") if r.strip()]
    valid = {4, 8, 12}
    bad = [r for r in holdout_runs if r not in valid]
    if bad:
        raise ValueError(f"--runs must be a subset of 4,8,12 — got invalid {bad}")

    mode = "resuming" if resume else "training"
    print(f"Launching {len(holdout_runs)} subject-conditional {mode} job(s) on {GPU}: "
          f"runs {holdout_runs} (p_subj_uncond={p_subj_uncond}, "
          f"bank_size_per_subject={bank_size_per_subject}, "
          f"subject_guidance_scale={subject_guidance_scale:g})")

    handles = {
        r: train_one.spawn(r, n_steps, batch_size, seed, resume, p_subj_uncond,
                           bank_size_per_subject, subject_guidance_scale)
        for r in holdout_runs
    }

    print("\n================ RESULTS ================")
    for r, handle in handles.items():
        result = handle.get()
        flag = "MEMORIZING (reject)" if result["memorizing"] else "clean"
        memo = result.get("subject_memo_min_ratio", float("nan"))
        spread = result.get("subject_embedding_spread_sameseed",
                            result.get("subject_embedding_spread", float("nan")))
        print(
            f"subj run {r:>2}: loss={result['train_loss']:.4f} "
            f"psd_div={result['psd_divergence']:.4f} "
            f"nn={result['nn_distance']:.3f} (real {result['nn_distance_real']:.3f}) [{flag}] "
            f"| per-subject nn_gen/nn_real min={memo:.3f} emb_spread={spread:.4f}  "
            f"mlflow={result['run_id']}"
        )
    print("=========================================")
    print(
        "\nPull checkpoints + cached banks down with:\n"
        "  modal volume get neurosynth-checkpoints diffusion_subj_holdout_run4.pt   models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_subj_holdout_run8.pt   models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_subj_holdout_run12.pt  models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints subject_banks_holdout_run4.npz   models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints subject_banks_holdout_run8.npz   models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints subject_banks_holdout_run12.npz  models/checkpoints/\n"
        "then run the TSTR passes:  modal run --detach modal_train_tstr.py --variant subj"
    )
