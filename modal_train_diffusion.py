"""
Modal launcher for Phase 0 Step 4 diffusion training (cloud GPU).

This runs `eval/train_diffusion.py::train` on Modal's GPUs — one container per
held-out run (4/8/12), in parallel. Nothing in eval/ changes; this file only
imports and calls train() on the remote worker.

Data (861 MB processed.npz) and outputs (checkpoints, fidelity figures, mlruns)
live in two persistent Modal Volumes so you upload the data once and can pull
the results down afterwards.

────────────────────────────────────────────────────────────────────────────
RUNBOOK (all commands from the repo root, on your laptop)
────────────────────────────────────────────────────────────────────────────
1. Install + authenticate (one time):
       pip install modal
       modal setup                       # opens a browser to link your account

2. Upload the dataset into the data volume (one time, ~861 MB):
       modal volume create neurosynth-data
       modal volume put neurosynth-data data/processed.npz /processed.npz

3. Train. Runs all three held-out models in parallel. Use --detach so the job
   survives your laptop sleeping/closing the lid (recommended — a full run is
   ~2.5-3 h/model on A10G). Watch progress at the app URL printed on launch.
       modal run --detach modal_train_diffusion.py
   Options:
       modal run --detach modal_train_diffusion.py --resume       # finish from newest checkpoints
       modal run --detach modal_train_diffusion.py --runs 12      # just one model
       NEUROSYNTH_GPU=A100 modal run --detach modal_train_diffusion.py   # faster GPU (~1-1.5 h/model)
       modal run modal_train_diffusion.py --n-steps 5000          # quick cloud smoke test

4. Pull the results down to your laptop:
       modal volume get neurosynth-checkpoints diffusion_holdout_run4.pt  models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_holdout_run8.pt  models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_holdout_run12.pt models/checkpoints/
       modal volume get neurosynth-checkpoints reports models/../modal_reports   # fidelity PNGs
   (or grab everything: `modal volume get neurosynth-checkpoints / ./modal_outputs`)

Cost estimate: A10G ≈ $1.10/hr, ~2.5-3 h/model → ~$3/model, ~$8-9 for all three
(parallel = same wall-clock, same total GPU-cost). --resume only trains the
remaining steps, so finishing a timed-out batch is much cheaper.
"""
from __future__ import annotations

import os

import modal

app = modal.App("neurosynth-diffusion")

# GPU type is read at import time (works on every Modal version — no with_options).
# Override with:  NEUROSYNTH_GPU=A100 modal run modal_train_diffusion.py
GPU = os.environ.get("NEUROSYNTH_GPU", "A10G")

# Training only reads the .npz — no MNE needed on the worker. Torch's default
# linux PyPI wheel is CUDA-enabled, so the GPU is used automatically.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"GIT_PYTHON_REFRESH": "quiet"})  # silence mlflow's git-not-found warnings
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "mlflow>=2.10",
    )
    # Ship the source packages (code only — exclude the 861 MB npz and caches).
    .add_local_dir("eval", "/root/eval", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("data", "/root/data", ignore=["*.npz", "__pycache__", "*.pyc"])
)

# Persistent volumes: dataset in, artifacts out.
data_volume = modal.Volume.from_name("neurosynth-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("neurosynth-checkpoints", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/checkpoints"


@app.function(
    image=image,
    gpu=GPU,  # set via NEUROSYNTH_GPU env var (default A10G)
    volumes={DATA_MOUNT: data_volume, OUT_MOUNT: ckpt_volume},
    timeout=60 * 60 * 6,  # 6 h ceiling per model (A10G ~2.5-3 h for 100K steps)
)
def train_one(
    holdout_run: int,
    n_steps: int = 100_000,
    batch_size: int = 32,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    """Train one run-holdout model on a GPU worker and persist its artifacts."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    # Run from the output volume so train()'s relative paths (reports/figures,
    # mlruns) persist alongside the checkpoints and can be downloaded later.
    os.makedirs(OUT_MOUNT, exist_ok=True)
    os.chdir(OUT_MOUNT)

    # Resume from the newest intermediate checkpoint for this run, if requested.
    # The volume is mounted with its latest committed state at container start, so
    # the previous run's checkpoints are already visible — no reload() needed (and
    # reload() would fail anyway once cwd is inside the volume).
    resume_path = None
    if resume:
        ckpts = sorted(
            Path(OUT_MOUNT).glob(f"diffusion_run{holdout_run}_step_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )
        if ckpts:
            resume_path = ckpts[-1]
            print(f"[run {holdout_run}] resuming from {resume_path.name}")
        else:
            print(f"[run {holdout_run}] --resume set but no checkpoint found — training from scratch")

    import torch

    from eval.train_diffusion import train

    print(f"[run {holdout_run}] CUDA available: {torch.cuda.is_available()} "
          f"| device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    result = train(
        data_path=Path(DATA_MOUNT) / "processed.npz",
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
        mlflow_experiment="phase0-diffusion",
        n_eval_samples=100,
        nn_probe=32,
    )

    # Flush writes so `modal volume get` sees the checkpoints/figures.
    ckpt_volume.commit()
    return result


@app.local_entrypoint()
def main(
    runs: str = "4,8,12",
    n_steps: int = 100_000,
    batch_size: int = 32,
    seed: int = 42,
    resume: bool = False,
) -> None:
    """Launch one GPU container per held-out run (in parallel) and print results."""
    holdout_runs = [int(r) for r in runs.split(",") if r.strip()]
    valid = {4, 8, 12}
    bad = [r for r in holdout_runs if r not in valid]
    if bad:
        raise ValueError(f"--runs must be a subset of 4,8,12 — got invalid {bad}")

    mode = "resuming" if resume else "training"
    print(f"Launching {len(holdout_runs)} {mode} job(s) on {GPU}: runs {holdout_runs}")

    # spawn() is async → all requested models train concurrently.
    handles = {r: train_one.spawn(r, n_steps, batch_size, seed, resume) for r in holdout_runs}

    print("\n================ RESULTS ================")
    for r, handle in handles.items():
        result = handle.get()
        flag = "MEMORIZING (reject)" if result["memorizing"] else "clean"
        print(
            f"run {r:>2}: loss={result['train_loss']:.4f} "
            f"psd_div={result['psd_divergence']:.4f} "
            f"nn={result['nn_distance']:.3f} (real {result['nn_distance_real']:.3f}) "
            f"[{flag}]  mlflow={result['run_id']}"
        )
    print("=========================================")
    print(
        "\nPull checkpoints down with:\n"
        "  modal volume get neurosynth-checkpoints diffusion_holdout_run4.pt  models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_holdout_run8.pt  models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_holdout_run12.pt models/checkpoints/"
    )
