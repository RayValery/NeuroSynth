"""
Modal launcher for pivot_iter_01 — execution-conditioned diffusion training (cloud GPU).

Same as cloud/modal_train_diffusion.py, but each of the three run-holdout models is trained
on imagery + real motor-execution jointly, domain-conditioned (imagery=0 / execution=1),
and saved as diffusion_exec_holdout_run{4,8,12}.pt. It calls
`src/neurosynth/train_diffusion.py::train(use_domain_cond=True, ckpt_prefix="diffusion_exec")`;
nothing in src/neurosynth/ changes here.

The combined dataset `processed_with_exec.npz` must be built and uploaded first (it
includes execution runs 3/7/11 and a `domains` array). The imagery-only test data used
by the classifier is unchanged.

────────────────────────────────────────────────────────────────────────────
RUNBOOK (from the repo root)
────────────────────────────────────────────────────────────────────────────
1. Build the combined dataset locally (needs MNE + downloads execution runs):
       python scripts/build_dataset.py --subjects $(seq 1 109) --task exec_imagery_lr \
           --out data/processed_with_exec.npz

2. Upload it to the data volume (one time):
       modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz

3. Train the three execution-conditioned models in parallel (~2.5-3 h/model on A10G):
       modal run --detach cloud/modal_train_diffusion_exec.py
   Options:
       modal run --detach cloud/modal_train_diffusion_exec.py --resume        # finish from newest checkpoints
       modal run --detach cloud/modal_train_diffusion_exec.py --runs 12        # just one model
       NEUROSYNTH_GPU=A100 modal run --detach cloud/modal_train_diffusion_exec.py
       modal run cloud/modal_train_diffusion_exec.py --n-steps 300            # quick cloud smoke test

4. Pull the checkpoints down:
       modal volume get neurosynth-checkpoints diffusion_exec_holdout_run4.pt  models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_exec_holdout_run8.pt  models/checkpoints/
       modal volume get neurosynth-checkpoints diffusion_exec_holdout_run12.pt models/checkpoints/

Then run the TSTR passes:  modal run --detach cloud/modal_train_tstr.py --variant exec
"""
from __future__ import annotations

import os

import modal

app = modal.App("neurosynth-diffusion-exec")

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
    .add_local_dir("src/neurosynth", "/root/neurosynth", ignore=["__pycache__", "*.pyc"])
)

data_volume = modal.Volume.from_name("neurosynth-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("neurosynth-checkpoints", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/checkpoints"
CKPT_PREFIX = "diffusion_exec"


@app.function(
    image=image,
    gpu=GPU,
    volumes={DATA_MOUNT: data_volume, OUT_MOUNT: ckpt_volume},
    timeout=60 * 60 * 6,
)
def train_one(
    holdout_run: int,
    n_steps: int = 100_000,
    batch_size: int = 32,
    seed: int = 42,
    resume: bool = False,
) -> dict:
    """Train one execution-conditioned run-holdout model and persist its artifacts."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    os.makedirs(OUT_MOUNT, exist_ok=True)
    os.chdir(OUT_MOUNT)

    # Always resume from the newest committed checkpoint for this run if one exists —
    # independent of the --resume flag. Modal preemption restarts the function with the
    # SAME inputs (i.e. resume=False), so gating resumption behind the flag would make an
    # automatic restart begin again at step 0. Checking for a checkpoint unconditionally
    # makes the container idempotent: a fresh run finds nothing and trains from scratch;
    # a preempted/restarted run continues from its last save. (--resume is now advisory.)
    _ = resume
    ckpts = sorted(
        Path(OUT_MOUNT).glob(f"{CKPT_PREFIX}_run{holdout_run}_step_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    resume_path = ckpts[-1] if ckpts else None
    if resume_path is not None:
        print(f"[exec run {holdout_run}] resuming from {resume_path.name}")
    else:
        print(f"[exec run {holdout_run}] no checkpoint found — training from scratch")

    import torch

    from neurosynth.train_diffusion import train

    data_path = Path(DATA_MOUNT) / "processed_with_exec.npz"
    if not data_path.exists():
        raise SystemExit(
            f"{data_path} not found on the data volume. Upload it first:\n"
            "  modal volume put neurosynth-data data/processed_with_exec.npz /processed_with_exec.npz"
        )

    print(f"[exec run {holdout_run}] CUDA available: {torch.cuda.is_available()} "
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
        mlflow_experiment="pivot1-diffusion",
        n_eval_samples=100,
        nn_probe=32,
        use_domain_cond=True,
        ckpt_prefix=CKPT_PREFIX,
        on_checkpoint=ckpt_volume.commit,   # commit each save so preemption resumes, not restarts
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
) -> None:
    """Launch one GPU container per held-out run (in parallel) and print results."""
    holdout_runs = [int(r) for r in runs.split(",") if r.strip()]
    valid = {4, 8, 12}
    bad = [r for r in holdout_runs if r not in valid]
    if bad:
        raise ValueError(f"--runs must be a subset of 4,8,12 — got invalid {bad}")

    mode = "resuming" if resume else "training"
    print(f"Launching {len(holdout_runs)} exec-conditioned {mode} job(s) on {GPU}: runs {holdout_runs}")

    handles = {r: train_one.spawn(r, n_steps, batch_size, seed, resume) for r in holdout_runs}

    print("\n================ RESULTS ================")
    for r, handle in handles.items():
        result = handle.get()
        flag = "MEMORIZING (reject)" if result["memorizing"] else "clean"
        print(
            f"exec run {r:>2}: loss={result['train_loss']:.4f} "
            f"psd_div={result['psd_divergence']:.4f} "
            f"nn={result['nn_distance']:.3f} (real {result['nn_distance_real']:.3f}) "
            f"[{flag}]  mlflow={result['run_id']}"
        )
    print("=========================================")
    print(
        "\nPull checkpoints down with:\n"
        "  modal volume get neurosynth-checkpoints diffusion_exec_holdout_run4.pt  models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_exec_holdout_run8.pt  models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints diffusion_exec_holdout_run12.pt models/checkpoints/\n"
        "then run the TSTR passes:  modal run --detach cloud/modal_train_tstr.py --variant exec"
    )
