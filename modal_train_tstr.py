"""
Modal launcher for Phase 0 Step 5 TSTR go/no-go gate (cloud GPU).

This runs `eval/train_baseline.py::train(run_name="tstr", ...)` on Modal's GPUs —
one container per n_copies value (1/2/3 → N=30/60/90), in parallel. Each container
builds the three synthetic banks once (from the frozen Step 4 diffusion checkpoints
already on the volume), then runs all 318 LORO folds with real+synthetic training.
Nothing in eval/ changes; this file only imports and calls train() on the remote worker.

Both inputs already live on Modal volumes from Step 4:
  * neurosynth-data        — processed.npz
  * neurosynth-checkpoints — diffusion_holdout_run{4,8,12}.pt (frozen weights)

The launcher ships the CURRENT eval/diffusion.py (with the clip_denoised + z-score
sampler fix); the checkpoint *weights* on the volume are identical to the local copies.
The per-fold synthetic draws are keyed by test run, so a fold is only ever augmented
by the model that never saw its test run (leakage guard asserted in _load_synthetic_set).

────────────────────────────────────────────────────────────────────────────
RUNBOOK (all commands from the repo root, on your laptop)
────────────────────────────────────────────────────────────────────────────
1. One-time setup (if not already done for Step 4):
       pip install modal
       modal setup
       modal volume put neurosynth-data data/processed.npz /processed.npz

2. Ensure the three diffusion checkpoints are on the checkpoints volume (present
   from Step 4). Optionally refresh them with the corrected local copies:
       modal volume put neurosynth-checkpoints models/checkpoints/diffusion_holdout_run4.pt  /diffusion_holdout_run4.pt
       modal volume put neurosynth-checkpoints models/checkpoints/diffusion_holdout_run8.pt  /diffusion_holdout_run8.pt
       modal volume put neurosynth-checkpoints models/checkpoints/diffusion_holdout_run12.pt /diffusion_holdout_run12.pt

3. Run all three TSTR passes in parallel. Use --detach so the job survives your
   laptop sleeping (bank generation + 3×318 EEGNet fold-trainings, ~1 h wall-clock):
       modal run --detach modal_train_tstr.py
   Options:
       modal run --detach modal_train_tstr.py --copies 2          # just the gate (N=60)
       modal run modal_train_tstr.py --max-epochs 20 --bank-size 128   # quick cloud smoke test
       NEUROSYNTH_GPU=A100 modal run --detach modal_train_tstr.py  # faster GPU

4. Pull the three result JSONs down to your laptop:
       modal volume get neurosynth-checkpoints per_subject_results_tstr_1x.json models/checkpoints/
       modal volume get neurosynth-checkpoints per_subject_results_tstr_2x.json models/checkpoints/
       modal volume get neurosynth-checkpoints per_subject_results_tstr_3x.json models/checkpoints/

5. Compute the RR-3 verdict and figures locally:
       python eval/compare_baselines.py

Cost estimate: A10G ≈ $1.10/hr, ~15-30 min/pass, three passes in parallel ≈ same
wall-clock → ~$1-2 total. Cost is dominated by the many short EEGNet fold-trainings,
not the (tiny) diffusion bank sampling.
"""
from __future__ import annotations

import os

import modal

app = modal.App("neurosynth-tstr")

# GPU type is read at import time (works on every Modal version — no with_options).
# Override with:  NEUROSYNTH_GPU=A100 modal run modal_train_tstr.py
GPU = os.environ.get("NEUROSYNTH_GPU", "A10G")

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
    # Ship the source packages (code only — exclude the large npz and caches).
    .add_local_dir("eval", "/root/eval", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("data", "/root/data", ignore=["*.npz", "__pycache__", "*.pyc"])
)

# Persistent volumes: dataset + frozen diffusion checkpoints in, result JSONs out.
data_volume = modal.Volume.from_name("neurosynth-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("neurosynth-checkpoints", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/checkpoints"


@app.function(
    image=image,
    gpu=GPU,  # set via NEUROSYNTH_GPU env var (default A10G)
    volumes={DATA_MOUNT: data_volume, OUT_MOUNT: ckpt_volume},
    timeout=60 * 60 * 6,  # 6 h ceiling per pass (banks + 318 fold-trainings ≈ 15-30 min)
)
def train_tstr(
    n_copies: int,
    bank_size: int = 2048,
    seed: int = 42,
    max_epochs: int = 500,
) -> dict:
    """Run one TSTR pass (real + n_copies synthetic) on a GPU worker; persist the JSON."""
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    # Run from the output volume so train()'s relative paths (mlruns) persist next to
    # the result JSONs and can be downloaded later. The checkpoints volume already
    # carries the three diffusion models at container start — no reload() needed.
    os.makedirs(OUT_MOUNT, exist_ok=True)
    os.chdir(OUT_MOUNT)

    import torch

    from eval.train_baseline import train

    print(f"[tstr {n_copies}x] CUDA available: {torch.cuda.is_available()} "
          f"| device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    result = train(
        data_path=Path(DATA_MOUNT) / "processed.npz",
        run_name="tstr",
        checkpoint_dir=Path(OUT_MOUNT),
        mlflow_experiment="phase0-tstr",
        seed=seed,
        max_epochs=max_epochs,
        n_copies=n_copies,
        bank_size=bank_size,
    )

    # Flush writes so `modal volume get` sees per_subject_results_tstr_{n}x.json.
    ckpt_volume.commit()
    return result


@app.local_entrypoint()
def main(
    copies: str = "1,2,3",
    bank_size: int = 2048,
    seed: int = 42,
    max_epochs: int = 500,
) -> None:
    """Launch one GPU container per n_copies value (in parallel) and print results."""
    n_copies_list = [int(c) for c in copies.split(",") if c.strip()]
    bad = [c for c in n_copies_list if c not in (1, 2, 3)]
    if bad:
        raise ValueError(f"--copies must be a subset of 1,2,3 — got invalid {bad}")

    print(f"Launching {len(n_copies_list)} TSTR pass(es) on {GPU}: n_copies {n_copies_list}")

    # spawn() is async → all requested passes run concurrently.
    handles = {
        c: train_tstr.spawn(c, bank_size, seed, max_epochs) for c in n_copies_list
    }

    print("\n================ TSTR RESULTS ================")
    for c, handle in handles.items():
        result = handle.get()
        n_extra = c * 30  # nominal N per fold (30 real samples typical)
        print(
            f"tstr {c}x (N≈{n_extra:>3}): "
            f"acc={result['test_accuracy']*100:.1f}% ± {result['test_accuracy_std']*100:.1f}%  "
            f"f1={result['test_f1']:.3f}  mlflow={result['run_id']}"
        )
    print("=============================================")
    print(
        "\nPull the result JSONs down with:\n"
        "  modal volume get neurosynth-checkpoints per_subject_results_tstr_1x.json models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints per_subject_results_tstr_2x.json models/checkpoints/\n"
        "  modal volume get neurosynth-checkpoints per_subject_results_tstr_3x.json models/checkpoints/\n"
        "then compute the RR-3 verdict:\n"
        "  python eval/compare_baselines.py"
    )
