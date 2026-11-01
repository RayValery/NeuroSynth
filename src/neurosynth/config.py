"""Paths and constants shared across the project.

Training code (train_baseline, train_diffusion) keeps paths relative to the working
directory on purpose: the cloud launchers run it from inside a Modal volume. Run every
script from the repository root.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PROCESSED_IMAGERY = DATA_DIR / "processed.npz"            # imagery runs 4/8/12
PROCESSED_WITH_EXEC = DATA_DIR / "processed_with_exec.npz"  # + execution runs 3/7/11

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"   # training outputs (gitignored)
RESULTS_DIR = ROOT / "results"                     # summary tables (CSV)
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"      # per-subject result JSON, one per arm
FIGURES_DIR = ROOT / "figures"                     # README figures
REPORT_FIGURES_DIR = ROOT / "reports" / "figures"  # research-log figures

# PhysioNet EEGBCI, left vs right fist
IMAGERY_RUNS = (4, 8, 12)
EXECUTION_RUNS = (3, 7, 11)
DROPPED_SUBJECTS = (88, 92, 100)

SEED = 42

# Subject tiers by real-only accuracy (%), lower bound inclusive
TIERS = (
    ("Non-responders", 0, 45),
    ("Data-limited", 45, 56),
    ("Learning", 56, 70),
    ("Strong", 70, 101),
)
