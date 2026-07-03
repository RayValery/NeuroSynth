# NeuroSynth — Phase 0: Research Validation

Python-only research repo for the go/no-go gate: does training on synthetic EEG data beat standard signal augmentation for motor imagery classification?

Nothing here depends on the eventual backend. Do not build billing, dashboards, or APIs until Phase 0 produces a go decision on RR-3 (synthetic-augmented training beats the standard-augmentation baseline by a meaningful margin).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python environment_check.py
```

`environment_check.py` verifies all dependencies and reports whether CUDA or Apple Silicon MPS is available. Diffusion model training (Step 4) will be impractically slow on CPU only — plan to use Modal/RunPod/Colab if no GPU is available locally.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Environment setup | ✅ Done — `.venv`, `requirements.txt`, `environment_check.py` |
| 2 | EEGBCI preprocessing pipeline | ✅ Done — `data/eegbci_pipeline.py`, output: `data/processed.npz` |
| 3a | EEGNet vanilla baseline | ✅ Done — **56.7% ± 10.8%** within-subject LORO |
| 3b | EEGNet augmented baseline | ✅ Done — additive offline aug, best: **59.5% ± 11.2%** at 2× |
| 4 | 1D U-Net diffusion model (class-conditional) | Not started |
| 5 | TSTR harness + fidelity metrics → go/no-go report | Not started |
| 6 | Human review of go/no-go | Blocked on 3–5 |

## Baseline Results (Step 3a)

Within-subject Leave-One-Run-Out (LORO) evaluation on 106 subjects, imagery runs only (4, 8, 12):

| Run | Accuracy | Notes |
|---|---|---|
| Vanilla (real data only) | **56.7% ± 10.8%** | MLflow `fe46e741342f4cf1b7ae2645d224e0c1` |
| Aug offline 1× (real + 30 aug copies) | 58.6% ± 9.1% | MLflow `3bdca79ef06e482ea26c864e4e4ddf98` |
| Aug offline 2× (real + 60 aug copies) | **59.5% ± 11.2%** | Peak — primary TSTR target, MLflow `3c78eda0974042608a6636b863e8805b` |
| Aug offline 3× (real + 90 aug copies) | 59.0% ± 10.0% | MLflow `02b9a34efc764e0a8f5307708dcd0bd3` |
| TSTR (real + synthetic) | pending | go/no-go: must beat **59.5%** |

Subject distribution: ~13% non-responders (<45%), ~39% data-limited (45–56%), ~40% learning well (56–70%), ~8% strong responders (>70%). Full analysis in [`reports/phase0_baseline_report.md`](reports/phase0_baseline_report).

## Layout

```
NeuroSynth/
├── data/
│   ├── eegbci_pipeline.py      # download → bandpass → epoch → z-score → .npz
│   └── processed.npz           # generated, gitignored — rebuild with pipeline
├── eval/
│   ├── eegnet.py               # EEGNet architecture (Lawhern et al. 2018, 2,930 params)
│   ├── augmentations.py        # gaussian noise, time-shift, channel dropout, crop
│   ├── dataset.py              # EEGTorchDataset, within_subject_loro_split
│   └── train_baseline.py       # LORO training loop + MLflow logging
├── models/
│   └── checkpoints/            # saved .pt files, gitignored — rebuild by rerunning training
├── specs/                      # experiment design documents
├── reports/
│   ├── phase0_baseline_report.md   # full experiment log with analysis
│   ├── generate_figures.py          # regenerate all report figures
│   └── figures/                     # PNG figures referenced by the report
├── mlruns/                     # MLflow experiment tracking (kept in git as experiment ledger)
├── workers/                    # Phase 1+, not yet
├── api/                        # Phase 1+, not yet
├── requirements.txt
└── environment_check.py
```

## Data Pipeline

Downloads PhysioNet EEGBCI motor imagery via MNE, band-pass filters to 7–30 Hz (mu/beta), epochs around T1/T2 event markers (−1 s to +4 s), and applies per-epoch per-channel z-score normalization. Saves `X` (epochs × channels × time), `y` (0/1 labels), `subject_ids`, and `run_ids`.

```bash
# Full dataset — all 109 subjects, imagery_lr task
source .venv/bin/activate
python data/eegbci_pipeline.py --subjects $(seq 1 109) --task imagery_lr --out data/processed.npz
```

- `--task imagery_lr`: left fist vs. right fist (runs 4, 8, 12) — used for all baselines
- `--task imagery_ff`: both fists vs. both feet (runs 6, 10, 14) — alternative task

Subjects 88, 92, 100 are automatically dropped due to non-standard epoch lengths. Final output: 4,748 epochs, 106 subjects, shape `(4748, 64, 801)`.

## Running the Baselines

```bash
source .venv/bin/activate

# Vanilla baseline (real data only)
python eval/train_baseline.py --data data/processed.npz --run vanilla

# Augmented baseline — additive offline copies (recommended for TSTR comparison)
python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 1
python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 2
python eval/train_baseline.py --data data/processed.npz --run augmented_offline --n-copies 3

# Compare all N values, generate fig10/fig11, save subject_deltas.csv
python eval/compare_baselines.py

# View results
mlflow ui   # → http://localhost:5000, experiment: phase0-baseline
```

## Next Steps

1. ~~Run augmented baseline~~ — done. Ceiling: 59.5% ± 11.2% at 2×.
2. Train 1D U-Net diffusion model on `data/processed.npz` (class-conditional generation).
3. TSTR evaluation: for each LORO fold, add N synthetic epochs to the real training set and compare accuracy against the 59.5% augmented ceiling at matched N values.
4. Human review of go/no-go results before any Phase 1 work.
