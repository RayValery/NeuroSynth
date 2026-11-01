# NeuroSynth — Initial Research Plan
## Synthetic EEG for few-shot motor-imagery calibration

> **Document purpose:** The research plan written at the start of the project (2026-06-22), before any experiment was run. It defines the research question, the pre-registered success criterion (RR-3), the datasets, the validation methodology and the Phase 0 work plan. Later specs in `specs/` refine each step before it was executed.
>
> *Editorial note (2026-10-31): the original planning document also covered personal goals, market analysis, pricing and a product/backend architecture. Those sections are not part of the research and have been removed from this repository; the research sections below are unchanged except for wording that referred to them.*

---

## 1. Motivation

Neural data is expensive and scarce: collecting a labeled EEG dataset requires ethics approval, participant time and lab equipment. Brain-computer interfaces (BCIs) need a per-user classifier, but a new user typically provides only a few minutes of calibration data. Generative models (here, diffusion models) can produce unlimited synthetic EEG — **if** that synthetic data improves a downstream classifier more than trivial signal augmentation does.

---

## 2. The critical question (must be addressed first)
> **THE go/no-go question:** Does training on (synthetic EEG + real EEG) meaningfully outperform training on (real EEG + standard augmentation: noise injection, time-shift, channel dropout, sliding window)?

- If **yes** → synthetic EEG is a practical tool for few-shot BCI calibration.
- If **no/marginal** → a well-controlled negative result that explains why.

Literature is **encouraging but not conclusive**. This must be validated empirically first; it drives the implementation order below.

---

## 3. Research / Validation Requirements (gate the whole project)
| ID | Requirement |
|----|-------------|
| RR-1 | Implement TSTR (Train-on-Synthetic, Test-on-Real) benchmark harness |
| RR-2 | Implement baseline: standard augmentation (noise, time-shift, channel dropout, sliding window) |
| RR-3 | Synthetic-augmented training must beat standard-augmentation baseline on a real test set (target: meaningful accuracy/F1 improvement, e.g. >=5-10% relative) |
| RR-4 | Compute distributional fidelity metrics: PSD divergence, ERP preservation, inter-channel correlation, spectral entropy |
| RR-5 | Results reproducible and documented for preprint |

---

## 4. Datasets (open, for training + validation)
| Dataset | Use | Notes |
|---------|-----|-------|
| **PhysioNet EEG Motor Movement/Imagery (EEGBCI)** | Primary training (Phase 0) | 109 subjects, 64ch, 160Hz, motor imagery; via MNE `eegbci` loader |
| BCI Competition IV (2a/2b) | Benchmark / TSTR | Standard motor-imagery benchmark |
| DEAP | Emotion paradigm (later) | 32 subjects |
| Temple University EEG Corpus (TUH) | Scale-up (later) | 30,000+ sessions |
| OpenNeuro.org (BIDS) | fMRI/EEG expansion (later) | Hundreds of datasets |
| Sleep-EDF (PhysioNet) | Sleep staging (later) | Sleep paradigm |

---

## 5. Validation Methodology (for RR-* and the paper)
1. **TSTR (Train-on-Synthetic, Test-on-Real):** Train a standard EEG classifier (e.g., EEGNet) on synthetic data, test on held-out real data. Compare to TRTR (train-real-test-real) and to standard-augmentation baseline.
2. **Baseline to beat:** real data + standard augmentation (Gaussian noise, time-shift, channel dropout, sliding-window crops).
3. **Distributional fidelity:** Power Spectral Density (Welch) divergence, ERP preservation, inter-channel correlation matrices, spectral entropy, t-SNE/UMAP of real vs synthetic embeddings.
4. **Downstream gain metric:** classification accuracy / F1 uplift when synthetic data augments a small real training set (the practically relevant scenario: few-shot per-subject calibration).
5. All experiments seeded, tracked (MLflow/W&B), and written up for the preprint.

---

## 6. Implementation Plan


### Phase 0 — Research validation & go/no-go (weeks 1–4)
- [ ] Set up Python ML repo + environment (PyTorch, MNE-Python, NumPy, SciPy, MLflow/W&B).
- [ ] Download + explore EEGBCI (motor imagery, 64ch, 160Hz). Build preprocessing pipeline (filtering, epoching, normalization).
- [ ] Implement EEGNet (or ShallowConvNet) classifier as the downstream evaluator.
- [ ] Implement standard-augmentation **baseline** (RR-2).
- [ ] Implement **1D U-Net denoising diffusion model** for class-conditional EEG generation.
- [ ] Train diffusion model on EEGBCI.
- [ ] Build **TSTR benchmark harness** (RR-1) + distributional fidelity metrics (RR-4).
- [ ] **GO/NO-GO DECISION (RR-3):** Does synthetic-augmented training beat the augmentation baseline by a meaningful margin?


> *Outcome (added 2026-10-31):* Phase 0 was completed and extended with two follow-up experiments. RR-3 was not met by any synthetic arm. See the top-level `README.md`.
