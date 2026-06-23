# NeuroSynth - `ml/` (Phase 0: Research Validation)

This is the Python-only research repo for the go/no-go gate described in
`specs/2026_06_22_neurosynth_synthetic_eeg_data_platform_project_spec_implementation.md`,
Section 9 ("Phase 0") and Section 13 ("First Steps for the Implementing Agent").

**Nothing here depends on the eventual Java backend.** Do not build
billing/dashboards/APIs until Phase 0 produces a go decision on RR-3
(does synthetic-augmented training beat the standard-augmentation
baseline by a meaningful margin?).

## Setup

```bash
cd /Users/valeriesalivon/Desktop/Study/Projects/Synthetic_Brain/NeuroSynth
python3 -m venv venv && source venv/bin/activate   # or use conda
pip install -r requirements.txt
python environment_check.py
```

`environment_check.py` verifies all dependencies import correctly and
reports whether a CUDA GPU or Apple Silicon MPS device is available.
Diffusion model training will be impractically slow on CPU only - if no
GPU is available locally, plan to use Modal/RunPod/Colab per Section 6.6
of the spec for the training step.

## Status

| Step (spec Section 13)                                | Status |
|-------------------------------------------------------|---|
| 1. NeuroSynth/ project + environment                  | Done - structure + requirements.txt + environment_check.py |
| 2. EEGBCI load + preprocessing/epoching pipeline      | Done - `data/eegbci_pipeline.py` |
| 3. EEGNet classifier + augmentation baseline          | Not started |
| 4. 1D U-Net diffusion model, class-conditional        | Not started |
| 5. TSTR harness + fidelity metrics -> go/no-go report | Not started |
| 6. Human review of go/no-go                           | Blocked on 3-5 |

## Layout

```
NeuroSynth/
     data/
       eegbci_pipeline.py   # download, filter, epoch, normalize EEGBCI -> .npz
     models/                # (next) EEGNet + 1D U-Net diffusion model
     eval/                  # (next) augmentation baseline, TSTR harness, fidelity metrics
     workers/               # (Phase 1+, not yet) queue-consuming generation/format/validation workers
     api/                   # (Phase 1+, not yet) minimal FastAPI service
     requirements.txt
     environment_check.py
```

## Data pipeline (`data/eegbci_pipeline.py`)

Downloads the PhysioNet EEGBCI motor-imagery runs via MNE, band-pass
filters to the mu/beta band (7-30 Hz), epochs around the T1/T2 event
markers, and per-epoch z-score normalizes. Produces a binary
classification dataset (`X`: epochs x channels x time, `y`: 0/1 labels).

```bash
python data/eegbci_pipeline.py --subjects 1 2 3 --task imagery_lr --out processed/eegbci.npz
```

- `--task imagery_lr`: left fist vs. right fist (runs 4, 8, 12)
- `--task imagery_ff`: both fists vs. both feet (runs 6, 10, 14)

First run downloads via `mne.datasets.eegbci` (cached under
`~/mne_data` afterward). Start with 3-5 subjects to validate the
pipeline before scaling to all 109.

**Note on testing:** the pipeline logic that doesn't require MNE
(normalization, dataset save/load round-trip) has been unit-verified.
The MNE-dependent download/filter/epoch path follows the standard
MNE EEGBCI motor-imagery tutorial pattern but has not yet been run
end-to-end in this environment (no network/MNE install available here) -
run it locally and sanity-check the printed shapes/class balance before
moving on to the classifier.

## Next steps

1. Run the pipeline locally for a handful of subjects; eyeball PSD plots
   and epoch counts to confirm preprocessing looks sane.
2. Implement EEGNet (or ShallowConvNet) in `models/` as the downstream
   evaluator.
3. Implement the standard-augmentation baseline (noise injection,
   time-shift, channel dropout, sliding-window crops) in `eval/`.
4. Implement the 1D U-Net denoising diffusion model in `models/`; train
   class-conditional generation on EEGBCI.
5. Build the TSTR harness + distributional fidelity metrics in `eval/`;
   produce the go/no-go report (RR-1 through RR-5).
6. Pause for human review of the go/no-go results before touching
   Phase 1 (preprint, minimal API) or anything Java-related.
