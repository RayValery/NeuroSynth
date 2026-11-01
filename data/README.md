# Data

**PhysioNet EEG Motor Movement/Imagery Dataset** (Schalk et al. 2004; Goldberger et al. 2000),
available at [physionet.org/content/eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/).
The raw recordings are not stored in this repository. `scripts/build_dataset.py` downloads them
through MNE (`mne.datasets.eegbci`, cached under `~/mne_data`) and writes the preprocessed arrays
here.

## Recordings

109 subjects, 64 EEG channels (10-10 montage), 160 Hz, 14 runs each:

| Runs | Task | Used here |
|---|---|---|
| 1, 2 | resting baseline, eyes open / closed | no |
| 3, 7, 11 | **executed** left- vs right-fist movement | generator training data in one arm only |
| 4, 8, 12 | **imagined** left- vs right-fist movement | classifier training and testing (all arms) |
| 5, 9, 13 | executed both fists vs both feet | no |
| 6, 10, 14 | imagined both fists vs both feet | no (available as `--task imagery_ff`) |

Event codes: T0 = rest (dropped), T1 = left fist, T2 = right fist. Each run has about 15 T1/T2 trials.

## Preprocessing (`src/neurosynth/data.py`)

1. Load each run, standardize channel names, set the `standard_1005` montage.
2. Band-pass 7–30 Hz (FIR), covering the mu and beta rhythms that carry motor-imagery signal.
3. Epoch each T1/T2 event from −1 s to +4 s, no baseline correction: 801 samples per epoch.
4. Z-score every epoch per channel over time.
5. Drop subjects 88, 92 and 100, whose epochs have a non-standard length.

Euclidean alignment is implemented but **not** applied: it removed the within-subject spatial signal
(see the baseline report, §5.4).

## Built files (gitignored)

| File | Command | Contents |
|---|---|---|
| `processed.npz` | `python scripts/build_dataset.py --subjects $(seq 1 109) --task imagery_lr --out data/processed.npz` | Imagery runs 4/8/12: `X` (4748, 64, 801) float32, `y` (0 = left, 1 = right), `subject_ids`, `run_ids`, `ch_names`, `sfreq`; ~0.9 GB |
| `processed_with_exec.npz` | `python scripts/build_dataset.py --subjects $(seq 1 109) --task exec_imagery_lr --out data/processed_with_exec.npz` | Imagery + execution runs (~9.5k epochs) with an extra `domains` array (0 = imagery, 1 = execution); ~1.7 GB |

Classes are nearly balanced (2,396 left / 2,352 right in `processed.npz`).

`python scripts/inspect_data.py` prints shapes and class balance and plots the average spectrum, as
a quick check of a freshly built file.

## References

- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. & Wolpaw, J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering* 51(6), 1034–1043.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220.
