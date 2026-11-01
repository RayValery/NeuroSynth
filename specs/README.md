# Design specs

Each experiment was specified **before** it was run: the hypothesis, the design, the leakage
argument, the success criterion and the decision rule. The specs are kept unedited (apart from
`iter_01`, trimmed to its research content, and minor formatting) as a record of what was planned versus what was found.
The corresponding results are in `reports/`.

File paths in the specs refer to the layout at the time they were written: `eval/*.py` is now
`src/neurosynth/*.py` (command-line entry points in `scripts/`), `data/eegbci_pipeline.py` is now
`src/neurosynth/data.py`, and the Modal launchers are in `cloud/`.

| Date | Spec | What it set up | Outcome |
|---|---|---|---|
| 22 Jun | `iter_01/…initial-plan` | Research question, pre-registered success criterion RR-3, datasets, Phase 0 plan | — |
| 28 Jun | `iter_02/V01–V03` | EEGNet + cross-subject training, three training-loop fixes | ~50–52%: gradients cancel across subjects |
| 28 Jun | `iter_02/V04` | Euclidean alignment for cross-subject transfer | Hurt; removes the spatial signal |
| 28 Jun | `iter_02/V05–V06` | Switch to within-subject leave-one-run-out; fix early-stopping/BatchNorm failure | Vanilla baseline 56.7% |
| 28–29 Jun | `iter_03/V07–V08` | Augmented baseline; **additive, sample-matched design** so augmentation and synthetic get the same N | Augmentation ceiling 59.5% at 2× |
| 3 Jul | `iter_04/step4_eeg-diffusion-model` | 1D U-Net DDPM, **run-holdout cross-fitting** (3 models) + memorization guard | Healthy, non-memorizing generators |
| 3 Jul | `iter_04/step4-probe_pooled-real-augmentation` | Cheap real-data probe of population vs subject-specific data | Not run — the diffusion route was taken directly (listed as future work) |
| 7 Jul | `iter_05/step5_tstr-go-no-go-gate` | TSTR harness, routing assertion, gate statistics | RR-3 failed (54.7%) |
| 10 Jul | `pivot_iter_01/…execution-conditioned…` | Test "imagery is too weak a teacher": add motor-execution data to the generator | No lift (+0.01) |
| 18 Jul | `pivot_iter_02/…subject-conditional…` | Per-subject embedding as a residual over a population fallback; two-key leakage routing | Best synthetic arm (56.1%), still below augmentation |

Reading tip: most specs have a "Non-Goals" section listing the alternatives that were
considered and why they were deferred.
