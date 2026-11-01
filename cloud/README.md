# Cloud launchers (Modal)

Thin wrappers that run the training code in `src/neurosynth/` on [Modal](https://modal.com) GPUs (A10G).
They contain no model logic — each one imports and calls `neurosynth.train_diffusion.train` or
`neurosynth.train_baseline.train` inside a container.

**Run them from the repository root** (they ship `src/neurosynth/` into the container by relative path):

| Launcher | What it runs | Report |
|---|---|---|
| `modal_train_diffusion.py` | 3 population DDPMs, one per held-out imagery run | `reports/phase0_baseline_report.md` §9 |
| `modal_train_tstr.py` | TSTR passes (`--variant imagery / exec / subj`) | all three TSTR reports |
| `modal_train_diffusion_exec.py` | 3 execution-enriched, domain-conditioned DDPMs | `reports/pivot_iter01_exec_conditioned_tstr_report.md` |
| `modal_train_diffusion_subj.py` | 3 subject-conditional DDPMs + per-subject synthetic banks | `reports/pivot_iter02_subject_conditional_tstr_report.md` |

```bash
modal run --detach cloud/modal_train_diffusion.py
modal run --detach cloud/modal_train_tstr.py --variant imagery
```

Data and checkpoints live on two Modal volumes (`neurosynth-data`, `neurosynth-checkpoints`);
see each launcher's docstring for upload/download commands.
