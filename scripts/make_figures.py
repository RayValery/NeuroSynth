"""Rebuild every figure and summary table from results/per_subject/*.json.

  figures/                README figures (overview, paired differences, accuracy vs extra
                          samples, tier selection check)
  results/*.csv           summary tables behind every number in the README
  reports/figures/        research-log figures (fig1–fig14)

The output is deterministic: with the pinned versions in requirements.txt, rerunning
produces byte-identical files. The three
reports/figures/diffusion_fidelity_run*.png plots are the exception; they are written
by the generator training itself (scripts/train_generator.py), not from results/.

Usage (from the repository root):
    python scripts/make_figures.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from neurosynth import config

SCRIPTS = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / script)], check=True, cwd=config.ROOT)


def main() -> None:
    run("analyze_results.py")    # results/*.csv + README figures
    run("compare_baselines.py")  # gate verdicts, results/subject_deltas.csv, fig10–fig14
    run("report_figures.py")     # fig1–fig9

    # README copy of the accuracy-vs-N figure
    sys.path.insert(0, str(SCRIPTS))
    from compare_baselines import load_all_results, plot_accuracy_vs_n

    plot_accuracy_vs_n(load_all_results(config.PER_SUBJECT_DIR),
                       config.FIGURES_DIR / "accuracy_vs_extra_samples.png")


if __name__ == "__main__":
    main()
