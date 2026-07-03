#!/usr/bin/env python3
"""
Compare baseline runs across all N values and produce paper figures.

Loads per-subject result JSONs from models/checkpoints/ and produces:
  fig10_accuracy_vs_n.png       — main paper figure: accuracy vs extra samples
  fig11_tier_breakdown_vs_n.png — accuracy vs N broken down by subject tier

Usage:
    python eval/compare_baselines.py
    python eval/compare_baselines.py --checkpoint-dir models/checkpoints --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

CHECKPOINT_DIR = Path("models/checkpoints")
OUT_DIR = Path("reports/figures")

C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_BLUE   = "#378ADD"
C_GREEN  = "#639922"
C_PURPLE = "#534AB7"
C_TEAL   = "#1D9E75"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#aaaaaa",
})

# Tier boundaries (by vanilla accuracy, in percent)
TIERS = [
    ("Non-responders",  0,   45,  C_RED),
    ("Data-limited",   45,   56,  C_GRAY),
    ("Learning",       56,   70,  C_BLUE),
    ("Strong",         70,  101,  C_GREEN),
]


@dataclass
class RunResult:
    run_name: str
    n_extra: int
    subject_accs: dict[int, float]   # {subject_id: mean_accuracy_pct}

    @property
    def overall_mean(self) -> float:
        return float(np.mean(list(self.subject_accs.values())))

    @property
    def overall_std(self) -> float:
        return float(np.std(list(self.subject_accs.values())))


def _load_run(json_path: Path, run_name: str, n_extra: int) -> RunResult:
    data = json.loads(json_path.read_text())
    subject_accs = {entry["subject"]: entry["mean_accuracy"] * 100.0 for entry in data}
    return RunResult(run_name=run_name, n_extra=n_extra, subject_accs=subject_accs)


def load_all_results(checkpoint_dir: Path) -> dict[str, RunResult]:
    """Load all available per-subject JSON files. Missing files are silently skipped."""
    candidates = [
        ("vanilla",    checkpoint_dir / "per_subject_results_vanilla.json",       0),
        ("aug_online", checkpoint_dir / "per_subject_results_augmented.json",      0),
        ("aug_1x",     checkpoint_dir / "per_subject_results_augmented_1x.json",  30),
        ("aug_2x",     checkpoint_dir / "per_subject_results_augmented_2x.json",  60),
        ("aug_3x",     checkpoint_dir / "per_subject_results_augmented_3x.json",  90),
    ]
    results: dict[str, RunResult] = {}
    for name, path, n_extra in candidates:
        if path.exists():
            results[name] = _load_run(path, name, n_extra)
            r = results[name]
            print(f"  Loaded {name:12s}: {len(r.subject_accs)} subjects, "
                  f"mean={r.overall_mean:.1f}% ± {r.overall_std:.1f}%")
        else:
            print(f"  Skipping {name:12s}: {path.name} not found")
    return results


def _wilcoxon_vs_vanilla(vanilla: RunResult, other: RunResult) -> float:
    """One-sided Wilcoxon signed-rank test: is other > vanilla? Returns p-value."""
    shared = sorted(set(vanilla.subject_accs) & set(other.subject_accs))
    if len(shared) < 10:
        return float("nan")
    a = np.array([vanilla.subject_accs[s] for s in shared])
    b = np.array([other.subject_accs[s] for s in shared])
    if np.allclose(a, b):
        return 1.0
    _, p = stats.wilcoxon(b, a, alternative="greater")
    return float(p)


def print_summary_table(results: dict[str, RunResult]) -> None:
    """Print summary table with Δ and Wilcoxon p-values vs vanilla."""
    vanilla = results.get("vanilla")
    col = f"{'Run':<18} {'N extra':>8} {'Mean':>8} {'Std':>7} {'Δ vs vanilla':>14} {'p-value':>10}"
    sep = "=" * len(col)
    print(f"\n{sep}\n{col}\n{'-' * len(col)}")

    for key in ["vanilla", "aug_online", "aug_1x", "aug_2x", "aug_3x"]:
        if key not in results:
            continue
        r = results[key]
        delta_str, p_str = "—", "—"
        if vanilla and key != "vanilla":
            delta = r.overall_mean - vanilla.overall_mean
            delta_str = f"{delta:+.1f}%"
            p = _wilcoxon_vs_vanilla(vanilla, r)
            p_str = f"{p:.4f}" if not np.isnan(p) else "n/a"
        print(f"{r.run_name:<18} {r.n_extra:>8} {r.overall_mean:>7.1f}% "
              f"{r.overall_std:>6.1f}% {delta_str:>14} {p_str:>10}")

    print(f"{sep}\n")


def plot_accuracy_vs_n(results: dict[str, RunResult], out: Path) -> None:
    """
    Figure 10: accuracy vs N extra training samples.
    Vanilla at N=0, aug_online as a diamond reference at N=0,
    aug_offline at N=30/60/90 connected by a line.
    TSTR placeholder lane added for visual context.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    vanilla = results.get("vanilla")

    # --- Offline augmented curve (including vanilla N=0 anchor) ---
    offline_keys = [k for k in ("aug_1x", "aug_2x", "aug_3x") if k in results]
    if offline_keys:
        ns   = [0] + [results[k].n_extra for k in offline_keys] if vanilla else [results[k].n_extra for k in offline_keys]
        means = ([vanilla.overall_mean] if vanilla else []) + [results[k].overall_mean for k in offline_keys]
        stds  = ([vanilla.overall_std]  if vanilla else []) + [results[k].overall_std  for k in offline_keys]

        ax.plot(ns, means, "o-", color=C_BLUE, linewidth=2.2, markersize=7,
                label="Additive augmentation", zorder=3)
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=C_BLUE)
        for k in offline_keys:
            r = results[k]
            ax.annotate(f"{r.overall_mean:.1f}%",
                        xy=(r.n_extra, r.overall_mean),
                        xytext=(r.n_extra + 2, r.overall_mean + 1.0),
                        fontsize=8.5, color=C_BLUE)

    # --- Vanilla reference line ---
    if vanilla:
        ax.axhline(vanilla.overall_mean, color=C_GRAY, linestyle="--", linewidth=1.5, alpha=0.85,
                   label=f"Vanilla baseline — {vanilla.overall_mean:.1f}% (N=0)")

    # --- TSTR placeholder lane ---
    ax.plot([], [], "s--", color=C_GREEN, linewidth=2, markersize=7,
            label="TSTR / synthetic data (Phase 5 — pending)")

    # --- Chance line ---
    ax.axhline(50, color=C_RED, linestyle=":", linewidth=1, alpha=0.5, label="Chance (50%)")

    ax.set_xlabel("N extra training samples per fold", fontsize=11)
    ax.set_ylabel("Mean accuracy (%) ± std across subjects", fontsize=11)
    ax.set_title("Figure 10 — Accuracy vs. Additional Training Samples\n"
                 "Additive augmented copies vs. vanilla baseline", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.92, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    x_max = max(90, max((r.n_extra for r in results.values()), default=0)) + 10
    ax.set_xlim(-5, x_max)
    ax.set_ylim(48, 78)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_tier_breakdown_vs_n(
    results: dict[str, RunResult],
    vanilla: RunResult,
    out: Path,
) -> None:
    """
    Figure 11: 4-panel accuracy vs N, one panel per subject tier.
    Tier assignment is fixed from vanilla accuracy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    offline_keys = [k for k in ("aug_1x", "aug_2x", "aug_3x") if k in results]

    for panel_idx, (tier_name, lo, hi, color) in enumerate(TIERS):
        ax = axes_flat[panel_idx]
        tier_subjects = [s for s, acc in vanilla.subject_accs.items() if lo <= acc < hi]
        n_tier = len(tier_subjects)

        def tier_mean(r: RunResult) -> float:
            vals = [r.subject_accs[s] for s in tier_subjects if s in r.subject_accs]
            return float(np.mean(vals)) if vals else float("nan")

        def tier_std(r: RunResult) -> float:
            vals = [r.subject_accs[s] for s in tier_subjects if s in r.subject_accs]
            return float(np.std(vals)) if vals else float("nan")

        # Offline augmented curve anchored at vanilla N=0
        if offline_keys:
            ns    = [0] + [results[k].n_extra for k in offline_keys]
            means = [tier_mean(vanilla)] + [tier_mean(results[k]) for k in offline_keys]
            stds  = [tier_std(vanilla)]  + [tier_std(results[k])  for k in offline_keys]
            valid = [(n, m, s) for n, m, s in zip(ns, means, stds) if not np.isnan(m)]
            if valid:
                vns, vms, vss = zip(*valid)
                ax.plot(vns, vms, "o-", color=color, linewidth=2, markersize=6,
                        label="Additive augmentation")
                ax.fill_between(vns,
                                [m - s for m, s in zip(vms, vss)],
                                [m + s for m, s in zip(vms, vss)],
                                alpha=0.15, color=color)

        ax.axhline(50, color=C_RED, linestyle=":", linewidth=0.8, alpha=0.45)
        ax.axhline(tier_mean(vanilla), color=color, linestyle="--", linewidth=1,
                   alpha=0.5, label="Vanilla")

        ax.set_title(f"{tier_name}  (n={n_tier},  vanilla {lo}–{hi}%)",
                     fontsize=10, color=color, fontweight="bold")
        ax.set_xlabel("N extra samples")
        ax.set_ylabel("Mean accuracy (%)")
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(axis="y", alpha=0.25)

        x_max = max(90, max((r.n_extra for r in results.values()), default=0)) + 10
        ax.set_xlim(-5, x_max)

    fig.suptitle("Figure 11 — Accuracy vs. N Extra Samples by Subject Tier\n"
                 "(tier assigned from vanilla accuracy)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def save_delta_csv(results: dict[str, RunResult], out_dir: Path) -> None:
    """Save per-subject accuracy and deltas vs vanilla for all offline N values."""
    vanilla = results.get("vanilla")
    if vanilla is None:
        print("  delta CSV skipped: vanilla results not available")
        return

    def get_tier(acc: float) -> str:
        for name, lo, hi, _ in TIERS:
            if lo <= acc < hi:
                return name
        return "Unknown"

    offline_keys = [k for k in ("aug_1x", "aug_2x", "aug_3x") if k in results]
    out = out_dir / "subject_deltas.csv"
    fieldnames = ["subject", "vanilla_acc", "tier"] + [
        col for k in offline_keys for col in (f"{k}_acc", f"{k}_delta")
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(vanilla.subject_accs):
            v_acc = vanilla.subject_accs[s]
            row: dict = {
                "subject": s,
                "vanilla_acc": round(v_acc, 1),
                "tier": get_tier(v_acc),
            }
            for k in offline_keys:
                if s in results[k].subject_accs:
                    acc = results[k].subject_accs[s]
                    row[f"{k}_acc"] = round(acc, 1)
                    row[f"{k}_delta"] = round(acc - v_acc, 1)
                else:
                    row[f"{k}_acc"] = ""
                    row[f"{k}_delta"] = ""
            writer.writerow(row)
    print(f"  Saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-subject results...")
    results = load_all_results(args.checkpoint_dir)

    if not results:
        print("No result files found. Run train_baseline.py first.")
        return

    print_summary_table(results)
    save_delta_csv(results, args.out_dir)

    if len(results) < 2:
        print("Need at least 2 runs to plot. Exiting.")
        return

    plot_accuracy_vs_n(results, args.out_dir / "fig10_accuracy_vs_n.png")

    if "vanilla" in results:
        plot_tier_breakdown_vs_n(
            results,
            results["vanilla"],
            args.out_dir / "fig11_tier_breakdown_vs_n.png",
        )


if __name__ == "__main__":
    main()
