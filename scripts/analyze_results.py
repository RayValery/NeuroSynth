#!/usr/bin/env python3
"""
Summary tables and README figures, computed from the per-subject results.

Every comparison is paired over the same 106 subjects (each subject's accuracy is
the mean over its 3 leave-one-run-out folds). For each pair we report:
  - mean paired difference (percentage points)
  - 95% bootstrap CI (10,000 resamples of subjects, percentile method, seed 0)
  - two-sided Wilcoxon signed-rank p-value (zero differences dropped)
  - number of subjects better / worse

Inputs:   results/per_subject/*.json
Outputs:  results/summary.csv               one row per arm: accuracy, Δ vs real-only and vs augmentation
          results/paired_comparisons.csv    all paired comparisons between arms
          results/tier_comparisons.csv      within-tier comparisons between arms
          results/tier_selection_check.csv  regression-to-the-mean check on the tier analysis
          figures/overview.png              README headline figure
          figures/paired_differences.png    paired differences with CIs
          figures/tier_selection_check.png  regression-to-the-mean check

Usage (from the repository root):
    python scripts/analyze_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from neurosynth import config

N_BOOT = 10_000
SEED = 0

# arm key -> (file stem, display label)
ARMS = {
    "vanilla": ("per_subject_results_vanilla", "Real data only"),
    "aug_1x": ("per_subject_results_augmented_1x", "Signal augmentation 1×"),
    "aug_2x": ("per_subject_results_augmented_2x", "Signal augmentation"),
    "aug_3x": ("per_subject_results_augmented_3x", "Signal augmentation 3×"),
    "pop_2x": ("per_subject_results_tstr_2x", "Population generator (imagery)"),
    "exec_2x": ("per_subject_results_tstr_exec_2x", "Population generator (+ execution)"),
    "subj_2x": ("per_subject_results_tstr_subj_2x", "Subject-conditional generator"),
    "subjg_2x": ("per_subject_results_tstr_subj_g1.5_2x", "Subject-conditional + guidance 1.5"),
}

# (a, b) pairs: difference is a − b
COMPARISONS = [
    ("aug_2x", "vanilla"),
    ("pop_2x", "vanilla"),
    ("exec_2x", "vanilla"),
    ("subj_2x", "vanilla"),
    ("subjg_2x", "vanilla"),
    ("pop_2x", "aug_2x"),
    ("exec_2x", "aug_2x"),
    ("subj_2x", "aug_2x"),
    ("subjg_2x", "aug_2x"),
    ("exec_2x", "pop_2x"),
    ("subj_2x", "pop_2x"),
    ("subj_2x", "exec_2x"),
    ("subjg_2x", "subj_2x"),
]

TIERS = [("Non-responders", 0, 45), ("Data-limited", 45, 56), ("Learning", 56, 70), ("Strong", 70, 101)]

C_AUG, C_POP, C_EXEC, C_SUBJ, C_GUID, C_GRAY = "#378ADD", "#639922", "#1D9E75", "#D9822B", "#B5651D", "#888780"


def load(results_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    accs: dict[str, dict[int, float]] = {}
    for key, (stem, _) in ARMS.items():
        path = results_dir / f"{stem}.json"
        if not path.exists():
            print(f"  skip {key}: {path} not found")
            continue
        accs[key] = {e["subject"]: 100.0 * e["mean_accuracy"] for e in json.loads(path.read_text())}
    subjects = np.array(sorted(set.intersection(*(set(v) for v in accs.values()))))
    return subjects, {k: np.array([v[s] for s in subjects]) for k, v in accs.items()}


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else 1.0
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wilcoxon_p_two_sided": p,
        "n_better": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()),
        "n": int(len(d)),
    }


def tier_of(v: float) -> str:
    for name, lo, hi in TIERS:
        if lo <= v < hi:
            return name
    return TIERS[-1][0]


def fig15(rows: dict, data: dict, out: Path) -> None:
    order = [("aug_2x", C_AUG), ("pop_2x", C_POP), ("exec_2x", C_EXEC), ("subj_2x", C_SUBJ), ("subjg_2x", C_GUID)]
    order = [(k, c) for k, c in order if k in data]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    panels = [("vanilla", "vs. real data only"), ("aug_2x", "vs. signal augmentation (success criterion)")]
    for ax, (ref, title) in zip(axes, panels):
        for i, (k, c) in enumerate(order):
            y = len(order) - 1 - i
            if k == ref:
                ax.text(0, y, "reference", ha="center", va="center", fontsize=8.5, color=C_GRAY)
                continue
            r = rows[(k, ref)]
            ax.errorbar(r["mean_diff"], y, xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                        fmt="o", color=c, ecolor=c, elinewidth=2, capsize=4, markersize=7)
            ax.text(r["ci_high"] + 0.25, y, f"{r['mean_diff']:+.1f}  [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]",
                    va="center", fontsize=8, color="#333333")
        ax.axvline(0, color=C_GRAY, linewidth=1)
        if ref == "aug_2x":
            ax.axvspan(1.0, 9, color="#639922", alpha=0.08)
            ax.text(1.2, -0.45, "pre-registered\nwin region (> +1 pt)", fontsize=7.5, color="#3B6D11", va="bottom")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Paired difference, percentage points (95% bootstrap CI)")
        ax.set_xlim(-8, 8)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([ARMS[k][1] for k, _ in reversed(order)], fontsize=9)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    fig.suptitle("Each arm adds 60 extra training samples per fold (2×) · 106 subjects, paired", fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def fig16(data: dict, out: Path) -> list[dict]:
    """Regression-to-the-mean check: the same tier pattern appears in either direction."""
    v, a1 = data["vanilla"], data["aug_1x"]
    names = [t[0] for t in TIERS]
    forward, backward, rows = [], [], []
    for name in names:
        m_f = np.array([tier_of(x) == name for x in v])
        m_b = np.array([tier_of(x) == name for x in a1])
        f = float((a1 - v)[m_f].mean()) if m_f.any() else np.nan
        b = float((v - a1)[m_b].mean()) if m_b.any() else np.nan
        forward.append(f)
        backward.append(b)
        rows.append({"tier": name, "n_tiered_by_real_only": int(m_f.sum()), "aug1x_minus_real_tiered_by_real": f,
                     "n_tiered_by_aug1x": int(m_b.sum()), "real_minus_aug1x_tiered_by_aug1x": b})
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, forward, 0.38, color=C_AUG, label="aug 1× − real-only   (tiers defined on real-only)")
    ax.bar(x + 0.2, backward, 0.38, color=C_GRAY, label="real-only − aug 1×   (tiers defined on aug 1×)")
    for xi, f, b in zip(x, forward, backward):
        ax.text(xi - 0.2, f + (0.4 if f >= 0 else -0.4), f"{f:+.1f}", ha="center", va="bottom" if f >= 0 else "top", fontsize=8)
        ax.text(xi + 0.2, b + (0.4 if b >= 0 else -0.4), f"{b:+.1f}", ha="center", va="bottom" if b >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    lim = np.nanmax(np.abs(forward + backward)) + 2.5
    ax.set_ylim(-lim, lim)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in names])
    ax.set_ylabel("Mean change in tier (pp)")
    ax.set_title("Tiering on a noisy score creates the 'helps weak / hurts strong' pattern by itself\n"
                 "(both directions show it → per-tier 'Δ vs. the tiering run' is regression to the mean)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")
    return rows


def fig_overview(data: dict, rows: dict, out: Path) -> None:
    """README headline figure: plain accuracy per method, read without any background."""
    C_BASE, C_AUGM, C_GEN = "#898781", "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    items = [
        ("vanilla", "Real EEG only (no extra trials)", C_BASE),
        ("aug_2x", "+ augmented copies of the person's own trials", C_AUGM),
        (None, None, None),
        ("pop_2x", "+ generated: population model", C_GEN),
        ("exec_2x", "+ generated: population model, trained on more data", C_GEN),
        ("subj_2x", "+ generated: subject-specific model", C_GEN),
        ("subjg_2x", "+ generated: subject-specific model, with guidance", C_GEN),
    ]
    aug = float(data["aug_2x"].mean())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = 0.0
    ticks, labels = [], []
    for key, label, color in items:
        if key is None:
            y -= 0.45
            continue
        v = float(data[key].mean())
        ax.plot([50, v], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [y], s=90, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.text(v + 0.18, y, f"{v:.1f}%", va="center", ha="left", fontsize=9.5, color=INK)
        if key not in ("vanilla", "aug_2x"):
            d = f"{v - aug:+.1f}".replace("-", "\u2212")
            ax.text(60.4, y, f"{d} points vs. augmentation", va="center", ha="left", fontsize=9.5, color=INK2)
        ticks.append(y)
        labels.append(label)
        y -= 1.0
    ax.axvline(aug, ymax=0.86, color=C_AUGM, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(aug, 0.75, "score to beat", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(50, 64.5)
    ax.set_ylim(y + 0.4, 1.35)
    ax.set_xticks(range(50, 61, 2))
    ax.set_xticklabels([f"{t}%" for t in range(50, 61, 2)], fontsize=9, color=INK2)
    ax.set_xlabel("Accuracy on held-out real EEG (left vs. right hand imagery; 50% = guessing)",
                  fontsize=9.5, color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.text(0.01, 0.985, "No method using diffusion-generated EEG beat simple data augmentation",
             fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.01, 0.925, f"Average over {len(data['vanilla'])} people. Each person's classifier is trained on ~30 real trials "
             "plus 60 extra trials from the method shown.", fontsize=9.5, color=INK2, ha="left", va="top")
    r = rows[("subj_2x", "aug_2x")]
    m = lambda x: f"{x:+.1f}".replace("-", "\u2212")
    fig.text(0.01, 0.01, f"Best generated method vs. augmentation: {m(r['mean_diff'])} points "
             f"(95% CI {m(r['ci_low'])} to {m(r['ci_high'])}).",
             fontsize=8.5, color=INK2, ha="left", va="bottom")
    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out.name}")


# Every arm in results/per_subject/, for results/summary.csv: (file stem, family, extra samples per fold)
SUMMARY_ARMS = [
    ("vanilla", "Real data only", 0),
    ("augmented_1x", "Signal augmentation", 30),
    ("augmented_2x", "Signal augmentation", 60),
    ("augmented_3x", "Signal augmentation", 90),
    ("tstr_1x", "Population generator (imagery)", 30),
    ("tstr_2x", "Population generator (imagery)", 60),
    ("tstr_3x", "Population generator (imagery)", 90),
    ("tstr_exec_1x", "Population generator (+ execution)", 30),
    ("tstr_exec_2x", "Population generator (+ execution)", 60),
    ("tstr_exec_3x", "Population generator (+ execution)", 90),
    ("tstr_subj_1x", "Subject-conditional generator", 30),
    ("tstr_subj_2x", "Subject-conditional generator", 60),
    ("tstr_subj_3x", "Subject-conditional generator", 90),
    ("tstr_subj_g1.5_1x", "Subject-conditional + guidance 1.5", 30),
    ("tstr_subj_g1.5_2x", "Subject-conditional + guidance 1.5", 60),
    ("tstr_subj_g1.5_3x", "Subject-conditional + guidance 1.5", 90),
    ("augmented", "Online augmentation (reference only)", 0),
]


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else v


def write_summary(results_dir: Path, out: Path, rng: np.random.Generator) -> None:
    """One row per arm: mean accuracy, and paired Δ vs real-only and vs augmentation 2×."""
    accs = {}
    for stem, _, _ in SUMMARY_ARMS:
        path = results_dir / f"per_subject_results_{stem}.json"
        if path.exists():
            accs[stem] = {e["subject"]: 100.0 * e["mean_accuracy"] for e in json.loads(path.read_text())}
    subjects = sorted(set.intersection(*(set(v) for v in accs.values())))
    vec = {k: np.array([v[s] for s in subjects]) for k, v in accs.items()}
    fields = ["arm", "family", "extra_samples_per_fold", "n_subjects", "accuracy_mean", "accuracy_sd",
              "diff_vs_real_only", "diff_vs_real_only_ci_low", "diff_vs_real_only_ci_high",
              "diff_vs_augmentation", "diff_vs_augmentation_ci_low", "diff_vs_augmentation_ci_high",
              "p_vs_augmentation_two_sided"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for stem, family, n_extra in SUMMARY_ARMS:
            if stem not in vec:
                continue
            row = {"arm": stem, "family": family, "extra_samples_per_fold": n_extra,
                   "n_subjects": len(subjects), "accuracy_mean": float(vec[stem].mean()),
                   "accuracy_sd": float(vec[stem].std())}
            for ref, tag in (("vanilla", "real_only"), ("augmented_2x", "augmentation")):
                if stem == ref:
                    continue
                r = paired(vec[stem], vec[ref], rng)
                row[f"diff_vs_{tag}"] = r["mean_diff"]
                row[f"diff_vs_{tag}_ci_low"] = r["ci_low"]
                row[f"diff_vs_{tag}_ci_high"] = r["ci_high"]
                if tag == "augmentation":
                    row["p_vs_augmentation_two_sided"] = r["wilcoxon_p_two_sided"]
            w.writerow({k: _fmt(v) for k, v in row.items()})
    print(f"  saved {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=config.PER_SUBJECT_DIR)
    ap.add_argument("--tables-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    args = ap.parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    subjects, data = load(args.results_dir)
    print(f"Loaded {len(data)} arms × {len(subjects)} subjects")
    rng = np.random.default_rng(SEED)

    rows = {}
    for a, b in COMPARISONS:
        if a in data and b in data:
            rows[(a, b)] = paired(data[a], data[b], rng)

    print(f"\n{'comparison':52s} {'Δ':>6s} {'95% CI':>16s} {'p(2s)':>7s} {'better/worse':>13s}")
    with open(args.tables_dir / "paired_comparisons.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "reference", "mean_diff_pp", "ci95_low", "ci95_high", "wilcoxon_p_two_sided",
                    "n_better", "n_worse", "n_subjects"])
        for (a, b), r in rows.items():
            label = f"{ARMS[a][1]} − {ARMS[b][1]}"
            print(f"{label:52s} {r['mean_diff']:+6.2f} [{r['ci_low']:+6.2f},{r['ci_high']:+6.2f}] "
                  f"{r['wilcoxon_p_two_sided']:7.3f} {r['n_better']:>6d}/{r['n_worse']:<6d}")
            w.writerow([a, b, f"{r['mean_diff']:.3f}", f"{r['ci_low']:.3f}", f"{r['ci_high']:.3f}",
                        f"{r['wilcoxon_p_two_sided']:.4f}", r["n_better"], r["n_worse"], r["n"]])

    # Within-tier comparisons BETWEEN two non-tiering arms are not biased by tier selection.
    tiers = np.array([tier_of(x) for x in data["vanilla"]])
    tier_rows = []
    for a, b in [("subj_2x", "pop_2x"), ("subj_2x", "aug_2x"), ("pop_2x", "aug_2x")]:
        for name, _, _ in TIERS:
            m = tiers == name
            r = paired(data[a][m], data[b][m], rng)
            tier_rows.append({"arm": a, "reference": b, "tier": name, **r})
    with open(args.tables_dir / "tier_comparisons.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tier_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in tier_rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    print("\nWithin-tier (tiers from real-only accuracy; arm-vs-arm, not vs. the tiering run):")
    for r in tier_rows:
        print(f"  {r['arm']:8s} − {r['reference']:8s} {r['tier']:15s} n={r['n']:3d}  "
              f"{r['mean_diff']:+5.1f} [{r['ci_low']:+5.1f},{r['ci_high']:+5.1f}]  p={r['wilcoxon_p_two_sided']:.3f}")

    fig_overview(data, rows, args.figures_dir / "overview.png")
    fig15(rows, data, args.figures_dir / "paired_differences.png")
    rtm = fig16(data, args.figures_dir / "tier_selection_check.png")
    with open(args.tables_dir / "tier_selection_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rtm[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rtm:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    print("\nRegression-to-the-mean check:")
    for r in rtm:
        print(f"  {r['tier']:15s} aug1x−real (tiers by real, n={r['n_tiered_by_real_only']:2d}) "
              f"{r['aug1x_minus_real_tiered_by_real']:+5.1f}  |  real−aug1x (tiers by aug1x, "
              f"n={r['n_tiered_by_aug1x']:2d}) {r['real_minus_aug1x_tiered_by_aug1x']:+5.1f}")

    write_summary(args.results_dir, args.tables_dir / "summary.csv", rng)


if __name__ == "__main__":
    main()
