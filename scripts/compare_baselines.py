#!/usr/bin/env python3
"""
Pre-registered gate verdicts and the research-log comparison figures.

Reads results/per_subject/*.json and produces:
  printed gate verdicts (RR-3) for every synthetic arm
  results/subject_deltas.csv                       per-subject accuracy and Δ vs real-only, all arms
  reports/figures/fig10_accuracy_vs_n.png          accuracy vs extra samples per fold
  reports/figures/fig11_tier_breakdown_vs_n.png    accuracy vs N by subject tier
  reports/figures/fig12–fig14_*.png                per-subject paired-difference plots

Usage (from the repository root):
    python scripts/compare_baselines.py
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

from neurosynth import config

# Version-controlled per-subject results. Training writes fresh JSONs to
# models/checkpoints/ (gitignored); `make collect` copies them here.
CHECKPOINT_DIR = config.PER_SUBJECT_DIR
OUT_DIR = config.REPORT_FIGURES_DIR
TABLES_DIR = config.RESULTS_DIR

# RR-3 go/no-go gate: TSTR@60 must beat the augmented@60 ceiling by a margin that
# is both practically meaningful and statistically significant.
GATE_MARGIN = 1.0    # minimum mean per-subject Δ (percentage points)
GATE_ALPHA = 0.05    # one-sided Wilcoxon significance threshold

C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_BLUE   = "#378ADD"
C_GREEN  = "#639922"
C_PURPLE = "#534AB7"
C_TEAL   = "#1D9E75"
C_ORANGE = "#D9822B"

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
        ("tstr_1x",    checkpoint_dir / "per_subject_results_tstr_1x.json",       30),
        ("tstr_2x",    checkpoint_dir / "per_subject_results_tstr_2x.json",       60),
        ("tstr_3x",    checkpoint_dir / "per_subject_results_tstr_3x.json",       90),
        # pivot_iter_01 — execution-conditioned generator (sampled imagery-domain)
        ("tstr_exec_1x", checkpoint_dir / "per_subject_results_tstr_exec_1x.json", 30),
        ("tstr_exec_2x", checkpoint_dir / "per_subject_results_tstr_exec_2x.json", 60),
        ("tstr_exec_3x", checkpoint_dir / "per_subject_results_tstr_exec_3x.json", 90),
        # pivot_iter_01 ablations (2x only)
        ("tstr_execpool_2x", checkpoint_dir / "per_subject_results_tstr_execpool_2x.json", 60),
        ("tstr_execdom_2x",  checkpoint_dir / "per_subject_results_tstr_execdom_2x.json",  60),
        # pivot_iter_02 — subject-conditional generator (per-subject banks, imagery-domain)
        ("tstr_subj_1x", checkpoint_dir / "per_subject_results_tstr_subj_1x.json", 30),
        ("tstr_subj_2x", checkpoint_dir / "per_subject_results_tstr_subj_2x.json", 60),
        ("tstr_subj_3x", checkpoint_dir / "per_subject_results_tstr_subj_3x.json", 90),
    ]
    # pivot_iter_02 subject-guidance retries: auto-discover any
    # per_subject_results_tstr_subj_g{scale}_{n}x.json produced by --subject-guidance-scale.
    import re
    guided = []
    for p in sorted(checkpoint_dir.glob("per_subject_results_tstr_subj_g*_*x.json")):
        m = re.match(r"per_subject_results_(tstr_subj_g[0-9.]+)_(\d+)x\.json", p.name)
        if m:
            guided.append((f"{m.group(1)}_{m.group(2)}x", p, int(m.group(2)) * 30))
    candidates = candidates + guided

    results: dict[str, RunResult] = {}
    for name, path, n_extra in candidates:
        if path.exists():
            results[name] = _load_run(path, name, n_extra)
            r = results[name]
            print(f"  Loaded {name:16s}: {len(r.subject_accs)} subjects, "
                  f"mean={r.overall_mean:.1f}% ± {r.overall_std:.1f}%")
        else:
            print(f"  Skipping {name:16s}: {path.name} not found")
    return results


def guided_subj_keys(results: dict[str, RunResult], n: str = "2x") -> list[str]:
    """Sorted subject-guidance retry keys present at copy-count ``n`` (e.g. tstr_subj_g3_2x)."""
    import re
    keys = [k for k in results if re.fullmatch(rf"tstr_subj_g[0-9.]+_{n}", k)]
    return sorted(keys, key=lambda k: float(re.search(r"_g([0-9.]+)_", k).group(1)))


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


def _wilcoxon_paired(a: RunResult, b: RunResult) -> tuple[float, float, int]:
    """One-sided paired Wilcoxon: is ``a > b`` across the subjects both share?

    Returns ``(mean_delta_pct, p_value, n_subjects)``. With fewer than 10 shared
    subjects the p-value is NaN (too little power); with identical arrays it is 1.0.
    """
    shared = sorted(set(a.subject_accs) & set(b.subject_accs))
    n = len(shared)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    av = np.array([a.subject_accs[s] for s in shared])
    bv = np.array([b.subject_accs[s] for s in shared])
    mean_delta = float(np.mean(av - bv))
    if n < 10:
        return (mean_delta, float("nan"), n)
    if np.allclose(av, bv):
        return (mean_delta, 1.0, n)
    _, p = stats.wilcoxon(av, bv, alternative="greater")
    return (mean_delta, float(p), n)


def print_summary_table(results: dict[str, RunResult]) -> None:
    """Print summary table with Δ and Wilcoxon p-values vs vanilla."""
    vanilla = results.get("vanilla")
    col = f"{'Run':<18} {'N extra':>8} {'Mean':>8} {'Std':>7} {'Δ vs vanilla':>14} {'p-value':>10}"
    sep = "=" * len(col)
    print(f"\n{sep}\n{col}\n{'-' * len(col)}")

    import re
    guided = sorted(k for k in results if re.fullmatch(r"tstr_subj_g[0-9.]+_\dx", k))
    for key in ["vanilla", "aug_online", "aug_1x", "aug_2x", "aug_3x",
                "tstr_1x", "tstr_2x", "tstr_3x",
                "tstr_exec_1x", "tstr_exec_2x", "tstr_exec_3x",
                "tstr_execpool_2x", "tstr_execdom_2x",
                "tstr_subj_1x", "tstr_subj_2x", "tstr_subj_3x", *guided]:
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


def print_gate_verdict(
    results: dict[str, RunResult],
    tstr_key: str = "tstr_2x",
    secondary_keys: tuple[str, ...] = ("tstr_1x", "tstr_2x", "tstr_3x"),
    lift_key: str | None = None,
    lift_keys: tuple[str, ...] = (),
) -> None:
    """
    RR-3 go/no-go gate: is the TSTR@60 arm ``tstr_key`` better than augmentation@60 (aug_2x)?

    PASS iff the mean per-subject Δ exceeds +GATE_MARGIN% AND the one-sided paired
    Wilcoxon (tstr_key > aug_2x across shared subjects) has p < GATE_ALPHA.
    Also reports the secondary vs-vanilla comparisons, and the paired lift of ``tstr_key``
    over each arm in ``lift_key`` / ``lift_keys`` — e.g. tstr_subj_2x over tstr_exec_2x
    and tstr_2x (how much subject conditioning gained over the population arms).
    """
    title = f"RR-3 GO/NO-GO GATE  —  {tstr_key} vs aug_2x"
    bar = "=" * max(60, len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")

    tstr = results.get(tstr_key)
    aug = results.get("aug_2x")
    vanilla = results.get("vanilla")

    if tstr is None or aug is None:
        missing = [k for k in ("aug_2x", tstr_key) if k not in results]
        print(f"  Cannot evaluate the gate — missing: {', '.join(missing)}.")
        print(f"  Need vanilla, aug_2x, and {tstr_key} per-subject JSONs.")
        print(f"{bar}\n")
        return

    delta, p, n = _wilcoxon_paired(tstr, aug)
    p_ok = (not np.isnan(p)) and (p < GATE_ALPHA)
    margin_ok = delta > GATE_MARGIN
    passed = p_ok and margin_ok
    p_disp = f"{p:.4f}" if not np.isnan(p) else "n/a"

    print(f"  {tstr_key} mean : {tstr.overall_mean:.1f}%   (N=60 synthetic)")
    print(f"  aug_2x  mean : {aug.overall_mean:.1f}%   (N=60 augmented — the ceiling)")
    print(f"  paired Δ     : {delta:+.2f}%   across {n} shared subjects")
    print(f"  Wilcoxon p   : {p_disp}   (one-sided, {tstr_key} > aug_2x)")
    print(f"  margin test  : Δ {delta:+.2f}% {'>' if margin_ok else '≤'} "
          f"+{GATE_MARGIN:.1f}%   -> {'ok' if margin_ok else 'FAIL'}")
    print(f"  signif. test : p {p_disp} {'<' if p_ok else '≥'} "
          f"{GATE_ALPHA}   -> {'ok' if p_ok else 'FAIL'}")
    verdict = "PASS — synthetic rung validated" if passed \
        else "FAIL — synthetic does not beat augmentation"
    print(f"\n  VERDICT: {verdict}")

    # Optional lift(s): how much this arm improves over another (e.g. subj over exec/imagery).
    all_lift_keys = ([lift_key] if lift_key is not None else []) + list(lift_keys)
    for lk in all_lift_keys:
        if lk not in results:
            continue
        ld, lp, ln = _wilcoxon_paired(tstr, results[lk])
        lp_disp = f"{lp:.4f}" if not np.isnan(lp) else "n/a"
        print(f"\n  Lift over {lk} ({results[lk].overall_mean:.1f}%): "
              f"Δ={ld:+.2f}%  p={lp_disp}  (n={ln})")

    # Secondary (informative): the arm at each N vs the vanilla real-only baseline.
    if vanilla is not None:
        print(f"\n  Secondary — vs vanilla ({vanilla.overall_mean:.1f}%):")
        for key in secondary_keys:
            if key not in results:
                continue
            d, pv, nn = _wilcoxon_paired(results[key], vanilla)
            pv_disp = f"{pv:.4f}" if not np.isnan(pv) else "n/a"
            print(f"    {key:<14}: {results[key].overall_mean:>5.1f}%  "
                  f"Δ={d:+.2f}%  p={pv_disp}  (n={nn})")

    print(f"{bar}\n")


def plot_accuracy_vs_n(results: dict[str, RunResult], out: Path) -> None:
    """
    Figure 10: mean per-subject accuracy vs N extra training samples per fold.

    Every curve is anchored at the vanilla (real-only) point at N=0 and adds N samples
    of a different kind: augmented copies of the subject's own epochs, or synthetic
    epochs from one of the generators. Spread across subjects (~±10 pts) is not drawn —
    it is dominated by between-subject differences that cancel in the paired
    comparisons; see fig15 for paired differences with 95% bootstrap CIs.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    vanilla = results.get("vanilla")

    arms = [
        (("aug_1x", "aug_2x", "aug_3x"), "Signal augmentation (own epochs)", "o-", C_BLUE),
        (("tstr_1x", "tstr_2x", "tstr_3x"), "Population generator (imagery)", "s--", C_GREEN),
        (("tstr_exec_1x", "tstr_exec_2x", "tstr_exec_3x"), "Population generator (+ execution data)", "^--", C_TEAL),
        (("tstr_subj_1x", "tstr_subj_2x", "tstr_subj_3x"), "Subject-conditional generator", "D-", C_ORANGE),
    ]
    for keys, label, style, color in arms:
        keys = [k for k in keys if k in results]
        if not keys:
            continue
        ns = ([0] if vanilla else []) + [results[k].n_extra for k in keys]
        ms = ([vanilla.overall_mean] if vanilla else []) + [results[k].overall_mean for k in keys]
        ax.plot(ns, ms, style, color=color, linewidth=2.0, markersize=6.5, label=label, zorder=3)

    if vanilla:
        ax.axhline(vanilla.overall_mean, color=C_GRAY, linestyle=":", linewidth=1.2,
                   label=f"Real data only — {vanilla.overall_mean:.1f}%")
    aug_gate = results.get("aug_2x")
    if aug_gate is not None:
        ax.axhline(aug_gate.overall_mean, color=C_PURPLE, linestyle="-.", linewidth=1.1, alpha=0.8,
                   label=f"Success criterion: beat {aug_gate.overall_mean:.1f}% (aug, N=60)")

    ax.set_xticks([0, 30, 60, 90])
    ax.set_xticklabels(["0\n(real only)", "30\n(1×)", "60\n(2×)", "90\n(3×)"])
    ax.set_xlabel("Extra training samples per fold (on top of ~30 real epochs)")
    ax.set_ylabel("Mean per-subject accuracy (%), 106 subjects")
    ax.set_title("Accuracy vs. extra training samples: augmentation vs. diffusion-generated EEG", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.95, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-6, 102)
    ax.set_ylim(52, 61)

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


def plot_paired_delta(
    results: dict[str, RunResult],
    out: Path,
    key_a: str,
    key_b: str,
    title: str,
    ylabel: str,
    improved_word: str = "improved",
) -> None:
    """
    Per-subject Δ (key_a − key_b) at N=60, sorted ascending, bars colored by the
    subject's vanilla tier; the dashed line marks the cohort mean Δ. Used for fig12
    (tstr_2x − aug_2x) and fig13 (tstr_exec_2x − tstr_2x).
    """
    a = results.get(key_a)
    b = results.get(key_b)
    vanilla = results.get("vanilla")
    if a is None or b is None:
        print(f"  {out.name} skipped: need {key_a} and {key_b}")
        return

    shared = sorted(set(a.subject_accs) & set(b.subject_accs))
    if not shared:
        print(f"  {out.name} skipped: no shared subjects between {key_a} and {key_b}")
        return

    def tier_color(subj: int) -> str:
        acc = vanilla.subject_accs.get(subj) if vanilla else None
        if acc is None:
            return C_GRAY
        for _, lo, hi, color in TIERS:
            if lo <= acc < hi:
                return color
        return C_GRAY

    deltas = sorted(
        ((s, a.subject_accs[s] - b.subject_accs[s]) for s in shared),
        key=lambda t: t[1],
    )
    subs = [s for s, _ in deltas]
    vals = [d for _, d in deltas]
    colors = [tier_color(s) for s in subs]
    mean_delta = float(np.mean(vals))
    n_pos = sum(1 for v in vals if v > 0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(vals)), vals, color=colors, width=1.0, edgecolor="white", linewidth=0.2)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhline(mean_delta, color=C_PURPLE, linestyle="--", linewidth=1.5,
               label=f"Mean Δ = {mean_delta:+.2f}%")

    ax.set_xlabel(f"Subjects sorted by Δ  ({n_pos}/{len(vals)} {improved_word})")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    tier_handles = [mpatches.Patch(color=color, label=name) for name, _, _, color in TIERS]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + tier_handles, fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-1, len(vals))

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

    import re
    guided = sorted(k for k in results if re.fullmatch(r"tstr_subj_g[0-9.]+_\dx", k))
    extra_keys = [
        k for k in ("aug_1x", "aug_2x", "aug_3x",
                    "tstr_1x", "tstr_2x", "tstr_3x",
                    "tstr_exec_1x", "tstr_exec_2x", "tstr_exec_3x",
                    "tstr_execpool_2x", "tstr_execdom_2x",
                    "tstr_subj_1x", "tstr_subj_2x", "tstr_subj_3x", *guided)
        if k in results
    ]
    out = out_dir / "subject_deltas.csv"
    fieldnames = ["subject", "vanilla_acc", "tier"] + [
        col for k in extra_keys for col in (f"{k}_acc", f"{k}_delta")
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for s in sorted(vanilla.subject_accs):
            v_acc = vanilla.subject_accs[s]
            row: dict = {
                "subject": s,
                "vanilla_acc": round(v_acc, 1),
                "tier": get_tier(v_acc),
            }
            for k in extra_keys:
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
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading per-subject results...")
    results = load_all_results(args.checkpoint_dir)

    if not results:
        print("No result files found. Run train_baseline.py first.")
        return

    print_summary_table(results)
    # Step 5 gate (imagery-only synthetic).
    print_gate_verdict(results)
    # pivot_iter_01 gate (execution-conditioned synthetic), if present, with the
    # paired lift over the imagery-only arm.
    if "tstr_exec_2x" in results:
        print_gate_verdict(
            results,
            tstr_key="tstr_exec_2x",
            secondary_keys=("tstr_exec_1x", "tstr_exec_2x", "tstr_exec_3x",
                            "tstr_execpool_2x", "tstr_execdom_2x"),
            lift_key="tstr_2x",
        )
    # pivot_iter_02 gate (subject-conditional synthetic), if present, with the paired
    # lift over both population arms (execution-enriched and imagery-only).
    if "tstr_subj_2x" in results:
        print_gate_verdict(
            results,
            tstr_key="tstr_subj_2x",
            secondary_keys=("tstr_subj_1x", "tstr_subj_2x", "tstr_subj_3x"),
            lift_keys=("tstr_exec_2x", "tstr_2x"),
        )
    # pivot_iter_02 subject-guidance retries: gate each vs aug_2x with the lift over the
    # un-guided subject arm (did sharpening toward the subject help?).
    for gk in guided_subj_keys(results, "2x"):
        stem = gk.rsplit("_", 1)[0]  # tstr_subj_g{scale}
        print_gate_verdict(
            results,
            tstr_key=gk,
            secondary_keys=tuple(f"{stem}_{n}" for n in ("1x", "2x", "3x") if f"{stem}_{n}" in results),
            lift_keys=("tstr_subj_2x", "tstr_exec_2x"),
        )
    save_delta_csv(results, args.tables_dir)

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

    plot_paired_delta(
        results, args.out_dir / "fig12_tstr_vs_aug_delta.png",
        key_a="tstr_2x", key_b="aug_2x",
        title="Figure 12 — Per-Subject Synthetic vs. Augmentation Advantage (N=60)\n"
              "bars colored by vanilla tier",
        ylabel="TSTR@60 − Augmented@60 (percentage points)",
        improved_word="improved with synthetic",
    )
    if "tstr_exec_2x" in results:
        plot_paired_delta(
            results, args.out_dir / "fig13_tstr_exec_vs_imagery_delta.png",
            key_a="tstr_exec_2x", key_b="tstr_2x",
            title="Figure 13 — Execution-Conditioned vs. Imagery-Only Synthetic (N=60)\n"
                  "did execution rescue the tiers imagery-only harmed? (colored by vanilla tier)",
            ylabel="TSTR_exec@60 − TSTR_imagery@60 (percentage points)",
            improved_word="helped by execution",
        )
    # pivot_iter_02: did per-subject generation rescue the Learning/Strong tiers the
    # population (execution-enriched) arm still degraded?
    if "tstr_subj_2x" in results and "tstr_exec_2x" in results:
        plot_paired_delta(
            results, args.out_dir / "fig14_tstr_subj_vs_exec_delta.png",
            key_a="tstr_subj_2x", key_b="tstr_exec_2x",
            title="Figure 14 — Subject-Conditional vs. Population (Execution) Synthetic (N=60)\n"
                  "did per-subject generation help the tiers population synthetic harmed? "
                  "(colored by vanilla tier)",
            ylabel="TSTR_subj@60 − TSTR_exec@60 (percentage points)",
            improved_word="helped by subject conditioning",
        )


if __name__ == "__main__":
    main()
