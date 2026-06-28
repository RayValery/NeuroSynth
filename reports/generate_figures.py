#!/usr/bin/env python3
"""
Generate all figures for the Phase 0 Baseline Evaluation Report.

Usage:
    cd /path/to/NeuroSynth
    source .venv/bin/activate
    python reports/generate_figures.py

Output: reports/figures/fig*.png
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
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

C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_BLUE   = "#378ADD"
C_GREEN  = "#639922"
C_PURPLE = "#534AB7"
C_AMBER  = "#BA7517"
C_TEAL   = "#1D9E75"

# ─── V06 per-subject results ──────────────────────────────────────────────────

SUBJECT_ACCS: dict[int, float] = {
    1:48.9, 2:57.8, 3:53.3, 4:68.9, 5:66.7, 6:53.3, 7:64.4, 8:62.2, 9:62.2, 10:68.9,
    11:37.8, 12:46.7, 13:48.9, 14:53.3, 15:68.9, 16:64.4, 17:73.3, 18:71.1, 19:57.8, 20:53.3,
    21:68.9, 22:48.9, 23:55.6, 24:48.9, 25:55.6, 26:68.9, 27:62.2, 28:60.0, 29:64.4, 30:53.3,
    31:53.3, 32:60.0, 33:55.6, 34:62.5, 35:62.2, 36:40.0, 37:56.3, 38:48.9, 39:37.8, 40:75.6,
    41:43.2, 42:64.4, 43:46.7, 44:62.2, 45:55.6, 46:33.3, 47:44.4, 48:44.4, 49:62.2, 50:64.4,
    51:86.7, 52:51.1, 53:84.4, 54:57.8, 55:51.1, 56:55.6, 57:40.0, 58:53.3, 59:42.2, 60:51.1,
    61:55.6, 62:80.0, 63:44.4, 64:49.8, 65:53.3, 66:46.7, 67:57.8, 68:51.1, 69:57.8, 70:64.4,
    71:71.1, 72:88.4, 73:42.9, 74:44.0, 75:46.7, 76:45.2, 77:53.3, 78:57.8, 79:46.7, 80:55.6,
    81:55.6, 82:53.3, 83:35.6, 84:44.4, 85:84.4, 86:68.9, 87:53.3, 89:57.8, 90:46.7, 91:64.4,
    93:62.2, 94:64.4, 95:57.8, 96:60.0, 97:51.1, 98:48.9, 99:64.4, 101:62.2, 102:47.6,
    103:57.8, 104:46.7, 105:64.4, 106:64.4, 107:57.8, 108:62.2, 109:46.7,
}


def tier_color(acc: float) -> str:
    if acc < 45:
        return C_RED
    elif acc < 56:
        return C_GRAY
    elif acc < 70:
        return C_BLUE
    else:
        return C_GREEN


# ─── Figure 1: Preprocessing pipeline ───────────────────────────────────────

def fig1_preprocessing_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        ("Download\n(MNE/PhysioNet)", "109 subjects\n64 ch, 160 Hz"),
        ("Band-pass Filter\n7–30 Hz", "mu + beta\nmotor rhythm"),
        ("Epoch per Run\n−1 s to +4 s", "801 samples\nT1/T2 events"),
        ("Z-score\nNormalize", "per channel\nper epoch"),
        ("Save as\n.npz", "X, y,\nsubject_ids,\nrun_ids"),
    ]

    box_w, box_h = 1.9, 1.3
    gap = 0.45
    y0 = 0.85
    xs = [0.3 + i * (box_w + gap) for i in range(len(steps))]

    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        color = C_PURPLE if i < len(steps) - 1 else C_TEAL
        fancy = FancyBboxPatch((x, y0), box_w, box_h,
                               boxstyle="round,pad=0.05",
                               facecolor=color + "18", edgecolor=color, linewidth=1.5)
        ax.add_patch(fancy)
        ax.text(x + box_w / 2, y0 + box_h * 0.63, title,
                ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x + box_w / 2, y0 + box_h * 0.22, subtitle,
                ha="center", va="center", fontsize=7.5, color="#444444")

        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1], y0 + box_h / 2),
                        xytext=(x + box_w, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    ax.set_title("Figure 1 — EEGBCI Data Preprocessing Pipeline", fontsize=11, pad=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_preprocessing_pipeline.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_preprocessing_pipeline.png")


# ─── Figure 2: EEGNet architecture ──────────────────────────────────────────

def fig2_eegnet_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        {
            "label": "Input",
            "lines": ["(B, 1, 64, 801)"],
            "x": 0.1, "w": 1.7, "color": C_GRAY,
        },
        {
            "label": "Block 1 — Temporal Filter",
            "lines": ["Conv2d(1→8, kernel 1×64)", "BatchNorm2d(8)", "→ (B, 8, 64, 801)"],
            "x": 2.1, "w": 2.8, "color": C_PURPLE,
        },
        {
            "label": "Block 2 — Spatial Filter",
            "lines": ["DepthwiseConv2d(8→16, 64×1)", "BatchNorm, ELU", "AvgPool(1×4), Dropout(0.5)", "→ (B, 16, 1, 200)"],
            "x": 5.2, "w": 2.8, "color": C_BLUE,
        },
        {
            "label": "Block 3 — Separable Conv",
            "lines": ["DepthwiseConv2d(16×16, 1×16)", "PointwiseConv2d(16→16, 1×1)", "BatchNorm, ELU, AvgPool(1×8)", "→ (B, 16, 1, 25)"],
            "x": 8.3, "w": 2.8, "color": C_TEAL,
        },
        {
            "label": "Classifier",
            "lines": ["Flatten → 400", "Linear(400 → 2)", "→ logits"],
            "x": 11.4, "w": 1.5, "color": C_GREEN,
        },
    ]

    for blk in blocks:
        x, w, color = blk["x"], blk["w"], blk["color"]
        box_h = 2.8
        y0 = 0.6
        fancy = FancyBboxPatch((x, y0), w, box_h,
                               boxstyle="round,pad=0.07",
                               facecolor=color + "15", edgecolor=color, linewidth=1.5)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y0 + box_h - 0.25, blk["label"],
                ha="center", va="top", fontsize=8.5, fontweight="bold", color=color)
        for j, line in enumerate(blk["lines"]):
            ax.text(x + w / 2, y0 + box_h - 0.65 - j * 0.55, line,
                    ha="center", va="top", fontsize=7.5, color="#333333", family="monospace")

    for i in range(len(blocks) - 1):
        x_end = blocks[i]["x"] + blocks[i]["w"]
        x_start = blocks[i + 1]["x"]
        y_mid = 2.0
        ax.annotate("", xy=(x_start, y_mid), xytext=(x_end, y_mid),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    ax.text(6.5, 0.25, "Total parameters: 2,930   |   Subject-independent architecture   |   Lawhern et al. 2018",
            ha="center", fontsize=8, color="#555555", style="italic")

    ax.set_title("Figure 2 — EEGNet Architecture (F1=8, D=2, F2=16, n_channels=64, n_times=801)",
                 fontsize=11, pad=6)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_eegnet_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_eegnet_architecture.png")


# ─── Figure 3: LORO protocol ─────────────────────────────────────────────────

def fig3_loro_protocol() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 3.8)
    ax.axis("off")

    runs = ["Run 4", "Run 8", "Run 12"]
    folds = ["Fold 1\n(test run=4)", "Fold 2\n(test run=8)", "Fold 3\n(test run=12)"]

    for fi, fold_label in enumerate(folds):
        y = 2.5 - fi * 1.1
        ax.text(-0.3, y, fold_label, ha="right", va="center", fontsize=9, fontweight="bold")
        for ri, run_label in enumerate(runs):
            x = 0.5 + ri * 1.1
            is_test = ri == fi
            color = C_RED if is_test else C_BLUE
            label = "TEST" if is_test else "TRAIN"
            sub_label = "~15 samples" if is_test else "~15 samples"
            fancy = FancyBboxPatch((x - 0.45, y - 0.35), 0.9, 0.7,
                                   boxstyle="round,pad=0.04",
                                   facecolor=color + "25", edgecolor=color, linewidth=1.5)
            ax.add_patch(fancy)
            ax.text(x, y + 0.08, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
            ax.text(x, y - 0.17, sub_label, ha="center", va="center",
                    fontsize=7, color="#555555")

    for ri, run_label in enumerate(runs):
        x = 0.5 + ri * 1.1
        ax.text(x, 3.55, run_label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#222222")

    patch_train = mpatches.Patch(facecolor=C_BLUE + "25", edgecolor=C_BLUE, linewidth=1.2, label="Training set (~30 samples)")
    patch_test  = mpatches.Patch(facecolor=C_RED + "25", edgecolor=C_RED, linewidth=1.2, label="Test set (~15 samples)")
    ax.legend(handles=[patch_train, patch_test], loc="lower center",
              bbox_to_anchor=(0.45, -0.12), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Leave-One-Run-Out (LORO) folds per subject", fontsize=10, pad=4)

    ax2 = axes[1]
    ax2.axis("off")
    info = [
        ("Subjects", "106 (3 dropped: 88, 92, 100)"),
        ("Runs per subject", "3 imagery runs (4, 8, 12)"),
        ("Total folds", "318  (106 × 3)"),
        ("Training set size", "~30 epochs per fold"),
        ("Test set size", "~15 epochs per fold"),
        ("Aggregation", "mean over 3 folds → subject mean\nmean over 106 subjects → final"),
        ("No leakage", "test run epochs never seen\nduring training"),
    ]
    y = 3.5
    ax2.text(0.05, y, "Protocol Summary", fontsize=10, fontweight="bold", color="#222222")
    for k, v in info:
        y -= 0.44
        ax2.text(0.05, y, f"• {k}:", fontsize=8.5, color="#555555")
        ax2.text(0.05, y - 0.22, f"  {v}", fontsize=8.5, color="#222222")
        y -= 0.22

    fig.suptitle("Figure 3 — Leave-One-Run-Out Evaluation Protocol", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_loro_protocol.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_loro_protocol.png")


# ─── Figure 4: Experiment evolution ─────────────────────────────────────────

def fig4_experiment_evolution() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6.5),
                                          gridspec_kw={"height_ratios": [2, 1.5]})

    versions = ["V01", "V02", "V03", "V04", "V05", "V06"]
    accuracies = [50.0, 50.0, 52.0, 50.5, 54.1, 56.7]
    colors_bar = [C_GRAY, C_GRAY, C_GRAY, C_RED, C_AMBER, C_GREEN]

    ax_top.bar(versions, accuracies, color=[c + "99" for c in colors_bar],
               edgecolor=colors_bar, linewidth=1.5, width=0.55)
    ax_top.axhline(50, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    ax_top.axhline(65, color=C_GREEN, linestyle="--", linewidth=1, alpha=0.6, label="Literature target (65–75%)")
    ax_top.axhspan(65, 75, alpha=0.07, color=C_GREEN, label="_nolegend_")
    for i, (v, acc) in enumerate(zip(versions, accuracies)):
        ax_top.text(i, acc + 0.4, f"{acc:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=colors_bar[i])
    ax_top.set_ylim(47, 72)
    ax_top.set_ylabel("Test Accuracy (%)")
    ax_top.set_title("Accuracy progression across experiments", fontsize=11)
    ax_top.legend(fontsize=8, framealpha=0.8)
    ax_top.grid(axis="y", alpha=0.3)

    ax_bot.axis("off")
    rows = [
        ["V01", "Cross-subject, lr=1e-3, patience=15", "~50%", "Chance. Loss oscillates, no convergence."],
        ["V02", "Add ReduceLROnPlateau, patience=30", "~50%", "Scheduler fires too late. Still stuck."],
        ["V03", "lr → 1e-4, CosineAnnealingLR, epochs→300", "~52%", "Convergent but gradients cancel across subjects."],
        ["V04", "Add Euclidean Alignment (EA)", "~50%", "EA destroys ERD spatial signal. Reverted."],
        ["V05", "Switch to within-subject LORO", "54.1%", "6-sample val set causes bimodal failure (62.9% dead zone)."],
        ["V06", "Full-batch, fixed epochs, no val split", "56.7%", "Final baseline. Non-responders limit the mean."],
    ]
    col_labels = ["Exp.", "Key Change", "Accuracy", "Observation"]
    col_widths = [0.055, 0.35, 0.075, 0.52]
    col_starts = [0.0, 0.06, 0.42, 0.505]
    header_y = 1.02
    data_y_start = 0.88

    for ci, (label, x) in enumerate(zip(col_labels, col_starts)):
        ax_bot.text(x, header_y, label, fontsize=8.5, fontweight="bold", color="#222222",
                    transform=ax_bot.transAxes)

    for ri, row in enumerate(rows):
        y = data_y_start - ri * 0.155
        row_color = "#f8f8f8" if ri % 2 == 0 else "white"
        for ci, (cell, x) in enumerate(zip(row, col_starts)):
            color = colors_bar[ri] if ci == 2 else "#333333"
            weight = "bold" if ci == 2 else "normal"
            ax_bot.text(x, y, cell, fontsize=8, color=color, fontweight=weight,
                        transform=ax_bot.transAxes, va="top")

    fig.suptitle("Figure 4 — Experiment Evolution: V01–V06", fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "fig4_experiment_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_experiment_evolution.png")


# ─── Figure 5: V05 fold distribution (bimodal failure) ──────────────────────

def fig5_v05_fold_distribution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins_labels = ["<40%\n(below\nchance)", "40–56%\n(dead\nzone)", "60–70%\n(working)", "70–80%", ">80%"]
    counts = [20, 200, 63, 17, 16]
    percents = [c / 316 * 100 for c in counts]
    bar_colors = [C_RED, C_GRAY, C_BLUE, C_GREEN, C_GREEN]

    ax = axes[0]
    bars = ax.bar(range(len(bins_labels)), counts, color=[c + "88" for c in bar_colors],
                  edgecolor=bar_colors, linewidth=1.5, width=0.6)
    for bar, pct, cnt in zip(bars, percents, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(bins_labels)))
    ax.set_xticklabels(bins_labels, fontsize=8)
    ax.set_ylabel("Number of folds")
    ax.set_title("V05: Fold accuracy distribution (318 folds)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    dead_zone_pct = (20 + 200) / 316 * 100
    working_pct = (63 + 17 + 16) / 316 * 100
    ax.annotate(f"Dead zone:\n{dead_zone_pct:.1f}%\nof all folds",
                xy=(0.5, 200), xytext=(1.5, 160),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
                fontsize=8, color=C_RED, fontweight="bold")

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    causes = [
        ("Root Cause 1: Premature Early Stopping",
         "20% val split of 30 training samples = 6 val epochs.\n"
         "Val loss on 6 binary samples is dominated by chance fluctuation.\n"
         "Stopping fires at epoch ~120/500. CosineAnnealingLR barely\n"
         "decayed (lr still ≈9.9×10⁻⁴). Model terminates underfitted.",
         C_RED),
        ("Root Cause 2: Class-Biased Checkpoint",
         "Early-stage models predict strongly toward one class\n"
         "(a natural intermediate in gradient descent). If 6 val\n"
         "samples are skewed toward that class, the biased checkpoint\n"
         "looks good on val → saved → scores 27–40% on balanced test.",
         C_AMBER),
        ("Root Cause 3: Noisy BatchNorm",
         "batch_size=16 with 24 actual training samples gives a\n"
         "last batch of only 8 samples every epoch. BatchNorm\n"
         "statistics on 8 samples are noisy, compounding the\n"
         "checkpoint selection problem.",
         C_GRAY),
    ]

    y = 0.95
    for title, body, color in causes:
        ax2.text(0.03, y, title, fontsize=8.5, fontweight="bold", color=color, va="top")
        y -= 0.09
        for line in body.split("\n"):
            ax2.text(0.05, y, line, fontsize=7.8, color="#333333", va="top")
            y -= 0.075
        y -= 0.04

    ax2.text(0.03, y, "Working folds (>60%) averaged 73.3% — architecture is correct.",
             fontsize=8.5, color=C_GREEN, fontweight="bold", va="top")

    fig.suptitle("Figure 5 — V05 Failure Analysis: Bimodal Fold Distribution", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_v05_fold_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_v05_fold_distribution.png")


# ─── Figure 6: V06 subject accuracy histogram ───────────────────────────────

def fig6_v06_histogram() -> None:
    accs = np.array(list(SUBJECT_ACCS.values()))
    mean_acc = accs.mean()
    median_acc = np.median(accs)

    fig, ax = plt.subplots(figsize=(9, 5))

    bin_edges = np.arange(28, 93, 6.67)
    counts, edges = np.histogram(accs, bins=bin_edges)

    for i in range(len(counts)):
        mid = (edges[i] + edges[i + 1]) / 2
        color = tier_color(mid)
        ax.bar(mid, counts[i], width=(edges[i + 1] - edges[i]) * 0.88,
               color=color + "88", edgecolor=color, linewidth=1.2)

    ax.axvline(mean_acc, color=C_PURPLE, linestyle="-", linewidth=2,
               label=f"Mean = {mean_acc:.1f}%")
    ax.axvline(median_acc, color=C_AMBER, linestyle="--", linewidth=1.5,
               label=f"Median = {median_acc:.1f}%")
    ax.axvline(50, color="#cc0000", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Chance (50%)")

    tier_patches = [
        mpatches.Patch(facecolor=C_RED + "88", edgecolor=C_RED, linewidth=1, label="Non-responders <45% (n=14, 13%)"),
        mpatches.Patch(facecolor=C_GRAY + "88", edgecolor=C_GRAY, linewidth=1, label="Mud zone 45–56% (n=41, 39%)"),
        mpatches.Patch(facecolor=C_BLUE + "88", edgecolor=C_BLUE, linewidth=1, label="Learners 56–70% (n=42, 40%)"),
        mpatches.Patch(facecolor=C_GREEN + "88", edgecolor=C_GREEN, linewidth=1, label="Strong responders >70% (n=9, 8%)"),
    ]
    ax.legend(handles=tier_patches + [
        mpatches.Patch(facecolor="white", edgecolor="none",
                       label=f"Mean={mean_acc:.1f}%, Median={median_acc:.1f}%, σ={accs.std():.1f}%")
    ], fontsize=8, loc="upper right", framealpha=0.9)

    ax.set_xlabel("Per-subject mean accuracy (%)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Figure 6 — V06: Distribution of Per-Subject Accuracy (106 subjects)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.02, 0.97,
            f"Mean:   {mean_acc:.1f}%\nMedian: {median_acc:.1f}%\nStd:      {accs.std():.1f}%\nN:         {len(accs)}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.4", alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_v06_subject_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_v06_subject_histogram.png")


# ─── Figure 7: V06 per-subject sorted bar ───────────────────────────────────

def fig7_sorted_bar() -> None:
    sorted_items = sorted(SUBJECT_ACCS.items(), key=lambda x: x[1])
    subj_ids = [f"S{k:03d}" for k, _ in sorted_items]
    accs_sorted = [v for _, v in sorted_items]
    colors_sorted = [tier_color(a) for a in accs_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(accs_sorted))
    bars = ax.bar(x, accs_sorted, color=[c + "aa" for c in colors_sorted],
                  edgecolor=colors_sorted, linewidth=0.6, width=0.8)

    ax.axhline(50, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")
    mean_v = np.mean(accs_sorted)
    ax.axhline(mean_v, color=C_PURPLE, linestyle="-", linewidth=1.5,
               label=f"Mean = {mean_v:.1f}%")
    ax.axhline(65, color=C_GREEN, linestyle=":", linewidth=1, alpha=0.6,
               label="Literature target (65%)")

    tick_step = 5
    xticks = list(range(0, len(accs_sorted), tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([subj_ids[i] for i in xticks], rotation=45, ha="right", fontsize=7.5)

    notable = {"S072": 88.4, "S051": 86.7, "S085": 84.4, "S053": 84.4,
               "S046": 33.3, "S083": 35.6, "S011": 37.8, "S039": 37.8}
    id_to_idx = {f"S{k:03d}": i for i, (k, _) in enumerate(sorted_items)}
    for sid, acc in notable.items():
        if sid in id_to_idx:
            idx = id_to_idx[sid]
            offset = 1.5 if acc >= 50 else -4.5
            ax.text(idx, acc + offset, sid, ha="center", fontsize=6.5,
                    color="#222222", rotation=90)

    tier_patches = [
        mpatches.Patch(facecolor=C_RED + "aa", edgecolor=C_RED, linewidth=1, label="Non-responders <45%"),
        mpatches.Patch(facecolor=C_GRAY + "aa", edgecolor=C_GRAY, linewidth=1, label="Mud zone 45–56%"),
        mpatches.Patch(facecolor=C_BLUE + "aa", edgecolor=C_BLUE, linewidth=1, label="Learners 56–70%"),
        mpatches.Patch(facecolor=C_GREEN + "aa", edgecolor=C_GREEN, linewidth=1, label="Strong responders >70%"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + tier_patches, fontsize=8, loc="upper left",
              framealpha=0.9, ncol=2)

    ax.set_xlabel("Subjects (sorted by accuracy)")
    ax.set_ylabel("Mean accuracy over 3 LORO folds (%)")
    ax.set_title("Figure 7 — V06: Per-Subject Accuracy, All 106 Subjects (sorted ascending)", fontsize=11)
    ax.set_ylim(25, 96)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_v06_sorted_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_v06_sorted_bar.png")


# ─── Figure 8: V05 vs V06 comparison ────────────────────────────────────────

def fig8_v05_vs_v06() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    v05_bins = ["<40%", "40–56%\n(dead zone)", "60–70%", "70–80%", ">80%"]
    v05_counts = [20, 200, 63, 17, 16]
    v05_total = sum(v05_counts)

    v06_accs = np.array(list(SUBJECT_ACCS.values()))
    v06_fold_accs = v06_accs
    v06_bins_boundaries = [0, 40, 56, 70, 80, 101]
    v06_fold_labels = ["<40%", "40–56%", "56–70%", "70–80%", ">80%"]
    v06_fold_counts = [
        sum(1 for a in v06_accs if 0 <= a < 40),
        sum(1 for a in v06_accs if 40 <= a < 56),
        sum(1 for a in v06_accs if 56 <= a < 70),
        sum(1 for a in v06_accs if 70 <= a < 80),
        sum(1 for a in v06_accs if a >= 80),
    ]
    v06_total = sum(v06_fold_counts)

    bar_colors = [C_RED, C_GRAY, C_BLUE, C_GREEN, C_GREEN]

    for ax, bins, counts, total, title, key_label in [
        (axes[0], v05_bins, v05_counts, v05_total,
         "V05: With val split + early stopping\n(54.1% mean)", "62.9% dead zone"),
        (axes[1], v06_fold_labels, v06_fold_counts, v06_total,
         "V06: Full-batch, fixed epochs, no val split\n(56.7% mean)", ""),
    ]:
        bars = ax.bar(range(len(bins)), [c / total * 100 for c in counts],
                      color=[c + "88" for c in bar_colors],
                      edgecolor=bar_colors, linewidth=1.5, width=0.6)
        for bar, cnt, pct in zip(bars, counts, [c / total * 100 for c in counts]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=8)
        ax.set_ylabel("% of subjects")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 75)
        ax.grid(axis="y", alpha=0.3)
        if key_label:
            ax.text(0.5, 0.88, key_label, transform=ax.transAxes,
                    ha="center", color=C_RED, fontsize=9, fontweight="bold")

    fig.suptitle("Figure 8 — Accuracy Distribution: V05 (bimodal failure) vs V06 (fixed training)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_v05_vs_v06.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_v05_vs_v06.png")


# ─── Figure 9: Subject tier pie ──────────────────────────────────────────────

def fig9_tier_breakdown() -> None:
    accs = list(SUBJECT_ACCS.values())
    tiers = {
        "Non-responders\n(<45%)\nn=14": sum(1 for a in accs if a < 45),
        "Mud zone\n(45–56%)\nn=41": sum(1 for a in accs if 45 <= a < 56),
        "Learners\n(56–70%)\nn=42": sum(1 for a in accs if 56 <= a < 70),
        "Strong responders\n(>70%)\nn=9": sum(1 for a in accs if a >= 70),
    }
    colors = [C_RED + "cc", C_GRAY + "cc", C_BLUE + "cc", C_GREEN + "cc"]

    fig, (ax_pie, ax_text) = plt.subplots(1, 2, figsize=(10, 4.5),
                                           gridspec_kw={"width_ratios": [1, 1.3]})

    wedges, texts, autotexts = ax_pie.pie(
        tiers.values(), labels=tiers.keys(), colors=colors,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.72, labeldistance=1.12,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for t in texts:
        t.set_fontsize(8.5)
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    ax_pie.set_title("Subject distribution by accuracy tier", fontsize=10)

    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    lines = [
        (C_RED, "Non-responders (13%): genuine physiological limitation.",
         "These subjects don't produce reliable ERD patterns during motor"),
        ("", "imagery. Known as 'BCI illiteracy'. No augmentation will help.", ""),
        ("", "", ""),
        (C_GRAY, "Mud zone (39%): data-limited, weak signal.",
         "30 training samples is not enough to reliably learn the"),
        ("", "spatial filter for weak-signal subjects. Key augmentation target.", ""),
        ("", "", ""),
        (C_BLUE, "Learners (40%): model working correctly.",
         "Meaningful learning from 30 samples. Synthetic data could"),
        ("", "push these subjects toward the strong responder tier.", ""),
        ("", "", ""),
        (C_GREEN, "Strong responders (8%): excellent signal quality.",
         "Consistent ERD patterns, robust across runs. Represent"),
        ("", "the upper bound with current data volume.", ""),
    ]

    y = 0.96
    for color, line1, line2 in lines:
        if not line1 and not line2:
            y -= 0.05
            continue
        if color:
            ax_text.add_patch(mpatches.Rectangle((0.01, y - 0.04), 0.012, 0.07,
                                                   color=color, transform=ax_text.transAxes))
        ax_text.text(0.06, y, line1, fontsize=8.5,
                     color="#222222" if color else "#444444",
                     fontweight="bold" if color else "normal",
                     transform=ax_text.transAxes, va="top")
        if line2:
            y -= 0.1
            ax_text.text(0.06, y, line2, fontsize=8.2, color="#555555",
                         transform=ax_text.transAxes, va="top")
        y -= 0.085

    fig.suptitle("Figure 9 — V06: Subject Tier Breakdown and Interpretation", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig9_tier_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig9_tier_breakdown.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures → reports/figures/")
    fig1_preprocessing_pipeline()
    fig2_eegnet_architecture()
    fig3_loro_protocol()
    fig4_experiment_evolution()
    fig5_v05_fold_distribution()
    fig6_v06_histogram()
    fig7_sorted_bar()
    fig8_v05_vs_v06()
    fig9_tier_breakdown()
    print(f"Done. {len(list((Path(__file__).parent / 'figures').glob('*.png')))} figures saved.")
