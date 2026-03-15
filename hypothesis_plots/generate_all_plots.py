#!/usr/bin/env python3
"""
Generate 20 plots for 5 hypotheses (4 plot types each).

Hypotheses (re-indexed):
  H1: Overall Recognition Accuracy
  H2: Before-Boundary (BB) Frame Accuracy
  H3: Overall Confidence Ratings
  H4: Proportion of Low-Confidence Trials
  H5: Movie-Level Accuracy Variability

Plot types per hypothesis (raincloud ALWAYS + 3 of {box, violin, histogram, raindrop}):
  H1: Raincloud + Box + Violin + Histogram
  H2: Raincloud + Violin + Histogram + Raindrop
  H3: Raincloud + Box + Histogram + Raindrop
  H4: Raincloud + Box + Violin + Raindrop
  H5: Raincloud + Violin + Histogram + Raindrop
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
trials, pp, pp_tt = load_data()
n_ab = (pp.condition == "AB").sum()
n_nb = (pp.condition == "NB").sum()
print(f"  {len(trials)} trials, {n_ab + n_nb} participants (AB={n_ab}, NB={n_nb})\n")

BASE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE, "plots")

# Colors
C_AB, C_NB = "#e74c3c", "#27ae60"

# Matplotlib style — larger labels and ticks
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8f9fb",
    "axes.edgecolor": "#bdc3c7",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 28,
    "axes.labelsize": 34,
    "axes.titlesize": 36,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 26,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

# Larger figure size to accommodate bigger labels
FIG_W, FIG_H = 9.5, 7.0


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def make_dir(name):
    d = os.path.join(PLOT_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def half_violin(ax, data, positions, colors, width=0.35):
    """Draw right-sided half-violin (KDE density) for each group."""
    for d, pos, col in zip(data, positions, colors):
        d = d.dropna().values
        if len(d) < 3:
            continue
        kde = stats.gaussian_kde(d, bw_method=0.3)
        y = np.linspace(d.min() - 0.02, d.max() + 0.02, 300)
        dn = kde(y)
        dn = dn / dn.max() * width
        ax.fill_betweenx(y, pos, pos + dn, color=col, alpha=0.4, lw=0)


def jitter_strip(ax, data, positions, colors, w=0.06, s=18, alpha=0.5):
    """Draw jittered strip points."""
    rng = np.random.default_rng(42)
    for d, pos, col in zip(data, positions, colors):
        d = d.dropna().values
        ax.scatter(
            pos + rng.uniform(-w, w, len(d)), d,
            color=col, s=s, alpha=alpha,
            edgecolors="white", linewidth=0.3, zorder=3,
        )


# ── Standard plot builders ────────────────────────────────────────────

def plot_boxplot(data_list, labels, colors, ylabel, title, save_path, chance=None):
    """Standard box plot comparing two groups."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    positions = [1, 2]
    bp = ax.boxplot(
        [d.dropna().values for d in data_list],
        positions=positions, widths=0.45, patch_artist=True,
        showfliers=True, zorder=4,
        medianprops=dict(color="black", lw=2.2),
        whiskerprops=dict(lw=1.3, color="#555"),
        capprops=dict(lw=1.3, color="#555"),
        flierprops=dict(marker='o', markersize=5, alpha=0.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.65, edgecolor=c)
    for flier, c in zip(bp["fliers"], colors):
        flier.set(markeredgecolor=c, markerfacecolor=c)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if chance is not None:
        ax.axhline(chance, ls="--", color="grey", alpha=0.45, label="Chance (0.5)")
        ax.legend(framealpha=0.7, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_violin(data_list, labels, colors, ylabel, title, save_path, chance=None):
    """Violin plot with inner quartile box."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    positions = [1, 2]
    parts = ax.violinplot(
        [d.dropna().values for d in data_list],
        positions=positions, widths=0.7,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.4)
        pc.set_edgecolor(c)

    # Add inner box (quartile markers)
    for d, pos, c in zip(data_list, positions, colors):
        vals = d.dropna().values
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=c, lw=5, alpha=0.7, zorder=4)
        ax.scatter([pos], [med], color="white", s=45, zorder=5,
                   edgecolors=c, linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if chance is not None:
        ax.axhline(chance, ls="--", color="grey", alpha=0.45, label="Chance (0.5)")
        ax.legend(framealpha=0.7, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_raincloud(data_list, labels, colors, ylabel, title, save_path, chance=None):
    """Raincloud: half-violin + box + jittered strip."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    positions = [1, 2]

    # Half violin (right side)
    half_violin(ax, data_list, positions, colors)

    # Box plot (center)
    bp = ax.boxplot(
        [d.dropna().values for d in data_list],
        positions=positions, widths=0.14, patch_artist=True,
        showfliers=False, zorder=4,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.7)

    # Jittered strip (left side)
    jitter_strip(ax, data_list, [p - 0.2 for p in positions], colors)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if chance is not None:
        ax.axhline(chance, ls="--", color="grey", alpha=0.45, label="Chance (0.5)")
        ax.legend(framealpha=0.7, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_histogram(data_list, labels, colors, xlabel, ylabel, title, save_path,
                   bins=15, chance=None):
    """Overlaid histogram for two groups."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for d, label, c in zip(data_list, labels, colors):
        vals = d.dropna().values
        ax.hist(vals, bins=bins, alpha=0.5, color=c, label=label,
                edgecolor="white", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.7)
    if chance is not None:
        ax.axvline(chance, ls="--", color="grey", alpha=0.45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_raindrop(data_list, labels, colors, ylabel, title, save_path, chance=None):
    """
    Raindrop plot: jittered scatter points overlaid on a boxplot.
    Each data point is plotted as a 'raindrop' with random horizontal jitter.
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    positions = [1, 2]
    rng = np.random.default_rng(42)

    # Jittered scatter (the 'raindrops')
    for d, pos, col in zip(data_list, positions, colors):
        vals = d.dropna().values
        jit = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            pos + jit, vals,
            color=col, alpha=0.55, s=28,
            edgecolors="white", linewidth=0.4, zorder=3,
        )

    # Boxplot overlay
    bp = ax.boxplot(
        [d.dropna().values for d in data_list],
        positions=positions, widths=0.35, patch_artist=True,
        showfliers=False, zorder=4,
        medianprops=dict(color="black", lw=2.2),
        whiskerprops=dict(lw=1.3, color="#333"),
        capprops=dict(lw=1.3, color="#333"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c, linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if chance is not None:
        ax.axhline(chance, ls="--", color="grey", alpha=0.45, label="Chance (0.5)")
        ax.legend(framealpha=0.7, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PREPARE HYPOTHESIS-SPECIFIC DATA
# ═══════════════════════════════════════════════════════════════════════

# H1: Overall Recognition Accuracy
ab_acc = pp.loc[pp.condition == "AB", "acc"]
nb_acc = pp.loc[pp.condition == "NB", "acc"]

# H2: Before-Boundary (BB) Frame Accuracy
bb_ab = pp_tt[(pp_tt.condition == "AB") & (pp_tt.target_type == "BB")]["acc"]
bb_nb = pp_tt[(pp_tt.condition == "NB") & (pp_tt.target_type == "BB")]["acc"]

# H3: Overall Confidence Ratings
ab_conf = pp.loc[pp.condition == "AB", "conf"]
nb_conf = pp.loc[pp.condition == "NB", "conf"]

# H4: Proportion of Low-Confidence Trials
pp_lc = trials.groupby(["sub_id", "condition"]).apply(
    lambda g: (g["conf"] <= 2).mean()
).reset_index(name="prop_lc")
ab_lc = pp_lc.loc[pp_lc.condition == "AB", "prop_lc"]
nb_lc = pp_lc.loc[pp_lc.condition == "NB", "prop_lc"]

# H5: Movie-Level Accuracy Variability
pp_mv = trials.groupby(["sub_id", "condition", "movie_id_int"]).agg(
    acc=("acc", "mean")
).reset_index()
pp_sd = pp_mv.groupby(["sub_id", "condition"])["acc"].std().reset_index(name="acc_sd")
ab_sd = pp_sd.loc[pp_sd.condition == "AB", "acc_sd"]
nb_sd = pp_sd.loc[pp_sd.condition == "NB", "acc_sd"]


# ═══════════════════════════════════════════════════════════════════════
# GENERATE PLOTS — 4 per hypothesis (raincloud always + 3 varied others)
# ═══════════════════════════════════════════════════════════════════════

LABELS = ["Abrupt Cut (AB)", "Natural Cut (NB)"]
COLORS = [C_AB, C_NB]

# Plot selection per hypothesis (raincloud always included):
#   H1: raincloud + box + violin + histogram    (no raindrop)
#   H2: raincloud + violin + histogram + raindrop (no box)
#   H3: raincloud + box + histogram + raindrop   (no violin)
#   H4: raincloud + box + violin + raindrop      (no histogram)
#   H5: raincloud + violin + histogram + raindrop (no box)

hypotheses = [
    {
        "id": "H1",
        "name": "Overall Recognition Accuracy",
        "data": [ab_acc, nb_acc],
        "ylabel": "Recognition Accuracy",
        "xlabel_hist": "Recognition Accuracy",
        "chance": 0.5,
        "bins": 15,
        "plots": ["raincloud", "box", "violin", "histogram"],
    },
    {
        "id": "H2",
        "name": "BB Frame Accuracy",
        "data": [bb_ab, bb_nb],
        "ylabel": "BB Frame Accuracy",
        "xlabel_hist": "BB Frame Accuracy",
        "chance": 0.5,
        "bins": 15,
        "plots": ["raincloud", "violin", "histogram", "raindrop"],
    },
    {
        "id": "H3",
        "name": "Overall Confidence Ratings",
        "data": [ab_conf, nb_conf],
        "ylabel": "Mean Confidence (1–5)",
        "xlabel_hist": "Mean Confidence Rating",
        "chance": None,
        "bins": 20,
        "plots": ["raincloud", "box", "histogram", "raindrop"],
    },
    {
        "id": "H4",
        "name": "Low-Confidence Proportion",
        "data": [ab_lc, nb_lc],
        "ylabel": "Proportion (conf ≤ 2)",
        "xlabel_hist": "Proportion of Low-Confidence Trials",
        "chance": None,
        "bins": 15,
        "plots": ["raincloud", "box", "violin", "raindrop"],
    },
    {
        "id": "H5",
        "name": "Movie-Level Accuracy Variability",
        "data": [ab_sd, nb_sd],
        "ylabel": "SD of Per-Movie Accuracy",
        "xlabel_hist": "SD of Per-Movie Accuracy",
        "chance": None,
        "bins": 15,
        "plots": ["raincloud", "violin", "histogram", "raindrop"],
    },
]

PLOT_FUNCS = {
    "box": lambda h, out: plot_boxplot(
        h["data"], LABELS, COLORS, h["ylabel"],
        f"{h['id']}: {h['name']} — Box Plot",
        os.path.join(out, f"{h['id'].lower()}_boxplot.pdf"),
        chance=h["chance"],
    ),
    "violin": lambda h, out: plot_violin(
        h["data"], LABELS, COLORS, h["ylabel"],
        f"{h['id']}: {h['name']} — Violin Plot",
        os.path.join(out, f"{h['id'].lower()}_violin.pdf"),
        chance=h["chance"],
    ),
    "raincloud": lambda h, out: plot_raincloud(
        h["data"], LABELS, COLORS, h["ylabel"],
        f"{h['id']}: {h['name']} — Raincloud Plot",
        os.path.join(out, f"{h['id'].lower()}_raincloud.pdf"),
        chance=h["chance"],
    ),
    "histogram": lambda h, out: plot_histogram(
        h["data"], LABELS, COLORS, h["xlabel_hist"], "Count",
        f"{h['id']}: {h['name']} — Histogram",
        os.path.join(out, f"{h['id'].lower()}_histogram.pdf"),
        bins=h["bins"], chance=h["chance"],
    ),
    "raindrop": lambda h, out: plot_raindrop(
        h["data"], LABELS, COLORS, h["ylabel"],
        f"{h['id']}: {h['name']} — Raindrop Plot",
        os.path.join(out, f"{h['id'].lower()}_raindrop.pdf"),
        chance=h["chance"],
    ),
}

for h in hypotheses:
    hid = h["id"]
    out_dir = make_dir(hid)
    print(f"\n{'─'*60}")
    print(f"  {hid}: {h['name']}")
    print(f"{'─'*60}")

    for ptype in h["plots"]:
        PLOT_FUNCS[ptype](h, out_dir)
        suffix = {"box": "boxplot", "violin": "violin", "raincloud": "raincloud",
                  "histogram": "histogram", "raindrop": "raindrop"}[ptype]
        print(f"  ✓ {hid.lower()}_{suffix}.pdf")

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
total_plots = 0
for h in hypotheses:
    d = os.path.join(PLOT_DIR, h["id"])
    n = len([f for f in os.listdir(d) if f.endswith(".pdf")])
    total_plots += n

print(f"\n{'='*60}")
print(f"  All done — {total_plots} plots generated in {PLOT_DIR}/")
print(f"{'='*60}")
