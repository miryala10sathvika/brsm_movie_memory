#!/usr/bin/env python3
"""Generates the 3 final RQ plots into final_plots/."""

import os, re, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "BRSM data csv")
OUT = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#f5f6fa",
    "axes.edgecolor": "#bdc3c7", "axes.grid": True, "grid.alpha": 0.25,
    "font.family": "sans-serif", "font.size": 12, "axes.labelsize": 13,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})
C_AB, C_NB = "#e74c3c", "#27ae60"
C_BB, C_EM = "#2980b9", "#f39c12"

# ── Helpers ──────────────────────────────────────────────────────────────
def half_violin(ax, data, positions, colors, width=0.35):
    for d, pos, col in zip(data, positions, colors):
        d = d.dropna()
        if len(d) < 3: continue
        kde = stats.gaussian_kde(d)
        y = np.linspace(d.min(), d.max(), 200)
        dens = kde(y); dens = dens / dens.max() * width
        ax.fill_betweenx(y, pos, pos + dens, color=col, alpha=0.45, lw=0)

def jitter(ax, data, positions, colors, w=0.12, s=18, a=0.45):
    rng = np.random.default_rng(42)
    for d, pos, col in zip(data, positions, colors):
        d = d.dropna()
        ax.scatter(pos + rng.uniform(-w, w, len(d)), d,
                   color=col, s=s, alpha=a, edgecolors="white", linewidth=0.3, zorder=3)

# ── Load data ────────────────────────────────────────────────────────────
all_t = []
for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    fname = os.path.basename(f)
    if "sub42_nb" in fname.lower(): continue
    try: df = pd.read_csv(f, on_bad_lines="skip")
    except: continue
    if len(df) < 5: continue
    p = df["participant"].dropna().iloc[0] if "participant" in df.columns else ""
    p_str = str(p)

    # Extract condition from participant string or filename
    p_upper = p_str.upper()
    fname_upper = fname.upper()
    if "_AB" in p_upper or "_AB" in fname_upper:
        cond = "AB"
    elif "_NB" in p_upper or "_NB" in fname_upper:
        cond = "NB"
    else:
        continue  # skip test/junk accounts (test1, aru5467598, hello001, etc.)

    # Extract sub_id: handle typos like suub, subh, SUB
    sid = None
    for src in [p_str, fname]:
        m = re.search(r'[Ss][Uu]*[Bb][Hh]?(\d+)', src)
        if m:
            sid = int(m.group(1))
            break
    if sid is None:
        continue  # can't extract a numeric ID

    r = df[df["movie_id"].notna() & df["resp.corr"].notna()].copy()
    if r.empty: continue
    r["condition"], r["sub_id"] = cond, sid
    r["target_type"] = r["target_img"].apply(
        lambda s: "BB" if "_BB_" in str(s) else ("EM" if "_EM_" in str(s) else np.nan))
    r["acc"] = pd.to_numeric(r["resp.corr"], errors="coerce")
    r["_file"] = fname
    all_t.append(r)

trials = pd.concat(all_t, ignore_index=True)

# Deduplicate: keep first file per (sub_id, condition) in case of duplicate entries
first_file = trials.groupby(["sub_id", "condition"])["_file"].first().reset_index()
first_file_set = set(zip(first_file["sub_id"], first_file["condition"], first_file["_file"]))
trials = trials[trials.apply(lambda r: (r["sub_id"], r["condition"], r["_file"]) in first_file_set, axis=1)]
trials = trials.drop(columns=["_file"])
pp = trials.groupby(["sub_id","condition"]).agg(acc=("acc","mean")).reset_index()
pp_tt = trials.dropna(subset=["target_type"]).groupby(
    ["sub_id","condition","target_type"]).agg(acc=("acc","mean")).reset_index()

ab_acc = pp.loc[pp.condition=="AB","acc"].dropna()
nb_acc = pp.loc[pp.condition=="NB","acc"].dropna()
pos, cols = [1, 2], [C_AB, C_NB]

# ── Stats helper ─────────────────────────────────────────────────────────
def print_stats(label, series):
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    print(f"  {label:25s}  n={len(s):4d}   Mean={s.mean():.4f}   "
          f"Median={s.median():.4f}   SD={s.std():.4f}   "
          f"Q1={q1:.4f}   Q3={q3:.4f}   Min={s.min():.4f}   Max={s.max():.4f}")

# ── Data summary ─────────────────────────────────────────────────────────
n_participants = len(pp)  # unique (sub_id, condition) pairs
n_ab = (pp.condition == "AB").sum()
n_nb = (pp.condition == "NB").sum()
trials_per = len(trials) // n_participants if n_participants else 0
print(f"\n{'='*85}")
print(f"DATA SUMMARY")
print(f"{'='*85}")
print(f"  Total valid recognition trials : {len(trials)}")
print(f"  Total valid participants       : {n_participants}  (AB={n_ab}, NB={n_nb})")
print(f"  Trials per participant         : {trials_per}")
print(f"  Excluded                       : sub42_NB (incomplete session)")
print(f"{'='*85}")

# ── Plot 1  RQ1: Overall accuracy raincloud ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
data = [ab_acc, nb_acc]
half_violin(ax, data, pos, cols, width=0.4)
bp = ax.boxplot(data, positions=pos, widths=0.15, patch_artist=True,
                showfliers=False, zorder=4, medianprops=dict(color="black",lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
for patch, c in zip(bp["boxes"], cols): patch.set(facecolor=c, alpha=0.7)
jitter(ax, data, [p-0.18 for p in pos], cols, w=0.06, s=20, a=0.5)
ax.set_xticks(pos); ax.set_xticklabels(["Abrupt Cut (AB)","Natural Cut (NB)"])
ax.set_ylabel("Recognition Accuracy (per participant)")
ax.axhline(0.5, ls="--", color="grey", alpha=0.4, label="Chance")
ax.legend(framealpha=0.7)
plt.savefig(os.path.join(OUT, "rq1_accuracy_raincloud.png")); plt.close()
print("✓ rq1_accuracy_raincloud.png")

print(f"\n── RQ1 Statistics: Overall Recognition Accuracy ──")
print_stats("Abrupt Cut  (AB)", ab_acc)
print_stats("Natural Cut (NB)", nb_acc)

# ── Plot 2  RQ2: BB accuracy raincloud ───────────────────────────────────
bb = pp_tt[pp_tt.target_type=="BB"]
bb_ab = bb.loc[bb.condition=="AB","acc"].dropna()
bb_nb = bb.loc[bb.condition=="NB","acc"].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
data = [bb_ab, bb_nb]
half_violin(ax, data, pos, cols, width=0.4)
bp = ax.boxplot(data, positions=pos, widths=0.15, patch_artist=True,
                showfliers=False, zorder=4, medianprops=dict(color="black",lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
for patch, c in zip(bp["boxes"], cols): patch.set(facecolor=c, alpha=0.7)
jitter(ax, data, [p-0.18 for p in pos], cols, w=0.06, s=20, a=0.5)
ax.set_xticks(pos); ax.set_xticklabels(["Abrupt Cut (AB)","Natural Cut (NB)"])
ax.set_ylabel("Accuracy — Before-Boundary (BB) Frames")
ax.axhline(0.5, ls="--", color="grey", alpha=0.4)
plt.savefig(os.path.join(OUT, "rq2_bb_accuracy_raincloud.png")); plt.close()
print("✓ rq2_bb_accuracy_raincloud.png")

print(f"\n── RQ2 Statistics: Before-Boundary (BB) Accuracy ──")
print_stats("BB — Abrupt Cut  (AB)", bb_ab)
print_stats("BB — Natural Cut (NB)", bb_nb)

# ── Plot 3  RQ3: BB vs EM violin+strip (2-panel) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for i, (cond, label, cc) in enumerate([("AB","Abrupt Cut (AB)",C_AB),
                                         ("NB","Natural Cut (NB)",C_NB)]):
    ax = axes[i]
    bb_d = pp_tt[(pp_tt.condition==cond)&(pp_tt.target_type=="BB")]["acc"].dropna()
    em_d = pp_tt[(pp_tt.condition==cond)&(pp_tt.target_type=="EM")]["acc"].dropna()
    parts = ax.violinplot([bb_d, em_d], positions=[1,2],
                           showmeans=False, showmedians=True, showextrema=False)
    for j, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([C_BB, C_EM][j]); pc.set_alpha(0.4)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    jitter(ax, [bb_d, em_d], [1,2], [C_BB, C_EM], w=0.08, s=22, a=0.5)
    ax.set_xticks([1,2]); ax.set_xticklabels(["BB","EM"])
    ax.set_xlabel(label)
    ax.axhline(0.5, ls="--", color="grey", alpha=0.4)
    if i == 0: ax.set_ylabel("Recognition Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rq3_bb_em_violin_strip.png")); plt.close()
print("✓ rq3_bb_em_violin_strip.png")

print(f"\n── RQ3 Statistics: BB vs EM Accuracy by Condition ──")
for cond in ["AB", "NB"]:
    for tt in ["BB", "EM"]:
        s = pp_tt[(pp_tt.condition==cond)&(pp_tt.target_type==tt)]["acc"]
        print_stats(f"{tt} — {cond}", s)

print(f"\nDone — 3 plots in {OUT}")
