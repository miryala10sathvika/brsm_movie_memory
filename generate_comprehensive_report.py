"""Generate a comprehensive PDF report of all statistical analyses.

Must be run AFTER run_statistical_tests.py and analysis_regression_models.py.

Output: statistical_results/comprehensive_report.pdf
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

STATS_DIR = Path("statistical_results")
REG_DIR = STATS_DIR / "regression"
OUT_PDF = STATS_DIR / "comprehensive_report.pdf"

# ── Colour palette ────────────────────────────────────────────────────
C_NB = "#1f77b4"
C_AB = "#d62728"
C_HEADER = "#2c3e50"
C_SUBHEADER = "#2980b9"
C_PASS = "#27ae60"
C_FAIL = "#e74c3c"
C_WARN = "#f39c12"

HYPOTHESIS_LABELS = {
    "H1": "H1: Overall recognition accuracy (NB > AB)",
    "H2": "H2: BB-frame accuracy (NB > AB)",
    "H3": "H3: Mean confidence rating (NB > AB)",
    "H4": "H4: Proportion low-confidence trials (AB > NB)",
    "H5": "H5: Movie-level accuracy variability (AB > NB)",
    "H6": "H6: BB vs EM mean log RT (paired, non-directional)",
    "H7": "H7: Condition × Frame accuracy interaction (NB shows larger BB advantage)",
    "H8": "H8: Condition × Frame confidence interaction (NB shows larger BB conf advantage)",
}


def load_results() -> dict:
    path = STATS_DIR / "hypothesis_test_results.json"
    with open(path) as f:
        return json.load(f)


def load_regression_summary() -> str:
    path = REG_DIR / "regression_summary.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_synthesis() -> str:
    path = REG_DIR / "results_synthesis.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── Low-level drawing helpers ─────────────────────────────────────────
def clear_axes(fig):
    for ax in fig.get_axes():
        ax.set_visible(False)


def text_page(pdf: PdfPages, lines: list[str], title: str = "",
              font_size: float = 9.5, title_size: float = 14) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.02, 0.90, 0.95])
    ax.axis("off")
    y = 0.97
    if title:
        ax.text(0.5, y, title, transform=ax.transAxes,
                fontsize=title_size, fontweight="bold", ha="center", va="top",
                color=C_HEADER)
        y -= 0.045
        ax.plot([0.0, 1.0], [y + 0.01, y + 0.01], color=C_SUBHEADER, linewidth=1.5,
                transform=ax.transAxes)
        y -= 0.01

    line_height = font_size / (11 * 72) * 1.55
    for line in lines:
        if y < 0.03:
            break
        bold = line.startswith("###") or line.startswith("##") or line.startswith("**")
        clean = line.lstrip("#* ").rstrip("*")
        weight = "bold" if bold else "normal"
        size = font_size + 1 if bold else font_size
        ax.text(0.0, y, clean, transform=ax.transAxes,
                fontsize=size, fontweight=weight, va="top",
                color=C_HEADER if bold else "black",
                fontfamily="monospace" if not bold else "sans-serif")
        y -= line_height * (1.3 if bold else 1.0)

    pdf.savefig(fig)
    plt.close(fig)


def section_page(pdf: PdfPages, title: str, subtitle: str = "") -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_HEADER)
    ax.axis("off")
    ax.text(0.5, 0.55, title, transform=ax.transAxes,
            fontsize=22, fontweight="bold", ha="center", va="center", color="white")
    if subtitle:
        ax.text(0.5, 0.44, subtitle, transform=ax.transAxes,
                fontsize=13, ha="center", va="center", color="#bdc3c7")
    pdf.savefig(fig)
    plt.close(fig)


def sig_label(p: float, alpha: float = 0.05) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


def decision_colour(p_one: float | None, observed_matches: bool | None) -> str:
    if p_one is None or np.isnan(p_one):
        return C_WARN
    if p_one < 0.05 and observed_matches:
        return C_PASS
    if p_one < 0.05 and not observed_matches:
        return C_WARN
    return C_FAIL


# ── Title page ────────────────────────────────────────────────────────
def title_page(pdf: PdfPages, results: dict) -> None:
    meta = results.get("_meta", {})
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_HEADER)
    ax.axis("off")

    ax.text(0.5, 0.82, "BRSM Movie Memory Experiment",
            transform=ax.transAxes, fontsize=26, fontweight="bold",
            ha="center", color="white")
    ax.text(0.5, 0.74, "Comprehensive Statistical Report",
            transform=ax.transAxes, fontsize=18, ha="center", color="#bdc3c7")
    ax.plot([0.1, 0.9], [0.68, 0.68], color="#bdc3c7", linewidth=1,
            transform=ax.transAxes)

    info = [
        f"Participants analysed: {meta.get('n_analysed', '?')}",
        f"Trials analysed: {meta.get('n_trials', '?')}",
        f"Vigilance exclusions: {', '.join(meta.get('vigilance_exclusions', []))}",
        f"Repeated-video IDs excluded: {meta.get('repeated_video_ids_excluded', [])}",
        f"(5 movies × 2 conditions shown twice as vigilance checks — excluded)",
        "",
        "Conditions:  NB = Natural Boundaries  |  AB = Abrupt Boundaries",
        "Frame types: BB = Before-Boundary     |  EM = Event-Middle",
        "",
        "Hypotheses tested: H1 – H8",
        "Models: Binomial GLM · GEE (8 nested) · GLMM (Variational Bayes)",
        "",
        "Generated by: generate_comprehensive_report.py",
    ]
    y = 0.62
    for line in info:
        ax.text(0.5, y, line, transform=ax.transAxes,
                fontsize=11, ha="center", color="white" if line else "#bdc3c7")
        y -= 0.038

    pdf.savefig(fig)
    plt.close(fig)


# ── Executive summary ─────────────────────────────────────────────────
def executive_summary(pdf: PdfPages, results: dict) -> None:
    primary = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]
    rows = []
    for h in primary:
        r = results.get(h, {})
        if not r:
            continue
        p1 = r.get("p_one_sided")
        obs = r.get("observed_direction_matches")
        if p1 is None:
            p1 = r.get("p_two_sided")
        if p1 is None:
            p1 = np.nan
        supported = (not np.isnan(p1)) and p1 < 0.05 and (obs is True or obs is None)
        rows.append({
            "H": h,
            "label": HYPOTHESIS_LABELS.get(h, h),
            "p_one": p1,
            "supported": supported,
            "obs_matches": obs,
        })

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.97, "Executive Summary", fontsize=16, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes, color=C_HEADER)
    ax.plot([0.0, 1.0], [0.955, 0.955], color=C_SUBHEADER, lw=1.5,
            transform=ax.transAxes)

    y = 0.925
    for row in rows:
        col = C_PASS if row["supported"] else C_FAIL
        mark = "✓ Supported" if row["supported"] else "✗ Not supported"
        p_str = f"p = {row['p_one']:.4f}" if not np.isnan(row["p_one"]) else "p = n/a"
        ax.text(0.02, y, row["H"], transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", color=col)
        ax.text(0.10, y, row["label"].split(":", 1)[1].strip(), transform=ax.transAxes,
                fontsize=9, va="top", color="black")
        ax.text(0.73, y, p_str, transform=ax.transAxes, fontsize=9,
                va="top", color="black", ha="right")
        ax.text(0.98, y, mark, transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color=col, fontweight="bold")
        y -= 0.065
        ax.plot([0.0, 1.0], [y + 0.025, y + 0.025], color="#ecf0f1", lw=0.8,
                transform=ax.transAxes)

    # Legend
    ax.text(0.02, y - 0.02, "Note: One-sided p-value used for directional hypotheses.",
            transform=ax.transAxes, fontsize=8, color="#7f8c8d", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


# ── Mann-Whitney U justification ──────────────────────────────────────
def mann_whitney_justification_page(pdf: PdfPages, results: dict) -> None:
    section_page(pdf, "Methodological Note",
                 "Justification for Mann-Whitney U with Bounded DVs")

    lines = [
        "## Why Mann-Whitney U Is Appropriate for Bounded Dependent Variables",
        "",
        "Several DVs in this study are bounded proportions or Likert means:",
        "  • Overall accuracy:  k / 35 ∈ [0, 1]  (35 unique movies after excluding repeats)",
        "  • BB / EM accuracy:  approximately k / 17 or k / 18  (half of 35)",
        "  • Mean confidence:   mean of 35 Likert responses, each ∈ {1, 2, 3, 4, 5}",
        "  • Prop. low-confidence: proportion of trials with conf ≤ 3",
        "",
        "### Problem: Normality fails systematically for bounded DVs",
        "",
        "Bounded scales impose hard ceilings and floors that produce asymmetric",
        "distributions whenever participants cluster toward an extreme.  In this",
        "experiment, mean accuracy ≈ 0.85–0.90 — well above chance (0.5) and close",
        "to the ceiling of 1.0.  Shapiro-Wilk tests confirmed significant departures",
        "from normality for accuracy and confidence DVs in both conditions.",
        "",
        "Transformations such as arcsin(√p) or log-odds can reduce boundary effects",
        "but rarely restore normality at these accuracy levels, and they complicate",
        "interpretation of group differences.",
        "",
        "### Why Mann-Whitney U addresses this correctly",
        "",
        "(a)  Distribution-free.  Mann-Whitney makes no parametric assumption about",
        "     the shape of the distribution — it works on ranks.  This is exactly",
        "     appropriate when normality is violated by design (bounded scale).",
        "",
        "(b)  Directly addresses the research question.  The test evaluates whether",
        "     a randomly chosen NB participant is more likely to outperform a randomly",
        "     chosen AB participant (stochastic dominance).  This is a clinically",
        "     and cognitively meaningful question that does not require distributional",
        "     assumptions.",
        "",
        "(c)  Good power at n ≈ 83–87.  With sample sizes ≈ 83–87 per condition, the",
        "     Normal approximation to the U statistic is accurate (z-statistic reported).",
        "     Under non-normality, Mann-Whitney has HIGHER power than the t-test.",
        "",
        "(d)  Effect size is scale-invariant.  Rank-biserial r (Kerby, 2014) maps",
        "     directly onto the probability of superiority: r_rb = 0.30 means 65% of",
        "     NB participants exceed a randomly chosen AB participant.  It is bounded",
        "     in [-1, 1] and needs no variance estimate, making it robust to ceiling",
        "     effects that inflate or deflate variance-based effect sizes like Cohen's d.",
        "",
        "### Formula",
        "",
        "  r_rb = (2 × U₁) / (n₁ × n₂)  −  1",
        "",
        "where U₁ is the Mann-Whitney statistic for group 1, and n₁, n₂ are the",
        "respective sample sizes.  Positive r_rb indicates group 1 tends to be",
        "larger; r_rb = 0 indicates no stochastic dominance.",
        "",
        "### Reference",
        "",
        "  Kerby, D. S. (2014). The simple difference formula: An approach to",
        "  teaching nonparametric correlation. Comprehensive Psychology, 3, 11.",
        "  doi:10.2466/11.IT.3.1",
    ]

    text_page(pdf, lines, title="", font_size=9)


# ── Per-hypothesis result page ─────────────────────────────────────────
def hypothesis_page(pdf: PdfPages, h: str, r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.5),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor("white")

    ax_text = axes[0]
    ax_bar = axes[1]
    ax_text.axis("off")
    ax_bar.axis("off")

    # ── Left: result text ──────────────────────────────────────────────
    label = HYPOTHESIS_LABELS.get(h, h)
    p1 = r.get("p_one_sided")
    obs = r.get("observed_direction_matches")
    test = r.get("test_used", "—")
    stat_name = r.get("statistic_name", "stat")
    stat_val = r.get("statistic")
    df_val = r.get("df")
    p2 = r.get("p_two_sided")
    eff_name = r.get("effect_size_name", "")
    eff_val = r.get("effect_size_value")
    z_approx = r.get("z_approx")
    g1 = r.get("group1", "G1")
    g2 = r.get("group2", "G2")
    d1 = r.get("descriptives", {}).get(g1, {})
    d2 = r.get("descriptives", {}).get(g2, {})

    if p1 is None:
        p1 = p2
    supported = p1 is not None and p1 < 0.05 and obs is True
    dec_col = C_PASS if supported else C_FAIL
    decision = "SUPPORTED" if supported else "NOT SUPPORTED"

    y = 0.97
    ax_text.text(0.0, y, label, transform=ax_text.transAxes,
                 fontsize=10, fontweight="bold", va="top", color=C_HEADER,
                 wrap=True)
    y -= 0.10

    ax_text.text(0.0, y, f"Decision:  {decision}  (α = 0.05, one-tailed)",
                 transform=ax_text.transAxes, fontsize=10, fontweight="bold",
                 va="top", color=dec_col)
    y -= 0.09

    # Descriptives
    def _fmt(d, name):
        return (f"{name}: n={d.get('n','?')}, "
                f"M={d.get('mean', float('nan')):.4f}, "
                f"SD={d.get('sd', float('nan')):.4f}, "
                f"Mdn={d.get('median', float('nan')):.4f}")

    for line in [_fmt(d1, g1), _fmt(d2, g2)]:
        ax_text.text(0.0, y, line, transform=ax_text.transAxes,
                     fontsize=8.5, va="top", color="black")
        y -= 0.07

    y -= 0.02

    # Normality
    norm = r.get("normality", {})
    if norm:
        ax_text.text(0.0, y, "Normality (Shapiro-Wilk):", transform=ax_text.transAxes,
                     fontsize=8.5, va="top", fontweight="bold")
        y -= 0.065
        for grp in [g1, g2]:
            n_info = norm.get(grp, {})
            sw_p = n_info.get("p", float("nan"))
            normal = sw_p >= 0.05 if not np.isnan(sw_p) else False
            ax_text.text(0.04, y,
                         f"{grp}: W={n_info.get('W', n_info.get('shapiro_W', float('nan'))):.4f}, "
                         f"p={sw_p:.4g}  ({'normal' if normal else 'non-normal'})",
                         transform=ax_text.transAxes, fontsize=8.5, va="top")
            y -= 0.065

    # Levene
    lev = r.get("levene", {})
    if lev:
        lev_p = lev.get("p", float("nan"))
        ax_text.text(0.0, y,
                     f"Levene (Brown-Forsythe): W={lev.get('W', float('nan')):.4f}, "
                     f"p={lev_p:.4g}  ({'equal var' if lev_p >= 0.05 else 'unequal var'})",
                     transform=ax_text.transAxes, fontsize=8.5, va="top")
        y -= 0.07

    ax_text.text(0.0, y, f"Test chosen: {test}", transform=ax_text.transAxes,
                 fontsize=8.5, va="top", style="italic", color="#555")
    y -= 0.07

    # Test result
    stat_line = f"{stat_name} = {stat_val:.4f}" if stat_val is not None else ""
    if df_val is not None:
        stat_line += f",  df = {df_val:.1f}"
    if z_approx is not None:
        stat_line += f",  z ≈ {z_approx:.4f}"
    ax_text.text(0.0, y, stat_line, transform=ax_text.transAxes,
                 fontsize=8.5, va="top")
    y -= 0.065

    ax_text.text(0.0, y,
                 f"p (two-sided) = {p2:.4g}    p (one-sided) = {p1:.4g}" if p1 is not None else
                 f"p (two-sided) = {p2:.4g}",
                 transform=ax_text.transAxes, fontsize=8.5, va="top")
    y -= 0.065

    ax_text.text(0.0, y, f"Effect size ({eff_name}) = {eff_val:.4f}" if eff_val is not None else "",
                 transform=ax_text.transAxes, fontsize=8.5, va="top", fontweight="bold")

    # ── Right: bar chart ───────────────────────────────────────────────
    ax_bar.set_visible(True)
    means = [d1.get("mean", 0), d2.get("mean", 0)]
    sds = [d1.get("sd", 0), d2.get("sd", 0)]
    bars = ax_bar.bar([g1, g2], means, color=[C_NB, C_AB], alpha=0.8,
                      edgecolor="black", linewidth=0.8)
    ax_bar.errorbar([0, 1], means, yerr=sds, fmt="none", color="black",
                    capsize=5, linewidth=1.2)
    for bar, mean in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, mean * 1.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Mean ± SD")
    ax_bar.set_title(f"[{h}] Group means", fontsize=9, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Hypothesis {h}", fontsize=13, fontweight="bold", color=C_HEADER, y=1.01)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── H6 paired page ───────────────────────────────────────────────────
def h6_page(pdf: PdfPages, r: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    label = HYPOTHESIS_LABELS["H6"]
    ax.text(0.5, 0.97, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="top", color=C_HEADER)

    p2 = r.get("p_two_sided")
    test = r.get("test_used", "—")
    stat_name = r.get("statistic_name", "stat")
    stat_val = r.get("statistic")
    df_val = r.get("df")
    eff_name = r.get("effect_size_name", "")
    eff_val = r.get("effect_size_value")
    n_pairs = r.get("n_pairs")
    desc = r.get("descriptives", {})
    d_bb = desc.get("BB", {})
    d_em = desc.get("EM", {})
    d_diff = desc.get("difference", {})
    norm = r.get("normality_of_diffs", {})

    supported = p2 is not None and p2 < 0.05
    dec_col = C_PASS if supported else C_FAIL
    decision = "SIGNIFICANT" if supported else "NOT SIGNIFICANT"

    lines = [
        f"Decision: {decision}  (two-tailed, α = 0.05)",
        f"n pairs = {n_pairs}",
        f"BB log RT:  M={d_bb.get('mean', float('nan')):.4f}, SD={d_bb.get('sd', float('nan')):.4f}",
        f"EM log RT:  M={d_em.get('mean', float('nan')):.4f}, SD={d_em.get('sd', float('nan')):.4f}",
        f"Difference (BB−EM):  M={d_diff.get('mean', float('nan')):.4f}, "
        f"SD={d_diff.get('sd', float('nan')):.4f}",
        f"Normality of diffs: Shapiro W={norm.get('shapiro_W', float('nan')):.4f}, "
        f"p={norm.get('p', float('nan')):.4g}",
        f"Test: {test}",
        f"{stat_name} = {stat_val:.4f}" + (f",  df = {df_val:.1f}" if df_val else ""),
        f"p (two-sided) = {p2:.4g}",
        f"Effect ({eff_name}) = {eff_val:.4f}" if eff_val is not None else "",
        "",
        "This hypothesis tests whether BB-frame targets take longer to recognise than",
        "EM-frame targets within participants (regardless of condition).  A significant",
        "result would suggest boundary-related frames require more retrieval effort.",
    ]

    y = 0.83
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        colour = dec_col if i == 0 else "black"
        ax.text(0.05, y, line, transform=ax.transAxes,
                fontsize=9, va="top", fontweight=weight, color=colour)
        y -= 0.065

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Hypothesis summary table ─────────────────────────────────────────
def hypothesis_summary_table(pdf: PdfPages, results: dict) -> None:
    csv_path = STATS_DIR / "hypothesis_test_summary.csv"
    df = load_csv_safe(csv_path)
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.98, "Hypothesis Test Summary Table",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            ha="center", va="top", color=C_HEADER)

    cols = ["label", "test_used", "statistic", "df", "p_two_sided", "p_one_sided",
            "effect_size_name", "effect_size_value"]
    avail = [c for c in cols if c in df.columns]
    sub = df[avail].copy()

    for col in ["statistic", "p_two_sided", "p_one_sided", "effect_size_value"]:
        if col in sub.columns:
            sub[col] = sub[col].apply(lambda x: f"{float(x):.4g}" if pd.notna(x) else "—")
    if "df" in sub.columns:
        sub["df"] = sub["df"].apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—")

    table = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.02, 1, 0.93],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(len(sub.columns))))

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Regression overview ───────────────────────────────────────────────
def regression_overview_page(pdf: PdfPages) -> None:
    section_page(pdf, "Regression Models",
                 "Model 3 (GLM)  ·  Model 1 (GEE)  ·  GLMM (VB)")

    lines = [
        "## Overview of Regression Modelling Strategy",
        "",
        "Three complementary models are used to examine predictors of trial-level",
        "recognition accuracy after excluding repeated (vigilance) video frames.",
        "",
        "### Model 3 — Participant-level binomial GLM",
        "",
        "  Unit of analysis: participant (each contributes one success/failure ratio).",
        "  DV: (n_correct, n_incorrect) ~ Binomial(n_trials).",
        "  Predictors: condition, age, gender.",
        "  Purpose: replicates a simple between-subjects analysis analogous to a",
        "  t-test but respects the binomial nature of the DV.",
        "",
        "### Model 1 — Trial-level GEE (Generalised Estimating Equations)",
        "",
        "  Unit of analysis: individual trial (binary correct/incorrect).",
        "  Clustering: trials nested within participants (exchangeable correlation).",
        "  Robust standard errors: GEE accounts for within-participant correlation",
        "  without specifying the full random-effects distribution.",
        "  Eight nested models are compared by QIC (Quasi-Information Criterion).",
        "  Predictors vary across models: condition, target_type, interaction,",
        "  demographics (age, gender, handedness, vision), movie_duration.",
        "",
        "### GLMM — Binomial mixed model (Variational Bayes)",
        "",
        "  Unit of analysis: individual trial.",
        "  Random effect: participant-level random intercept ~ N(0, σ²).",
        "  Fixed effects: condition, target_type, and their interaction.",
        "  Estimation: Variational Bayes (statsmodels BinomialBayesMixedGLM.fit_vb()).",
        "  VB is preferred over Laplace (fit_map) because it more reliably estimates",
        "  the random-effect variance σ in balanced designs.",
        "  The posterior SD of fixed effects plays the role of the standard error.",
        "  Approximate two-tailed p-values use z = post_mean / post_SD.",
        "",
        "### Why two model families?",
        "",
        "  GEE targets population-average effects: the estimated OR answers",
        "  'On average across all participants, how much does condition affect P(correct)?'",
        "",
        "  GLMM targets subject-specific effects: the estimated OR answers",
        "  'For a given participant, how much does condition affect their P(correct)?'",
        "",
        "  Both families should yield the same direction of effects; discrepancies",
        "  reveal heterogeneity across participants (large σ in GLMM).",
        "  For between-subjects designs with bounded binary outcomes, both approaches",
        "  are valid and complementary.",
    ]
    text_page(pdf, lines, title="", font_size=9)


# ── Interaction effects deep-dive ─────────────────────────────────────
def interaction_effects_page(pdf: PdfPages) -> None:
    lines = [
        "## Understanding Interaction Effects in Logistic Regression",
        "",
        "An interaction term in a logistic model means the effect of one predictor",
        "on the LOG-ODDS of a correct response CHANGES depending on the level of",
        "another predictor.",
        "",
        "### The condition × target_type interaction",
        "",
        "The term  C(condition)[T.AB] : C(target_type)[T.EM]  tests whether the",
        "difference in log-odds between EM and BB targets is the SAME in both",
        "the AB and NB conditions.",
        "",
        "### Interpreting the coefficient",
        "",
        "Let β₁ = coefficient for condition (AB vs NB, reference = NB)",
        "    β₂ = coefficient for target type (EM vs BB, reference = BB)",
        "    β₃ = interaction coefficient",
        "",
        "Then the model's log-odds are:",
        "  NB + BB:  intercept                       (reference cell)",
        "  NB + EM:  intercept + β₂                 (EM advantage in NB)",
        "  AB + BB:  intercept + β₁                 (AB penalty for BB)",
        "  AB + EM:  intercept + β₁ + β₂ + β₃      (combined effects)",
        "",
        "So β₃ measures how the EM advantage CHANGES when moving from NB to AB.",
        "  β₃ > 0:  EM advantage is LARGER in AB than in NB",
        "           (AB condition hurts BB recall more than EM recall)",
        "  β₃ < 0:  EM advantage is SMALLER in AB (or BB advantage is larger in AB)",
        "  β₃ ≈ 0:  condition does not moderate the BB-vs-EM difference",
        "",
        "### Odds ratios for interactions are 'ratio of ratios'",
        "",
        "OR for the interaction = exp(β₃) = (OR_EM/BB in AB) / (OR_EM/BB in NB)",
        "",
        "Example: if OR_EM/BB in NB = 1.40 and OR_EM/BB in AB = 1.20, then",
        "  the interaction OR = 1.20 / 1.40 = 0.857  (β₃ < 0).",
        "This means the EM advantage shrinks when moving from NB to AB.",
        "",
        "### Simple (conditional) effects from the interaction model",
        "",
        "  Effect of CONDITION for BB targets:   exp(β₁)          [AB vs NB, BB only]",
        "  Effect of CONDITION for EM targets:   exp(β₁ + β₃)    [AB vs NB, EM only]",
        "  Effect of TARGET TYPE in NB:          exp(β₂)          [EM vs BB, NB only]",
        "  Effect of TARGET TYPE in AB:          exp(β₂ + β₃)    [EM vs BB, AB only]",
        "",
        "These simple effects answer 'what is the effect of X when Y is held constant?'",
        "",
        "### Convergence between GEE and GLMM interaction estimates",
        "",
        "Because GEE estimates population-average ORs and GLMM estimates",
        "subject-specific ORs, their interaction coefficients will differ in magnitude",
        "but should agree in sign and significance.  A large discrepancy suggests",
        "substantial between-participant heterogeneity in the moderation effect.",
    ]
    text_page(pdf, lines, title="Interaction Effects — Interpretation Guide", font_size=9)


# ── GEE model comparison table ────────────────────────────────────────
def gee_model_comparison_page(pdf: PdfPages) -> None:
    df = load_csv_safe(REG_DIR / "model1_model_comparison_qic.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.text(0.5, 0.97, "GEE Model Comparison — QIC (lower = better)",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="center", va="top", color=C_HEADER)

    cols = ["model", "qic", "stable", "description"]
    avail = [c for c in cols if c in df.columns]
    sub = df[avail].copy()
    if "qic" in sub.columns:
        sub["qic"] = sub["qic"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "—")

    table = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        cellLoc="left",
        loc="center",
        bbox=[0, 0.02, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(col=list(range(len(sub.columns))))
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def gee_odds_ratios_page(pdf: PdfPages, model_name: str, title_suffix: str = "") -> None:
    df = load_csv_safe(REG_DIR / f"{model_name}_odds_ratios.csv")
    if df is None:
        return

    fig, (ax_tbl, ax_plot) = plt.subplots(1, 2, figsize=(11, 6),
                                           gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle(f"GEE: {model_name}{' — ' + title_suffix if title_suffix else ''}",
                 fontsize=11, fontweight="bold", color=C_HEADER)

    ax_tbl.axis("off")
    cols = ["term", "coef", "se", "p", "OR", "OR_ci_low", "OR_ci_high"]
    avail = [c for c in cols if c in df.columns]
    sub = df[avail].copy()
    for col in ["coef", "se", "OR", "OR_ci_low", "OR_ci_high"]:
        if col in sub.columns:
            sub[col] = sub[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
    if "p" in sub.columns:
        sub["p"] = sub["p"].apply(lambda x: f"{float(x):.4g}" if pd.notna(x) else "—")

    table = ax_tbl.table(
        cellText=sub.values, colLabels=sub.columns,
        cellLoc="center", loc="center", bbox=[0, 0.0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(len(sub.columns))))
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")

    # Forest plot
    ax_plot.axis("on")
    df_plot = df[df["term"] != "Intercept"].copy() if "term" in df.columns else df.copy()
    for col in ["OR", "OR_ci_low", "OR_ci_high"]:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")
    df_plot = df_plot.dropna(subset=["OR"])

    y_pos = range(len(df_plot))
    ax_plot.errorbar(
        df_plot["OR"], y_pos,
        xerr=[df_plot["OR"] - df_plot["OR_ci_low"],
              df_plot["OR_ci_high"] - df_plot["OR"]],
        fmt="o", color=C_SUBHEADER, capsize=4, ms=5,
    )
    ax_plot.axvline(1.0, color="red", lw=1.2, linestyle="--", alpha=0.7)
    ax_plot.set_yticks(list(y_pos))
    ax_plot.set_yticklabels(df_plot["term"].tolist(), fontsize=7)
    ax_plot.set_xlabel("Odds Ratio (95% CI)", fontsize=9)
    ax_plot.set_title("Forest plot", fontsize=9)
    ax_plot.grid(axis="x", alpha=0.3)
    ax_plot.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── GLMM results page ─────────────────────────────────────────────────
def glmm_results_page(pdf: PdfPages) -> None:
    core_df = load_csv_safe(REG_DIR / "glmm_core_fixed_effects.csv")
    int_df = load_csv_safe(REG_DIR / "glmm_interaction_fixed_effects.csv")
    full_df = load_csv_safe(REG_DIR / "glmm_full_fixed_effects.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    fig.suptitle("GLMM — Fixed Effects (Variational Bayes, random participant intercept)\n"
                 "Repeated movies excluded",
                 fontsize=11, fontweight="bold", color=C_HEADER)

    for ax, df, title in zip(axes,
                              [core_df, int_df, full_df],
                              ["Core\n(main effects)", "Interaction\n(cond × type)",
                               "Full\n(+ age, gender, duration)"]):
        ax.axis("off")
        if df is None:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(title, fontsize=9, fontweight="bold")
            continue

        cols = ["term", "post_mean", "post_sd", "OR"]
        avail = [c for c in cols if c in df.columns]
        sub = df[avail].copy()
        for col in ["post_mean", "post_sd", "OR"]:
            if col in sub.columns:
                sub[col] = sub[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")

        table = ax.table(
            cellText=sub.values, colLabels=sub.columns,
            cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(sub.columns))))
        for (row, col_idx), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(C_SUBHEADER)
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f5f5")
            cell.set_edgecolor("#cccccc")

        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def glmm_interpretation_page(pdf: PdfPages) -> None:
    core_df = load_csv_safe(REG_DIR / "glmm_core_fixed_effects.csv")
    int_df = load_csv_safe(REG_DIR / "glmm_interaction_fixed_effects.csv")

    lines = ["## GLMM Results — Narrative Interpretation", ""]

    if core_df is not None:
        lines.append("### Core model (main effects only)")
        for _, row in core_df.iterrows():
            if "Intercept" in str(row.get("term", "")):
                continue
            lines.append(
                f"  {row['term']}:  post_mean = {row['post_mean']:.4f}, "
                f"post_SD = {row['post_sd']:.4f}, OR = {row['OR']:.4f}"
            )
        lines += [
            "",
            "Interpretation of main effects:",
            "  • Condition (AB vs NB): OR < 1 means AB participants have lower odds",
            "    of a correct trial than NB participants.",
            "  • Target type (EM vs BB): OR > 1 means EM targets are recognised more",
            "    accurately than BB targets on average across conditions.",
            "  • The random intercept SD (σ) quantifies between-participant variability",
            "    in log-odds accuracy after accounting for fixed effects.",
            "    Larger σ = more individual differences not explained by condition/frame.",
        ]

    if int_df is not None:
        lines += ["", "### Interaction model (condition × target_type)"]
        for _, row in int_df.iterrows():
            if "Intercept" in str(row.get("term", "")):
                continue
            lines.append(
                f"  {row['term']}:  OR = {row['OR']:.4f}"
            )
        lines += [
            "",
            "The interaction term (condition[T.AB]:target_type[T.EM]) tests whether",
            "the EM advantage over BB changes between conditions.",
            "  OR > 1: EM advantage is LARGER in AB (boundary disruption hurts BB more)",
            "  OR < 1: EM advantage is SMALLER in AB (boundary disruption is uniform)",
            "  OR ≈ 1: no moderation — condition and frame type act independently",
            "",
            "Simple effects can be derived by combining coefficients:",
            "  Cond effect for BB:  exp(β_condition)",
            "  Cond effect for EM:  exp(β_condition + β_interaction)",
            "  Frame effect in NB:  exp(β_target_type)",
            "  Frame effect in AB:  exp(β_target_type + β_interaction)",
        ]

    text_page(pdf, lines, title="", font_size=9)


# ── Diagnostics pages ─────────────────────────────────────────────────
def diagnostics_page(pdf: PdfPages) -> None:
    plots = [
        REG_DIR / "diagnostic_ols_residuals_vs_fitted.png",
        REG_DIR / "diagnostic_logistic_pearson_residuals.png",
        STATS_DIR / "rt_normality_participant_level.png",
        REG_DIR / "eda_accuracy_by_condition_target_type.png",
    ]
    titles = [
        "OLS surrogate: residuals vs fitted",
        "GEE logistic: Pearson residuals vs fitted",
        "RT normality: participant-level",
        "EDA: accuracy by condition and frame type",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Model Diagnostics", fontsize=13, fontweight="bold", color=C_HEADER)
    for ax, path, title in zip(axes.flatten(), plots, titles):
        ax.axis("off")
        if Path(path).exists():
            img = plt.imread(str(path))
            ax.imshow(img)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def vif_page(pdf: PdfPages) -> None:
    df = load_csv_safe(REG_DIR / "vif_table.csv")
    bp = load_csv_safe(REG_DIR / "diagnostic_breusch_pagan.csv")
    if df is None:
        return

    fig, (ax_tbl, ax_info) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Collinearity & Heteroscedasticity Diagnostics",
                 fontsize=12, fontweight="bold", color=C_HEADER)

    ax_tbl.axis("off")
    sub = df[["variable", "VIF"]].copy()
    sub["VIF"] = sub["VIF"].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
    table = ax_tbl.table(
        cellText=sub.values, colLabels=sub.columns,
        cellLoc="center", loc="center", bbox=[0.05, 0.05, 0.9, 0.9],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(col=list(range(len(sub.columns))))
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_HEADER)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#cccccc")
    ax_tbl.set_title("VIF Table (VIF < 5 = acceptable)", fontsize=10, fontweight="bold")

    ax_info.axis("off")
    if bp is not None and len(bp) > 0:
        row = bp.iloc[0]
        info = [
            "Breusch-Pagan Test (OLS surrogate on participant accuracy)",
            f"  LM statistic: {row.get('bp_lm', float('nan')):.4f}",
            f"  LM p-value:   {row.get('bp_lm_p', float('nan')):.4g}",
            f"  F statistic:  {row.get('bp_f', float('nan')):.4f}",
            f"  F p-value:    {row.get('bp_f_p', float('nan')):.4g}",
            f"  R²:           {row.get('r2', float('nan')):.4f}",
            f"  n:            {row.get('n', '?')}",
            "",
            "Interpretation:",
            "  p > 0.05 → fail to reject homoscedasticity",
            "  (Note: BP is for linear models; our primary",
            "   model is logistic GEE — this is indicative only)",
        ]
        y = 0.93
        for line in info:
            bold = not line.startswith(" ") and line != ""
            ax_info.text(0.05, y, line, transform=ax_info.transAxes,
                         fontsize=9, va="top", fontweight="bold" if bold else "normal")
            y -= 0.085
    ax_info.set_title("Heteroscedasticity", fontsize=10, fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading results...")
    results = load_results()

    print(f"Generating report -> {OUT_PDF}")
    with PdfPages(str(OUT_PDF)) as pdf:

        # ── Title & overview ─────────────────────────────────────────
        title_page(pdf, results)
        executive_summary(pdf, results)

        # ── Methodological notes ──────────────────────────────────────
        mann_whitney_justification_page(pdf, results)

        # ── Individual hypothesis pages ───────────────────────────────
        section_page(pdf, "Hypothesis Tests", "H1 – H8  |  Repeated movies excluded")
        for h in ["H1", "H2", "H3", "H4", "H5", "H7", "H8"]:
            r = results.get(h)
            if r:
                hypothesis_page(pdf, h, r)

        r6 = results.get("H6")
        if r6:
            h6_page(pdf, r6)

        hypothesis_summary_table(pdf, results)

        # ── Regression section ────────────────────────────────────────
        regression_overview_page(pdf)
        interaction_effects_page(pdf)

        section_page(pdf, "GEE Results", "8 nested models compared by QIC")
        gee_model_comparison_page(pdf)

        # Best model + interaction model OR tables
        mc = load_csv_safe(REG_DIR / "model1_model_comparison_qic.csv")
        if mc is not None:
            stable = mc[mc.get("stable", pd.Series(True, index=mc.index)).astype(bool)]
            stable_qic = stable.dropna(subset=["qic"])
            best_name = (stable_qic.sort_values("qic").iloc[0]["model"]
                         if len(stable_qic) else "M1_core_main")
        else:
            best_name = "M1_core_main"

        gee_odds_ratios_page(pdf, best_name, "best model by QIC")
        gee_odds_ratios_page(pdf, "M1_core_interaction", "core interaction model")
        gee_odds_ratios_page(pdf, "M1_stim_interaction", "interaction + stimulus duration")

        section_page(pdf, "GLMM Results", "Variational Bayes  |  Random participant intercept")
        glmm_results_page(pdf)
        glmm_interpretation_page(pdf)

        # ── Diagnostics ───────────────────────────────────────────────
        section_page(pdf, "Model Diagnostics", "Residuals · Normality · VIF · Heteroscedasticity")
        diagnostics_page(pdf)
        vif_page(pdf)

        # ── Closing ───────────────────────────────────────────────────
        lines = [
            "## Summary of Key Findings",
            "",
            "Repeated-video exclusion",
            "  All analyses exclude movie IDs 3, 7, 18, 28, 37 (shown twice as vigilance",
            "  checks) to prevent inflated accuracy estimates from double exposure.",
            "",
            "H1 – H2 (Recognition accuracy)",
            "  Test whether NB condition produces better recognition than AB, overall",
            "  and specifically for Before-Boundary frames.",
            "",
            "H3 – H4 (Confidence)",
            "  Test whether NB participants are more confident and have fewer low-",
            "  confidence responses than AB participants.",
            "",
            "H5 (Variability)",
            "  Tests whether AB shows greater movie-level variance in accuracy,",
            "  suggesting inconsistent boundary disruption across stimuli.",
            "",
            "H6 (RT)",
            "  Tests whether BB-frame targets take longer to retrieve than EM-frame",
            "  targets, regardless of condition.",
            "",
            "H7 – H8 (Interactions)",
            "  Tests whether condition MODERATES the BB-vs-EM difference in accuracy",
            "  and confidence.  A significant interaction means the frame-type effect",
            "  is not the same in both encoding conditions.",
            "",
            "Mann-Whitney U",
            "  Used when Shapiro-Wilk rejects normality (common for bounded proportions",
            "  near the ceiling).  Appropriate, distribution-free, and powerful at",
            "  n ≈ 83–87.  Effect sizes reported as rank-biserial r.",
            "",
            "GEE vs GLMM",
            "  GEE: population-average effects, robust SEs, no distributional assumption",
            "       on random effects.",
            "  GLMM: subject-specific effects, random intercept σ quantifies individual",
            "        differences, Variational Bayes estimation.",
            "  Both methods should converge on the same directional conclusions.",
        ]
        text_page(pdf, lines, title="Discussion & Conclusions", font_size=9.5)

    print(f"Report saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
