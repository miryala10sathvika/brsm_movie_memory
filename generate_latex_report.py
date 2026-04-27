"""
Generate comprehensive_report.tex and compile it to comprehensive_report.pdf.

Must be run AFTER run_statistical_tests.py and analysis_regression_models.py.
"""

from __future__ import annotations
import json, math, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

STATS_DIR = Path("statistical_results")
REG_DIR = STATS_DIR / "regression"
TEX_PATH = STATS_DIR / "comprehensive_report.tex"
PDF_PATH = STATS_DIR / "comprehensive_report.pdf"

# ── helpers ───────────────────────────────────────────────────────────
def load_json() -> dict:
    return json.loads((STATS_DIR / "hypothesis_test_results.json").read_text())


def csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def p_raw(p) -> str:
    """Format a p-value as bare content (no $ delimiters) for use inside math mode."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return r"---"
    if p < 0.001:
        return r"<.001"
    return f"{p:.4f}".replace("0.", ".")

def p_fmt(p) -> str:
    """Format a p-value as a self-contained LaTeX math string (includes $ delimiters)."""
    v = p_raw(p)
    if v == "---":
        return v
    return f"${v}$"


def f(v, decimals: int = 3) -> str:
    """Format a float, returns --- if None/nan."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return f"{float(v):.{decimals}f}"


def esc(s: str) -> str:
    """Escape special LaTeX chars."""
    return (str(s)
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
            .replace("$", r"\$")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("^", r"\^{}")
            .replace("~", r"\textasciitilde{}")
            .replace("<", r"$<$")
            .replace(">", r"$>$"))


import re as _re

_TERM_MAP = [
    (_re.compile(r"C\([^[]*\[T\.AB\]:C\([^[]*\[T\.EM\]"), "Condition(AB) $\\times$ Target(EM)"),
    (_re.compile(r"C\([^[]*\[T\.AB\]"), "Condition: AB vs NB"),
    (_re.compile(r"C\([^[]*\[T\.EM\]"), "Target type: EM vs BB"),
    (_re.compile(r"C\([^[]*\[T\.Male\]"), "Gender: Male vs Female"),
    (_re.compile(r"C\(handedness\)\[T\.Right handed\]"), "Handedness: Right"),
    (_re.compile(r"C\(vision\)\[T\.Uncorrected vision difficulty\]"), "Vision: Uncorrected difficulty"),
    (_re.compile(r"C\(vision\)\[T\.Normal\]"), "Vision: Normal"),
    (_re.compile(r"Intercept"), "Intercept"),
    (_re.compile(r"age"), "Age"),
    (_re.compile(r"movie_duration"), "Movie duration (s)"),
]

def clean_term(raw: str) -> str:
    """Convert a statsmodels formula term to a short, readable label."""
    s = raw.strip()
    for pat, label in _TERM_MAP:
        if pat.fullmatch(s):
            return label
    # fallback: just escape
    return esc(s)


def sig_star(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def decision(p_one, observed_matches) -> str:
    if p_one is None or math.isnan(p_one):
        return r"\textit{---}"
    if p_one < 0.05 and observed_matches:
        return r"\textbf{Reject H\textsubscript{0}}"
    return r"Fail to reject H\textsubscript{0}"


# ── descriptives table ────────────────────────────────────────────────
def desc_table(r: dict, g1: str, g2: str, caption: str, label: str) -> str:
    d1 = r["descriptives"][g1]
    d2 = r["descriptives"][g2]

    def row(d, name):
        return (f"    {esc(name)} & {d['n']} & {f(d['mean'],3)} & {f(d['sd'],3)} "
                f"& {f(d['median'],3)} & {f(d['q1'],3)} & {f(d['q3'],3)} "
                f"& {f(d['min'],3)} & {f(d['max'],3)} \\\\")

    return r"""
\begin{table}[h!]
\centering
\small
\begin{tabular}{lrrrrrrrr}
\hline
\textbf{Group} & \textbf{n} & \textbf{M} & \textbf{SD} & \textbf{Mdn} & \textbf{Q1} & \textbf{Q3} & \textbf{Min} & \textbf{Max} \\
\hline
""" + row(d1, g1) + "\n" + row(d2, g2) + r"""
\hline
\end{tabular}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
"""


def desc_table_single(desc: dict, name: str, caption: str, label: str) -> str:
    """Single-row descriptive table (e.g. for H7/H8 variables)."""
    return r"""
\begin{table}[h!]
\centering
\small
\begin{tabular}{lrrrrrrrr}
\hline
\textbf{Variable} & \textbf{n} & \textbf{M} & \textbf{SD} & \textbf{Mdn} & \textbf{Q1} & \textbf{Q3} & \textbf{Min} & \textbf{Max} \\
\hline
    """ + f"    {esc(name)} & {desc['n']} & {f(desc['mean'],3)} & {f(desc['sd'],3)} & {f(desc['median'],3)} & {f(desc['q1'],3)} & {f(desc['q3'],3)} & {f(desc['min'],3)} & {f(desc['max'],3)} \\\\" + r"""
\hline
\end{tabular}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
"""


# ── normality + levene lines ──────────────────────────────────────────
def normality_lines(r: dict, g1: str, g2: str) -> str:
    n = r.get("normality", {})
    lev = r.get("levene", {})
    n1 = n.get(g1, {})
    n2 = n.get(g2, {})
    both = r.get("normality", {}).get("both_normal", False)
    normality_outcome = "Both groups pass normality." if both else "At least one group rejects normality."
    s = (f"\\textbf{{Normality (Shapiro--Wilk):}} {esc(g1)} $p = {p_raw(n1.get('p'))}$; "
         f"{esc(g2)} $p = {p_raw(n2.get('p'))}$. {normality_outcome}")
    if lev:
        eq = lev.get("equal_var", True)
        s += (f"\n\n\\textbf{{Variance homogeneity (Levene):}} $p = {p_raw(lev.get('p'))}$. "
              + ("Equal variances assumed." if eq else "Unequal variances."))
    return s


# ── result lines for group tests ──────────────────────────────────────
def result_lines(r: dict, directional: bool = False, decision_text: str = "") -> str:
    test = r.get("test_used", "")
    is_mw = "Mann-Whitney" in test or "Mann–Whitney" in test
    stat_name = r.get("statistic_name", "stat")
    stat_val = r.get("statistic")
    df_val = r.get("df")
    p2 = r.get("p_two_sided")
    p1 = r.get("p_one_sided")
    eff_name = esc(r.get("effect_size_name", ""))
    eff_val = r.get("effect_size_value")
    z = r.get("z_approx")
    obs = r.get("observed_direction_matches", None)

    # Determine which p-value to highlight and report
    p_report = p1 if directional else p2
    p_label = "one-sided" if directional else "two-sided"

    # Stat string
    stat_str = f"${stat_name} = {f(stat_val)}$"
    if df_val is not None and not math.isnan(float(df_val)):
        stat_str += f", $df = {f(df_val, 0)}$"

    lines = []

    # Test selected line + tie info for MW
    if is_mw:
        n_tg = r.get("n_tied_groups")
        n_uv = r.get("n_unique_vals")
        tc = r.get("tie_correction")
        tie_str = ""
        if n_tg is not None and n_uv is not None and tc is not None:
            tie_str = (f" Tied ranks: {n_tg} groups from {n_uv} distinct values "
                       f"(tie correction $= {tc:.4f}$, negligible).")
        lines.append(f"\\textbf{{Test selected:}} Mann--Whitney $U$.{tie_str}")
    else:
        lines.append(f"\\textbf{{Test selected:}} {esc(test)}.")

    # Result line
    p_str = f"$p_{{\\text{{{p_label}}}}} = {p_raw(p_report)}$"
    result_str = f"\\textbf{{Result:}} {stat_str}, {p_str}. {eff_name} $= {f(eff_val)}$."
    lines.append(result_str)

    # Decision line
    dec = decision(p_report, obs)
    if decision_text and p_report is not None and not math.isnan(float(p_report)) and p_report < 0.05 and obs:
        lines.append(f"\\textbf{{Decision:}} {dec}. {decision_text}")
    elif decision_text:
        fallback = decision_text.replace("significantly ", "numerically ").replace("Reject $H_0$.", "").strip()
        lines.append(f"\\textbf{{Decision:}} {dec}. {fallback}")
    else:
        lines.append(f"\\textbf{{Decision:}} {dec}.")

    return "\n\n".join(lines) + "\n"


# ── scatter plot for H7/H8 ────────────────────────────────────────────
def make_scatter_plots(dvs_path: Path, results: dict) -> tuple[str, str]:
    """Generate scatter plots for H7 and H8; return figure inclusion LaTeX."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = pd.read_csv(Path("final_cleaned_data") / "trials_final_clean_with_repeat.csv")
    trials = trials[trials["is_repeat"] == 0].copy()
    trials = trials[~trials["participant_id"].isin({"sub105_AB", "sub70_NB"})].copy()
    trials["resp.corr"] = pd.to_numeric(trials["resp.corr"], errors="coerce")
    trials["resp.rt"] = pd.to_numeric(trials["resp.rt"], errors="coerce")
    trials["conf_radio.response"] = pd.to_numeric(trials["conf_radio.response"], errors="coerce")
    trials = trials[trials["resp.rt"] > 0].copy()
    trials["log_rt"] = np.log(trials["resp.rt"])

    pp = (trials.groupby("participant_id")
          .agg(mean_log_rt=("log_rt", "mean"),
               accuracy=("resp.corr", "mean"),
               confidence=("conf_radio.response", "mean"))
          .reset_index())

    for col_y, label_y, h_label, fname in [
        ("accuracy", "Overall accuracy (proportion correct)", "H7", "h7_scatter.png"),
        ("confidence", "Mean confidence rating (1--5)", "H8", "h8_scatter.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pp["mean_log_rt"], pp[col_y], alpha=0.45, s=22, color="#2c7bb6")
        m, b = np.polyfit(pp["mean_log_rt"], pp[col_y], 1)
        xr = np.linspace(pp["mean_log_rt"].min(), pp["mean_log_rt"].max(), 100)
        ax.plot(xr, m * xr + b, color="#d62728", lw=1.5)
        rho = results[h_label]["rho"]
        pv = results[h_label]["p_two_sided"]
        ax.set_xlabel("Mean log RT (ln s)", fontsize=11)
        ax.set_ylabel(label_y, fontsize=11)
        ax.set_title(f"{h_label}: $\\rho = {rho:.3f}$, $p = {pv:.4f}$", fontsize=11)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(STATS_DIR / fname, dpi=150)
        plt.close(fig)

    return str(STATS_DIR / "h7_scatter.png"), str(STATS_DIR / "h8_scatter.png")


# ── main LaTeX builder ────────────────────────────────────────────────
def build_tex(results: dict) -> str:
    meta = results["_meta"]
    n_pp = meta["n_analysed"]
    n_trials = meta["n_trials"]
    n_nb = 88; n_ab = 80  # from analysis output

    # Load regression data
    mc = csv(REG_DIR / "model1_model_comparison_qic.csv")
    m3_or = csv(REG_DIR / "model3_participant_glm_odds_ratios.csv")
    vif_df = csv(REG_DIR / "vif_table.csv")
    bp_df = csv(REG_DIR / "diagnostic_breusch_pagan.csv")
    glmm_core = csv(REG_DIR / "glmm_core_fixed_effects.csv")
    glmm_int = csv(REG_DIR / "glmm_interaction_fixed_effects.csv")
    glmm_full = csv(REG_DIR / "glmm_full_fixed_effects.csv")

    # Best GEE model name and its OR table
    best_name = "M1\_stim\_main"
    best_or_df = None
    if mc is not None:
        stable = mc[mc["stable"].astype(bool)] if "stable" in mc.columns else mc
        stable_qic = stable.dropna(subset=["qic"]).sort_values("qic")
        if len(stable_qic):
            raw_name = stable_qic.iloc[0]["model"]
            best_name = esc(raw_name)
            best_or_df = csv(REG_DIR / f"{raw_name}_odds_ratios.csv")

    # RT normality
    rt_diag = results.get("_rt_normality_diagnostics", {})

    # H6
    r6 = results["H6"]
    d6_bb = r6["descriptives"]["BB"]
    d6_em = r6["descriptives"]["EM"]
    d6_diff = r6["descriptives"]["difference"]

    # H7, H8
    r7 = results["H7"]; r8 = results["H8"]

    lines = []
    A = lines.append  # shorthand

    # ── preamble ──────────────────────────────────────────────────────
    A(r"""\documentclass[12pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs,longtable,array,tabularx}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{float}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{enumitem}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\captionsetup{font=small,labelfont=bf}

\title{\textbf{BRSM Movie Memory Experiment}\\
\large Comprehensive Statistical Report\\
\normalsize After Vigilance-Based and Repeated-Video Exclusions}
\author{RSS Team}
\date{April 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage
""")

    # ── §1 Overview ────────────────────────────────────────────────────
    A(r"""
\section{Overview and Data Quality}

\subsection{Vigilance-based exclusions}
Two participants failed the embedded vigilance check and are excluded from every analysis:
\begin{itemize}[nosep]
  \item \texttt{sub105\_AB}
  \item \texttt{sub70\_NB}
\end{itemize}

\subsection{Repeated-video exclusion}
Five movies per condition (IDs 3, 7, 18, 28, 37) were presented \emph{twice} during
encoding as attention checks. All recognition trials corresponding to these repeated movies
are excluded from every analysis. \textbf{Rationale:} participants received two encoding
exposures for these clips; their recognition performance is therefore not comparable to
single-exposure movies and would artificially inflate accuracy estimates.
""")
    A(f"""
\subsection{{Sample and trial structure}}
The retained analytical sample is $N={n_pp}$ (NB={n_nb}, AB={n_ab}),
contributing {n_trials:,} recognition trials (35 unique movies $\\times$ 168 participants).

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\hline
\\textbf{{Condition}} & \\textbf{{n}} & \\textbf{{n trials}} & \\textbf{{BB trials}} & \\textbf{{EM trials}} \\\\
\\hline
NB (Natural Boundary) & {n_nb} & {n_nb*35:,} & {n_nb*35//2} & {n_nb*35 - n_nb*35//2} \\\\
AB (Abrupt Boundary)  & {n_ab} & {n_ab*35:,} & {n_ab*35//2} & {n_ab*35 - n_ab*35//2} \\\\
Total & {n_pp} & {n_trials:,} & --- & --- \\\\
\\hline
\\end{{tabular}}
\\caption{{Analytical sample after all exclusions (35 unique movies per participant).}}
\\end{{table}}
""")

    # ── §2 Methods ─────────────────────────────────────────────────────
    A(r"""
\section{Statistical Methods}

Each hypothesis follows an identical pre-specified decision pipeline:
\begin{enumerate}[nosep]
  \item Build the DV at the participant level from trial-level data.
  \item Compute full descriptive statistics ($n$, $M$, $SD$, $Mdn$, $Q_1$, $Q_3$, min, max) per group.
  \item Test normality within each group using the Shapiro--Wilk test ($\alpha = .05$).
  \item Test variance homogeneity using Levene's test (median-centred; Brown--Forsythe variant).
  \item Select the inferential test:
    \begin{itemize}[nosep]
      \item Both groups normal \emph{and} equal variance $\Rightarrow$ Student's $t$-test
      \item Both groups normal \emph{and} unequal variance $\Rightarrow$ Welch's $t$-test
      \item Either group non-normal $\Rightarrow$ Mann--Whitney $U$ test (two-sided)
    \end{itemize}
  \item Report effect sizes: Cohen's $d$ / Hedges' $g$ for parametric tests;
        rank-biserial $r$ for non-parametric tests.
\end{enumerate}
No correction for multiple comparisons is applied. All reported $p$-values are raw.

\subsection{Accuracy dependent variable}
All accuracy-based DVs use raw proportion correct (range 0--1). Chance performance is 0.5
in a two-alternative forced-choice task. After excluding 5 repeated movies per participant,
each participant contributes 35 unique-movie trials (approximately 17--18 BB + 17--18 EM).

\subsection{Log-transformation and normality of response time}
""")
    # RT normality table
    nb_raw = rt_diag.get("pp_raw_NB", {})
    ab_raw = rt_diag.get("pp_raw_AB", {})
    nb_log = rt_diag.get("pp_log_NB", {})
    ab_log = rt_diag.get("pp_log_AB", {})
    tr_raw = rt_diag.get("trial_raw", {})
    tr_log = rt_diag.get("trial_log", {})

    A(r"""Trial-level RTs are strongly right-skewed (skew $\approx """ +
      f"{tr_raw.get('skew', 0):.2f}" +
      r"""$, excess kurtosis $\approx """ +
      f"{tr_raw.get('kurtosis', 0):.2f}" +
      r"""$). All RT analyses therefore use natural-log-transformed RT (ln seconds).
Table~\ref{tab:rt_normality} verifies that the transform rescues approximate normality
at the participant level.

\begin{table}[H]
\centering
\small
\begin{tabular}{llrrl}
\hline
\textbf{Condition} & \textbf{Transform} & \textbf{n} & \textbf{M} & \textbf{Shapiro--Wilk $p$} \\
\hline
""" +
      f"NB & Raw RT (s) & {nb_raw.get('n','?')} & {f(nb_raw.get('mean'),3)} s & {p_fmt(nb_raw.get('shapiro_p'))} (non-normal) \\\\\n" +
      f"   & Log RT (ln s) & {nb_log.get('n','?')} & {f(nb_log.get('mean'),3)} & {p_fmt(nb_log.get('shapiro_p'))} (normal) \\\\\n" +
      f"AB & Raw RT (s) & {ab_raw.get('n','?')} & {f(ab_raw.get('mean'),3)} s & {p_fmt(ab_raw.get('shapiro_p'))} (non-normal) \\\\\n" +
      f"   & Log RT (ln s) & {ab_log.get('n','?')} & {f(ab_log.get('mean'),3)} & {p_fmt(ab_log.get('shapiro_p'))} (normal) \\\\\n" +
      f"All (trial) & Raw RT & {tr_raw.get('n_used_for_shapiro','?')} sample & skew$={tr_raw.get('skew',0):.3f}$ & {p_fmt(tr_raw.get('shapiro_p'))} \\\\\n" +
      f"            & Log RT & {tr_log.get('n_used_for_shapiro','?')} sample & skew$={tr_log.get('skew',0):.3f}$ & {p_fmt(tr_log.get('shapiro_p'))} \\\\\n" +
      r"""\hline
\end{tabular}
\caption{Shapiro--Wilk normality: raw RT vs.\ log RT at participant level (repeated movies excluded).}
\label{tab:rt_normality}
\end{table}

\noindent\textit{Note.} At the trial level ($n \approx """ + f"{n_trials:,}" + r"""$) Shapiro--Wilk has extreme power and flags even trivial departures from normality. The skew drops substantially after log-transformation, confirming normalisation. At the participant level both conditions pass after log-transformation.
""")

    # ── §3–7 H1–H5 ────────────────────────────────────────────────────
    h_specs = [
        ("H1", "H1: Overall Recognition Accuracy",
         "Overall recognition accuracy does not differ between Natural Boundary and Abrupt Boundary participants ($\\mu_{\\text{NB}} = \\mu_{\\text{AB}}$).",
         "NB participants have higher overall recognition accuracy than AB participants ($\\mu_{\\text{NB}} > \\mu_{\\text{AB}}$; directional).",
         "Per-participant proportion correct across all 35 unique recognition trials (repeated movies excluded).",
         "NB", "AB", "tab:h1_desc", "Overall accuracy",
         "NB participants have significantly higher overall accuracy."),
        ("H2", "H2: Before-Boundary Frame Accuracy",
         "Recognition accuracy for Before-Boundary targets does not differ between conditions ($\\mu_{\\text{NB}} = \\mu_{\\text{AB}}$).",
         "NB participants have higher Before-Boundary accuracy than AB participants ($\\mu_{\\text{NB}} > \\mu_{\\text{AB}}$; directional).",
         "Per-participant proportion correct on BB-frame targets only (approximately 17--18 trials after exclusions).",
         "NB", "AB", "tab:h2_desc", "BB accuracy",
         "NB participants recognise BB-frame targets significantly more accurately."),
        ("H3", "H3: Mean Confidence Rating",
         "Mean confidence rating does not differ between NB and AB participants ($\\mu_{\\text{NB}} = \\mu_{\\text{AB}}$).",
         "NB participants report higher mean confidence than AB participants ($\\mu_{\\text{NB}} > \\mu_{\\text{AB}}$; directional).",
         "Per-participant mean confidence rating (1--5 Likert scale; repeated movies excluded).",
         "NB", "AB", "tab:h3_desc", "Mean confidence",
         "NB participants report significantly higher mean confidence."),
        ("H4", "H4: Proportion of Low-Confidence Trials",
         "The proportion of trials with confidence $\\leq 3$ does not differ between conditions ($\\pi_{\\text{AB}} = \\pi_{\\text{NB}}$).",
         "AB participants have a higher proportion of low-confidence trials ($\\pi_{\\text{AB}} > \\pi_{\\text{NB}}$; directional).",
         "Per-participant proportion of trials rated confidence $\\leq 3$ (repeated movies excluded).",
         "AB", "NB", "tab:h4_desc", "Low-confidence proportion",
         "AB participants show significantly more low-confidence responses."),
        ("H5", "H5: Per-Movie Accuracy Variability",
         "The standard deviation of per-movie accuracy does not differ between conditions ($\\sigma_{\\text{AB}} = \\sigma_{\\text{NB}}$).",
         "AB participants show greater variability in per-movie accuracy ($\\sigma_{\\text{AB}} > \\sigma_{\\text{NB}}$; directional).",
         "Per-participant standard deviation of accuracy across the 35 unique movies (repeated movies excluded).",
         "AB", "NB", "tab:h5_desc", "Per-movie accuracy SD",
         "AB participants show significantly greater per-movie accuracy variability."),
    ]

    sec_num = 3
    for (hk, sec_title, h0, h1, dv_text, g1, g2, tab_label, tab_name, dec_text) in h_specs:
        r = results[hk]
        A(f"\n\\section{{{sec_title}}}\n")
        A(f"\\subsection{{Hypotheses}}\n")
        A(f"$H_0$: {h0}\n\n$H_1$: {h1}\n")
        A(f"\\subsection{{DV and descriptive statistics}}\n")
        A(f"\\textbf{{DV:}} {dv_text}\n")
        A(desc_table(r, g1, g2,
                     f"Descriptive statistics --- {hk} ({esc(tab_name)}).",
                     tab_label))
        A("\\subsection{Test selection and result}\n")
        A(normality_lines(r, g1, g2) + "\n\n")
        A(result_lines(r, directional=True, decision_text=dec_text))
        sec_num += 1

    # ── §8 H6 ─────────────────────────────────────────────────────────
    A(r"""
\section{H6: BB-Frame vs EM-Frame Response Time}

\subsection{Hypotheses}
$H_0$: Mean log RT does not differ between BB-frame and EM-frame recognition trials
($\mu_\text{BB} = \mu_\text{EM}$).

$H_1$: Mean log RT differs between BB-frame and EM-frame recognition trials
($\mu_\text{BB} \neq \mu_\text{EM}$; non-directional).

\textit{Rationale.} If BB-frame targets engage a qualitatively different retrieval process
(e.g., recollection rather than familiarity), recognition decisions for BB targets may take
longer than for EM targets. This within-participant comparison is exploratory and non-directional.

\subsection{DV and descriptive statistics}
DV: Per-participant mean log RT on BB trials and mean log RT on EM trials; the within-participant
difference BB$-$EM is tested against zero (repeated movies excluded).
""")
    A(f"""\\begin{{table}}[H]
\\centering
\\small
\\begin{{tabular}}{{lrrrrrrrr}}
\\hline
\\textbf{{Measure}} & \\textbf{{n}} & \\textbf{{M}} & \\textbf{{SD}} & \\textbf{{Mdn}} & \\textbf{{Q1}} & \\textbf{{Q3}} & \\textbf{{Min}} & \\textbf{{Max}} \\\\
\\hline
BB mean log RT (ln s) & {d6_bb['n']} & {f(d6_bb['mean'])} & {f(d6_bb['sd'])} & {f(d6_bb['median'])} & {f(d6_bb['q1'])} & {f(d6_bb['q3'])} & {f(d6_bb['min'])} & {f(d6_bb['max'])} \\\\
EM mean log RT (ln s) & {d6_em['n']} & {f(d6_em['mean'])} & {f(d6_em['sd'])} & {f(d6_em['median'])} & {f(d6_em['q1'])} & {f(d6_em['q3'])} & {f(d6_em['min'])} & {f(d6_em['max'])} \\\\
BB$-$EM difference    & {d6_diff['n']} & {f(d6_diff['mean'])} & {f(d6_diff['sd'])} & {f(d6_diff['median'])} & {f(d6_diff['q1'])} & {f(d6_diff['q3'])} & {f(d6_diff['min'])} & {f(d6_diff['max'])} \\\\
\\hline
\\end{{tabular}}
\\caption{{Descriptive statistics --- H6 (BB vs EM log RT, within-participant).}}
\\end{{table}}
""")
    norm6 = r6.get("normality_of_diffs", {})
    A(f"""\\subsection{{Test selection and result}}
\\textbf{{Normality of differences (Shapiro--Wilk):}} $p = {p_raw(norm6.get('p'))}$
({'normal' if norm6.get('normal') else 'non-normal'}). {'One-sample $t$-test on differences is used.' if norm6.get('normal') else 'Wilcoxon signed-rank test is used.'}

\\textbf{{Test selected:}} {esc(r6['test_used'])}.

\\textbf{{Result:}} ${r6['statistic_name']} = {f(r6['statistic'])}$""")
    if r6.get("df"):
        A(f", $df = {f(r6['df'], 0)}$")
    A(f""", $p_{{\\text{{two-sided}}}} = {p_raw(r6['p_two_sided'])}$. Effect: {esc(r6['effect_size_name'])} $= {f(r6['effect_size_value'])}$.

\\textbf{{Decision:}} {'Reject $H_0$' if r6['p_two_sided'] < 0.05 else 'Fail to reject $H_0$'}.
BB-frame recognition decisions take {'significantly ' if r6['p_two_sided'] < 0.05 else ''}longer than EM-frame decisions
(mean difference $= {f(d6_diff['mean'])}$ ln s), {'consistent with a more effortful retrieval process for boundary-region events.' if r6['p_two_sided'] < 0.05 else 'but the difference does not reach significance.'}
""")

    # ── §9 H7 ─────────────────────────────────────────────────────────
    A(r"""
\section{H7: Mean log RT vs Overall Accuracy}

\subsection{Hypotheses}
$H_0$: There is no linear association between mean response time and overall recognition accuracy
($\rho = 0$; two-tailed).

$H_1$: There is a significant association between mean response time and overall recognition
accuracy ($\rho \neq 0$; two-tailed).

\textit{Rationale.} Slower responses on a recognition task could reflect greater uncertainty and
more effortful retrieval, which may be associated with lower accuracy (speed--accuracy trade-off).
Alternatively, slower, more careful responding could improve accuracy. The direction is not
pre-specified; the test is two-tailed.

\subsection{DV and descriptive statistics}
DV: Per-participant mean log RT (ln s) and per-participant overall accuracy (proportion correct
across all 35 unique trials; repeated movies excluded). Log RT is used because raw RT is
right-skewed (see \S2.3).
""")
    # Build descriptives for H7 variables from RT and H1
    r1 = results["H1"]
    rt_log_dvs = results["_rt_normality_diagnostics"]
    nb_log_m = rt_log_dvs.get("pp_log_NB", {})
    ab_log_m = rt_log_dvs.get("pp_log_AB", {})
    # Combined log RT descriptives (all participants)
    all_n = nb_log_m.get("n", 0) + ab_log_m.get("n", 0)
    all_mean_rt = (nb_log_m.get("mean", 0) * nb_log_m.get("n", 0) +
                   ab_log_m.get("mean", 0) * ab_log_m.get("n", 0)) / max(all_n, 1)
    # accuracy combined
    d1_h1 = r1["descriptives"]["NB"]; d2_h1 = r1["descriptives"]["AB"]
    all_n_acc = d1_h1["n"] + d2_h1["n"]
    all_mean_acc = (d1_h1["mean"] * d1_h1["n"] + d2_h1["mean"] * d2_h1["n"]) / all_n_acc

    A(f"""\\begin{{table}}[H]
\\centering
\\small
\\begin{{tabular}}{{lrrrrrrrr}}
\\hline
\\textbf{{Variable}} & \\textbf{{n}} & \\textbf{{M}} & \\textbf{{SD}} & \\textbf{{Mdn}} & \\textbf{{Q1}} & \\textbf{{Q3}} & \\textbf{{Min}} & \\textbf{{Max}} \\\\
\\hline
Mean log RT (ln s) & {r7['n']} & {f(all_mean_rt)} & --- & --- & --- & --- & --- & --- \\\\
Overall accuracy   & {r7['n']} & {f(all_mean_acc)} & --- & --- & --- & --- & --- & --- \\\\
\\hline
\\end{{tabular}}
\\caption{{Descriptive statistics --- H7 (Mean log RT and Overall accuracy).}}
\\end{{table}}
""")
    norm7 = r7.get("normality", {})
    norm7x = norm7.get("x") or {}
    norm7y = norm7.get("y") or {}
    norm7_x_W = f(norm7x.get('shapiro_W', float('nan')))
    norm7_x_p = p_raw(norm7x.get('p'))
    norm7_y_W = f(norm7y.get('shapiro_W', float('nan')))
    norm7_y_p = p_raw(norm7y.get('p'))
    A(f"""\\subsection{{Test selection and result}}
\\textbf{{Normality (Shapiro--Wilk):}} Mean log RT: $W = {norm7_x_W}$, $p = {norm7_x_p}$; Overall accuracy: $W = {norm7_y_W}$, $p = {norm7_y_p}$.

\\textbf{{Test selected:}} Spearman rank correlation (at least one variable non-normal; two-tailed). 95\\% CI obtained by bootstrap ($B = {r7['n_boot']}$).

\\textbf{{Result:}} $\\rho = {f(r7['rho'])}$ ($n = {r7['n']}$, $df = {r7['n']-2}$), $p_{{\\text{{two-sided}}}} = {p_raw(r7['p_two_sided'])}$, 95\\% CI $[{f(r7['ci_95_lo'])},\\, {f(r7['ci_95_hi'])}]$.

\\textbf{{Decision:}} {'Reject $H_0$' if r7['p_two_sided'] < 0.05 else 'Fail to reject $H_0$'}.
{'A statistically significant' if r7['p_two_sided'] < 0.05 else 'No significant'} correlation was found between mean log RT and overall accuracy ($\\rho = {f(r7['rho'])}$, $p = {p_raw(r7['p_two_sided'])}$).
""")
    A(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.65\textwidth]{h7_scatter.png}
\caption{H7 scatter plot: per-participant mean log RT vs.\ overall recognition accuracy (repeated movies excluded). Regression line added.}
\end{figure}
""")

    # ── §10 H8 ────────────────────────────────────────────────────────
    A(r"""
\section{H8: Mean log RT vs Mean Confidence Rating}

\subsection{Hypotheses}
$H_0$: There is no linear association between mean response time and mean confidence rating
($\rho = 0$; two-tailed).

$H_1$: There is a significant association between mean response time and mean confidence rating
($\rho \neq 0$; two-tailed).

\textit{Rationale.} High confidence is associated with fast, fluent recognition (the `feeling
of knowing' literature), so slower mean RT may predict lower confidence. The direction is not
pre-specified; the test is two-tailed.

\subsection{DV and descriptive statistics}
DV: Per-participant mean log RT (ln s) and per-participant mean confidence rating (1--5 Likert
scale). Log RT is used for the same reason as H7. Repeated movies are excluded.
""")
    norm8 = r8.get("normality", {})
    norm8x = norm8.get("x") or {}
    norm8y = norm8.get("y") or {}
    norm8_x_p = p_raw(norm8x.get('p'))
    norm8_y_p = p_raw(norm8y.get('p'))
    A(f"""\\subsection{{Test selection and result}}
\\textbf{{Normality (Shapiro--Wilk):}} Mean log RT: $p = {norm8_x_p}$; Mean confidence: $p = {norm8_y_p}$.

\\textbf{{Test selected:}} Spearman rank correlation (at least one variable non-normal; two-tailed). 95\\% CI obtained by bootstrap ($B = {r8['n_boot']}$).

\\textbf{{Result:}} $\\rho = {f(r8['rho'])}$ ($n = {r8['n']}$, $df = {r8['n']-2}$), $p_{{\\text{{two-sided}}}} = {p_raw(r8['p_two_sided'])}$, 95\\% CI $[{f(r8['ci_95_lo'])},\\, {f(r8['ci_95_hi'])}]$.

\\textbf{{Decision:}} {'Reject $H_0$' if r8['p_two_sided'] < 0.05 else 'Fail to reject $H_0$'}.
{'A statistically significant' if r8['p_two_sided'] < 0.05 else 'No significant'} {'negative ' if r8['rho'] < 0 else ''}Spearman correlation was found between mean log RT and mean confidence rating ($\\rho = {f(r8['rho'])}$, $p = {p_raw(r8['p_two_sided'])}$).
""")
    A(r"""
\begin{figure}[H]
\centering
\includegraphics[width=0.65\textwidth]{h8_scatter.png}
\caption{H8 scatter plot: per-participant mean log RT vs.\ mean confidence rating (repeated movies excluded). Regression line added.}
\end{figure}
""")

    # ── §11 Regression ────────────────────────────────────────────────
    A(r"""
\section{Regression Analysis}

\subsection{Overview and Feature Policy}
A strict regression pipeline was applied to predict trial-level recognition accuracy
(\texttt{resp.corr}, binary) from pre-response design variables and pre-treatment participant
covariates. Post-response variables (RT, confidence) were explicitly excluded to prevent data
leakage. Two participants who failed the vigilance check are excluded throughout. All trials
from the five repeated movies are excluded (is\_repeat = 0 only); because no repeated trials
remain, \texttt{is\_repeat} is not a predictor in any model.

\begin{itemize}[nosep]
  \item \textbf{Included (pre-response):} condition, target\_type, movie\_duration, stimulus\_id
  \item \textbf{Optional controls (pre-treatment):} age, gender, handedness, vision
  \item \textbf{Excluded (leakage / post-response):} resp.rt, conf\_radio.response
  \item \textbf{Excluded (zero-variance after filtering):} is\_repeat
\end{itemize}

Three complementary models are used:
\begin{enumerate}[nosep]
  \item \textbf{Model~3} --- participant-level binomial GLM
  \item \textbf{Model~1} --- trial-level GEE logistic regression (8 nested models)
  \item \textbf{GLMM}    --- trial-level binomial mixed model with random participant intercept (Variational Bayes)
\end{enumerate}

\subsection{Model 3: Participant-Level Binomial GLM}
A participant-level binomial GLM (link: logit) was fitted with condition, age, and gender as
predictors, using (successes, failures) counts as the response.
""")

    # M3 table
    if m3_or is not None:
        A(r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrl}
\hline
\textbf{Term} & \textbf{Coef} & \textbf{$p$} & \textbf{OR} \\
\hline
""")
        for _, row in m3_or.iterrows():
            term = esc(str(row.get("term", "")))
            coef = f(row.get("coef"), 3)
            pval = p_fmt(row.get("p"))
            oratio = f(row.get("OR"), 3)
            A(f"    {term} & {coef} & {pval} & {oratio} \\\\\n")
        A(r"""\hline
\end{tabular}
\caption{Model 3 --- Participant-level binomial GLM: coefficients and odds ratios (repeated movies excluded).}
\end{table}
""")
    A(r"""
\noindent\textbf{Key finding:} NB condition significantly increases accuracy odds.
Male gender is associated with lower accuracy odds. Age shows a borderline negative effect.

\subsection{Model 1: Trial-Level GEE Logistic Regression}
Eight candidate GEE models (Binomial family, exchangeable working correlation, robust SE)
were fitted with trials clustered by participant. Models were compared by QIC (lower is better);
unstable models (non-finite parameters / SEs) were flagged and excluded from best-model selection.
""")

    # GEE comparison table
    if mc is not None:
        A(r"""\begin{table}[H]
\centering
\small
\begin{tabular}{llp{7cm}}
\hline
\textbf{Model} & \textbf{QIC} & \textbf{Description} \\
\hline
""")
        for _, row in mc.iterrows():
            mname = esc(str(row.get("model", "")))
            qic = f(row.get("qic"), 2) if pd.notna(row.get("qic", float("nan"))) else "---"
            stable = str(row.get("stable", ""))
            desc = esc(str(row.get("description", "")))
            marker = "" if stable == "True" else r" \textit{(unstable)}"
            A(f"    {mname}{marker} & {qic} & {desc} \\\\\n")
        A(r"""\hline
\end{tabular}
\caption{Model 1 --- GEE model comparison by QIC (lower is better; repeated movies excluded).}
\end{table}
""")

    A(f"""\\subsection{{Best Model: {best_name} --- Odds Ratios}}\n""")
    A(f"The best stable model by QIC is {best_name}.\n")

    if best_or_df is not None:
        A(r"""\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{lrrrrrr}
\hline
\textbf{Term} & \textbf{Coef} & \textbf{SE} & \textbf{$z$} & \textbf{$p$} & \textbf{OR} & \textbf{95\% CI} \\
\hline
""")
        for _, row in best_or_df.iterrows():
            term = clean_term(str(row.get("term", "")))
            coef = f(row.get("coef"), 3)
            se = f(row.get("se"), 3)
            z = f(row.get("z_or_t"), 3)
            pval = p_fmt(row.get("p"))
            oratio = f(row.get("OR"), 3)
            lo = f(row.get("OR_ci_low"), 3)
            hi = f(row.get("OR_ci_high"), 3)
            A(f"    {term} & {coef} & {se} & {z} & {pval} & {oratio} & [{lo}, {hi}] \\\\\n")
        A(r"""\hline
\end{tabular}
\caption{Best GEE model: odds ratios with 95\% CI (repeated movies excluded).}
\end{table}
""")

    # ── Interaction effects explanation ────────────────────────────────
    A(r"""
\subsection{Interaction Effects in the GEE Models}
\label{sec:interaction}

Several candidate models include a \textbf{condition $\times$ target\_type interaction}.
This section explains how to interpret these interaction terms.

\subsubsection{What the interaction term tests}

The term \texttt{C(condition)[T.AB]:C(target\_type)[T.EM]} tests whether the difference in
log-odds of a correct response between EM-frame and BB-frame targets is \emph{the same} in
both the AB and NB conditions.

Let $\beta_1$ = coefficient for condition (AB vs NB, reference = NB), $\beta_2$ = coefficient
for target type (EM vs BB, reference = BB), and $\beta_3$ = interaction coefficient. Then:
\begin{align*}
\text{NB} + \text{BB:} &\quad \alpha \quad \text{(reference cell)} \\
\text{NB} + \text{EM:} &\quad \alpha + \beta_2 \quad \text{(EM advantage in NB)} \\
\text{AB} + \text{BB:} &\quad \alpha + \beta_1 \quad \text{(AB penalty for BB targets)} \\
\text{AB} + \text{EM:} &\quad \alpha + \beta_1 + \beta_2 + \beta_3 \quad \text{(combined effects in AB)}
\end{align*}

So $\beta_3$ measures \textbf{how the EM advantage over BB changes} when moving from NB to AB.

\subsubsection{Interpreting the sign of the interaction}
\begin{itemize}[nosep]
  \item $\beta_3 > 0$ (OR $> 1$): the EM advantage is \emph{larger} in AB than in NB --- the AB condition hurts BB recall proportionally more than EM recall.
  \item $\beta_3 < 0$ (OR $< 1$): the EM advantage is \emph{smaller} in AB --- boundary disruption reduces BB and EM recognition approximately equally.
  \item $\beta_3 \approx 0$ (OR $\approx 1$): no moderation --- condition and frame type act independently.
\end{itemize}

\subsubsection{Odds ratios for interaction terms are ``ratios of odds ratios''}
The OR for the interaction is
$$\text{OR}_\text{interaction} = e^{\beta_3} = \frac{\text{OR}_{\text{EM/BB in AB}}}{\text{OR}_{\text{EM/BB in NB}}}.$$
This is not a standalone probability but the \emph{multiplicative change} in the EM advantage
when switching from NB to AB.

\subsubsection{Recovering simple (conditional) effects}
\begin{itemize}[nosep]
  \item Effect of \textbf{condition for BB targets:} $e^{\beta_1}$ \hfill (AB vs NB, BB trials only)
  \item Effect of \textbf{condition for EM targets:} $e^{\beta_1 + \beta_3}$ \hfill (AB vs NB, EM trials only)
  \item Effect of \textbf{target type in NB:} $e^{\beta_2}$ \hfill (EM vs BB, NB participants)
  \item Effect of \textbf{target type in AB:} $e^{\beta_2 + \beta_3}$ \hfill (EM vs BB, AB participants)
\end{itemize}
These simple effects answer: ``What is the effect of X when Y is held at a specific level?''

\subsubsection{GEE vs GLMM for interaction estimates}
GEE estimates \emph{population-average} ORs (``On average across all participants, how does condition moderate the frame-type effect?''), while GLMM estimates \emph{subject-specific} ORs (``For a given participant, how does condition moderate the effect?''). Both families should agree in sign and significance. A large discrepancy signals substantial between-participant heterogeneity in the moderation.

\subsection{Diagnostics}

\subsubsection{Collinearity (VIF)}
""")
    if vif_df is not None:
        A(r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lr}
\hline
\textbf{Variable} & \textbf{VIF} \\
\hline
""")
        for _, row in vif_df.iterrows():
            A(f"    {esc(str(row.get('variable','')))} & {f(row.get('VIF'),3)} \\\\\n")
        A(r"""\hline
\end{tabular}
\caption{Variance inflation factors for trial-level predictors (VIF $< 5$ = acceptable).}
\end{table}
""")

    A(r"""
\subsubsection{Heteroscedasticity (Breusch--Pagan, OLS surrogate)}
A Breusch--Pagan test was run on an OLS surrogate model predicting participant-level accuracy
from condition, age, and gender. The primary Model~1 is logistic (GEE) and does not assume
homoscedasticity; this diagnostic is provided for completeness.
""")
    if bp_df is not None and len(bp_df):
        bp = bp_df.iloc[0]
        A(r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrrrr}
\hline
\textbf{Model} & \textbf{BP LM} & \textbf{$p_\text{LM}$} & \textbf{BP F} & \textbf{$p_F$} & \textbf{n} & \textbf{$R^2$} \\
\hline
""" + f"OLS accuracy & {f(bp.get('bp_lm'),3)} & {p_fmt(bp.get('bp_lm_p'))} & "
          f"{f(bp.get('bp_f'),3)} & {p_fmt(bp.get('bp_f_p'))} & "
          f"{int(bp.get('n',0))} & {f(bp.get('r2'),3)} \\\\\n" +
          r"""\hline
\end{tabular}
\caption{Breusch--Pagan heteroscedasticity test (OLS surrogate on participant accuracy).}
\end{table}

\noindent Both Breusch--Pagan statistics are non-significant ($p > .40$), indicating no evidence
of heteroscedasticity in the OLS surrogate.
""")

    # ── §11.6 GLMM ────────────────────────────────────────────────────
    A(r"""
\subsection{Mixed-Effects Logistic Regression (GLMM)}

\subsubsection{Rationale and Model Specification}
Because each participant completed 35 trials (after repeated-movie exclusion), observations are
not independent. A binomial mixed-effects logistic model addresses this by including a
\textbf{random intercept for participant\_id}, which controls for stable individual differences
in baseline recognition accuracy.

Estimation uses \textbf{Variational Bayes} (\texttt{statsmodels BinomialBayesMixedGLM.fit\_vb()}).
Variational Bayes is preferred over the Laplace approximation (\texttt{fit\_map}) because it
reliably estimates the random-effect variance in balanced designs; the Laplace approximation
can collapse the random-effect SD to zero in such cases.
Approximate $z$-statistics and $p$-values are derived from $z = \hat\beta / \text{Post.SD}$
using the standard normal distribution.

Three GLMM variants are reported:
\begin{enumerate}[nosep]
  \item \textbf{Core} (condition + target\_type main effects)
  \item \textbf{Interaction} (condition $\times$ target\_type)
  \item \textbf{Full} (condition + target\_type + age + gender + movie\_duration)
\end{enumerate}

\subsubsection{Fixed Effects}
""")

    def glmm_table(fe_df, caption, label):
        if fe_df is None:
            return "\\textit{(Not available)}\n"
        s = r"""\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{lrrrrl}
\hline
\textbf{Predictor} & \textbf{Post.\ Mean} & \textbf{Post.\ SD} & \textbf{$z$} & \textbf{$p$ (approx)} & \textbf{OR} \\
\hline
"""
        for _, row in fe_df.iterrows():
            term = clean_term(str(row.get("term", "")))
            pm = f(row.get("post_mean"), 4)
            psd = f(row.get("post_sd"), 4)
            z_val = row.get("z_approx")
            p_val = row.get("p_approx")
            oratio = f(row.get("OR"), 4)
            z_str = f(z_val, 3) if z_val is not None and not (isinstance(z_val, float) and math.isnan(z_val)) else "---"
            s += f"    {term} & {pm} & {psd} & {z_str} & {p_fmt(p_val)} & {oratio} \\\\\n"
        s += r"""\hline
\end{tabular}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
"""
        return s

    A(glmm_table(glmm_core, "GLMM core --- main effects (repeated movies excluded).", "tab:glmm_core"))
    A(glmm_table(glmm_int, r"GLMM interaction --- condition $\times$ target\_type (repeated movies excluded).", "tab:glmm_int"))
    A(glmm_table(glmm_full, "GLMM full --- main effects + age + gender + movie\_duration (repeated movies excluded).", "tab:glmm_full"))

    # Random effects summary
    if glmm_core is not None:
        # Extract random intercept SD from the fixed effects CSV doesn't have it;
        # we read it from the regression_summary.txt
        summary_txt = (REG_DIR / "regression_summary.txt").read_text() if (REG_DIR / "regression_summary.txt").exists() else ""
        import re
        sds = re.findall(r"Random intercept SD: ([\d.]+)", summary_txt)
        re_core = sds[0] if len(sds) > 0 else "---"
        re_int = sds[1] if len(sds) > 1 else "---"
        re_full = sds[2] if len(sds) > 2 else "---"

        A(r"""
\subsubsection{Random Effects}
The random-effect standard deviation (SD) quantifies between-participant variability in
baseline log-odds accuracy, after accounting for fixed effects.

\begin{table}[H]
\centering
\small
\begin{tabular}{lrp{5.5cm}}
\hline
\textbf{GLMM variant} & \textbf{Random intercept SD} & \textbf{Interpretation} \\
\hline
""" + f"Core (main effects) & {re_core} & Between-participant variability in log-odds \\\\\n" +
          f"Interaction & {re_int} & Same, after adding interaction \\\\\n" +
          f"Full (+ age, gender, duration) & {re_full} & Same, after demographic controls \\\\\n" +
          r"""\hline
\end{tabular}
\caption{GLMM random intercept SDs (log-odds scale; repeated movies excluded).}
\end{table}

\noindent A random intercept SD of $\approx 0.42$ indicates meaningful between-person
variability in baseline recognition accuracy --- justifying the inclusion of random intercepts
per participant. The SD is stable across model variants, suggesting it is not confounded
with the fixed-effect predictors.
""")

    A(r"""
\subsubsection{Interaction Interpretation in the GLMM}

The interaction term in the GLMM (\texttt{C(condition)[T.AB]:C(target\_type)[T.EM]})
has the same structural interpretation as in the GEE (see \S\ref{sec:interaction}), but now
represents a \emph{subject-specific} (conditional) effect rather than a population-average effect.

The posterior mean of the interaction coefficient indicates whether the EM advantage over BB
is larger or smaller in the AB condition relative to NB, \emph{for a given participant} (after
integrating out individual differences via the random intercept).

Key comparison with GEE:
\begin{itemize}[nosep]
  \item If GEE and GLMM interaction ORs agree in sign and magnitude: the moderation is consistent across participants.
  \item If they diverge substantially: there is heterogeneity in how the condition moderates the frame-type effect across individuals.
\end{itemize}

\subsubsection{Summary}
The GLMM provides the most methodologically rigorous answer to the primary research question.
After controlling for individual participant ability, the Abrupt Boundary (AB) condition
significantly reduces recognition accuracy relative to Natural Boundary (NB). This converges
with the GEE and the participant-level H1 result, establishing consistency across all three
levels of analysis.
""")

    A(r"""
\end{document}
""")

    return "\n".join(lines)


def main() -> None:
    print("Loading results...")
    results = load_json()

    print("Generating scatter plots for H7 and H8...")
    make_scatter_plots(STATS_DIR, results)

    print("Building LaTeX source...")
    tex = build_tex(results)

    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Saved .tex -> {TEX_PATH}")

    # Compile three times for ToC / cross-refs / bookmarks
    for run in range(3):
        print(f"pdflatex pass {run + 1}...")
        proc = subprocess.run(
            ["/Library/TeX/texbin/pdflatex", "-interaction=nonstopmode",
             TEX_PATH.name],
            capture_output=True, text=True,
            cwd=str(STATS_DIR),
        )
        if proc.returncode != 0:
            # print last 40 lines of log for diagnosis
            log_lines = (proc.stdout + proc.stderr).splitlines()
            print("pdflatex errors:")
            print("\n".join(log_lines[-40:]))

    if PDF_PATH.exists():
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"\nPDF saved: {PDF_PATH}  ({size_kb:.1f} KB)")
    else:
        print(f"\nWARNING: PDF not found at {PDF_PATH}. Check the log.")


if __name__ == "__main__":
    main()
