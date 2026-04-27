"""
Statistical tests for the BRSM Movie Memory experiment.

Vigilance exclusions
--------------------
Two participants failed the built-in vigilance check and are excluded:
    sub105_AB  (AB condition)
    sub70_NB   (NB condition)

Repeated-video exclusion
------------------------
Five movies per condition (IDs 3, 7, 18, 28, 37) were shown TWICE during
encoding as attention/vigilance checks.  These repeated trials are excluded
from ALL recognition analyses here.  Reason: participants had two exposure
opportunities for those movies, so their recognition is not directly
comparable to the single-exposure movies.  Including them would inflate
accuracy estimates and confound condition comparisons.

Hypotheses
----------
    H1: Overall recognition accuracy  (NB > AB)   — DV: mean accuracy
    H2: Before-Boundary (BB) accuracy (NB > AB)   — DV: BB accuracy
    H3: Mean confidence rating        (NB > AB)   — DV: mean conf (1-5)
    H4: Proportion of low-confidence  (AB > NB)   — DV: prop conf <= 3
    H5: Movie-level accuracy variability (AB > NB) — DV: SD of per-movie acc
    H6: BB vs EM log RT — within-participant paired comparison (non-directional)
    H7: Correlation between mean log RT and overall accuracy (two-tailed)
    H8: Correlation between mean log RT and mean confidence rating (two-tailed)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("final_cleaned_data")
OUT_DIR = Path("statistical_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
RNG_SEED = 20260415

VIGILANCE_EXCLUSIONS = {"sub105_AB", "sub70_NB"}
REPEATED_VIDEO_IDS = {3, 7, 18, 28, 37}


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def load_data():
    trials = pd.read_csv(DATA_DIR / "trials_final_clean_with_repeat.csv")
    pp = pd.read_csv(DATA_DIR / "participants_final_clean_with_vigilance.csv")

    pp = pp[~pp["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()
    trials = trials[~trials["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()

    # Exclude repeated video frames (vigilance check movies)
    trials["movie_id"] = pd.to_numeric(trials["movie_id"], errors="coerce")
    trials["is_repeat"] = pd.to_numeric(trials["is_repeat"], errors="coerce").fillna(0).astype(int)
    n_before = len(trials)
    trials = trials[trials["is_repeat"] == 0].copy()
    n_after = len(trials)
    print(f"  Excluded {n_before - n_after} repeated-video trials "
          f"(movie IDs {sorted(REPEATED_VIDEO_IDS)}); {n_after} trials retained.")

    trials["resp.corr"] = pd.to_numeric(trials["resp.corr"], errors="coerce")
    trials["resp.rt"] = pd.to_numeric(trials["resp.rt"], errors="coerce")
    trials["conf_radio.response"] = pd.to_numeric(
        trials["conf_radio.response"], errors="coerce"
    )

    def _target_type(s):
        s = str(s)
        if "_BB_" in s:
            return "BB"
        if "_EM_" in s:
            return "EM"
        return np.nan

    trials["target_type"] = trials["target_img"].apply(_target_type)

    return trials, pp


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def describe(series) -> dict:
    s = pd.Series(series).dropna().astype(float)
    q1, med, q3 = np.percentile(s, [25, 50, 75]) if len(s) else (np.nan,) * 3
    return {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
    }


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return np.nan, np.nan
    d = (x.mean() - y.mean()) / sp
    J = 1 - (3 / (4 * (nx + ny) - 9))
    return d, d * J


def welch_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sp = math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    if sp == 0:
        return np.nan
    return (x.mean() - y.mean()) / sp


def rank_biserial_from_u(u, n1, n2):
    return (2.0 * u) / (n1 * n2) - 1.0


def mw_z_from_u(u, n1, n2):
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return np.nan
    return (u - mu) / sigma


def spearman_bootstrap_ci(x, y, n_boot: int = 5000, seed: int = RNG_SEED) -> dict:
    """Spearman ρ with bootstrap 95% CI (BCa not needed; percentile CI reported)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        br, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(br)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])
    sw_x = stats.shapiro(x) if 3 <= len(x) <= 5000 else (np.nan, np.nan)
    sw_y = stats.shapiro(y) if 3 <= len(y) <= 5000 else (np.nan, np.nan)
    return {
        "n": int(n),
        "rho": float(rho),
        "p_two_sided": float(p),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "n_boot": n_boot,
        "normality": {
            "x": {"shapiro_W": float(sw_x[0]), "p": float(sw_x[1])},
            "y": {"shapiro_W": float(sw_y[0]), "p": float(sw_y[1])},
        },
    }


def directional_pvalue(two_sided_p, observed_mean_diff, predicted_direction):
    if np.isnan(two_sided_p):
        return np.nan
    in_direction = (
        (predicted_direction == ">" and observed_mean_diff > 0)
        or (predicted_direction == "<" and observed_mean_diff < 0)
    )
    return two_sided_p / 2 if in_direction else 1 - two_sided_p / 2


# ──────────────────────────────────────────────────────────────────────
# Core two-group testing pipeline
# ──────────────────────────────────────────────────────────────────────
def two_group_test(
    label: str,
    dv_name: str,
    group1_name: str,
    group1_values: np.ndarray,
    group2_name: str,
    group2_values: np.ndarray,
    predicted_direction: str,
    alpha: float = ALPHA,
) -> dict:
    """Full decision pipeline: normality -> variance -> choose test -> report.

    Test selection rationale
    ------------------------
    1. Shapiro-Wilk normality test on each group separately (α = 0.05).
    2. If both groups are normally distributed, test homogeneity of variance
       with Levene's test (median-centred, i.e. Brown-Forsythe variant, which
       is robust to non-normality).
    3. Decision tree:
         both normal + equal variance  → Student's independent t-test
         both normal + unequal variance → Welch's t-test
         either non-normal             → Mann-Whitney U test

    Justification for Mann-Whitney U with bounded DVs
    -------------------------------------------------
    Several DVs in this study are bounded: overall accuracy is a proportion
    in [0, 1] (discrete: k/35 where k ∈ {0, …, 35} after excluding the 5
    repeated movies per participant); BB/EM sub-accuracies are k/17 or k/18;
    mean confidence is bounded [1, 5].

    Bounded scales impose a hard ceiling and floor that necessarily induce
    skew when participants cluster at either extreme.  In this experiment,
    mean accuracy ≈ 0.86 with ceiling pressure: the distribution is left-
    skewed and Shapiro-Wilk reliably rejects normality.  Transformations
    (e.g., arcsin-square-root) can reduce but rarely eliminate boundary
    effects for proportion data at these accuracy levels.

    Mann-Whitney U (Wilcoxon rank-sum) is appropriate here for three reasons:
      (a) It makes no distributional assumption beyond continuity — correct
          given the bounded, discretised nature of accuracy proportions.
      (b) It tests the null hypothesis that a randomly selected score from
          group 1 is equally likely to exceed or fall below a randomly
          selected score from group 2 (stochastic dominance), which directly
          addresses the research question.
      (c) With n ≈ 83-87 per condition, the Normal approximation to the U
          statistic is accurate (|z| reported), and the test has good power
          against location shifts.
    Effect size is the rank-biserial correlation r_rb (Kerby, 2014), which
    maps directly onto the probability of superiority and is interpretable
    on a [-1, 1] scale.
    """
    x = np.asarray(pd.Series(group1_values).dropna(), dtype=float)
    y = np.asarray(pd.Series(group2_values).dropna(), dtype=float)

    desc = {group1_name: describe(x), group2_name: describe(y)}

    sw_x = stats.shapiro(x) if 3 <= len(x) <= 5000 else (np.nan, np.nan)
    sw_y = stats.shapiro(y) if 3 <= len(y) <= 5000 else (np.nan, np.nan)
    normal_x = (not np.isnan(sw_x[1])) and sw_x[1] >= alpha
    normal_y = (not np.isnan(sw_y[1])) and sw_y[1] >= alpha
    both_normal = normal_x and normal_y

    lev_stat, lev_p = stats.levene(x, y, center="median")
    equal_var = lev_p >= alpha

    if both_normal and equal_var:
        test_name = "Student's independent-samples t-test"
        t_stat, t_p = stats.ttest_ind(x, y, equal_var=True)
        df = len(x) + len(y) - 2
        d_raw, d_hedges = cohens_d(x, y)
        effect_name = "Cohen's d (pooled)"
        effect_value = d_raw
        effect_hedges = d_hedges
    elif both_normal and not equal_var:
        test_name = "Welch's independent-samples t-test"
        t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        nx, ny = len(x), len(y)
        df_num = (vx / nx + vy / ny) ** 2
        df_den = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
        df = float(df_num / df_den) if df_den > 0 else np.nan
        effect_name = "Cohen's d (Welch)"
        effect_value = welch_d(x, y)
        _, effect_hedges = cohens_d(x, y)
    else:
        test_name = "Mann-Whitney U test (two-sided, continuity-corrected)"
        u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")
        z = mw_z_from_u(u_stat, len(x), len(y))
        rb = rank_biserial_from_u(u_stat, len(x), len(y))
        # Count tied ranks: observations sharing a value with at least one other
        from collections import Counter
        all_vals = np.concatenate([x, y])
        val_counts = Counter(np.round(all_vals, 10).tolist())
        n_tied_obs = int(sum(count for count in val_counts.values() if count > 1))
        n_tied_groups = int(sum(1 for count in val_counts.values() if count > 1))
        n_unique_vals = int(len(val_counts))
        n_total_obs = int(len(all_vals))
        # Tie correction factor: how much variance is reduced by ties
        tie_correction = float(
            1.0 - sum(t**3 - t for t in val_counts.values()) / (n_total_obs**3 - n_total_obs)
        ) if n_total_obs > 1 else 1.0
        return {
            "label": label,
            "dv": dv_name,
            "group1": group1_name,
            "group2": group2_name,
            "descriptives": desc,
            "normality": {
                group1_name: {"shapiro_W": float(sw_x[0]), "p": float(sw_x[1])},
                group2_name: {"shapiro_W": float(sw_y[0]), "p": float(sw_y[1])},
                "both_normal": bool(both_normal),
            },
            "levene": {"W": float(lev_stat), "p": float(lev_p),
                       "equal_var": bool(equal_var)},
            "test_used": test_name,
            "statistic_name": "U",
            "statistic": float(u_stat),
            "z_approx": float(z) if not np.isnan(z) else None,
            "df": None,
            "p_two_sided": float(u_p),
            "p_one_sided": float(
                directional_pvalue(u_p, x.mean() - y.mean(), predicted_direction)
            ),
            "predicted_direction": f"{group1_name} {predicted_direction} {group2_name}",
            "observed_direction_matches": bool(
                (predicted_direction == ">" and x.mean() > y.mean())
                or (predicted_direction == "<" and x.mean() < y.mean())
            ),
            "effect_size_name": "Rank-biserial r",
            "effect_size_value": float(rb),
            "n_ties": n_tied_obs,
            "n_tied_groups": n_tied_groups,
            "n_unique_vals": n_unique_vals,
            "n_total": n_total_obs,
            "tie_correction": tie_correction,
        }

    one_sided_p = directional_pvalue(
        float(t_p), x.mean() - y.mean(), predicted_direction
    )
    return {
        "label": label,
        "dv": dv_name,
        "group1": group1_name,
        "group2": group2_name,
        "descriptives": desc,
        "normality": {
            group1_name: {"shapiro_W": float(sw_x[0]), "p": float(sw_x[1])},
            group2_name: {"shapiro_W": float(sw_y[0]), "p": float(sw_y[1])},
            "both_normal": bool(both_normal),
        },
        "levene": {"W": float(lev_stat), "p": float(lev_p),
                   "equal_var": bool(equal_var)},
        "test_used": test_name,
        "statistic_name": "t",
        "statistic": float(t_stat),
        "z_approx": None,
        "df": float(df),
        "p_two_sided": float(t_p),
        "p_one_sided": float(one_sided_p),
        "predicted_direction": f"{group1_name} {predicted_direction} {group2_name}",
        "observed_direction_matches": bool(
            (predicted_direction == ">" and x.mean() > y.mean())
            or (predicted_direction == "<" and x.mean() < y.mean())
        ),
        "effect_size_name": effect_name,
        "effect_size_value": float(effect_value),
        "effect_size_hedges_g": float(effect_hedges)
        if effect_hedges is not None
        else None,
    }


# ──────────────────────────────────────────────────────────────────────
# Paired test (H6 — BB vs EM log RT within participants)
# ──────────────────────────────────────────────────────────────────────
def paired_test(
    label: str,
    dv_name: str,
    group1_name: str,
    group1_values,
    group2_name: str,
    group2_values,
) -> dict:
    x = np.asarray(pd.Series(group1_values).dropna(), dtype=float)
    y = np.asarray(pd.Series(group2_values).dropna(), dtype=float)
    diffs = x - y
    n = len(diffs)

    desc_g1 = describe(x)
    desc_g2 = describe(y)
    desc_diff = describe(diffs)

    sw = stats.shapiro(diffs) if 3 <= n <= 5000 else (np.nan, np.nan)
    normal_diffs = (not np.isnan(sw[1])) and sw[1] >= ALPHA

    if normal_diffs:
        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        d_z = float(diffs.mean() / diffs.std(ddof=1))
        return {
            "label": label,
            "dv": dv_name,
            "group1": group1_name,
            "group2": group2_name,
            "n_pairs": int(n),
            "descriptives": {group1_name: desc_g1, group2_name: desc_g2,
                             "difference": desc_diff},
            "normality_of_diffs": {
                "shapiro_W": float(sw[0]), "p": float(sw[1]),
                "normal": bool(normal_diffs),
            },
            "test_used": "One-sample t-test on differences (paired, two-tailed)",
            "statistic_name": "t",
            "statistic": float(t_stat),
            "df": float(n - 1),
            "p_two_sided": float(p_val),
            "effect_size_name": "Cohen's d_z",
            "effect_size_value": float(d_z),
        }
    else:
        w_stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
        mu_w = n * (n + 1) / 4.0
        sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z_w = (w_stat - mu_w) / sigma_w if sigma_w > 0 else np.nan
        r_rb = float(z_w / math.sqrt(n)) if not np.isnan(z_w) else np.nan
        return {
            "label": label,
            "dv": dv_name,
            "group1": group1_name,
            "group2": group2_name,
            "n_pairs": int(n),
            "descriptives": {group1_name: desc_g1, group2_name: desc_g2,
                             "difference": desc_diff},
            "normality_of_diffs": {
                "shapiro_W": float(sw[0]), "p": float(sw[1]),
                "normal": bool(normal_diffs),
            },
            "test_used": "Wilcoxon signed-rank test (paired, two-tailed)",
            "statistic_name": "W",
            "statistic": float(w_stat),
            "df": None,
            "p_two_sided": float(p_val),
            "effect_size_name": "Rank-biserial r (z/sqrt(n))",
            "effect_size_value": float(r_rb),
        }


# ──────────────────────────────────────────────────────────────────────
# Per-hypothesis DV construction
# ──────────────────────────────────────────────────────────────────────
def build_dvs(trials: pd.DataFrame) -> dict:
    """Construct per-participant DVs.

    After repeated-video exclusion each participant sees 35 unique movies:
    approximately 17-18 BB + 17-18 EM targets depending on whether any of the
    5 excluded movies had BB or EM frames.  All proportions are computed over
    the retained trials only.
    """
    t = trials.copy()

    pp_acc = (
        t.groupby(["participant_id", "condition"])["resp.corr"]
        .mean()
        .reset_index(name="acc")
    )

    bb = t[t.target_type == "BB"]
    pp_bb = (
        bb.groupby(["participant_id", "condition"])["resp.corr"]
        .mean()
        .reset_index(name="bb_acc")
    )

    em = t[t.target_type == "EM"]
    pp_em = (
        em.groupby(["participant_id", "condition"])["resp.corr"]
        .mean()
        .reset_index(name="em_acc")
    )

    pp_conf = (
        t.groupby(["participant_id", "condition"])["conf_radio.response"]
        .mean()
        .reset_index(name="conf")
    )

    pp_lc = (
        t.assign(low=lambda d: (d["conf_radio.response"] <= 3).astype(float))
        .groupby(["participant_id", "condition"])["low"]
        .mean()
        .reset_index(name="prop_lc")
    )

    pp_movie_acc = (
        t.groupby(["participant_id", "condition", "movie_id"])["resp.corr"]
        .mean()
        .reset_index(name="movie_acc")
    )
    pp_acc_sd = (
        pp_movie_acc.groupby(["participant_id", "condition"])["movie_acc"]
        .std(ddof=1)
        .reset_index(name="movie_acc_sd")
    )

    t_rt = t.dropna(subset=["resp.rt"]).copy()
    t_rt = t_rt[t_rt["resp.rt"] > 0]
    t_rt["log_rt"] = np.log(t_rt["resp.rt"])
    pp_logrt = (
        t_rt.groupby(["participant_id", "condition"])["log_rt"]
        .mean()
        .reset_index(name="mean_log_rt")
    )
    pp_rawrt = (
        t_rt.groupby(["participant_id", "condition"])["resp.rt"]
        .mean()
        .reset_index(name="mean_rt")
    )

    bb_rt = (
        t_rt[t_rt.target_type == "BB"]
        .groupby(["participant_id", "condition"])["log_rt"]
        .mean()
        .reset_index(name="bb_log_rt")
    )
    em_rt = (
        t_rt[t_rt.target_type == "EM"]
        .groupby(["participant_id", "condition"])["log_rt"]
        .mean()
        .reset_index(name="em_log_rt")
    )
    pp_h6 = bb_rt.merge(em_rt, on=["participant_id", "condition"], how="inner")

    # H7 / H8 — participant-level: mean log RT paired with accuracy and confidence
    pp_h78 = pp_logrt.merge(pp_acc, on=["participant_id", "condition"], how="inner")
    pp_h78 = pp_h78.merge(pp_conf, on=["participant_id", "condition"], how="inner")

    return {
        "H1": pp_acc,
        "H2": pp_bb,
        "H3": pp_conf,
        "H4": pp_lc,
        "H5": pp_acc_sd,
        "H6": pp_h6,
        "H78": pp_h78,   # mean_log_rt, acc, conf per participant
        "EM_rec": pp_em,
        "RT_log": pp_logrt,
        "RT_raw": pp_rawrt,
        "trial_rt": t_rt,
    }


# ──────────────────────────────────────────────────────────────────────
# RT normality diagnostic plots
# ──────────────────────────────────────────────────────────────────────
def plot_rt_normality(dvs: dict, out_dir: Path) -> dict:
    pp_raw = dvs["RT_raw"]
    pp_log = dvs["RT_log"]
    trial_rt = dvs["trial_rt"]

    diagnostics: dict = {}

    for cond in ["NB", "AB"]:
        raw_vals = pp_raw.loc[pp_raw.condition == cond, "mean_rt"].values
        log_vals = pp_log.loc[pp_log.condition == cond, "mean_log_rt"].values
        sw_raw = stats.shapiro(raw_vals)
        sw_log = stats.shapiro(log_vals)
        diagnostics[f"pp_raw_{cond}"] = {
            "n": int(len(raw_vals)),
            "mean": float(raw_vals.mean()),
            "sd": float(raw_vals.std(ddof=1)),
            "shapiro_W": float(sw_raw.statistic),
            "shapiro_p": float(sw_raw.pvalue),
            "normal_alpha05": bool(sw_raw.pvalue >= ALPHA),
        }
        diagnostics[f"pp_log_{cond}"] = {
            "n": int(len(log_vals)),
            "mean": float(log_vals.mean()),
            "sd": float(log_vals.std(ddof=1)),
            "shapiro_W": float(sw_log.statistic),
            "shapiro_p": float(sw_log.pvalue),
            "normal_alpha05": bool(sw_log.pvalue >= ALPHA),
        }

    trial_raw = trial_rt["resp.rt"].values
    trial_log = trial_rt["log_rt"].values
    rng = np.random.default_rng(RNG_SEED)
    if len(trial_raw) > 5000:
        idx = rng.choice(len(trial_raw), size=5000, replace=False)
        sw_trial_raw = stats.shapiro(trial_raw[idx])
        sw_trial_log = stats.shapiro(trial_log[idx])
        diagnostics["trial_sampled"] = True
    else:
        sw_trial_raw = stats.shapiro(trial_raw)
        sw_trial_log = stats.shapiro(trial_log)
        diagnostics["trial_sampled"] = False
    diagnostics["trial_raw"] = {
        "n_used_for_shapiro": min(len(trial_raw), 5000),
        "shapiro_W": float(sw_trial_raw.statistic),
        "shapiro_p": float(sw_trial_raw.pvalue),
        "skew": float(stats.skew(trial_raw)),
        "kurtosis": float(stats.kurtosis(trial_raw)),
    }
    diagnostics["trial_log"] = {
        "n_used_for_shapiro": min(len(trial_log), 5000),
        "shapiro_W": float(sw_trial_log.statistic),
        "shapiro_p": float(sw_trial_log.pvalue),
        "skew": float(stats.skew(trial_log)),
        "kurtosis": float(stats.kurtosis(trial_log)),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    colours = {"NB": "#1f77b4", "AB": "#d62728"}

    for col, cond in enumerate(["NB", "AB"]):
        raw_vals = pp_raw.loc[pp_raw.condition == cond, "mean_rt"].values
        log_vals = pp_log.loc[pp_log.condition == cond, "mean_log_rt"].values

        ax = axes[0, col * 2]
        ax.hist(raw_vals, bins=20, color=colours[cond], alpha=0.75, edgecolor="black")
        sw_p = diagnostics[f"pp_raw_{cond}"]["shapiro_p"]
        ax.set_title(f"{cond} — raw mean RT\nShapiro p = {sw_p:.4g}")
        ax.set_xlabel("mean RT (s)")
        ax.set_ylabel("count")

        ax = axes[1, col * 2]
        stats.probplot(raw_vals, dist="norm", plot=ax)
        ax.set_title(f"{cond} — raw Q-Q")
        ax.get_lines()[0].set_color(colours[cond])

        ax = axes[0, col * 2 + 1]
        ax.hist(log_vals, bins=20, color=colours[cond], alpha=0.75, edgecolor="black")
        sw_p_log = diagnostics[f"pp_log_{cond}"]["shapiro_p"]
        ax.set_title(f"{cond} — log mean RT\nShapiro p = {sw_p_log:.4g}")
        ax.set_xlabel("ln mean RT")
        ax.set_ylabel("count")

        ax = axes[1, col * 2 + 1]
        stats.probplot(log_vals, dist="norm", plot=ax)
        ax.set_title(f"{cond} — log Q-Q")
        ax.get_lines()[0].set_color(colours[cond])

    fig.suptitle(
        "Participant-level RT normality — raw vs log transform",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pp_path = out_dir / "rt_normality_participant_level.png"
    fig.savefig(pp_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.hist(trial_raw, bins=80, color="#555", alpha=0.8, edgecolor="black")
    sw = diagnostics["trial_raw"]
    ax.set_title(
        f"Trial raw RT  (n={len(trial_raw)}, skew={sw['skew']:.2f})\n"
        f"Shapiro (n≤5000) p = {sw['shapiro_p']:.3g}"
    )
    ax.set_xlabel("RT (s)")
    ax.set_ylabel("count")

    ax = axes[0, 1]
    ax.hist(trial_log, bins=80, color="#2ca02c", alpha=0.8, edgecolor="black")
    sw = diagnostics["trial_log"]
    ax.set_title(
        f"Trial log RT  (skew={sw['skew']:.2f})\n"
        f"Shapiro (n≤5000) p = {sw['shapiro_p']:.3g}"
    )
    ax.set_xlabel("ln RT")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    sample_raw = trial_raw if len(trial_raw) <= 2000 else rng.choice(
        trial_raw, size=2000, replace=False
    )
    stats.probplot(sample_raw, dist="norm", plot=ax)
    ax.set_title("Trial raw RT — Q-Q")

    ax = axes[1, 1]
    sample_log = trial_log if len(trial_log) <= 2000 else rng.choice(
        trial_log, size=2000, replace=False
    )
    stats.probplot(sample_log, dist="norm", plot=ax)
    ax.set_title("Trial log RT — Q-Q")

    fig.suptitle(
        "Trial-level RT normality — raw vs log transform",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    trial_path = out_dir / "rt_normality_trial_level.png"
    fig.savefig(trial_path, dpi=150)
    plt.close(fig)

    diagnostics["plot_paths"] = {
        "participant_level": str(pp_path),
        "trial_level": str(trial_path),
    }
    return diagnostics


# ──────────────────────────────────────────────────────────────────────
# H6 plot
# ──────────────────────────────────────────────────────────────────────
def plot_h6_bb_em_logrt(dvs: dict, out_dir: Path) -> str:
    h6 = dvs["H6"]
    bb_vals = h6["bb_log_rt"].values
    em_vals = h6["em_log_rt"].values
    diffs = bb_vals - em_vals

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = {"BB": "#e07b39", "EM": "#5b7fba"}

    ax = axes[0]
    parts = ax.violinplot([bb_vals, em_vals], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], [colours["BB"], colours["EM"]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    rng = np.random.default_rng(RNG_SEED)
    jitter = rng.uniform(-0.06, 0.06, size=len(bb_vals))
    ax.scatter(1 + jitter, bb_vals, color=colours["BB"], alpha=0.45, s=18, zorder=3)
    ax.scatter(2 + jitter, em_vals, color=colours["EM"], alpha=0.45, s=18, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["BB targets", "EM targets"], fontsize=11)
    ax.set_ylabel("Mean log RT (ln s)", fontsize=11)
    ax.set_title("BB vs EM: mean log RT per participant", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.hist(diffs, bins=25, color="#7a5c9e", alpha=0.8, edgecolor="black")
    ax.axvline(0, color="red", lw=1.8, linestyle="--", label="Zero (no diff)")
    ax.axvline(float(np.mean(diffs)), color="black", lw=1.8,
               linestyle="-", label=f"Mean diff = {np.mean(diffs):.3f}")
    ax.set_xlabel("BB − EM mean log RT (ln s)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of BB − EM log RT differences", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "H6: BB-frame vs EM-frame mean log RT (within-participant)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "h6_bb_em_logrt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    trials, pp = load_data()

    print("=" * 72)
    print(f"Vigilance exclusions: {sorted(VIGILANCE_EXCLUSIONS)}")
    print(f"Repeated-video exclusion: movie IDs {sorted(REPEATED_VIDEO_IDS)}")
    print(f"Analysed: {len(pp)} participants, {len(trials)} trials")
    print(pp.condition.value_counts().to_dict())
    print("=" * 72)

    dvs = build_dvs(trials)
    rt_diag = plot_rt_normality(dvs, OUT_DIR)
    h6_plot_path = plot_h6_bb_em_logrt(dvs, OUT_DIR)

    results = {}
    results["_meta"] = {
        "vigilance_exclusions": sorted(VIGILANCE_EXCLUSIONS),
        "repeated_video_ids_excluded": sorted(REPEATED_VIDEO_IDS),
        "n_excluded_vigilance": len(VIGILANCE_EXCLUSIONS),
        "n_analysed": int(len(pp)),
        "n_trials": int(len(trials)),
    }
    results["_rt_normality_diagnostics"] = rt_diag
    results["_h6_plot_path"] = h6_plot_path

    # ── H1: overall recognition accuracy ──────────────────────────────
    H1 = dvs["H1"]
    results["H1"] = two_group_test(
        label="H1",
        dv_name="Overall accuracy (proportion correct, repeated movies excluded)",
        group1_name="NB",
        group1_values=H1.loc[H1.condition == "NB", "acc"].values,
        group2_name="AB",
        group2_values=H1.loc[H1.condition == "AB", "acc"].values,
        predicted_direction=">",
    )

    # ── H2: BB-frame accuracy ──────────────────────────────────────────
    H2 = dvs["H2"]
    results["H2"] = two_group_test(
        label="H2",
        dv_name="Before-Boundary accuracy (repeated movies excluded)",
        group1_name="NB",
        group1_values=H2.loc[H2.condition == "NB", "bb_acc"].values,
        group2_name="AB",
        group2_values=H2.loc[H2.condition == "AB", "bb_acc"].values,
        predicted_direction=">",
    )

    # ── H3: mean confidence ───────────────────────────────────────────
    H3 = dvs["H3"]
    results["H3"] = two_group_test(
        label="H3",
        dv_name="Mean confidence (1-5 Likert, repeated movies excluded)",
        group1_name="NB",
        group1_values=H3.loc[H3.condition == "NB", "conf"].values,
        group2_name="AB",
        group2_values=H3.loc[H3.condition == "AB", "conf"].values,
        predicted_direction=">",
    )

    # ── H4: proportion of low-confidence trials ───────────────────────
    H4 = dvs["H4"]
    results["H4"] = two_group_test(
        label="H4",
        dv_name="Proportion of low-confidence trials (conf<=3, repeated excluded)",
        group1_name="AB",
        group1_values=H4.loc[H4.condition == "AB", "prop_lc"].values,
        group2_name="NB",
        group2_values=H4.loc[H4.condition == "NB", "prop_lc"].values,
        predicted_direction=">",
    )

    # ── H5: per-movie accuracy SD ──────────────────────────────────────
    H5 = dvs["H5"]
    results["H5"] = two_group_test(
        label="H5",
        dv_name="SD of per-movie accuracy (repeated movies excluded)",
        group1_name="AB",
        group1_values=H5.loc[H5.condition == "AB", "movie_acc_sd"].values,
        group2_name="NB",
        group2_values=H5.loc[H5.condition == "NB", "movie_acc_sd"].values,
        predicted_direction=">",
    )

    # ── H6: BB vs EM log RT — paired ────────────────────────────────
    H6 = dvs["H6"]
    results["H6"] = paired_test(
        label="H6",
        dv_name="BB vs EM mean log RT (within-participant paired, repeated excluded)",
        group1_name="BB",
        group1_values=H6["bb_log_rt"].values,
        group2_name="EM",
        group2_values=H6["em_log_rt"].values,
    )

    # ── H7: Spearman correlation — mean log RT vs overall accuracy ────
    H78 = dvs["H78"]
    results["H7"] = {
        "label": "H7",
        "dv_x": "Mean log RT (ln s)",
        "dv_y": "Overall accuracy (proportion correct, repeated excluded)",
        "type": "spearman_correlation",
        **spearman_bootstrap_ci(H78["mean_log_rt"].values, H78["acc"].values),
    }

    # ── H8: Spearman correlation — mean log RT vs mean confidence ─────
    results["H8"] = {
        "label": "H8",
        "dv_x": "Mean log RT (ln s)",
        "dv_y": "Mean confidence rating (1–5, repeated excluded)",
        "type": "spearman_correlation",
        **spearman_bootstrap_ci(H78["mean_log_rt"].values, H78["conf"].values),
    }

    # ── Secondary: log-RT NB vs AB ────────────────────────────────────
    ppRT = dvs["RT_log"]
    results["Secondary_RT"] = two_group_test(
        label="Secondary_RT",
        dv_name="Participant-level mean log response time (ln s, repeated excluded)",
        group1_name="NB",
        group1_values=ppRT.loc[ppRT.condition == "NB", "mean_log_rt"].values,
        group2_name="AB",
        group2_values=ppRT.loc[ppRT.condition == "AB", "mean_log_rt"].values,
        predicted_direction="<",
    )

    # ── Secondary: EM-frame accuracy ──────────────────────────────────
    EM = dvs["EM_rec"]
    results["Secondary_EM"] = two_group_test(
        label="Secondary_EM",
        dv_name="Event-middle accuracy (repeated excluded)",
        group1_name="NB",
        group1_values=EM.loc[EM.condition == "NB", "em_acc"].values,
        group2_name="AB",
        group2_values=EM.loc[EM.condition == "AB", "em_acc"].values,
        predicted_direction=">",
    )

    # ── Secondary: trial-level log-RT Welch + MW ──────────────────────
    t_rt = dvs["trial_rt"]
    rt_nb = np.log(t_rt.loc[t_rt.condition == "NB", "resp.rt"].values.astype(float))
    rt_ab = np.log(t_rt.loc[t_rt.condition == "AB", "resp.rt"].values.astype(float))
    t_stat, t_p = stats.ttest_ind(rt_nb, rt_ab, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(rt_nb, rt_ab, alternative="two-sided")
    results["Secondary_RT_trial_level"] = {
        "label": "Secondary_RT_trial",
        "dv": "Trial-level log response time (ln s, repeated excluded)",
        "n_NB": int(len(rt_nb)),
        "n_AB": int(len(rt_ab)),
        "mean_log_rt_NB": float(rt_nb.mean()),
        "mean_log_rt_AB": float(rt_ab.mean()),
        "welch_t": float(t_stat),
        "welch_p_two_sided": float(t_p),
        "mann_whitney_U": float(u_stat),
        "mann_whitney_p_two_sided": float(u_p),
        "rank_biserial": float(rank_biserial_from_u(u_stat, len(rt_nb), len(rt_ab))),
    }

    # ── Save JSON ─────────────────────────────────────────────────────
    out_json = OUT_DIR / "hypothesis_test_results.json"

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    out_json.write_text(json.dumps(_clean(results), indent=2))
    print(f"\nSaved JSON -> {out_json}")

    # ── CSV summary ───────────────────────────────────────────────────
    rows = []
    group_labels = ["H1", "H2", "H3", "H4", "H5"]
    secondary_labels = ["Secondary_RT", "Secondary_EM"]

    for h in group_labels + secondary_labels:
        r = results[h]
        g1, g2 = r["group1"], r["group2"]
        d1, d2 = r["descriptives"][g1], r["descriptives"][g2]
        rows.append({
            "label": r["label"],
            "dv": r["dv"],
            "g1": g1, "g2": g2,
            "n_g1": d1["n"], "n_g2": d2["n"],
            "mean_g1": d1["mean"], "mean_g2": d2["mean"],
            "sd_g1": d1["sd"], "sd_g2": d2["sd"],
            "median_g1": d1["median"], "median_g2": d2["median"],
            "shapiro_p_g1": r["normality"][g1]["p"],
            "shapiro_p_g2": r["normality"][g2]["p"],
            "levene_p": r["levene"]["p"],
            "test_used": r["test_used"],
            "statistic": r["statistic"],
            "df": r.get("df"),
            "p_two_sided": r["p_two_sided"],
            "p_one_sided": r["p_one_sided"],
            "effect_size_name": r["effect_size_name"],
            "effect_size_value": r["effect_size_value"],
        })

    # H6 row
    r6 = results["H6"]
    d_bb = r6["descriptives"]["BB"]
    d_em = r6["descriptives"]["EM"]
    rows.append({
        "label": r6["label"],
        "dv": r6["dv"],
        "g1": "BB", "g2": "EM",
        "n_g1": r6["n_pairs"], "n_g2": r6["n_pairs"],
        "mean_g1": d_bb["mean"], "mean_g2": d_em["mean"],
        "sd_g1": d_bb["sd"], "sd_g2": d_em["sd"],
        "median_g1": d_bb["median"], "median_g2": d_em["median"],
        "shapiro_p_g1": r6["normality_of_diffs"]["p"],
        "shapiro_p_g2": None,
        "levene_p": None,
        "test_used": r6["test_used"],
        "statistic": r6["statistic"],
        "df": r6.get("df"),
        "p_two_sided": r6["p_two_sided"],
        "p_one_sided": None,
        "effect_size_name": r6["effect_size_name"],
        "effect_size_value": r6["effect_size_value"],
    })

    # H7 and H8 rows (Spearman correlations)
    for h in ["H7", "H8"]:
        rc = results[h]
        rows.append({
            "label": rc["label"],
            "dv": f"{rc['dv_x']} ~ {rc['dv_y']}",
            "g1": "x", "g2": "y",
            "n_g1": rc["n"], "n_g2": rc["n"],
            "mean_g1": None, "mean_g2": None,
            "sd_g1": None, "sd_g2": None,
            "median_g1": None, "median_g2": None,
            "shapiro_p_g1": rc["normality"]["x"]["p"],
            "shapiro_p_g2": rc["normality"]["y"]["p"],
            "levene_p": None,
            "test_used": "Spearman rank correlation (two-tailed, bootstrap CI)",
            "statistic": rc["rho"],
            "df": rc["n"] - 2,
            "p_two_sided": rc["p_two_sided"],
            "p_one_sided": None,
            "effect_size_name": "Spearman rho",
            "effect_size_value": rc["rho"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "hypothesis_test_summary.csv", index=False)
    print(f"Saved CSV  -> {OUT_DIR / 'hypothesis_test_summary.csv'}")

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" PRIMARY HYPOTHESES (H1–H8) — DECISION SUMMARY")
    print("=" * 72)
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        r = results[h]
        g1, g2 = r["group1"], r["group2"]
        d1, d2 = r["descriptives"][g1], r["descriptives"][g2]
        print(
            f"[{h}] {r['dv']}\n"
            f"    {g1}: n={d1['n']} M={d1['mean']:.4f} SD={d1['sd']:.4f}"
            f"  |  {g2}: n={d2['n']} M={d2['mean']:.4f} SD={d2['sd']:.4f}\n"
            f"    Shapiro p: {g1}={r['normality'][g1]['p']:.4g}  "
            f"{g2}={r['normality'][g2]['p']:.4g}  "
            f"Levene p={r['levene']['p']:.4g}\n"
            f"    Test: {r['test_used']}\n"
            f"    Stat={r['statistic']:.4f}  df={r.get('df')}  "
            f"p2={r['p_two_sided']:.4g}  p1={r['p_one_sided']:.4g}\n"
            f"    Effect ({r['effect_size_name']}) = {r['effect_size_value']:.4f}\n"
        )
    for h in ["H7", "H8"]:
        rc = results[h]
        print(
            f"[{h}] {rc['dv_x']} ~ {rc['dv_y']}\n"
            f"    n={rc['n']}, Spearman rho={rc['rho']:.4f}, "
            f"p2={rc['p_two_sided']:.4g}, "
            f"95% CI [{rc['ci_95_lo']:.4f}, {rc['ci_95_hi']:.4f}]\n"
        )

    print("=" * 72)
    print(" H6 — BB vs EM LOG RT (PAIRED)")
    print("=" * 72)
    r6 = results["H6"]
    d_bb = r6["descriptives"]["BB"]
    d_em = r6["descriptives"]["EM"]
    d_diff = r6["descriptives"]["difference"]
    print(
        f"    n pairs = {r6['n_pairs']}\n"
        f"    BB: M={d_bb['mean']:.4f}  EM: M={d_em['mean']:.4f}\n"
        f"    Diff BB-EM: M={d_diff['mean']:.4f} SD={d_diff['sd']:.4f}\n"
        f"    Shapiro on diffs p={r6['normality_of_diffs']['p']:.4g}\n"
        f"    Test: {r6['test_used']}\n"
        f"    Stat={r6['statistic']:.4f}  p2={r6['p_two_sided']:.4g}\n"
        f"    Effect ({r6['effect_size_name']}) = {r6['effect_size_value']:.4f}\n"
    )

    return results


if __name__ == "__main__":
    np.random.seed(RNG_SEED)
    main()
