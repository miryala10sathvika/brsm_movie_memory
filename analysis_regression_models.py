"""Regression models for the BRSM Movie Memory experiment.

This script implements a *strict, assignment-style* regression pipeline for:

  - Participant-level baseline model (Model 3)
  - Trial-level clustered logistic regression (Model 1) using GEE

Key principles (to avoid "random models")
-----------------------------------------
1) Feature eligibility is pre-defined *before* fitting:
   - Include design variables that occur *before* the response:
       condition (AB vs NB), target_type (BB vs EM), is_repeat
   - Include pre-treatment participant covariates (optional controls):
       age, gender, handedness, vision

2) Explicit exclusions (data leakage):
   - Exclude confidence and RT when predicting accuracy.
       conf_radio.response and resp.rt are measured during/after the decision.

3) Repeated measures handled correctly:
   - Trials are clustered within participant -> use GEE (Binomial + robust SE).

4) Model comparison is systematic:
   - Compare nested models by QIC (when available) rather than ad-hoc choices.

Outputs
-------
Writes tables/plots into statistical_results/regression/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


DATA_DIR = Path("final_cleaned_data")
OUT_DIR = Path("statistical_results") / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Participants excluded for failing the vigilance check (consistent with run_statistical_tests.py)
VIGILANCE_EXCLUSIONS = {"sub105_AB", "sub70_NB"}


def _target_type(target_img: str) -> str | float:
    s = str(target_img)
    if "_BB_" in s:
        return "BB"
    if "_EM_" in s:
        return "EM"
    return np.nan


def load_and_merge_trial_level() -> pd.DataFrame:
    """Load trials with `is_repeat` and merge participant demographics.

    Why not use trials_with_demographics_final.csv?
    - In this repo, `trials_with_demographics_final.csv` often has `is_repeat` missing.
      We therefore load `trials_final_clean_with_repeat.csv` for `is_repeat` and merge
      demographics from `participants_final_clean_with_vigilance.csv`.
    """

    trials = pd.read_csv(DATA_DIR / "trials_final_clean_with_repeat.csv")
    pp = pd.read_csv(DATA_DIR / "participants_final_clean_with_vigilance.csv")

    # Exclude vigilance failures up front
    pp = pp[~pp["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()
    trials = trials[~trials["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()

    # Core typing
    trials["resp.corr"] = pd.to_numeric(trials["resp.corr"], errors="coerce")
    trials["is_repeat"] = pd.to_numeric(trials["is_repeat"], errors="coerce")

    # Derive target_type
    trials["target_type"] = trials["target_img"].apply(_target_type)

    # Merge covariates (pre-treatment)
    covars = pp[[
        "participant_id",
        "age",
        "gender",
        "handedness",
        "vision",
    ]].copy()

    merged = trials.merge(covars, on="participant_id", how="left")
    merged = merge_stimulus_metadata(merged)
    return merged


def merge_stimulus_metadata(trials: pd.DataFrame) -> pd.DataFrame:
    """Merge stimulus-level metadata (duration) by (condition, movie_id).

    `abruptmovies.csv` and `naturalmovies.csv` each contain 40 unique movies,
    with additional duplicate rows where `is_repeat==1`. We build a stable
    movie_id mapping from the *first occurrence* of each unique path.

    This adds:
      - movie_duration (seconds)
    """

    def _make_movie_table(csv_path: Path, condition: str) -> pd.DataFrame:
        m = pd.read_csv(csv_path)
        m = m.dropna(subset=["path", "duration"]).copy()
        m["path"] = m["path"].astype(str)
        # Keep first occurrence of each unique movie path (preserve order)
        m_first = m.drop_duplicates(subset=["path"], keep="first").reset_index(drop=True)
        m_first["movie_id"] = np.arange(1, len(m_first) + 1)
        m_first["condition"] = condition
        m_first = m_first.rename(columns={"duration": "movie_duration"})
        return m_first[["condition", "movie_id", "movie_duration"]]

    ab = _make_movie_table(Path("abruptmovies.csv"), "AB")
    nb = _make_movie_table(Path("naturalmovies.csv"), "NB")
    movie_table = pd.concat([ab, nb], ignore_index=True)
    out = trials.merge(movie_table, on=["condition", "movie_id"], how="left")
    out["movie_duration"] = pd.to_numeric(out["movie_duration"], errors="coerce")
    out["movie_id"] = pd.to_numeric(out["movie_id"], errors="coerce")
    out["stimulus_id"] = (
        out["condition"].astype(str)
        + "_"
        + out["movie_id"].round(0).astype("Int64").astype(str)
    )
    return out


def is_gee_result_stable(result) -> bool:
    """Heuristic: treat models with non-finite params/SEs as unstable."""
    params = np.asarray(result.params)
    bse = np.asarray(getattr(result, "bse", np.full_like(params, np.nan)))
    return bool(np.isfinite(params).all() and np.isfinite(bse).all())


def load_participant_level() -> pd.DataFrame:
    pp = pd.read_csv(DATA_DIR / "participants_final_clean_with_vigilance.csv")
    pp = pp[~pp["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()
    return pp


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = OUT_DIR / filename
    df.to_csv(path, index=False)


def compute_vif_from_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIFs on a numeric design matrix.

    Notes:
    - VIF is most interpretable for linear models, but we use it here as a
      pragmatic collinearity diagnostic for the covariate set.
    """
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    X = sm.add_constant(X, has_constant="add")

    vifs = []
    values = X.to_numpy(dtype=float)
    for i, col in enumerate(X.columns):
        vifs.append({"variable": col, "VIF": float(variance_inflation_factor(values, i))})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


def odds_ratio_table(result) -> pd.DataFrame:
    ci = result.conf_int()
    out = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "se": result.bse.values,
            "z_or_t": result.tvalues.values,
            "p": result.pvalues.values,
            "OR": np.exp(result.params.values),
            "OR_ci_low": np.exp(ci[0].values),
            "OR_ci_high": np.exp(ci[1].values),
        }
    )
    return out


@dataclass(frozen=True)
class ModelSpec:
    name: str
    formula: str
    description: str


def fit_gee_binomial(formula: str, data: pd.DataFrame, group_col: str = "participant_id"):
    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()
    # For binomial models, scale is effectively fixed at 1.
    # In statsmodels, scale is set on .fit(), not on model construction.
    model = smf.gee(
        formula,
        groups=data[group_col],
        data=data,
        family=fam,
        cov_struct=cov,
    )
    return model.fit(scale=1.0)


def try_get_qic(result) -> float | None:
    """Return QIC if available, else None."""
    qic = getattr(result, "qic", None)
    if qic is None:
        return None
    try:
        # statsmodels returns a tuple (QIC, QICu)
        if callable(qic):
            try:
                return float(qic(scale=1.0)[0])
            except TypeError:
                return float(qic()[0])
        return float(qic[0])
    except Exception:
        return None


def plot_accuracy_bars(df: pd.DataFrame) -> None:
    """Simple EDA plot: mean accuracy by condition and target_type."""
    agg = (
        df.groupby(["condition", "target_type"])["resp.corr"]
        .mean()
        .reset_index(name="mean_acc")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, sub in agg.groupby("condition"):
        ax.plot(sub["target_type"], sub["mean_acc"], marker="o", label=cond)
    ax.set_title("Trial accuracy by condition and target type")
    ax.set_xlabel("target_type")
    ax.set_ylabel("mean(resp.corr)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="condition")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_accuracy_by_condition_target_type.png", dpi=160)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Assignment-style EDA: histograms of numeric predictors."""
    numeric_cols = ["age", "movie_duration", "is_repeat"]
    available = [c for c in numeric_cols if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(x, bins=30, color="#4c78a8", alpha=0.85, edgecolor="black")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_numeric_distributions.png", dpi=160)
    plt.close(fig)


def plot_numeric_correlations(df: pd.DataFrame) -> None:
    """Assignment-style EDA: correlation heatmap for numeric predictors."""
    numeric_cols = ["age", "movie_duration", "is_repeat"]
    x = df[[c for c in numeric_cols if c in df.columns]].copy()
    if x.shape[1] < 2:
        return
    x = x.apply(pd.to_numeric, errors="coerce")
    corr = x.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[0]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap (numeric predictors)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_numeric_correlation_heatmap.png", dpi=160)
    plt.close(fig)


def heteroscedasticity_diagnostics_ols(pp: pd.DataFrame) -> pd.DataFrame:
    """Run assignment-style heteroscedasticity diagnostics on an OLS surrogate.

    Note: Breusch–Pagan is for linear regression. Our primary Model 1 is
    logistic (GEE), so this is provided as a familiar diagnostic analogous
    to your assignment, using participant-level accuracy.
    """
    d = pp[["accuracy", "condition", "age", "gender"]].copy()
    d["accuracy"] = pd.to_numeric(d["accuracy"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d = d.dropna(subset=["accuracy", "condition"]).copy()

    model = smf.ols("accuracy ~ C(condition, Treatment('NB')) + age + C(gender)", data=d).fit()

    # BP test
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)

    # Residual plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.4, s=18, color="#4c78a8")
    ax.axhline(0, color="red", lw=1)
    ax.set_title("OLS surrogate: residuals vs fitted")
    ax.set_xlabel("fitted accuracy")
    ax.set_ylabel("residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostic_ols_residuals_vs_fitted.png", dpi=160)
    plt.close(fig)

    out = pd.DataFrame(
        [
            {
                "model": "OLS_surrogate_accuracy",
                "bp_lm": float(lm_stat),
                "bp_lm_p": float(lm_p),
                "bp_f": float(f_stat),
                "bp_f_p": float(f_p),
                "n": int(len(d)),
                "r2": float(model.rsquared),
            }
        ]
    )
    save_dataframe(out, "diagnostic_breusch_pagan.csv")
    return out


def plot_logistic_residuals(result, df: pd.DataFrame, filename: str) -> None:
    """Diagnostic: Pearson residuals vs fitted probabilities (logistic GEE)."""
    fitted = result.predict(df)
    y = df["resp.corr"].astype(float).values
    var = np.clip(fitted * (1 - fitted), 1e-6, None)
    pearson = (y - fitted) / np.sqrt(var)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fitted, pearson, alpha=0.25, s=10, color="#4c78a8")
    ax.axhline(0, color="red", lw=1)
    ax.set_title("Model 1 diagnostic: Pearson residuals vs fitted")
    ax.set_xlabel("fitted P(correct)")
    ax.set_ylabel("Pearson residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_predicted_probabilities(model_result, pred_df: pd.DataFrame, filename: str) -> None:
    pred_df = pred_df.copy()
    pred_df["pred_prob"] = model_result.predict(pred_df)

    # Plot lines by target_type within each condition
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["NB", "AB"]
    markers = {"BB": "o", "EM": "s"}
    for target_type, sub in pred_df.groupby("target_type"):
        y = []
        for cond in order:
            row = sub[sub["condition"] == cond]
            y.append(float(row["pred_prob"].iloc[0]))
        ax.plot(order, y, marker=markers.get(target_type, "o"), label=f"{target_type}")

    ax.set_title("Predicted P(correct) by condition and target type")
    ax.set_xlabel("condition")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="target_type")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    # ─────────────────────────────────────────────────────────────
    # 0) Load data
    # ─────────────────────────────────────────────────────────────
    trials = load_and_merge_trial_level()
    pp = load_participant_level()

    print("=" * 72)
    print("STRICT REGRESSION PIPELINE — BRSM MOVIE MEMORY")
    print("=" * 72)
    print(f"Vigilance exclusions applied: {sorted(VIGILANCE_EXCLUSIONS)}")
    print(f"Trial rows loaded: {len(trials):,}")
    print(f"Participants loaded: {len(pp):,}")
    print("\nCondition counts (trial-level):")
    print(trials["condition"].value_counts(dropna=False).to_string())

    # ─────────────────────────────────────────────────────────────
    # 1) Define candidate predictors (exhaustive + rationale)
    # ─────────────────────────────────────────────────────────────
    # Design variables (pre-response)
    design_predictors = ["condition", "target_type", "is_repeat"]
    # Pre-treatment participant covariates (optional controls)
    participant_covariates = ["age", "gender", "handedness", "vision"]
    # Explicit leakage exclusions for accuracy models
    leakage_exclusions = ["resp.rt", "conf_radio.response", "conf_radio.rt"]

    # ─────────────────────────────────────────────────────────────
    # 2) Clean and restrict columns (accuracy DV)
    # ─────────────────────────────────────────────────────────────
    selected = [
        "participant_id",
        "resp.corr",
        "movie_id",
        "movie_duration",
        "stimulus_id",
    ] + design_predictors + participant_covariates
    df = trials[selected].copy()

    # Type cleaning
    df["resp.corr"] = pd.to_numeric(df["resp.corr"], errors="coerce")
    df["resp.corr"] = df["resp.corr"].astype("Int64")
    df["is_repeat"] = pd.to_numeric(df["is_repeat"], errors="coerce")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["movie_duration"] = pd.to_numeric(df["movie_duration"], errors="coerce")

    # Drop unusable rows (strict)
    df = df.dropna(subset=["participant_id", "resp.corr", "condition", "target_type"])
    df = df[df["resp.corr"].isin([0, 1])].copy()
    df["resp.corr"] = df["resp.corr"].astype(int)
    # Keep movie_duration even if missing; models that use it will naturally drop NA rows.

    # `is_repeat` should be 0/1; if missing, we impute to 0 only when truly missing
    # (if the column was never recorded, it will be missing for all rows — we detect this).
    if df["is_repeat"].isna().mean() > 0.9:
        raise RuntimeError(
            "is_repeat is missing for most rows. "
            "Use trials_final_clean_with_repeat.csv (this script expects it)."
        )
    df["is_repeat"] = df["is_repeat"].fillna(0).astype(int)

    # Save cleaned modeling frame
    save_dataframe(df, "model_frame_trial_accuracy.csv")

    # ─────────────────────────────────────────────────────────────
    # 3) EDA (minimal, saved)
    # ─────────────────────────────────────────────────────────────
    plot_accuracy_bars(df)
    plot_numeric_distributions(df)
    plot_numeric_correlations(df)

    # ─────────────────────────────────────────────────────────────
    # 4) Collinearity diagnostic (VIF) on covariate set
    # ─────────────────────────────────────────────────────────────
    # Build a numeric design matrix using one-hot encoding for categoricals
    X_for_vif = df[[
        "condition",
        "target_type",
        "is_repeat",
        "movie_duration",
        "age",
        "gender",
        "handedness",
        "vision",
    ]].copy()
    X_for_vif = pd.get_dummies(
        X_for_vif,
        columns=["condition", "target_type", "gender", "handedness", "vision"],
        drop_first=True,
        dummy_na=False,
    )
    vif_table = compute_vif_from_design_matrix(X_for_vif)
    save_dataframe(vif_table, "vif_table.csv")
    print("\nSaved VIF table ->", OUT_DIR / "vif_table.csv")

    # Heteroscedasticity diagnostics (OLS surrogate; assignment-style)
    bp_diag = heteroscedasticity_diagnostics_ols(pp)

    # ─────────────────────────────────────────────────────────────
    # 5) Model 3 (participant-level baseline)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MODEL 3 — PARTICIPANT-LEVEL BASELINE")
    print("=" * 72)
    # Prefer a binomial GLM using (successes, failures) over OLS on proportions.
    pp_m3 = pp[["participant_id", "condition", "n_correct", "n_trials", "age", "gender"]].copy()
    pp_m3["n_correct"] = pd.to_numeric(pp_m3["n_correct"], errors="coerce")
    pp_m3["n_trials"] = pd.to_numeric(pp_m3["n_trials"], errors="coerce")
    pp_m3["age"] = pd.to_numeric(pp_m3["age"], errors="coerce")
    pp_m3 = pp_m3.dropna(subset=["n_correct", "n_trials", "condition"]).copy()
    pp_m3 = pp_m3.reset_index(drop=True)
    pp_m3["n_trials"] = pp_m3["n_trials"].astype(int)
    pp_m3["n_correct"] = pp_m3["n_correct"].astype(int)
    pp_m3["n_incorrect"] = pp_m3["n_trials"] - pp_m3["n_correct"]

    # Endog as 2-col for binomial GLM: [success, failure]
    endog = np.column_stack([pp_m3["n_correct"].values, pp_m3["n_incorrect"].values])
    exog_df = pd.get_dummies(pp_m3[["condition", "age", "gender"]], drop_first=True)
    exog_df = exog_df.replace([np.inf, -np.inf], np.nan)
    exog = sm.add_constant(exog_df, has_constant="add").astype(float)
    glm_m3 = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    print(glm_m3.summary())
    m3_or = pd.DataFrame(
        {
            "term": exog.columns,
            "coef": glm_m3.params,
            "p": glm_m3.pvalues,
            "OR": np.exp(glm_m3.params),
        }
    )
    save_dataframe(m3_or, "model3_participant_glm_odds_ratios.csv")

    # ─────────────────────────────────────────────────────────────
    # 6) Model 1 (trial-level GEE logistic), with systematic comparisons
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MODEL 1 — TRIAL-LEVEL CLUSTERED LOGISTIC REGRESSION (GEE)")
    print("=" * 72)

    # Pre-specified covariate sets (systematic; not ad-hoc)
    #   - core: design variables only
    #   - demo: core + demographics
    #   - stim: demo + stimulus duration
    #   - movieFE: stim + fixed effects for stimulus_id (stimulus difficulty)
    base_terms = (
        "C(condition, Treatment('NB')) + "
        "C(target_type, Treatment('BB')) + "
        "is_repeat"
    )
    demo_controls = " + age + C(gender, Treatment('Female')) + C(handedness) + C(vision)"
    stim_controls = demo_controls + " + movie_duration"
    movie_fe = stim_controls + " + C(stimulus_id)"

    specs: list[ModelSpec] = [
        ModelSpec(
            name="M1_core_main",
            formula=f"Q('resp.corr') ~ {base_terms}",
            description="Main effects; core design variables only.",
        ),
        ModelSpec(
            name="M1_core_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                "C(target_type, Treatment('BB')) + is_repeat"
            ),
            description="Adds condition × target_type interaction; core design variables only.",
        ),
        ModelSpec(
            name="M1_demo_main",
            formula=f"Q('resp.corr') ~ {base_terms}{demo_controls}",
            description="Main effects; design + demographics (age, gender, handedness, vision).",
        ),
        ModelSpec(
            name="M1_demo_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')) + is_repeat{demo_controls}"
            ),
            description="Adds interaction; design + demographics.",
        ),
        ModelSpec(
            name="M1_stim_main",
            formula=f"Q('resp.corr') ~ {base_terms}{stim_controls}",
            description="Main effects; + stimulus duration.",
        ),
        ModelSpec(
            name="M1_stim_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')) + is_repeat{stim_controls}"
            ),
            description="Adds interaction; + stimulus duration.",
        ),
        ModelSpec(
            name="M1_movieFE_main",
            formula=f"Q('resp.corr') ~ {base_terms}{movie_fe}",
            description="Main effects; + demographics + duration + movie fixed effects.",
        ),
        ModelSpec(
            name="M1_movieFE_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')) + is_repeat{movie_fe}"
            ),
            description="Adds interaction; + movie fixed effects.",
        ),
    ]

    fitted = []
    for spec in specs:
        print(f"\n--- Fitting {spec.name} ---")
        print(spec.description)
        res = fit_gee_binomial(spec.formula, df, group_col="participant_id")
        qic = try_get_qic(res)
        stable = is_gee_result_stable(res)
        fitted.append(
            {
                "name": spec.name,
                "description": spec.description,
                "qic": qic,
                "stable": stable,
                "result": res,
            }
        )
        print(res.summary())
        or_table = odds_ratio_table(res)
        or_table["stable"] = stable
        save_dataframe(or_table, f"{spec.name}_odds_ratios.csv")

    model_compare = pd.DataFrame(
        [
            {
                "model": f["name"],
                "qic": f["qic"],
                "stable": f["stable"],
                "description": f["description"],
            }
            for f in fitted
        ]
    ).sort_values(["stable", "qic", "model"], ascending=[False, True, True], na_position="last")
    save_dataframe(model_compare, "model1_model_comparison_qic.csv")
    print("\nSaved model comparison ->", OUT_DIR / "model1_model_comparison_qic.csv")

    # Pick a default "best" model (prefer stable + lowest QIC)
    best = None
    stable_compare = model_compare[model_compare["stable"] == True]
    if (stable_compare["qic"].notna().any()):
        best_name = stable_compare.iloc[0]["model"]
        best = next(f for f in fitted if f["name"] == best_name)
    elif model_compare["qic"].notna().any():
        # Fall back to best QIC even if unstable (still report)
        best_name = model_compare.iloc[0]["model"]
        best = next(f for f in fitted if f["name"] == best_name)
    else:
        best = next(f for f in fitted if f["name"] == "M1_core_main")

    print("\nBest model used for predictions:", best["name"])

    # Write a concise summary artifact (assignment-style)
    summary_path = OUT_DIR / "regression_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("STRICT REGRESSION PIPELINE — SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"Vigilance exclusions: {sorted(VIGILANCE_EXCLUSIONS)}\n")
        f.write(f"Trial rows (after cleaning for modeling): {len(df):,}\n")
        f.write("\nFeature policy (strict; assignment-style):\n")
        f.write("  Included (pre-response): condition, target_type, is_repeat, movie_duration, stimulus_id\n")
        f.write("  Optional controls (pre-treatment): age, gender, handedness, vision\n")
        f.write("  Excluded (leakage / post-response): response time, confidence, and response-derived fields\n")

        # Document observed leakage-like columns in this dataset (pattern-based)
        trial_cols = set(trials.columns)
        leakage_like = sorted(
            c
            for c in trial_cols
            if (
                c.startswith("resp.")
                or c.startswith("conf_")
                or c.startswith("conf_radio")
                or c.startswith("vigilance")
            )
            and c not in {"resp.corr"}
        )
        if leakage_like:
            f.write("  Observed excluded columns (pattern match): " + ", ".join(leakage_like) + "\n")

        f.write("\nDiagnostics written:\n")
        f.write("  - EDA: eda_numeric_distributions.png, eda_numeric_correlation_heatmap.png\n")
        f.write("  - Collinearity: vif_table.csv\n")
        f.write("  - Heteroscedasticity (OLS surrogate): diagnostic_breusch_pagan.csv, diagnostic_ols_residuals_vs_fitted.png\n")
        f.write("  - Logistic residual check: diagnostic_logistic_pearson_residuals.png\n")

        if bp_diag is not None and len(bp_diag) > 0:
            row = bp_diag.iloc[0].to_dict()
            f.write(
                "\nBreusch–Pagan (OLS surrogate) summary: "
                + f"LM p={row.get('bp_lm_p', np.nan):.4g}, F p={row.get('bp_f_p', np.nan):.4g}, R2={row.get('r2', np.nan):.4g}"\
                + "\n"
            )
        f.write("\nModel comparison (QIC; lower is better, when available):\n")
        f.write(model_compare.to_string(index=False) + "\n")
        f.write("\nBest model used for predictions: " + str(best["name"]) + "\n")
        f.write("\nOdds ratios (best model):\n")
        best_or = odds_ratio_table(best["result"]).round(6)
        f.write(best_or.to_string(index=False) + "\n")

    # ─────────────────────────────────────────────────────────────
    # 7) Prediction grid + plot
    # ─────────────────────────────────────────────────────────────
    # Hold numeric covariates at mean (age) and categorical at baseline.
    age_mean = float(pd.to_numeric(df["age"], errors="coerce").dropna().mean())

    def _mode_or_default(series: pd.Series, default: str) -> str:
        s = series.dropna().astype(str)
        if len(s) == 0:
            return default
        return str(s.mode().iloc[0])

    handedness_baseline = _mode_or_default(df["handedness"], "Right handed")
    vision_baseline = _mode_or_default(df["vision"], "Normal")
    duration_mean = float(pd.to_numeric(df["movie_duration"], errors="coerce").dropna().mean())
    pred_grid = pd.DataFrame(
        {
            "condition": ["NB", "NB", "AB", "AB"],
            "target_type": ["BB", "EM", "BB", "EM"],
            "is_repeat": [0, 0, 0, 0],
            "age": [age_mean] * 4,
            "gender": ["Female"] * 4,
            # For extended models, include baseline categories as needed
            "handedness": [handedness_baseline] * 4,
            "vision": [vision_baseline] * 4,
            "movie_duration": [duration_mean] * 4,
            "movie_id": [1, 1, 1, 1],
            "stimulus_id": ["NB_1", "NB_1", "AB_1", "AB_1"],
        }
    )
    save_dataframe(pred_grid, "prediction_grid.csv")
    plot_predicted_probabilities(best["result"], pred_grid, "predicted_probabilities.png")

    # Save a logistic residual diagnostic for the best model
    plot_logistic_residuals(best["result"], df, "diagnostic_logistic_pearson_residuals.png")

    print("\nArtifacts written to:")
    print(" -", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

