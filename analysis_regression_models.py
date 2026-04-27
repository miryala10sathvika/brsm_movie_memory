"""Regression models for the BRSM Movie Memory experiment.

Repeated-video exclusion
------------------------
Five movies per condition (IDs 3, 7, 18, 28, 37) served as vigilance checks
and were shown twice during encoding.  All trials with is_repeat == 1 are
excluded before modelling.  Because no repeated-video trials remain, is_repeat
is not a predictor in any model (it would be a zero-variance constant).

Models
------
  Model 3  — participant-level binomial GLM (aggregated successes/failures)
  Model 1  — trial-level GEE logistic regression (clustered by participant)
  GLMM     — trial-level binomial mixed model with random participant intercept
              fitted by Variational Bayes (statsmodels BinomialBayesMixedGLM)

Interaction effects — interpretation guide
-------------------------------------------
An interaction term in a logistic model means that the effect of one predictor
on the log-odds of a correct response changes depending on the level of another
predictor.  Concretely, the term

    C(condition)[T.AB] : C(target_type)[T.EM]

captures whether the difference in log-odds between EM and BB targets is the
same in the AB condition as in the NB (reference) condition.

  - If the coefficient is POSITIVE:  the EM advantage over BB is *larger* in AB
    than in NB (or equivalently, the AB penalty is smaller for EM targets than
    for BB targets).
  - If the coefficient is NEGATIVE:  the EM advantage is *smaller* (or the BB
    advantage is larger) in AB relative to NB.
  - If NOT significant: there is no evidence that condition moderates the
    BB-vs-EM accuracy difference.

To recover simple (marginal) effects from an interaction model:
  - Effect of condition for BB trials  = main effect of condition alone
  - Effect of condition for EM trials  = main effect + interaction coefficient
  - Effect of frame type in NB         = main effect of target_type alone
  - Effect of frame type in AB         = main effect of target_type + interaction

Odds ratios for interaction terms should always be interpreted as the *ratio of
odds ratios* (how much the simple OR for one factor changes at each level of the
other factor), not as a standalone probability.

Feature policy
--------------
  Included (pre-response):  condition, target_type, movie_duration
  Optional controls:        age, gender, handedness, vision
  Excluded (leakage):       resp.rt, conf_radio.response, conf_radio.rt
  Excluded (zero-variance after filtering): is_repeat
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
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


DATA_DIR = Path("final_cleaned_data")
OUT_DIR = Path("statistical_results") / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIGILANCE_EXCLUSIONS = {"sub105_AB", "sub70_NB"}
REPEATED_VIDEO_IDS = {3, 7, 18, 28, 37}


def _target_type(target_img: str) -> str | float:
    s = str(target_img)
    if "_BB_" in s:
        return "BB"
    if "_EM_" in s:
        return "EM"
    return np.nan


def load_and_merge_trial_level() -> pd.DataFrame:
    trials = pd.read_csv(DATA_DIR / "trials_final_clean_with_repeat.csv")
    pp = pd.read_csv(DATA_DIR / "participants_final_clean_with_vigilance.csv")

    pp = pp[~pp["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()
    trials = trials[~trials["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()

    trials["is_repeat"] = pd.to_numeric(trials["is_repeat"], errors="coerce").fillna(0).astype(int)
    n_before = len(trials)
    trials = trials[trials["is_repeat"] == 0].copy()
    print(f"  Excluded {n_before - len(trials)} repeated-video trials; {len(trials)} retained.")

    trials["resp.corr"] = pd.to_numeric(trials["resp.corr"], errors="coerce")
    trials["target_type"] = trials["target_img"].apply(_target_type)

    covars = pp[["participant_id", "age", "gender", "handedness", "vision"]].copy()
    merged = trials.merge(covars, on="participant_id", how="left")
    merged = merge_stimulus_metadata(merged)
    return merged


def merge_stimulus_metadata(trials: pd.DataFrame) -> pd.DataFrame:
    def _make_movie_table(csv_path: Path, condition: str) -> pd.DataFrame:
        m = pd.read_csv(csv_path)
        m = m.dropna(subset=["path", "duration"]).copy()
        m["path"] = m["path"].astype(str)
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


def load_participant_level() -> pd.DataFrame:
    pp = pd.read_csv(DATA_DIR / "participants_final_clean_with_vigilance.csv")
    pp = pp[~pp["participant_id"].isin(VIGILANCE_EXCLUSIONS)].copy()
    return pp


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUT_DIR / filename, index=False)


def compute_vif_from_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    X = sm.add_constant(X, has_constant="add")
    values = X.to_numpy(dtype=float)
    return pd.DataFrame(
        [{"variable": col, "VIF": float(variance_inflation_factor(values, i))}
         for i, col in enumerate(X.columns)]
    ).sort_values("VIF", ascending=False)


def odds_ratio_table(result) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame({
        "term": result.params.index,
        "coef": result.params.values,
        "se": result.bse.values,
        "z_or_t": result.tvalues.values,
        "p": result.pvalues.values,
        "OR": np.exp(result.params.values),
        "OR_ci_low": np.exp(ci[0].values),
        "OR_ci_high": np.exp(ci[1].values),
    })


@dataclass(frozen=True)
class ModelSpec:
    name: str
    formula: str
    description: str


def fit_gee_binomial(formula: str, data: pd.DataFrame, group_col: str = "participant_id"):
    model = smf.gee(
        formula,
        groups=data[group_col],
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(scale=1.0)


def fit_glmm_vb(formula: str, data: pd.DataFrame) -> tuple:
    """Fit a binomial GLMM with a random participant intercept using Variational Bayes.

    Returns (result, summary_dict).

    The model is:
        logit P(correct) = fixed effects + u_i
    where u_i ~ N(0, σ²) is a participant-level random intercept.

    Variational Bayes (fit_vb) is used instead of Laplace (fit_map) because
    it consistently estimates the random-effect variance (σ) whereas the Laplace
    approximation can collapse σ to zero in balanced designs.

    The posterior SD column for fixed effects (M-type rows) plays the role of
    the standard error; the V-type row reports the log-SD of the random effect.
    To obtain σ: σ = exp(posterior_mean of the V parameter).
    """
    vc_formulas = {"participant": "0 + C(participant_id)"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data=data)
    result = model.fit_vb()
    summary = result.summary()

    # Extract fixed effects using the correct VB result attributes
    fe_mean = result.fe_mean            # posterior means for fixed effects
    fe_sd = result.fe_sd                # posterior SDs for fixed effects
    names = model.fep_names             # fixed-effect parameter names

    # RE variance: vcp_mean is the log-SD of the random intercept
    re_log_sd = float(result.vcp_mean[0])
    re_sd = float(np.exp(re_log_sd))

    fe_df = pd.DataFrame({
        "term": names,
        "post_mean": fe_mean,
        "post_sd": fe_sd,
        "OR": np.exp(fe_mean),
        "z_approx": fe_mean / np.where(fe_sd > 0, fe_sd, np.nan),
    })
    fe_df["p_approx"] = 2 * (1 - pd.Series(
        [float(__import__("scipy").stats.norm.cdf(abs(z))) for z in fe_df["z_approx"]]
    ))

    summary_dict = {
        "fixed_effects": fe_df.to_dict(orient="records"),
        "random_intercept_log_sd": re_log_sd,
        "random_intercept_sd": re_sd,
        "n_participants": int(data["participant_id"].nunique()),
        "n_trials": int(len(data)),
        "summary_text": str(summary),
    }
    return result, summary_dict, fe_df


def is_gee_result_stable(result) -> bool:
    params = np.asarray(result.params)
    bse = np.asarray(getattr(result, "bse", np.full_like(params, np.nan)))
    return bool(np.isfinite(params).all() and np.isfinite(bse).all())


def try_get_qic(result) -> float | None:
    qic = getattr(result, "qic", None)
    if qic is None:
        return None
    try:
        if callable(qic):
            try:
                return float(qic(scale=1.0)[0])
            except TypeError:
                return float(qic()[0])
        return float(qic[0])
    except Exception:
        return None


def plot_accuracy_bars(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["condition", "target_type"])["resp.corr"]
        .mean()
        .reset_index(name="mean_acc")
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, sub in agg.groupby("condition"):
        ax.plot(sub["target_type"], sub["mean_acc"], marker="o", label=cond)
    ax.set_title("Trial accuracy by condition and target type\n(repeated movies excluded)")
    ax.set_xlabel("target_type")
    ax.set_ylabel("mean(resp.corr)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="condition")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_accuracy_by_condition_target_type.png", dpi=160)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["age", "movie_duration"]
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
    numeric_cols = ["age", "movie_duration"]
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
    d = pp[["accuracy", "condition", "age", "gender"]].copy()
    d["accuracy"] = pd.to_numeric(d["accuracy"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d = d.dropna(subset=["accuracy", "condition"]).copy()

    model = smf.ols("accuracy ~ C(condition, Treatment('NB')) + age + C(gender)", data=d).fit()
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)

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

    out = pd.DataFrame([{
        "model": "OLS_surrogate_accuracy",
        "bp_lm": float(lm_stat),
        "bp_lm_p": float(lm_p),
        "bp_f": float(f_stat),
        "bp_f_p": float(f_p),
        "n": int(len(d)),
        "r2": float(model.rsquared),
    }])
    save_dataframe(out, "diagnostic_breusch_pagan.csv")
    return out


def plot_logistic_residuals(result, df: pd.DataFrame, filename: str) -> None:
    fitted = result.predict(df)
    y = df["resp.corr"].astype(float).values
    var = np.clip(fitted * (1 - fitted), 1e-6, None)
    pearson = (y - fitted) / np.sqrt(var)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fitted, pearson, alpha=0.25, s=10, color="#4c78a8")
    ax.axhline(0, color="red", lw=1)
    ax.set_title("Model 1 GEE diagnostic: Pearson residuals vs fitted")
    ax.set_xlabel("fitted P(correct)")
    ax.set_ylabel("Pearson residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_predicted_probabilities(model_result, pred_df: pd.DataFrame, filename: str) -> None:
    pred_df = pred_df.copy()
    pred_df["pred_prob"] = model_result.predict(pred_df)
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["NB", "AB"]
    markers = {"BB": "o", "EM": "s"}
    for target_type, sub in pred_df.groupby("target_type"):
        y = []
        for cond in order:
            row = sub[sub["condition"] == cond]
            y.append(float(row["pred_prob"].iloc[0]))
        ax.plot(order, y, marker=markers.get(target_type, "o"), label=f"{target_type}")
    ax.set_title("Predicted P(correct) by condition and target type\n(repeated movies excluded)")
    ax.set_xlabel("condition")
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(title="target_type")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_glmm_forest(fe_df: pd.DataFrame, filename: str) -> None:
    """Forest plot of GLMM fixed-effect odds ratios."""
    terms = fe_df[fe_df["term"] != "Intercept"].copy()
    terms["OR"] = np.exp(terms["post_mean"])
    terms["OR_lo"] = np.exp(terms["post_mean"] - 1.96 * terms["post_sd"])
    terms["OR_hi"] = np.exp(terms["post_mean"] + 1.96 * terms["post_sd"])

    fig, ax = plt.subplots(figsize=(8, max(3, len(terms) * 0.6 + 1)))
    y_pos = range(len(terms))
    ax.errorbar(
        terms["OR"], y_pos,
        xerr=[terms["OR"] - terms["OR_lo"], terms["OR_hi"] - terms["OR"]],
        fmt="o", color="#2c7bb6", capsize=4, ms=6,
    )
    ax.axvline(1.0, color="red", lw=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(terms["term"].tolist(), fontsize=9)
    ax.set_xlabel("Odds Ratio (approx 95% CI from posterior SD × 1.96)", fontsize=10)
    ax.set_title("GLMM (Variational Bayes) — Fixed-effect Odds Ratios\n(repeated movies excluded)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=160)
    plt.close(fig)


def main() -> None:
    # ── 0) Load data ──────────────────────────────────────────────────
    trials = load_and_merge_trial_level()
    pp = load_participant_level()

    print("=" * 72)
    print("REGRESSION PIPELINE — BRSM MOVIE MEMORY (repeated movies excluded)")
    print("=" * 72)
    print(f"Vigilance exclusions: {sorted(VIGILANCE_EXCLUSIONS)}")
    print(f"Trial rows after filtering: {len(trials):,}")
    print(f"Participants: {len(pp):,}")
    print("\nCondition counts (trial-level):")
    print(trials["condition"].value_counts(dropna=False).to_string())

    # ── 1) Build modelling frame ──────────────────────────────────────
    selected = [
        "participant_id", "resp.corr", "movie_id", "movie_duration", "stimulus_id",
        "condition", "target_type", "age", "gender", "handedness", "vision",
    ]
    df = trials[selected].copy()
    df["resp.corr"] = pd.to_numeric(df["resp.corr"], errors="coerce")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["movie_duration"] = pd.to_numeric(df["movie_duration"], errors="coerce")
    df = df.dropna(subset=["participant_id", "resp.corr", "condition", "target_type"])
    df = df[df["resp.corr"].isin([0, 1])].copy()
    df["resp.corr"] = df["resp.corr"].astype(int)

    # Column name for formula compatibility
    df["resp_corr"] = df["resp.corr"]

    save_dataframe(df, "model_frame_trial_accuracy.csv")

    # ── 2) EDA ────────────────────────────────────────────────────────
    plot_accuracy_bars(df)
    plot_numeric_distributions(df)
    plot_numeric_correlations(df)

    # ── 3) VIF ────────────────────────────────────────────────────────
    X_for_vif = df[["condition", "target_type", "movie_duration", "age",
                     "gender", "handedness", "vision"]].copy()
    X_for_vif = pd.get_dummies(
        X_for_vif,
        columns=["condition", "target_type", "gender", "handedness", "vision"],
        drop_first=True, dummy_na=False,
    )
    vif_table = compute_vif_from_design_matrix(X_for_vif)
    save_dataframe(vif_table, "vif_table.csv")
    print("\nVIF table saved.")

    bp_diag = heteroscedasticity_diagnostics_ols(pp)

    # ── 4) Model 3 — participant-level GLM ────────────────────────────
    print("\n" + "=" * 72)
    print("MODEL 3 — PARTICIPANT-LEVEL BASELINE (binomial GLM)")
    print("=" * 72)
    pp_m3 = pp[["participant_id", "condition", "n_correct", "n_trials", "age", "gender"]].copy()
    for col in ["n_correct", "n_trials", "age"]:
        pp_m3[col] = pd.to_numeric(pp_m3[col], errors="coerce")
    pp_m3 = pp_m3.dropna(subset=["n_correct", "n_trials", "condition"]).reset_index(drop=True)
    pp_m3["n_trials"] = pp_m3["n_trials"].astype(int)
    pp_m3["n_correct"] = pp_m3["n_correct"].astype(int)
    pp_m3["n_incorrect"] = pp_m3["n_trials"] - pp_m3["n_correct"]

    endog = np.column_stack([pp_m3["n_correct"].values, pp_m3["n_incorrect"].values])
    exog_df = pd.get_dummies(pp_m3[["condition", "age", "gender"]], drop_first=True)
    exog = sm.add_constant(exog_df.replace([np.inf, -np.inf], np.nan), has_constant="add").astype(float)
    glm_m3 = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    print(glm_m3.summary())

    m3_or = pd.DataFrame({
        "term": exog.columns,
        "coef": glm_m3.params,
        "p": glm_m3.pvalues,
        "OR": np.exp(glm_m3.params),
    })
    save_dataframe(m3_or, "model3_participant_glm_odds_ratios.csv")

    # ── 5) Model 1 — GEE logistic ────────────────────────────────────
    print("\n" + "=" * 72)
    print("MODEL 1 — TRIAL-LEVEL GEE LOGISTIC (clustered by participant)")
    print("=" * 72)

    base_terms = (
        "C(condition, Treatment('NB')) + "
        "C(target_type, Treatment('BB'))"
    )
    demo_controls = " + age + C(gender, Treatment('Female')) + C(handedness) + C(vision)"
    stim_controls = demo_controls + " + movie_duration"
    movie_fe = stim_controls + " + C(stimulus_id)"

    specs: list[ModelSpec] = [
        ModelSpec(
            name="M1_core_main",
            formula=f"Q('resp.corr') ~ {base_terms}",
            description="Main effects: condition + target_type.",
        ),
        ModelSpec(
            name="M1_core_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                "C(target_type, Treatment('BB'))"
            ),
            description=(
                "Condition × target_type interaction. "
                "The interaction term tests whether the BB-vs-EM accuracy difference "
                "is moderated by condition (NB vs AB)."
            ),
        ),
        ModelSpec(
            name="M1_demo_main",
            formula=f"Q('resp.corr') ~ {base_terms}{demo_controls}",
            description="Main effects + demographics (age, gender, handedness, vision).",
        ),
        ModelSpec(
            name="M1_demo_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')){demo_controls}"
            ),
            description="Interaction + demographics.",
        ),
        ModelSpec(
            name="M1_stim_main",
            formula=f"Q('resp.corr') ~ {base_terms}{stim_controls}",
            description="Main effects + demographics + stimulus duration.",
        ),
        ModelSpec(
            name="M1_stim_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')){stim_controls}"
            ),
            description="Interaction + demographics + stimulus duration.",
        ),
        ModelSpec(
            name="M1_movieFE_main",
            formula=f"Q('resp.corr') ~ {base_terms}{movie_fe}",
            description="Main effects + demographics + duration + movie fixed effects.",
        ),
        ModelSpec(
            name="M1_movieFE_interaction",
            formula=(
                "Q('resp.corr') ~ C(condition, Treatment('NB')) * "
                f"C(target_type, Treatment('BB')){movie_fe}"
            ),
            description="Interaction + movie fixed effects.",
        ),
    ]

    fitted = []
    for spec in specs:
        print(f"\n--- Fitting {spec.name} ---")
        print(spec.description)
        res = fit_gee_binomial(spec.formula, df, group_col="participant_id")
        qic = try_get_qic(res)
        stable = is_gee_result_stable(res)
        fitted.append({
            "name": spec.name,
            "description": spec.description,
            "qic": qic,
            "stable": stable,
            "result": res,
        })
        print(res.summary())
        or_table = odds_ratio_table(res)
        or_table["stable"] = stable
        save_dataframe(or_table, f"{spec.name}_odds_ratios.csv")

    model_compare = pd.DataFrame([
        {"model": f["name"], "qic": f["qic"], "stable": f["stable"],
         "description": f["description"]}
        for f in fitted
    ]).sort_values(["stable", "qic", "model"], ascending=[False, True, True], na_position="last")
    save_dataframe(model_compare, "model1_model_comparison_qic.csv")
    print("\nModel comparison saved.")

    stable_compare = model_compare[model_compare["stable"]]
    if stable_compare["qic"].notna().any():
        best_name = stable_compare.iloc[0]["model"]
    elif model_compare["qic"].notna().any():
        best_name = model_compare.iloc[0]["model"]
    else:
        best_name = "M1_core_main"
    best = next(f for f in fitted if f["name"] == best_name)
    print(f"\nBest GEE model: {best_name}")

    # ── 6) GLMM — Variational Bayes ──────────────────────────────────
    print("\n" + "=" * 72)
    print("GLMM — BINOMIAL MIXED MODEL (Variational Bayes, random participant intercept)")
    print("=" * 72)

    # Core GLMM (main effects)
    print("\n--- GLMM core (main effects) ---")
    glmm_core_formula = (
        "resp_corr ~ C(condition, Treatment('NB')) + C(target_type, Treatment('BB'))"
    )
    _, glmm_core_summary, glmm_core_fe = fit_glmm_vb(glmm_core_formula, df)
    save_dataframe(glmm_core_fe, "glmm_core_fixed_effects.csv")
    print(glmm_core_summary["summary_text"])
    print(f"  Random intercept SD: {glmm_core_summary['random_intercept_sd']:.4f}")

    # GLMM with interaction
    print("\n--- GLMM interaction (condition × target_type) ---")
    glmm_int_formula = (
        "resp_corr ~ C(condition, Treatment('NB')) * C(target_type, Treatment('BB'))"
    )
    _, glmm_int_summary, glmm_int_fe = fit_glmm_vb(glmm_int_formula, df)
    save_dataframe(glmm_int_fe, "glmm_interaction_fixed_effects.csv")
    print(glmm_int_summary["summary_text"])
    print(f"  Random intercept SD: {glmm_int_summary['random_intercept_sd']:.4f}")

    # GLMM with demographics + stimulus duration
    print("\n--- GLMM full (+ age, gender, movie_duration) ---")
    df_glmm_full = df.dropna(subset=["age", "gender", "movie_duration"]).copy()
    glmm_full_formula = (
        "resp_corr ~ C(condition, Treatment('NB')) + C(target_type, Treatment('BB')) "
        "+ age + C(gender, Treatment('Female')) + movie_duration"
    )
    _, glmm_full_summary, glmm_full_fe = fit_glmm_vb(glmm_full_formula, df_glmm_full)
    save_dataframe(glmm_full_fe, "glmm_full_fixed_effects.csv")
    print(glmm_full_summary["summary_text"])
    print(f"  Random intercept SD: {glmm_full_summary['random_intercept_sd']:.4f}")

    # Forest plot for the full GLMM
    plot_glmm_forest(glmm_full_fe, "glmm_full_forest_plot.png")

    # Consolidated GLMM summary
    glmm_results = {
        "core": glmm_core_summary,
        "interaction": glmm_int_summary,
        "full": glmm_full_summary,
    }

    # ── 7) Prediction grid + residuals for best GEE ──────────────────
    age_mean = float(pd.to_numeric(df["age"], errors="coerce").dropna().mean())

    def _mode_or_default(series: pd.Series, default: str) -> str:
        s = series.dropna().astype(str)
        return str(s.mode().iloc[0]) if len(s) else default

    handedness_baseline = _mode_or_default(df["handedness"], "Right handed")
    vision_baseline = _mode_or_default(df["vision"], "Normal")
    duration_mean = float(pd.to_numeric(df["movie_duration"], errors="coerce").dropna().mean())

    pred_grid = pd.DataFrame({
        "condition": ["NB", "NB", "AB", "AB"],
        "target_type": ["BB", "EM", "BB", "EM"],
        "age": [age_mean] * 4,
        "gender": ["Female"] * 4,
        "handedness": [handedness_baseline] * 4,
        "vision": [vision_baseline] * 4,
        "movie_duration": [duration_mean] * 4,
        "movie_id": [1, 1, 1, 1],
        "stimulus_id": ["NB_1", "NB_1", "AB_1", "AB_1"],
    })
    save_dataframe(pred_grid, "prediction_grid.csv")
    plot_predicted_probabilities(best["result"], pred_grid, "predicted_probabilities.png")
    plot_logistic_residuals(best["result"], df, "diagnostic_logistic_pearson_residuals.png")

    # ── 8) Write summary text ─────────────────────────────────────────
    summary_path = OUT_DIR / "regression_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("REGRESSION PIPELINE — SUMMARY (repeated movies excluded)\n")
        f.write("=" * 72 + "\n")
        f.write(f"Vigilance exclusions: {sorted(VIGILANCE_EXCLUSIONS)}\n")
        f.write(f"Repeated movie IDs excluded: {sorted(REPEATED_VIDEO_IDS)}\n")
        f.write(f"Trial rows (after all filtering): {len(df):,}\n")
        f.write("\nFeature policy:\n")
        f.write("  Included: condition, target_type, movie_duration, age, gender, handedness, vision\n")
        f.write("  Excluded (leakage): resp.rt, conf_radio.response\n")
        f.write("  Excluded (zero-variance after filtering): is_repeat\n")

        bp = bp_diag.iloc[0].to_dict() if bp_diag is not None and len(bp_diag) > 0 else {}
        f.write(f"\nBreusch-Pagan (OLS surrogate): LM p={bp.get('bp_lm_p', 'n/a'):.4g}, "
                f"F p={bp.get('bp_f_p', 'n/a'):.4g}, R2={bp.get('r2', 'n/a'):.4g}\n")

        f.write("\n--- GEE Model Comparison (QIC) ---\n")
        f.write(model_compare.to_string(index=False) + "\n")
        f.write(f"\nBest GEE model: {best_name}\n")
        f.write("\nBest GEE odds ratios:\n")
        f.write(odds_ratio_table(best["result"]).round(4).to_string(index=False) + "\n")

        f.write("\n--- GLMM Core (VB) ---\n")
        for row in glmm_core_summary["fixed_effects"]:
            f.write(f"  {row['term']}: post_mean={row['post_mean']:.4f}, "
                    f"post_sd={row['post_sd']:.4f}, OR={row['OR']:.4f}\n")
        f.write(f"  Random intercept SD: {glmm_core_summary['random_intercept_sd']:.4f}\n")

        f.write("\n--- GLMM Interaction (VB) ---\n")
        for row in glmm_int_summary["fixed_effects"]:
            f.write(f"  {row['term']}: post_mean={row['post_mean']:.4f}, "
                    f"post_sd={row['post_sd']:.4f}, OR={row['OR']:.4f}\n")
        f.write(f"  Random intercept SD: {glmm_int_summary['random_intercept_sd']:.4f}\n")

        f.write("\n--- GLMM Full (VB) ---\n")
        for row in glmm_full_summary["fixed_effects"]:
            f.write(f"  {row['term']}: post_mean={row['post_mean']:.4f}, "
                    f"post_sd={row['post_sd']:.4f}, OR={row['OR']:.4f}\n")
        f.write(f"  Random intercept SD: {glmm_full_summary['random_intercept_sd']:.4f}\n")

    # Write results synthesis
    synthesis_path = OUT_DIR / "results_synthesis.txt"
    with synthesis_path.open("w", encoding="utf-8") as f:
        f.write("RESULTS SYNTHESIS (repeated movies excluded)\n")
        f.write("=" * 72 + "\n\n")
        f.write("### Model 3 — Participant-level binomial GLM\n\n")
        for term, coef, pval, oratio in zip(
            m3_or["term"], m3_or["coef"], m3_or["p"], m3_or["OR"]
        ):
            if term == "const":
                continue
            f.write(f"  {term}: coef={coef:.4f}, p={pval:.4g}, OR={oratio:.4f}\n")

        f.write("\n### Model 1 — Best GEE model: " + best_name + "\n\n")
        best_or_df = odds_ratio_table(best["result"])
        for _, row in best_or_df.iterrows():
            if row["term"] == "Intercept":
                continue
            f.write(f"  {row['term']}: OR={row['OR']:.4f}, p={row['p']:.4g}, "
                    f"95% CI [{row['OR_ci_low']:.4f}, {row['OR_ci_high']:.4f}]\n")

        f.write("\n### GLMM (Variational Bayes) — Core model\n\n")
        f.write("Random participant intercept SD: "
                f"{glmm_core_summary['random_intercept_sd']:.4f}\n")
        for row in glmm_core_summary["fixed_effects"]:
            if "Intercept" in row["term"]:
                continue
            f.write(f"  {row['term']}: OR={row['OR']:.4f}, post_mean={row['post_mean']:.4f}, "
                    f"post_SD={row['post_sd']:.4f}\n")

        f.write("\n### GLMM Interaction note\n\n")
        f.write(
            "The interaction term C(condition)[T.AB]:C(target_type)[T.EM] captures\n"
            "whether the EM-vs-BB accuracy difference is the same in both conditions.\n"
            "  OR > 1: EM advantage is LARGER in AB than NB\n"
            "  OR < 1: EM advantage is SMALLER in AB (or BB advantage is larger in AB)\n"
            "  Simple effects:\n"
            "    Effect of condition for BB trials = main effect of condition only\n"
            "    Effect of condition for EM trials = main effect + interaction coefficient\n"
        )
        for row in glmm_int_summary["fixed_effects"]:
            f.write(f"  {row['term']}: OR={row['OR']:.4f}\n")
        f.write(f"  Random intercept SD: {glmm_int_summary['random_intercept_sd']:.4f}\n")

    print(f"\nArtifacts written to: {OUT_DIR}")
    print("Done.")

    return glmm_results


if __name__ == "__main__":
    main()
