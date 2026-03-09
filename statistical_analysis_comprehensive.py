#!/usr/bin/env python3
"""
BRSM Movie Memory - Comprehensive Statistical Analysis
NON-PARAMETRIC TESTS (Data is not normally distributed)

Includes:
- Mann-Whitney U tests for between-group comparisons
- Effect sizes (rank-biserial correlation, Cliff's Delta)
- Multiple comparisons corrections (Bonferroni, Holm, FDR)
- Post-hoc tests
- Confidence intervals via bootstrapping
- Comprehensive reporting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, spearmanr, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "final_cleaned_data"
OUTPUT_DIR = BASE_DIR / "statistical_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("BRSM MOVIE MEMORY - COMPREHENSIVE STATISTICAL ANALYSIS")
print("NON-PARAMETRIC TESTS (Data violates normality assumptions)")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df_participants = pd.read_csv(INPUT_DIR / "participants_final_clean.csv")
df_trials = pd.read_csv(INPUT_DIR / "trials_final_clean.csv")

# Extract frame type from target_img
df_trials['frame_type'] = df_trials['target_img'].str.extract(r'_(BB|EM)_T\.png')[0]

# Extract movie ID
df_trials['movie_id'] = df_trials['target_img'].str.extract(r'Vid(\d+)_')[0].astype(float)

# Add stimulus category more explicitly
df_trials['is_target'] = df_trials['target_img'].str.contains('_T.png').astype(int)

print(f"   Participants: {len(df_participants)}")
print(f"   Trials: {len(df_trials)}")
print(f"   Frame types found: {df_trials['frame_type'].value_counts().to_dict()}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def mann_whitney_with_effect_size(group1, group2, alternative='two-sided'):
    """
    Perform Mann-Whitney U test with comprehensive statistics.
    
    Returns:
        dict with U-stat, p-value, z-score, effect size (rank-biserial r), 
        medians, and interpretation
    """
    n1, n2 = len(group1), len(group2)
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    
    # Z-score approximation
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0
    
    # Rank-biserial correlation (effect size)
    # r = 1 - (2U)/(n1*n2)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    
    # Cliff's Delta (alternative effect size)
    cliffs_delta = calculate_cliffs_delta(group1, group2)
    
    # Descriptive statistics
    median1, median2 = np.median(group1), np.median(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    # Effect size interpretation
    abs_r = abs(rank_biserial)
    if abs_r < 0.1:
        effect_interp = "negligible"
    elif abs_r < 0.3:
        effect_interp = "small"
    elif abs_r < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    return {
        'U_statistic': u_stat,
        'p_value': p_value,
        'z_score': z_score,
        'rank_biserial_r': rank_biserial,
        'cliffs_delta': cliffs_delta,
        'n1': n1,
        'n2': n2,
        'median1': median1,
        'median2': median2,
        'mean1': mean1,
        'mean2': mean2,
        'effect_size_interpretation': effect_interp,
        'significant': p_value < 0.05
    }


def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's Delta effect size.
    
    Cliff's Delta = (# pairs where x > y - # pairs where x < y) / (n1 * n2)
    
    Interpretation:
      |d| < 0.147: negligible
      |d| < 0.33: small
      |d| < 0.474: medium
      |d| >= 0.474: large
    """
    n1, n2 = len(group1), len(group2)
    
    # Count pairs
    greater = sum(x > y for x in group1 for y in group2)
    less = sum(x < y for x in group1 for y in group2)
    
    delta = (greater - less) / (n1 * n2)
    
    return delta


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """
    Calculate bootstrap confidence interval for median.
    """
    bootstrapped_medians = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_medians.append(np.median(sample))
    
    alpha = 1 - ci
    lower = np.percentile(bootstrapped_medians, alpha/2 * 100)
    upper = np.percentile(bootstrapped_medians, (1 - alpha/2) * 100)
    
    return lower, upper


def apply_multiple_comparison_correction(p_values, method='holm', alpha=0.05):
    """
    Apply multiple comparison correction.
    
    Methods:
      - 'bonferroni': Most conservative
      - 'holm': Sequentially rejective Bonferroni
      - 'fdr_bh': Benjamini-Hochberg FDR control (recommended)
    """
    reject, p_corrected, alphacSidak, alphacBonf = multipletests(
        p_values, alpha=alpha, method=method
    )
    
    return {
        'reject': reject,
        'p_corrected': p_corrected,
        'method': method,
        'alpha': alpha
    }


def format_p_value(p):
    """Format p-value for reporting."""
    if p < 0.001:
        return "p < .001"
    else:
        return f"p = {p:.3f}"


def report_mann_whitney_result(result, group1_name, group2_name, dv_name, hypothesis_direction="two-sided"):
    """
    Generate APA-style report for Mann-Whitney U test.
    """
    report = f"\n{'='*70}\n"
    report += f"Mann-Whitney U Test: {dv_name}\n"
    report += f"Comparing: {group1_name} vs {group2_name}\n"
    report += f"{'='*70}\n\n"
    
    report += f"Sample Sizes: n₁ = {result['n1']}, n₂ = {result['n2']}\n\n"
    
    report += f"Descriptive Statistics:\n"
    report += f"  {group1_name}:\n"
    report += f"    Median = {result['median1']:.3f}\n"
    report += f"    Mean = {result['mean1']:.3f}\n"
    report += f"  {group2_name}:\n"
    report += f"    Median = {result['median2']:.3f}\n"
    report += f"    Mean = {result['mean2']:.3f}\n\n"
    
    report += f"Test Statistics:\n"
    report += f"  U = {result['U_statistic']:.2f}\n"
    report += f"  Z = {result['z_score']:.3f}\n"
    report += f"  {format_p_value(result['p_value'])}\n\n"
    
    report += f"Effect Sizes:\n"
    report += f"  Rank-biserial r = {result['rank_biserial_r']:.3f} ({result['effect_size_interpretation']})\n"
    report += f"  Cliff's Delta = {result['cliffs_delta']:.3f}\n\n"
    
    if result['significant']:
        direction = "higher" if result['median2'] > result['median1'] else "lower"
        report += f"✓ SIGNIFICANT: {group2_name} shows {direction} {dv_name} than {group1_name}\n"
    else:
        report += f"✗ NOT SIGNIFICANT: No reliable difference detected\n"
    
    report += "\nInterpretation:\n"
    if result['significant']:
        if result['median2'] > result['median1']:
            report += f"  {group2_name} (Mdn = {result['median2']:.3f}) had significantly higher\n"
            report += f"  {dv_name} than {group1_name} (Mdn = {result['median1']:.3f}),\n"
        else:
            report += f"  {group1_name} (Mdn = {result['median1']:.3f}) had significantly higher\n"
            report += f"  {dv_name} than {group2_name} (Mdn = {result['median2']:.3f}),\n"
        report += f"  U = {result['U_statistic']:.0f}, {format_p_value(result['p_value'])}, r = {result['rank_biserial_r']:.3f}.\n"
        report += f"  This represents a {result['effect_size_interpretation']} effect.\n"
    else:
        report += f"  No significant difference was found between {group1_name} and {group2_name}\n"
        report += f"  on {dv_name}, {format_p_value(result['p_value'])}.\n"
    
    return report


# ============================================================================
# SPLIT DATA BY CONDITION
# ============================================================================
print("\n[2] Splitting data by condition...")
ab_participants = df_participants[df_participants['condition'] == 'AB']
nb_participants = df_participants[df_participants['condition'] == 'NB']

ab_trials = df_trials[df_trials['condition'] == 'AB']
nb_trials = df_trials[df_trials['condition'] == 'NB']

print(f"   AB: {len(ab_participants)} participants, {len(ab_trials)} trials")
print(f"   NB: {len(nb_participants)} participants, {len(nb_trials)} trials")

# ============================================================================
# HYPOTHESIS 1: OVERALL RECOGNITION ACCURACY
# ============================================================================
print("\n[3] HYPOTHESIS 1: Overall Recognition Accuracy")
print("   H0: No difference between AB and NB")
print("   H1: NB shows higher accuracy than AB")

h1_result = mann_whitney_with_effect_size(
    ab_participants['accuracy'].dropna(),
    nb_participants['accuracy'].dropna(),
    alternative='two-sided'
)

# Bootstrap confidence intervals
ab_ci = bootstrap_ci(ab_participants['accuracy'].dropna())
nb_ci = bootstrap_ci(nb_participants['accuracy'].dropna())

h1_report = report_mann_whitney_result(
    h1_result, "AB", "NB", "Recognition Accuracy"
)

print(h1_report)
print(f"95% Bootstrap CI for AB: [{ab_ci[0]:.3f}, {ab_ci[1]:.3f}]")
print(f"95% Bootstrap CI for NB: [{nb_ci[0]:.3f}, {nb_ci[1]:.3f}]")

# ============================================================================
# HYPOTHESIS 2: BB-FRAME RECOGNITION ACCURACY
# ============================================================================
print("\n[4] HYPOTHESIS 2: BB-Frame Recognition Accuracy")
print("   H0: No difference in BB-frame accuracy between AB and NB")
print("   H1: NB shows higher BB-frame accuracy than AB")

# Calculate BB-frame accuracy for each participant
ab_bb_trials = ab_trials[ab_trials['frame_type'] == 'BB']
nb_bb_trials = nb_trials[nb_trials['frame_type'] == 'BB']

ab_bb_acc = ab_bb_trials.groupby('participant_id')['resp.corr'].mean()
nb_bb_acc = nb_bb_trials.groupby('participant_id')['resp.corr'].mean()

print(f"   AB participants with BB trials: {len(ab_bb_acc)}")
print(f"   NB participants with BB trials: {len(nb_bb_acc)}")

h2_result = mann_whitney_with_effect_size(
    ab_bb_acc.values,
    nb_bb_acc.values,
    alternative='two-sided'
)

h2_report = report_mann_whitney_result(
    h2_result, "AB", "NB", "BB-Frame Accuracy"
)

print(h2_report)

# ============================================================================
# HYPOTHESIS 3: OVERALL CONFIDENCE RATINGS
# ============================================================================
print("\n[5] HYPOTHESIS 3: Overall Confidence Ratings")
print("   H0: No difference in confidence between AB and NB")
print("   H1: NB shows higher confidence than AB")

h3_result = mann_whitney_with_effect_size(
    ab_participants['confidence_mean'].dropna(),
    nb_participants['confidence_mean'].dropna(),
    alternative='two-sided'
)

h3_report = report_mann_whitney_result(
    h3_result, "AB", "NB", "Mean Confidence Rating"
)

print(h3_report)

# ============================================================================
# HYPOTHESIS 4: FRAME-SPECIFIC EFFECTS (CRITICAL TEST)
# ============================================================================
print("\n[6] HYPOTHESIS 4: Frame-Specific Effects (BB vs EM)")
print("   Testing for Condition × Frame Type interaction")
print("   Approach: Separate tests for BB and EM, then compare effect sizes")

# BB frames
ab_bb_conf = ab_bb_trials.groupby('participant_id')['conf_radio.response'].mean()
nb_bb_conf = nb_bb_trials.groupby('participant_id')['conf_radio.response'].mean()

h4_bb_result = mann_whitney_with_effect_size(
    ab_bb_conf.values,
    nb_bb_conf.values,
    alternative='two-sided'
)

print("\n--- BB Frames (Boundary frames) ---")
print(report_mann_whitney_result(h4_bb_result, "AB", "NB", "BB-Frame Confidence"))

# EM frames
ab_em_trials = ab_trials[ab_trials['frame_type'] == 'EM']
nb_em_trials = nb_trials[nb_trials['frame_type'] == 'EM']

ab_em_conf = ab_em_trials.groupby('participant_id')['conf_radio.response'].mean()
nb_em_conf = nb_em_trials.groupby('participant_id')['conf_radio.response'].mean()

h4_em_result = mann_whitney_with_effect_size(
    ab_em_conf.values,
    nb_em_conf.values,
    alternative='two-sided'
)

print("\n--- EM Frames (Mid-event frames) ---")
print(report_mann_whitney_result(h4_em_result, "AB", "NB", "EM-Frame Confidence"))

# Compare effect sizes
print("\n--- Interaction Assessment ---")
print(f"BB-Frame effect size (r): {h4_bb_result['rank_biserial_r']:.3f}")
print(f"EM-Frame effect size (r): {h4_em_result['rank_biserial_r']:.3f}")
print(f"Difference in effect sizes: {abs(h4_bb_result['rank_biserial_r'] - h4_em_result['rank_biserial_r']):.3f}")

if abs(h4_bb_result['rank_biserial_r']) > abs(h4_em_result['rank_biserial_r']) * 1.5:
    print("✓ EVIDENCE FOR INTERACTION: BB effect substantially larger than EM effect")
else:
    print("✗ NO STRONG INTERACTION: Effects are similar across frame types")

# ============================================================================
# HYPOTHESIS 5: ACCURACY VARIABILITY ACROSS MOVIES
# ============================================================================
print("\n[7] HYPOTHESIS 5: Accuracy Variability Across Movies")
print("   H0: No difference in accuracy variability between AB and NB")
print("   H1: AB shows higher variability than NB")

# Calculate SD of accuracy across movies for each participant
def calculate_movie_level_variability(df_trials_subset, participant_col='participant_id'):
    """Calculate within-participant SD across movies."""
    variability = []
    participant_ids = []
    
    for pid, group in df_trials_subset.groupby(participant_col):
        # Calculate accuracy for each movie
        movie_acc = group.groupby('movie_id')['resp.corr'].mean()
        
        if len(movie_acc) >= 2:  # Need at least 2 movies to calculate SD
            variability.append(movie_acc.std())
            participant_ids.append(pid)
    
    return pd.Series(variability, index=participant_ids)

ab_variability = calculate_movie_level_variability(ab_trials)
nb_variability = calculate_movie_level_variability(nb_trials)

print(f"   Participants with movie-level data: AB={len(ab_variability)}, NB={len(nb_variability)}")

h5_result = mann_whitney_with_effect_size(
    ab_variability.values,
    nb_variability.values,
    alternative='two-sided'
)

h5_report = report_mann_whitney_result(
    h5_result, "AB", "NB", "Accuracy Variability (SD across movies)"
)

print(h5_report)

# Additionally: Test variance equality with Levene's test
from scipy.stats import levene
levene_stat, levene_p = levene(ab_variability.values, nb_variability.values)
print(f"\nLevene's Test for Equality of Variances:")
print(f"  W = {levene_stat:.3f}, {format_p_value(levene_p)}")
if levene_p < 0.05:
    print("  ✓ SIGNIFICANT: Variances differ between groups")
else:
    print("  ✗ NOT SIGNIFICANT: Variances are similar")

# ============================================================================
# ADDITIONAL ANALYSIS: RESPONSE TIME
# ============================================================================
print("\n[8] ADDITIONAL ANALYSIS: Response Time")
print("   Exploratory test (use corrected alpha if needed)")

h6_result = mann_whitney_with_effect_size(
    ab_participants['rt_mean'].dropna(),
    nb_participants['rt_mean'].dropna(),
    alternative='two-sided'
)

h6_report = report_mann_whitney_result(
    h6_result, "AB", "NB", "Mean Response Time"
)

print(h6_report)

# ============================================================================
# MULTIPLE COMPARISONS CORRECTION
# ============================================================================
print("\n[9] MULTIPLE COMPARISONS CORRECTION")
print("="*80)

# Collect all p-values from primary hypotheses
primary_tests = {
    'H1: Overall Accuracy': h1_result['p_value'],
    'H2: BB-Frame Accuracy': h2_result['p_value'],
    'H3: Overall Confidence': h3_result['p_value'],
    'H4a: BB-Frame Confidence': h4_bb_result['p_value'],
    'H4b: EM-Frame Confidence': h4_em_result['p_value'],
    'H5: Accuracy Variability': h5_result['p_value']
}

test_names = list(primary_tests.keys())
p_values = list(primary_tests.values())

# Apply different correction methods
corrections = {}
for method in ['bonferroni', 'holm', 'fdr_bh']:
    corrections[method] = apply_multiple_comparison_correction(p_values, method=method)

# Create summary table
print("\nPrimary Hypotheses (6 tests) - Multiple Comparisons Correction:")
print(f"{'Test':<30} {'Uncorrected':<12} {'Bonferroni':<12} {'Holm':<12} {'FDR (BH)':<12}")
print("-" * 80)

for i, test_name in enumerate(test_names):
    uncorr_sig = "✓" if p_values[i] < 0.05 else "✗"
    bonf_sig = "✓" if corrections['bonferroni']['reject'][i] else "✗"
    holm_sig = "✓" if corrections['holm']['reject'][i] else "✗"
    fdr_sig = "✓" if corrections['fdr_bh']['reject'][i] else "✗"
    
    print(f"{test_name:<30} {p_values[i]:<6.4f} {uncorr_sig:<6} "
          f"{corrections['bonferroni']['p_corrected'][i]:<6.4f} {bonf_sig:<6} "
          f"{corrections['holm']['p_corrected'][i]:<6.4f} {holm_sig:<6} "
          f"{corrections['fdr_bh']['p_corrected'][i]:<6.4f} {fdr_sig:<6}")

print("\n" + "="*80)
print("RECOMMENDED APPROACH: Use Holm-Bonferroni or FDR (Benjamini-Hochberg)")
print("  - Holm: Better balance of power and control (recommended for confirmatory)")
print("  - FDR: Good for exploratory analyses with many tests")
print("  - Bonferroni: Most conservative (may be too strict)")
print("="*80)

# ============================================================================
# SIGNAL DETECTION THEORY ANALYSIS
# ============================================================================
print("\n[10] SIGNAL DETECTION THEORY ANALYSIS")
print("="*80)

def calculate_sdt_measures(df_trials_subset):
    """
    Calculate SDT measures for each participant.
    
    Assumes:
      - 'R' response for targets = Hit
      - 'R' response for lures = False Alarm
    """
    results = []
    
    for pid, group in df_trials_subset.groupby('participant_id'):
        # Identify targets and lures
        targets = group[group['is_target'] == 1]
        lures = group[group['is_target'] == 0]
        
        # Calculate hit rate and false alarm rate
        hits = (targets['resp.corr'] == 1).sum()
        n_targets = len(targets)
        
        false_alarms = (lures['resp.corr'] == 0).sum()
        n_lures = len(lures)
        
        hit_rate = hits / n_targets if n_targets > 0 else 0
        fa_rate = false_alarms / n_lures if n_lures > 0 else 0
        
        # Adjust extreme values (loglinear correction)
        hit_rate = max(0.01, min(0.99, hit_rate))
        fa_rate = max(0.01, min(0.99, fa_rate))
        
        # Calculate d' (sensitivity) and c (criterion)
        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
        c = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate))
        
        results.append({
            'participant_id': pid,
            'hit_rate': hit_rate,
            'fa_rate': fa_rate,
            'd_prime': d_prime,
            'criterion_c': c
        })
    
    return pd.DataFrame(results)

print("\nCalculating SDT measures...")
ab_sdt = calculate_sdt_measures(ab_trials)
nb_sdt = calculate_sdt_measures(nb_trials)

print(f"   AB: {len(ab_sdt)} participants")
print(f"   NB: {len(nb_sdt)} participants")

# Test d' (sensitivity)
sdt_dprime_result = mann_whitney_with_effect_size(
    ab_sdt['d_prime'].values,
    nb_sdt['d_prime'].values,
    alternative='two-sided'
)

print("\n--- Sensitivity (d') ---")
print(report_mann_whitney_result(sdt_dprime_result, "AB", "NB", "Sensitivity (d')"))

# Test c (criterion)
sdt_criterion_result = mann_whitney_with_effect_size(
    ab_sdt['criterion_c'].values,
    nb_sdt['criterion_c'].values,
    alternative='two-sided'
)

print("\n--- Response Criterion (c) ---")
print(report_mann_whitney_result(sdt_criterion_result, "AB", "NB", "Response Criterion (c)"))
print("\nNote: Positive c = conservative bias (prefer 'lure'), Negative c = liberal bias (prefer 'target')")

# ============================================================================
# CONFIDENCE-ACCURACY CALIBRATION
# ============================================================================
print("\n[11] CONFIDENCE-ACCURACY CALIBRATION")
print("="*80)

def calculate_gamma_correlation(df_trials_subset):
    """
    Calculate Goodman-Kruskal gamma correlation between confidence and accuracy.
    
    Gamma is appropriate for ordinal data and ranges from -1 to 1.
    """
    from scipy.stats import kendalltau
    
    gammas = []
    participant_ids = []
    
    for pid, group in df_trials_subset.groupby('participant_id'):
        if len(group) > 1 and group['conf_radio.response'].nunique() > 1:
            # Use Kendall's tau as approximation to gamma
            tau, _ = kendalltau(group['conf_radio.response'], group['resp.corr'])
            gammas.append(tau)
            participant_ids.append(pid)
    
    return pd.Series(gammas, index=participant_ids)

print("\nCalculating confidence-accuracy calibration (Kendall's tau)...")
ab_gamma = calculate_gamma_correlation(ab_trials)
nb_gamma = calculate_gamma_correlation(nb_trials)

print(f"   Valid participants: AB={len(ab_gamma)}, NB={len(nb_gamma)}")

calibration_result = mann_whitney_with_effect_size(
    ab_gamma.values,
    nb_gamma.values,
    alternative='two-sided'
)

print("\n--- Confidence-Accuracy Relationship ---")
print(report_mann_whitney_result(calibration_result, "AB", "NB", "Confidence-Accuracy Calibration (τ)"))

# ============================================================================
# SAVE ALL RESULTS
# ============================================================================
print("\n[12] Saving results to files...")

# Create comprehensive results summary
results_summary = pd.DataFrame({
    'Hypothesis': [
        'H1: Overall Accuracy',
        'H2: BB-Frame Accuracy',
        'H3: Overall Confidence',
        'H4a: BB-Frame Confidence',
        'H4b: EM-Frame Confidence',
        'H5: Accuracy Variability',
        'Additional: Response Time',
        'SDT: Sensitivity (d\')',
        'SDT: Criterion (c)',
        'Calibration: Conf-Acc'
    ],
    'Test': ['Mann-Whitney U'] * 10,
    'AB_Median': [
        h1_result['median1'], h2_result['median1'], h3_result['median1'],
        h4_bb_result['median1'], h4_em_result['median1'], h5_result['median1'],
        h6_result['median1'], sdt_dprime_result['median1'], 
        sdt_criterion_result['median1'], calibration_result['median1']
    ],
    'NB_Median': [
        h1_result['median2'], h2_result['median2'], h3_result['median2'],
        h4_bb_result['median2'], h4_em_result['median2'], h5_result['median2'],
        h6_result['median2'], sdt_dprime_result['median2'],
        sdt_criterion_result['median2'], calibration_result['median2']
    ],
    'U_statistic': [
        h1_result['U_statistic'], h2_result['U_statistic'], h3_result['U_statistic'],
        h4_bb_result['U_statistic'], h4_em_result['U_statistic'], h5_result['U_statistic'],
        h6_result['U_statistic'], sdt_dprime_result['U_statistic'],
        sdt_criterion_result['U_statistic'], calibration_result['U_statistic']
    ],
    'z_score': [
        h1_result['z_score'], h2_result['z_score'], h3_result['z_score'],
        h4_bb_result['z_score'], h4_em_result['z_score'], h5_result['z_score'],
        h6_result['z_score'], sdt_dprime_result['z_score'],
        sdt_criterion_result['z_score'], calibration_result['z_score']
    ],
    'p_value': [
        h1_result['p_value'], h2_result['p_value'], h3_result['p_value'],
        h4_bb_result['p_value'], h4_em_result['p_value'], h5_result['p_value'],
        h6_result['p_value'], sdt_dprime_result['p_value'],
        sdt_criterion_result['p_value'], calibration_result['p_value']
    ],
    'rank_biserial_r': [
        h1_result['rank_biserial_r'], h2_result['rank_biserial_r'], h3_result['rank_biserial_r'],
        h4_bb_result['rank_biserial_r'], h4_em_result['rank_biserial_r'], h5_result['rank_biserial_r'],
        h6_result['rank_biserial_r'], sdt_dprime_result['rank_biserial_r'],
        sdt_criterion_result['rank_biserial_r'], calibration_result['rank_biserial_r']
    ],
    'cliffs_delta': [
        h1_result['cliffs_delta'], h2_result['cliffs_delta'], h3_result['cliffs_delta'],
        h4_bb_result['cliffs_delta'], h4_em_result['cliffs_delta'], h5_result['cliffs_delta'],
        h6_result['cliffs_delta'], sdt_dprime_result['cliffs_delta'],
        sdt_criterion_result['cliffs_delta'], calibration_result['cliffs_delta']
    ],
    'effect_size': [
        h1_result['effect_size_interpretation'], h2_result['effect_size_interpretation'], 
        h3_result['effect_size_interpretation'], h4_bb_result['effect_size_interpretation'],
        h4_em_result['effect_size_interpretation'], h5_result['effect_size_interpretation'],
        h6_result['effect_size_interpretation'], sdt_dprime_result['effect_size_interpretation'],
        sdt_criterion_result['effect_size_interpretation'], calibration_result['effect_size_interpretation']
    ],
    'significant_uncorrected': [r['significant'] for r in [
        h1_result, h2_result, h3_result, h4_bb_result, h4_em_result, h5_result,
        h6_result, sdt_dprime_result, sdt_criterion_result, calibration_result
    ]]
})

# Add corrected p-values (using Holm for first 6)
primary_p_corrected = corrections['holm']['p_corrected']
results_summary['p_value_holm_corrected'] = list(primary_p_corrected) + [np.nan] * 4
results_summary['significant_holm'] = list(corrections['holm']['reject']) + [np.nan] * 4

# Save
results_summary.to_csv(OUTPUT_DIR / 'statistical_results_summary.csv', index=False)
print(f"   ✓ Saved: statistical_results_summary.csv")

# Create comprehensive text report
with open(OUTPUT_DIR / 'statistical_analysis_full_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BRSM MOVIE MEMORY - COMPLETE STATISTICAL ANALYSIS REPORT\n")
    f.write("NON-PARAMETRIC TESTS (Data violates normality assumptions)\n")
    f.write("="*80 + "\n\n")
    
    f.write(h1_report + "\n\n")
    f.write(h2_report + "\n\n")
    f.write(h3_report + "\n\n")
    f.write("HYPOTHESIS 4: Frame-Specific Effects\n")
    f.write(report_mann_whitney_result(h4_bb_result, "AB", "NB", "BB-Frame Confidence") + "\n\n")
    f.write(report_mann_whitney_result(h4_em_result, "AB", "NB", "EM-Frame Confidence") + "\n\n")
    f.write(h5_report + "\n\n")
    f.write(h6_report + "\n\n")
    f.write(report_mann_whitney_result(sdt_dprime_result, "AB", "NB", "Sensitivity (d')") + "\n\n")
    f.write(report_mann_whitney_result(sdt_criterion_result, "AB", "NB", "Response Criterion (c)") + "\n\n")
    f.write(report_mann_whitney_result(calibration_result, "AB", "NB", "Calibration") + "\n\n")

print(f"   ✓ Saved: statistical_analysis_full_report.txt")

# ============================================================================
# CREATE VISUALIZATION OF RESULTS
# ============================================================================
print("\n[13] Creating results visualization...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Statistical Results Summary - Key Findings', fontsize=16, fontweight='bold')

# Plot 1: Overall Accuracy
axes[0, 0].bar(['AB', 'NB'], [h1_result['median1'], h1_result['median2']], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy (Median)', fontweight='bold')
axes[0, 0].set_title(f'H1: Overall Accuracy\n{format_p_value(h1_result["p_value"])}, r={h1_result["rank_biserial_r"]:.3f}')
axes[0, 0].set_ylim(0, 1)
if h1_result['significant']:
    axes[0, 0].text(0.5, 0.95, '✓ SIGNIFICANT', ha='center', va='top', 
                    transform=axes[0, 0].transAxes, fontsize=10, color='green', fontweight='bold')

# Plot 2: BB-Frame Accuracy
axes[0, 1].bar(['AB', 'NB'], [h2_result['median1'], h2_result['median2']], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('BB-Frame Accuracy (Median)', fontweight='bold')
axes[0, 1].set_title(f'H2: BB-Frame Accuracy\n{format_p_value(h2_result["p_value"])}, r={h2_result["rank_biserial_r"]:.3f}')
axes[0, 1].set_ylim(0, 1)
if h2_result['significant']:
    axes[0, 1].text(0.5, 0.95, '✓ SIGNIFICANT', ha='center', va='top',
                    transform=axes[0, 1].transAxes, fontsize=10, color='green', fontweight='bold')

# Plot 3: Overall Confidence
axes[0, 2].bar(['AB', 'NB'], [h3_result['median1'], h3_result['median2']], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[0, 2].set_ylabel('Confidence (Median)', fontweight='bold')
axes[0, 2].set_title(f'H3: Overall Confidence\n{format_p_value(h3_result["p_value"])}, r={h3_result["rank_biserial_r"]:.3f}')
axes[0, 2].set_ylim(1, 5)
if h3_result['significant']:
    axes[0, 2].text(0.5, 0.95, '✓ SIGNIFICANT', ha='center', va='top',
                    transform=axes[0, 2].transAxes, fontsize=10, color='green', fontweight='bold')

# Plot 4: Frame-Specific Effects
frame_types = ['BB Frames', 'EM Frames']
ab_values = [h4_bb_result['median1'], h4_em_result['median1']]
nb_values = [h4_bb_result['median2'], h4_em_result['median2']]

x = np.arange(len(frame_types))
width = 0.35

axes[1, 0].bar(x - width/2, ab_values, width, label='AB', color='#FF6B6B', alpha=0.7, edgecolor='black')
axes[1, 0].bar(x + width/2, nb_values, width, label='NB', color='#4ECDC4', alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Confidence (Median)', fontweight='bold')
axes[1, 0].set_title(f'H4: Frame-Specific Effects')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(frame_types)
axes[1, 0].legend()
axes[1, 0].set_ylim(1, 5)

# Plot 5: Variability
axes[1, 1].bar(['AB', 'NB'], [h5_result['median1'], h5_result['median2']], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Accuracy SD (Median)', fontweight='bold')
axes[1, 1].set_title(f'H5: Accuracy Variability\n{format_p_value(h5_result["p_value"])}, r={h5_result["rank_biserial_r"]:.3f}')
if h5_result['significant']:
    axes[1, 1].text(0.5, 0.95, '✓ SIGNIFICANT', ha='center', va='top',
                    transform=axes[1, 1].transAxes, fontsize=10, color='green', fontweight='bold')

# Plot 6: Sensitivity (d')
axes[1, 2].bar(['AB', 'NB'], [sdt_dprime_result['median1'], sdt_dprime_result['median2']], 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
axes[1, 2].set_ylabel('Sensitivity (d\')', fontweight='bold')
axes[1, 2].set_title(f'SDT: Sensitivity\n{format_p_value(sdt_dprime_result["p_value"])}, r={sdt_dprime_result["rank_biserial_r"]:.3f}')
if sdt_dprime_result['significant']:
    axes[1, 2].text(0.5, 0.95, '✓ SIGNIFICANT', ha='center', va='top',
                    transform=axes[1, 2].transAxes, fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'statistical_results_visualization.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: statistical_results_visualization.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFiles saved to: {OUTPUT_DIR}/")
print("  - statistical_results_summary.csv (table of all results)")
print("  - statistical_analysis_full_report.txt (complete text report)")
print("  - statistical_results_visualization.png (key findings plot)")
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS:")
print("="*80)

significant_count = sum([
    h1_result['significant'], h2_result['significant'], h3_result['significant'],
    h4_bb_result['significant'], h4_em_result['significant'], h5_result['significant']
])

print(f"\nPrimary Hypotheses: {significant_count}/6 significant (uncorrected α = .05)")
print(f"After Holm correction: Check 'significant_holm' column in CSV")

print("\nEffect Sizes (using uncorrected tests):")
for name, result in zip(
    ['H1: Overall Accuracy', 'H2: BB-Frame Accuracy', 'H3: Confidence', 
     'H4a: BB Confidence', 'H4b: EM Confidence', 'H5: Variability'],
    [h1_result, h2_result, h3_result, h4_bb_result, h4_em_result, h5_result]
):
    sig_marker = "✓" if result['significant'] else "✗"
    print(f"  {sig_marker} {name}: r = {result['rank_biserial_r']:.3f} ({result['effect_size_interpretation']})")

print("\n" + "="*80)
print("NEXT STEPS:")
print("  1. Review the full report text file for APA-style reporting")
print("  2. Check effect sizes and decide on practical significance")
print("  3. Consider which correction method to use (Holm recommended)")
print("  4. Create publication-quality figures from the visualization script")
print("  5. Interpret findings in context of your theoretical framework")
print("="*80)
