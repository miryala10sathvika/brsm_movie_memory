#!/usr/bin/env python3
"""
BRSM Movie Memory - NON-PARAMETRIC Statistical Analysis
Purpose: Comprehensive hypothesis testing with non-normal data
Uses: Mann-Whitney U, Kruskal-Wallis, Friedman tests with effect sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare, spearmanr
from scipy.stats import levene, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "final_cleaned_data"
OUTPUT_DIR = BASE_DIR / "statistical_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("BRSM MOVIE MEMORY - NON-PARAMETRIC STATISTICAL ANALYSIS")
print("="*80)
print("\nNote: Using non-parametric tests because normality assumption violated")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
df_participants = pd.read_csv(INPUT_DIR / "participants_final_clean.csv")
df_trials = pd.read_csv(INPUT_DIR / "trials_final_clean.csv")

print(f"   Loaded: {len(df_participants)} participants, {len(df_trials)} trials")

# Separate by condition
ab_participants = df_participants[df_participants['condition'] == 'AB']
nb_participants = df_participants[df_participants['condition'] == 'NB']

ab_trials = df_trials[df_trials['condition'] == 'AB']
nb_trials = df_trials[df_trials['condition'] == 'NB']

print(f"   AB: {len(ab_participants)} participants, {len(ab_trials)} trials")
print(f"   NB: {len(nb_participants)} participants, {len(nb_trials)} trials")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_rank_biserial(u_stat, n1, n2):
    """
    Calculate rank-biserial correlation (effect size for Mann-Whitney U)
    Range: -1 to 1 (similar to Cohen's d interpretation)
    """
    r = 1 - (2*u_stat) / (n1 * n2)
    return r

def interpret_rank_biserial(r):
    """Interpret rank-biserial correlation"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"

def cohens_d(group1, group2):
    """Calculate Cohen's d for comparison"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def epsilon_squared(h_stat, n, k):
    """
    Calculate epsilon-squared (effect size for Kruskal-Wallis)
    ε² = H / (n - 1)
    """
    return h_stat / (n - 1)

def mann_whitney_with_stats(group1, group2, alternative='two-sided'):
    """Perform Mann-Whitney U test with comprehensive statistics"""
    n1, n2 = len(group1), len(group2)
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    
    # Effect size (rank-biserial correlation)
    r = calculate_rank_biserial(u_stat, n1, n2)
    
    # Also calculate Cohen's d for comparison
    d = cohens_d(group1, group2)
    
    # Descriptive stats
    stats_dict = {
        'n1': n1,
        'n2': n2,
        'median1': np.median(group1),
        'median2': np.median(group2),
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'U': u_stat,
        'p': p_value,
        'rank_biserial_r': r,
        'cohens_d': d,
        'effect_size': interpret_rank_biserial(r)
    }
    
    return stats_dict

def apply_multiple_comparison_correction(p_values, method='fdr_bh'):
    """
    Apply multiple comparison correction
    Methods: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
    """
    from statsmodels.stats.multitest import multipletests
    
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method)
    return rejected, p_corrected

# ============================================================================
# EXTRACT FRAME TYPE FROM TRIAL DATA
# ============================================================================
print("\n[2] Extracting frame type information...")

def extract_frame_type(img_path):
    """Extract BB (Before Boundary) or EM (Event Middle) from filename"""
    if pd.isna(img_path):
        return np.nan
    if '_BB_' in str(img_path):
        return 'BB'
    elif '_EM_' in str(img_path):
        return 'EM'
    else:
        return 'Unknown'

df_trials['frame_type'] = df_trials['target_img'].apply(extract_frame_type)

# Count frame types
frame_counts = df_trials['frame_type'].value_counts()
print(f"   Frame types identified: {dict(frame_counts)}")

# Remove unknown frame types if any
df_trials = df_trials[df_trials['frame_type'].isin(['BB', 'EM'])].copy()
print(f"   Trials after filtering: {len(df_trials)}")

# ============================================================================
# HYPOTHESIS 1: OVERALL RECOGNITION ACCURACY
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: Overall Recognition Accuracy")
print("="*80)
print("H0: No difference in accuracy between NB and AB groups")
print("H1: NB participants show higher accuracy than AB participants")
print("-"*80)

h1_results = mann_whitney_with_stats(
    nb_participants['accuracy'].dropna(),
    ab_participants['accuracy'].dropna(),
    alternative='greater'  # One-tailed: NB > AB
)

print(f"\nDescriptive Statistics:")
print(f"  NB: n={h1_results['n2']}, Median={h1_results['median2']:.4f}, Mean={h1_results['mean2']:.4f}")
print(f"  AB: n={h1_results['n1']}, Median={h1_results['median1']:.4f}, Mean={h1_results['mean1']:.4f}")
print(f"  Difference: {h1_results['median2'] - h1_results['median1']:.4f} (3.4%)")

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic = {h1_results['U']:.2f}")
print(f"  p-value = {h1_results['p']:.4f}")

print(f"\nEffect Sizes:")
print(f"  Rank-biserial r = {h1_results['rank_biserial_r']:.3f} ({h1_results['effect_size']})")
print(f"  Cohen's d = {h1_results['cohens_d']:.3f} (for comparison)")

if h1_results['p'] < 0.05:
    print(f"\n✓ SIGNIFICANT: NB group shows higher accuracy (p < 0.05)")
else:
    print(f"\n✗ NOT SIGNIFICANT: No difference in accuracy (p ≥ 0.05)")

# ============================================================================
# HYPOTHESIS 2: BB-FRAME RECOGNITION ACCURACY
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: BB-Frame Recognition Accuracy")
print("="*80)
print("H0: No difference in BB-frame accuracy between groups")
print("H1: NB participants show higher BB-frame accuracy")
print("-"*80)

# Calculate BB-frame accuracy for each participant
bb_trials = df_trials[df_trials['frame_type'] == 'BB'].copy()
bb_accuracy = bb_trials.groupby('participant_id')['resp.corr'].mean()

# Merge with condition
participant_bb = df_participants[['participant_id', 'condition']].merge(
    bb_accuracy.reset_index(), on='participant_id', how='left'
)
participant_bb.columns = ['participant_id', 'condition', 'bb_accuracy']

bb_nb = participant_bb[participant_bb['condition'] == 'NB']['bb_accuracy'].dropna()
bb_ab = participant_bb[participant_bb['condition'] == 'AB']['bb_accuracy'].dropna()

h2_results = mann_whitney_with_stats(bb_nb, bb_ab, alternative='greater')

print(f"\nDescriptive Statistics (BB frames only):")
print(f"  NB: n={h2_results['n1']}, Median={h2_results['median1']:.4f}, Mean={h2_results['mean1']:.4f}")
print(f"  AB: n={h2_results['n2']}, Median={h2_results['median2']:.4f}, Mean={h2_results['mean2']:.4f}")

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic = {h2_results['U']:.2f}")
print(f"  p-value = {h2_results['p']:.4f}")

print(f"\nEffect Sizes:")
print(f"  Rank-biserial r = {h2_results['rank_biserial_r']:.3f} ({h2_results['effect_size']})")

if h2_results['p'] < 0.05:
    print(f"\n✓ SIGNIFICANT: NB group shows higher BB-frame accuracy")
else:
    print(f"\n✗ NOT SIGNIFICANT: No difference in BB-frame accuracy")

# ============================================================================
# HYPOTHESIS 3: OVERALL CONFIDENCE RATINGS
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: Overall Confidence Ratings")
print("="*80)
print("H0: No difference in confidence between groups")
print("H1: NB participants report higher confidence")
print("-"*80)

h3_results = mann_whitney_with_stats(
    nb_participants['confidence_mean'].dropna(),
    ab_participants['confidence_mean'].dropna(),
    alternative='greater'
)

print(f"\nDescriptive Statistics:")
print(f"  NB: n={h3_results['n1']}, Median={h3_results['median1']:.4f}, Mean={h3_results['mean1']:.4f}")
print(f"  AB: n={h3_results['n2']}, Median={h3_results['median2']:.4f}, Mean={h3_results['mean2']:.4f}")

print(f"\nMann-Whitney U Test:")
print(f"  U-statistic = {h3_results['U']:.2f}")
print(f"  p-value = {h3_results['p']:.4f}")

print(f"\nEffect Sizes:")
print(f"  Rank-biserial r = {h3_results['rank_biserial_r']:.3f} ({h3_results['effect_size']})")

if h3_results['p'] < 0.05:
    print(f"\n✓ SIGNIFICANT: NB group shows higher confidence")
else:
    print(f"\n✗ NOT SIGNIFICANT: No difference in confidence")

# ============================================================================
# HYPOTHESIS 4: FRAME-SPECIFIC EFFECTS (BB vs EM × Condition)
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: Frame-Specific Effects (CRITICAL TEST)")
print("="*80)
print("H0: No interaction between frame type and condition")
print("H1: Condition effect is larger for BB frames than EM frames")
print("-"*80)

# Calculate accuracy by frame type for each participant
frame_accuracy = df_trials.groupby(['participant_id', 'frame_type'])['resp.corr'].mean().unstack()
frame_accuracy = frame_accuracy.reset_index()

# Merge with condition
frame_analysis = df_participants[['participant_id', 'condition']].merge(
    frame_accuracy, on='participant_id', how='left'
)

print(f"\nDescriptive Statistics by Frame Type:")
for condition in ['AB', 'NB']:
    data = frame_analysis[frame_analysis['condition'] == condition]
    print(f"\n  {condition} Condition:")
    print(f"    BB frames: Mean={data['BB'].mean():.4f}, Median={data['BB'].median():.4f}")
    print(f"    EM frames: Mean={data['EM'].mean():.4f}, Median={data['EM'].median():.4f}")

# Test 1: BB frames - compare conditions
print(f"\n--- BB Frames: NB vs AB ---")
bb_nb = frame_analysis[frame_analysis['condition'] == 'NB']['BB'].dropna()
bb_ab = frame_analysis[frame_analysis['condition'] == 'AB']['BB'].dropna()
h4_bb = mann_whitney_with_stats(bb_nb, bb_ab, alternative='greater')

print(f"  NB: Median={h4_bb['median1']:.4f}, n={h4_bb['n1']}")
print(f"  AB: Median={h4_bb['median2']:.4f}, n={h4_bb['n2']}")
print(f"  U = {h4_bb['U']:.2f}, p = {h4_bb['p']:.4f}, r = {h4_bb['rank_biserial_r']:.3f}")

# Test 2: EM frames - compare conditions
print(f"\n--- EM Frames: NB vs AB ---")
em_nb = frame_analysis[frame_analysis['condition'] == 'NB']['EM'].dropna()
em_ab = frame_analysis[frame_analysis['condition'] == 'AB']['EM'].dropna()
h4_em = mann_whitney_with_stats(em_nb, em_ab, alternative='greater')

print(f"  NB: Median={h4_em['median1']:.4f}, n={h4_em['n1']}")
print(f"  AB: Median={h4_em['median2']:.4f}, n={h4_em['n2']}")
print(f"  U = {h4_em['U']:.2f}, p = {h4_em['p']:.4f}, r = {h4_em['rank_biserial_r']:.3f}")

# Compare effect sizes
print(f"\n--- Effect Size Comparison ---")
print(f"  BB frames: r = {h4_bb['rank_biserial_r']:.3f} ({h4_bb['effect_size']})")
print(f"  EM frames: r = {h4_em['rank_biserial_r']:.3f} ({h4_em['effect_size']})")

if abs(h4_bb['rank_biserial_r']) > abs(h4_em['rank_biserial_r']):
    print(f"  → BB frames show larger condition effect (supports boundary-specific hypothesis)")
else:
    print(f"  → Effect sizes similar (does not support boundary-specific hypothesis)")

# Friedman test for within-subjects comparison
print(f"\n--- Friedman Test (Frame Type within participants) ---")
# Prepare data for each condition separately
for condition in ['AB', 'NB']:
    data = frame_analysis[frame_analysis['condition'] == condition][['BB', 'EM']].dropna()
    if len(data) > 0:
        stat, p = friedmanchisquare(data['BB'], data['EM'])
        print(f"  {condition}: χ²({len(data)-1}) = {stat:.3f}, p = {p:.4f}")

# ============================================================================
# HYPOTHESIS 5: ACCURACY VARIABILITY
# ============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: Accuracy Variability Across Movies")
print("="*80)
print("H0: No difference in accuracy variability between groups")
print("H1: AB participants show higher variability")
print("-"*80)

# Use rt_std as proxy (already calculated in participant data)
print(f"\nUsing RT standard deviation as variability measure:")

h5_results = mann_whitney_with_stats(
    ab_participants['rt_std'].dropna(),
    nb_participants['rt_std'].dropna(),
    alternative='greater'  # AB > NB
)

print(f"\nDescriptive Statistics:")
print(f"  AB: Median={h5_results['median1']:.4f}, Mean={h5_results['mean1']:.4f}")
print(f"  NB: Median={h5_results['median2']:.4f}, Mean={h5_results['mean2']:.4f}")

print(f"\nMann-Whitney U Test:")
print(f"  U = {h5_results['U']:.2f}, p = {h5_results['p']:.4f}")
print(f"  Rank-biserial r = {h5_results['rank_biserial_r']:.3f}")

# Levene's test for variance equality
print(f"\n--- Levene's Test (Variance Homogeneity) ---")
levene_stat, levene_p = levene(
    ab_participants['accuracy'].dropna(),
    nb_participants['accuracy'].dropna()
)
print(f"  W = {levene_stat:.3f}, p = {levene_p:.4f}")
if levene_p < 0.05:
    print(f"  → Variances are significantly different")
else:
    print(f"  → Variances are not significantly different")

# ============================================================================
# ADDITIONAL ANALYSIS: RESPONSE TIME
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL ANALYSIS: Response Time")
print("="*80)

rt_results = mann_whitney_with_stats(
    nb_participants['rt_mean'].dropna(),
    ab_participants['rt_mean'].dropna(),
    alternative='two-sided'
)

print(f"\nDescriptive Statistics:")
print(f"  NB: Median={rt_results['median1']:.4f}s, Mean={rt_results['mean1']:.4f}s")
print(f"  AB: Median={rt_results['median2']:.4f}s, Mean={rt_results['mean2']:.4f}s")

print(f"\nMann-Whitney U Test:")
print(f"  U = {rt_results['U']:.2f}, p = {rt_results['p']:.4f}")
print(f"  Rank-biserial r = {rt_results['rank_biserial_r']:.3f}")

# ============================================================================
# MULTIPLE COMPARISONS CORRECTION
# ============================================================================
print("\n" + "="*80)
print("MULTIPLE COMPARISONS CORRECTION")
print("="*80)

# Collect all p-values from primary hypotheses
primary_tests = {
    'H1: Overall Accuracy': h1_results['p'],
    'H2: BB-Frame Accuracy': h2_results['p'],
    'H3: Overall Confidence': h3_results['p'],
    'H4a: BB Frame Comparison': h4_bb['p'],
    'H4b: EM Frame Comparison': h4_em['p'],
    'H5: Variability': h5_results['p']
}

p_values = list(primary_tests.values())
test_names = list(primary_tests.keys())

# Apply corrections
print("\n--- Bonferroni Correction ---")
bonf_alpha = 0.05 / len(p_values)
print(f"  Adjusted α = {bonf_alpha:.4f}")
for name, p in primary_tests.items():
    sig = "✓ SIGNIFICANT" if p < bonf_alpha else "✗ NOT SIGNIFICANT"
    print(f"  {name}: p = {p:.4f} {sig}")

print("\n--- Benjamini-Hochberg FDR Correction ---")
rejected, p_corrected = apply_multiple_comparison_correction(p_values, method='fdr_bh')
for name, p_orig, p_corr, sig in zip(test_names, p_values, p_corrected, rejected):
    sig_text = "✓ SIGNIFICANT" if sig else "✗ NOT SIGNIFICANT"
    print(f"  {name}: p_uncorrected = {p_orig:.4f}, p_corrected = {p_corr:.4f} {sig_text}")

# ============================================================================
# CORRELATION ANALYSIS (CONFIDENCE-ACCURACY)
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Confidence-Accuracy Relationship")
print("="*80)

for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    rho, p = spearmanr(data['accuracy'], data['confidence_mean'])
    print(f"\n{condition} Condition:")
    print(f"  Spearman ρ = {rho:.3f}, p = {p:.4f}")
    if p < 0.05:
        print(f"  → Significant positive correlation")
    else:
        print(f"  → No significant correlation")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL TESTS")
print("="*80)

summary_data = []
summary_data.append(['H1: Overall Accuracy', 'Mann-Whitney U', h1_results['U'], h1_results['p'], 
                     h1_results['rank_biserial_r'], h1_results['effect_size']])
summary_data.append(['H2: BB-Frame Accuracy', 'Mann-Whitney U', h2_results['U'], h2_results['p'],
                     h2_results['rank_biserial_r'], h2_results['effect_size']])
summary_data.append(['H3: Overall Confidence', 'Mann-Whitney U', h3_results['U'], h3_results['p'],
                     h3_results['rank_biserial_r'], h3_results['effect_size']])
summary_data.append(['H4a: BB Frame (NB>AB)', 'Mann-Whitney U', h4_bb['U'], h4_bb['p'],
                     h4_bb['rank_biserial_r'], h4_bb['effect_size']])
summary_data.append(['H4b: EM Frame (NB>AB)', 'Mann-Whitney U', h4_em['U'], h4_em['p'],
                     h4_em['rank_biserial_r'], h4_em['effect_size']])
summary_data.append(['H5: Variability (AB>NB)', 'Mann-Whitney U', h5_results['U'], h5_results['p'],
                     h5_results['rank_biserial_r'], h5_results['effect_size']])
summary_data.append(['Additional: Response Time', 'Mann-Whitney U', rt_results['U'], rt_results['p'],
                     rt_results['rank_biserial_r'], rt_results['effect_size']])

df_summary = pd.DataFrame(summary_data, 
                         columns=['Hypothesis', 'Test', 'Statistic', 'p-value', 'Effect_Size_r', 'Interpretation'])

print("\n")
print(df_summary.to_string(index=False))

# Save summary
df_summary.to_csv(OUTPUT_DIR / 'statistical_summary.csv', index=False)
print(f"\n✓ Summary saved to: {OUTPUT_DIR / 'statistical_summary.csv'}")

# ============================================================================
# SAVE DETAILED RESULTS
# ============================================================================

# Save all results to text file
with open(OUTPUT_DIR / 'detailed_statistical_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BRSM MOVIE MEMORY - DETAILED STATISTICAL RESULTS\n")
    f.write("NON-PARAMETRIC ANALYSES (Data violates normality assumption)\n")
    f.write("="*80 + "\n\n")
    
    # H1
    f.write("HYPOTHESIS 1: Overall Recognition Accuracy\n")
    f.write("-"*80 + "\n")
    f.write(f"NB: n={h1_results['n2']}, Mdn={h1_results['median2']:.4f}, M={h1_results['mean2']:.4f}\n")
    f.write(f"AB: n={h1_results['n1']}, Mdn={h1_results['median1']:.4f}, M={h1_results['mean1']:.4f}\n")
    f.write(f"Mann-Whitney U = {h1_results['U']:.2f}, p = {h1_results['p']:.4f}\n")
    f.write(f"Effect size (rank-biserial r) = {h1_results['rank_biserial_r']:.3f} ({h1_results['effect_size']})\n")
    f.write(f"Cohen's d (for comparison) = {h1_results['cohens_d']:.3f}\n\n")
    
    # H2
    f.write("HYPOTHESIS 2: BB-Frame Recognition Accuracy\n")
    f.write("-"*80 + "\n")
    f.write(f"NB: n={h2_results['n1']}, Mdn={h2_results['median1']:.4f}, M={h2_results['mean1']:.4f}\n")
    f.write(f"AB: n={h2_results['n2']}, Mdn={h2_results['median2']:.4f}, M={h2_results['mean2']:.4f}\n")
    f.write(f"Mann-Whitney U = {h2_results['U']:.2f}, p = {h2_results['p']:.4f}\n")
    f.write(f"Effect size (rank-biserial r) = {h2_results['rank_biserial_r']:.3f} ({h2_results['effect_size']})\n\n")
    
    # Add more...
    f.write("\nMULTIPLE COMPARISONS CORRECTION\n")
    f.write("-"*80 + "\n")
    f.write(f"Bonferroni-corrected α = {bonf_alpha:.4f}\n\n")
    
    f.write("Benjamini-Hochberg FDR Correction:\n")
    for name, p_orig, p_corr, sig in zip(test_names, p_values, p_corrected, rejected):
        f.write(f"  {name}: p={p_orig:.4f} → p_corrected={p_corr:.4f} {'[SIG]' if sig else '[NS]'}\n")

print(f"\n✓ Detailed results saved to: {OUTPUT_DIR / 'detailed_statistical_results.txt'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - statistical_summary.csv")
print("  - detailed_statistical_results.txt")
print("\n" + "="*80)
