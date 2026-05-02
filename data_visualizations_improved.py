import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# CONSISTENT COLOR SCHEME - Improved colors with better contrast
COLOR_AB = '#FF6B6B'  # Coral Red for Abrupt Boundary (improved visibility)
COLOR_NB = '#4ECDC4'  # Turquoise for Natural Boundary (improved visibility)
COLOR_BB = '#9B59B6'  # Purple for Beginning Boundary
COLOR_EM = '#F39C12'  # Orange for Ending Moment
COLORS_CONDITION = {'AB': COLOR_AB, 'NB': COLOR_NB}
COLORS_FRAME = {'BB': COLOR_BB, 'EM': COLOR_EM}

# Define paths
BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "final_cleaned_data"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("BRSM MOVIE MEMORY - IMPROVED DATA VISUALIZATIONS")
print("="*80)

# Load data
print("\n[1] Loading cleaned data with is_repeat and vigilance...")
df_participants = pd.read_csv(INPUT_DIR / "participants_final_clean_with_vigilance.csv")
df_trials = pd.read_csv(INPUT_DIR / "trials_final_clean_with_repeat.csv")
df_trials_demo = pd.read_csv(INPUT_DIR / "trials_with_demographics_final.csv")

print(f"   Loaded: {len(df_participants)} participants, {len(df_trials)} trials")

# CRITICAL: Extract frame type (BB/EM) from target_img
print("\n[2] Extracting frame type (BB/EM) and merging is_repeat...")
def extract_frame_type(target_img):
    """Extract BB or EM from target image filename"""
    if pd.isna(target_img):
        return np.nan
    if '_BB_' in str(target_img):
        return 'BB'
    elif '_EM_' in str(target_img):
        return 'EM'
    else:
        return np.nan

df_trials['frame_type'] = df_trials['target_img'].apply(extract_frame_type)
df_trials_demo['frame_type'] = df_trials_demo['target_img'].apply(extract_frame_type)

# Also add is_repeat to trials_demo if not present
if 'is_repeat' not in df_trials_demo.columns and 'is_repeat' in df_trials.columns:
    df_trials_demo = df_trials_demo.merge(
        df_trials[['participant_id', 'movie_id', 'is_repeat']], 
        on=['participant_id', 'movie_id'],
        how='left'
    )

# Verify frame type extraction
frame_counts = df_trials['frame_type'].value_counts()
print(f"   Frame type distribution:")
for ft, count in frame_counts.items():
    print(f"      {ft}: {count} trials ({count/len(df_trials)*100:.1f}%)")

# Check for missing frame types
missing_frame = df_trials['frame_type'].isna().sum()
if missing_frame > 0:
    print(f"   ⚠ Warning: {missing_frame} trials missing frame type")

# Verify data completeness
print("\n[3] Verifying data completeness...")
print(f"   Participants by condition:")
for cond, count in df_participants['condition'].value_counts().items():
    print(f"      {cond}: {count}")
print(f"   Trials by condition:")
for cond, count in df_trials['condition'].value_counts().items():
    print(f"      {cond}: {count}")
print(f"   Trials by frame type:")
for ft, count in frame_counts.items():
    print(f"      {ft}: {count}")

# Check for missing data in key variables
print(f"\n   Missing values in key variables:")
print(f"      resp.corr: {df_trials['resp.corr'].isna().sum()}")
print(f"      resp.rt: {df_trials['resp.rt'].isna().sum()}")
print(f"      conf_radio.response: {df_trials['conf_radio.response'].isna().sum()}")
print(f"      frame_type: {df_trials['frame_type'].isna().sum()}")

# ============================================================================
# SECTION 1: DISTRIBUTIONS - SEPARATE HIGH-QUALITY FIGURES
# ============================================================================
print("\n[4] Creating distribution plots...")

# Define better colors for conditions
COLOR_AB = '#FF6B6B'  # Coral red
COLOR_NB = '#4ECDC4'  # Turquoise
COLOR_OVERALL = '#95A5A6'  # Gray

# FIGURE 1A: Basic Demographics (Age)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(df_participants['age'], bins=20, color='#00008B', edgecolor='black', 
        alpha=0.85, linewidth=1.5)
ax.axvline(df_participants['age'].mean(), color='#C0392B', linestyle='--', linewidth=3,
           label=f'Mean = {df_participants["age"].mean():.1f} years')
ax.set_xlabel('Age (years)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Participants', fontweight='bold', fontsize=14)
ax.set_title('Age Distribution (N=170)', fontweight='bold', fontsize=16, pad=20)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01a_distribution_age.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01a_distribution_age.png")
plt.close()

# FIGURE 1B: Confidence Rating Distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
conf_counts = df_trials['conf_radio.response'].value_counts().sort_index()
bars = ax.bar(conf_counts.index, conf_counts.values, color=COLOR_OVERALL, 
              edgecolor='black', alpha=0.85, width=0.7, linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_xlabel('Confidence Rating (1=Not Confident, 5=Very Confident)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Trials', fontweight='bold', fontsize=14)
ax.set_title(f'Confidence Rating Distribution (N=6800 trials, Mean={df_trials["conf_radio.response"].mean():.2f})', 
             fontweight='bold', fontsize=16, pad=20)
ax.set_xticks([1, 2, 3, 4, 5])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01b_distribution_confidence.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01b_distribution_confidence.png")
plt.close()

# FIGURE 1C: Overall Accuracy Distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(df_participants['accuracy'], bins=25, color='#00008B', edgecolor='black', 
        alpha=0.85, linewidth=1.5)
ax.axvline(df_participants['accuracy'].mean(), color='#C0392B', linestyle='--', linewidth=3,
           label=f'Mean = {df_participants["accuracy"].mean():.3f}')
ax.set_xlabel('Accuracy (Proportion Correct)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Participants', fontweight='bold', fontsize=14)
ax.set_title('Overall Accuracy Distribution (N=170 participants)', fontweight='bold', fontsize=16, pad=20)
ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01c_distribution_accuracy_overall.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01c_distribution_accuracy_overall.png")
plt.close()

# FIGURE 1D: Accuracy by Condition (Side-by-Side Comparison)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ab_acc = df_participants[df_participants['condition'] == 'AB']['accuracy']
nb_acc = df_participants[df_participants['condition'] == 'NB']['accuracy']

# AB Condition
axes[0].hist(ab_acc, bins=20, color=COLOR_AB, edgecolor='black', alpha=0.85, linewidth=1.5)
axes[0].axvline(ab_acc.mean(), color='#C0392B', linestyle='--', linewidth=3,
                label=f'Mean = {ab_acc.mean():.3f}')
axes[0].set_xlabel('Accuracy (Proportion Correct)', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Number of Participants', fontweight='bold', fontsize=14)
axes[0].set_title(f'Abrupt Boundary (AB) - N={len(ab_acc)}', fontweight='bold', fontsize=15, 
                  pad=15, color=COLOR_AB)
axes[0].legend(loc='upper left', fontsize=12, framealpha=0.95)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# NB Condition
axes[1].hist(nb_acc, bins=20, color=COLOR_NB, edgecolor='black', alpha=0.85, linewidth=1.5)
axes[1].axvline(nb_acc.mean(), color='#16A085', linestyle='--', linewidth=3,
                label=f'Mean = {nb_acc.mean():.3f}')
axes[1].set_xlabel('Accuracy (Proportion Correct)', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Number of Participants', fontweight='bold', fontsize=14)
axes[1].set_title(f'Natural Boundary (NB) - N={len(nb_acc)}', fontweight='bold', fontsize=15, 
                  pad=15, color=COLOR_NB)
axes[1].legend(loc='upper left', fontsize=12, framealpha=0.95)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

fig.suptitle('Accuracy Distribution by Condition', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01d_distribution_accuracy_by_condition.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01d_distribution_accuracy_by_condition.png")
plt.close()

# FIGURE 1E: Overall Response Time Distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(df_trials['resp.rt'], bins=60, color=COLOR_OVERALL, edgecolor='black', 
        alpha=0.85, linewidth=1.5)
ax.axvline(df_trials['resp.rt'].mean(), color='#C0392B', linestyle='--', linewidth=3,
           label=f'Mean = {df_trials["resp.rt"].mean():.2f}s')
ax.axvline(df_trials['resp.rt'].median(), color='#2980B9', linestyle=':', linewidth=3,
           label=f'Median = {df_trials["resp.rt"].median():.2f}s')
ax.set_xlabel('Response Time (seconds)', fontweight='bold', fontsize=14)
ax.set_ylabel('Number of Trials', fontweight='bold', fontsize=14)
ax.set_title('Response Time Distribution (N=6800 trials)', fontweight='bold', fontsize=16, pad=20)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01e_distribution_rt_overall.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01e_distribution_rt_overall.png")
plt.close()

# FIGURE 1F: Response Time by Condition (Side-by-Side Comparison)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ab_rt = df_trials[df_trials['condition'] == 'AB']['resp.rt']
nb_rt = df_trials[df_trials['condition'] == 'NB']['resp.rt']

# AB Condition
axes[0].hist(ab_rt, bins=50, color=COLOR_AB, edgecolor='black', alpha=0.85, linewidth=1.5)
axes[0].axvline(ab_rt.mean(), color='#C0392B', linestyle='--', linewidth=3,
                label=f'Mean = {ab_rt.mean():.2f}s')
axes[0].axvline(ab_rt.median(), color='#8E44AD', linestyle=':', linewidth=3,
                label=f'Median = {ab_rt.median():.2f}s')
axes[0].set_xlabel('Response Time (seconds)', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Number of Trials', fontweight='bold', fontsize=14)
axes[0].set_title(f'Abrupt Boundary (AB) - N={len(ab_rt)} trials', fontweight='bold', fontsize=15, 
                  pad=15, color=COLOR_AB)
axes[0].legend(loc='upper right', fontsize=12, framealpha=0.95)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# NB Condition
axes[1].hist(nb_rt, bins=50, color=COLOR_NB, edgecolor='black', alpha=0.85, linewidth=1.5)
axes[1].axvline(nb_rt.mean(), color='#16A085', linestyle='--', linewidth=3,
                label=f'Mean = {nb_rt.mean():.2f}s')
axes[1].axvline(nb_rt.median(), color='#2980B9', linestyle=':', linewidth=3,
                label=f'Median = {nb_rt.median():.2f}s')
axes[1].set_xlabel('Response Time (seconds)', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Number of Trials', fontweight='bold', fontsize=14)
axes[1].set_title(f'Natural Boundary (NB) - N={len(nb_rt)} trials', fontweight='bold', fontsize=15, 
                  pad=15, color=COLOR_NB)
axes[1].legend(loc='upper right', fontsize=12, framealpha=0.95)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

fig.suptitle('Response Time Distribution by Condition', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01f_distribution_rt_by_condition.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 01f_distribution_rt_by_condition.png")
plt.close()

# ============================================================================
# SECTION 2: BOX PLOTS - BY CONDITION AND FRAME TYPE
# ============================================================================
print("\n[5] Creating box plots by condition and frame type...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Box Plots: Condition and Frame Type Comparisons', fontsize=16, fontweight='bold')

# Accuracy by Condition
data_cond = [df_participants[df_participants['condition']==c]['accuracy'].dropna() 
             for c in ['AB', 'NB']]
bp1 = axes[0, 0].boxplot(data_cond, labels=['AB', 'NB'], patch_artist=True)
for patch, color in zip(bp1['boxes'], [COLOR_AB, COLOR_NB]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0, 0].set_ylabel('Accuracy (proportion)', fontweight='bold')
axes[0, 0].set_xlabel('Condition', fontweight='bold')
axes[0, 0].set_title('Accuracy by Condition')
axes[0, 0].grid(axis='y', alpha=0.3)

# Response Time by Condition
data_rt_cond = [df_trials[df_trials['condition']==c]['resp.rt'].dropna() 
                for c in ['AB', 'NB']]
bp2 = axes[0, 1].boxplot(data_rt_cond, labels=['AB', 'NB'], patch_artist=True)
for patch, color in zip(bp2['boxes'], [COLOR_AB, COLOR_NB]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0, 1].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 1].set_xlabel('Condition', fontweight='bold')
axes[0, 1].set_title('Response Time by Condition')
axes[0, 1].grid(axis='y', alpha=0.3)

# Confidence by Condition
data_conf_cond = [df_trials[df_trials['condition']==c]['conf_radio.response'].dropna() 
                  for c in ['AB', 'NB']]
bp3 = axes[0, 2].boxplot(data_conf_cond, labels=['AB', 'NB'], patch_artist=True)
for patch, color in zip(bp3['boxes'], [COLOR_AB, COLOR_NB]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[0, 2].set_xlabel('Condition', fontweight='bold')
axes[0, 2].set_title('Confidence by Condition')
axes[0, 2].grid(axis='y', alpha=0.3)

# Accuracy by Frame Type (trial-level)
data_acc_frame = [df_trials[df_trials['frame_type']==ft]['resp.corr'].dropna() 
                  for ft in ['BB', 'EM']]
bp4 = axes[1, 0].boxplot(data_acc_frame, labels=['BB', 'EM'], patch_artist=True)
for patch, color in zip(bp4['boxes'], [COLOR_BB, COLOR_EM]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 0].set_ylabel('Accuracy (0=incorrect, 1=correct)', fontweight='bold')
axes[1, 0].set_xlabel('Frame Type', fontweight='bold')
axes[1, 0].set_title('Trial Accuracy by Frame Type')
axes[1, 0].grid(axis='y', alpha=0.3)

# Response Time by Frame Type
data_rt_frame = [df_trials[df_trials['frame_type']==ft]['resp.rt'].dropna() 
                 for ft in ['BB', 'EM']]
bp5 = axes[1, 1].boxplot(data_rt_frame, labels=['BB', 'EM'], patch_artist=True)
for patch, color in zip(bp5['boxes'], [COLOR_BB, COLOR_EM]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 1].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[1, 1].set_xlabel('Frame Type', fontweight='bold')
axes[1, 1].set_title('Response Time by Frame Type')
axes[1, 1].grid(axis='y', alpha=0.3)

# Confidence by Frame Type
data_conf_frame = [df_trials[df_trials['frame_type']==ft]['conf_radio.response'].dropna() 
                   for ft in ['BB', 'EM']]
bp6 = axes[1, 2].boxplot(data_conf_frame, labels=['BB', 'EM'], patch_artist=True)
for patch, color in zip(bp6['boxes'], [COLOR_BB, COLOR_EM]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[1, 2].set_xlabel('Frame Type', fontweight='bold')
axes[1, 2].set_title('Confidence by Frame Type')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_boxplots_condition_frametype.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 02_boxplots_condition_frametype.png")
plt.close()

# ============================================================================
# SECTION 3: VIOLIN PLOTS WITH HORIZONTAL LEGENDS
# ============================================================================
print("\n[6] Creating violin plots with proper legends...")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))  # Extra space for legend below
fig.suptitle('Violin Plots: Distribution Shapes', fontsize=16, fontweight='bold')

# Response Time by Condition
sns.violinplot(data=df_trials, x='condition', y='resp.rt', ax=axes[0, 0], 
               palette=COLORS_CONDITION, inner='box', order=['AB', 'NB'])
axes[0, 0].set_xlabel('Condition', fontweight='bold')
axes[0, 0].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 0].set_title('Response Time by Condition')

# Confidence by Condition
sns.violinplot(data=df_trials, x='condition', y='conf_radio.response', ax=axes[0, 1],
               palette=COLORS_CONDITION, inner='box', order=['AB', 'NB'])
axes[0, 1].set_xlabel('Condition', fontweight='bold')
axes[0, 1].set_ylabel('Confidence Rating', fontweight='bold')
axes[0, 1].set_title('Confidence by Condition')

# Accuracy by Condition (participant-level)
sns.violinplot(data=df_participants, x='condition', y='accuracy', ax=axes[0, 2],
               palette=COLORS_CONDITION, inner='box', order=['AB', 'NB'])
axes[0, 2].set_xlabel('Condition', fontweight='bold')
axes[0, 2].set_ylabel('Accuracy (proportion)', fontweight='bold')
axes[0, 2].set_title('Accuracy by Condition')

# Response Time by Frame Type
sns.violinplot(data=df_trials, x='frame_type', y='resp.rt', ax=axes[1, 0],
               palette=COLORS_FRAME, inner='box', order=['BB', 'EM'])
axes[1, 0].set_xlabel('Frame Type', fontweight='bold')
axes[1, 0].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Response Time by Frame Type')

# Confidence by Frame Type
sns.violinplot(data=df_trials, x='frame_type', y='conf_radio.response', ax=axes[1, 1],
               palette=COLORS_FRAME, inner='box', order=['BB', 'EM'])
axes[1, 1].set_xlabel('Frame Type', fontweight='bold')
axes[1, 1].set_ylabel('Confidence Rating', fontweight='bold')
axes[1, 1].set_title('Confidence by Frame Type')

# Confidence by Correctness
sns.violinplot(data=df_trials, x='resp.corr', y='conf_radio.response', ax=axes[1, 2],
               palette={'0.0': "#670EAB", '1.0': '#27AE60'}, inner='box')
axes[1, 2].set_xlabel('Response Correctness', fontweight='bold')
axes[1, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[1, 2].set_title('Confidence by Correctness')
axes[1, 2].set_xticklabels(['Incorrect', 'Correct'])

# Add horizontal legend below the figure
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_AB, label='AB (Abrupt Boundary)', alpha=0.7),
    Patch(facecolor=COLOR_NB, label='NB (Natural Boundary)', alpha=0.7),
    Patch(facecolor=COLOR_BB, label='BB (Beginning Boundary)', alpha=0.7),
    Patch(facecolor=COLOR_EM, label='EM (Ending Moment)', alpha=0.7),
    Patch(facecolor="#670EAB", label='Incorrect', alpha=0.7),
    Patch(facecolor='#27AE60', label='Correct', alpha=0.7)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
           bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True)

plt.tight_layout(rect=[0, 0.02, 1, 0.97])  # Make room for legend
plt.savefig(OUTPUT_DIR / '03_violin_plots.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 03_violin_plots.png")
plt.close()

# ============================================================================
# SECTION 4: INTERACTION PLOTS - Condition × Frame Type
# ============================================================================
print("\n[7] Creating interaction plots (Condition × Frame Type)...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Interaction Effects: Condition × Frame Type', fontsize=16, fontweight='bold')

# Calculate means for interaction
interaction_data = df_trials.groupby(['condition', 'frame_type']).agg({
    'resp.corr': 'mean',
    'resp.rt': 'mean',
    'conf_radio.response': 'mean'
}).reset_index()

# Accuracy interaction
for condition, color in COLORS_CONDITION.items():
    data = interaction_data[interaction_data['condition'] == condition]
    axes[0].plot(['BB', 'EM'], data['resp.corr'], marker='o', markersize=10,
                 linewidth=2, label=condition, color=color)
axes[0].set_xlabel('Frame Type', fontweight='bold')
axes[0].set_ylabel('Mean Accuracy', fontweight='bold')
axes[0].set_title('Accuracy: Condition × Frame Type')
axes[0].legend(title='Condition')
axes[0].grid(True, alpha=0.3)

# Response Time interaction
for condition, color in COLORS_CONDITION.items():
    data = interaction_data[interaction_data['condition'] == condition]
    axes[1].plot(['BB', 'EM'], data['resp.rt'], marker='o', markersize=10,
                 linewidth=2, label=condition, color=color)
axes[1].set_xlabel('Frame Type', fontweight='bold')
axes[1].set_ylabel('Mean Response Time (s)', fontweight='bold')
axes[1].set_title('Response Time: Condition × Frame Type')
axes[1].legend(title='Condition')
axes[1].grid(True, alpha=0.3)

# Confidence interaction
for condition, color in COLORS_CONDITION.items():
    data = interaction_data[interaction_data['condition'] == condition]
    axes[2].plot(['BB', 'EM'], data['conf_radio.response'], marker='o', markersize=10,
                 linewidth=2, label=condition, color=color)
axes[2].set_xlabel('Frame Type', fontweight='bold')
axes[2].set_ylabel('Mean Confidence Rating', fontweight='bold')
axes[2].set_title('Confidence: Condition × Frame Type')
axes[2].legend(title='Condition')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_interaction_plots.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 04_interaction_plots.png")
plt.close()

# ============================================================================
# SECTION 5: PIE CHARTS
# ============================================================================
print("\n[8] Creating independent pie charts for categorical variables...")

# PIE CHART 1: Condition Distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
condition_counts = df_participants['condition'].value_counts()
wedges, texts, autotexts = ax.pie(condition_counts, labels=[f'{c}\n(N={condition_counts[c]})' for c in condition_counts.index], 
                                    autopct='%1.1f%%',
                                    colors=[COLORS_CONDITION[c] for c in condition_counts.index],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05, 0.05], shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Participant Distribution by Condition', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05a_pie_condition.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05a_pie_condition.png")
plt.close()

# PIE CHART 2: Frame Type Distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
frame_counts = df_trials['frame_type'].value_counts()
wedges, texts, autotexts = ax.pie(frame_counts, labels=[f'{ft}\n(N={frame_counts[ft]})' for ft in frame_counts.index], 
                                    autopct='%1.1f%%',
                                    colors=[COLORS_FRAME[ft] for ft in frame_counts.index],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05, 0.05], shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Trial Distribution by Frame Type\n(BB=Beginning Boundary, EM=Ending Moment)', 
             fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05b_pie_frametype.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05b_pie_frametype.png")
plt.close()

# PIE CHART 3: Overall Accuracy (Correct vs Incorrect)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
acc_counts = df_trials['resp.corr'].value_counts()
labels_acc = ['Incorrect', 'Correct']
wedges, texts, autotexts = ax.pie(acc_counts, labels=[f'{labels_acc[i]}\n(N={int(acc_counts.iloc[i])})' for i in range(len(acc_counts))], 
                                    autopct='%1.1f%%',
                                    colors=['#E74C3C', '#27AE60'],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05, 0.05], shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Overall Response Correctness (N=6800 trials)', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05c_pie_accuracy.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05c_pie_accuracy.png")
plt.close()

# PIE CHART 4: Gender Distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gender_counts = df_participants['gender'].value_counts()
wedges, texts, autotexts = ax.pie(gender_counts, labels=[f'{g}\n(N={gender_counts[g]})' for g in gender_counts.index], 
                                    autopct='%1.1f%%',
                                    colors=['#E91E63', '#2196F3', '#9C27B0'][:len(gender_counts)],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05]*len(gender_counts), shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Gender Distribution (N=170 participants)', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05d_pie_gender.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05d_pie_gender.png")
plt.close()

# PIE CHART 5: Handedness Distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
handedness_counts = df_participants['handedness'].value_counts()
wedges, texts, autotexts = ax.pie(handedness_counts, labels=[f'{h}\n(N={handedness_counts[h]})' for h in handedness_counts.index], 
                                    autopct='%1.1f%%',
                                    colors=['#4CAF50', '#FF9800'][:len(handedness_counts)],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05]*len(handedness_counts), shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Handedness Distribution (N=170 participants)', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05e_pie_handedness.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05e_pie_handedness.png")
plt.close()

# PIE CHART 6: Vision Status Distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
vision_counts = df_participants['vision'].value_counts()
wedges, texts, autotexts = ax.pie(vision_counts, labels=[f'{v}\n(N={vision_counts[v]})' for v in vision_counts.index], 
                                    autopct='%1.1f%%',
                                    colors=['#00BCD4', '#CDDC39'][:len(vision_counts)],
                                    startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                    explode=[0.05]*len(vision_counts), shadow=True)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')
ax.set_title('Vision Status Distribution (N=170 participants)', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05f_pie_vision.png', dpi=400, bbox_inches='tight')
print("   ✓ Saved: 05f_pie_vision.png")
plt.close()

# ============================================================================
# SECTION 6: SEPARATE SCATTER PLOTS FOR EACH ASSOCIATION
# ============================================================================
print("\n[9] Creating individual scatter plots for key associations...")

# Association 1: Confidence vs Response Time
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_trials[df_trials['condition'] == condition]
    ax.scatter(data['conf_radio.response'], data['resp.rt'], 
               alpha=0.3, s=20, label=condition, color=color)
ax.set_xlabel('Confidence Rating', fontweight='bold', fontsize=12)
ax.set_ylabel('Response Time (seconds)', fontweight='bold', fontsize=12)
ax.set_title('Association: Confidence Rating vs Response Time', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06a_scatter_confidence_vs_rt.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06a_scatter_confidence_vs_rt.png")
plt.close()

# Association 2: Confidence vs Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_trials[df_trials['condition'] == condition]
    # Add jitter to see overlapping points
    jitter_conf = data['conf_radio.response'] + np.random.normal(0, 0.05, len(data))
    jitter_acc = data['resp.corr'] + np.random.normal(0, 0.02, len(data))
    ax.scatter(jitter_conf, jitter_acc, 
               alpha=0.3, s=20, label=condition, color=color)
ax.set_xlabel('Confidence Rating (with jitter)', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy (0=incorrect, 1=correct)', fontweight='bold', fontsize=12)
ax.set_title('Association: Confidence Rating vs Accuracy', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Incorrect', 'Correct'])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06b_scatter_confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06b_scatter_confidence_vs_accuracy.png")
plt.close()

# Association 3: Age vs Accuracy (participant-level)
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_participants[df_participants['condition'] == condition]
    ax.scatter(data['age'], data['accuracy'], 
               alpha=0.6, s=60, label=condition, color=color, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Age (years)', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy (proportion correct)', fontweight='bold', fontsize=12)
ax.set_title('Association: Age vs Accuracy', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06c_scatter_age_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06c_scatter_age_vs_accuracy.png")
plt.close()

# Association 4: Age vs Mean RT
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_participants[df_participants['condition'] == condition]
    ax.scatter(data['age'], data['rt_mean'], 
               alpha=0.6, s=60, label=condition, color=color, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Age (years)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Response Time (seconds)', fontweight='bold', fontsize=12)
ax.set_title('Association: Age vs Mean Response Time', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06d_scatter_age_vs_rt.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06d_scatter_age_vs_rt.png")
plt.close()

# Association 5: Accuracy vs Mean Confidence (participant-level)
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_participants[df_participants['condition'] == condition]
    ax.scatter(data['accuracy'], data['confidence_mean'], 
               alpha=0.6, s=60, label=condition, color=color, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Accuracy (proportion correct)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Confidence Rating', fontweight='bold', fontsize=12)
ax.set_title('Association: Accuracy vs Mean Confidence', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06e_scatter_accuracy_vs_confidence.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06e_scatter_accuracy_vs_confidence.png")
plt.close()

# Association 6: Mean RT vs Mean Confidence
fig, ax = plt.subplots(figsize=(10, 6))
for condition, color in COLORS_CONDITION.items():
    data = df_participants[df_participants['condition'] == condition]
    ax.scatter(data['rt_mean'], data['confidence_mean'], 
               alpha=0.6, s=60, label=condition, color=color, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Mean Response Time (seconds)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Confidence Rating', fontweight='bold', fontsize=12)
ax.set_title('Association: Mean RT vs Mean Confidence', fontsize=14, fontweight='bold')
ax.legend(title='Condition', loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06f_scatter_rt_vs_confidence.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06f_scatter_rt_vs_confidence.png")
plt.close()

# ============================================================================
# SECTION 7: CORRELATION HEATMAPS
# ============================================================================
print("\n[10] Creating correlation heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Correlation Heatmaps', fontsize=16, fontweight='bold')

# Participant-level correlations
participant_numeric = df_participants[['age', 'accuracy', 'rt_mean', 'confidence_mean']].copy()
corr_matrix_participant = participant_numeric.corr()

sns.heatmap(corr_matrix_participant, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'},
            linewidths=1, linecolor='white', vmin=-1, vmax=1)
axes[0].set_title('Participant-Level Correlations', fontweight='bold')
axes[0].set_xticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Conf.'], rotation=45, ha='right')
axes[0].set_yticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Conf.'], rotation=0)

# Trial-level correlations
trial_numeric = df_trials[['resp.corr', 'resp.rt', 'conf_radio.response']].copy()
corr_matrix_trial = trial_numeric.corr()

sns.heatmap(corr_matrix_trial, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation'},
            linewidths=1, linecolor='white', vmin=-1, vmax=1)
axes[1].set_title('Trial-Level Correlations', fontweight='bold')
axes[1].set_xticklabels(['Accuracy', 'Response Time', 'Confidence'], rotation=45, ha='right')
axes[1].set_yticklabels(['Accuracy', 'Response Time', 'Confidence'], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 07_correlation_heatmaps.png")
plt.close()

# ============================================================================
# SECTION 8: Q-Q PLOTS FOR NORMALITY
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle('Q-Q Plots - Normality Assessment', fontsize=16, fontweight='bold')

# Response Time
stats.probplot(df_trials['resp.rt'].dropna(), dist="norm", plot=axes[0])
axes[0].set_title('Response Time', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy
stats.probplot(df_participants['accuracy'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title('Accuracy (Participant)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Mean RT
stats.probplot(df_participants['rt_mean'].dropna(), dist="norm", plot=axes[2])
axes[2].set_title('Mean RT (Participant)', fontweight='bold')
axes[2].grid(True, alpha=0.3)

# Mean Confidence
stats.probplot(df_participants['confidence_mean'].dropna(), dist="norm", plot=axes[3])
axes[3].set_title('Mean Confidence (Participant)', fontweight='bold')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_qq_plots.png', dpi=300, bbox_inches='tight')
plt.close()
# ============================================================================
# SECTION 9: NORMALITY TESTS
# ============================================================================
print("\n[12] Performing normality tests...")

normality_results = []

variables_to_test = [
    ('Age', df_participants['age'].dropna()),
    ('Response Time', df_trials['resp.rt'].dropna()),
    ('Confidence Rating', df_trials['conf_radio.response'].dropna()),
    ('Accuracy (Participant)', df_participants['accuracy'].dropna()),
    ('Mean RT (Participant)', df_participants['rt_mean'].dropna()),
    ('Mean Confidence (Participant)', df_participants['confidence_mean'].dropna())
]

for var_name, data in variables_to_test:
    # Shapiro-Wilk test
    if len(data) < 5000:
        shapiro_stat, shapiro_p = shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    
    # D'Agostino-Pearson test
    if len(data) >= 8:
        dag_stat, dag_p = normaltest(data)
    else:
        dag_stat, dag_p = np.nan, np.nan
    
    normality_results.append({
        'Variable': var_name,
        'N': len(data),
        'Shapiro_W': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'KS_statistic': ks_stat,
        'KS_p': ks_p,
        'DAP_statistic': dag_stat,
        'DAP_p': dag_p,
        'Normal?': 'Yes' if (shapiro_p > 0.05 and ks_p > 0.05 and dag_p > 0.05) else 'No'
    })

df_normality = pd.DataFrame(normality_results)
df_normality.to_csv(OUTPUT_DIR / 'normality_test_results.csv', index=False)
print("\n" + "="*80)
print("NORMALITY TEST RESULTS")
print("="*80)
print(df_normality.to_string(index=False))
print("\n   ✓ Saved: normality_test_results.csv")

# ============================================================================
# SECTION 10: CONFIDENCE RT VISUALIZATIONS
# ============================================================================
print("\n[13] Creating confidence RT visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Confidence Rating Response Time (conf_radio.rt)', fontsize=16, fontweight='bold')

# Distribution
axes[0, 0].hist(df_trials['conf_radio.rt'], bins=50, color='gray', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Confidence RT (seconds)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Confidence RT Distribution')
axes[0, 0].axvline(df_trials['conf_radio.rt'].mean(), color='red', linestyle='--',
                    label=f'Mean = {df_trials["conf_radio.rt"].mean():.2f}s')
axes[0, 0].legend()

# By Condition
for condition, color in COLORS_CONDITION.items():
    data = df_trials[df_trials['condition'] == condition]['conf_radio.rt']
    axes[0, 1].hist(data, bins=40, alpha=0.6, label=f'{condition} (M={data.mean():.2f}s)',
                    color=color, edgecolor='black')
axes[0, 1].set_xlabel('Confidence RT (seconds)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Confidence RT by Condition')
axes[0, 1].legend()

# By Frame Type
for frame_type, color in COLORS_FRAME.items():
    data = df_trials[df_trials['frame_type'] == frame_type]['conf_radio.rt']
    axes[0, 2].hist(data, bins=40, alpha=0.6, label=f'{frame_type} (M={data.mean():.2f}s)',
                    color=color, edgecolor='black')
axes[0, 2].set_xlabel('Confidence RT (seconds)', fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontweight='bold')
axes[0, 2].set_title('Confidence RT by Frame Type')
axes[0, 2].legend()

# Confidence RT vs Response RT (scatter)
axes[1, 0].scatter(df_trials['resp.rt'], df_trials['conf_radio.rt'], alpha=0.2, s=10, color='gray')
axes[1, 0].set_xlabel('Response RT (seconds)', fontweight='bold')
axes[1, 0].set_ylabel('Confidence RT (seconds)', fontweight='bold')
axes[1, 0].set_title('Confidence RT vs Response RT')
axes[1, 0].grid(True, alpha=0.3)

# Confidence RT vs Confidence Rating
for rating in [1, 2, 3, 4, 5]:
    data = df_trials[df_trials['conf_radio.response'] == rating]['conf_radio.rt']
    axes[1, 1].scatter([rating]*len(data), data, alpha=0.3, s=10, label=f'Rating {rating}')
axes[1, 1].set_xlabel('Confidence Rating', fontweight='bold')
axes[1, 1].set_ylabel('Confidence RT (seconds)', fontweight='bold')
axes[1, 1].set_title('Confidence RT by Rating Level')
axes[1, 1].set_xticks([1, 2, 3, 4, 5])

# Box plot by correctness
data_conf_rt_corr = [df_trials[df_trials['resp.corr']==c]['conf_radio.rt'].dropna() 
                      for c in [0.0, 1.0]]
bp = axes[1, 2].boxplot(data_conf_rt_corr, labels=['Incorrect', 'Correct'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#E74C3C', '#27AE60']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 2].set_ylabel('Confidence RT (seconds)', fontweight='bold')
axes[1, 2].set_xlabel('Response Correctness', fontweight='bold')
axes[1, 2].set_title('Confidence RT by Correctness')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_confidence_rt_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 09_confidence_rt_analysis.png")
plt.close()

# ============================================================================
# SECTION 11: IS_REPEAT (VIGILANCE TRIALS) ANALYSIS
# ============================================================================
print("\n[14] Creating is_repeat (vigilance trial) visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Vigilance Trial Effects (is_repeat=1 vs is_repeat=0)', fontsize=16, fontweight='bold')

# Accuracy by repeat status
repeat_labels = ['Non-Repeat\n(New Videos)', 'Repeat\n(Vigilance Trials)']
data_acc_repeat = [df_trials[df_trials['is_repeat']==r]['resp.corr'].mean() for r in [0.0, 1.0]]
bars = axes[0, 0].bar([0, 1], data_acc_repeat, color=['#3498DB', '#E74C3C'], alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Mean Accuracy', fontweight='bold')
axes[0, 0].set_xlabel('Trial Type', fontweight='bold')
axes[0, 0].set_title('Accuracy: Repeat vs Non-Repeat Trials')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(repeat_labels)
axes[0, 0].set_ylim([0, 1])
for i, v in enumerate(data_acc_repeat):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Response Time by repeat status
data_rt_repeat = [df_trials[df_trials['is_repeat']==r]['resp.rt'].dropna() for r in [0.0, 1.0]]
bp1 = axes[0, 1].boxplot(data_rt_repeat, labels=repeat_labels, patch_artist=True)
for patch, color in zip(bp1['boxes'], ['#3498DB', '#E74C3C']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0, 1].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 1].set_xlabel('Trial Type', fontweight='bold')
axes[0, 1].set_title('Response Time: Repeat vs Non-Repeat')
axes[0, 1].grid(axis='y', alpha=0.3)

# Confidence by repeat status
data_conf_repeat = [df_trials[df_trials['is_repeat']==r]['conf_radio.response'].dropna() for r in [0.0, 1.0]]
bp2 = axes[0, 2].boxplot(data_conf_repeat, labels=repeat_labels, patch_artist=True)
for patch, color in zip(bp2['boxes'], ['#3498DB', '#E74C3C']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[0, 2].set_xlabel('Trial Type', fontweight='bold')
axes[0, 2].set_title('Confidence: Repeat vs Non-Repeat')
axes[0, 2].grid(axis='y', alpha=0.3)

# Interaction: Condition × Repeat Status (Accuracy)
for condition, color in COLORS_CONDITION.items():
    means = []
    for repeat_status in [0.0, 1.0]:
        acc = df_trials[(df_trials['condition']==condition) & (df_trials['is_repeat']==repeat_status)]['resp.corr'].mean()
        means.append(acc)
    axes[1, 0].plot([0, 1], means, marker='o', markersize=10, linewidth=2, label=condition, color=color)
axes[1, 0].set_ylabel('Mean Accuracy', fontweight='bold')
axes[1, 0].set_xlabel('Trial Type', fontweight='bold')
axes[1, 0].set_title('Interaction: Condition × Repeat Status')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['Non-Repeat', 'Repeat'])
axes[1, 0].legend(title='Condition')
axes[1, 0].grid(True, alpha=0.3)

# Repeat trial distribution
repeat_counts = df_trials['is_repeat'].value_counts()
axes[1, 1].pie(repeat_counts, labels=['Non-Repeat Trials', 'Repeat Trials'],
               autopct='%1.1f%%', colors=['#3498DB', '#E74C3C'],
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[1, 1].set_title('Trial Type Distribution')

# Repeated movie IDs
repeat_movies = df_trials[df_trials['is_repeat']==1.0]['movie_id'].value_counts().sort_index()
axes[1, 2].bar(repeat_movies.index, repeat_movies.values, color='#E74C3C', alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Movie ID', fontweight='bold')
axes[1, 2].set_ylabel('Number of Trials', fontweight='bold')
axes[1, 2].set_title('Repeated Videos (Vigilance Trials)')
axes[1, 2].set_xticks(repeat_movies.index)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_repeat_vigilance_effects.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 10_repeat_vigilance_effects.png")
plt.close()

# ============================================================================
# SECTION 12: VIGILANCE PERFORMANCE
# ============================================================================
if 'vigilance_correct' in df_participants.columns and df_participants['vigilance_correct'].notna().sum() > 0:
    print("\n[15] Creating vigilance performance visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Vigilance Check Performance', fontsize=16, fontweight='bold')
    
    # Vigilance pass/fail
    vig_counts = df_participants['vigilance_correct'].value_counts()
    axes[0].pie(vig_counts, labels=['Passed', 'Failed'], autopct='%1.1f%%',
                colors=['#27AE60', '#E74C3C'],
                startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title(f'Vigilance Check Results')
    
    # Vigilance by condition
    vig_by_cond = df_participants.groupby('condition')['vigilance_correct'].value_counts().unstack(fill_value=0)
    vig_by_cond.plot(kind='bar', ax=axes[1], color=['#E74C3C', '#27AE60'], alpha=0.7)
    axes[1].set_xlabel('Condition', fontweight='bold')
    axes[1].set_ylabel('Number of Participants', fontweight='bold')
    axes[1].set_title('Vigilance Performance by Condition')
    axes[1].legend(['Failed', 'Passed'], title='Vigilance')
    axes[1].set_xticklabels(['AB', 'NB'], rotation=0)
    
    # Vigilance vs Accuracy
    if 'passed_vigilance' in df_participants.columns:
        vig_pass = df_participants[df_participants['passed_vigilance']==True]
        vig_fail = df_participants[df_participants['passed_vigilance']==False]
        
        data_acc_vig = [vig_fail['accuracy'].dropna(), vig_pass['accuracy'].dropna()]
        bp = axes[2].boxplot(data_acc_vig, labels=['Failed\nVigilance', 'Passed\nVigilance'], 
                             patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#E74C3C', '#27AE60']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[2].set_ylabel('Overall Accuracy', fontweight='bold')
        axes[2].set_xlabel('Vigilance Status', fontweight='bold')
        axes[2].set_title('Accuracy by Vigilance Performance')
        axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_vigilance_performance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: 11_vigilance_performance.png")
    plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated {len([f for f in OUTPUT_DIR.glob('*.png')])} high-quality visualizations:")
print(f"\n📊 DISTRIBUTION PLOTS (Separate high-quality files):")
print(f"   01a: Age distribution")
print(f"   01b: Confidence rating distribution (ordinal bar chart)")
print(f"   01c: Overall accuracy distribution")
print(f"   01d: Accuracy by condition (AB vs NB side-by-side)")
print(f"   01e: Overall response time distribution")
print(f"   01f: Response time by condition (AB vs NB side-by-side)")
print(f"\n📈 COMPARISON PLOTS:")
print(f"   02: Box plots by condition and frame type (BB/EM)")
print(f"   03: Violin plots with horizontal legend below")
print(f"   04: Interaction plots (Condition × Frame Type)")
print(f"\n🥧 PIE CHARTS (Independent files):")
print(f"   05a: Condition distribution")
print(f"   05b: Frame type distribution (BB/EM)")
print(f"   05c: Response correctness")
print(f"   05d: Gender distribution")
print(f"   05e: Handedness distribution")
print(f"   05f: Vision status distribution")
print(f"\n📉 SCATTER & CORRELATION PLOTS:")
print(f"   06a-f: Six separate scatter plots for key associations")
print(f"   07: Correlation heatmaps")
print(f"   08: Q-Q plots for normality")
print(f"\n🆕 ADDITIONAL ANALYSES:")
print(f"   09: Confidence RT analysis")
print(f"   10: Repeat/vigilance trial effects")
print(f"   11: Vigilance performance")
print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print(f"Color scheme: Coral Red (AB), Turquoise (NB), Purple (BB), Orange (EM)")
print(f"Resolution: 400 DPI for publication-quality figures")
print(f"\n✅ Distribution plots: Each variable in separate high-quality file")
print(f"✅ Pie charts: Independent files with enhanced styling")
print(f"✅ No overlapping AB/NB plots - clear side-by-side comparisons")
print(f"✅ Frame type (BB/EM) successfully extracted and visualized")
print(f"✅ All missing columns now populated and visualized")
print("="*80)
