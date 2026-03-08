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

# Define paths
BASE_DIR = Path(".")
INPUT_DIR = BASE_DIR / "final_cleaned_data"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("BRSM MOVIE MEMORY - DATA VISUALIZATIONS")
print("="*70)

# Load data
print("\n[1] Loading cleaned data...")
df_participants = pd.read_csv(INPUT_DIR / "participants_final_clean.csv")
df_trials = pd.read_csv(INPUT_DIR / "trials_final_clean.csv")
df_trials_demo = pd.read_csv(INPUT_DIR / "trials_with_demographics_final.csv")

print(f"   Loaded: {len(df_participants)} participants, {len(df_trials)} trials")

# ============================================================================
# SECTION 1: DISTRIBUTIONS - HISTOGRAMS
# ============================================================================
print("\n[2] Creating histograms for continuous variables...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Continuous Variables', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df_participants['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age (years)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].axvline(df_participants['age'].mean(), color='red', linestyle='--', 
                    label=f'Mean = {df_participants["age"].mean():.1f}')
axes[0, 0].legend()

# Response Time distribution
axes[0, 1].hist(df_trials['resp.rt'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Response Time (seconds)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Response Time Distribution')
axes[0, 1].axvline(df_trials['resp.rt'].mean(), color='darkred', linestyle='--',
                    label=f'Mean = {df_trials["resp.rt"].mean():.2f}s')
axes[0, 1].legend()

# Confidence Rating distribution
axes[0, 2].hist(df_trials['conf_radio.response'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Confidence Rating', fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontweight='bold')
axes[0, 2].set_title('Confidence Rating Distribution')
axes[0, 2].axvline(df_trials['conf_radio.response'].mean(), color='darkgreen', linestyle='--',
                    label=f'Mean = {df_trials["conf_radio.response"].mean():.2f}')
axes[0, 2].legend()

# Accuracy distribution
axes[1, 0].hist(df_participants['accuracy'], bins=20, color='plum', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Accuracy (proportion correct)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Accuracy Distribution (by Participant)')
axes[1, 0].axvline(df_participants['accuracy'].mean(), color='purple', linestyle='--',
                    label=f'Mean = {df_participants["accuracy"].mean():.3f}')
axes[1, 0].legend()

# Mean Response Time by Participant
axes[1, 1].hist(df_participants['rt_mean'], bins=20, color='peachpuff', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Mean Response Time (seconds)', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Mean RT Distribution (by Participant)')
axes[1, 1].axvline(df_participants['rt_mean'].mean(), color='orange', linestyle='--',
                    label=f'Mean = {df_participants["rt_mean"].mean():.2f}s')
axes[1, 1].legend()

# Mean Confidence by Participant
axes[1, 2].hist(df_participants['confidence_mean'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Mean Confidence Rating', fontweight='bold')
axes[1, 2].set_ylabel('Frequency', fontweight='bold')
axes[1, 2].set_title('Mean Confidence Distribution (by Participant)')
axes[1, 2].axvline(df_participants['confidence_mean'].mean(), color='blue', linestyle='--',
                    label=f'Mean = {df_participants["confidence_mean"].mean():.2f}')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_histograms_continuous_variables.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 01_histograms_continuous_variables.png")
plt.close()

# ============================================================================
# SECTION 2: BOX PLOTS FOR OUTLIER DETECTION
# ============================================================================
print("\n[3] Creating box plots for outlier detection...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')

# Age boxplot
axes[0, 0].boxplot(df_participants['age'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='skyblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
axes[0, 0].set_ylabel('Age (years)', fontweight='bold')
axes[0, 0].set_title('Age - Outlier Detection')
axes[0, 0].set_xticklabels(['Age'])

# Response Time boxplot
axes[0, 1].boxplot(df_trials['resp.rt'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7),
                   medianprops=dict(color='darkred', linewidth=2))
axes[0, 1].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 1].set_title('Response Time - Outlier Detection')
axes[0, 1].set_xticklabels(['RT'])

# Confidence Rating boxplot
axes[0, 2].boxplot(df_trials['conf_radio.response'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   medianprops=dict(color='darkgreen', linewidth=2))
axes[0, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[0, 2].set_title('Confidence Rating - Outlier Detection')
axes[0, 2].set_xticklabels(['Confidence'])

# Accuracy boxplot by participant
axes[1, 0].boxplot(df_participants['accuracy'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='plum', alpha=0.7),
                   medianprops=dict(color='purple', linewidth=2))
axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
axes[1, 0].set_title('Accuracy - Outlier Detection')
axes[1, 0].set_xticklabels(['Accuracy'])

# Mean RT by participant
axes[1, 1].boxplot(df_participants['rt_mean'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='peachpuff', alpha=0.7),
                   medianprops=dict(color='orange', linewidth=2))
axes[1, 1].set_ylabel('Mean RT (seconds)', fontweight='bold')
axes[1, 1].set_title('Mean RT - Outlier Detection')
axes[1, 1].set_xticklabels(['Mean RT'])

# Mean Confidence by participant
axes[1, 2].boxplot(df_participants['confidence_mean'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='blue', linewidth=2))
axes[1, 2].set_ylabel('Mean Confidence', fontweight='bold')
axes[1, 2].set_title('Mean Confidence - Outlier Detection')
axes[1, 2].set_xticklabels(['Mean Conf'])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_boxplots_outlier_detection.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 02_boxplots_outlier_detection.png")
plt.close()

# ============================================================================
# SECTION 3: VIOLIN PLOTS - COMPARING CONDITIONS
# ============================================================================
print("\n[4] Creating violin plots to compare conditions (AB vs NB)...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Violin Plots - Condition Comparison (AB vs NB)', fontsize=16, fontweight='bold')

# Response Time by Condition
sns.violinplot(data=df_trials, x='condition', y='resp.rt', ax=axes[0, 0], 
               palette=['#FF6B6B', '#4ECDC4'], inner='box')
axes[0, 0].set_xlabel('Condition', fontweight='bold')
axes[0, 0].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 0].set_title('Response Time by Condition')
axes[0, 0].legend(title='Condition', labels=['AB=Abrupt', 'NB=Natural'], loc='upper right')

# Confidence Rating by Condition
sns.violinplot(data=df_trials, x='condition', y='conf_radio.response', ax=axes[0, 1],
               palette=['#FF6B6B', '#4ECDC4'], inner='box')
axes[0, 1].set_xlabel('Condition', fontweight='bold')
axes[0, 1].set_ylabel('Confidence Rating', fontweight='bold')
axes[0, 1].set_title('Confidence Rating by Condition')
axes[0, 1].legend(title='Condition', labels=['AB=Abrupt', 'NB=Natural'], loc='upper right')

# Accuracy by Condition (participant level)
sns.violinplot(data=df_participants, x='condition', y='accuracy', ax=axes[0, 2],
               palette=['#FF6B6B', '#4ECDC4'], inner='box')
axes[0, 2].set_xlabel('Condition', fontweight='bold')
axes[0, 2].set_ylabel('Accuracy (proportion)', fontweight='bold')
axes[0, 2].set_title('Accuracy by Condition')
axes[0, 2].legend(title='Condition', labels=['AB=Abrupt', 'NB=Natural'], loc='lower right')

# Response Time by Response Keys (left/right)
sns.violinplot(data=df_trials, x='resp.keys', y='resp.rt', ax=axes[1, 0],
               palette='Set2', inner='box')
axes[1, 0].set_xlabel('Response Key', fontweight='bold')
axes[1, 0].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Response Time by Key Pressed')

# Confidence by Response Correctness and Condition
for condition in ['AB', 'NB']:
    data_cond = df_trials[df_trials['condition'] == condition]
    for corr in [0, 1]:
        data = data_cond[data_cond['resp.corr'] == corr]
        if len(data) > 0:
            pos = corr + (0.2 if condition == 'AB' else -0.2)
            parts = axes[1, 1].violinplot([data['conf_radio.response'].dropna()], 
                                         positions=[pos], widths=0.35,
                                         showmeans=True, showmedians=True)
            color = '#FF6B6B' if condition == 'AB' else '#4ECDC4'
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
axes[1, 1].set_xlabel('Response Correctness', fontweight='bold')
axes[1, 1].set_ylabel('Confidence Rating', fontweight='bold')
axes[1, 1].set_title('Confidence by Correctness & Condition')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Incorrect', 'Correct'])

# Confidence by Accuracy (simple version)
sns.violinplot(data=df_trials, x='resp.corr', y='conf_radio.response', ax=axes[1, 2],
               palette=['#E74C3C', '#2ECC71'], inner='box')
axes[1, 2].set_xlabel('Response Correctness', fontweight='bold')
axes[1, 2].set_ylabel('Confidence Rating', fontweight='bold')
axes[1, 2].set_title('Confidence by Response Correctness')
axes[1, 2].set_xticklabels(['Incorrect', 'Correct'])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_violin_plots_condition_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 03_violin_plots_condition_comparison.png")
plt.close()

# ============================================================================
# SECTION 4: PIE CHARTS FOR CATEGORICAL VARIABLES
# ============================================================================
print("\n[5] Creating pie charts for categorical variables...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Categorical Variable Distributions', fontsize=16, fontweight='bold')

# Gender distribution
gender_counts = df_participants['gender'].value_counts()
colors_gender = ['#FF9999', '#66B2FF', '#99FF99']
axes[0, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               colors=colors_gender, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[0, 0].set_title('Gender Distribution')

# Handedness distribution
handedness_counts = df_participants['handedness'].value_counts()
colors_hand = ['#FFD700', '#C0C0C0', '#CD7F32']
axes[0, 1].pie(handedness_counts, labels=handedness_counts.index, autopct='%1.1f%%',
               colors=colors_hand, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[0, 1].set_title('Handedness Distribution')

# Vision distribution
vision_counts = df_participants['vision'].value_counts()
colors_vision = ['#90EE90', '#FFB6C1', '#87CEEB']
axes[0, 2].pie(vision_counts, labels=vision_counts.index, autopct='%1.1f%%',
               colors=colors_vision, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[0, 2].set_title('Vision Status Distribution')

# Condition distribution
condition_counts = df_participants['condition'].value_counts()
colors_cond = ['#FF6B6B', '#4ECDC4']
axes[1, 0].pie(condition_counts, labels=condition_counts.index, autopct='%1.1f%%',
               colors=colors_cond, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[1, 0].set_title('Condition Distribution (Participants)')

# Remove the second pie (no stimulus category column)
axes[1, 1].text(0.5, 0.5, 'Stimulus categories\nnot explicitly coded\nin this dataset', 
                ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
axes[1, 1].set_title('Note: Stimulus Info in movie_id')
axes[1, 1].axis('off')

# Overall Accuracy (Correct vs Incorrect) - using trial-level data
acc_trial_counts = df_trials['resp.corr'].value_counts()
colors_acc = ['#E74C3C', '#2ECC71']
labels_acc = ['Incorrect', 'Correct']
axes[1, 2].pie(acc_trial_counts, labels=labels_acc, autopct='%1.1f%%',
               colors=colors_acc, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[1, 2].set_title('Overall Response Correctness (Trials)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_pie_charts_categorical_variables.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 04_pie_charts_categorical_variables.png")
plt.close()

# ============================================================================
# SECTION 5: SCATTER PLOTS - ASSOCIATIONS BETWEEN VARIABLES
# ============================================================================
print("\n[6] Creating scatter plots for variable associations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Scatter Plots - Variable Associations', fontsize=16, fontweight='bold')

# Confidence vs Response Time
for condition in ['AB', 'NB']:
    data = df_trials[df_trials['condition'] == condition]
    axes[0, 0].scatter(data['conf_radio.response'], data['resp.rt'], 
                       alpha=0.3, s=20, label=condition)
axes[0, 0].set_xlabel('Confidence Rating', fontweight='bold')
axes[0, 0].set_ylabel('Response Time (seconds)', fontweight='bold')
axes[0, 0].set_title('Confidence vs Response Time')
axes[0, 0].legend(title='Condition')
axes[0, 0].grid(True, alpha=0.3)

# Confidence vs Accuracy
for condition in ['AB', 'NB']:
    data = df_trials[df_trials['condition'] == condition]
    axes[0, 1].scatter(data['conf_radio.response'], data['resp.corr'], 
                       alpha=0.3, s=20, label=condition)
axes[0, 1].set_xlabel('Confidence Rating', fontweight='bold')
axes[0, 1].set_ylabel('Accuracy (0=incorrect, 1=correct)', fontweight='bold')
axes[0, 1].set_title('Confidence vs Accuracy')
axes[0, 1].legend(title='Condition')
axes[0, 1].grid(True, alpha=0.3)

# Age vs Accuracy (participant level)
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[0, 2].scatter(data['age'], data['accuracy'], 
                       alpha=0.5, s=50, label=condition)
axes[0, 2].set_xlabel('Age (years)', fontweight='bold')
axes[0, 2].set_ylabel('Accuracy (proportion)', fontweight='bold')
axes[0, 2].set_title('Age vs Accuracy')
axes[0, 2].legend(title='Condition')
axes[0, 2].grid(True, alpha=0.3)

# Age vs Mean RT
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[1, 0].scatter(data['age'], data['rt_mean'], 
                       alpha=0.5, s=50, label=condition)
axes[1, 0].set_xlabel('Age (years)', fontweight='bold')
axes[1, 0].set_ylabel('Mean Response Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Age vs Mean Response Time')
axes[1, 0].legend(title='Condition')
axes[1, 0].grid(True, alpha=0.3)

# Accuracy vs Mean Confidence (participant level)
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[1, 1].scatter(data['accuracy'], data['confidence_mean'], 
                       alpha=0.5, s=50, label=condition)
axes[1, 1].set_xlabel('Accuracy (proportion)', fontweight='bold')
axes[1, 1].set_ylabel('Mean Confidence Rating', fontweight='bold')
axes[1, 1].set_title('Accuracy vs Mean Confidence')
axes[1, 1].legend(title='Condition')
axes[1, 1].grid(True, alpha=0.3)

# Mean RT vs Mean Confidence
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[1, 2].scatter(data['rt_mean'], data['confidence_mean'], 
                       alpha=0.5, s=50, label=condition)
axes[1, 2].set_xlabel('Mean Response Time (seconds)', fontweight='bold')
axes[1, 2].set_ylabel('Mean Confidence Rating', fontweight='bold')
axes[1, 2].set_title('Mean RT vs Mean Confidence')
axes[1, 2].legend(title='Condition')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_scatter_plots_associations.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 05_scatter_plots_associations.png")
plt.close()

# ============================================================================
# SECTION 6: BUBBLE PLOTS (Size = additional dimension)
# ============================================================================
print("\n[7] Creating bubble plots with additional dimensions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bubble Plots - Multi-dimensional Associations', fontsize=16, fontweight='bold')

# Bubble plot: Age vs Accuracy (size = mean RT)
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[0].scatter(data['age'], data['accuracy'], 
                    s=data['rt_mean']*100, alpha=0.5, label=condition)
axes[0].set_xlabel('Age (years)', fontweight='bold')
axes[0].set_ylabel('Accuracy (proportion)', fontweight='bold')
axes[0].set_title('Age vs Accuracy (bubble size = Mean RT)')
axes[0].legend(title='Condition')
axes[0].grid(True, alpha=0.3)

# Bubble plot: Accuracy vs Mean Confidence (size = mean RT)
for condition in ['AB', 'NB']:
    data = df_participants[df_participants['condition'] == condition]
    axes[1].scatter(data['accuracy'], data['confidence_mean'], 
                    s=data['rt_mean']*100, alpha=0.5, label=condition)
axes[1].set_xlabel('Accuracy (proportion)', fontweight='bold')
axes[1].set_ylabel('Mean Confidence Rating', fontweight='bold')
axes[1].set_title('Accuracy vs Confidence (bubble size = Mean RT)')
axes[1].legend(title='Condition')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_bubble_plots.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 06_bubble_plots.png")
plt.close()

# ============================================================================
# SECTION 7: CORRELATION HEATMAPS
# ============================================================================
print("\n[8] Creating correlation heatmaps...")

# Participant-level correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Correlation Heatmaps', fontsize=16, fontweight='bold')

# Select numeric columns for correlation
participant_numeric = df_participants[['age', 'accuracy', 'rt_mean', 'confidence_mean']].copy()
corr_matrix_participant = participant_numeric.corr()

# Heatmap 1: Participant-level correlations
sns.heatmap(corr_matrix_participant, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=1, linecolor='white')
axes[0].set_title('Participant-Level Variable Correlations', fontweight='bold')
axes[0].set_xticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=45, ha='right')
axes[0].set_yticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=0)

# Trial-level correlations
trial_numeric = df_trials[['resp.corr', 'resp.rt', 'conf_radio.response']].copy()
corr_matrix_trial = trial_numeric.corr()

# Heatmap 2: Trial-level correlations
sns.heatmap(corr_matrix_trial, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=1, linecolor='white')
axes[1].set_title('Trial-Level Variable Correlations', fontweight='bold')
axes[1].set_xticklabels(['Accuracy', 'Response Time', 'Confidence'], rotation=45, ha='right')
axes[1].set_yticklabels(['Accuracy', 'Response Time', 'Confidence'], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 07_correlation_heatmaps.png")
plt.close()

# ============================================================================
# SECTION 8: CONDITION-SPECIFIC CORRELATION HEATMAPS
# ============================================================================
print("\n[9] Creating condition-specific correlation heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Correlation Heatmaps by Condition', fontsize=16, fontweight='bold')

# AB Condition
ab_numeric = df_participants[df_participants['condition'] == 'AB'][['age', 'accuracy', 'rt_mean', 'confidence_mean']].copy()
corr_ab = ab_numeric.corr()

sns.heatmap(corr_ab, annot=True, fmt='.3f', cmap='Reds', 
            center=0, square=True, ax=axes[0], cbar_kws={'label': 'Correlation'},
            linewidths=1, linecolor='white', vmin=-1, vmax=1)
axes[0].set_title('AB Condition (Abrupt Boundaries)', fontweight='bold')
axes[0].set_xticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=45, ha='right')
axes[0].set_yticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=0)

# NB Condition
nb_numeric = df_participants[df_participants['condition'] == 'NB'][['age', 'accuracy', 'rt_mean', 'confidence_mean']].copy()
corr_nb = nb_numeric.corr()

sns.heatmap(corr_nb, annot=True, fmt='.3f', cmap='Blues', 
            center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation'},
            linewidths=1, linecolor='white', vmin=-1, vmax=1)
axes[1].set_title('NB Condition (Natural Boundaries)', fontweight='bold')
axes[1].set_xticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=45, ha='right')
axes[1].set_yticklabels(['Age', 'Accuracy', 'Mean RT', 'Mean Confidence'], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_correlation_heatmaps_by_condition.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 08_correlation_heatmaps_by_condition.png")
plt.close()

# ============================================================================
# SECTION 9: NORMALITY TESTING - Q-Q PLOTS
# ============================================================================
print("\n[10] Creating Q-Q plots for normality assessment...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Q-Q Plots - Normality Assessment', fontsize=16, fontweight='bold')

# Age Q-Q plot
stats.probplot(df_participants['age'].dropna(), dist="norm", plot=axes[0, 0])
axes[0, 0].set_title('Age - Q-Q Plot', fontweight='bold')
axes[0, 0].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[0, 0].set_ylabel('Sample Quantiles', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Response Time Q-Q plot
stats.probplot(df_trials['resp.rt'].dropna(), dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Response Time - Q-Q Plot', fontweight='bold')
axes[0, 1].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[0, 1].set_ylabel('Sample Quantiles', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Confidence Rating Q-Q plot
stats.probplot(df_trials['conf_radio.response'].dropna(), dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Confidence Rating - Q-Q Plot', fontweight='bold')
axes[0, 2].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[0, 2].set_ylabel('Sample Quantiles', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Accuracy Q-Q plot
stats.probplot(df_participants['accuracy'].dropna(), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Accuracy - Q-Q Plot', fontweight='bold')
axes[1, 0].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[1, 0].set_ylabel('Sample Quantiles', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Mean RT Q-Q plot
stats.probplot(df_participants['rt_mean'].dropna(), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Mean RT - Q-Q Plot', fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[1, 1].set_ylabel('Sample Quantiles', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Mean Confidence Q-Q plot
stats.probplot(df_participants['confidence_mean'].dropna(), dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('Mean Confidence - Q-Q Plot', fontweight='bold')
axes[1, 2].set_xlabel('Theoretical Quantiles', fontweight='bold')
axes[1, 2].set_ylabel('Sample Quantiles', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_qq_plots_normality.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 09_qq_plots_normality.png")
plt.close()

# ============================================================================
# SECTION 10: STATISTICAL TESTS FOR NORMALITY
# ============================================================================
print("\n[11] Performing statistical tests for normality...")

print("\n" + "="*70)
print("NORMALITY TEST SELECTION RATIONALE")
print("="*70)
print("""
We use THREE complementary tests to assess normality comprehensively:

1. SHAPIRO-WILK TEST
   Rationale: Considered the most powerful test for normality
   - Best for: Small to medium samples (n < 5000)
   - Pros: Highest statistical power to detect non-normality
   - Cons: Can be overly sensitive with large samples
   - Null hypothesis: Data is normally distributed (p > 0.05 = normal)
   
2. KOLMOGOROV-SMIRNOV TEST
   Rationale: Tests goodness-of-fit to theoretical normal distribution
   - Best for: Continuous distributions, detecting specific departures
   - Pros: Non-parametric, tests overall distribution shape
   - Cons: Less powerful than Shapiro-Wilk for small samples
   - Null hypothesis: Data follows specified distribution (p > 0.05 = normal)
   
3. D'AGOSTINO-PEARSON TEST
   Rationale: Tests both skewness AND kurtosis simultaneously
   - Best for: Identifying specific types of non-normality
   - Pros: Reveals whether skewness or kurtosis causes non-normality
   - Cons: Requires n ≥ 8, works best with larger samples
   - Null hypothesis: Data is normally distributed (p > 0.05 = normal)

WHY USE MULTIPLE TESTS?
- Each test is sensitive to different aspects of non-normality
- Convergent evidence: If all tests agree, we have stronger conclusions
- Divergent results indicate specific types of departures from normality
- Provides comprehensive assessment rather than relying on single method

INTERPRETATION GUIDELINES:
- If ALL tests show p > 0.05: Strong evidence for normality
- If ALL tests show p < 0.05: Strong evidence against normality
- If tests disagree: Examine Q-Q plots and descriptive statistics
- For large samples: Tests may be overly sensitive; also check effect sizes
""")
print("="*70)

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
    # Shapiro-Wilk test (most powerful, best for n < 5000)
    # Chosen because: Gold standard for normality testing with small-medium samples
    if len(data) < 5000:
        shapiro_stat, shapiro_p = shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Kolmogorov-Smirnov test
    # Chosen because: Tests overall distribution fit, complements Shapiro-Wilk
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    
    # D'Agostino and Pearson's test
    # Chosen because: Specifically tests skewness AND kurtosis components
    dagostino_stat, dagostino_p = normaltest(data)
    
    normality_results.append({
        'Variable': var_name,
        'N': len(data),
        'Mean': data.mean(),
        'Std': data.std(),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Shapiro_W': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'KS_stat': ks_stat,
        'KS_p': ks_p,
        'DAgostino_stat': dagostino_stat,
        'DAgostino_p': dagostino_p
    })

df_normality = pd.DataFrame(normality_results)

# Save normality test results
df_normality.to_csv(OUTPUT_DIR / 'normality_test_results.csv', index=False)
print("   ✓ Saved: normality_test_results.csv")

# Create a summary table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_data = []
table_data.append(['Variable', 'N', 'Mean', 'Std', 'Skew', 'Kurt', 
                   'Shapiro p', 'KS p', 'D\'Agostino p', 'Normal?'])

for _, row in df_normality.iterrows():
    # Determine if normal (p > 0.05 for all tests)
    is_normal = 'Yes' if (row['Shapiro_p'] > 0.05 and row['KS_p'] > 0.05 and row['DAgostino_p'] > 0.05) else 'No'
    
    table_data.append([
        row['Variable'],
        f"{int(row['N'])}",
        f"{row['Mean']:.2f}",
        f"{row['Std']:.2f}",
        f"{row['Skewness']:.2f}",
        f"{row['Kurtosis']:.2f}",
        f"{row['Shapiro_p']:.4f}" if not pd.isna(row['Shapiro_p']) else 'N/A',
        f"{row['KS_p']:.4f}",
        f"{row['DAgostino_p']:.4f}",
        is_normal
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.2, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(10):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(10):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

plt.title('Normality Test Results Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig(OUTPUT_DIR / '10_normality_test_summary.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 10_normality_test_summary.png")
plt.close()

# Print summary to console
print("\n" + "="*70)
print("NORMALITY TEST RESULTS")
print("="*70)
for _, row in df_normality.iterrows():
    print(f"\n{row['Variable']}:")
    print(f"  N = {int(row['N'])}, Mean = {row['Mean']:.3f}, Std = {row['Std']:.3f}")
    print(f"  Skewness = {row['Skewness']:.3f}, Kurtosis = {row['Kurtosis']:.3f}")
    
    # Interpret skewness
    if abs(row['Skewness']) < 0.5:
        skew_interp = "approximately symmetric"
    elif row['Skewness'] > 0:
        skew_interp = "positively skewed (right tail)"
    else:
        skew_interp = "negatively skewed (left tail)"
    
    # Interpret kurtosis
    if abs(row['Kurtosis']) < 0.5:
        kurt_interp = "mesokurtic (normal-like tails)"
    elif row['Kurtosis'] > 0:
        kurt_interp = "leptokurtic (heavy tails, peaked)"
    else:
        kurt_interp = "platykurtic (light tails, flat)"
    
    print(f"  Distribution shape: {skew_interp}, {kurt_interp}")
    
    if not pd.isna(row['Shapiro_p']):
        print(f"  Shapiro-Wilk: W = {row['Shapiro_W']:.4f}, p = {row['Shapiro_p']:.4f}")
    print(f"  Kolmogorov-Smirnov: D = {row['KS_stat']:.4f}, p = {row['KS_p']:.4f}")
    print(f"  D'Agostino-Pearson: stat = {row['DAgostino_stat']:.4f}, p = {row['DAgostino_p']:.4f}")
    
    # Detailed interpretation
    if row['Shapiro_p'] > 0.05 and row['KS_p'] > 0.05 and row['DAgostino_p'] > 0.05:
        print("  ✓ NORMALLY DISTRIBUTED (all tests p > 0.05)")
        print("  → Recommended tests: Parametric (t-test, ANOVA, Pearson correlation)")
    else:
        print("  ✗ NOT NORMALLY DISTRIBUTED (at least one test p < 0.05)")
        
        # Identify which tests failed
        failed_tests = []
        if not pd.isna(row['Shapiro_p']) and row['Shapiro_p'] <= 0.05:
            failed_tests.append("Shapiro-Wilk")
        if row['KS_p'] <= 0.05:
            failed_tests.append("K-S")
        if row['DAgostino_p'] <= 0.05:
            failed_tests.append("D'Agostino")
        
        print(f"  → Failed: {', '.join(failed_tests)}")
        print("  → Recommended tests: Non-parametric (Mann-Whitney U, Kruskal-Wallis, Spearman)")
        
        # Suggest transformations if appropriate
        if row['Skewness'] > 1:
            print("  → Consider: Log transformation to reduce positive skew")
        elif row['Skewness'] < -1:
            print("  → Consider: Square transformation or reflection + log")
        
        if abs(row['Kurtosis']) > 2:
            print("  → Note: Extreme values present; check for outliers")


