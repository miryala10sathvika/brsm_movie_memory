
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BRSM MOVIE MEMORY - DATA CLEANING & DESCRIPTIVE STATISTICS")
print("="*80)


BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "final_cleaned_data"
STATS_DIR = BASE_DIR / "descriptive_statistics"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
STATS_DIR.mkdir(exist_ok=True, parents=True)

# Load the main datasets
df_summary = pd.read_csv(DATA_DIR / "participant_summary_statistics.csv")
df_recognition = pd.read_csv(DATA_DIR / "master_recognition_data.csv")
df_demographics = pd.read_csv(DATA_DIR / "master_demographics.csv")

print(f"Loaded participant summary: {len(df_summary)} rows")
print(f"Loaded recognition data: {len(df_recognition)} trials")
print(f"Loaded demographics: {len(df_demographics)} participants")



# Initial counts
print(f"\nInitial data:")
print(f"  Summary participants: {len(df_summary)}")
print(f"  Recognition trials: {len(df_recognition)}")

# Keep all trial rows; only standardize/parse values so we do not lose data.
df_recognition_clean = df_recognition.copy()

for col in ['resp.corr', 'resp.rt', 'conf_radio.response']:
    if col in df_recognition_clean.columns:
        df_recognition_clean[col] = pd.to_numeric(df_recognition_clean[col], errors='coerce')

print("\n2.1: Missingness profile (no row deletion)...")
trial_key_vars = ['resp.corr', 'resp.rt', 'conf_radio.response']
for var in trial_key_vars:
    if var in df_recognition_clean.columns:
        missing = df_recognition_clean[var].isna().sum()
        pct = 100 * missing / len(df_recognition_clean)
        print(f"  {var}: {missing}/{len(df_recognition_clean)} missing ({pct:.1f}%)")

print("\n2.2: Rebuilding participant summary from trial-level data...")
df_summary_clean = (
    df_recognition_clean
    .groupby(['participant_id', 'condition'], as_index=False)
    .agg(
        n_trials=('movie_id', 'count'),
        n_correct=('resp.corr', 'sum'),
        accuracy=('resp.corr', 'mean'),
        rt_mean=('resp.rt', 'mean'),
        rt_median=('resp.rt', 'median'),
        rt_std=('resp.rt', 'std'),
        confidence_mean=('conf_radio.response', 'mean'),
        confidence_std=('conf_radio.response', 'std')
    )
)
print(f"  Built summary for {len(df_summary_clean)} participant-condition rows")

print("\n2.3: Merging demographics without dropping participants...")
demo = df_demographics.copy()
demo.columns = [c.strip() for c in demo.columns]

rename_map = {
    'Sub ID': 'participant_id',
    'Age': 'age',
    'Gender': 'gender',
    'Gender ': 'gender',
    'Handedness': 'handedness',
    'Vision': 'vision'
}
demo = demo.rename(columns={k: v for k, v in rename_map.items() if k in demo.columns})

for c in ['participant_id', 'gender', 'handedness', 'vision']:
    if c in demo.columns:
        demo[c] = demo[c].astype(str).str.strip()
        demo.loc[demo[c].isin(['nan', 'None', '']), c] = np.nan

if 'age' in demo.columns:
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')

demo['id_key'] = demo['participant_id'].str.lower().str.strip()
df_summary_clean['id_key'] = df_summary_clean['participant_id'].astype(str).str.lower().str.strip()

demo_keep = [c for c in ['id_key', 'age', 'gender', 'handedness', 'vision'] if c in demo.columns]
df_summary_clean = df_summary_clean.merge(demo[demo_keep], on='id_key', how='left')

print("\n2.4: Elegant missing-data handling for demographics (impute + flags)...")
for col in ['age', 'gender', 'handedness', 'vision']:
    df_summary_clean[f'{col}_was_missing'] = df_summary_clean[col].isna()

# Condition-wise imputation to preserve sample size while avoiding cross-condition leakage.
age_med = df_summary_clean.groupby('condition')['age'].transform('median')
df_summary_clean['age'] = df_summary_clean['age'].fillna(age_med)

for col in ['gender', 'handedness', 'vision']:
    mode_by_cond = (
        df_summary_clean.groupby('condition')[col]
        .transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    )
    df_summary_clean[col] = df_summary_clean[col].fillna(mode_by_cond)

for col in ['age', 'gender', 'handedness', 'vision']:
    df_summary_clean[f'{col}_imputed'] = df_summary_clean[f'{col}_was_missing'] & df_summary_clean[col].notna()

test_patterns = ['test', 'aru', 'hello', 'suub', 'subh']
df_summary_clean['id_suspect_nonstandard'] = df_summary_clean['participant_id'].str.contains(
    '|'.join(test_patterns), case=False, na=False
)

print(f"  Participants kept: {len(df_summary_clean)}")
print(f"  Trials kept: {len(df_recognition_clean)}")
print(
    "  Imputed demographics counts: "
    f"age={df_summary_clean['age_imputed'].sum()}, "
    f"gender={df_summary_clean['gender_imputed'].sum()}, "
    f"handedness={df_summary_clean['handedness_imputed'].sum()}, "
    f"vision={df_summary_clean['vision_imputed'].sum()}"
)

df_summary_clean = df_summary_clean.drop(columns=['id_key'])

print(f"\n  Final trial count: {len(df_recognition_clean)} trials")



print("\n" + "="*80)
print("STEP 3: Data quality checks...")
print("="*80)

# Check accuracy range (should be 0-1)
print("\n3.1: Checking accuracy range...")
acc_out_of_range = df_summary_clean[
    (df_summary_clean['accuracy'] < 0) | (df_summary_clean['accuracy'] > 1)
]
if len(acc_out_of_range) > 0:
    print(f"  ⚠ {len(acc_out_of_range)} participants with accuracy out of range [0,1]")
    print(f"    Range: {acc_out_of_range['accuracy'].min():.3f} - {acc_out_of_range['accuracy'].max():.3f}")
else:
    print(f"  ✓ All accuracy values in valid range")
    print(f"    Range: {df_summary_clean['accuracy'].min():.3f} - {df_summary_clean['accuracy'].max():.3f}")

# Check RT outliers (extremely fast or slow)
print("\n3.2: Checking response time outliers...")
rt_q1 = df_summary_clean['rt_mean'].quantile(0.25)
rt_q3 = df_summary_clean['rt_mean'].quantile(0.75)
rt_iqr = rt_q3 - rt_q1
rt_lower = rt_q1 - 3 * rt_iqr  # Using 3*IQR for extreme outliers
rt_upper = rt_q3 + 3 * rt_iqr

rt_outliers = df_summary_clean[
    (df_summary_clean['rt_mean'] < rt_lower) | (df_summary_clean['rt_mean'] > rt_upper)
]
print(f"  IQR method (3×IQR): {rt_lower:.2f}s - {rt_upper:.2f}s")
print(f"  Found {len(rt_outliers)} extreme RT outliers")
if len(rt_outliers) > 0:
    print(f"    Mean RT range of outliers: {rt_outliers['rt_mean'].min():.2f}s - {rt_outliers['rt_mean'].max():.2f}s")
    print(f"    Participant IDs: {', '.join(rt_outliers['participant_id'].tolist()[:5])}")
    # Keep outliers but flag them
    df_summary_clean['rt_outlier'] = (
        (df_summary_clean['rt_mean'] < rt_lower) | (df_summary_clean['rt_mean'] > rt_upper)
    )
else:
    df_summary_clean['rt_outlier'] = False

# Check confidence range (should be 1-5)
print("\n3.3: Checking confidence rating range...")
conf_min = df_summary_clean['confidence_mean'].min()
conf_max = df_summary_clean['confidence_mean'].max()
print(f"  Confidence range: {conf_min:.2f} - {conf_max:.2f}")
if conf_min < 1 or conf_max > 5:
    print(f"  ⚠ Some confidence ratings outside expected range [1,5]")
else:
    print(f"  ✓ All confidence ratings in valid range")

# Check trials per participant
print("\n3.4: Checking trial counts per participant...")
trials_per_participant = df_recognition_clean.groupby('participant_id').size()
print(f"  Expected: 40 trials per participant")
print(f"  Actual range: {trials_per_participant.min()} - {trials_per_participant.max()}")
print(f"  Mode: {trials_per_participant.mode()[0]} trials")

unusual_trial_counts = trials_per_participant[trials_per_participant != 40]
if len(unusual_trial_counts) > 0:
    print(f"  ⚠ {len(unusual_trial_counts)} participants with ≠40 trials:")
    for pid, count in list(unusual_trial_counts.items())[:5]:
        print(f"    - {pid}: {count} trials")
else:
    print(f"  ✓ All participants have exactly 40 trials")


print("\n" + "="*80)
print("STEP 4: Saving final cleaned data...")
print("="*80)

# Save cleaned datasets
df_summary_clean.to_csv(OUTPUT_DIR / "participants_final_clean.csv", index=False)
df_recognition_clean.to_csv(OUTPUT_DIR / "trials_final_clean.csv", index=False)

# Create a merged dataset for analysis
df_merged = df_recognition_clean.merge(
    df_summary_clean[['participant_id', 'age', 'gender', 'handedness', 'vision']], 
    on='participant_id', 
    how='left'
)
df_merged.to_csv(OUTPUT_DIR / "trials_with_demographics_final.csv", index=False)

print(f"  ✓ Saved participants_final_clean.csv ({len(df_summary_clean)} rows)")
print(f"  ✓ Saved trials_final_clean.csv ({len(df_recognition_clean)} rows)")
print(f"  ✓ Saved trials_with_demographics_final.csv ({len(df_merged)} rows)")


print("\n" + "="*80)
print("STEP 5: DESCRIPTIVE STATISTICS - Sample Characteristics")
print("="*80)

stats_report = []
stats_report.append("="*80)
stats_report.append("DESCRIPTIVE STATISTICS REPORT")
stats_report.append("BRSM Movie Memory Experiment")
stats_report.append("="*80)
stats_report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Overall sample
stats_report.append("\n" + "="*80)
stats_report.append("1. SAMPLE CHARACTERISTICS")
stats_report.append("="*80)

stats_report.append(f"\nTotal Participants: {len(df_summary_clean)}")
stats_report.append(f"Total Trials: {len(df_recognition_clean)}")
stats_report.append(f"Average Trials per Participant: {len(df_recognition_clean)/len(df_summary_clean):.1f}")

# By condition
stats_report.append("\n--- By Condition ---")
cond_counts = df_summary_clean['condition'].value_counts()
for cond, count in cond_counts.items():
    pct = count / len(df_summary_clean) * 100
    stats_report.append(f"  {cond}: {count} participants ({pct:.1f}%)")

# Demographics
stats_report.append("\n--- Demographics ---")

# Age
age_stats = df_summary_clean['age'].describe()
stats_report.append(f"\nAge:")
stats_report.append(f"  Mean ± SD: {age_stats['mean']:.1f} ± {age_stats['std']:.1f} years")
stats_report.append(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
stats_report.append(f"  Median: {age_stats['50%']:.0f} years")

# Age by condition
stats_report.append(f"\n  By Condition:")
for cond in df_summary_clean['condition'].unique():
    cond_age = df_summary_clean[df_summary_clean['condition']==cond]['age']
    stats_report.append(f"    {cond}: {cond_age.mean():.1f} ± {cond_age.std():.1f} years")

# Gender
stats_report.append(f"\nGender:")
gender_counts = df_summary_clean['gender'].value_counts()
for gender, count in gender_counts.items():
    pct = count / len(df_summary_clean) * 100
    stats_report.append(f"  {gender}: {count} ({pct:.1f}%)")

# Gender by condition
stats_report.append(f"\n  By Condition:")
for cond in df_summary_clean['condition'].unique():
    cond_gender = df_summary_clean[df_summary_clean['condition']==cond]['gender'].value_counts()
    stats_report.append(f"    {cond}:")
    for gender, count in cond_gender.items():
        stats_report.append(f"      {gender}: {count}")

# Handedness
stats_report.append(f"\nHandedness:")
hand_counts = df_summary_clean['handedness'].value_counts()
for hand, count in hand_counts.items():
    pct = count / len(df_summary_clean) * 100
    stats_report.append(f"  {hand}: {count} ({pct:.1f}%)")

# Vision
stats_report.append(f"\nVision:")
vision_counts = df_summary_clean['vision'].value_counts()
for vis, count in vision_counts.items():
    pct = count / len(df_summary_clean) * 100
    stats_report.append(f"  {vis}: {count} ({pct:.1f}%)")


stats_report.append("\n" + "="*80)
stats_report.append("2. PERFORMANCE MEASURES")
stats_report.append("="*80)

# Overall accuracy
stats_report.append("\n--- Recognition Accuracy ---")
acc_stats = df_summary_clean['accuracy'].describe()
stats_report.append(f"\nOverall:")
stats_report.append(f"  Mean ± SD: {acc_stats['mean']:.3f} ± {acc_stats['std']:.3f} ({acc_stats['mean']*100:.1f}% ± {acc_stats['std']*100:.1f}%)")
stats_report.append(f"  Median: {acc_stats['50%']:.3f} ({acc_stats['50%']*100:.1f}%)")
stats_report.append(f"  Range: {acc_stats['min']:.3f} - {acc_stats['max']:.3f} ({acc_stats['min']*100:.1f}% - {acc_stats['max']*100:.1f}%)")
stats_report.append(f"  IQR: {acc_stats['25%']:.3f} - {acc_stats['75%']:.3f}")

# Accuracy by condition
stats_report.append(f"\nBy Condition:")
for cond in sorted(df_summary_clean['condition'].unique()):
    cond_acc = df_summary_clean[df_summary_clean['condition']==cond]['accuracy']
    stats_report.append(f"  {cond}:")
    stats_report.append(f"    Mean ± SD: {cond_acc.mean():.3f} ± {cond_acc.std():.3f} ({cond_acc.mean()*100:.1f}% ± {cond_acc.std()*100:.1f}%)")
    stats_report.append(f"    Median: {cond_acc.median():.3f} ({cond_acc.median()*100:.1f}%)")
    stats_report.append(f"    Range: {cond_acc.min():.3f} - {cond_acc.max():.3f}")
    stats_report.append(f"    N: {len(cond_acc)}")

# Response Time
stats_report.append("\n--- Response Time (seconds) ---")
rt_stats = df_summary_clean['rt_mean'].describe()
stats_report.append(f"\nOverall:")
stats_report.append(f"  Mean ± SD: {rt_stats['mean']:.3f} ± {rt_stats['std']:.3f} seconds")
stats_report.append(f"  Median: {rt_stats['50%']:.3f} seconds")
stats_report.append(f"  Range: {rt_stats['min']:.3f} - {rt_stats['max']:.3f} seconds")
stats_report.append(f"  IQR: {rt_stats['25%']:.3f} - {rt_stats['75%']:.3f}")

# RT by condition
stats_report.append(f"\nBy Condition:")
for cond in sorted(df_summary_clean['condition'].unique()):
    cond_rt = df_summary_clean[df_summary_clean['condition']==cond]['rt_mean']
    stats_report.append(f"  {cond}:")
    stats_report.append(f"    Mean ± SD: {cond_rt.mean():.3f} ± {cond_rt.std():.3f} seconds")
    stats_report.append(f"    Median: {cond_rt.median():.3f} seconds")
    stats_report.append(f"    Range: {cond_rt.min():.3f} - {cond_rt.max():.3f} seconds")
    stats_report.append(f"    N: {len(cond_rt)}")

# Confidence ratings
stats_report.append("\n--- Confidence Ratings (1-5 scale) ---")
conf_stats = df_summary_clean['confidence_mean'].describe()
stats_report.append(f"\nOverall:")
stats_report.append(f"  Mean ± SD: {conf_stats['mean']:.3f} ± {conf_stats['std']:.3f}")
stats_report.append(f"  Median: {conf_stats['50%']:.3f}")
stats_report.append(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
stats_report.append(f"  IQR: {conf_stats['25%']:.3f} - {conf_stats['75%']:.3f}")

# Confidence by condition
stats_report.append(f"\nBy Condition:")
for cond in sorted(df_summary_clean['condition'].unique()):
    cond_conf = df_summary_clean[df_summary_clean['condition']==cond]['confidence_mean']
    stats_report.append(f"  {cond}:")
    stats_report.append(f"    Mean ± SD: {cond_conf.mean():.3f} ± {cond_conf.std():.3f}")
    stats_report.append(f"    Median: {cond_conf.median():.3f}")
    stats_report.append(f"    Range: {cond_conf.min():.3f} - {cond_conf.max():.3f}")
    stats_report.append(f"    N: {len(cond_conf)}")


stats_report.append("\n" + "="*80)
stats_report.append("3. TRIAL-LEVEL STATISTICS")
stats_report.append("="*80)

# Overall trial accuracy
trial_acc = df_recognition_clean['resp.corr'].mean()
stats_report.append(f"\nTrial-level accuracy: {trial_acc:.3f} ({trial_acc*100:.1f}%)")
stats_report.append(f"  Correct trials: {df_recognition_clean['resp.corr'].sum():.0f}")
stats_report.append(f"  Incorrect trials: {(df_recognition_clean['resp.corr'] == 0).sum():.0f}")

# By condition
stats_report.append(f"\nBy Condition:")
for cond in sorted(df_recognition_clean['condition'].unique()):
    cond_trials = df_recognition_clean[df_recognition_clean['condition']==cond]
    cond_trial_acc = cond_trials['resp.corr'].mean()
    stats_report.append(f"  {cond}: {cond_trial_acc:.3f} ({cond_trial_acc*100:.1f}%)")
    stats_report.append(f"    Correct: {cond_trials['resp.corr'].sum():.0f} / {len(cond_trials)}")

# Response distribution (left vs right)
stats_report.append(f"\n--- Response Distribution ---")
resp_dist = df_recognition_clean['resp.keys'].value_counts()
stats_report.append(f"\nOverall:")
for resp, count in resp_dist.items():
    pct = count / len(df_recognition_clean) * 100
    stats_report.append(f"  '{resp}': {count} ({pct:.1f}%)")


print("\n" + "="*80)
print("STEP 6: Creating summary tables...")
print("="*80)

# Table 1: Overall summary by condition
summary_by_condition = df_summary_clean.groupby('condition').agg({
    'participant_id': 'count',
    'age': ['mean', 'std'],
    'accuracy': ['mean', 'std', 'min', 'max'],
    'rt_mean': ['mean', 'std', 'min', 'max'],
    'confidence_mean': ['mean', 'std']
}).round(3)

summary_by_condition.columns = ['_'.join(col).strip('_') for col in summary_by_condition.columns.values]
summary_by_condition = summary_by_condition.rename(columns={'participant_id_count': 'N'})
summary_by_condition.to_csv(STATS_DIR / "summary_by_condition.csv")
print(f"  ✓ Saved summary_by_condition.csv")

# Table 2: Gender distribution by condition
gender_by_condition = pd.crosstab(
    df_summary_clean['condition'], 
    df_summary_clean['gender'], 
    margins=True
)
gender_by_condition.to_csv(STATS_DIR / "gender_by_condition.csv")
print(f"  ✓ Saved gender_by_condition.csv")

# Table 3: Individual participant summary (sorted by accuracy)
participant_table = df_summary_clean[[
    'participant_id', 'condition', 'age', 'gender', 
    'n_trials', 'accuracy', 'rt_mean', 'confidence_mean'
]].sort_values('accuracy', ascending=False)
participant_table.to_csv(STATS_DIR / "all_participants_summary.csv", index=False)
print(f"  ✓ Saved all_participants_summary.csv")

# Table 4: Top and bottom performers
top_performers = df_summary_clean.nlargest(10, 'accuracy')[[
    'participant_id', 'condition', 'accuracy', 'rt_mean', 'confidence_mean'
]]
top_performers.to_csv(STATS_DIR / "top_10_performers.csv", index=False)

bottom_performers = df_summary_clean.nsmallest(10, 'accuracy')[[
    'participant_id', 'condition', 'accuracy', 'rt_mean', 'confidence_mean'
]]
bottom_performers.to_csv(STATS_DIR / "bottom_10_performers.csv", index=False)
print(f"  ✓ Saved top/bottom performers tables")


stats_report.append("\n" + "="*80)
stats_report.append("END OF REPORT")
stats_report.append("="*80)

report_text = "\n".join(stats_report)
with open(STATS_DIR / "descriptive_statistics_report.txt", 'w') as f:
    f.write(report_text)

print(f"\n  ✓ Saved descriptive_statistics_report.txt")

# Print report to console
print("\n" + "="*80)
print(report_text)
print("="*80)


print("\n" + "="*80)
print("✅ DATA CLEANING & DESCRIPTIVE STATISTICS COMPLETE!")
print("="*80)

print("\n📊 FINAL CLEAN DATASET:")
print(f"   ✓ {len(df_summary_clean)} participants")
print(f"   ✓ {len(df_recognition_clean)} trials")
print(f"   ✓ {df_summary_clean['condition'].value_counts()['AB']} AB participants, {df_summary_clean['condition'].value_counts()['NB']} NB participants")

