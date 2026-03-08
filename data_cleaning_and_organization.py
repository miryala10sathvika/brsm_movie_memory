import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = Path(".")  # Current directory
RAW_DATA_DIR = BASE_DIR / "BRSM data csv"
OUTPUT_DIR = BASE_DIR / "cleaned_data"
REPORTS_DIR = BASE_DIR / "data_reports"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "individual_cleaned").mkdir(exist_ok=True, parents=True)


# Load movie information
abrupt_movies = pd.read_csv(BASE_DIR / "abruptmovies.csv")
natural_movies = pd.read_csv(BASE_DIR / "naturalmovies.csv")
target_lures = pd.read_csv(BASE_DIR / "target_and_lures.csv")

print(f"Abrupt movies: {len(abrupt_movies)} entries")
print(f"Natural movies: {len(natural_movies)} entries")
print(f"Target & Lures: {len(target_lures)} pairs")

# Standardize paths (convert backslashes to forward slashes)
abrupt_movies['path'] = abrupt_movies['path'].str.replace('\\', '/')
if 'target_img' in target_lures.columns:
    target_lures['target_img'] = target_lures['target_img'].str.replace('\\', '/')
    target_lures['lure_img'] = target_lures['lure_img'].str.replace('\\', '/')

# Save cleaned stimulus files
abrupt_movies.to_csv(OUTPUT_DIR / "abrupt_movies_clean.csv", index=False)
natural_movies.to_csv(OUTPUT_DIR / "natural_movies_clean.csv", index=False)
target_lures.to_csv(OUTPUT_DIR / "target_lures_clean.csv", index=False)






# Get all CSV files
all_files = list(RAW_DATA_DIR.glob("*.csv"))
print(f"  Found {len(all_files)} total files")

# Categorize files
ab_files = []
nb_files = []
problem_files = []

for file in all_files:
    filename = file.name
    if '_AB_' in filename or '_AB ' in filename:
        ab_files.append(file)
    elif '_NB_' in filename or '_NB ' in filename:
        nb_files.append(file)
    else:
        problem_files.append(file)

print(f"AB (Abrupt) participants: {len(ab_files)}")
print(f"NB (Natural) participants: {len(nb_files)}")
if problem_files:
    print(f"  ⚠ Files with naming issues: {len(problem_files)}")
    for pf in problem_files:
        print(f"      - {pf.name}")



# Core columns needed for analysis
DEMOGRAPHIC_COLS = ['age', 'gender', 'handedness', 'vision', 'caffeine_2h', 'alcohol_smoke_12h']

VIDEO_PHASE_COLS = [
    'path', 'duration', 'is_repeat', 'movie_id',
    'vigilance_pressed', 'vigilance_correct'
]

RECOGNITION_COLS = [
    'movie_id', 'target_img', 'lure_img',
    'resp.keys', 'resp.corr', 'resp.rt',
    'conf_radio.response', 'conf_radio.rt'
]

METADATA_COLS = [
    'participant', 'session', 'date', 'expName', 'frameRate'
]

print("  ✓ Defined column sets for extraction\n")


def extract_participant_id(filename):
    """Extract standardized participant ID from filename"""
    # Handle cases like sub133_AB _ (with extra space)
    parts = filename.split('_')
    if len(parts) >= 2:
        sub_id = parts[0]
        condition = parts[1].strip()  # Remove spaces
        return f"{sub_id}_{condition}"
    return filename.split('_recognitionstage')[0]

def process_participant_file(filepath, condition):
    """Process a single participant file and extract key data"""
    try:
        # Read the file
        df = pd.read_csv(filepath, low_memory=False)
        
        # Extract participant ID
        if 'participant' in df.columns and df['participant'].notna().any():
            participant_id = df['participant'].dropna().iloc[0]
        else:
            participant_id = extract_participant_id(filepath.name)
        
        # Standardize participant ID (ensure consistent format)
        participant_id = participant_id.strip()
        
        # Demographics will be merged from separate file later
        # Not extracted from individual trial files (they only have question labels)
        demographics = {col: None for col in DEMOGRAPHIC_COLS}
        
        # Extract metadata
        metadata = {'participant_id': participant_id, 'condition': condition}
        for col in METADATA_COLS:
            if col in df.columns and df[col].notna().any():
                metadata[col] = df[col].dropna().iloc[0]
        
        # Filter to recognition trials (rows with movie_id)
        recognition_data = df[df['movie_id'].notna()].copy()
        
        if len(recognition_data) == 0:
            print(f"  ⚠ No recognition data found for {participant_id}")
            return None, None, None
        
        # Add participant ID and condition to recognition data
        recognition_data['participant_id'] = participant_id
        recognition_data['condition'] = condition
        
        # Standardize paths
        if 'path' in recognition_data.columns:
            recognition_data['path'] = recognition_data['path'].str.replace('\\', '/')
        if 'target_img' in recognition_data.columns:
            recognition_data['target_img'] = recognition_data['target_img'].str.replace('\\', '/')
        if 'lure_img' in recognition_data.columns:
            recognition_data['lure_img'] = recognition_data['lure_img'].str.replace('\\', '/')
        
        # Convert response keys to standardized format
        if 'resp.keys' in recognition_data.columns:
            # Handle list format ['l'] or ['r']
            recognition_data['resp.keys'] = recognition_data['resp.keys'].astype(str)
            recognition_data['resp.keys'] = recognition_data['resp.keys'].str.replace("['", "").str.replace("']", "")
            recognition_data['resp.keys'] = recognition_data['resp.keys'].str.strip()
        
        # Extract response times and confidence
        if 'resp.rt' in recognition_data.columns:
            recognition_data['resp.rt'] = recognition_data['resp.rt'].astype(str).str.replace('[', '').str.replace(']', '')
            recognition_data['resp.rt'] = pd.to_numeric(recognition_data['resp.rt'], errors='coerce')
        
        # Select key columns for master dataset
        keep_cols = ['participant_id', 'condition', 'movie_id']
        for col in RECOGNITION_COLS:
            if col in recognition_data.columns and col != 'movie_id':
                keep_cols.append(col)
        
        # Add is_repeat and vigilance columns for AB participants
        if condition == 'AB':
            if 'is_repeat' in recognition_data.columns:
                keep_cols.append('is_repeat')
            if 'vigilance_pressed' in recognition_data.columns:
                keep_cols.append('vigilance_pressed')
            if 'vigilance_correct' in recognition_data.columns:
                keep_cols.append('vigilance_correct')
        
        recognition_clean = recognition_data[[col for col in keep_cols if col in recognition_data.columns]].copy()
        
        return participant_id, demographics, recognition_clean
        
    except Exception as e:
        print(f"  ✗ Error processing {filepath.name}: {str(e)}")
        return None, None, None

# Process all files
all_demographics = []
all_recognition_data = []
processing_log = []

total_files = len(ab_files) + len(nb_files)
processed = 0

print(f"\n  Processing {total_files} participant files...\n")

for file_list, condition in [(ab_files, 'AB'), (nb_files, 'NB')]:
    for filepath in file_list:
        participant_id, demographics, recognition_data = process_participant_file(filepath, condition)
        
        if participant_id:
            # Add demographics
            demographics['participant_id'] = participant_id
            demographics['condition'] = condition
            demographics['original_filename'] = filepath.name
            all_demographics.append(demographics)
            
            # Add recognition data
            if recognition_data is not None and len(recognition_data) > 0:
                all_recognition_data.append(recognition_data)
                
                # Save individual cleaned file
                clean_filename = f"{participant_id}_clean.csv"
                recognition_data.to_csv(OUTPUT_DIR / "individual_cleaned" / clean_filename, index=False)
                
                processing_log.append({
                    'participant_id': participant_id,
                    'condition': condition,
                    'n_trials': len(recognition_data),
                    'status': 'success'
                })
            else:
                processing_log.append({
                    'participant_id': participant_id,
                    'condition': condition,
                    'n_trials': 0,
                    'status': 'no_recognition_data'
                })
        else:
            processing_log.append({
                'participant_id': filepath.name,
                'condition': 'unknown',
                'n_trials': 0,
                'status': 'processing_error'
            })
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Progress: {processed}/{total_files} files processed...")

print(f"\n Completed processing {processed} files\n")


print("STEP 5: Creating master datasets...")

# Load demographics from the separate Demographic_data.csv file
print("  Loading demographics from Demographic_data.csv...")
demo_file = BASE_DIR / "Demographic_data.csv"
if demo_file.exists():
    df_demographics_raw = pd.read_csv(demo_file)
    
    # Clean column names (remove extra spaces)
    df_demographics_raw.columns = df_demographics_raw.columns.str.strip()
    
    # Rename to match our standard names
    df_demographics_raw = df_demographics_raw.rename(columns={
        'Sub ID': 'participant_id',
        'Age': 'age',
        'Gender': 'gender',
        'Handedness': 'handedness',
        'Vision': 'vision'
    })
    
    # Standardize participant IDs (handle "Sub" vs "sub")
    df_demographics_raw['participant_id'] = df_demographics_raw['participant_id'].str.strip()
    
    # Extract condition from participant ID if not present
    df_demographics_raw['condition'] = df_demographics_raw['participant_id'].str.extract(r'_(AB|NB)', expand=False)
    
    # Save cleaned demographics
    df_demographics_raw = df_demographics_raw.sort_values(['condition', 'participant_id'])
    df_demographics_raw.to_csv(OUTPUT_DIR / "master_demographics.csv", index=False)
    print(f"  ✓ Master demographics loaded: {len(df_demographics_raw)} participants from file")
    df_demographics = df_demographics_raw
else:
    print("  ⚠ Demographic data.csv not found, using extracted demographics")
    df_demographics = pd.DataFrame(all_demographics)
    df_demographics = df_demographics.sort_values(['condition', 'participant_id'])
    df_demographics.to_csv(OUTPUT_DIR / "master_demographics.csv", index=False)
    print(f"  ✓ Master demographics: {len(df_demographics)} participants")

# Create master recognition data file
df_recognition = pd.concat(all_recognition_data, ignore_index=True)
df_recognition = df_recognition.sort_values(['participant_id', 'movie_id'])
df_recognition.to_csv(OUTPUT_DIR / "master_recognition_data.csv", index=False)
print(f"  ✓ Master recognition data: {len(df_recognition)} trials")

# Create processing log
df_log = pd.DataFrame(processing_log)
df_log.to_csv(REPORTS_DIR / "processing_log.csv", index=False)
print(f"  ✓ Processing log saved\n")


print("STEP 6: Creating summary statistics datasets...")

# Calculate per-participant summary statistics
participant_summary = df_recognition.groupby(['participant_id', 'condition']).agg({
    'movie_id': 'count',  # Number of trials
    'resp.corr': ['sum', 'mean'],  # Correct responses
    'resp.rt': ['mean', 'median', 'std'],  # Response times
    'conf_radio.response': ['mean', 'std']  # Confidence ratings
}).reset_index()

# Flatten column names
participant_summary.columns = ['_'.join(col).strip('_') for col in participant_summary.columns.values]
participant_summary = participant_summary.rename(columns={
    'movie_id_count': 'n_trials',
    'resp.corr_sum': 'n_correct',
    'resp.corr_mean': 'accuracy',
    'resp.rt_mean': 'rt_mean',
    'resp.rt_median': 'rt_median',
    'resp.rt_std': 'rt_std',
    'conf_radio.response_mean': 'confidence_mean',
    'conf_radio.response_std': 'confidence_std'
})

# Merge with demographics
# Create matching keys (case-insensitive)
participant_summary['id_match'] = participant_summary['participant_id'].str.lower().str.strip()
df_demographics_match = df_demographics.copy()
df_demographics_match['id_match'] = df_demographics_match['participant_id'].str.lower().str.strip()

df_summary = participant_summary.merge(
    df_demographics_match[['id_match', 'age', 'gender', 'handedness', 'vision']], 
    on='id_match', 
    how='left'
).drop('id_match', axis=1)

df_summary.to_csv(OUTPUT_DIR / "participant_summary_statistics.csv", index=False)
print(f"  ✓ Participant summary statistics: {len(df_summary)} participants")

# Report on demographic matching
missing_demos = df_summary[df_summary['age'].isna()]
if len(missing_demos) > 0:
    print(f"  ⚠ {len(missing_demos)} participants missing demographic data:")
    print(f"    {', '.join(missing_demos['participant_id'].tolist()[:10])}" + 
          (f" and {len(missing_demos)-10} more..." if len(missing_demos) > 10 else ""))

# Create by-condition summary
condition_summary = df_summary.groupby('condition').agg({
    'n_trials': ['mean', 'std'],
    'accuracy': ['mean', 'std'],
    'rt_mean': ['mean', 'std'],
    'confidence_mean': ['mean', 'std']
}).round(3)

condition_summary.to_csv(REPORTS_DIR / "condition_summary.csv")
print(f"  ✓ Condition summary statistics saved\n")


print("STEP 7: Generating data quality report...")

quality_report = []

# Overall statistics
quality_report.append("="*80)
quality_report.append("DATA QUALITY REPORT")
quality_report.append("="*80)
quality_report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
quality_report.append(f"\nTotal participants: {len(df_demographics)}")
quality_report.append(f"  - AB (Abrupt): {len(df_demographics[df_demographics['condition']=='AB'])}")
quality_report.append(f"  - NB (Natural): {len(df_demographics[df_demographics['condition']=='NB'])}")
quality_report.append(f"\nTotal recognition trials: {len(df_recognition)}")

# Demographics completeness
quality_report.append("\n" + "-"*80)
quality_report.append("DEMOGRAPHICS COMPLETENESS")
quality_report.append("-"*80)
for col in DEMOGRAPHIC_COLS:
    if col in df_demographics.columns:
        missing = df_demographics[col].isna().sum()
        pct = (missing / len(df_demographics)) * 100
        quality_report.append(f"  {col:20s}: {len(df_demographics)-missing:3d}/{len(df_demographics):3d} complete ({pct:5.1f}% missing)")

# Recognition data completeness
quality_report.append("\n" + "-"*80)
quality_report.append("RECOGNITION DATA COMPLETENESS")
quality_report.append("-"*80)
for col in ['resp.corr', 'resp.rt', 'conf_radio.response']:
    if col in df_recognition.columns:
        missing = df_recognition[col].isna().sum()
        pct = (missing / len(df_recognition)) * 100
        quality_report.append(f"  {col:20s}: {len(df_recognition)-missing:5d}/{len(df_recognition):5d} complete ({pct:5.1f}% missing)")

# Accuracy distribution
quality_report.append("\n" + "-"*80)
quality_report.append("ACCURACY DISTRIBUTION")
quality_report.append("-"*80)
for condition in ['AB', 'NB']:
    cond_data = df_summary[df_summary['condition'] == condition]
    quality_report.append(f"\n  {condition} Condition:")
    quality_report.append(f"    Mean accuracy: {cond_data['accuracy'].mean():.3f} ± {cond_data['accuracy'].std():.3f}")
    quality_report.append(f"    Range: {cond_data['accuracy'].min():.3f} - {cond_data['accuracy'].max():.3f}")

# Response time distribution
quality_report.append("\n" + "-"*80)
quality_report.append("RESPONSE TIME DISTRIBUTION (seconds)")
quality_report.append("-"*80)
for condition in ['AB', 'NB']:
    cond_data = df_summary[df_summary['condition'] == condition]
    quality_report.append(f"\n  {condition} Condition:")
    quality_report.append(f"    Mean RT: {cond_data['rt_mean'].mean():.3f} ± {cond_data['rt_mean'].std():.3f}")
    quality_report.append(f"    Range: {cond_data['rt_mean'].min():.3f} - {cond_data['rt_mean'].max():.3f}")

# Files with issues
quality_report.append("\n" + "-"*80)
quality_report.append("PROCESSING ISSUES")
quality_report.append("-"*80)
issues = df_log[df_log['status'] != 'success']
if len(issues) > 0:
    quality_report.append(f"\n  Files with issues: {len(issues)}")
    for _, row in issues.iterrows():
        quality_report.append(f"    - {row['participant_id']}: {row['status']}")
else:
    quality_report.append("\n  ✓ All files processed successfully!")

quality_report.append("\n" + "="*80)

# Write report
report_text = "\n".join(quality_report)
with open(REPORTS_DIR / "data_quality_report.txt", 'w') as f:
    f.write(report_text)

print(report_text)


print("\n" + "="*80)
print("STEP 8: Creating analysis-ready datasets...")

# Dataset for statistical tests
df_stats = df_summary.copy()
df_stats.to_csv(OUTPUT_DIR / "data_for_statistics.csv", index=False)
print(f"  ✓ Statistics-ready data: {len(df_stats)} participants")

# Dataset for visualization (long format)
df_viz = df_recognition.copy()
df_viz = df_viz.merge(
    df_demographics[['participant_id', 'age', 'gender', 'condition']], 
    on='participant_id', 
    how='left',
    suffixes=('', '_demo')
)
# Keep the condition from recognition data if there's a conflict
if 'condition_demo' in df_viz.columns:
    df_viz['condition'] = df_viz['condition'].fillna(df_viz['condition_demo'])
    df_viz = df_viz.drop('condition_demo', axis=1)

df_viz.to_csv(OUTPUT_DIR / "data_for_visualization.csv", index=False)
print(f"  ✓ Visualization-ready data: {len(df_viz)} trials")
