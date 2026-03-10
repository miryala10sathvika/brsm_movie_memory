import pandas as pd
import numpy as np

print("="*80)
print("ADDING is_repeat AND VIGILANCE DATA TO CLEANED FILES")
print("="*80)

# Load movie metadata with is_repeat info
abrupt = pd.read_csv('abruptmovies.csv')
natural = pd.read_csv('naturalmovies.csv')

# Extract movie_id from path
def extract_movie_id(path):
    # abrupt_videos\abrupt_3.mp4 -> 3
    # Trimmed_videos/natural_3.mp4 -> 3
    filename = path.split('/')[-1].split('\\')[-1]
    movie_num = filename.split('_')[-1].replace('.mp4', '')
    return int(movie_num)

abrupt['movie_id'] = abrupt['path'].apply(extract_movie_id)
natural['movie_id'] = natural['path'].apply(extract_movie_id)

# Get unique movie-repeat pairs
abrupt_repeat_map = abrupt.drop_duplicates('movie_id')[['movie_id', 'is_repeat']].set_index('movie_id')['is_repeat'].to_dict()
natural_repeat_map = natural.drop_duplicates('movie_id')[['movie_id', 'is_repeat']].set_index('movie_id')['is_repeat'].to_dict()

print("\n1. REPEAT VIDEOS IDENTIFIED:")
print(f"   Abrupt videos shown twice: {sorted([k for k,v in abrupt_repeat_map.items() if v == 1])}")
print(f"   Natural videos shown twice: {sorted([k for k,v in natural_repeat_map.items() if v == 1])}")

# Load cleaned trial data
df_trials = pd.read_csv('final_cleaned_data/trials_final_clean.csv')
print(f"\n2. LOADED TRIAL DATA: {len(df_trials)} trials")

# Map is_repeat based on movie_id and condition
def get_is_repeat(row):
    movie_id = int(row['movie_id'])
    condition = row['condition']
    
    if condition == 'AB':
        return abrupt_repeat_map.get(movie_id, 0)
    elif condition == 'NB':
        return natural_repeat_map.get(movie_id, 0)
    else:
        return np.nan

df_trials['is_repeat'] = df_trials.apply(get_is_repeat, axis=1)

print(f"\n3. MAPPED is_repeat TO TRIALS:")
print(f"   Trials with is_repeat=1: {(df_trials['is_repeat'] == 1).sum()}")
print(f"   Trials with is_repeat=0: {(df_trials['is_repeat'] == 0).sum()}")

# Check by condition
print(f"\n4. REPEAT TRIALS BY CONDITION:")
for condition in ['AB', 'NB']:
    cond_data = df_trials[df_trials['condition'] == condition]
    n_repeat = (cond_data['is_repeat'] == 1).sum()
    n_total = len(cond_data)
    print(f"   {condition}: {n_repeat}/{n_total} repeat trials ({n_repeat/n_total*100:.1f}%)")
    
    # Show which movies
    repeat_movies = cond_data[cond_data['is_repeat'] == 1]['movie_id'].unique()
    print(f"      Repeated movie IDs: {sorted(repeat_movies)}")

# Now extract vigilance data from raw files
print(f"\n5. EXTRACTING VIGILANCE DATA FROM RAW FILES...")

import os
from pathlib import Path

RAW_DATA_DIR = Path("BRSM data csv")
all_files = list(RAW_DATA_DIR.glob("*.csv"))

vigilance_data = []

for filepath in all_files:
    try:
        df_raw = pd.read_csv(filepath, low_memory=False)
        
        # Extract participant ID
        parts = filepath.name.split('_')
        if len(parts) >= 2:
            participant_id = f"{parts[0]}_{parts[1]}"
            condition = parts[1]
        else:
            continue
        
        # Find vigilance row (where vigilance_pressed is not NaN)
        vigilance_rows = df_raw[df_raw['vigilance_pressed'].notna()]
        
        if len(vigilance_rows) > 0:
            for _, row in vigilance_rows.iterrows():
                vigilance_data.append({
                    'participant_id': participant_id,
                    'condition': condition,
                    'vigilance_pressed': row['vigilance_pressed'],
                    'vigilance_correct': row['vigilance_correct'],
                    'encoding_duration': row.get('Videos.stopped', np.nan) - row.get('instruction_2.stopped', np.nan) if 'Videos.stopped' in row and 'instruction_2.stopped' in row else np.nan
                })
    except Exception as e:
        print(f"   Warning: Could not process {filepath.name}: {e}")

df_vigilance = pd.DataFrame(vigilance_data)

print(f"   Extracted vigilance data for {len(df_vigilance)} participants")
print(f"\n   Vigilance response distribution:")
print(df_vigilance['vigilance_pressed'].value_counts())
print(f"\n   Vigilance correctness:")
print(df_vigilance['vigilance_correct'].value_counts())

# Calculate vigilance metrics per participant
df_vigilance['vigilance_accuracy'] = df_vigilance['vigilance_correct']
df_vigilance['passed_vigilance'] = (df_vigilance['vigilance_correct'] == 1.0)

print(f"\n6. VIGILANCE PERFORMANCE:")
print(f"   Participants who passed vigilance check: {df_vigilance['passed_vigilance'].sum()}/{len(df_vigilance)}")
print(f"   Vigilance pass rate: {df_vigilance['passed_vigilance'].mean()*100:.1f}%")

# Check encoding duration if available
if 'encoding_duration' in df_vigilance.columns:
    valid_durations = df_vigilance['encoding_duration'].dropna()
    if len(valid_durations) > 0:
        print(f"\n7. ENCODING DURATION (instruction_2.stopped to Videos.stopped):")
        print(f"   Mean: {valid_durations.mean():.1f} seconds ({valid_durations.mean()/60:.1f} minutes)")
        print(f"   Range: {valid_durations.min():.1f} - {valid_durations.max():.1f} seconds")
        
        # Flag participants with >27 minutes encoding
        long_encoders = df_vigilance[df_vigilance['encoding_duration'] > 1620]  # 27 minutes
        if len(long_encoders) > 0:
            print(f"   ⚠️  Participants with >27 minutes encoding: {len(long_encoders)}")

# Save updated trial data
df_trials.to_csv('final_cleaned_data/trials_final_clean_with_repeat.csv', index=False)
print(f"\n✅ SAVED: trials_final_clean_with_repeat.csv (with is_repeat column)")

# Save vigilance summary
df_vigilance.to_csv('final_cleaned_data/vigilance_summary.csv', index=False)
print(f"✅ SAVED: vigilance_summary.csv")

# Load participant summary and merge vigilance data
df_participants = pd.read_csv('final_cleaned_data/participants_final_clean.csv')

# Merge vigilance data
df_participants = df_participants.merge(
    df_vigilance[['participant_id', 'vigilance_pressed', 'vigilance_correct', 'passed_vigilance']],
    on='participant_id',
    how='left'
)

df_participants.to_csv('final_cleaned_data/participants_final_clean_with_vigilance.csv', index=False)
print(f"✅ SAVED: participants_final_clean_with_vigilance.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
✅ is_repeat column successfully added to trial data
   - 5 videos per condition were shown twice (vigilance checks)
   - Mapped based on movie_id and condition
   
✅ Vigilance data extracted from raw files
   - Vigilance response and correctness per participant
   - Pass rate: {df_vigilance['passed_vigilance'].mean()*100:.1f}%
   
✅ All files updated with complete data
""")
