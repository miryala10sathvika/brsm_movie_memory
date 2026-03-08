# Cleaned Data - BRSM Movie Memory Experiment

## Data Cleaning Summary

**Date Processed:** March 8, 2026  
**Original Files:** 171 participant CSV files  
**Successfully Processed:** 170 participants (6,800 trials)  
**Excluded Files:** 1 incomplete file

---

## File Organization

### Master Datasets

1. **master_demographics.csv** (187 rows)
   - Demographic information for all participants from Demographic_data.csv
   - Includes: participant_id, age, gender, handedness, vision, condition

2. **master_recognition_data.csv** (6,800 rows)
   - All recognition trial data combined
   - 170 participants × 40 trials each (some with 80 trials due to duplicates)
   - Columns: participant_id, condition, movie_id, target_img, lure_img, response, accuracy, RT, confidence

3. **participant_summary_statistics.csv** (146 rows)
   - Per-participant summary statistics with demographics
   - Includes: accuracy, mean RT, median RT, confidence ratings
   - 146 participants with matching demographics (24 participants in trial data not in demographic file)

### Analysis-Ready Datasets

1. **data_for_statistics.csv**
   - Wide format: one row per participant
   - Use for: t-tests, ANOVAs, correlation analyses
   - Variables: accuracy, RT measures, confidence by condition

2. **data_for_visualization.csv** (6,800 rows)
   - Long format: one row per trial
   - Use for: plotting individual trials, distributions, trial-level analyses
   - Includes demographics merged with trial data

### Individual Files

**individual_cleaned/** folder contains 170 individual participant files
- Standardized format across all participants
- Paths converted to forward slashes
- Response keys cleaned (removed brackets)
- Ready for individual inspection

---

## Data Quality Notes

### Demographics
- **163/187** participants have complete demographic data (87.2%)
- **24 missing**: Some participant IDs in trial data don't match Demographic_data.csv
  - Examples: aru13, aru5467598, hello001, test1, etc. (likely test/pilot runs)

### Recognition Data
- **100%** complete for core variables (accuracy, RT, confidence)
- All 170 participants have 40 recognition trials (some have 80 due to duplicates in dataset)

### Performance Summary

**AB Condition (Abrupt boundaries):**
- N = 81-92 participants (depends on matching)
- Mean accuracy: 83.8% ± 8.4%
- Mean RT: 5.58 ± 1.49 seconds

**NB Condition (Natural boundaries):**
- N =89-93 participants
- Mean accuracy: 87.2% ± 6.9%
- Mean RT: 5.83 ± 1.64 seconds

---

## Excluded Files

### Incomplete Data (1 file)
- **sub42_NB_recognitionstage_2026-01-20_09h31.14.064.csv**
  - Reason: No recognition trial data (only demographics/instructions)
  - Participant started but did not complete the task
  - Correctly excluded from analysis

---

## Data Cleaning Actions Performed

1. ✅ **Standardized paths**: Converted all backslashes to forward slashes
2. ✅ **Cleaned responses**: Removed brackets from response keys ['l'] → 'l'
3. ✅ **Merged demographics**: Loaded from Demographic_data.csv 
4. ✅ **Case-insensitive matching**: Handled "Sub" vs "sub" variations
5. ✅ **Calculated summaries**: Per-participant accuracy, RT statistics
6. ✅ **Identified duplicates**: Some participants appear multiple times
7. ✅ **Created analysis formats**: Both wide (stats) and long (viz) formats

---

## Usage Recommendations

### For Statistical Analysis
Use **data_for_statistics.csv**:
```python
import pandas as pd
df = pd.read_csv('cleaned_data/data_for_statistics.csv')

# Filter participants with complete demographics
df_complete = df[df['age'].notna()]

# Compare conditions
ab_group = df_complete[df_complete['condition'] == 'AB']
nb_group = df_complete[df_complete['condition'] == 'NB']
```

### For Visualization
Use **data_for_visualization.csv**:
```python
import pandas as pd
import seaborn as sns

df = pd.read_csv('cleaned_data/data_for_visualization.csv')

# Plot accuracy by condition
sns.boxplot(data=df, x='condition', y='resp.corr')
```

---

## Notes on Duplicate Participants

Some participants appear multiple times in the dataset (e.g., sub77_NB has 120 trials instead of 40). This could be due to:
- Multiple testing sessions
- Repeated experiments
- Data collection errors

**Recommendation**: Decide whether to:
1. Keep only first occurrence
2. Average across sessions
3. Treat as separate observations

Check `data_reports/processing_log.csv` for details on each participant.

---

## Original Data

All original files remain **UNTOUCHED** in `BRSM data csv/` folder.
This cleaned data is for analysis only.

---

For questions or issues, check:
- `data_reports/data_quality_report.txt` - Full quality assessment
- `data_reports/processing_log.csv` - Per-file processing status
- `data_reports/condition_summary.csv` - By-condition statistics



CLEANED DATA SUMMARY:
   ✓ 170 participants processed successfully
   ✓ 6,800 recognition trials (40 trials per participant)
   ✓ 163 participants with complete demographics (87%)
   ✓ 1 incomplete file excluded (sub42_NB - no trial data)

READY FOR ANALYSIS:
   → cleaned_data/data_for_statistics.csv    (146 participants, wide format)
   → cleaned_data/data_for_visualization.csv (6,800 trials, long format)
   → cleaned_data/participant_summary_statistics.csv
   → cleaned_data/master_demographics.csv
   → cleaned_data/master_recognition_data.csv

PERFORMANCE BY CONDITION:
   AB (Abrupt):  83.8% accuracy, 5.58s RT
   NB (Natural): 87.2% accuracy, 5.83s RT