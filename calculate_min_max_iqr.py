#!/usr/bin/env python3
"""
Calculate Min, Max, and IQR for Recognition Accuracy and Response Time
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
INPUT_DIR = Path("final_cleaned_data")
df_participants = pd.read_csv(INPUT_DIR / "participants_final_clean.csv")
df_trials = pd.read_csv(INPUT_DIR / "trials_final_clean.csv")

print("="*80)
print("MIN, MAX, and IQR CALCULATIONS")
print("="*80)

# Function to calculate statistics
def calculate_stats(data, variable_name, unit=""):
    """Calculate min, max, Q1, Q3, and IQR"""
    data_clean = data.dropna()
    
    minimum = data_clean.min()
    maximum = data_clean.max()
    q1 = data_clean.quantile(0.25)  # 25th percentile
    q3 = data_clean.quantile(0.75)  # 75th percentile
    iqr = q3 - q1
    median = data_clean.median()
    mean = data_clean.mean()
    
    print(f"\n{variable_name}:")
    print(f"  Min: {minimum:.4f}{unit}")
    print(f"  Q1 (25th percentile): {q1:.4f}{unit}")
    print(f"  Median (50th percentile): {median:.4f}{unit}")
    print(f"  Q3 (75th percentile): {q3:.4f}{unit}")
    print(f"  Max: {maximum:.4f}{unit}")
    print(f"  IQR (Q3 - Q1): {iqr:.4f}{unit}")
    print(f"  Mean: {mean:.4f}{unit} (for reference)")
    print(f"  Range: {minimum:.4f} to {maximum:.4f}{unit}")
    
    return {
        'variable': variable_name,
        'min': minimum,
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': maximum,
        'iqr': iqr,
        'mean': mean,
        'n': len(data_clean)
    }

# ============================================================================
# OVERALL STATISTICS
# ============================================================================
print("\n" + "="*80)
print("OVERALL (All Participants)")
print("="*80)

results = []

# Recognition Accuracy (participant-level)
results.append(calculate_stats(df_participants['accuracy'], 
                               "Recognition Accuracy (participant-level)", ""))

# Response Time (participant-level mean)
results.append(calculate_stats(df_participants['rt_mean'], 
                               "Mean Response Time (participant-level)", " seconds"))

# Response Time (trial-level)
results.append(calculate_stats(df_trials['resp.rt'], 
                               "Response Time (trial-level)", " seconds"))

# Confidence (participant-level)
results.append(calculate_stats(df_participants['confidence_mean'], 
                               "Mean Confidence Rating (participant-level)", " /5"))

# ============================================================================
# BY CONDITION
# ============================================================================
print("\n" + "="*80)
print("BY CONDITION")
print("="*80)

for condition in ['AB', 'NB']:
    print(f"\n{'='*80}")
    print(f"{condition} CONDITION")
    print(f"{'='*80}")
    
    # Filter data
    participants_cond = df_participants[df_participants['condition'] == condition]
    trials_cond = df_trials[df_trials['condition'] == condition]
    
    # Recognition Accuracy
    results.append(calculate_stats(participants_cond['accuracy'], 
                                   f"Recognition Accuracy ({condition})", ""))
    
    # Response Time (participant-level)
    results.append(calculate_stats(participants_cond['rt_mean'], 
                                   f"Mean Response Time ({condition})", " seconds"))
    
    # Response Time (trial-level)
    results.append(calculate_stats(trials_cond['resp.rt'], 
                                   f"Response Time - trial level ({condition})", " seconds"))
    
    # Confidence
    results.append(calculate_stats(participants_cond['confidence_mean'], 
                                   f"Mean Confidence ({condition})", " /5"))

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
df_results = pd.DataFrame(results)
output_file = Path("statistical_results") / "min_max_iqr_statistics.csv"
output_file.parent.mkdir(exist_ok=True)
df_results.to_csv(output_file, index=False)

print("\n" + "="*80)
print(f"Results saved to: {output_file}")
print("="*80)

# ============================================================================
# CREATE SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE: Recognition Accuracy")
print("="*80)

acc_overall = df_participants['accuracy']
acc_nb = df_participants[df_participants['condition'] == 'NB']['accuracy']
acc_ab = df_participants[df_participants['condition'] == 'AB']['accuracy']

print("\n{:<20} {:>12} {:>12} {:>12}".format("Statistic", "Overall", "NB", "AB"))
print("-"*80)
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Min", acc_overall.min(), acc_nb.min(), acc_ab.min()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Q1 (25%)", acc_overall.quantile(0.25), acc_nb.quantile(0.25), acc_ab.quantile(0.25)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Median (50%)", acc_overall.median(), acc_nb.median(), acc_ab.median()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Q3 (75%)", acc_overall.quantile(0.75), acc_nb.quantile(0.75), acc_ab.quantile(0.75)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Max", acc_overall.max(), acc_nb.max(), acc_ab.max()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("IQR", acc_overall.quantile(0.75) - acc_overall.quantile(0.25), 
                                                      acc_nb.quantile(0.75) - acc_nb.quantile(0.25),
                                                      acc_ab.quantile(0.75) - acc_ab.quantile(0.25)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Mean", acc_overall.mean(), acc_nb.mean(), acc_ab.mean()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("SD", acc_overall.std(), acc_nb.std(), acc_ab.std()))

print("\n" + "="*80)
print("SUMMARY TABLE: Response Time (seconds)")
print("="*80)

rt_overall = df_participants['rt_mean']
rt_nb = df_participants[df_participants['condition'] == 'NB']['rt_mean']
rt_ab = df_participants[df_participants['condition'] == 'AB']['rt_mean']

print("\n{:<20} {:>12} {:>12} {:>12}".format("Statistic", "Overall", "NB", "AB"))
print("-"*80)
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Min", rt_overall.min(), rt_nb.min(), rt_ab.min()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Q1 (25%)", rt_overall.quantile(0.25), rt_nb.quantile(0.25), rt_ab.quantile(0.25)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Median (50%)", rt_overall.median(), rt_nb.median(), rt_ab.median()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Q3 (75%)", rt_overall.quantile(0.75), rt_nb.quantile(0.75), rt_ab.quantile(0.75)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Max", rt_overall.max(), rt_nb.max(), rt_ab.max()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("IQR", rt_overall.quantile(0.75) - rt_overall.quantile(0.25), 
                                                      rt_nb.quantile(0.75) - rt_nb.quantile(0.25),
                                                      rt_ab.quantile(0.75) - rt_ab.quantile(0.25)))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Mean", rt_overall.mean(), rt_nb.mean(), rt_ab.mean()))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("SD", rt_overall.std(), rt_nb.std(), rt_ab.std()))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
