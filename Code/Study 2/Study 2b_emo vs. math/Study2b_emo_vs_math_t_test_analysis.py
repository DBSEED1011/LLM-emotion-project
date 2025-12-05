#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Independent samples t-test analysis
Analyze stage=2 data from emotion & control data, comparing group=1(emo) vs group=0(math) differences
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_filter_data():
    """Load data and filter stage=2 data"""
    print("Loading data...")
    df = pd.read_excel('Study2b_emo_vs_math.xlsx', sheet_name='bytrial')
    print(f"Original data shape: {df.shape}")
    
    # Filter stage=2 data
    stage2_data = df[df['stage'] == 2].copy()
    print(f"Stage=2 data shape: {stage2_data.shape}")
    
    # Check group distribution
    print("\nGroup distribution:")
    print(stage2_data['group'].value_counts())
    
    return stage2_data

def perform_t_tests(data):
    """Perform independent samples t-tests for each variable"""
    print("\n" + "="*60)
    print("Independent samples t-test results (Group 1: emo vs Group 0: math)")
    print("="*60)
    
    # Numeric variables to analyze
    numeric_vars = ['fairness', 'choice', 'current_valence', 'current_arousal', 
                   'predicted_valence', 'predicted_arousal', 'actual_valence', 'actual_arousal']
    
    results = []
    
    for var in numeric_vars:
        if var in data.columns:
            # Get data for two groups
            group1_data = data[data['group'] == 1][var].dropna()
            group0_data = data[data['group'] == 0][var].dropna()
            
            if len(group1_data) > 0 and len(group0_data) > 0:
                # Perform independent samples t-test
                t_stat, p_value = stats.ttest_ind(group1_data, group0_data)
                
                # Calculate descriptive statistics
                group1_mean = group1_data.mean()
                group0_mean = group0_data.mean()
                group1_std = group1_data.std()
                group0_std = group0_data.std()
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_std**2 + 
                                    (len(group0_data) - 1) * group0_std**2) / 
                                   (len(group1_data) + len(group0_data) - 2))
                cohens_d = (group1_mean - group0_mean) / pooled_std if pooled_std > 0 else 0
                
                result = {
                    'Variable': var,
                    'Group1_n': len(group1_data),
                    'Group0_n': len(group0_data),
                    'Group1_mean': group1_mean,
                    'Group0_mean': group0_mean,
                    'Group1_std': group1_std,
                    'Group0_std': group0_std,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
                results.append(result)
                
                # Print results
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"\n{var}:")
                print(f"  Group 1 (emo):  n={len(group1_data):3d}, M={group1_mean:.3f}, SD={group1_std:.3f}")
                print(f"  Group 0 (math): n={len(group0_data):3d}, M={group0_mean:.3f}, SD={group0_std:.3f}")
                print(f"  t({len(group1_data) + len(group0_data) - 2}) = {t_stat:.3f}, p = {p_value:.4f} {significance}")
                print(f"  Cohen's d = {cohens_d:.3f}")
    
    return pd.DataFrame(results)



def main():
    """Main function"""
    print("Stage 2 Independent samples t-test analysis")
    print("="*50)
    
    # Load and filter data
    data = load_and_filter_data()
    
    # Perform t-tests
    results_df = perform_t_tests(data)
    
    
    
    # Save results to CSV
    results_df.to_csv('emo_math_t_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: emo_math_t_test_results.csv")
    
    # Summary
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    significant_count = results_df['significant'].sum()
    total_tests = len(results_df)
    print(f"Total t-tests performed for {total_tests} variables")
    print(f"Among them, {significant_count} variables showed significant differences (p < 0.05)")
    
    if significant_count > 0:
        print("\nVariables with significant differences:")
        for _, row in results_df[results_df['significant']].iterrows():
            print(f"  - {row['Variable']}: p = {row['p_value']:.4f}, Cohen's d = {row['cohens_d']:.3f}")

if __name__ == "__main__":
    main()

