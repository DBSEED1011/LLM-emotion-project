#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze how group moderates the effects of different traits on choice behavior
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data"""
    try:
        df = pd.read_excel('data_with_grouping_variables.xlsx')
        print(f"Data loaded successfully, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None


def basic_descriptive_stats(df):
    """Basic descriptive statistics"""
    print("\n" + "="*60)
    print("Basic Descriptive Statistics")
    print("="*60)
    
    # Basic statistics for Choice variable
    print("\nChoice variable statistics:")
    print(f"Mean: {df['choice'].mean():.3f}")
    print(f"Standard deviation: {df['choice'].std():.3f}")
    print(f"Minimum: {df['choice'].min()}")
    print(f"Maximum: {df['choice'].max()}")
    
    # Group distribution
    print(f"\nGroup distribution:")
    group_counts = df['group'].value_counts()
    for group, count in group_counts.items():
        print(f"{group}: {count} ({count/len(df)*100:.1f}%)")
    
    # Distribution of grouping variables
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    for var in grouping_vars:
        if var in df.columns:
            print(f"\n{var} distribution:")
            value_counts = df[var].value_counts()
            for value, count in value_counts.items():
                if pd.notna(value):
                    print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

def analyze_moderation_effect(df, grouping_var, outcome_var='choice'):
    """Analyze moderation effect"""
    print(f"\n" + "="*60)
    print(f"Moderation effect analysis: {grouping_var} × group → {outcome_var}")
    print("="*60)
    
    # Check if variable exists
    if grouping_var not in df.columns:
        print(f"Warning: Variable '{grouping_var}' does not exist")
        return None
    
    # Remove missing values
    analysis_df = df[[outcome_var, 'group', grouping_var]].dropna()
    print(f"Analysis sample size: {len(analysis_df)}")
    
    # Descriptive statistics
    print(f"\nMean choice by {grouping_var} levels:")
    desc_stats = analysis_df.groupby([grouping_var, 'group'])[outcome_var].agg(['mean', 'std', 'count'])
    print(desc_stats)
    
    # 2×5 ANOVA (grouping variable × group)
    try:
        # Build formula
        formula = f"{outcome_var} ~ C({grouping_var}) * C(group)"
        model = ols(formula, data=analysis_df).fit()
        
        # ANOVA table
        anova_table = anova_lm(model, typ=2)
        print(f"\nANOVA results:")
        print(anova_table)
        
        # Extract key statistics
        main_effect_grouping = anova_table.loc[f'C({grouping_var})', 'PR(>F)']
        main_effect_group = anova_table.loc['C(group)', 'PR(>F)']
        interaction_effect = anova_table.loc[f'C({grouping_var}):C(group)', 'PR(>F)']
        
        print(f"\nKey results:")
        print(f"{grouping_var} main effect: p = {main_effect_grouping:.4f}")
        print(f"Group main effect: p = {main_effect_group:.4f}")
        print(f"Interaction effect ({grouping_var} × group): p = {interaction_effect:.4f}")
        
        # Determine moderation effect
        if interaction_effect < 0.05:
            print(f"✓ Significant moderation effect found! {grouping_var} moderates the effect of group on {outcome_var}")
            
            # Simple effects analysis
            print(f"\nSimple effects analysis:")
            simple_effects = analyze_simple_effects(analysis_df, grouping_var, outcome_var)
            return {
                'significant_moderation': True,
                'interaction_p': interaction_effect,
                'simple_effects': simple_effects,
                'descriptive_stats': desc_stats
            }
        else:
            print(f"✗ No significant moderation effect (p = {interaction_effect:.4f})")
            return {
                'significant_moderation': False,
                'interaction_p': interaction_effect,
                'descriptive_stats': desc_stats
            }
            
    except Exception as e:
        print(f"ANOVA analysis error: {e}")
        return None

def analyze_simple_effects(df, grouping_var, outcome_var):
    """Simple effects analysis"""
    simple_effects = {}
    
    # Compare differences between groups at each grouping variable level
    for level in df[grouping_var].unique():
        if pd.notna(level):
            subset = df[df[grouping_var] == level]
            if len(subset) > 0:
                # One-way ANOVA to compare different groups
                try:
                    groups = [subset[subset['group'] == g][outcome_var].values 
                             for g in subset['group'].unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    simple_effects[level] = {
                        'f_stat': f_stat,
                        'p_value': p_value,
                        'group_means': subset.groupby('group')[outcome_var].mean().to_dict()
                    }
                    
                    print(f"  {grouping_var}={level}: F={f_stat:.3f}, p={p_value:.4f}")
                    for group, mean_val in simple_effects[level]['group_means'].items():
                        print(f"    {group}: M={mean_val:.3f}")
                        
                except Exception as e:
                    print(f"  {grouping_var}={level}: Analysis failed - {e}")
    
    return simple_effects

def run_all_moderation_analyses(df):
    """Run all moderation effect analyses"""
    print("\n" + "="*80)
    print("Starting all moderation effect analyses")
    print("="*80)
    
    # All grouping variables
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    
    results = {}
    
    for var in grouping_vars:
        if var in df.columns:
            # Analyze moderation effect
            result = analyze_moderation_effect(df, var)
            if result:
                results[var] = result
            
            print("\n" + "-"*60)
    
    return results

def generate_summary_report(results):
    """Generate summary report"""
    print("\n" + "="*80)
    print("Moderation Effect Analysis Summary Report")
    print("="*80)
    
    significant_moderations = []
    non_significant_moderations = []
    
    for var, result in results.items():
        if result['significant_moderation']:
            significant_moderations.append((var, result['interaction_p']))
        else:
            non_significant_moderations.append((var, result['interaction_p']))
    
    print(f"\nVariables with significant moderation effects ({len(significant_moderations)}):")
    for var, p_val in significant_moderations:
        print(f"  ✓ {var}: p = {p_val:.4f}")
    
    print(f"\nVariables without significant moderation effects ({len(non_significant_moderations)}):")
    for var, p_val in non_significant_moderations:
        print(f"  ✗ {var}: p = {p_val:.4f}")
    
    # Save detailed results to Excel file
    try:
        with pd.ExcelWriter('choice_moderation_analysis_summary.xlsx', engine='openpyxl') as writer:
            
            # 1. Basic summary table
            summary_df = pd.DataFrame([
                {
                    'grouping_variable': var,
                    'significant_moderation': result['significant_moderation'],
                    'interaction_p_value': result['interaction_p'],
                    'interaction_significant': 'Yes' if result['interaction_p'] < 0.05 else 'No'
                }
                for var, result in results.items()
            ])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. Descriptive statistics table
            desc_stats_list = []
            for var, result in results.items():
                desc_stats = result['descriptive_stats']
                for (trait_level, group), stats_row in desc_stats.iterrows():
                    desc_stats_list.append({
                        'grouping_variable': var,
                        'trait_level': trait_level,
                        'group': group,
                        'mean': stats_row['mean'],
                        'std': stats_row['std'],
                        'count': stats_row['count']
                    })
            
            desc_stats_df = pd.DataFrame(desc_stats_list)
            desc_stats_df.to_excel(writer, sheet_name='Descriptive_Stats', index=False)
            
            # 3. Simple effects analysis table (only significant moderation effects)
            simple_effects_list = []
            for var, result in results.items():
                if result['significant_moderation'] and 'simple_effects' in result:
                    simple_effects = result['simple_effects']
                    for trait_level, effect_data in simple_effects.items():
                        simple_effects_list.append({
                            'grouping_variable': var,
                            'trait_level': trait_level,
                            'f_statistic': effect_data['f_stat'],
                            'p_value': effect_data['p_value'],
                            'significant': 'Yes' if effect_data['p_value'] < 0.05 else 'No'
                        })
                        
                        # Add mean for each group
                        for group, mean_val in effect_data['group_means'].items():
                            simple_effects_list.append({
                                'grouping_variable': var,
                                'trait_level': f"{trait_level}_group_{group}",
                                'f_statistic': '',
                                'p_value': '',
                                'significant': '',
                                'group_mean': mean_val
                            })
            
            if simple_effects_list:
                simple_effects_df = pd.DataFrame(simple_effects_list)
                simple_effects_df.to_excel(writer, sheet_name='Simple_Effects', index=False)
            
            # 4. Detailed results table (contains all statistical information)
            detailed_results_list = []
            for var, result in results.items():
                detailed_results_list.append({
                    'grouping_variable': var,
                    'significant_moderation': result['significant_moderation'],
                    'interaction_p_value': result['interaction_p'],
                    'interaction_significant': 'Yes' if result['interaction_p'] < 0.05 else 'No',
                    'num_trait_levels': len(result['descriptive_stats'].index.get_level_values(0).unique()),
                    'total_sample_size': result['descriptive_stats']['count'].sum(),
                    'has_simple_effects': 'Yes' if 'simple_effects' in result else 'No'
                })
            
            detailed_df = pd.DataFrame(detailed_results_list)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        print(f"\nDetailed summary report saved to: choice_moderation_analysis_summary.xlsx")
        print("Contains the following worksheets:")
        print("  - Summary: Basic moderation effect summary")
        print("  - Descriptive_Stats: Descriptive statistics")
        print("  - Simple_Effects: Simple effects analysis results")
        print("  - Detailed_Results: Detailed analysis results")
        
    except Exception as e:
        print(f"Failed to save summary report: {e}")

def main():
    """Main function"""
    print("Starting moderation effect analysis...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic descriptive statistics
    basic_descriptive_stats(df)
    
    # Run all moderation effect analyses
    results = run_all_moderation_analyses(df)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\nModeration effect analysis completed!")

if __name__ == "__main__":
    main()

