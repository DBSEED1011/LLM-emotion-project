#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze how group moderates the effects of different traits on emotion variables
Perform moderation effect analysis and plotting for 6 emotion variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load data"""
    try:
        df = pd.read_excel('data_with_grouping_variables.xlsx')
        print(f"Data loaded successfully, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

def setup_group_order():
    """Set group order and labels"""
    group_mapping = {
        'human': 'Human',
        'gpt3.5': 'GPT-3.5', 
        'o3': 'o3-mini',
        'V3': 'DeepSeek-V3',
        'R1': 'DeepSeek-R1'
    }
    
    group_order = ['human', 'gpt3.5', 'o3', 'V3', 'R1']
    group_labels = [group_mapping[g] for g in group_order]
    
    return group_order, group_labels

def analyze_moderation_effect(df, grouping_var, outcome_var):
    """Analyze moderation effect"""
    print(f"\n" + "="*60)
    print(f"Moderation effect analysis: {grouping_var} × group → {outcome_var}")
    print("="*60)
    
    # Check if variables exist
    if grouping_var not in df.columns or outcome_var not in df.columns:
        print(f"Warning: Variable '{grouping_var}' or '{outcome_var}' does not exist")
        return None
    
    # Remove missing values
    analysis_df = df[[outcome_var, 'group', grouping_var]].dropna()
    print(f"Analysis sample size: {len(analysis_df)}")
    
    # Descriptive statistics
    print(f"\nMean {outcome_var} by {grouping_var} levels:")
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
            return {
                'significant_moderation': True,
                'interaction_p': interaction_effect,
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

def create_combined_plot(df, outcome_var, outcome_label):
    """Create combined moderation effect plots"""
    print(f"Starting to create combined plot for {outcome_var}...")
    
    # Set group order
    group_order, group_labels = setup_group_order()
    
    # Define grouping variables and labels
    grouping_vars = {
        'gender': 'Gender',
        'SVO': 'Social Value Orientation', 
        'CESD_2g': 'Depression Tendency',
        'AQ_2g': 'Autism Spectrum Quotient',
        'age_2g': 'Age',
        'JSI_2g': 'Justice Sensitivity',
        'ERS_2g': 'Emotional Reactivity'
    }
    
    # Create large figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    axes = axes.flatten()
    
    # Set color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (var, label) in enumerate(grouping_vars.items()):
        if i >= 7:  # Only process first 7 variables
            break
            
        ax = axes[i]
        
        # Calculate mean outcome for each group and trait level
        mean_data = df.groupby([var, 'group'])[outcome_var].mean().reset_index()
        
        # Reorder groups
        mean_data['group'] = pd.Categorical(mean_data['group'], categories=group_order, ordered=True)
        mean_data = mean_data.sort_values('group')
        
        # Get trait levels
        trait_levels = sorted(mean_data[var].unique())
        if pd.isna(trait_levels).any():
            trait_levels = [x for x in trait_levels if pd.notna(x)]
        
        # Create bar plots for each trait level
        x_pos = np.arange(len(group_order))
        width = 0.35
        
        for j, level in enumerate(trait_levels):
            if pd.notna(level):
                level_data = mean_data[mean_data[var] == level]
                
                # Ensure all groups have data
                values = []
                for group in group_order:
                    group_data = level_data[level_data['group'] == group]
                    if len(group_data) > 0:
                        values.append(group_data[outcome_var].iloc[0])
                    else:
                        values.append(0)
                
                # Draw bar plot
                offset = (j - 0.5) * width
                bars = ax.bar(x_pos + offset, values, width, 
                             label=str(level), 
                             color=colors[j % len(colors)],
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=6)
        
        # Set plot properties
        ax.set_title(f'{label}', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel(f'{outcome_label}', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        
        # Set y-axis range based on outcome variable
        if 'AA_' in outcome_var or 'AC_' in outcome_var:
            ax.set_ylim(-100, 100)  # AA and AC variables range from -100 to +100
        elif 'EmoFDBK_' in outcome_var:
            ax.set_ylim(-200, 200)  # EmoFDBK variables range from -200 to +200
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper right')
        
        # Add background color to distinguish human and LLM
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='Human')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue', label='LLMs')
    
    # Hide the 8th subplot (only 7 variables)
    axes[7].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # Save plot
    filename = f'combined_moderation_{outcome_var}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved: {filename}")
    
    plt.show()

def run_emotion_analysis(df):
    """Run analysis for all emotion variables"""
    print("\n" + "="*80)
    print("Starting moderation effect analysis for all emotion variables")
    print("="*80)
    
    # Define emotion variables and labels
    emotion_vars = {
        'AA_valence': 'After-allocation valence',
        'AA_arousal': 'After-allocation arousal',
        'AC_valence': 'After-choice valence', 
        'AC_arousal': 'After-choice arousal',
        'EmoFDBK_valence': 'Emotional outcome valence',
        'EmoFDBK_arousal': 'Emotional outcome arousal'
    }
    
    # All grouping variables
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    
    all_results = {}
    
    for emotion_var, emotion_label in emotion_vars.items():
        print(f"\n{'='*20} Analyzing {emotion_var} {'='*20}")
        
        emotion_results = {}
        
        # Perform moderation effect analysis for each grouping variable
        for var in grouping_vars:
            if var in df.columns:
                result = analyze_moderation_effect(df, var, emotion_var)
                if result:
                    emotion_results[var] = result
        
        all_results[emotion_var] = emotion_results
        
        # Create combined plot
        create_combined_plot(df, emotion_var, emotion_label)
        
        print(f"\n{emotion_var} analysis completed!")
        print("-"*60)
    
    return all_results

def generate_emotion_summary_report(all_results):
    """Generate summary report for emotion variable analysis"""
    print("\n" + "="*80)
    print("Emotion Variable Moderation Effect Analysis Summary Report")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for emotion_var, emotion_results in all_results.items():
        for grouping_var, result in emotion_results.items():
            summary_data.append({
                'emotion_variable': emotion_var,
                'grouping_variable': grouping_var,
                'significant_moderation': result['significant_moderation'],
                'interaction_p_value': result['interaction_p']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Statistics grouped by emotion variable
    print("\nModeration effect statistics by emotion variable:")
    for emotion_var in summary_df['emotion_variable'].unique():
        emotion_data = summary_df[summary_df['emotion_variable'] == emotion_var]
        significant_count = emotion_data['significant_moderation'].sum()
        total_count = len(emotion_data)
        print(f"{emotion_var}: {significant_count}/{total_count} trait variables show significant moderation effect")
    
    # Statistics grouped by grouping variable
    print("\nModeration effect statistics by grouping variable:")
    for grouping_var in summary_df['grouping_variable'].unique():
        grouping_data = summary_df[summary_df['grouping_variable'] == grouping_var]
        significant_count = grouping_data['significant_moderation'].sum()
        total_count = len(grouping_data)
        print(f"{grouping_var}: {significant_count}/{total_count} emotion variables show significant moderation effect")
    
    # Save summary report
    try:
        summary_df.to_excel('emotion_moderation_analysis_summary.xlsx', index=False)
        print(f"\nSummary report saved to: emotion_moderation_analysis_summary.xlsx")
    except Exception as e:
        print(f"Failed to save summary report: {e}")

def main():
    """Main function"""
    print("Starting emotion variable moderation effect analysis...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run analysis for all emotion variables
    all_results = run_emotion_analysis(df)
    
    # Generate summary report
    generate_emotion_summary_report(all_results)
    
    print("\nEmotion variable moderation effect analysis completed!")

if __name__ == "__main__":
    main()
