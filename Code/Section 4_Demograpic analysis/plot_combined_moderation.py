#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create combined moderation effect plots: comparison of choice means for 7 traits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

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

def create_combined_plot(df):
    """Create combined moderation effect plots"""
    print("Starting to create combined plot...")
    
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
        
        # Calculate mean choice for each group and trait level
        mean_data = df.groupby([var, 'group'])['choice'].mean().reset_index()
        
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
                        values.append(group_data['choice'].iloc[0])
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
        #ax.set_xlabel('Group', fontsize=10)
        ax.set_ylabel('Punishment Probability', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper right')
        
        # Add background color to distinguish human and LLM
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='Human')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue', label='LLMs')
    
    # Hide the 8th subplot (only 7 variables)
    axes[7].set_visible(False)
    
    # Set overall title
    #fig.suptitle('Moderating Role of Participant Group in Trait-Behavior Associations\nChoice Means by Trait Level and Group', 
                 #fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # Save plot
    plt.savefig('combined_choice_moderation_effects.png', dpi=300, bbox_inches='tight')
    print("Combined plot saved: combined_choice_moderation_effects.png")
    
    plt.show()

def create_individual_plots(df):
    """Create individual plots (optional)"""
    print("\nCreating individual plots...")
    
    group_order, group_labels = setup_group_order()
    
    grouping_vars = {
        'gender': 'Gender',
        'SVO': 'Social Value Orientation', 
        'CESD_2g': 'Depressive Symptoms',
        'AQ_2g': 'Autism Spectrum Traits',
        'age_2g': 'Age',
        'JSI_2g': 'Joint Attention',
        'ERS_2g': 'Emotional Reactivity'
    }
    
    for var, label in grouping_vars.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate mean data
        mean_data = df.groupby([var, 'group'])['choice'].mean().reset_index()
        mean_data['group'] = pd.Categorical(mean_data['group'], categories=group_order, ordered=True)
        mean_data = mean_data.sort_values('group')
        
        # Get trait levels
        trait_levels = sorted(mean_data[var].unique())
        if pd.isna(trait_levels).any():
            trait_levels = [x for x in trait_levels if pd.notna(x)]
        
        # Create bar plot
        x_pos = np.arange(len(group_order))
        width = 0.35
        
        for j, level in enumerate(trait_levels):
            if pd.notna(level):
                level_data = mean_data[mean_data[var] == level]
                
                values = []
                for group in group_order:
                    group_data = level_data[level_data['group'] == group]
                    if len(group_data) > 0:
                        values.append(group_data['choice'].iloc[0])
                    else:
                        values.append(0)
                
                offset = (j - 0.5) * width
                bars = ax.bar(x_pos + offset, values, width, 
                             label=str(level), 
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Set plot properties
        ax.set_title(f'{label} × Group Interaction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Choice Mean', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add background color
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue')
        
        plt.tight_layout()
        filename = f'individual_{var}_moderation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Individual plot saved: {filename}")
        plt.show()

def main():
    """Main function"""
    print("Starting to create moderation effect plots...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create combined plot
    create_combined_plot(df)
    
    # Optional: create individual plots
    # create_individual_plots(df)
    
    print("\nPlot creation completed!")

if __name__ == "__main__":
    main()
