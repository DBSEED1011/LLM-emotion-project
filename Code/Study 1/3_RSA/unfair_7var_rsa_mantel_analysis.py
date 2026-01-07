"""
Unfair 7-variable RSA Mantel Analysis Script
Functions: Data processing, RSA matrix calculation, Mantel analysis, Pearson analysis, result saving
Does not include visualization. For visualization, use unfair_7var_rsa_mantel_visualization.py

DATA FILE NOTE:
This script uses demo.xlsx as input data file, which contains a subset of the full dataset
(5 participants per group: human, gpt3.5, o3, V3, R1).

To use the full dataset:
1. Access the full dataset at: [INSERT LINK HERE]
2. Download the file named "Study1_experimental_data.xlsx"
3. Replace "demo.xlsx" with "Study1_experimental_data.xlsx" in the main() function

Note: When switching to full data, computation time and memory usage may increase significantly.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
import os
warnings.filterwarnings('ignore')

def create_condition_order(amount_of_allocation, amount_of_cost):
    """
    Create condition_order based on amount_of_allocation and amount_of_cost
    
    Mapping rules:
    - 1-10: allocation = 14, cost = 9-0
    - 11-20: allocation = 13, cost = 9-0
    - 21-30: allocation = 12, cost = 9-0
    - 31-40: allocation = 11, cost = 9-0
    - 41-50: allocation = 10, cost = 9-0
    """
    # Calculate base offset
    # allocation 14 -> 0, 13 -> 10, 12 -> 20, 11 -> 30, 10 -> 40
    allocation_offset = (14 - amount_of_allocation) * 10
    
    # cost 9 -> 0, 8 -> 1, ..., 0 -> 9
    cost_offset = 9 - amount_of_cost
    
    condition_order = allocation_offset + cost_offset + 1
    
    return condition_order

def process_unfair_data(input_file, output_file):
    """
    Process raw data to generate mean data in unfair all datasets means with emotions format
    
    Processing steps:
    1. Filter all unfair data (amount_of_allocation != 15)
    2. Create condition_order
    3. For each condition, group, and variable, calculate mean using raw scores
       (e.g., for condition=1, human_choice: filter human group, allocation=14, cost=9, 
        then calculate mean of all choice values)
    
    Parameters:
    input_file: Path to input data file (demo.xlsx for testing, Study1_experimental_data.xlsx for full data)
    output_file: Path to output processed data file
    """
    print(f"Reading raw data file: {input_file}")
    # Read data - Using demo data by default (see main() function DATA FILE NOTE for full data access)
    df = pd.read_excel(input_file)
    print(f"Raw data shape: {df.shape}")
    
    # Filter unfair data (amount_of_allocation != 15)
    unfair_df = df[df['amount_of_allocation'] != 15].copy()
    print(f"Filtered unfair data shape: {unfair_df.shape}")
    
    # Create condition_order
    unfair_df['condition_order'] = unfair_df.apply(
        lambda row: create_condition_order(row['amount_of_allocation'], row['amount_of_cost']),
        axis=1
    )
    
    # Map group names (gpt3.5 -> gpt35)
    unfair_df['group'] = unfair_df['group'].replace('gpt3.5', 'gpt35')
    
    # Define groups and variables
    # Group order: ['human', 'gpt35', 'V3', 'R1', 'o3'] to match compute_rsa_matrices_7var
    groups = ['human', 'gpt35', 'V3', 'R1', 'o3']
    variables = ['choice', 'AA_valence', 'AA_arousal', 'AC_valence', 'AC_arousal', 
                 'EmoFDBK_valence', 'EmoFDBK_arousal']
    
    # Calculate mean values for each condition using raw scores
    print("\nCalculating mean values for each condition (using raw scores)...")
    result_data = []
    
    for condition_order in range(1, 51):
        condition_data = {'condition_order': condition_order}
        # Filter data for this condition
        condition_df = unfair_df[unfair_df['condition_order'] == condition_order]
        
        if len(condition_df) == 0:
            # Fill with NaN
            for group in groups:
                for var in variables:
                    condition_data[f'{group}_{var}'] = np.nan
        else:
            # Calculate mean for each group
            for group in groups:
                group_condition_df = condition_df[condition_df['group'] == group]
                
                if len(group_condition_df) == 0:
                    # Fill with NaN
                    for var in variables:
                        condition_data[f'{group}_{var}'] = np.nan
                else:
                    # Calculate mean for each variable using raw scores
                    for var in variables:
                        mean_value = group_condition_df[var].mean()
                        condition_data[f'{group}_{var}'] = mean_value
        
        result_data.append(condition_data)
    
    # Create result DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Ensure column order matches target file
    columns_order = ['condition_order']
    for group in groups:
        for var in variables:
            columns_order.append(f'{group}_{var}')
    
    result_df = result_df[columns_order]
    
    # Save results
    result_df.to_excel(output_file, index=False)
    print(f"\nProcessing complete! Results saved to: {output_file}")
    print(f"Result data shape: {result_df.shape}")
    
    # Verify with examples
    print("\nVerification examples:")
    
    # Example 1: Condition 1 (allocation=14, cost=9)
    print("Condition 1, human_choice (allocation=14, cost=9):")
    cond1_human = unfair_df[(unfair_df['condition_order'] == 1) & (unfair_df['group'] == 'human')]
    if len(cond1_human) > 0:
        print(f"  Filtered rows: {len(cond1_human)}")
        print(f"  Allocation: {cond1_human['amount_of_allocation'].unique()}")
        print(f"  Cost: {cond1_human['amount_of_cost'].unique()}")
        print(f"  Mean choice (raw): {cond1_human['choice'].mean():.10f}")
        print(f"  Saved value: {result_df.loc[0, 'human_choice']:.10f}")
    
    # Example 2: Condition 10 (allocation=14, cost=0)
    print("\nCondition 10, human_choice (allocation=14, cost=0):")
    cond10_human = unfair_df[(unfair_df['condition_order'] == 10) & (unfair_df['group'] == 'human')]
    if len(cond10_human) > 0:
        print(f"  Filtered rows: {len(cond10_human)}")
        print(f"  Allocation: {cond10_human['amount_of_allocation'].unique()}")
        print(f"  Cost: {cond10_human['amount_of_cost'].unique()}")
        print(f"  Mean choice (raw): {cond10_human['choice'].mean():.10f}")
        print(f"  Saved value: {result_df.loc[9, 'human_choice']:.10f}")
    
    return result_df

def mantel_test(x, y, n_perm=10000, metric='pearson'):
    """
    Mantel test function - compare similarity of two matrices
    
    Parameters:
    x, y: matrices to compare
    n_perm: number of permutations
    metric: correlation measure ('pearson' or 'spearman')
    
    Returns:
    observed_r: observed correlation
    p_value: p-value
    """
    
    n = x.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    x_flat = x[upper_tri_indices]
    y_flat = y[upper_tri_indices]
    
    if metric == 'pearson':
        observed_r, _ = pearsonr(x_flat, y_flat)
    else:
        from scipy.stats import spearmanr
        observed_r, _ = spearmanr(x_flat, y_flat)
    
    # Permutation test
    permuted_rs = []
    for _ in range(n_perm):
        perm_indices = np.random.permutation(n)
        y_perm = y[np.ix_(perm_indices, perm_indices)]
        y_perm_flat = y_perm[upper_tri_indices]
        
        if metric == 'pearson':
            perm_r, _ = pearsonr(x_flat, y_perm_flat)
        else:
            from scipy.stats import spearmanr
            perm_r, _ = spearmanr(x_flat, y_perm_flat)
        
        permuted_rs.append(perm_r)
    
    p_value = np.mean(np.abs(permuted_rs) >= np.abs(observed_r))
    return observed_r, p_value

def compute_rsa_matrices_7var(data_df):
    """
    Calculate RSA correlation matrices for all groups (using 7 variables)
    
    Parameters:
    data_df: original data DataFrame
    
    Returns:
    rsa_matrices: dictionary containing RSA matrix for each group
    """
    groups = ['human', 'gpt35', 'V3', 'R1', 'o3']
    rsa_matrices = {}
    
    for group in groups:
        # Select corresponding columns based on group name (7 variables)
        if group == 'human':
            columns = ['human_choice', 'human_AA_valence', 'human_AA_arousal', 
                      'human_AC_valence', 'human_AC_arousal', 
                      'human_EmoFDBK_valence', 'human_EmoFDBK_arousal']
        elif group == 'gpt35':
            columns = ['gpt35_choice', 'gpt35_AA_valence', 'gpt35_AA_arousal',
                      'gpt35_AC_valence', 'gpt35_AC_arousal',
                      'gpt35_EmoFDBK_valence', 'gpt35_EmoFDBK_arousal']
        elif group == 'o3':
            columns = ['o3_choice', 'o3_AA_valence', 'o3_AA_arousal',
                      'o3_AC_valence', 'o3_AC_arousal',
                      'o3_EmoFDBK_valence', 'o3_EmoFDBK_arousal']
        elif group == 'V3':
            columns = ['V3_choice', 'V3_AA_valence', 'V3_AA_arousal',
                      'V3_AC_valence', 'V3_AC_arousal',
                      'V3_EmoFDBK_valence', 'V3_EmoFDBK_arousal']
        elif group == 'R1':
            columns = ['R1_choice', 'R1_AA_valence', 'R1_AA_arousal',
                      'R1_AC_valence', 'R1_AC_arousal',
                      'R1_EmoFDBK_valence', 'R1_EmoFDBK_arousal']
        
        # Extract group data and standardize
        # group_data shape: (50 conditions, 7 variables)
        group_data = data_df[columns].values.astype(float)
        group_data_std = (group_data - group_data.mean(axis=0)) / group_data.std(axis=0)
        
        # Calculate RSA correlation matrix
        # Compute correlations between 50 conditions (not between 7 variables)
        rsa_matrix = np.corrcoef(group_data_std)
        rsa_matrices[group] = rsa_matrix
        
        print(f"{group.upper()} RSA matrix: shape={rsa_matrix.shape}, range=[{rsa_matrix.min():.3f}, {rsa_matrix.max():.3f}]")
        print(f"  Variables used: {len(columns)} ({', '.join([col.split('_')[-1] for col in columns])})")
    
    return rsa_matrices


def rsa_mantel_analysis(rsa_matrices, analysis_name=""):
    """
    Perform RSA Mantel analysis
    
    Parameters:
    rsa_matrices: dictionary containing RSA matrices for all groups
    analysis_name: analysis name
    
    Returns:
    mantel_matrix: Mantel correlation matrix
    p_matrix: p-value matrix
    """
    groups = list(rsa_matrices.keys())
    n_groups = len(groups)
    
    mantel_matrix = np.zeros((n_groups, n_groups))
    p_matrix = np.zeros((n_groups, n_groups))
    
    print(f"\n=== {analysis_name}RSA Mantel Test Results ===")
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i != j:
                rsa1 = rsa_matrices[group1]
                rsa2 = rsa_matrices[group2]
                
                mantel_r, p_value = mantel_test(rsa1, rsa2, n_perm=10000)
                mantel_matrix[i, j] = mantel_r
                p_matrix[i, j] = p_value
                
                print(f"{group1.upper()} vs {group2.upper()}: r = {mantel_r:.3f}, p = {p_value:.3f}")
            else:
                mantel_matrix[i, j] = 1.0  # Diagonal
                p_matrix[i, j] = 0.0
    
    return mantel_matrix, p_matrix

def pearson_analysis(rsa_matrices, analysis_name=""):
    """
    Perform Pearson correlation analysis (flatten RSA matrices)
    
    Parameters:
    rsa_matrices: dictionary containing RSA matrices for all groups
    analysis_name: analysis name
    
    Returns:
    pearson_matrix: Pearson correlation matrix
    pearson_p_matrix: p-value matrix
    """
    groups = list(rsa_matrices.keys())
    n_groups = len(groups)
    
    pearson_matrix = np.zeros((n_groups, n_groups))
    pearson_p_matrix = np.zeros((n_groups, n_groups))
    
    print(f"\n=== {analysis_name}Pearson Correlation Test Results ===")
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i != j:
                rsa1 = rsa_matrices[group1]
                rsa2 = rsa_matrices[group2]
                
                # Flatten RSA matrices for Pearson correlation
                rsa1_flat = rsa1.flatten()
                rsa2_flat = rsa2.flatten()
                
                pearson_r, pearson_p = pearsonr(rsa1_flat, rsa2_flat)
                pearson_matrix[i, j] = pearson_r
                pearson_p_matrix[i, j] = pearson_p
                
                print(f"{group1.upper()} vs {group2.upper()}: r = {pearson_r:.3f}, p = {pearson_p:.3f}")
            else:
                pearson_matrix[i, j] = 1.0  # Diagonal
                pearson_p_matrix[i, j] = 0.0
    
    return pearson_matrix, pearson_p_matrix



def main():
    """
    Main function: perform Unfair 7-variable validation RSA Mantel analysis
    
    DATA FILE NOTE:
    This script uses demo.xlsx, which contains a subset of the full dataset
    (5 participants per group: human, gpt3.5, o3, V3, R1).
    
    To use the full dataset:
    1. Access the full dataset at: [INSERT LINK HERE]
    2. Download the file named "Study1_experimental_data.xlsx"
    3. Replace "demo.xlsx" with "Study1_experimental_data.xlsx" in the input_file path below
    
    Note: When switching to full data, you may need to adjust:
    - Computation time may increase significantly (especially for Mantel test permutations)
    - Memory usage may increase
    - RSA matrix calculations will be based on more participants per group
    """
    print("=" * 60)
    print("Unfair 7-variable validation RSA Mantel analysis")
    print("=" * 60)
    
    # Get script directory and set file paths
    # Using demo data - see DATA FILE NOTE above for full data access
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "demo.xlsx")
    output_file = os.path.join(script_dir, "processed_rsa_data.xlsx")
    
    print("\nStep 1: Process raw data, generate mean data")
    print("-" * 60)
    try:
        data_df = process_unfair_data(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Raw data file not found: {input_file}")
        print("Attempting to read processed data file...")
        try:
            data_df = pd.read_excel(output_file)
            print(f"Successfully read processed data file, shape: {data_df.shape}")
        except FileNotFoundError:
            print(f"Error: Processed data file also not found: {output_file}")
            return
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Check data quality
    print(f"\nStep 2: Data quality check")
    print("-" * 60)
    print(f"- Total rows: {len(data_df)}")
    print(f"- Total columns: {len(data_df.columns)}")
    print(f"- Missing values: {data_df.isnull().sum().sum()}")
    
    # Calculate 7-variable RSA matrices
    print(f"\nStep 3: Calculate RSA correlation matrices (7 variables)")
    print("-" * 60)
    rsa_matrices_7var = compute_rsa_matrices_7var(data_df)
    
    # Perform 7-variable RSA Mantel analysis
    print(f"\nStep 4: Perform RSA Mantel analysis")
    print("-" * 60)
    groups = ['human', 'gpt35', 'V3', 'R1', 'o3']
    mantel_matrix_7var, p_matrix_7var = rsa_mantel_analysis(rsa_matrices_7var, "Unfair 7-variable ")
    
    # Perform Pearson analysis
    print(f"\nStep 5: Perform Pearson correlation analysis")
    print("-" * 60)
    pearson_matrix_7var, pearson_p_matrix_7var = pearson_analysis(rsa_matrices_7var, "Unfair 7-variable ")
    
    # Save results to CSV
    print(f"\nStep 6: Save results")
    print("-" * 60)
    mantel_df_7var = pd.DataFrame(mantel_matrix_7var, index=groups, columns=groups)
    p_df_7var = pd.DataFrame(p_matrix_7var, index=groups, columns=groups)
    pearson_df_7var = pd.DataFrame(pearson_matrix_7var, index=groups, columns=groups)
    pearson_p_df_7var = pd.DataFrame(pearson_p_matrix_7var, index=groups, columns=groups)
    
    # Save files
    mantel_df_7var.to_csv(os.path.join(script_dir, 'unfair_7var_rsa_mantel_correlations.csv'))
    p_df_7var.to_csv(os.path.join(script_dir, 'unfair_7var_rsa_mantel_pvalues.csv'))
    pearson_df_7var.to_csv(os.path.join(script_dir, 'unfair_7var_rsa_pearson_correlations.csv'))
    pearson_p_df_7var.to_csv(os.path.join(script_dir, 'unfair_7var_rsa_pearson_pvalues.csv'))
    
    print("Results saved:")
    print("- processed_rsa_data.xlsx: Processed standardized mean data")
    print("- unfair_7var_rsa_mantel_correlations.csv: RSA Mantel correlation matrix")
    print("- unfair_7var_rsa_mantel_pvalues.csv: RSA Mantel p-value matrix")
    print("- unfair_7var_rsa_pearson_correlations.csv: Pearson correlation matrix")
    print("- unfair_7var_rsa_pearson_pvalues.csv: Pearson p-value matrix")
    
    print("\n" + "=" * 60)
    print("Unfair 7-variable RSA Mantel analysis completed!")
    print("Note: Run unfair_7var_rsa_mantel_visualization.py for visualization")
    print("=" * 60)

if __name__ == "__main__":
    main()
