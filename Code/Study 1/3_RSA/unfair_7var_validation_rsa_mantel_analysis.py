import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style and font
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['axes.unicode_minus'] = False

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
        group_data = data_df[columns].values.astype(float)
        group_data_std = (group_data - group_data.mean(axis=0)) / group_data.std(axis=0)
        
        # Calculate RSA correlation matrix
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

def plot_unfair_7var_validation_results(mantel_matrix_7var, p_matrix_7var, pearson_matrix_7var, pearson_p_matrix_7var, groups):
    """
    Plot Unfair 7-variable validation RSA Mantel and Pearson analysis results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 7-variable RSA Mantel correlation matrix
    sns.heatmap(mantel_matrix_7var, 
                xticklabels=groups, 
                yticklabels=groups,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                ax=ax1)
    ax1.set_title('Unfair 7 Variables RSA Mantel Correlations', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Groups', fontsize=14)
    ax1.set_ylabel('Groups', fontsize=14)
    
    # Plot 7-variable RSA Mantel p-value matrix
    sns.heatmap(p_matrix_7var, 
                xticklabels=groups, 
                yticklabels=groups,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                square=True,
                ax=ax2)
    ax2.set_title('Unfair 7 Variables RSA Mantel P-values', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Groups', fontsize=14)
    ax2.set_ylabel('Groups', fontsize=14)
    
    # Plot 7-variable Pearson correlation matrix
    sns.heatmap(pearson_matrix_7var, 
                xticklabels=groups, 
                yticklabels=groups,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                ax=ax3)
    ax3.set_title('Unfair 7 Variables Pearson Correlations', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Groups', fontsize=14)
    ax3.set_ylabel('Groups', fontsize=14)
    
    # Plot 7-variable Pearson p-value matrix
    sns.heatmap(pearson_p_matrix_7var, 
                xticklabels=groups, 
                yticklabels=groups,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                square=True,
                ax=ax4)
    ax4.set_title('Unfair 7 Variables Pearson P-values', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Groups', fontsize=14)
    ax4.set_ylabel('Groups', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('unfair_7var_validation_rsa_mantel_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\nUnfair 7-variable validation RSA Mantel and Pearson results plot saved to: unfair_7var_validation_rsa_mantel_results.png")
    plt.show()


def main():
    """
    Main function: perform Unfair 7-variable validation RSA Mantel analysis
    """
    print("Starting Unfair 7-variable validation RSA Mantel analysis...")
    print("Using data file: unfair_all_datasets_means_with_emotions_7_variables.xlsx")
    
    # Read data
    try:
        data_df = pd.read_excel("unfair_all_datasets_means_with_emotions_7_variables.xlsx")
        print(f"Successfully read data file, shape: {data_df.shape}")
        print(f"Data columns: {list(data_df.columns)}")
    except FileNotFoundError:
        print("Data file not found: unfair_all_datasets_means_with_emotions_7_variables.xlsx")
        return
    except Exception as e:
        print(f"Error reading data file: {e}")
        return
    
    # Check data quality
    print(f"\nData overview:")
    print(f"- Total rows: {len(data_df)}")
    print(f"- Total columns: {len(data_df.columns)}")
    print(f"- Missing values: {data_df.isnull().sum().sum()}")
    
    # Calculate 7-variable RSA matrices
    print("\nCalculating RSA correlation matrices (7 variables)...")
    rsa_matrices_7var = compute_rsa_matrices_7var(data_df)
    
    # Perform 7-variable RSA Mantel analysis
    groups = ['human', 'gpt35', 'V3', 'R1', 'o3']
    mantel_matrix_7var, p_matrix_7var = rsa_mantel_analysis(rsa_matrices_7var, "Unfair 7-variable ")
    
    # Perform Pearson analysis
    pearson_matrix_7var, pearson_p_matrix_7var = pearson_analysis(rsa_matrices_7var, "Unfair 7-variable ")
    
    # Plot results
    print("\nPlotting Unfair 7-variable validation RSA Mantel and Pearson analysis results...")
    plot_unfair_7var_validation_results(mantel_matrix_7var, p_matrix_7var, pearson_matrix_7var, pearson_p_matrix_7var, groups)
    
    # Save results to CSV
    mantel_df_7var = pd.DataFrame(mantel_matrix_7var, index=groups, columns=groups)
    p_df_7var = pd.DataFrame(p_matrix_7var, index=groups, columns=groups)
    pearson_df_7var = pd.DataFrame(pearson_matrix_7var, index=groups, columns=groups)
    pearson_p_df_7var = pd.DataFrame(pearson_p_matrix_7var, index=groups, columns=groups)
    
    # Save files
    mantel_df_7var.to_csv('unfair_7var_validation_rsa_mantel_correlations.csv')
    p_df_7var.to_csv('unfair_7var_validation_rsa_mantel_pvalues.csv')
    pearson_df_7var.to_csv('unfair_7var_validation_rsa_pearson_correlations.csv')
    pearson_p_df_7var.to_csv('unfair_7var_validation_rsa_pearson_pvalues.csv')
    
    print("\nUnfair 7-variable validation analysis results saved:")
    print("- unfair_7var_validation_rsa_mantel_correlations.csv: RSA Mantel correlation matrix")
    print("- unfair_7var_validation_rsa_mantel_pvalues.csv: RSA Mantel p-value matrix")
    print("- unfair_7var_validation_rsa_pearson_correlations.csv: Pearson correlation matrix")
    print("- unfair_7var_validation_rsa_pearson_pvalues.csv: Pearson p-value matrix")
    print("- unfair_7var_validation_rsa_mantel_results.png: Unfair 7-variable validation RSA Mantel and Pearson results plot")
    
    print("\nUnfair 7-variable validation RSA Mantel and Pearson analysis completed!")

if __name__ == "__main__":
    main()
