import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Read participant level results data containing all groups
participant_df = pd.read_csv('participant_level_results_all_groups.csv')

print("=== Partial Correlation Analysis: Choice Mean vs EmoFDBK_valence Mean (Reference unfair format) ===")
print("Control variable: cost_level")
print("Sample 1/4 of points for each condition combination of each group\n")

def calculate_partial_correlation(x, y, z):
    """Calculate partial correlation coefficient"""
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    z_clean = z[valid_mask]
    
    if len(x_clean) < 3:
        return np.nan, np.nan
    
    # Calculate simple correlation coefficients
    r_xy, _ = pearsonr(x_clean, y_clean)
    r_xz, _ = pearsonr(x_clean, z_clean)
    r_yz, _ = pearsonr(y_clean, z_clean)
    
    # Calculate partial correlation coefficient
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    
    if denominator == 0:
        return np.nan, np.nan
    
    partial_r = numerator / denominator
    
    # Calculate significance
    n = len(x_clean)
    df_degrees = n - 3
    
    if df_degrees <= 0:
        return partial_r, np.nan
    
    t_stat = partial_r * np.sqrt(df_degrees / (1 - partial_r**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_degrees))
    
    return partial_r, p_value

def calculate_partial_regression_slope(x, y, z):
    """Calculate partial regression coefficient (correct slope based on partial correlation coefficient)"""
    partial_r, p_value = calculate_partial_correlation(x, y, z)
    
    if np.isnan(partial_r):
        return np.nan, np.nan, np.nan, np.nan
    
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Partial regression coefficient = partial correlation coefficient × (Y standard deviation / X standard deviation)
    x_std = np.std(x_clean)
    y_std = np.std(y_clean)
    partial_slope = partial_r * (y_std / x_std)
    
    return partial_r, p_value, partial_slope, len(x_clean)

# Define group colors (reference unfair format)
palette = {
    'human': '#4C72B0',
    'gpt3.5': '#DD8452', 
    'o3': '#55A868',
    'V3': '#C44E52',
    'R1': '#8172B3'
}

# Define scatter plot styles (reference unfair format)
markers = {
    'human': 'o',        # Circle
    'gpt3.5': 's',      # Square
    'o3': '^',          # Triangle
    'V3': 'D',          # Diamond
    'R1': 'v'           # Inverted triangle
}

# Define group display names
group_display_names = {
    'human': 'Human',
    'gpt3.5': 'GPT-3.5',
    'o3': 'o3-mini',
    'V3': 'DeepSeek-V3',
    'R1': 'DeepSeek-R1'
}

# Convert cost_level to numeric values (for partial correlation calculation)
participant_df['cost_level_numeric'] = participant_df['cost_level'].map({'low': 0, 'high': 1})

print("Data overview:")
print(f"Total number of participants: {len(participant_df['participant_id'].unique())}")
print(f"Total number of data points: {len(participant_df)}")

# Store all results
all_results = []

# Create separate plots for each condition
conditions = ['unfair', 'fair']

for condition in conditions:
    # Create separate plots for each condition
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))  # Single plot size
    condition_data = participant_df[participant_df['fairness_group'] == condition]
    
    print(f"\n--- {condition.upper()} Condition ---")
    
    # Draw scatter plots and fitting lines for each group
    for group_name in condition_data['group'].unique():
        if pd.isna(group_name):
            continue
            
        group_data = condition_data[condition_data['group'] == group_name]
        
        x = group_data['choice_mean'].values      # Use choice mean as x-axis
        y = group_data['EmoFDBK_valence_mean'].values  # Use EmoFDBK_valence mean as y-axis
        z = group_data['cost_level_numeric'].values  # Use cost_level as control variable
        
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        z_clean = z[valid_mask]
        
        if len(x_clean) == 0:
            continue
        
        # Calculate partial correlation coefficient and partial regression coefficient
        partial_r, p_value, partial_slope, n = calculate_partial_regression_slope(x, y, z)
        
        if not np.isnan(partial_r):
            # Sample 1/4 of points for display (reference unfair format)
            n_total = len(x_clean)
            n_sample = max(1, n_total // 4)  # Sample 1/4 of points
            
            # Random sampling
            np.random.seed(42)  # Set random seed to ensure reproducible results
            sample_indices = np.random.choice(n_total, n_sample, replace=False)
            
            x_sampled = x_clean[sample_indices]
            y_sampled = y_clean[sample_indices]
            
            print(f"{group_name}: Total data points={n_total}, Sampled for display={n_sample}")
            
            # Add random jitter to y values (reference unfair format)
            jitter_amount = 5.0  # Jitter amount (adjusted for valence values)
            y_jittered = y_sampled + np.random.normal(0, jitter_amount, len(y_sampled))
            
            # Draw scatter plot (reference unfair format style)
            ax.scatter(x_sampled, y_jittered, 
                      alpha=0.15, 
                      s=7, 
                      marker=markers[group_name],
                      color=palette[group_name], 
                      label='_nolegend_')  # Don't show scatter plot legend
            
            # Use partial regression coefficient to draw fitting line (calculated based on all data points)
            x_mean = np.mean(x_clean)
            y_mean = np.mean(y_clean)
            
            # Generate x value range
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            
            # Calculate fitting line: y = y_mean + partial_slope * (x - x_mean)
            y_fit = y_mean + partial_slope * (x_range - x_mean)
            
            # Draw fitting line (reference unfair format style)
            # First draw black outline
            ax.plot(x_range, y_fit, 
                   color='black', 
                   linewidth=2.5, 
                   alpha=1.0)
            
            # Then draw colored fitting curve
            ax.plot(x_range, y_fit, 
                   color=palette[group_name], 
                   linewidth=2.0, 
                   alpha=0.8,
                   label=group_display_names[group_name])
            
            # Calculate confidence interval (reference unfair format bootstrap method)
            n_bootstrap = 1000
            y_bootstrap = np.zeros((n_bootstrap, len(x_range)))
            
            for i in range(n_bootstrap):
                # Random sampling (with replacement)
                indices = np.random.choice(len(x_clean), len(x_clean), replace=True)
                x_boot = x_clean[indices]
                y_boot = y_clean[indices]
                
                # Calculate partial regression coefficient
                try:
                    partial_r_boot, _, partial_slope_boot, _ = calculate_partial_regression_slope(x_boot, y_boot, z_clean[indices])
                    if not np.isnan(partial_slope_boot):
                        x_mean_boot = np.mean(x_boot)
                        y_mean_boot = np.mean(y_boot)
                        y_bootstrap[i, :] = y_mean_boot + partial_slope_boot * (x_range - x_mean_boot)
                except:
                    continue
            
            # Calculate standard deviation (for building confidence interval)
            y_std = np.std(y_bootstrap, axis=0)
            
            # Draw confidence interval (reference unfair format)
            ax.fill_between(x_range, 
                           y_fit - y_std, 
                           y_fit + y_std, 
                           color=palette[group_name], 
                           alpha=0.15)
            
            # Significance marking
            if p_value < 0.001:
                sig_mark = "***"
            elif p_value < 0.01:
                sig_mark = "**"
            elif p_value < 0.05:
                sig_mark = "*"
            else:
                sig_mark = "ns"
            
            print(f"{group_name}: Partial correlation r={partial_r:.3f}, Partial regression slope={partial_slope:.6f}, p={p_value:.3f}, n={n}")
            
            # Store results
            all_results.append({
                'condition': condition,
                'group': group_name,
                'display_name': group_display_names[group_name],
                'partial_r': partial_r,
                'p_value': p_value,
                'partial_slope': partial_slope,
                'n': n,
                'significance': sig_mark
            })
    
    # Set subplot properties (reference unfair format)
    ax.set_xlabel('P(Punishment)', fontsize=7)
    ax.set_ylabel('Emotional outcome valence', fontsize=7)
    ax.set_title(f'{condition.capitalize()} condition(controlling for cost level)', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-100, 100)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add simplified legend for each subplot (only using colors to represent groups)
    from matplotlib.lines import Line2D
    
    # Create simplified legend elements (only using colors)
    legend_elements = []
    for group_name in palette.keys():
        legend_elements.append(
            Line2D([0], [0], 
                   color=palette[group_name], 
                   linewidth=2.5,  # Slightly reduce line thickness
                   label=group_display_names[group_name])
        )
    
    # Add legend for current subplot, place in lower left corner, reduce overall size
    ax.legend(handles=legend_elements, fontsize=5, loc='lower left', 
              frameon=True, framealpha=0.9, ncol=1,
              handlelength=1.5,  # Shorten color bar length
              handletextpad=0.3,  # Reduce spacing between color bar and text
              columnspacing=0.5,  # Reduce column spacing
              borderpad=0.3,  # Reduce border padding
              labelspacing=0.3)  # Reduce spacing between labels
    
    # Adjust layout and save separate image
    plt.tight_layout()
    filename = f'partial_correlation_emofdbk_{condition}_format.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Chart saved as: {filename}")

# Create results summary table
results_df = pd.DataFrame(all_results)

print("\n=== Partial Correlation Analysis Results Summary ===")
print(results_df.to_string(index=False, float_format='%.4f'))

# Save results to CSV
results_df.to_csv('partial_correlation_emofdbk_valence_results.csv', index=False)
print("\nResults saved to: partial_correlation_emofdbk_valence_results.csv")

print(f"\n=== Key Notes ===")
print("1. Use participant level choice mean and EmoFDBK_valence mean")
print("2. Sample 1/4 of points for each condition combination of each group for display")
print("3. Fitting lines and confidence intervals calculated based on all data points")
print("4. Plot format completely references unfair_combined_scatter_plot")
print("5. Partial correlation coefficient: correlation strength after controlling for cost_level")
print("6. Charts saved separately as: partial_correlation_emofdbk_unfair_format.png and partial_correlation_emofdbk_fair_format.png")
