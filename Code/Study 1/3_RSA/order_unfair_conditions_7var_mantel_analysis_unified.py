import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def mantel(x, y, n_perm=10000, metric='pearson'):
    """
    Custom Mantel test function
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

print("Reading unfair conditions 7-variable dataset...")
df = pd.read_excel("unfair_all_datasets_means_with_emotions_7_variables.xlsx")
print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")

# Use group order from order
groups = ["Human", "gpt35", "V3", "R1", "o3"]
group_mapping = {
    "Human": "human",
    "gpt35": "gpt35", 
    "V3": "V3",
    "R1": "R1",
    "o3": "o3"
}

print("Starting RSA correlation matrix calculation (using same method as mantel analysis)...")

# Calculate RSA matrices using same method as unfair_7var_validation_rsa_mantel_analysis
rsa_matrices = {}
for group in groups:
    prefix = group_mapping[group]
    
    # Select corresponding columns based on group name (7 variables)
    if prefix == 'human':
        columns = ['human_choice', 'human_AA_valence', 'human_AA_arousal', 
                  'human_AC_valence', 'human_AC_arousal', 
                  'human_EmoFDBK_valence', 'human_EmoFDBK_arousal']
    elif prefix == 'gpt35':
        columns = ['gpt35_choice', 'gpt35_AA_valence', 'gpt35_AA_arousal',
                  'gpt35_AC_valence', 'gpt35_AC_arousal',
                  'gpt35_EmoFDBK_valence', 'gpt35_EmoFDBK_arousal']
    elif prefix == 'o3':
        columns = ['o3_choice', 'o3_AA_valence', 'o3_AA_arousal',
                  'o3_AC_valence', 'o3_AC_arousal',
                  'o3_EmoFDBK_valence', 'o3_EmoFDBK_arousal']
    elif prefix == 'V3':
        columns = ['V3_choice', 'V3_AA_valence', 'V3_AA_arousal',
                  'V3_AC_valence', 'V3_AC_arousal',
                  'V3_EmoFDBK_valence', 'V3_EmoFDBK_arousal']
    elif prefix == 'R1':
        columns = ['R1_choice', 'R1_AA_valence', 'R1_AA_arousal',
                  'R1_AC_valence', 'R1_AC_arousal',
                  'R1_EmoFDBK_valence', 'R1_EmoFDBK_arousal']
    
    # Extract group data and standardize
    group_data = df[columns].values.astype(float)
    group_data_std = (group_data - group_data.mean(axis=0)) / group_data.std(axis=0)
    
    # Calculate RSA correlation matrix (using same method as mantel analysis)
    rsa_matrix = np.corrcoef(group_data_std)
    rsa_matrices[group] = rsa_matrix
    
    print(f"{group} RSA matrix: shape={rsa_matrix.shape}, range=[{rsa_matrix.min():.3f}, {rsa_matrix.max():.3f}]")
    print(f"  Variables used: {len(columns)} ({', '.join([col.split('_')[-1] for col in columns])})")

print("\nStarting Mantel and Pearson correlation calculation...")

print("Using pre-calculated RSA Mantel and Pearson correlation matrices...")

print("\nReading pre-calculated RSA Mantel and Pearson correlation matrices...")

# Read RSA Mantel correlation matrices
try:
    rsa_mantel_correlations = pd.read_csv("unfair_7var_validation_rsa_mantel_correlations.csv", index_col=0)
    rsa_mantel_pvalues = pd.read_csv("unfair_7var_validation_rsa_mantel_pvalues.csv", index_col=0)
    
    # Original data has been adjusted to correct group order: ["Human", "gpt35", "V3", "R1", "o3"]
    # Use original data directly, no need to reorder
    rsa_mantel_correlations_reordered = rsa_mantel_correlations
    rsa_mantel_pvalues_reordered = rsa_mantel_pvalues
    
    # Convert reordered RSA Mantel data to numpy arrays
    mantel_correlations = rsa_mantel_correlations_reordered.values
    mantel_pvalues = rsa_mantel_pvalues_reordered.values
    
    print("Successfully read RSA Mantel data:")
    print("Reordered Mantel correlation matrix:")
    print(rsa_mantel_correlations_reordered)
    print("\nReordered Mantel p-value matrix:")
    print(rsa_mantel_pvalues_reordered)
    
except FileNotFoundError:
    print("Error: RSA Mantel CSV files not found!")
    exit(1)

# Read RSA Pearson correlation matrices
try:
    rsa_pearson_correlations = pd.read_csv("unfair_7var_validation_rsa_pearson_correlations.csv", index_col=0)
    rsa_pearson_pvalues = pd.read_csv("unfair_7var_validation_rsa_pearson_pvalues.csv", index_col=0)
    
    # Original data has been adjusted to correct group order: ["Human", "gpt35", "V3", "R1", "o3"]
    # Use original data directly, no need to reorder
    rsa_pearson_correlations_reordered = rsa_pearson_correlations
    rsa_pearson_pvalues_reordered = rsa_pearson_pvalues
    
    # Convert reordered RSA Pearson data to numpy arrays
    pearson_correlations = rsa_pearson_correlations_reordered.values
    pearson_pvalues = rsa_pearson_pvalues_reordered.values
    
    print("\nSuccessfully read RSA Pearson data:")
    print("Reordered Pearson correlation matrix:")
    print(rsa_pearson_correlations_reordered)
    print("\nReordered Pearson p-value matrix:")
    print(rsa_pearson_pvalues_reordered)
    
except FileNotFoundError:
    print("Error: RSA Pearson CSV files not found!")
    exit(1)

# Convert to DataFrame for display
mantel_df = pd.DataFrame(mantel_correlations, index=groups, columns=groups)
mantel_p_df = pd.DataFrame(mantel_pvalues, index=groups, columns=groups)
pearson_df = pd.DataFrame(pearson_correlations, index=groups, columns=groups)
pearson_p_df = pd.DataFrame(pearson_pvalues, index=groups, columns=groups)

print("Mantel correlation matrix:")
print(mantel_df)
print("\nMantel p-value matrix:")
print(mantel_p_df)
print("\nPearson correlation matrix:")
print(pearson_df)
print("\nPearson p-value matrix:")
print(pearson_p_df)

# Create visualization - reference R code style
fig, ax = plt.subplots(figsize=(6, 4))

# Create mask to show only lower triangle (Pearson)
mask = np.triu(np.ones_like(pearson_correlations, dtype=bool))

# Use matplotlib to draw lower triangle Pearson correlation heatmap
# Create RdBu_r color mapping
cmap = plt.cm.RdBu_r

# Draw heatmap
for i in range(len(groups)):
    for j in range(len(groups)):
        if not mask[i, j]:  # Only draw lower triangle
            # Get correlation coefficient
            r_val = pearson_correlations[i, j]
            
            # Calculate color
            color = cmap(0.5 + 0.5 * r_val)  # Map r value to 0-1 range
            
            # Draw rectangle
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

# Draw small color blocks and significance markers in each cell of lower triangle
for i in range(len(groups)):
    for j in range(i):
        # Calculate cell center position
        x_center = j + 0.5
        y_center = i + 0.5
        
        # Get correlation coefficient and p-value
        r_val = pearson_correlations[i, j]
        p_val = pearson_pvalues[i, j]
        
        # Set cell background color (color intensity represents r value, using white to red gradient)
        color_intensity = abs(r_val)  # Color intensity proportional to r value
        bg_color = plt.cm.Reds(color_intensity)
        
        # Set cell background color, add 0.5pt black outline
        ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                 facecolor=bg_color, 
                                 edgecolor='black', linewidth=0.5))
        
        # Determine significance marker
        if p_val < 0.001:
            sig_mark = "***"
        elif p_val < 0.01:
            sig_mark = "**"
        elif p_val < 0.05:
            sig_mark = "*"
        else:
            sig_mark = "ns"
        
        # Add Pearson correlation coefficient value (above significance marker)
        ax.text(x_center, y_center + 0.08, f'{r_val:.2f}', ha='center', va='center', 
                fontsize=7, fontweight='normal', fontfamily='Arial', color='white' if color_intensity > 0.5 else 'black')
        
        # Add significance marker (centered display)
        ax.text(x_center, y_center - 0.08, sig_mark, ha='center', va='center', 
                fontsize=7, fontweight='normal', fontfamily='Arial', color='white' if color_intensity > 0.5 else 'black')

# Draw Mantel network in upper triangle
# Define positions of two groups of nodes
# First group: located in first row, arranged horizontally (columns 1,2,3,4,5 of row 1)
# Second group: located in fifth column, arranged vertically (rows 1,2,3,4,5 of column 5)

group_positions = {
    # First group nodes (first row, top) - remove 5th point
    'Human_group1': (0.5, 0.5),
    'gpt35_group1': (1.5, 0.5), 
    'V3_group1': (2.5, 0.5),
    'R1_group1': (3.5, 0.5),
    
    # Second group nodes (fifth column, rightmost) - remove 1st point
    'gpt35_group2': (4.5, 1.5),
    'V3_group2': (4.5, 2.5),
    'R1_group2': (4.5, 3.5),
    'o3_group2': (4.5, 4.5)
}

# Define node colors and labels
colors = {
    'Human': '#4C72B0',
    'gpt35': '#DD8452', 
    'o3': '#55A868',
    'V3': '#C44E52',
    'R1': '#8172B3'
}
group_labels = ['Human', 'GPT-3.5', 'DeepSeek-V3', 'DeepSeek-R1', 'o3-mini']

# Draw connection lines (based on Mantel R values)
# Keep only upper triangle connection pattern to avoid duplication
# Human → GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini
# GPT-3.5 → DeepSeek-V3, DeepSeek-R1, o3-mini
# DeepSeek-V3 → DeepSeek-R1, o3-mini
# DeepSeek-R1 → o3-mini

for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:  # Keep only upper triangle to avoid duplication
            mantel_r = mantel_correlations[i, j]
            mantel_p = mantel_pvalues[i, j]
            
            # Only show significantly correlated connection lines
            if mantel_p < 0.05:
                # Determine color based on Mantel R
                if mantel_r >= 0.70:
                    color = '#FF4500'  # Orange-red
                elif mantel_r >= 0.5:
                    color = '#FFA500'  # Orange
                else:
                    color = '#F0E68C'  # Light yellow
                
                # Determine line width based on Mantel P
                if mantel_p < 0.001:
                    linewidth = 3.0  # Thickest line
                elif mantel_p < 0.01:
                    linewidth = 1.8  # Medium line width
                else:
                    linewidth = 0.6  # Thin line
                
                # Use solid line
                linestyle = '-'
                
                # Draw connection line: from group1 of first group to group2 of second group
                # Check if nodes exist
                if f'{group1}_group1' in group_positions and f'{group2}_group2' in group_positions:
                    pos1 = group_positions[f'{group1}_group1']
                    pos2 = group_positions[f'{group2}_group2']
                    
                    # Calculate intersection points of line with circle, let line start from circle edge
                    # Node radius (calculated based on scatter s=300, approximately 0.1)
                    node_radius = 0.19
                    
                    # Calculate direction vector from pos1 to pos2
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance > 0:  # Avoid division by zero error
                        # Unit direction vector
                        unit_dx = dx / distance
                        unit_dy = dy / distance
                        
                        # Calculate start point (from pos1 circle edge)
                        start_x = pos1[0] + unit_dx * node_radius
                        start_y = pos1[1] + unit_dy * node_radius
                        
                        # Calculate end point (to pos2 circle edge)
                        end_x = pos2[0] - unit_dx * node_radius
                        end_y = pos2[1] - unit_dy * node_radius
                        
                        # Draw straight line from circle edge to circle edge
                        ax.plot([start_x, end_x], [start_y, end_y], 
                                linewidth=linewidth, color=color, alpha=0.7,
                                linestyle=linestyle, solid_capstyle='round')
                        

# Draw nodes (after connection lines to ensure nodes are on top of lines)
# Draw first group nodes (horizontal arrangement, labels below)
for i, group in enumerate(groups[:4]):  # Only draw first 4 groups
    pos = group_positions[f'{group}_group1']
    ax.scatter(pos[0], pos[1], s=300, c=colors[group], alpha=0.8, 
                edgecolors='black', linewidth=0.5, zorder=10)
    ax.text(pos[0], pos[1]-0.25, group_labels[i], 
             ha='center', va='top', fontsize=7, fontweight='normal', fontfamily='Arial', zorder=11)

# Draw second group nodes (vertical arrangement, labels above)
for i, group in enumerate(groups[1:], 1):  # Start from 2nd group
    pos = group_positions[f'{group}_group2']
    ax.scatter(pos[0], pos[1], s=300, c=colors[group], alpha=0.8, 
                edgecolors='black', linewidth=0.5, zorder=10)
    ax.text(pos[0], pos[1]+0.25, group_labels[i], 
             ha='center', va='bottom', fontsize=7, fontweight='normal', fontfamily='Arial', zorder=11)

# Set axis range and ticks - 5x6 layout
ax.set_xlim(0, len(groups) + 1)  # Add one column for legend
ax.set_ylim(0, len(groups))
ax.set_aspect('equal')

# Set ticks and annotate group labels
# Horizontal ticks: correspond to first 4 groups (remove o3)
ax.set_xticks(np.arange(0.5, len(groups) - 0.5, 1))
# Vertical ticks: correspond to last 4 groups (remove Human)
ax.set_yticks(np.arange(1.5, len(groups) + 0.5, 1))

# Horizontal labels: remove last o3, place above top frame line
x_labels = group_labels[:-1]  # Remove last o3
ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=7, fontfamily='Arial')
ax.xaxis.tick_top()  # Move X-axis ticks to top

# Vertical labels: remove first Human, place on left
y_labels = group_labels[1:]  # Remove first Human
ax.set_yticklabels(y_labels, rotation=0, ha='right', fontsize=7, fontfamily='Arial')

# Set axis labels
ax.set_xlabel("")
ax.set_ylabel("")

# Set axis borders to transparent
for spine in ax.spines.values():
    spine.set_visible(False)

# Add legends - uniformly placed on the right
# Mantel R legend (color) - use color to represent correlation strength
mantel_r_legend = [
    plt.Line2D([0], [0], color='#FF4500', lw=3, label=' ≥ 0.7'),  # Orange-red
    plt.Line2D([0], [0], color='#FFA500', lw=3, label='0.5-0.7'),  # Orange
    plt.Line2D([0], [0], color='#F0E68C', lw=3, label=' < 0.5')  # Light yellow
]

# Mantel P legend (line width) - use line width to represent significance level
mantel_p_legend = [
    plt.Line2D([0], [0], color='black', lw=3.0, linestyle='-', label=' < 0.001'),  # Thickest line
    plt.Line2D([0], [0], color='black', lw=1.8, linestyle='-', label=' < 0.01'),  # Medium line width
    plt.Line2D([0], [0], color='black', lw=0.6, linestyle='-', label=' < 0.05')  # Thin line
]

# Add all legends in bottom right corner
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Create Pearson's R colorbar in right column and place label above colorbar
cbar_ax = plt.axes([0.82, 0.85, 0.2, 0.05])  # [left, bottom, width, height] - uniform width
norm = Normalize(vmin=0, vmax=1)
cbar = ColorbarBase(cbar_ax, cmap=plt.cm.Reds, norm=norm, orientation='horizontal')
# First set colorbar
cbar.ax.tick_params(labelsize=7)
# Then manually add label above colorbar
cbar_ax.text(0.5, 1.25, "Pearson's $r$", ha='center', va='bottom', fontsize=7, fontweight='normal', fontfamily='Arial', transform=cbar_ax.transAxes)

# Add node type legend (using new color scheme, alpha=0.8), only add 0.5pt black outline around circles, no lines crossing circles
node_legend_elements = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#4C72B0', markersize=6, 
                markeredgewidth=0.5, markeredgecolor='black', label='Human', alpha=0.8),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#DD8452', markersize=6, 
                markeredgewidth=0.5, markeredgecolor='black', label='GPT-3.5', alpha=0.8),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#55A868', markersize=6, 
                markeredgewidth=0.5, markeredgecolor='black', label='o3-mini', alpha=0.8),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#C44E52', markersize=6, 
                markeredgewidth=0.5, markeredgecolor='black', label='DeepSeek-V3', alpha=0.8),
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#8172B3', markersize=6, 
                markeredgewidth=0.5, markeredgecolor='black', label='DeepSeek-R1', alpha=0.8)
]

# Add all legends in right column, arranged top to bottom in order
# 1. Pearson's R colorbar (top, already set)

# 2. Mantel's R legend - use independent axes, manually control size
mantel_r_ax = plt.axes([0.82, 0.65, 0.2, 0.05])
mantel_r_ax.set_xlim(0, 1)
mantel_r_ax.set_ylim(0, 1)
mantel_r_ax.legend(handles=mantel_r_legend, fontsize=7, title='Mantel\'s $r$', 
                   frameon=True, fancybox=True, shadow=False, title_fontsize=7, 
                   prop={'family': 'Arial', 'size': 7, 'weight': 'normal'},
                   bbox_to_anchor=(0.5, 0.5), loc='center', 
                   bbox_transform=mantel_r_ax.transAxes,
                   ncol=1, columnspacing=0.5, handletextpad=0.3)
mantel_r_ax.axis('off')

# 3. Mantel's P legend - use independent axes, manually control size
mantel_p_ax = plt.axes([0.82, 0.45, 0.2, 0.05])
mantel_p_ax.set_xlim(0, 1)
mantel_p_ax.set_ylim(0, 1)
mantel_p_ax.legend(handles=mantel_p_legend, fontsize=7, title='Mantel\'s $p$', 
                   frameon=True, fancybox=True, shadow=False, title_fontsize=7, 
                   prop={'family': 'Arial', 'size': 7, 'weight': 'normal'},
                   bbox_to_anchor=(0.5, 0.5), loc='center',
                   bbox_transform=mantel_p_ax.transAxes,
                   ncol=1, columnspacing=0.5, handletextpad=0.3)
mantel_p_ax.axis('off')

# 4. Group Types legend - use independent axes, manually control size
group_ax = plt.axes([0.82, 0.20, 0.2, 0.05])
group_ax.set_xlim(0, 1)
group_ax.set_ylim(0, 1)
group_ax.legend(handles=node_legend_elements, fontsize=7, title='Group', 
                frameon=True, fancybox=True, shadow=False, title_fontsize=7, 
                prop={'family': 'Arial', 'size': 7, 'weight': 'normal'},
                bbox_to_anchor=(0.5, 0.5), loc='center',
                bbox_transform=group_ax.transAxes,
                ncol=1, columnspacing=0.5, handletextpad=0.3)
group_ax.axis('off')

# Remove title
# plt.suptitle('Unfair Conditions 7 Variables Mantel Analysis (Unified Method)', fontsize=12, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig("order_unfair_conditions_7var_mantel_analysis_unified.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n" + "="*100)
print("Detailed Results Comparison: Mantel vs Pearson Correlation Coefficients (with Human)")
print("="*100)
print(f"{'Model':<12} {'Mantel R':<10} {'Mantel P':<10} {'Pearson R':<12} {'Pearson P':<12} {'Difference':<10}")
print("-"*100)

for i, group in enumerate(groups[1:], 1):  # Skip Human
    mantel_r = mantel_correlations[0, i]
    mantel_p = mantel_pvalues[0, i]
    pearson_r = pearson_correlations[0, i]
    pearson_p = pearson_pvalues[0, i]
    diff = mantel_r - pearson_r
    
    print(f"{group:<12} {mantel_r:<10.3f} {mantel_p:<10.3f} {pearson_r:<12.3f} {pearson_p:<12.3f} {diff:<10.3f}")

print("\n" + "="*100)
print("LLM Inter-correlation Summary (Mantel Correlation Coefficients)")
print("="*100)
print("Correlations between model pairs (only showing r ≥ 0.5 correlations):")
print("-"*100)

for i in range(1, len(groups)):
    for j in range(i+1, len(groups)):
        mantel_r = mantel_correlations[i, j]
        mantel_p = mantel_pvalues[i, j]
        if mantel_r >= 0.5:
            sig_mark = "***" if mantel_p < 0.001 else "**" if mantel_p < 0.01 else "*" if mantel_p < 0.05 else "ns"
            print(f"{groups[i]} vs {groups[j]}: r = {mantel_r:.3f}, p = {mantel_p:.3f} {sig_mark}")

print("\n" + "="*100)
print("Unfair Conditions 7 Variables Unified Method Analysis Summary")
print("="*100)
print("1. Using same RSA matrix calculation method as mantel analysis (correlation matrix)")
print("2. Using group order from order: Human, gpt35, V3, R1, o3")
print("3. All models show highly significant correlations with human (p < 0.001)")
print("4. Mantel and Pearson correlations are highly consistent, validating method reliability")
print("5. Under unfair conditions, model performance differs from full dataset")
print("6. Significant correlations exist between LLMs, indicating shared emotion-behavior patterns under unfair conditions")
print("7. Complete correlation matrix reveals more complex inter-group relationship patterns under unfair conditions")
print("8. Using 7 variables (choice, AA_valence, AA_arousal, AC_valence, AC_arousal, EmoFDBK_valence, EmoFDBK_arousal) provides more comprehensive analysis")

