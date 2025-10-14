# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Category mapping
category_map = {
    1: 'Emotion',
    2: 'Fairness',
    3: 'Cost',
    4: 'Others'
}

# Set file paths
cot_path = '/Users/daiyiqing/Desktop/LLM_Emotion_Mechanism/LLM_annotator/Comparison/Top1000_CoT_Comparison_Claude_vs_Midsummer.xlsx'
human_path = '/Users/daiyiqing/Desktop/LLM_Emotion_Mechanism/LLM_annotator/Comparison/Top180_Comparison_Claude_vs_Midsummer.xlsx'

# Read data
cot_df = pd.read_excel(cot_path)
human_df = pd.read_excel(human_path)

# Clean column names to avoid issues with case or whitespace
cot_df.columns = cot_df.columns.str.strip().str.lower()
human_df.columns = human_df.columns.str.strip().str.lower()

# Add a Category column
cot_df['category'] = cot_df['word_sort'].map(category_map)
human_df['category'] = human_df['word_sort'].map(category_map)

# Ensure consistent category order
categories = ['Emotion', 'Fairness', 'Cost', 'Others']
cot_counts = cot_df['category'].value_counts(normalize=True).reindex(categories).fillna(0)
human_counts = human_df['category'].value_counts(normalize=True).reindex(categories).fillna(0)

# Combine proportions into a single DataFrame
proportions = pd.DataFrame({
    "DeepSeek-R1's CoT": cot_counts,
    "Human Reasoning": human_counts
})

# %%
# Plot settings
fig, ax = plt.subplots(figsize=(7, 3))

# Place the x-axis at the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Color palette
colors = ['#f26c6c', '#2171b5', '#6baed6', '#c6dbef']

# Create stacked horizontal bars
bar_height = 0.2
y_positions = [0.7, 0.45]  # y=0.7: CoT, y=0.45: Human
left_cot = 0
left_human = 0

for i, category in enumerate(categories):  # Iterate in fixed order
    cot_val = proportions.loc[category, "DeepSeek-R1's CoT"]
    human_val = proportions.loc[category, "Human Reasoning"]

    ax.barh(y_positions[0], cot_val, left=left_cot, height=bar_height, color=colors[i])
    ax.barh(y_positions[1], human_val, left=left_human, height=bar_height, color=colors[i])

    left_cot += cot_val
    left_human += human_val

# Set y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(["DeepSeek-R1's CoT", "Humans' Reasoning"])

# Configure x-axis
ax.set_xlim(0, 1)
ax.set_xlabel("Proportion", fontsize=13, labelpad=10)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Legend configuration
legend_elements = [
    Patch(facecolor=colors[0], label='Emotion'),
    Patch(facecolor=colors[1], label='Fairness'),
    Patch(facecolor=colors[2], label='Cost'),
    Patch(facecolor=colors[3], label='Others')
]

ax.legend(
    handles=legend_elements,
    title="Word Category",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.6),
    ncol=4,
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    handleheight=1.5,       # Controls the height of color boxes
    handlelength=2.0,       # Controls the width of color boxes
    borderaxespad=0.5       # Controls spacing between legend and plot
)

# Remove unnecessary borders
ax.spines[['right', 'left', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Ensure consistent category order
categories = ['Emotion', 'Fairness', 'Cost', 'Others']

# Raw counts
cot_counts_raw = cot_df['category'].value_counts().reindex(categories).fillna(0)
human_counts_raw = human_df['category'].value_counts().reindex(categories).fillna(0)

# Build the contingency table
contingency = pd.DataFrame({
    "DeepSeek-R1's CoT": cot_counts_raw,
    "Human Reasoning": human_counts_raw
}, index=categories)

print("Contingency Table:\n", contingency, "\n")

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency.T)

# Total sample size
N = contingency.to_numpy().sum()

# Cramér’s V
cramers_v = np.sqrt(chi2 / (N * (min(contingency.shape) - 1)))

print(f"Chi-square test: χ²({dof}, N={N}) = {chi2:.2f}, p = {p:.3e}")
print(f"Cramér's V = {cramers_v:.3f}\n")

# Calculate standardized residuals
residuals = (contingency.T - expected) / np.sqrt(expected)
residuals = residuals.T  # Convert back to original layout (categories × groups)

print("Standardized residuals (>|2| = major contributor):\n")
print(residuals.round(2))


# %%
