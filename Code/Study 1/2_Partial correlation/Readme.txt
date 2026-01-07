This folder contains partial correlation analysis scripts that compute partial correlations controlling for covariates.

Files:
- partial_correlation_emofdbk_arousal_unfair_format.py: Computes partial correlation between choice mean and emotion feedback arousal mean, controlling for cost_level. Analyzes data under unfair format conditions.
- partial_correlation_emofdbk_valence_unfair_format.py: Computes partial correlation between choice mean and emotion feedback valence mean, controlling for cost_level. Analyzes data under unfair format conditions.
- demo.xlsx: Input data file containing trial-level experimental data (subset of full dataset with 5 participants per group).
- participant_level_results_all_groups.csv: Previously used participant-level aggregated data file (no longer required, kept for reference).

Data Processing:
The scripts read trial-level data and automatically aggregate it to participant level by:
1. Grouping trials by participant_id, group, cost_level, and fairness_group
2. Calculating mean values for choice, EmoFDBK_valence, and EmoFDBK_arousal
3. Counting the number of trials per condition combination

Usage:
1. Ensure demo.xlsx (or Study1_experimental_data.xlsx for full dataset) is in the current directory.
2. Run the partial correlation scripts:
   python partial_correlation_emofdbk_arousal_unfair_format.py
   python partial_correlation_emofdbk_valence_unfair_format.py

Note on Data Files:
- demo.xlsx contains a subset of the full dataset (5 participants per group: human, gpt3.5, o3, V3, R1)
- To use the full dataset, replace "demo.xlsx" with "Study1_experimental_data.xlsx" in the scripts
- See the DATA FILE NOTE in each script for detailed instructions

Output:
- Partial correlation matrices
- Participant-level correlation results
- Statistical significance tests
- Visualization plots (saved as PNG files)

Note: The analysis samples 1/4 of points for each condition combination of each group to reduce computational load while maintaining statistical power.

