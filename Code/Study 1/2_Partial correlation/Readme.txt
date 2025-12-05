This folder contains partial correlation analysis scripts that compute partial correlations controlling for covariates.

Files:
- partial_correlation_emofdbk_arousal_unfair_format.py: Computes partial correlation between choice mean and emotion feedback arousal mean, controlling for cost_level. Analyzes data under unfair format conditions.
- partial_correlation_emofdbk_valence_unfair_format.py: Computes partial correlation between choice mean and emotion feedback valence mean, controlling for cost_level. Analyzes data under unfair format conditions.
- participant_level_results_all_groups.csv: Input data file containing participant-level aggregated results for all groups.

Usage:
1. Ensure participant_level_results_all_groups.csv is in the current directory.
2. Run the partial correlation scripts:
   python partial_correlation_emofdbk_arousal_unfair_format.py
   python partial_correlation_emofdbk_valence_unfair_format.py

Output:
- Partial correlation matrices
- Participant-level correlation results
- Statistical significance tests
- Visualization plots

Note: The analysis samples 1/4 of points for each condition combination of each group to reduce computational load while maintaining statistical power.

