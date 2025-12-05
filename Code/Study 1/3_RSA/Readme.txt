This folder contains Representational Similarity Analysis (RSA) scripts that compare representational structures across groups using Mantel tests.

Files:
- unfair_7var_validation_rsa_mantel_analysis.py: Main RSA analysis script. Computes representational matrices for 7 variables (choice, AA_valence, AA_arousal, AC_valence, AC_arousal, EmoFDBK_valence, EmoFDBK_arousal) and performs Mantel tests to compare matrices between human and LLM groups.
- order_unfair_conditions_7var_mantel_analysis_unified.py: Unified RSA analysis for unfair conditions with 7 variables. Performs cross-group validation and comparison.
- unfair_all_datasets_means_with_emotions_7_variables.xlsx: Input data file containing mean values for all datasets with 7 emotion and choice variables.

Usage:
1. Ensure unfair_all_datasets_means_with_emotions_7_variables.xlsx is in the current directory.
2. Run the RSA analysis scripts:
   python unfair_7var_validation_rsa_mantel_analysis.py
   python order_unfair_conditions_7var_mantel_analysis_unified.py

Output:
- Representational similarity matrices for each group
- Mantel test correlation coefficients and p-values
- Cross-group comparison results
- Visualization plots showing RSA correlation matrices

Note: RSA analysis compares the similarity of representational structures (correlation matrices) between groups. The Mantel test evaluates whether two representational matrices are significantly correlated.

