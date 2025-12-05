This folder contains mediation analysis scripts that examine mediation effects for unfair conditions.

Files:
- merged_all_data_unfair.R: R script for mediation analysis. Analyzes mediation effects of emotion variables on the relationship between allocation/cost and choice outcomes under unfair conditions. Uses standardized variables and group-wise analysis.
- merged_all_data_unfair.xlsx: Input data file containing all experimental data for unfair conditions.

Usage:
1. Ensure merged_all_data_unfair.xlsx is in the current directory (or modify the file path in the script).
2. Open merged_all_data_unfair.R in R or RStudio.
3. Modify the file path in the script (line 2) to match your system. The script currently reads from "Desktop/mod_med_unfair/merged_all_data_unfair.xlsx" - change this to your actual file path or move the file to match the path.
4. Run all code chunks to perform mediation analysis.

Output:
- Mediation effect estimates
- Direct and indirect effects
- Statistical significance tests
- Visualization plots

Note: The analysis splits data by groups (human, GPT-3.5, o3-mini, DeepSeek-V3, DeepSeek-R1) and performs group-wise standardization of emotion and fairness variables before mediation analysis.

