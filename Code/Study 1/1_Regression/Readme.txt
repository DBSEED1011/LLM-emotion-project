This folder contains regression analysis scripts that examine choice behavior and emotion responses using Generalized Linear Mixed Models (GLMM) and Linear Mixed Models (LMM).

Files:
- GLMM1_choice_continuous.R: R script for GLMM analysis of choice behavior. Analyzes choice (0-1 coded) as dependent variable with continuous predictors (standardized allocation and cost) and group (human, LLM) as fixed effects, and participant ID as random effect. Includes main effects, two-way interactions, and three-way interaction models with model comparison (AIC/BIC).
- GLMM2_choice_category.py: Python script for GLMM analysis of choice behavior using categorical predictors. Analyzes choice as dependent variable with fairness (fair/unfair), cost level, and group as predictors. Performs three-way interaction analysis and generates regression tables.
- LMM1_emotion.py: Python script for LMM analysis of emotion responses. Analyzes AA_valence and AA_arousal as dependent variables with fairness (fair/unfair) and group as predictors. Generates three-line regression tables and model information tables for paper reporting.
- LMM2_emotion_outcome.py: Python script for LMM analysis examining the relationship between emotional feedback valence and choice outcomes. Analyzes how group and choice interact to predict emotional feedback valence. Generates regression tables for paper reporting.
- demo.xlsx: Demo data file containing a subset of experimental data (5 participants per group: human, gpt3.5, o3, V3, R1) for Study 1, including choice, emotion, allocation, cost, and group variables. This is the default data file used by all scripts.

DATA FILE INFORMATION:
================================================================================
CURRENT SETUP: All scripts are configured to use demo.xlsx by default.

Demo Data (demo.xlsx):
- Contains 5 participants per group (human, gpt3.5, o3, V3, R1)
- Total: 25 participants, 1500 observations
- Suitable for testing and demonstration purposes

Full Dataset Access:
- To use the complete dataset, please access: [INSERT LINK HERE]
- Download the file named "Study1_experimental_data.xlsx"
- Replace "demo.xlsx" with "Study1_experimental_data.xlsx" in the relevant script(s)

IMPORTANT NOTES WHEN SWITCHING TO FULL DATA:
1. Update data file name in scripts:
   - GLMM1_choice_continuous.R: Line 40 (read_excel call)
   - GLMM2_choice_category.py: Line 40 (read_excel call)
   - LMM1_emotion.py: Line 188 (read_excel call)
   - LMM2_emotion_outcome.py: Line 189 (read_excel call)

2. Potential adjustments needed:
   - Model convergence settings may need tuning (optimizer parameters, maxfun values)
   - Computation time will increase significantly
   - Memory usage may increase
   - Sample size statistics (n_groups, n_obs) will change
   - Model fit indices (AIC, BIC, ICC) will differ

3. Code locations with data-dependent behavior:
   - All scripts: Sample size calculations (n_groups, n_obs)
   - GLMM1_choice_continuous.R: Model convergence settings (lines 95-108, 324-332)
   - GLMM2_choice_category.py: Model fitting method (line 80)
   - LMM1_emotion.py: Model fitting method (line 32)
   - LMM2_emotion_outcome.py: Model fitting method (line 32), data filtering (line 175)
================================================================================

Usage:
1. Ensure demo.xlsx is in the current directory (or replace with full dataset as described above).
2. For R analysis (GLMM1_choice_continuous.R):
   - Open the script in R or RStudio.
   - Modify the working directory path if needed (line 20).
   - Install required packages: lme4, lmerTest, glmmTMB, emmeans, ggplot2, dplyr, readxl, car, MuMIn.
   - Run all code chunks to perform GLMM analysis.
3. For Python analyses (GLMM2_choice_category.py, LMM1_emotion.py, LMM2_emotion_outcome.py):
   - Install required packages: pandas, numpy, statsmodels, scipy.
   - Run each script:
     python GLMM2_choice_category.py
     python LMM1_emotion.py
     python LMM2_emotion_outcome.py

Output:
- GLMM1_choice_continuous.R: Model comparison results (AIC/BIC), best model summary, regression coefficients, marginal means, and visualization plots.
- GLMM2_choice_category.py: Regression tables with main effects and interaction effects.
- LMM1_emotion.py: Three-line regression tables for AA_valence and AA_arousal, model information tables (N, ICC, etc.), saved to regression_tables_for_paper.xlsx.
- LMM2_emotion_outcome.py: Three-line regression tables for emotional feedback valence, model information tables, saved to regression_tables_for_paper.xlsx.

Note: The analyses compare behavior and emotion responses between human participants and different LLM groups (GPT-3.5, o3-mini, DeepSeek-V3, DeepSeek-R1). All models include participant ID as a random effect to account for within-participant dependencies. The Python scripts use human group as the reference category (coded as 0) for group comparisons.

IMPORTANT: This repository uses demo.xlsx by default. For full analysis results, please use the complete dataset (Study1_experimental_data.xlsx) as described in the DATA FILE INFORMATION section above.

