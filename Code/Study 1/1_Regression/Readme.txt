This folder contains regression analysis scripts that examine choice behavior and emotion responses using Generalized Linear Mixed Models (GLMM) and Linear Mixed Models (LMM).

Files:
- GLMM1_choice_continuous.R: R script for GLMM analysis of choice behavior. Analyzes choice (0-1 coded) as dependent variable with continuous predictors (standardized allocation and cost) and group (human, LLM) as fixed effects, and participant ID as random effect. Includes main effects, two-way interactions, and three-way interaction models with model comparison (AIC/BIC).
- GLMM2_choice_category.py: Python script for GLMM analysis of choice behavior using categorical predictors. Analyzes choice as dependent variable with fairness (fair/unfair), cost level, and group as predictors. Performs three-way interaction analysis and generates regression tables.
- LMM1_emotion.py: Python script for LMM analysis of emotion responses. Analyzes AA_valence and AA_arousal as dependent variables with fairness (fair/unfair) and group as predictors. Generates three-line regression tables and model information tables for paper reporting.
- LMM2_emotion_outcome.py: Python script for LMM analysis examining the relationship between emotional feedback valence and choice outcomes. Analyzes how group and choice interact to predict emotional feedback valence. Generates regression tables for paper reporting.
- merged_all_data.xlsx: Input data file containing all experimental data for Study 1, including choice, emotion, allocation, cost, and group variables.

Usage:
1. Ensure merged_all_data.xlsx is in the current directory.
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

