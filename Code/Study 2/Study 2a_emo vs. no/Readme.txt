README — Study 2a GLMM Analyses (Emotion vs. No Emotion Self-Report)
====================================================================

This directory contains all analysis scripts and output files associated with
the generalized linear mixed-effects models (GLMMs) reported in Study 2a of
the manuscript. These models test how emotion self-report modulates punitive
decisions in humans and LLM agents.

The folder includes model scripts, data files, and exported regression tables
(model coefficients and post-hoc comparisons). All files are directly
reproducible using the R Markdown script provided.

---------------------------------------------------------------------------
1. FILE LIST
---------------------------------------------------------------------------

1. Study2a_emovs.no.Rmd
   - The main R Markdown analysis script for Study 2a.
   - Runs GLMM3 (m1) and GLMM4 (m6).
   - Performs model comparison, generates predicted probabilities,
     computes odds ratios, and outputs model & post-hoc results.
   - Produces all CSV files listed below.

2. study2_GLMM.xlsx
   - Trial-level dataset used as input for GLMM analyses.
   - Contains variables:
        choice (0/1), Agent_type, Emotion_selfreport,
        amount_of_cost, amount_of_allocation, ID, and other covariates.
   - Continuous variables are mean-centered inside the Rmd script.

3. model_results_GLMM3_m1.csv
   - Fixed-effects summary table for GLMM3 (model m1).
   - Columns include:
        term, B, SE, z, P,
        CI_lower, CI_upper,
        OR, OR_low, OR_high.
   - These results correspond to Supplementary Table 6.

4. model_results_m6.csv  (GLMM4)
   - Fixed-effects summary table for GLMM4 (model m6).
   - Same column structure as above.
   - Corresponds to Supplementary Table 7.

5. posthoc_results_GLMM3_m1.csv
   - Post-hoc pairwise comparisons for GLMM3:
       • Agent_type comparisons within each Emotion_selfreport level
       • Emotion self-report comparisons within each Agent_type
   - Includes:
       B (log-odds), SE, z, P,
       95% CI (logit scale), OR, OR CI.

6. posthoc_results_m6.csv
   - Post-hoc pairwise results for GLMM4.
   - Same column structure as above.
   - Used for group-level interpretation of three-way interactions.

---------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS
---------------------------------------------------------------------------

R version:
    R ≥ 4.2.0  
Recommended editor:
    RStudio

Required R packages:
    tidyverse
    readxl
    lme4
    broom.mixed
    emmeans
    dplyr
    knitr
    rmarkdown

Install all dependencies:
    install.packages(c("tidyverse","readxl","lme4","broom.mixed",
                       "emmeans","dplyr","knitr","rmarkdown"))

---------------------------------------------------------------------------
3. HOW TO RUN THE ANALYSIS
---------------------------------------------------------------------------

To reproduce all GLMM analyses:

Step 1:
    Open `Study2a_emovs.no.Rmd` in RStudio.

Step 2:
    Ensure `study2_GLMM.xlsx` is in the same directory (or update its file path
    inside the Rmd file).

Step 3:
    Select "Run All" or knit document:
        rmarkdown::render("Study2a_emovs.no.Rmd")

Step 4:
    The script will automatically:
      • Load and preprocess the dataset
      • Fit GLMM3 (m1): Agent_type × Emotion_selfreport
      • Fit GLMM4 (m6): Group × Emotion × Allocation/Cost interactions
      • Compute OR and 95% confidence intervals
      • Compare nested models using likelihood ratio tests
      • Estimate marginal means and perform pairwise contrasts
      • Export all results to CSV files

Output files appear in the working directory.

---------------------------------------------------------------------------
4. EXPECTED OUTPUT
---------------------------------------------------------------------------

Running the Rmd file generates:

1. model_results_GLMM3_m1.csv
2. posthoc_results_GLMM3_m1.csv
3. model_results_m6.csv
4. posthoc_results_m6.csv

These correspond directly to the tables reported in the manuscript.

- GLMM3 focuses on testing whether emotion self-report amplifies punishment
  across agent types.
- GLMM4 evaluates how emotion reporting interacts with allocation fairness
  and punishment cost across LLM architectures.

---------------------------------------------------------------------------
5. NOTES
---------------------------------------------------------------------------

• All continuous predictors (allocation fairness, punishment cost)
  are mean-centered within the script.

• Models are fitted using:
      glmer(..., family = binomial("logit"),
             control = glmerControl(optimizer="bobyqa",
                                    optCtrl = list(maxfun=1e5)))

• All post-hoc pairwise tests are two-sided Wald z-tests performed via emmeans.

• Odds ratios and 95% CI are derived by exponentiating log-odds estimates.



