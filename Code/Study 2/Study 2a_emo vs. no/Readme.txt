README — Study 2a GLMM Analyses (Emotion vs. No Emotion Self-Report)
====================================================================

This directory contains analysis scripts associated with
the generalized linear mixed-effects models (GLMMs) reported in Study 2a of
the manuscript. These models test how emotion self-report modulates punitive
decisions in humans and LLM agents.

The folder includes the R Markdown analysis script and data file. Running the 
script will generate exported regression tables (model coefficients and 
post-hoc comparisons) as CSV files. All analyses are directly reproducible 
using the R Markdown script provided.

---------------------------------------------------------------------------
1. FILE LIST
---------------------------------------------------------------------------

1. Study2a_emovs.no.Rmd
   - The main R Markdown analysis script for Study 2a.
   - Runs GLMM3 (m1) and GLMM4 (m6).
   - Performs model comparison, generates predicted probabilities,
     computes odds ratios, and outputs model & post-hoc results.
   - Generates CSV output files (see Expected Output section below).

2. Study2a_experimental_data_emo_vs_no.xlsx
   - Trial-level dataset used as input for GLMM analyses.
   - Contains variables:
        choice (0/1), Agent_type, Emotion_selfreport,
        amount_of_cost, amount_of_allocation, ID, and other covariates.
   - Continuous variables are mean-centered inside the Rmd script.
   - Note: This data file is not included in this repository due to privacy 
     considerations. Please contact the authors to obtain the data download 
     link from the cloud storage.

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
    Ensure `Study2a_experimental_data_emo_vs_no.xlsx` is in the same directory 
    (or update its file path inside the Rmd file). Note: This file needs to be 
    obtained from the authors (see note above).

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

Running the Rmd file generates the following CSV output files:

1. model_results_GLMM3_m1.csv
   - Fixed-effects summary table for GLMM3 (model m1).
   - Columns include: term, B, SE, z, P, CI_lower, CI_upper, OR, OR_low, OR_high.
   - These results correspond to Supplementary Table 6.

2. posthoc_results_GLMM3_m1.csv
   - Post-hoc pairwise comparisons for GLMM3:
       • Agent_type comparisons within each Emotion_selfreport level
       • Emotion self-report comparisons within each Agent_type
   - Includes: B (log-odds), SE, z, P, 95% CI (logit scale), OR, OR CI.

3. model_results_m6.csv (GLMM4)
   - Fixed-effects summary table for GLMM4 (model m6).
   - Same column structure as model_results_GLMM3_m1.csv.
   - Corresponds to Supplementary Table 7.

4. posthoc_results_m6.csv
   - Post-hoc pairwise results for GLMM4.
   - Same column structure as posthoc_results_GLMM3_m1.csv.
   - Used for group-level interpretation of three-way interactions.

These output files correspond directly to the tables reported in the manuscript.

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



