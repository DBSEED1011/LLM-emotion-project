# -*- coding: utf-8 -*-
# Generalized Linear Mixed Effects Model Analysis (R code)
# Analyzing choice (0-1 coded) as dependent variable
# Fixed effects: Allocation (standardized continuous variable), cost (standardized continuous variable), group (human, LLM)
# Random effects: id
# Including all main effects and interaction effects analysis

# Load necessary packages
library(lme4)        # Mixed effects models
library(lmerTest)    # Provides p-values
library(glmmTMB)     # Alternative mixed effects model package
library(emmeans)     # Marginal means estimation
library(ggplot2)     # Plotting
library(dplyr)       # Data manipulation
library(readxl)      # Read Excel files
library(car)         # Analysis of variance
library(MuMIn)       # Model selection

# Set working directory (please modify according to your actual path)
# setwd("/Users/liuhao/Desktop/behvr patterns-regression")

# 1. Data loading and preprocessing
cat("Loading data...\n")
data <- read_excel("merged_all_data.xlsx")

cat("Basic data information:\n")
cat("Data shape:", dim(data), "\n")
cat("Column names:", colnames(data), "\n")

# 2. Data preprocessing
cat("\nPreparing analysis variables...\n")

# Process choice variable (ensure 0-1 coding)
data$choice <- as.numeric(data$choice)
cat("Choice variable: 0=", sum(data$choice==0), ", 1=", sum(data$choice==1), "\n")

# Process allocation variable (using amount_of_allocation, standardized as continuous variable)
data$allocation <- scale(data$amount_of_allocation)[,1]
cat("Allocation standardized range:", range(data$allocation), "\n")

# Process cost variable (using amount_of_cost, standardized as continuous variable)
data$cost <- scale(data$amount_of_cost)[,1]
cat("Cost standardized range:", range(data$cost), "\n")

# Process group variable (human, LLM)
# Set factor level order: human as reference group, others ordered as gpt3.5, o3, V3, R1
data$group <- as.factor(data$group)

# Define expected group names and order
expected_groups <- c("human", "gpt3.5", "o3", "V3", "R1")

# Check actual groups present in data
actual_groups <- levels(data$group)
cat("Actual groups in data:", actual_groups, "\n")

# Find matching groups (case insensitive)
matched_groups <- c()
for (expected in expected_groups) {
  # Find matching groups (supports case insensitive matching)
  matches <- actual_groups[grep(paste0("^", expected, "$"), actual_groups, ignore.case = TRUE)]
  if (length(matches) > 0) {
    matched_groups <- c(matched_groups, matches[1])  # Take first match
  }
}

# Add unmatched groups
unmatched_groups <- setdiff(actual_groups, matched_groups)
if (length(unmatched_groups) > 0) {
  cat("Warning: The following groups were not found in expected list:", unmatched_groups, "\n")
  matched_groups <- c(matched_groups, unmatched_groups)
}

# Reset factor level order
data$group <- factor(data$group, levels = matched_groups)
cat("Group categories after reordering:", levels(data$group), "\n")
cat("Reference group:", levels(data$group)[1], "\n")

# Process random effects variable
data$id <- as.factor(data$id)
cat("Number of IDs:", length(levels(data$id)), "\n")

# Remove missing values
original_len <- nrow(data)
data <- data[complete.cases(data[c("choice", "allocation", "cost", "group", "id")]), ]
cat("Remaining after removing missing values:", nrow(data), "rows (removed", original_len - nrow(data), "rows)\n")

# 3. Run generalized linear mixed effects model
cat("\n=== Starting Generalized Linear Mixed Effects Model Analysis ===\n")

# Method 1: Using lme4 package
cat("\nMethod 1: Using lme4 package\n")

# Main effects model
cat("Fitting main effects model...\n")
model_main <- glmer(choice ~ allocation + cost + group + (1|id), 
                    data = data, family = binomial, 
                    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

# Check model convergence
if (model_main@optinfo$conv$opt == 0) {
  cat("Main effects model converged successfully!\n")
  print(summary(model_main))
} else {
  cat("Main effects model failed to converge, trying other optimizer...\n")
  model_main <- glmer(choice ~ allocation + cost + group + (1|id), 
                      data = data, family = binomial, 
                      control = glmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 2e5)))
}

# Two-way interaction model
cat("\nFitting two-way interaction model...\n")
model_two_way <- glmer(choice ~ allocation + cost + group + 
                       allocation:cost + allocation:group + cost:group + 
                       (1|id), 
                       data = data, family = binomial, 
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

# Three-way interaction model
cat("\nFitting three-way interaction model...\n")
model_three_way <- glmer(choice ~ allocation + cost + group + 
                         allocation:cost + allocation:group + cost:group + 
                         allocation:cost:group + 
                         (1|id), 
                         data = data, family = binomial, 
                         control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

# 4. Model comparison and selection
cat("\n=== Model Comparison ===\n")
models <- list(
  "Main Effects" = model_main,
  "Two-way Interactions" = model_two_way,
  "Three-way Interactions" = model_three_way
)

# AIC comparison
aic_values <- sapply(models, AIC)
cat("AIC values:\n")
print(aic_values)

# BIC comparison
bic_values <- sapply(models, BIC)
cat("\nBIC values:\n")
print(bic_values)

# Select best model
best_model_name <- names(models)[which.min(aic_values)]
best_model <- models[[best_model_name]]
cat("\nBest model (based on AIC):", best_model_name, "\n")

# 5. Display best model regression results
cat("\n=== Best Model Regression Results ===\n")
cat("Model type:", best_model_name, "\n")
cat("AIC value:", AIC(best_model), "\n")
cat("BIC value:", BIC(best_model), "\n")

# Detailed model summary
cat("\n--- Model Summary ---\n")
print(summary(best_model))

# Coefficient table (including confidence intervals)
cat("\n--- Coefficient Estimates and Confidence Intervals ---\n")
coef_table <- summary(best_model)$coefficients
print(coef_table)

# Calculate confidence intervals
cat("\n--- 95% Confidence Intervals ---\n")
confint_table <- confint(best_model, method = "Wald")
print(confint_table)

# Model fit indicators
cat("\n--- Model Fit Indicators ---\n")
cat("Log-likelihood:", logLik(best_model), "\n")
cat("Deviance:", deviance(best_model), "\n")
cat("Residual degrees of freedom:", df.residual(best_model), "\n")

# Random effects variance
cat("\n--- Random Effects Variance ---\n")
random_effects <- VarCorr(best_model)
print(random_effects)

# Marginal R² and conditional R²
cat("\n--- R² Indicators ---\n")
if (require(MuMIn, quietly = TRUE)) {
  r_squared <- r.squaredGLMM(best_model)
  cat("Marginal R² (variance explained by fixed effects):", r_squared[1, "R2m"], "\n")
  cat("Conditional R² (variance explained by fixed+random effects):", r_squared[1, "R2c"], "\n")
}

# Model diagnostics
cat("\n--- Model Diagnostics ---\n")
cat("Convergence status:", ifelse(best_model@optinfo$conv$opt == 0, "Success", "Failed"), "\n")
if (best_model@optinfo$conv$opt != 0) {
  cat("Convergence warning:", best_model@optinfo$conv$opt, "\n")
}

# Effect size interpretation
cat("\n--- Effect Size Interpretation ---\n")
cat("Note: This is a logistic regression model, coefficients represent changes in log odds ratio\n")
cat("Positive coefficients indicate that as the variable increases, selection probability increases\n")
cat("Negative coefficients indicate that as the variable increases, selection probability decreases\n")

# Prediction probability examples
cat("\n--- Prediction Probability Examples ---\n")
# Create example data points for prediction
example_data <- data.frame(
  allocation = c(0, 1, -1),  # Standardized values
  cost = c(0, 0, 0),
  group = factor(rep(levels(data$group)[1], 3), levels = levels(data$group)),
  id = factor(rep(levels(data$id)[1], 3), levels = levels(data$id))
)

# Predict probabilities
predicted_probs <- predict(best_model, newdata = example_data, type = "response")
cat("Example prediction probabilities:\n")
for (i in 1:nrow(example_data)) {
  cat("Allocation =", example_data$allocation[i], 
      ", Cost =", example_data$cost[i], 
      ", Group =", as.character(example_data$group[i]),
      " -> Predicted probability =", round(predicted_probs[i], 4), "\n")
}

# Marginal effects analysis
cat("\n--- Marginal Effects Analysis ---\n")
cat("Calculating average marginal effects of each group on cost...\n")

# Use emmeans to calculate marginal effects
# First calculate marginal effects of each group on cost
marginal_effects <- emtrends(best_model, ~ group, var = "cost", 
                            at = list(allocation = 0))  # Calculate at allocation=0

# Display marginal effects results
cat("\nAverage marginal effects of each group on cost:\n")
print(marginal_effects)

# Get confidence intervals for marginal effects
marginal_effects_ci <- confint(marginal_effects)
cat("\n95% confidence intervals for marginal effects:\n")
print(marginal_effects_ci)

# Perform between-group comparisons
cat("\nBetween-group marginal effects comparisons:\n")
marginal_effects_pairs <- pairs(marginal_effects)
print(marginal_effects_pairs)

# Calculate effect size of marginal effects (probability scale)
cat("\nMarginal effects interpretation:\n")
cat("Marginal effects represent the change in selection probability when cost increases by 1 standard deviation unit\n")
cat("Positive values indicate that selection probability increases when cost increases, negative values indicate selection probability decreases\n")

cat("\n=== Analysis Complete ===\n")

library(openxlsx)
library(lme4)
library(lmerTest)
library(glmmTMB)
library(emmeans)
library(ggplot2)
library(dplyr)
library(readxl)
library(car)
library(MuMIn)
# 2. Create Excel workbook
wb <- createWorkbook()

# 3. Worksheet 1: Model summary
addWorksheet(wb, "Model Summary")
model_summary <- data.frame(
  Indicator = c("Model Type", "AIC Value", "BIC Value", "Log-likelihood", "Deviance", "Residual DF", 
         "Number of Observations", "Number of Groups", "Marginal R²", "Conditional R²", "Convergence Status"),
  Value = c("Three-way Interaction Model", 
        round(AIC(best_model), 4),
        round(BIC(best_model), 4),
        round(logLik(best_model), 4),
        round(deviance(best_model), 4),
        df.residual(best_model),
        nrow(data),
        length(levels(data$id)),
        round(r.squaredGLMM(best_model)[1, "R2m"], 6),
        round(r.squaredGLMM(best_model)[1, "R2c"], 6),
        ifelse(best_model@optinfo$conv$opt == 0, "Success", "Failed"))
)
writeData(wb, "Model Summary", model_summary)

# 4. Worksheet 2: Coefficient estimates and statistical tests
addWorksheet(wb, "Coefficient Estimates")
coef_summary <- summary(best_model)$coefficients
coef_df <- data.frame(
  Variable = rownames(coef_summary),
  Estimate = round(coef_summary[, "Estimate"], 6),
  Std_Error = round(coef_summary[, "Std. Error"], 6),
  z_value = round(coef_summary[, "z value"], 4),
  p_value = format(coef_summary[, "Pr(>|z|)"], scientific = TRUE, digits = 3),
  Significance = ifelse(coef_summary[, "Pr(>|z|)"] < 0.001, "***",
               ifelse(coef_summary[, "Pr(>|z|)"] < 0.01, "**",
                      ifelse(coef_summary[, "Pr(>|z|)"] < 0.05, "*",
                             ifelse(coef_summary[, "Pr(>|z|)"] < 0.1, ".", ""))))
)
writeData(wb, "Coefficient Estimates", coef_df)

# 5. Worksheet 3: Confidence intervals
addWorksheet(wb, "Confidence Intervals")
confint_table <- confint(best_model, method = "Wald")
confint_df <- data.frame(
  Variable = rownames(confint_table),
  Lower_2.5 = round(confint_table[, "2.5 %"], 6),
  Upper_97.5 = round(confint_table[, "97.5 %"], 6)
)
writeData(wb, "Confidence Intervals", confint_df)

# 6. Worksheet 4: Random effects
addWorksheet(wb, "Random Effects")
random_effects <- VarCorr(best_model)
random_df <- data.frame(
  Group = "id",
  Parameter = "(Intercept)",
  Std_Deviation = round(attr(random_effects$id, "stddev"), 6),
  Variance = round(attr(random_effects$id, "stddev")^2, 6)
)
writeData(wb, "Random Effects", random_df)

# 7. Worksheet 5: Model comparison
addWorksheet(wb, "Model Comparison")
# Refit all models for comparison
model_main <- glmer(choice ~ allocation + cost + group + (1|id), 
                    data = data, family = binomial, 
                    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

model_two_way <- glmer(choice ~ allocation + cost + group + 
                         allocation:cost + allocation:group + cost:group + 
                         (1|id), 
                       data = data, family = binomial, 
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

models <- list("Main Effects" = model_main, "Two-way Interactions" = model_two_way, "Three-way Interactions" = best_model)
model_comparison <- data.frame(
  Model = names(models),
  AIC = round(sapply(models, AIC), 4),
  BIC = round(sapply(models, BIC), 4),
  Log_Likelihood = round(sapply(models, logLik), 4),
  Degrees_of_Freedom = sapply(models, df.residual)
)
writeData(wb, "Model Comparison", model_comparison)

# 8. Worksheet 6: Effect interpretation
addWorksheet(wb, "Effect Interpretation")
effect_interpretation <- data.frame(
  Variable = c("allocation", "cost", "groupgpt3.5", "groupo3", "groupV3", "groupR1",
         "allocation:cost", "allocation:groupgpt3.5", "allocation:groupo3", 
         "allocation:groupV3", "allocation:groupR1", "cost:groupgpt3.5",
         "cost:groupo3", "cost:groupV3", "cost:groupR1", "allocation:cost:groupgpt3.5",
         "allocation:cost:groupo3", "allocation:cost:groupV3", "allocation:cost:groupR1"),
  Interpretation = c("Effect of allocation amount on selection probability (negative)",
         "Effect of cost on selection probability (negative)",
         "Selection tendency of GPT-3.5 group relative to human group",
         "Selection tendency of O3 group relative to human group", 
         "Selection tendency of V3 group relative to human group",
         "Selection tendency of R1 group relative to human group",
         "Interaction effect between allocation amount and cost",
         "Interaction effect between allocation amount and GPT-3.5 group",
         "Interaction effect between allocation amount and O3 group",
         "Interaction effect between allocation amount and V3 group",
         "Interaction effect between allocation amount and R1 group",
         "Interaction effect between cost and GPT-3.5 group",
         "Interaction effect between cost and O3 group",
         "Interaction effect between cost and V3 group",
         "Interaction effect between cost and R1 group",
         "Three-way interaction effect between allocation amount, cost and GPT-3.5 group",
         "Three-way interaction effect between allocation amount, cost and O3 group",
         "Three-way interaction effect between allocation amount, cost and V3 group",
         "Three-way interaction effect between allocation amount, cost and R1 group")
)
writeData(wb, "Effect Interpretation", effect_interpretation)

# 9. Worksheet 7: Prediction probability examples
addWorksheet(wb, "Prediction Probability Examples")
# Create example data
example_data <- expand.grid(
  allocation = c(-1, 0, 1),
  cost = c(-1, 0, 1),
  group = levels(data$group)[1:3]  # Only take first 3 groups
)
example_data$id <- factor(rep(levels(data$id)[1], nrow(example_data)), levels = levels(data$id))

# Predict probabilities
predicted_probs <- predict(best_model, newdata = example_data, type = "response")
example_data$Predicted_Probability <- round(predicted_probs, 4)
writeData(wb, "Prediction Probability Examples", example_data)

# 10. Worksheet 8: Marginal effects analysis
addWorksheet(wb, "Marginal Effects Analysis")

# Recalculate marginal effects (ensure they can be obtained in Excel output section)
marginal_effects <- emtrends(best_model, ~ group, var = "cost", 
                            at = list(allocation = 0))
marginal_effects_ci <- confint(marginal_effects)
marginal_effects_pairs <- pairs(marginal_effects)

# Marginal effects results table
marginal_effects_df <- data.frame(
  Group = marginal_effects@grid$group,
  Marginal_Effect = round(marginal_effects@bhat, 6),
  Std_Error = round(marginal_effects@SE, 6),
  Lower_2.5 = round(marginal_effects_ci[, "lower.CL"], 6),
  Upper_97.5 = round(marginal_effects_ci[, "upper.CL"], 6),
  Significance = ifelse(marginal_effects_ci[, "lower.CL"] > 0 | marginal_effects_ci[, "upper.CL"] < 0, "Significant", "Not Significant")
)
writeData(wb, "Marginal Effects Analysis", marginal_effects_df, startRow = 1, startCol = 1)

# Between-group comparison results
writeData(wb, "Marginal Effects Analysis", "Between-group comparison results:", startRow = nrow(marginal_effects_df) + 3, startCol = 1)
pairs_df <- data.frame(
  Comparison = paste(marginal_effects_pairs@grid$contrast),
  Estimate = round(marginal_effects_pairs@bhat, 6),
  Std_Error = round(marginal_effects_pairs@SE, 6),
  p_value = format(marginal_effects_pairs@misc$test$pvalues, scientific = TRUE, digits = 3)
)
writeData(wb, "Marginal Effects Analysis", pairs_df, startRow = nrow(marginal_effects_df) + 4, startCol = 1)


# 11. Save Excel file to current directory
file_path <- "Best_Model_Results.xlsx"
saveWorkbook(wb, file_path, overwrite = TRUE)
cat("Excel file saved as:", file_path, "\n")
cat("Contains the following worksheets:\n")
cat("1. Model Summary\n")
cat("2. Coefficient Estimates\n") 
cat("3. Confidence Intervals\n")
cat("4. Random Effects\n")
cat("5. Model Comparison\n")
cat("6. Effect Interpretation\n")
cat("7. Prediction Probability Examples\n")
cat("8. Marginal Effects Analysis\n")
