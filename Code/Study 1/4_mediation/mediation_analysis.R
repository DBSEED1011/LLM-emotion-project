# DATA FILE NOTE:
# This script uses unfair_data_for_mediation.xlsx, which is generated from demo.xlsx
# (contains a subset of the full dataset: 5 participants per group).
# 
# To use the full dataset:
# 1. Access the full dataset at: [INSERT LINK HERE]
# 2. Download "Study1_experimental_data.xlsx"
# 3. In process_unfair_data.py, replace "demo.xlsx" with "Study1_experimental_data.xlsx"
# 4. Run process_unfair_data.py to generate unfair_data_for_mediation.xlsx
# 5. Run this R script for mediation analysis

library(readxl)
unfair_data_for_mediation <- read_excel("unfair_data_for_mediation.xlsx")
View(unfair_data_for_mediation)

library(readxl)
library(dplyr)
library(ggplot2)
library(bruceR)

# Split dataset by groups
library(dplyr)
data_human <- unfair_data_for_mediation %>% filter(group == "human")
data_gpt3.5 <- unfair_data_for_mediation %>% filter(group == "gpt3.5")
data_o3 <- unfair_data_for_mediation %>% filter(group == "o3")
data_V3 <- unfair_data_for_mediation %>% filter(group == "V3")
data_R1 <- unfair_data_for_mediation %>% filter(group == "R1")

# Group-wise standardization of emotion and fairness variables
data_human$AA_valence_std <- scale(data_human$AA_valence, center = TRUE, scale = TRUE)
data_human$AA_arousal_std <- scale(data_human$AA_arousal, center = TRUE, scale = TRUE)
data_human$amount_of_allocation_std <- scale(data_human$amount_of_allocation, center = TRUE, scale = TRUE)
data_human$amount_of_cost_std <- scale(data_human$amount_of_cost, center = TRUE, scale = TRUE)

data_gpt3.5$AA_valence_std <- scale(data_gpt3.5$AA_valence, center = TRUE, scale = TRUE)
data_gpt3.5$AA_arousal_std <- scale(data_gpt3.5$AA_arousal, center = TRUE, scale = TRUE)
data_gpt3.5$amount_of_allocation_std <- scale(data_gpt3.5$amount_of_allocation, center = TRUE, scale = TRUE)
data_gpt3.5$amount_of_cost_std <- scale(data_gpt3.5$amount_of_cost, center = TRUE, scale = TRUE)

data_o3$AA_valence_std <- scale(data_o3$AA_valence, center = TRUE, scale = TRUE)
data_o3$AA_arousal_std <- scale(data_o3$AA_arousal, center = TRUE, scale = TRUE)
data_o3$amount_of_allocation_std <- scale(data_o3$amount_of_allocation, center = TRUE, scale = TRUE)
data_o3$amount_of_cost_std <- scale(data_o3$amount_of_cost, center = TRUE, scale = TRUE)


data_V3$AA_valence_std <- scale(data_V3$AA_valence, center = TRUE, scale = TRUE)
data_V3$AA_arousal_std <- scale(data_V3$AA_arousal, center = TRUE, scale = TRUE)
data_V3$amount_of_allocation_std <- scale(data_V3$amount_of_allocation, center = TRUE, scale = TRUE)
data_V3$amount_of_cost_std <- scale(data_V3$amount_of_cost, center = TRUE, scale = TRUE)


data_R1$AA_valence_std <- scale(data_R1$AA_valence, center = TRUE, scale = TRUE)
data_R1$AA_arousal_std <- scale(data_R1$AA_arousal, center = TRUE, scale = TRUE)
data_R1$amount_of_allocation_std <- scale(data_R1$amount_of_allocation, center = TRUE, scale = TRUE)
data_R1$amount_of_cost_std <- scale(data_R1$amount_of_cost, center = TRUE, scale = TRUE)


  
# Perform moderated mediation analysis
# Negative emotional response, moderated mediation, cluster=id, dependent variable=choice, moderator=personal cost cost_level
result_human <- PROCESS(data_human, y="choice", x="amount_of_allocation_std",
                    meds="AA_valence_std",
                    mods="cost_level",
                    mod.path=c("m-y","x-y"),
                    clusters="id",
                    ci="mcmc", nsim=1000, seed=1)
  
result_gpt3.5 <- PROCESS(data_gpt3.5, y="choice", x="amount_of_allocation_std",
                        meds="AA_valence_std",
                        mods="cost_level",
                        mod.path=c("m-y","x-y"),
                        clusters="id",
                        ci="mcmc", nsim=1000, seed=1)

result_o3 <- PROCESS(data_o3, y="choice", x="amount_of_allocation_std",
                        meds="AA_valence_std",
                        mods="cost_level",
                        mod.path=c("m-y","x-y"),
                        clusters="id",
                        ci="mcmc", nsim=1000, seed=1)

result_V3 <- PROCESS(data_V3, y="choice", x="amount_of_allocation_std",
                        meds="AA_valence_std",
                        mods="cost_level",
                        mod.path=c("m-y","x-y"),
                        clusters="id",
                        ci="mcmc", nsim=1000, seed=1)

result_R1 <- PROCESS(data_R1, y="choice", x="amount_of_allocation_std",
                        meds="AA_valence_std",
                        mods="cost_level",
                        mod.path=c("m-y","x-y"),
                        clusters="id",
                        ci="mcmc", nsim=1000, seed=1)



# Perform moderated mediation analysis
# Negative emotional response, moderated mediation, cluster=id, dependent variable=choice, moderator=personal cost amount_of_cost_std
result_human <- PROCESS(data_human, y="choice", x="amount_of_allocation_std",
                        meds="AA_valence_std",
                        mods="amount_of_cost_std",
                        mod.path=c("m-y","x-y"),
                        clusters="id",
                        ci="mcmc", nsim=1000, seed=1)

result_gpt3.5 <- PROCESS(data_gpt3.5, y="choice", x="amount_of_allocation_std",
                         meds="AA_valence_std",
                         mods="amount_of_cost_std",
                         mod.path=c("m-y","x-y"),
                         clusters="id",
                         ci="mcmc", nsim=1000, seed=1)

result_o3 <- PROCESS(data_o3, y="choice", x="amount_of_allocation_std",
                     meds="AA_valence_std",
                     mods="amount_of_cost_std",
                     mod.path=c("m-y","x-y"),
                     clusters="id",
                     ci="mcmc", nsim=1000, seed=1)

result_V3 <- PROCESS(data_V3, y="choice", x="amount_of_allocation_std",
                     meds="AA_valence_std",
                     mods="amount_of_cost_std",
                     mod.path=c("m-y","x-y"),
                     clusters="id",
                     ci="mcmc", nsim=1000, seed=1)

result_R1 <- PROCESS(data_R1, y="choice", x="amount_of_allocation_std",
                     meds="AA_valence_std",
                     mods="amount_of_cost_std",
                     mod.path=c("m-y","x-y"),
                     clusters="id",
                     ci="mcmc", nsim=1000, seed=1)