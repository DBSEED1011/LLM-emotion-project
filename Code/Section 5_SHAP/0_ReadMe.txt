This folder contains two data files ('merged_all_models_persona.csv' and 'Study1_demographic_data.xlsx'), two code files ('1_Analysis_of_SHAP.ipynb' and '2_LMM.Rmd'), and two completed HTML results ('1_Analysis_of_SHAP.html' and '2_LMM.html').
For simplicity, other potential output files have been removed, keeping only those necessary for plotting ('shap_results_weighted.csv' and 'xgb_oos_performance.csv').

Note: The following data files are not included in this repository due to privacy considerations. Please contact the authors to obtain the data download link from the cloud storage:
- Study1_demographic_data.xlsx
- merged_all_models_persona.csv

Please first run '1_Analysis_of_SHAP.ipynb', which generates a series of SVG vector figures and processed data files. Then, run '2_LMM.Rmd'.

The first file includes all content related to SHAP analysis in the main text and supplementary materials (corresponding to 'LLMs weighed emotions more and cost less than humans'  and 'Reasoning LLMs were more human-like in considering emotion and cost' in the main text and  Sections 5 in the supplementary materials).
The LMM file presents the results of a simple linear mixed-model analysis (corresponding to 'Reasoning LLMs were more human-like in considering emotion and cost' in the main text and Sections 5 in the supplementary materials).