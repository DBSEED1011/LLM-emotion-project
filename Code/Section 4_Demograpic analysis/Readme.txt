This folder contains demographic moderation analysis scripts that examine how demographic variables moderate the relationships between traits and outcomes (emotions and choices) across different groups.

Files:
- emotion_moderation_analysis.py: Analyzes how demographic variables moderate emotion outcomes (valence, arousal, etc.) across groups. Tests moderation effects on 6 emotion variables and generates statistical reports and visualizations.
- choice_moderation_analysis.py: Analyzes how demographic variables moderate choice outcomes across groups. Tests moderation effects of demographic variables (AQ, ERS, CESD, etc.) on allocation choices and compares moderation patterns across groups (Human, GPT-3.5, o3-mini, DeepSeek-V3, DeepSeek-R1).
- plot_combined_moderation.py: Generates combined plots for moderation analyses, producing publication-ready figures.
- Study1_data_with_grouping_variables.xlsx: Input data file containing demographic variables and grouping information.
  Note: This data file is not included in this repository due to privacy considerations. Please contact the authors to obtain the data download link from the cloud storage.

Usage:
1. Ensure Study1_data_with_grouping_variables.xlsx is in the current directory. Note: This file needs to be obtained from the authors (see note above).
2. Run emotion_moderation_analysis.py to analyze emotion moderation effects.
3. Run choice_moderation_analysis.py to analyze choice moderation effects.
4. Run plot_combined_moderation.py to generate combined visualization plots.

Output:
- Statistical reports and plots showing moderation effects
- Combined moderation plots for publication

