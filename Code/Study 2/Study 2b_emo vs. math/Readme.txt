This folder contains analysis scripts comparing emotion and math conditions in Study 2b.

Files:
- Study2b_emo_vs_math_t_test_analysis.py: Independent samples t-test analysis comparing outcomes between emotion condition (group=1) and math condition (group=0) for stage=2 data. Analyzes differences in choice behavior, emotion ratings, and other variables between the two conditions.
- Study2b_experimental_data_emo_vs_math.xlsx: Input data file containing experimental data for Study 2b with emotion and math conditions.
  Note: This data file is not included in this repository due to privacy considerations. Please contact the authors to obtain the data download link from the cloud storage.

Usage:
1. Ensure Study2b_experimental_data_emo_vs_math.xlsx is in the current directory. Note: This file needs to be obtained from the authors (see note above).
2. Run the t-test analysis script:
   python Study2b_emo_vs_math_t_test_analysis.py

Output:
Running the script generates the following output:
- T-test results comparing emotion vs. math conditions
- Statistical significance tests
- Descriptive statistics for each condition
- Visualization plots showing group differences
- Results saved to emo_math_t_test_results.csv (generated after running the script)

Note: The analysis focuses on stage=2 data and compares group=1 (emotion condition) with group=0 (math condition) to examine how different task conditions affect decision-making and emotional responses.

