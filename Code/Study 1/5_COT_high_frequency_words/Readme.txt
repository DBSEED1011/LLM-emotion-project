README — Semantic Reasoning & LLM Annotation Analysis
=====================================================

This folder contains all code and data needed to reproduce the semantic
categorization, word-frequency analysis, and word-cloud visualizations
reported in the manuscript. The materials are divided into two modules:

1. CoT_analysis — analyses based on chain-of-thought (CoT) reasoning data.
2. LLM_annotator — automated semantic annotation using language models
   (Claude 3 and o3-mini), plus associated data and summary outputs.

The code reproduces the semantic categorization results, chi-square tests,
agreement analyses, frequency tables, and visualizations used in the paper.


---------------------------------------------------------------------------
1. DIRECTORY STRUCTURE
---------------------------------------------------------------------------

5_COT_high_frequency_words/
│
├── CoT_analysis/
│     ├── COT_wordcloud.Rmd
│     ├── DeepSeek_R1_Word_Frequency.csv
│     ├── DeepSeek_WordCloud.png
│     ├── Human_reasoning.xlsx
│     ├── Human_Word_Frequency.csv
│     ├── Reasoning_example.xlsx
│     └── wordcloud_Humans_all_sq.png
│
└── LLM_annotator/
      ├── Annotation_code/
      │      ├── Claude_cot.py
      │      ├── Claude_human.py
      │      ├── error_prompt.txt
      │      ├── Human_Top180_Words.csv
      │      ├── o3mini_cot.py
      │      ├── o3mini_human.py
      │      └── Top1000_Words_DeepSeekR1.csv
      │
      ├── figure.py
      │
      ├── Human_coding_final_results/
      │      ├── Top180_Comparison_Claude_vs_Midsummer.xlsx
      │      └── Top1000_CoT_Comparison_Claude_vs_Midsummer.xlsx
      │
      └── LLM_annotation_results/
             ├── Top180_Annotated_Claude.csv
             ├── Top180_Annotated_Midsummer.csv
             ├── Top1000_Annotated_Claude.csv
             └── Top1000_Annotated_Midsummer.csv


---------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS
---------------------------------------------------------------------------

Software:
    - R ≥ 4.2.0
    - Python ≥ 3.8
    - RStudio (optional)
    - macOS / Linux / Windows

Required R packages:
    tidyverse
    readxl
    ggplot2
    wordcloud
    broom
    knitr
    rmarkdown
    (plus quanteda or similar if used for tokenization)

Install in R:
    install.packages(c("tidyverse","readxl","ggplot2",
                       "wordcloud","broom","knitr","rmarkdown"))

Required Python packages:
    pandas
    numpy
    matplotlib
    seaborn
    scipy
    httpx
    tqdm
    openai   # used as a client for the Claude-compatible API

Install in Python:
    pip install pandas numpy matplotlib seaborn scipy httpx tqdm openai


---------------------------------------------------------------------------
3. MODULE 1: CoT_analysis
---------------------------------------------------------------------------

The folder `CoT_analysis` contains all R code for analyzing chain-of-thought
(CoT) reasoning, computing word frequencies, generating word clouds, and
constructing semantic categories.

MAIN SCRIPT:
    COT_wordcloud.Rmd

This R Markdown script performs:
    • Import of CoT datasets (Human_reasoning.xlsx, DeepSeek_R1 etc.)
    • Tokenization, lowercasing, stopword removal
    • Construction of frequency dictionaries
    • Generation of word-cloud visualizations
    • Export of processed frequency tables

HOW TO RUN:
    1. Open `COT_wordcloud.Rmd` in RStudio.
    2. Ensure all CSV/XLSX files are in the same folder.
    3. Click “Run All” or knit the document.

Expected output:
    • DeepSeek_WordCloud.png
    • wordcloud_Humans_all_sq.png
    • Updated frequency tables:
        - DeepSeek_R1_Word_Frequency.csv
        - Human_Word_Frequency.csv


---------------------------------------------------------------------------
4. MODULE 2: LLM_annotator
---------------------------------------------------------------------------

This module generates semantic category labels (emotion, fairness, cost,
other) for high-frequency words using Claude 3.5 and o3-mini as independent
annotators. The Python scripts send batched word lists to the API, parse the
JSON responses, and save annotated CSV files.

FOLDER: Annotation_code/

    Claude_cot.py
        - Reads a CSV file containing a column named "word".
        - Sends words in batches (default: 30) to the Claude-compatible API.
        - Uses predefined seed lists for "emotion", "fairness", and "cost"
          to guide the classification.
        - Receives a JSON array with fields:
              "word", "category", "confidence", "rationale"
        - Saves results as CSV (e.g., Missing_Annotated_Claude.csv).
        - On error, writes the problematic prompt to `error_prompt.txt`.

        NOTE: The script uses a placeholder API key and base URL.
              Before running, users must:
                • Provide their own API key in `API_KEY` (or modify the code
                  to read from an environment variable).
                • Ensure that `BASE_URL` and `MODEL_NAME` correspond to a
                  valid Claude-compatible endpoint.

    Claude_human.py, o3mini_cot.py, o3mini_human.py
        - Analogous scripts for classifying human words and using o3-mini.
        - Input and output file names are specified at the bottom of each file
          and can be edited as needed.

    Input word lists:
        - Human_Top180_Words.csv          (top words from human reasoning)
        - Top1000_Words_DeepSeekR1.csv    (top words from DeepSeek-R1 CoT)

    Outputs:
        - CSV files in `LLM_annotation_results/` containing LLM-assigned
          categories for each word (emotion, fairness, cost, other).

HOW TO RUN (example for Claude on CoT words):

    cd LLM_annotator/Annotation_code
    python Claude_cot.py

    This will read the input CSV specified at the bottom of the script
    (default: "Missing_Words_DeepSeekR1.csv") and produce a corresponding
    annotated CSV (default: "Missing_Annotated_Claude.csv") in the same folder.


---------------------------------------------------------------------------
5. MODULE 2: Summary figures and statistics
---------------------------------------------------------------------------

FILE: figure.py

This script uses the final comparison tables to generate stacked bar plots
and perform statistical tests on semantic category distributions.

    - Reads:
        Top1000_CoT_Comparison_Claude_vs_Midsummer.xlsx
        Top180_Comparison_Claude_vs_Midsummer.xlsx

      (paths are currently hard-coded as absolute paths; users should
       replace them with paths appropriate for their environment, or move
       the script into the same directory and make paths relative.)

    - Maps numeric category codes (1–4) to:
        Emotion, Fairness, Cost, Others

    - Computes category proportions for:
        • DeepSeek-R1's CoT
        • Human Reasoning

    - Produces a stacked horizontal bar plot comparing these proportions.

    - Builds a 4×2 contingency table of raw counts and runs:
        • Chi-square test of independence (scipy.stats.chi2_contingency)
        • Cramér's V effect size
        • Standardized residuals to identify major contributors.

HOW TO RUN:

    1. Edit `cot_path` and `human_path` near the top of `figure.py` so that
       they point to the actual locations of the two comparison Excel files.
    2. From a Python environment with required packages installed:

           python figure.py

    3. The script will display the comparison figure and print:
           - Contingency table
           - χ² statistic, degrees of freedom, p-value
           - Cramér's V
           - Standardized residuals


