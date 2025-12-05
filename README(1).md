# LLM Emotion Project

## Project Overview

This project is a comprehensive research study on emotion analysis in Large Language Models (LLMs). Through multi-round experiments, the project explores LLM decision-making behavior under persona and emotion self-reporting conditions, conducting in-depth statistical analysis and model interpretability studies.

---

## Table of Contents

- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Usage Instructions](#usage-instructions)
- [Core Code Modules](#core-code-modules)
- [Data Flow](#data-flow)
- [License](#license)

---

## Project Features

- **Multi-model LLM Experiments**: Supports parallel execution of experiments across multiple LLM models (GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini)
- **Persona Integration**: Generates LLM personas based on real human demographic data (AQ, ERS, CESD scores, etc.)
- **Emotion Self-reporting**: Compares experimental results with/without emotion self-reporting conditions
- **Temperature Parameter Study**: Analyzes model performance under different temperature parameters
- **Model Interpretability Analysis (SHAP)**: Uses SHAP values to understand model decision-making processes
- **Chain of Thought (CoT) Reasoning Analysis**: Analyzes LLM reasoning processes and high-frequency words
- **Demographic Variable Moderation Analysis**: Statistical analysis of how demographic factors influence outcomes
- **Representational Similarity Analysis (RSA)**: Compares representational structures across different groups

---

## Project Structure

```
LLM-emotion-project/
├── Code/
│   ├── Section 1_experiment of LLM/       # LLM multi-round experiments (core experimental code)
│   ├── Section 3_CoT_mediation/           # Chain of Thought mediation analysis
│   ├── Section 4_Demograpic analysis/     # Demographic analysis
│   ├── Section 5_SHAP/                    # SHAP model interpretability analysis
│   ├── Section 6_Nopersona/               # No-persona condition analysis
│   ├── Section 7_Temperature/             # Temperature parameter comparison analysis
│   ├── Study 1/                           # Study 1 (regression, correlation, RSA analyses)
│   └── Study 2/                           # Study 2 (emotion vs. no-emotion, emotion vs. math comparisons)
├── SourceData/                            # Source data for figures
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
└── LICENSE                                # License
```

---

## System Requirements

### API Access
- **OpenAI API Key**: Required for GPT-3.5 and o3-mini models
- **DeepSeek API Key**: Required for DeepSeek-V3 and DeepSeek-R1 models
- Note: For security reasons, API keys are not included in the repository. Users must provide their own.

---

## Installation Guide

### Step 1: Clone or Download the Repository

```bash
# Using git
git clone <repository-url>
cd LLM-emotion-project

# Or download and extract the zip file
```

### Step 2: Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### Step 3: Install R Dependencies

Open R or RStudio and run:

```r
install.packages(c("lme4", "nlme", "ggplot2", "dplyr", 
                   "readxl", "corrplot", "vegan"))
```

### Step 4: Configure API Keys

1. Navigate to `Code/Section 1_experiment of LLM/multi_round_person.py`
2. Add your API keys:
   ```python
   api_key = "your-api-key-here"
   ```

---

## Quick Start

### Demo

The following demo uses a small data subset (2 subjects) to demonstrate the software functionality.

#### 1. Navigate to Section 1 Directory

```bash
cd "Code/Section 1_experiment of LLM"
```

#### 2. Configure Demo Settings

The demo is pre-configured in `run_par.py`:
- `subjnum_list = range(0, 2)` (2 subjects)
- Default conditions: persona=True, emotion=True, temperature=1.0

#### 3. Run the Demo

```bash
python run_par.py
```

This will:
1. Generate character prompts for 2 subjects
2. Generate game setting prompts
3. Run LLM experiments across 4 models (GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini)
4. Save results in `result_persona_emotion_1.0_*` folders

#### 4. Validate Results

```bash
jupyter notebook check.ipynb
```

Run all cells in the notebook to:
- Check output files
- Generate merged data files
- Validate results

### Expected Output

After running the demo, you should see:

1. **Generated prompt files** (in the `prompt/` folder):
   - `0_character.json`, `1_character.json`
   - `0_game_setting_prompt.json`, `1_game_setting_prompt.json`

2. **Result folders** (one for each LLM model):
   - `result_persona_emotion_1.0_gpt-3.5-turbo-0125/`
   - `result_persona_emotion_1.0_deepseek-v3/`
   - `result_persona_emotion_1.0_deepseek-r1/`
   - `result_persona_emotion_1.0_o3-mini-2025-01-31/`

3. **Output files** (in each result folder):
   - `output_0.txt`, `output_1.txt` (one per subject)

4. **Merged data file** (after running `check.ipynb`):
   - `merged_all_models_persona_emotion_1.0.txt`

### Expected Runtime

On a standard desktop/laptop:
- Prompt generation: < 1 minute
- LLM API calls (2 subjects × 4 models): 5-15 minutes (depends on API response time)
- Result validation: < 1 minute
- **Total**: Approximately 6-17 minutes

*Note: Runtime depends on API response time, network connection speed, and number of concurrent API calls.*

---

## Usage Instructions

### Section 1: LLM Experiments

#### Running on Your Data

1. **Configure Subject List**

Edit `run_par.py`:
```python
subjnum_list = range(0, N)  # N = number of subjects (max 1017)
```

2. **Adjust Experimental Conditions**

In `multi_round_person.py`:
```python
persona = True/False      # False for no-persona condition
TEMPERATURE = 1.0         # 0 for temperature=0
emotion = True/False      # False for no emotion self-reporting condition
```

3. **Run Experiments**

```bash
python run_par.py
```

4. **Validate and Merge Results**

```bash
jupyter notebook check.ipynb
```

### Other Sections

Each section has independent analysis scripts:

- **Section 3_CoT_mediation**: Chain of Thought mediation analysis (R Markdown)
- **Section 4_Demographic analysis**: Demographic moderation analysis (Python)
- **Section 5_SHAP**: SHAP value model interpretability analysis (Python/Jupyter)
- **Section 6_Nopersona**: No-persona condition analysis
- **Section 7_Temperature**: Model performance analysis under different temperature parameters
- **Study 1**: Regression analysis, partial correlation, RSA analysis (Python/R)
- **Study 2**: Emotion vs. no-emotion comparison analysis

Please refer to README files in each directory for specific usage instructions.

### Data Files

- **Input Data**: Located in `Code/Section 1_experiment of LLM/prompt/`
  - `demographic data.xlsx`: Human demographic information (n=1017)
  - `Emo&TPP data.xlsx`: Actual human experiment settings
- **Source Data for Figures**: Located in `SourceData/`
  - `SourceData_Figure2.xlsx`
  - `SourceData_Figure3.xlsx`
  - `SourceData_Figure4.xlsx`
  - `SourceData_Figure5.xlsx`

---

## Core Code Modules

### Section 1: LLM Experiments (`Code/Section 1_experiment of LLM/`)

#### `run_par.py` - Main Execution Script
- **Function**: Orchestrates parallel execution of LLM experiments across multiple subjects
- **Features**: Multiprocessing parallel execution, automatic file management

#### `generate_character_prompt.py` - Character Persona Generation
- **Function**: Generates LLM character personas based on human demographic data
- **Features**: 
  - Reads demographic data (n=1017)
  - Integrates personality traits: AQ, ERS, CESD, social value orientation, justice sensitivity
  - Converts Chinese location names to Pinyin
  - Generates JSON-formatted character description files

#### `generate_game_setting_prompt.py` - Game Scenario Generation
- **Function**: Generates game setting prompts for each experimental trial
- **Features**: Reads experimental settings, creates trial-specific scenarios with allocation amounts, cost levels, and cost amounts

#### `multi_round_person.py` - Core LLM Experiment Execution
- **Function**: Executes multi-round LLM experiments with persona and emotion conditions
- **Supported Models**: GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini
- **Configurable Conditions**:
  - `persona`: True/False (with/without persona)
  - `emotion`: True/False (with/without emotion self-reporting)
  - `TEMPERATURE`: 0.0 or 1.0 (temperature parameter)
- **Features**: Multi-round interaction (default: 60 rounds per subject), extracts emotion reports and allocation decisions from LLM responses

#### `check.ipynb` - Result Validation and Merging
- **Function**: Validates experimental outputs and merges results across models
- **Features**: Checks output file integrity, parses emotion and choice data, generates merged data files

### Section 4: Demographic Analysis

#### `choice_moderation_analysis.py` - Choice Moderation Analysis
- **Function**: Analyzes how demographic variables moderate choice outcomes across groups
- **Features**: Tests moderation effects of demographic variables on allocation choices, cross-group comparisons

#### `emotion_moderation_analysis.py` - Emotion Moderation Analysis
- **Function**: Analyzes how demographic variables moderate emotion outcomes
- **Features**: Tests moderation effects on 6 emotion variables, interaction effect analysis

### Section 5: SHAP Analysis

#### `1_Analysis_of_SHAP.ipynb` - SHAP Value Computation
- **Function**: Computes SHAP (SHapley Additive exPlanations) values for model interpretability
- **Features**: Uses XGBoost models to predict outcomes, computes SHAP values to understand feature importance

#### `2_LMM.Rmd` - Linear Mixed Model Analysis
- **Function**: Statistical analysis using linear mixed models

### Study 1: Statistical Analyses

#### `1_Regression/` - Regression Analysis
- **GLMM1_choice_continuous.R**: Generalized linear mixed models for continuous choices
- **GLMM2_choice_category.py**: GLMM for categorical choices
- **LMM1_emotion.py**: Linear mixed models for emotion variables

#### `2_Partial correlation/` - Partial Correlation Analysis
- **Function**: Computes partial correlations controlling for covariates

#### `3_RSA/` - Representational Similarity Analysis
- **Function**: RSA analysis comparing representational structures across groups
- **Features**: Mantel test, 7-variable comparison, cross-group validation

#### `4_mediation/` - Mediation Analysis
- **Function**: Mediation analysis for unfair conditions

### Study 2: Comparative Analyses

#### `Study 2a_emo vs. no/` - Emotion vs. No-emotion Comparison
- **Function**: Compares outcomes between emotion and no-emotion conditions

#### `Study 2b_emo vs. math/` - Emotion vs. Math Condition Comparison
- **Function**: T-test analysis comparing emotion and math conditions

---

## Data Flow

```
Demographic Data → Character Generation → Game Setting Generation → 
LLM Experiment Execution → Result Validation → Merged Data → 
Statistical Analysis → Visualization
```

### Input Data
- `demographic data.xlsx`: Human participant demographic information (n=1017)
- `Emo&TPP data.xlsx`: Experimental trial parameters

### Output Data
- Per-subject output files: `output_{subjnum}.txt`
- Merged data files: `merged_all_models_*.txt` or `*.csv`
- Analysis results: Various CSV/Excel files containing statistical results

---

## Reproducing Research Results (Optional)

To reproduce the quantitative results reported in the study:

1. **Run Full LLM Experiments**:
   - Set `subjnum_list = range(0, 1017)` in `run_par.py`
   - Configure conditions as specified in the study
   - Run `python run_par.py` (may take several hours)

2. **Run Statistical Analyses**:
   - Execute scripts in `Study 1/` and `Study 2/` directories
   - Follow the order specified in each section's README

3. **Generate Figures**:
   - Use source data from the `SourceData/` folder
   - Run visualization scripts in respective sections

*Note: Full reproduction requires API access and may incur significant API costs. The demo uses a data subset (2 subjects) for validation purposes.*

---


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or suggestions, please contact us through GitHub Issues.

---

**Note**: This project is intended for academic research purposes. API keys and sensitive data have been removed. Users must provide their own API access.

