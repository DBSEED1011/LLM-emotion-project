# LLM Emotion Project

## Project Overview
This project is a comprehensive research study on Large Language Model (LLM) emotion analysis, containing multiple research sections and data analysis modules.

## Project Structure

### Code/
- **Section 1_experiment of LLM**: Multi-round LLM experiments with persona and emotion conditions, including parallel execution scripts, prompt generation, and result validation across multiple LLM models (GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini)
- **Section 3_CoT_mediation**: Chain of Thought mediation analysis
- **Section 4_Demographic analysis**: Demographic analysis
- **Section 5_SHAP**: SHAP value analysis for model interpretability
- **Section 6_Nopersona**: Analysis under no persona conditions
- **Section 7_Temperature**: Model performance analysis under different temperature parameters
- **Study 1**: First study, including regression analysis, partial correlation analysis, RSA analysis, etc.
- **Study 2**: Second study, including emotion vs. no emotion, emotion vs. math comparison analysis

### SourceData/
Contains source data files for various figures (Figure2-5)

## Main Features
- **Multi-round LLM experiments**: Parallel execution of experiments across multiple LLM models with persona and emotion conditions
- **Prompt engineering**: Automated generation of character prompts and game settings for LLM experiments
- **Demographic integration**: Incorporation of human demographic data (AQ, ERS, CESD scores) into LLM persona generation
- **Emotion self-reporting**: Experimental conditions with and without emotion self-reporting
- **Temperature parameter studies**: Analysis of model performance under different temperature settings
- **Model interpretability analysis (SHAP)**: Explainable AI analysis for understanding model decisions
- **Chain of Thought reasoning analysis**: Analysis of LLM reasoning processes and high-frequency words
- **Demographic variable moderation effect analysis**: Statistical analysis of how demographic factors influence outcomes

## Tech Stack
- **Python**: Data analysis, machine learning, LLM API integration (OpenAI, DeepSeek)
- **R**: Statistical analysis, GLMM/LMM modeling, visualization
- **Jupyter Notebook**: Interactive analysis and result validation
- **Excel**: Data storage and preliminary processing
- **APIs**: OpenAI GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini integration
- **Libraries**: pandas, numpy, scikit-learn, SHAP, pypinyin for Chinese text processing

## Usage

### Section 1 - LLM Experiments
1. **Setup**: Configure API keys in `multi_round_person.py` (removed for security)
2. **Run experiments**: Execute `run_par.py` for parallel processing across multiple LLM models
3. **Validate results**: Use `check.ipynb` to verify outputs and generate merged data files
4. **Customize**: Modify `subnum_list` in `run_par.py` to adjust number of subjects (max 1017)
5. **Conditions**: Adjust persona, emotion, and temperature parameters in `multi_round_person.py`

### Other Sections
Each Section and Study has independent analysis scripts and documentation. Please refer to the README files in each directory for specific usage instructions.

## License
See [LICENSE](LICENSE) file for details
