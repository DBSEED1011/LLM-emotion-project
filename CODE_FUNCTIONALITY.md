# Code Functionality and Pseudocode

This document describes the functionality of the codebase, including main modules, input/output formats, execution workflows, and pseudocode descriptions of core algorithms.

---

## Code Functionality

This section describes the functionality of the codebase, including main modules, input/output formats, and execution workflows.

### Overall Functionality

The codebase implements a comprehensive pipeline for conducting LLM-based emotion and decision-making experiments, including:
1. **LLM Experiment Execution**: Multi-round experiments with persona and emotion conditions across multiple LLM models
2. **Data Analysis**: Statistical analyses including regression, moderation, mediation, and interpretability studies
3. **Result Validation**: Automated checking and merging of experimental outputs

### Main Code Modules

#### Section 1: LLM Experiments (`Code/Section 1_experiment of LLM/`)

**`run_par.py`** - Main execution script
- **Function**: Orchestrates parallel execution of LLM experiments across multiple subjects
- **Key Features**:
  - Parallel processing using multiprocessing (utilizes all CPU cores)
  - Sequential execution of three scripts per subject: character prompt generation → game setting generation → LLM experiment
  - Automatic file management and subject-specific file generation
- **Input**: Subject list (`subjnum_list`), configuration parameters
- **Output**: Subject-specific prompt files and result folders

**`generate_character_prompt.py`** - Character persona generation
- **Function**: Generates LLM character personas based on human demographic data
- **Key Features**:
  - Reads demographic data from `demographic data.xlsx` (n=1017)
  - Incorporates personality traits: AQ (autism quotient), ERS (emotional reactivity), CESD (depression), social value orientation, justice sensitivity
  - Converts Chinese location names to Pinyin for consistency
  - Generates JSON files with character descriptions (`{subjnum}_character.json`)
- **Input**: `demographic data.xlsx` (Excel file with demographic variables)
- **Output**: `{subjnum}_character.json` (JSON file with character persona descriptions)

**`generate_game_setting_prompt.py`** - Game scenario generation
- **Function**: Generates game setting prompts for each experimental trial
- **Key Features**:
  - Reads experimental settings from `Emo&TPP data.xlsx`
  - Creates trial-specific game scenarios with allocation amounts, cost levels, and cost amounts
  - Generates JSON files with game settings (`{subjnum}_game_setting_prompt.json`)
- **Input**: `Emo&TPP data.xlsx` (Excel file with experimental trial parameters)
- **Output**: `{subjnum}_game_setting_prompt.json` (JSON file with game scenario parameters)

**`multi_round_person.py`** - Core LLM experiment execution
- **Function**: Executes multi-round LLM experiments with persona and emotion conditions
- **Key Features**:
  - Supports multiple LLM models: GPT-3.5, DeepSeek-V3, DeepSeek-R1, o3-mini
  - Configurable experimental conditions:
    - `persona`: True/False (with or without persona)
    - `emotion`: True/False (with or without emotion self-reporting)
    - `TEMPERATURE`: 0.0 or 1.0 (temperature parameter)
  - Multi-round interaction (default: 60 rounds per subject)
  - Extracts emotion reports and allocation decisions from LLM responses
  - Saves results to model-specific folders (`result_persona_emotion_{temperature}_{model}/`)
- **Input**: 
  - Character JSON files (`{subjnum}_character.json`)
  - Game setting JSON files (`{subjnum}_game_setting_prompt.json`)
  - General prompt template (`person_all_game_prompt.json`)
- **Output**: 
  - Text files (`output_{subjnum}.txt`) containing round-by-round emotion reports and choices
  - Format: Each round includes "Emotion: [report]" and "Choice: [decision]"

**`check.ipynb`** - Result validation and merging
- **Function**: Validates experimental outputs and merges results across models
- **Key Features**:
  - Checks output file integrity
  - Parses emotion and choice data from text files
  - Merges data across all LLM models
  - Generates consolidated data files for downstream analysis
- **Input**: Result folders from `multi_round_person.py`
- **Output**: Merged data files (e.g., `merged_all_models_persona_emotion_1.0.txt`)

#### Section 3: Chain of Thought Mediation (`Code/Section 3_CoT_mediation/`)

**`CoT_mediation.Rmd`** - Chain of Thought mediation analysis
- **Function**: Analyzes mediation effects of Chain of Thought reasoning on outcomes
- **Input**: Reasoning example files (`Reasoning_example.xlsx`, `Reasoning_with_emotions.xlsx`)
- **Output**: Mediation analysis results and visualizations

#### Section 4: Demographic Analysis (`Code/Section 4_Demograpic analysis/`)

**`choice_moderation_analysis.py`** - Choice moderation analysis
- **Function**: Analyzes how demographic variables moderate choice outcomes across groups
- **Key Features**:
  - Tests moderation effects of demographic variables (AQ, ERS, CESD, etc.) on allocation choices
  - Compares moderation patterns across groups (Human, GPT-3.5, o3-mini, DeepSeek-V3, DeepSeek-R1)
  - Generates statistical reports and visualizations
- **Input**: `data_with_grouping_variables.xlsx`
- **Output**: Moderation analysis reports and plots

**`emotion_moderation_analysis.py`** - Emotion moderation analysis
- **Function**: Analyzes how demographic variables moderate emotion outcomes
- **Key Features**:
  - Tests moderation effects on 6 emotion variables (valence, arousal, etc.)
  - Group comparisons and interaction effect analysis
  - Statistical significance testing
- **Input**: `data_with_grouping_variables.xlsx`
- **Output**: Emotion moderation analysis reports

**`plot_combined_moderation.py`** - Combined visualization
- **Function**: Generates combined plots for moderation analyses
- **Output**: Publication-ready figures

#### Section 5: SHAP Analysis (`Code/Section 5_SHAP/`)

**`1_Analysis_of_SHAP.ipynb`** - SHAP value computation
- **Function**: Computes SHAP (SHapley Additive exPlanations) values for model interpretability
- **Key Features**:
  - Uses XGBoost models to predict outcomes
  - Computes SHAP values to understand feature importance
  - Generates SHAP plots and summary statistics
- **Input**: `merged_all_models_persona.csv`
- **Output**: `shap_results_weighted.csv`, `xgb_oos_performance.csv`

**`2_LMM.Rmd`** - Linear mixed model analysis
- **Function**: Statistical analysis using linear mixed models
- **Input**: SHAP results and demographic data
- **Output**: LMM results and visualizations

#### Study 1: Statistical Analyses (`Code/Study 1/`)

**`1_Regression/GLMM1_choice_continuous.R`** - Generalized linear mixed model for continuous choices
- **Function**: Fits GLMM models for continuous allocation choices
- **Input**: `merged_all_data.xlsx`
- **Output**: Model coefficients, significance tests, model fit statistics

**`1_Regression/GLMM2_choice_category.py`** - GLMM for categorical choices
- **Function**: Fits GLMM models for categorical choice outcomes
- **Output**: Model results, interaction effects, post-hoc tests

**`1_Regression/LMM1_emotion.py`** - Linear mixed model for emotions
- **Function**: Fits LMM models for emotion variables
- **Output**: Emotion model results

**`2_Partial correlation/`** - Partial correlation analysis
- **Function**: Computes partial correlations controlling for covariates
- **Input**: Emotion and feedback data
- **Output**: Partial correlation matrices and participant-level results

**`3_RSA/`** - Representational Similarity Analysis
- **Function**: RSA analysis comparing representational structures across groups
- **Key Features**:
  - Mantel test for correlation between representational matrices
  - 7-variable comparison (emotion and choice variables)
  - Cross-group validation
- **Input**: Mean data files with emotion variables
- **Output**: RSA correlation matrices and Mantel test results

**`4_mediation/merged_all_data_unfair.R`** - Mediation analysis
- **Function**: Mediation analysis for unfair conditions
- **Output**: Mediation effect estimates and indirect effects

#### Study 2: Comparative Analyses (`Code/Study 2/`)

**`Study 2a_emovs.no/Study2a_emovs.no.Rmd`** - Emotion vs. no emotion comparison
- **Function**: Compares outcomes between emotion and no-emotion conditions
- **Input**: `study2_GLMM.xlsx`
- **Output**: Model results (`model_results_m6.csv`), post-hoc tests (`posthoc_results_m6.csv`)

**`Study 2b_emo vs. math/Study2b_emo_vs_math_t_test_analysis.py`** - Emotion vs. math condition comparison
- **Function**: T-test analysis comparing emotion and math conditions
- **Input**: `Study2b_emo_vs_math.xlsx`
- **Output**: T-test results (`emo_math_t_test_results.csv`)

### Data Flow

1. **Input Data**:
   - `demographic data.xlsx`: Human participant demographics (n=1017)
   - `Emo&TPP data.xlsx`: Experimental trial parameters

2. **Processing Pipeline**:
   ```
   Demographic Data → Character Generation → Game Setting Generation → 
   LLM Experiment Execution → Result Validation → Merged Data
   ```

3. **Output Data**:
   - Per-subject output files: `output_{subjnum}.txt`
   - Merged data files: `merged_all_models_*.txt` or `*.csv`
   - Analysis results: Various CSV/Excel files with statistical results

### Key Functions and Classes

- **`extract_game_setting(cha_num, round)`**: Extracts game parameters for a specific character and round
- **`BaseMessage`**: Message class for LLM API communication
- **`RoleType`**: Enum for message roles (user/system)
- **`ExtendedModelType`**: Model type definitions for different LLMs
- **API Functions**: `deepseek_chat()`, `openai_chat()` for LLM API calls

### Execution Workflow

1. **Setup**: Configure API keys and experimental parameters
2. **Prompt Generation**: Generate character and game setting prompts for each subject
3. **Experiment Execution**: Run multi-round LLM experiments in parallel
4. **Result Validation**: Check outputs and merge data across models
5. **Statistical Analysis**: Run regression, moderation, mediation, and interpretability analyses
6. **Visualization**: Generate figures and summary reports

---

## Pseudocode

This section provides pseudocode descriptions of the core algorithms implemented in this codebase.

### Algorithm 1: Main Parallel Execution Pipeline (`run_par.py`)

```
FUNCTION process_subject(subjnum):
    // Step 1: Generate character prompt
    character_file = modify_and_generate_file('generate_character_prompt.py', subjnum)
    EXECUTE character_file
    
    // Step 2: Generate game setting prompt
    game_setting_file = modify_and_generate_file('generate_game_setting_prompt.py', subjnum)
    EXECUTE game_setting_file
    
    // Step 3: Run LLM experiment
    experiment_file = modify_and_generate_file('multi_round_person.py', subjnum)
    EXECUTE experiment_file
    
    RETURN success

FUNCTION main():
    subjnum_list = [0, 1, 2, ..., N-1]  // List of subject numbers
    max_processes = CPU_COUNT()
    active_processes = []
    
    FOR EACH subjnum IN subjnum_list:
        // Wait if too many processes running
        WHILE LENGTH(active_processes) >= max_processes:
            FOR EACH process IN active_processes:
                IF NOT process.is_alive():
                    REMOVE process FROM active_processes
        
        // Start new process
        process = CREATE_PROCESS(process_subject, subjnum)
        START process
        ADD process TO active_processes
    
    // Wait for all processes to complete
    FOR EACH process IN active_processes:
        WAIT_FOR process
    
    RETURN
```

### Algorithm 2: Character Persona Generation (`generate_character_prompt.py`)

```
FUNCTION is_chinese(text):
    FOR EACH character IN text:
        IF character NOT IN Chinese_Unicode_Range:
            RETURN False
    RETURN True

FUNCTION get_pinyin(chinese_text):
    IF chinese_text ENDS WITH "市" OR "省":
        chinese_text = REMOVE_LAST_CHARACTER(chinese_text)
    
    pinyin_list = CONVERT_TO_PINYIN(chinese_text)
    pinyin_str = CONCATENATE(pinyin_list, capitalize_first=True)
    RETURN pinyin_str

FUNCTION generate_personality_description(demographic_row, average_values):
    // Basic information
    IF demographic_row["city"] IS Chinese:
        city_output = get_pinyin(demographic_row["city"]) + " city"
    ELSE:
        city_output = demographic_row["city"]
    
    IF demographic_row["province"] IS Chinese:
        province_output = get_pinyin(demographic_row["province"]) + " province"
    ELSE:
        province_output = demographic_row["province"]
    
    basic_info = "You are a {age}-year-old {gender} from {city}, {province}."
    
    // Autism tendency description
    autism_tendency = "Your total AQ score is {score} (average: {avg})."
    autism_detail = "Sub-dimension scores: Social skills: {score1}, Routine: {score2}, ..."
    
    // Emotional reactivity description
    emotional_reactivity = "Your total ERS score is {score} (average: {avg})."
    emotional_detail = "Sub-dimension scores: Duration: {score1}, Sensitivity: {score2}, ..."
    
    // Depression tendency description
    depression_tendency = "Your total CESD score is {score} (average: {avg})."
    depression_detail = "Sub-dimension scores: Positive affect: {score1}, Depression: {score2}, ..."
    
    // Social value orientation
    IF demographic_row["SVO"] == 1:
        svo_meaning = "prosocial"
    ELSE IF demographic_row["SVO"] == 0:
        svo_meaning = "proself"
    ELSE:
        svo_meaning = "undifferentiated"
    
    // Personality type
    personality_type = "Your personality type is {type} based on experimental choices."
    
    // Justice sensitivity
    justice_sensitivity = "Your JSI score is {score} (average: {avg})."
    
    RETURN {
        "basic_information": basic_info,
        "autism_tendency": autism_tendency,
        "autism_tendency_detail": autism_detail,
        "emotional_reactivity": emotional_reactivity,
        "emotional_reactivity_detail": emotional_detail,
        "depression_tendency": depression_tendency,
        "depression_tendency_detail": depression_detail,
        "social_value_orientation": svo_description,
        "personality_type": personality_type,
        "personality_type_detail": personality_detail,
        "justice_sensitivity": justice_sensitivity
    }

FUNCTION main(subjnum):
    demographic_data = LOAD_EXCEL("demographic data.xlsx")
    average_values = LOAD_AVERAGE_VALUES()
    
    row = demographic_data[subjnum]
    character_description = generate_personality_description(row, average_values)
    
    character_list = [character_description]
    SAVE_TO_JSON(character_list, "{subjnum}_character.json")
    
    RETURN
```

### Algorithm 3: Game Setting Generation (`generate_game_setting_prompt.py`)

```
FUNCTION main(subjnum):
    experimental_data = LOAD_EXCEL("Emo&TPP data.xlsx")
    
    // Extract 60 trials for this subject
    start_row = 60 * subjnum
    end_row = 60 * (subjnum + 1)
    subject_trials = experimental_data[start_row:end_row]
    
    // Select relevant columns
    game_settings = SELECT_COLUMNS(subject_trials, 
        ["id", "trial", "amount_of_allocation", "cost_level", "amount_of_cost"])
    
    // Add index column
    game_settings["index"] = subjnum
    
    // Convert to JSON format
    json_data = CONVERT_TO_JSON(game_settings)
    SAVE_TO_FILE(json_data, "{subjnum}_game_setting_prompt.json")
    
    RETURN
```

### Algorithm 4: Multi-Round LLM Experiment (`multi_round_person.py`)

```
FUNCTION extract_game_setting(character_num, round_num):
    game_settings = LOAD_JSON("{subjnum}_game_setting_prompt.json")
    
    FOR EACH item IN game_settings:
        IF item["index"] == character_num AND item["trial"] == round_num + 1:
            RETURN {
                "amount_of_allocation": item["amount_of_allocation"],
                "cost_level": item["cost_level"],
                "amount_of_cost": item["amount_of_cost"]
            }
    
    RETURN None

FUNCTION call_llm_api(prompt, model_name, temperature, emotion_condition):
    IF emotion_condition == True:
        system_prompt = "You are a participant. Report emotions and make choices. 
                         Format: AA_valence = [number], AA_arousal = [number], 
                         choice = [0 or 1], AC_valence = [number], AC_arousal = [number]"
    ELSE:
        system_prompt = "You are a participant. Make choices. 
                         Format: choice = [0 or 1]"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = API_CALL(model_name, messages, temperature)
    RETURN response.content

FUNCTION parse_response(response_text, emotion_condition):
    IF emotion_condition == True:
        pattern = "AA_valence|AA_arousal|choice|AC_valence|AC_arousal = (-?\d+)"
        matches = EXTRACT_MATCHES(response_text, pattern)
        
        result = {
            "AA_valence": EXTRACT_VALUE(matches, "AA_valence"),
            "AA_arousal": EXTRACT_VALUE(matches, "AA_arousal"),
            "choice": EXTRACT_VALUE(matches, "choice"),
            "AC_valence": EXTRACT_VALUE(matches, "AC_valence"),
            "AC_arousal": EXTRACT_VALUE(matches, "AC_arousal")
        }
        
        // Calculate emotion feedback (change from before to after choice)
        result["EmoFDBK_valence"] = result["AC_valence"] - result["AA_valence"]
        result["EmoFDBK_arousal"] = result["AC_arousal"] - result["AA_arousal"]
    ELSE:
        pattern = "choice = (\d+)"
        matches = EXTRACT_MATCHES(response_text, pattern)
        
        result = {
            "choice": EXTRACT_VALUE(matches, "choice")
        }
    
    result["Output"] = response_text
    RETURN result

FUNCTION run_multi_round_experiment(subjnum, model_type, persona_flag, emotion_flag, temperature):
    // Load character descriptions
    IF persona_flag == True:
        characters = LOAD_JSON("{subjnum}_character.json")
    ELSE:
        characters = [EMPTY_DESCRIPTION]
    
    // Load game settings
    game_settings = LOAD_JSON("{subjnum}_game_setting_prompt.json")
    
    // Load general prompt template
    general_prompt = LOAD_JSON("person_all_game_prompt.json")
    
    all_results = []
    num_rounds = 60
    
    FOR EACH character IN characters:
        character_index = INDEX_OF(character, characters)
        
        // Set up role message
        IF persona_flag == True:
            role_message = character["description"] + "You are a human being, not an AI."
        ELSE:
            role_message = "You are a human being, not an AI."
        
        round_results = []
        
        FOR round = 0 TO num_rounds - 1:
            // Extract game parameters for this round
            game_params = extract_game_setting(character_index, round)
            allocation = game_params["amount_of_allocation"]
            cost_level = game_params["cost_level"]
            cost_amount = game_params["amount_of_cost"]
            
            // Construct round prompt
            round_prompt = "This is round {round+1}. "
            scenario_prompt = "Player 1 allocates {allocation} to Player 2 and {30-allocation} to themselves. 
                              Cost level: {cost_level}. If you punish, pay {cost_amount}. Make your choice."
            
            full_prompt = CONCATENATE(role_message, round_prompt, scenario_prompt)
            
            // Call LLM API
            response = call_llm_api(full_prompt, model_type, temperature, emotion_flag)
            
            // Parse response
            parsed_result = parse_response(response, emotion_flag)
            parsed_result["round"] = round + 1
            parsed_result["character_index"] = character_index
            
            ADD parsed_result TO round_results
            
            // Save intermediate results (after each round)
            output_file = "result_{persona_str}_{emotion_str}_{temperature}_{model}/output_{subjnum}.txt"
            SAVE_RESULTS(round_results, output_file)
        
        ADD round_results TO all_results
    
    RETURN all_results

FUNCTION main(subjnum):
    // Configuration
    persona = True/False
    emotion = True/False
    TEMPERATURE = 0.0 OR 1.0
    models = ["gpt-3.5-turbo-0125", "deepseek-v3", "deepseek-r1", "o3-mini-2025-01-31"]
    
    FOR EACH model IN models:
        results = run_multi_round_experiment(subjnum, model, persona, emotion, TEMPERATURE)
        SAVE_RESULTS(results, "result_{persona}_{emotion}_{TEMPERATURE}_{model}/output_{subjnum}.txt")
    
    RETURN
```

### Algorithm 5: Moderation Analysis (`emotion_moderation_analysis.py`)

```
FUNCTION analyze_moderation(data, grouping_variable, outcome_variable, group_variable):
    // Fit regression model with interaction term
    // outcome ~ grouping_var + group + grouping_var × group + covariates
    
    groups = UNIQUE_VALUES(data[group_variable])
    results = []
    
    FOR EACH group IN groups:
        group_data = FILTER(data, group_variable == group)
        
        // Fit OLS model with interaction
        model = FIT_MODEL(
            outcome_variable ~ grouping_variable + group + 
            grouping_variable:group + covariates,
            data = group_data
        )
        
        // Extract coefficients
        main_effect = GET_COEFFICIENT(model, grouping_variable)
        interaction_effect = GET_COEFFICIENT(model, "grouping_variable:group")
        p_value = GET_P_VALUE(model, interaction_effect)
        
        // Calculate simple slopes
        simple_slopes = CALCULATE_SIMPLE_SLOPES(model, grouping_variable, group_variable)
        
        ADD {
            "group": group,
            "main_effect": main_effect,
            "interaction_effect": interaction_effect,
            "p_value": p_value,
            "simple_slopes": simple_slopes
        } TO results
    
    RETURN results

FUNCTION main():
    data = LOAD_DATA("data_with_grouping_variables.xlsx")
    
    grouping_variables = ["AQ_score", "ERS_score", "CESD_score", ...]
    outcome_variables = ["valence", "arousal", "emotion_feedback_valence", ...]
    group_variable = "group"  // Human, GPT-3.5, o3-mini, DeepSeek-V3, DeepSeek-R1
    
    FOR EACH grouping_var IN grouping_variables:
        FOR EACH outcome_var IN outcome_variables:
            results = analyze_moderation(data, grouping_var, outcome_var, group_variable)
            SAVE_RESULTS(results, "moderation_{grouping_var}_{outcome_var}.csv")
            PLOT_MODERATION_EFFECTS(results)
    
    RETURN
```

### Algorithm 6: SHAP Value Computation (`1_Analysis_of_SHAP.ipynb`)

```
FUNCTION compute_shap_values(data, target_variable, features):
    // Split data into train and test sets
    train_data, test_data = SPLIT(data, train_ratio=0.8)
    
    // Train XGBoost model
    model = TRAIN_XGBOOST(
        features = features,
        target = target_variable,
        data = train_data
    )
    
    // Compute SHAP values
    shap_explainer = CREATE_SHAP_EXPLAINER(model)
    shap_values = COMPUTE_SHAP_VALUES(shap_explainer, test_data[features])
    
    // Calculate feature importance
    feature_importance = MEAN(ABSOLUTE(shap_values), axis=0)
    
    // Calculate out-of-sample performance
    predictions = PREDICT(model, test_data[features])
    performance_metrics = CALCULATE_METRICS(test_data[target_variable], predictions)
    
    RETURN {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "performance": performance_metrics
    }

FUNCTION main():
    data = LOAD_DATA("merged_all_models_persona.csv")
    
    features = ["allocation", "cost_level", "cost_amount", "AQ_score", "ERS_score", ...]
    target = "choice"  // or "valence", "arousal", etc.
    
    results = compute_shap_values(data, target, features)
    
    SAVE_CSV(results["shap_values"], "shap_results_weighted.csv")
    SAVE_CSV(results["performance"], "xgb_oos_performance.csv")
    
    // Generate SHAP plots
    PLOT_SHAP_SUMMARY(shap_values, features)
    PLOT_SHAP_WATERFALL(shap_values, example_instance)
    
    RETURN
```

### Algorithm 7: RSA Analysis (`3_RSA/unfair_7var_validation_rsa_mantel_analysis.py`)

```
FUNCTION compute_representational_matrix(data, variables):
    // Compute correlation matrix between variables
    correlation_matrix = CORRELATION(data[variables])
    RETURN correlation_matrix

FUNCTION mantel_test(matrix1, matrix2):
    // Mantel test for correlation between two representational matrices
    // H0: matrices are independent
    // H1: matrices are correlated
    
    observed_correlation = CORRELATION(matrix1, matrix2)
    
    // Permutation test
    n_permutations = 10000
    permuted_correlations = []
    
    FOR i = 1 TO n_permutations:
        permuted_matrix2 = PERMUTE_ROWS_AND_COLUMNS(matrix2)
        perm_corr = CORRELATION(matrix1, permuted_matrix2)
        ADD perm_corr TO permuted_correlations
    
    p_value = COUNT(permuted_correlations >= observed_correlation) / n_permutations
    
    RETURN {
        "correlation": observed_correlation,
        "p_value": p_value
    }

FUNCTION main():
    variables = ["valence", "arousal", "emotion_feedback_valence", 
                 "emotion_feedback_arousal", "choice", "allocation", "cost"]
    
    groups = ["human", "gpt3.5", "o3", "V3", "R1"]
    
    // Load data for each group
    group_matrices = {}
    FOR EACH group IN groups:
        group_data = LOAD_DATA("unfair_all_datasets_means_{group}.xlsx")
        matrix = compute_representational_matrix(group_data, variables)
        group_matrices[group] = matrix
    
    // Compare each group with human
    mantel_results = {}
    human_matrix = group_matrices["human"]
    
    FOR EACH group IN ["gpt3.5", "o3", "V3", "R1"]:
        result = mantel_test(human_matrix, group_matrices[group])
        mantel_results[group] = result
    
    // Create correlation matrix of correlations
    correlation_matrix = CREATE_MATRIX(mantel_results)
    p_value_matrix = CREATE_P_VALUE_MATRIX(mantel_results)
    
    PLOT_MANTEL_RESULTS(correlation_matrix, p_value_matrix, groups)
    SAVE_RESULTS(mantel_results, "rsa_mantel_results.csv")
    
    RETURN
```

