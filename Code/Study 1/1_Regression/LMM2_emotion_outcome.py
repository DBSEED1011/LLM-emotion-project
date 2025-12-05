import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

def format_p_value(p_value):
    """Format p-value"""
    if p_value < 0.001:
        return "< 0.001"
    elif p_value < 0.01:
        return f"= {p_value:.3f}"
    elif p_value < 0.05:
        return f"= {p_value:.3f}"
    else:
        return f"= {p_value:.3f}"

def run_mixed_model_for_table(dependent_var, df_clean):
    """Run mixed effects model and return results for table"""
    # Use existing choice variable in data: punish=1, accept=0
    
    # Recode group: human=0(reference group), other groups=1,2,3,4
    group_mapping = {'human': 0, 'R1': 4, 'V3': 3, 'gpt3.5': 1, 'o3': 2}
    df_clean['group_recoded'] = df_clean['group'].map(group_mapping)
    df_clean['group_recoded'] = df_clean['group_recoded'].astype('category')
    
    # New regression formula: Emotional feedback valence ~ Group + Choice + Group:Choice + (1 | ID)
    formula = f'{dependent_var} ~ group_recoded + choice + group_recoded:choice'
    model = mixedlm(formula, df_clean, groups=df_clean['participant_id'], re_formula='1')
    result = model.fit(method='powell')
    return result



def create_regression_table(result, dependent_var_name):
    """Create three-line table for regression results"""
    
    # Get parameter information
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    
    # Calculate 95% confidence intervals
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse
    
    # Create table data
    table_data = []
    
    # Intercept (Human group under Accept conditions)
    intercept_beta = params['Intercept']
    intercept_se = bse['Intercept']
    intercept_z = intercept_beta / intercept_se
    intercept_p = pvalues['Intercept']
    intercept_ci_lower = ci_lower['Intercept']
    intercept_ci_upper = ci_upper['Intercept']
    
    table_data.append({
        'Variable': 'Intercept (Human, Accept)',
        'β': f"{intercept_beta:.3f}",
        'SE': f"{intercept_se:.3f}",
        '95% CI': f"[{intercept_ci_lower:.3f}, {intercept_ci_upper:.3f}]",
        'z': f"{intercept_z:.3f}",
        'p': format_p_value(intercept_p)
    })
    
    # Choice main effect (Human group choice effect, since Human is reference group)
    choice_beta = params['choice']
    choice_se = bse['choice']
    choice_z = choice_beta / choice_se
    choice_p = pvalues['choice']
    choice_ci_lower = ci_lower['choice']
    choice_ci_upper = ci_upper['choice']
    
    table_data.append({
        'Variable': 'Choice (Human group)',
        'β': f"{choice_beta:.3f}",
        'SE': f"{choice_se:.3f}",
        '95% CI': f"[{choice_ci_lower:.3f}, {choice_ci_upper:.3f}]",
        'z': f"{choice_z:.3f}",
        'p': format_p_value(choice_p)
    })
    
    
    # Group effects (relative to Human group, average differences across all conditions)
    group_mapping = {1: 'gpt3.5', 2: 'o3', 3: 'V3', 4: 'R1'}
    for param in params.index:
        if param.startswith('group_recoded[T.') and not ':choice' in param:
            group_num = int(param.replace('group_recoded[T.', '').replace(']', ''))
            group_name = group_mapping[group_num]
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Group ({group_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Interaction effects (Additional effect differences of Choice across different groups)
    for param in params.index:
        if 'group_recoded[T.' in param and ':choice' in param:
            group_num = int(param.replace('group_recoded[T.', '').replace(']:choice', ''))
            group_name = group_mapping[group_num]
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Choice × {group_name}',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    return df_table

def create_model_info_table(result, dependent_var_name, n_groups):
    """Create model information table"""
    
    # Calculate ICC
    icc = result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale)
    
    model_info = {
        'Model Information': [
            'Dependent Variable',
            'Number of Participants',
            'Number of Observations',
            'Log-likelihood',
            'Convergence Status',
            'Between-group Variance',
            'Residual Variance',
            'Intraclass Correlation Coefficient (ICC)'
        ],
        'Value': [
            dependent_var_name,
            f"{n_groups}",
            f"{result.nobs}",
            f"{result.llf:.2f}",
            "Yes" if result.converged else "No",
            f"{result.cov_re.iloc[0,0]:.3f}",
            f"{result.scale:.3f}",
            f"{icc:.3f}"
        ]
    }
    
    return pd.DataFrame(model_info)


def main():
    print("=== Generate Regression Results Three-line Tables (Choice Analysis) ===\n")
    
    # Load data
    df = pd.read_excel('merged_all_data.xlsx')
    df_clean = df.dropna(subset=['EmoFDBK_valence', 'EmoFDBK_arousal', 'amount_of_allocation', 'group', 'id', 'trial', 'choice'])
    
    # Filter data points where amount_of_allocation is between 10-14
    df_clean = df_clean[(df_clean['amount_of_allocation'] >= 10) & (df_clean['amount_of_allocation'] <= 14)]
    
    # Create unique participant ID
    df_clean['participant_id'] = df_clean['id'].astype(str) + '_' + df_clean['group']
    
    print("=== Data Filtering and Encoding Settings ===")
    print("Data filtering: amount_of_allocation = 10-14")
    print("Choice variable: Use existing choice variable in data (punish=1, accept=0)")
    print("Group encoding: Human=0(reference group), R1=4, V3=3, gpt3.5=1, o3=2")
    print("Reference group: Human group under Accept conditions")
    print(f"Number of data points after filtering: {len(df_clean)}")
    print(f"Number of participants: {df_clean['participant_id'].nunique()}")
    print()
    
    # Run models
    print("Running EmoFDBK_valence model...")
    valence_result = run_mixed_model_for_table('EmoFDBK_valence', df_clean)
    
    print("Running EmoFDBK_arousal model...")
    arousal_result = run_mixed_model_for_table('EmoFDBK_arousal', df_clean)
    
    # Generate tables
    print("\nGenerating regression results tables...")
    valence_regression_table = create_regression_table(valence_result, 'EmoFDBK_valence')
    arousal_regression_table = create_regression_table(arousal_result, 'EmoFDBK_arousal')
    
    print("Generating model information tables...")
    n_groups = df_clean['participant_id'].nunique()
    valence_model_info = create_model_info_table(valence_result, 'EmoFDBK_valence', n_groups)
    arousal_model_info = create_model_info_table(arousal_result, 'EmoFDBK_arousal', n_groups)
    
    # Save to Excel file
    print("\nSaving tables to Excel file...")
    with pd.ExcelWriter('choice_regression_tables_for_paper.xlsx', engine='openpyxl') as writer:
        valence_regression_table.to_excel(writer, sheet_name='Valence_Regression', index=False)
        arousal_regression_table.to_excel(writer, sheet_name='Arousal_Regression', index=False)
        valence_model_info.to_excel(writer, sheet_name='Valence_Model_Info', index=False)
        arousal_model_info.to_excel(writer, sheet_name='Arousal_Model_Info', index=False)
    
    
    # Display table preview
    print("\n=== EmoFDBK_valence Regression Results Table ===")
    print(valence_regression_table.to_string(index=False))
    
    print("\n=== EmoFDBK_arousal Regression Results Table ===")
    print(arousal_regression_table.to_string(index=False))
    
    print("\n=== EmoFDBK_valence Model Information Table ===")
    print(valence_model_info.to_string(index=False))
    
    print("\n=== EmoFDBK_arousal Model Information Table ===")
    print(arousal_model_info.to_string(index=False))
    
    print(f"\n=== Files Saved ===")
    print("1. choice_regression_tables_for_paper.xlsx - Tables in Excel format")
    print("Analysis description: Based on data with amount_of_allocation=10-14, analyzing the effect of Choice on emotional responses.")

if __name__ == "__main__":
    main()
