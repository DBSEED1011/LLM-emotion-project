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
    # Recode fairness: fair=0(reference group), unfair=1
    df_clean['fairness_recoded'] = (df_clean['amount_of_allocation'] != 15).astype(int)
    
    # Recode group: human=0(reference group), other groups=1,2,3,4
    group_mapping = {'human': 0, 'R1': 4, 'V3': 3, 'gpt3.5': 1, 'o3': 2}
    df_clean['group_recoded'] = df_clean['group'].map(group_mapping)
    df_clean['group_recoded'] = df_clean['group_recoded'].astype('category')
    
    formula = f'{dependent_var} ~ fairness_recoded + group_recoded + fairness_recoded:group_recoded'
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
    
    # Intercept (Human group under Fair conditions)
    intercept_beta = params['Intercept']
    intercept_se = bse['Intercept']
    intercept_z = intercept_beta / intercept_se
    intercept_p = pvalues['Intercept']
    intercept_ci_lower = ci_lower['Intercept']
    intercept_ci_upper = ci_upper['Intercept']
    
    table_data.append({
        'Variable': 'Intercept (Human, Fair)',
        'β': f"{intercept_beta:.3f}",
        'SE': f"{intercept_se:.3f}",
        '95% CI': f"[{intercept_ci_lower:.3f}, {intercept_ci_upper:.3f}]",
        'z': f"{intercept_z:.3f}",
        'p': format_p_value(intercept_p)
    })
    
    # Unfairness main effect (Human group fairness effect, since Human is reference group)
    unfairness_beta = params['fairness_recoded']
    unfairness_se = bse['fairness_recoded']
    unfairness_z = unfairness_beta / unfairness_se
    unfairness_p = pvalues['fairness_recoded']
    unfairness_ci_lower = ci_lower['fairness_recoded']
    unfairness_ci_upper = ci_upper['fairness_recoded']
    
    table_data.append({
        'Variable': 'Unfairness (Human group)',
        'β': f"{unfairness_beta:.3f}",
        'SE': f"{unfairness_se:.3f}",
        '95% CI': f"[{unfairness_ci_lower:.3f}, {unfairness_ci_upper:.3f}]",
        'z': f"{unfairness_z:.3f}",
        'p': format_p_value(unfairness_p)
    })
    
    # Group effects (relative to Human group, average differences across all conditions)
    group_mapping = {1: 'gpt3.5', 2: 'o3', 3: 'V3', 4: 'R1'}
    for param in params.index:
        if param.startswith('group_recoded[T.') and not 'fairness_recoded:' in param:
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
    
    # Interaction effects (Additional effect differences of Unfairness across different groups)
    for param in params.index:
        if 'fairness_recoded:group_recoded[T.' in param:
            group_num = int(param.replace('fairness_recoded:group_recoded[T.', '').replace(']', ''))
            group_name = group_mapping[group_num]
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Unfairness × {group_name}',
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
    
    # Check covariance matrix dimensions
    cov_re_shape = result.cov_re.shape
    
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
    print("=== Generate Regression Results Three-line Tables (Corrected Encoding) ===\n")
    
    # 读取数据
    df = pd.read_excel('merged_all_data.xlsx')
    df_clean = df.dropna(subset=['AA_valence', 'AA_arousal', 'amount_of_allocation', 'group', 'id', 'trial'])
    
    # 创建唯一的参与者ID
    df_clean['participant_id'] = df_clean['id'].astype(str) + '_' + df_clean['group']
    
    print("=== New Encoding Settings ===")
    print("Fairness encoding: Fair=0(reference group), Unfair=1")
    print("Group encoding: Human=0(reference group), R1=4, V3=3, gpt3.5=1, o3=2")
    print("Reference group: Human group under Fair conditions")
    print()
    
    # 运行两个模型
    print("Running AA_valence model...")
    valence_result = run_mixed_model_for_table('AA_valence', df_clean)
    
    print("Running AA_arousal model...")
    arousal_result = run_mixed_model_for_table('AA_arousal', df_clean)
    
    # 生成表格
    print("\nGenerating regression results tables...")
    valence_regression_table = create_regression_table(valence_result, 'AA_valence')
    arousal_regression_table = create_regression_table(arousal_result, 'AA_arousal')
    
    print("Generating model information tables...")
    n_groups = df_clean['participant_id'].nunique()
    valence_model_info = create_model_info_table(valence_result, 'AA_valence', n_groups)
    arousal_model_info = create_model_info_table(arousal_result, 'AA_arousal', n_groups)
    
    # 保存为Excel文件
    print("\nSaving tables to Excel file...")
    with pd.ExcelWriter('regression_tables_for_paper.xlsx', engine='openpyxl') as writer:
        valence_regression_table.to_excel(writer, sheet_name='Valence_Regression', index=False)
        arousal_regression_table.to_excel(writer, sheet_name='Arousal_Regression', index=False)
        valence_model_info.to_excel(writer, sheet_name='Valence_Model_Info', index=False)
        arousal_model_info.to_excel(writer, sheet_name='Arousal_Model_Info', index=False)
    
    
    # 显示表格预览
    print("\n=== AA_valence Regression Results Table ===")
    print(valence_regression_table.to_string(index=False))
    
    print("\n=== AA_arousal Regression Results Table ===")
    print(arousal_regression_table.to_string(index=False))
    
    print("\n=== AA_valence Model Information Table ===")
    print(valence_model_info.to_string(index=False))
    
    print("\n=== AA_arousal Model Information Table ===")
    print(arousal_model_info.to_string(index=False))
    
    print(f"\n=== Files Saved ===")
    print("1. regression_tables_for_paper.xlsx - Tables in Excel format")

if __name__ == "__main__":
    main()
