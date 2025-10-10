import pandas as pd
import numpy as np
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

def prepare_data():
    """Prepare data"""
    print("=== Data Preparation ===")
    
    # Read data
    df = pd.read_excel('merged_all_data.xlsx')
    df_clean = df.dropna(subset=['choice', 'amount_of_allocation', 'amount_of_cost', 'group', 'id', 'trial'])
    
    # Create unique participant IDs
    df_clean['participant_id'] = df_clean['id'].astype(str) + '_' + df_clean['group']
    
    # Detailed encoding
    # Fairness encoding (6 levels)
    fairness_mapping = {
        15: 0,  # 15:15 (reference group)
        14: 1,  # 16:14
        13: 2,  # 17:13
        12: 3,  # 18:12
        11: 4,  # 19:11
        10: 5   # 20:10
    }
    
    # Cost encoding (9 levels)
    cost_mapping = {
        0: 0,   # 0%
        1: 1,   # 10%
        2: 2,   # 20%
        3: 3,   # 30%
        4: 4,   # 40%
        5: 5,   # 50%
        6: 6,   # 60%
        7: 7,   # 70%
        8: 8,   # 80%
        9: 9    # 90%
    }
    
    df_clean['fairness_detailed'] = df_clean['amount_of_allocation'].map(fairness_mapping)
    df_clean['cost_detailed'] = df_clean['amount_of_cost'].map(cost_mapping)
    
    # Group encoding: human=0(reference group), other groups=1,2,3,4
    group_mapping = {'human': 0, 'R1': 4, 'V3': 3, 'gpt3.5': 1, 'o3': 2}
    df_clean['group_recoded'] = df_clean['group'].map(group_mapping)
    df_clean['group_recoded'] = df_clean['group_recoded'].astype('category')
    df_clean['fairness_detailed'] = df_clean['fairness_detailed'].astype('category')
    df_clean['cost_detailed'] = df_clean['cost_detailed'].astype('category')
    
    print(f"Total observations: {len(df_clean)}")
    print(f"Number of participants: {df_clean['participant_id'].nunique()}")
    print(f"Punishment decision distribution: {df_clean['choice'].value_counts().to_dict()}")
    print()
    
    return df_clean

def fit_three_way_model(df_clean):
    """Fit three-way interaction model"""
    print("=== Fitting Three-way Interaction Model ===")
    
    # Three-way interaction model: main effects + two-way interactions + three-way interactions
    three_way_formula = 'choice ~ group_recoded + fairness_detailed + cost_detailed + group_recoded:fairness_detailed + group_recoded:cost_detailed + fairness_detailed:cost_detailed + group_recoded:fairness_detailed:cost_detailed'
    
    three_way_model = mixedlm(three_way_formula, df_clean, groups=df_clean['participant_id'], re_formula='1')
    three_way_result = three_way_model.fit(method='powell')
    
    print(f"Three-way interaction model converged: {three_way_result.converged}")
    print(f"Log-likelihood: {three_way_result.llf:.3f}")
    print(f"Number of parameters: {len(three_way_result.params)}")
    print()
    
    return three_way_result

def create_main_effects_table(result):
    """Create main effects table"""
    print("=== Main Effects Results ===")
    
    # Get parameter information
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    
    # Calculate 95% confidence intervals
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse
    
    # Create table data
    table_data = []
    
    # Intercept (Human group under fair conditions, 0% cost)
    intercept_beta = params['Intercept']
    intercept_se = bse['Intercept']
    intercept_z = intercept_beta / intercept_se
    intercept_p = pvalues['Intercept']
    intercept_ci_lower = ci_lower['Intercept']
    intercept_ci_upper = ci_upper['Intercept']
    
    table_data.append({
        'Variable': 'Intercept (Human, 15:15, 0% cost)',
        'β': f"{intercept_beta:.3f}",
        'SE': f"{intercept_se:.3f}",
        '95% CI': f"[{intercept_ci_lower:.3f}, {intercept_ci_upper:.3f}]",
        'z': f"{intercept_z:.3f}",
        'p': format_p_value(intercept_p)
    })
    
    # Group main effects (relative to Human group)
    group_mapping = {1: 'gpt3.5', 2: 'o3', 3: 'V3', 4: 'R1'}
    for param in params.index:
        if param.startswith('group_recoded[T.') and not ':' in param:
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
    
    # Fairness main effects (relative to 15:15)
    fairness_mapping = {1: '16:14', 2: '17:13', 3: '18:12', 4: '19:11', 5: '20:10'}
    for param in params.index:
        if param.startswith('fairness_detailed[T.') and not ':' in param:
            fairness_num = int(param.replace('fairness_detailed[T.', '').replace(']', ''))
            fairness_name = fairness_mapping[fairness_num]
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Fairness ({fairness_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Cost main effects (relative to 0%)
    cost_mapping = {1: '10%', 2: '20%', 3: '30%', 4: '40%', 5: '50%', 6: '60%', 7: '70%', 8: '80%', 9: '90%'}
    for param in params.index:
        if param.startswith('cost_detailed[T.') and not ':' in param:
            cost_num = int(param.replace('cost_detailed[T.', '').replace(']', ''))
            cost_name = cost_mapping[cost_num]
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Cost ({cost_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    return df_table

def create_two_way_interactions_table(result):
    """Create two-way interactions table"""
    print("=== Two-way Interaction Effects Results ===")
    
    # Get parameter information
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    
    # Calculate 95% confidence intervals
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse
    
    # Create table data
    table_data = []
    
    # Group × Fairness interaction effects
    group_mapping = {1: 'gpt3.5', 2: 'o3', 3: 'V3', 4: 'R1'}
    fairness_mapping = {1: '16:14', 2: '17:13', 3: '18:12', 4: '19:11', 5: '20:10'}
    
    for param in params.index:
        if 'group_recoded[T.' in param and 'fairness_detailed[T.' in param and 'cost_detailed' not in param:
            # Parse parameter names
            parts = param.split(':')
            group_part = parts[0]
            fairness_part = parts[1]
            
            group_num = int(group_part.replace('group_recoded[T.', '').replace(']', ''))
            fairness_num = int(fairness_part.replace('fairness_detailed[T.', '').replace(']', ''))
            
            group_name = group_mapping[group_num]
            fairness_name = fairness_mapping[fairness_num]
            
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Group×Fairness ({group_name} × {fairness_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Group × Cost interaction effects
    cost_mapping = {1: '10%', 2: '20%', 3: '30%', 4: '40%', 5: '50%', 6: '60%', 7: '70%', 8: '80%', 9: '90%'}
    
    for param in params.index:
        if 'group_recoded[T.' in param and 'cost_detailed[T.' in param and 'fairness_detailed' not in param:
            # Parse parameter names
            parts = param.split(':')
            group_part = parts[0]
            cost_part = parts[1]
            
            group_num = int(group_part.replace('group_recoded[T.', '').replace(']', ''))
            cost_num = int(cost_part.replace('cost_detailed[T.', '').replace(']', ''))
            
            group_name = group_mapping[group_num]
            cost_name = cost_mapping[cost_num]
            
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Group×Cost ({group_name} × {cost_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Fairness × Cost interaction effects
    for param in params.index:
        if 'fairness_detailed[T.' in param and 'cost_detailed[T.' in param and 'group_recoded' not in param:
            # Parse parameter names
            parts = param.split(':')
            fairness_part = parts[0]
            cost_part = parts[1]
            
            fairness_num = int(fairness_part.replace('fairness_detailed[T.', '').replace(']', ''))
            cost_num = int(cost_part.replace('cost_detailed[T.', '').replace(']', ''))
            
            fairness_name = fairness_mapping[fairness_num]
            cost_name = cost_mapping[cost_num]
            
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Fairness×Cost ({fairness_name} × {cost_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    return df_table

def create_three_way_interactions_table(result):
    """Create three-way interactions table"""
    print("=== Three-way Interaction Effects Results ===")
    
    # Get parameter information
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    
    # Calculate 95% confidence intervals
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse
    
    # Create table data
    table_data = []
    
    # Group × Fairness × Cost three-way interaction effects
    group_mapping = {1: 'gpt3.5', 2: 'o3', 3: 'V3', 4: 'R1'}
    fairness_mapping = {1: '16:14', 2: '17:13', 3: '18:12', 4: '19:11', 5: '20:10'}
    cost_mapping = {1: '10%', 2: '20%', 3: '30%', 4: '40%', 5: '50%', 6: '60%', 7: '70%', 8: '80%', 9: '90%'}
    
    for param in params.index:
        if 'group_recoded[T.' in param and 'fairness_detailed[T.' in param and 'cost_detailed[T.' in param:
            # Parse parameter names
            parts = param.split(':')
            group_part = parts[0]
            fairness_part = parts[1]
            cost_part = parts[2]
            
            group_num = int(group_part.replace('group_recoded[T.', '').replace(']', ''))
            fairness_num = int(fairness_part.replace('fairness_detailed[T.', '').replace(']', ''))
            cost_num = int(cost_part.replace('cost_detailed[T.', '').replace(']', ''))
            
            group_name = group_mapping[group_num]
            fairness_name = fairness_mapping[fairness_num]
            cost_name = cost_mapping[cost_num]
            
            beta = params[param]
            se = bse[param]
            z = beta / se
            p = pvalues[param]
            ci_lower_val = ci_lower[param]
            ci_upper_val = ci_upper[param]
            
            table_data.append({
                'Variable': f'Group×Fairness×Cost ({group_name} × {fairness_name} × {cost_name})',
                'β': f"{beta:.3f}",
                'SE': f"{se:.3f}",
                '95% CI': f"[{ci_lower_val:.3f}, {ci_upper_val:.3f}]",
                'z': f"{z:.3f}",
                'p': format_p_value(p)
            })
    
    # Create DataFrame
    df_table = pd.DataFrame(table_data)
    
    return df_table

def create_model_summary_table(result, n_obs, n_groups):
    """Create model summary table"""
    print("=== Model Summary Information ===")
    
    # Calculate ICC
    icc = result.cov_re.iloc[0,0] / (result.cov_re.iloc[0,0] + result.scale)
    
    # Calculate fit indices
    n_params = len(result.params) + 1  # +1 for variance parameter
    llf = result.llf
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + np.log(n_obs) * n_params
    
    # Pseudo R² (McFadden's R²)
    null_llf = n_obs * np.log(0.5)  # For binary classification, null model log-likelihood
    mcfadden_r2 = 1 - (llf / null_llf)
    mcfadden_r2_adj = 1 - ((llf - n_params) / null_llf)
    
    model_info = {
        'Model Information': [
            'Dependent Variable',
            'Number of Participants',
            'Number of Observations',
            'Number of Parameters',
            'Log-likelihood',
            'AIC',
            'BIC',
            'McFadden R²',
            'Adjusted McFadden R²',
            'Convergence Status',
            'Between-group Variance',
            'Residual Variance',
            'Intraclass Correlation Coefficient (ICC)'
        ],
        'Value': [
            'Punishment Decision (choice)',
            f"{n_groups}",
            f"{n_obs}",
            f"{n_params}",
            f"{llf:.2f}",
            f"{aic:.2f}",
            f"{bic:.2f}",
            f"{mcfadden_r2:.3f}",
            f"{mcfadden_r2_adj:.3f}",
            "Yes" if result.converged else "No",
            f"{result.cov_re.iloc[0,0]:.3f}",
            f"{result.scale:.3f}",
            f"{icc:.3f}"
        ]
    }
    
    return pd.DataFrame(model_info)


def main():
    print("=== Three-way Interaction Model Complete Report ===\n")
    
    # Prepare data
    df_clean = prepare_data()
    
    # Fit three-way interaction model
    three_way_result = fit_three_way_model(df_clean)
    
    # Generate various tables
    print("Generating regression results tables...")
    main_effects_table = create_main_effects_table(three_way_result)
    two_way_table = create_two_way_interactions_table(three_way_result)
    three_way_table = create_three_way_interactions_table(three_way_result)
    
    print("Generating model information table...")
    n_groups = df_clean['participant_id'].nunique()
    n_obs = len(df_clean)
    model_summary_table = create_model_summary_table(three_way_result, n_obs, n_groups)
    
    # Save as Excel file
    print("\nSaving tables to Excel file...")
    with pd.ExcelWriter('three_way_interaction_complete_report.xlsx', engine='openpyxl') as writer:
        main_effects_table.to_excel(writer, sheet_name='Main_Effects', index=False)
        two_way_table.to_excel(writer, sheet_name='Two_Way_Interactions', index=False)
        three_way_table.to_excel(writer, sheet_name='Three_Way_Interactions', index=False)
        model_summary_table.to_excel(writer, sheet_name='Model_Summary', index=False)
    
    
    # Display table preview
    print("\n=== Main Effects Table ===")
    print(main_effects_table.to_string(index=False))
    
    print("\n=== Two-way Interaction Effects Table (First 10 rows) ===")
    print(two_way_table.head(10).to_string(index=False))
    
    print("\n=== Three-way Interaction Effects Table (First 10 rows) ===")
    print(three_way_table.head(10).to_string(index=False))
    
    print("\n=== Model Summary Table ===")
    print(model_summary_table.to_string(index=False))
    
    print(f"\n=== Files Saved ===")
    print("1. three_way_interaction_complete_report.xlsx - Complete report in Excel format")

if __name__ == "__main__":
    main()
