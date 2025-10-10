#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析group如何调节不同特质对choice行为的影响
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    try:
        df = pd.read_excel('data_with_grouping_variables.xlsx')
        print(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def basic_descriptive_stats(df):
    """基本描述性统计"""
    print("\n" + "="*60)
    print("基本描述性统计")
    print("="*60)
    
    # Choice的基本统计
    print("\nChoice变量统计:")
    print(f"均值: {df['choice'].mean():.3f}")
    print(f"标准差: {df['choice'].std():.3f}")
    print(f"最小值: {df['choice'].min()}")
    print(f"最大值: {df['choice'].max()}")
    
    # Group分布
    print(f"\nGroup分布:")
    group_counts = df['group'].value_counts()
    for group, count in group_counts.items():
        print(f"{group}: {count} ({count/len(df)*100:.1f}%)")
    
    # 各分组变量的分布
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    for var in grouping_vars:
        if var in df.columns:
            print(f"\n{var}分布:")
            value_counts = df[var].value_counts()
            for value, count in value_counts.items():
                if pd.notna(value):
                    print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

def analyze_moderation_effect(df, grouping_var, outcome_var='choice'):
    """分析调节效应"""
    print(f"\n" + "="*60)
    print(f"调节效应分析: {grouping_var} × group → {outcome_var}")
    print("="*60)
    
    # 检查变量是否存在
    if grouping_var not in df.columns:
        print(f"警告: 变量 '{grouping_var}' 不存在")
        return None
    
    # 移除缺失值
    analysis_df = df[[outcome_var, 'group', grouping_var]].dropna()
    print(f"分析样本量: {len(analysis_df)}")
    
    # 描述性统计
    print(f"\n{grouping_var}各水平下的choice均值:")
    desc_stats = analysis_df.groupby([grouping_var, 'group'])[outcome_var].agg(['mean', 'std', 'count'])
    print(desc_stats)
    
    # 2×5 ANOVA (分组变量 × group)
    try:
        # 构建公式
        formula = f"{outcome_var} ~ C({grouping_var}) * C(group)"
        model = ols(formula, data=analysis_df).fit()
        
        # ANOVA表
        anova_table = anova_lm(model, typ=2)
        print(f"\nANOVA结果:")
        print(anova_table)
        
        # 提取关键统计量
        main_effect_grouping = anova_table.loc[f'C({grouping_var})', 'PR(>F)']
        main_effect_group = anova_table.loc['C(group)', 'PR(>F)']
        interaction_effect = anova_table.loc[f'C({grouping_var}):C(group)', 'PR(>F)']
        
        print(f"\n关键结果:")
        print(f"{grouping_var}主效应: p = {main_effect_grouping:.4f}")
        print(f"Group主效应: p = {main_effect_group:.4f}")
        print(f"交互效应 ({grouping_var} × group): p = {interaction_effect:.4f}")
        
        # 判断调节效应
        if interaction_effect < 0.05:
            print(f"✓ 发现显著调节效应! {grouping_var}调节了group对{outcome_var}的影响")
            
            # 简单效应分析
            print(f"\n简单效应分析:")
            simple_effects = analyze_simple_effects(analysis_df, grouping_var, outcome_var)
            return {
                'significant_moderation': True,
                'interaction_p': interaction_effect,
                'simple_effects': simple_effects,
                'descriptive_stats': desc_stats
            }
        else:
            print(f"✗ 未发现显著调节效应 (p = {interaction_effect:.4f})")
            return {
                'significant_moderation': False,
                'interaction_p': interaction_effect,
                'descriptive_stats': desc_stats
            }
            
    except Exception as e:
        print(f"ANOVA分析出错: {e}")
        return None

def analyze_simple_effects(df, grouping_var, outcome_var):
    """简单效应分析"""
    simple_effects = {}
    
    # 在每个分组变量水平下，比较不同group的差异
    for level in df[grouping_var].unique():
        if pd.notna(level):
            subset = df[df[grouping_var] == level]
            if len(subset) > 0:
                # 单因素ANOVA比较不同group
                try:
                    groups = [subset[subset['group'] == g][outcome_var].values 
                             for g in subset['group'].unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    simple_effects[level] = {
                        'f_stat': f_stat,
                        'p_value': p_value,
                        'group_means': subset.groupby('group')[outcome_var].mean().to_dict()
                    }
                    
                    print(f"  {grouping_var}={level}: F={f_stat:.3f}, p={p_value:.4f}")
                    for group, mean_val in simple_effects[level]['group_means'].items():
                        print(f"    {group}: M={mean_val:.3f}")
                        
                except Exception as e:
                    print(f"  {grouping_var}={level}: 分析失败 - {e}")
    
    return simple_effects

def run_all_moderation_analyses(df):
    """运行所有调节效应分析"""
    print("\n" + "="*80)
    print("开始所有调节效应分析")
    print("="*80)
    
    # 所有分组变量
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    
    results = {}
    
    for var in grouping_vars:
        if var in df.columns:
            # 分析调节效应
            result = analyze_moderation_effect(df, var)
            if result:
                results[var] = result
            
            print("\n" + "-"*60)
    
    return results

def generate_summary_report(results):
    """生成总结报告"""
    print("\n" + "="*80)
    print("调节效应分析总结报告")
    print("="*80)
    
    significant_moderations = []
    non_significant_moderations = []
    
    for var, result in results.items():
        if result['significant_moderation']:
            significant_moderations.append((var, result['interaction_p']))
        else:
            non_significant_moderations.append((var, result['interaction_p']))
    
    print(f"\n发现显著调节效应的变量 ({len(significant_moderations)}个):")
    for var, p_val in significant_moderations:
        print(f"  ✓ {var}: p = {p_val:.4f}")
    
    print(f"\n未发现显著调节效应的变量 ({len(non_significant_moderations)}个):")
    for var, p_val in non_significant_moderations:
        print(f"  ✗ {var}: p = {p_val:.4f}")
    
    # 保存详细结果到Excel文件
    try:
        with pd.ExcelWriter('choice_moderation_analysis_summary.xlsx', engine='openpyxl') as writer:
            
            # 1. 基本总结表
            summary_df = pd.DataFrame([
                {
                    'grouping_variable': var,
                    'significant_moderation': result['significant_moderation'],
                    'interaction_p_value': result['interaction_p'],
                    'interaction_significant': 'Yes' if result['interaction_p'] < 0.05 else 'No'
                }
                for var, result in results.items()
            ])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 2. 描述性统计表
            desc_stats_list = []
            for var, result in results.items():
                desc_stats = result['descriptive_stats']
                for (trait_level, group), stats_row in desc_stats.iterrows():
                    desc_stats_list.append({
                        'grouping_variable': var,
                        'trait_level': trait_level,
                        'group': group,
                        'mean': stats_row['mean'],
                        'std': stats_row['std'],
                        'count': stats_row['count']
                    })
            
            desc_stats_df = pd.DataFrame(desc_stats_list)
            desc_stats_df.to_excel(writer, sheet_name='Descriptive_Stats', index=False)
            
            # 3. 简单效应分析表（仅显著调节效应）
            simple_effects_list = []
            for var, result in results.items():
                if result['significant_moderation'] and 'simple_effects' in result:
                    simple_effects = result['simple_effects']
                    for trait_level, effect_data in simple_effects.items():
                        simple_effects_list.append({
                            'grouping_variable': var,
                            'trait_level': trait_level,
                            'f_statistic': effect_data['f_stat'],
                            'p_value': effect_data['p_value'],
                            'significant': 'Yes' if effect_data['p_value'] < 0.05 else 'No'
                        })
                        
                        # 添加每个group的均值
                        for group, mean_val in effect_data['group_means'].items():
                            simple_effects_list.append({
                                'grouping_variable': var,
                                'trait_level': f"{trait_level}_group_{group}",
                                'f_statistic': '',
                                'p_value': '',
                                'significant': '',
                                'group_mean': mean_val
                            })
            
            if simple_effects_list:
                simple_effects_df = pd.DataFrame(simple_effects_list)
                simple_effects_df.to_excel(writer, sheet_name='Simple_Effects', index=False)
            
            # 4. 详细结果表（包含所有统计信息）
            detailed_results_list = []
            for var, result in results.items():
                detailed_results_list.append({
                    'grouping_variable': var,
                    'significant_moderation': result['significant_moderation'],
                    'interaction_p_value': result['interaction_p'],
                    'interaction_significant': 'Yes' if result['interaction_p'] < 0.05 else 'No',
                    'num_trait_levels': len(result['descriptive_stats'].index.get_level_values(0).unique()),
                    'total_sample_size': result['descriptive_stats']['count'].sum(),
                    'has_simple_effects': 'Yes' if 'simple_effects' in result else 'No'
                })
            
            detailed_df = pd.DataFrame(detailed_results_list)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        print(f"\n详细总结报告已保存到: choice_moderation_analysis_summary.xlsx")
        print("包含以下工作表:")
        print("  - Summary: 基本调节效应总结")
        print("  - Descriptive_Stats: 描述性统计")
        print("  - Simple_Effects: 简单效应分析结果")
        print("  - Detailed_Results: 详细分析结果")
        
    except Exception as e:
        print(f"保存总结报告失败: {e}")

def main():
    """主函数"""
    print("开始调节效应分析...")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 基本描述性统计
    basic_descriptive_stats(df)
    
    # 运行所有调节效应分析
    results = run_all_moderation_analyses(df)
    
    # 生成总结报告
    generate_summary_report(results)
    
    print("\n调节效应分析完成!")

if __name__ == "__main__":
    main()

