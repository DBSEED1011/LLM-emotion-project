#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 独立样本t检验分析
分析情绪&对照数据中stage=2的数据，比较group=1(emo)和group=0(math)的差异
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_filter_data():
    """加载数据并筛选stage=2的数据"""
    print("正在加载数据...")
    df = pd.read_excel('Study2b_emo_vs_math.xlsx', sheet_name='bytrial')
    print(f"原始数据形状: {df.shape}")
    
    # 筛选stage=2的数据
    stage2_data = df[df['stage'] == 2].copy()
    print(f"Stage=2数据形状: {stage2_data.shape}")
    
    # 查看group分布
    print("\nGroup分布:")
    print(stage2_data['group'].value_counts())
    
    return stage2_data

def perform_t_tests(data):
    """对各个变量进行独立样本t检验"""
    print("\n" + "="*60)
    print("独立样本t检验结果 (Group 1: emo vs Group 0: math)")
    print("="*60)
    
    # 需要分析的数值变量
    numeric_vars = ['fairness', 'choice', 'current_valence', 'current_arousal', 
                   'predicted_valence', 'predicted_arousal', 'actual_valence', 'actual_arousal']
    
    results = []
    
    for var in numeric_vars:
        if var in data.columns:
            # 获取两个组的数据
            group1_data = data[data['group'] == 1][var].dropna()
            group0_data = data[data['group'] == 0][var].dropna()
            
            if len(group1_data) > 0 and len(group0_data) > 0:
                # 进行独立样本t检验
                t_stat, p_value = stats.ttest_ind(group1_data, group0_data)
                
                # 计算描述性统计
                group1_mean = group1_data.mean()
                group0_mean = group0_data.mean()
                group1_std = group1_data.std()
                group0_std = group0_data.std()
                
                # 计算效应量 (Cohen's d)
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_std**2 + 
                                    (len(group0_data) - 1) * group0_std**2) / 
                                   (len(group1_data) + len(group0_data) - 2))
                cohens_d = (group1_mean - group0_mean) / pooled_std if pooled_std > 0 else 0
                
                result = {
                    'Variable': var,
                    'Group1_n': len(group1_data),
                    'Group0_n': len(group0_data),
                    'Group1_mean': group1_mean,
                    'Group0_mean': group0_mean,
                    'Group1_std': group1_std,
                    'Group0_std': group0_std,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
                results.append(result)
                
                # 打印结果
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"\n{var}:")
                print(f"  Group 1 (emo):  n={len(group1_data):3d}, M={group1_mean:.3f}, SD={group1_std:.3f}")
                print(f"  Group 0 (math): n={len(group0_data):3d}, M={group0_mean:.3f}, SD={group0_std:.3f}")
                print(f"  t({len(group1_data) + len(group0_data) - 2}) = {t_stat:.3f}, p = {p_value:.4f} {significance}")
                print(f"  Cohen's d = {cohens_d:.3f}")
    
    return pd.DataFrame(results)



def main():
    """主函数"""
    print("Stage 2 独立样本t检验分析")
    print("="*50)
    
    # 加载和筛选数据
    data = load_and_filter_data()
    
    # 进行t检验
    results_df = perform_t_tests(data)
    
    
    
    # 保存结果到CSV
    results_df.to_csv('emo_math_t_test_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: emo_math_t_test_results.csv")
    
    # 总结
    print("\n" + "="*60)
    print("分析总结")
    print("="*60)
    significant_count = results_df['significant'].sum()
    total_tests = len(results_df)
    print(f"总共进行了 {total_tests} 个变量的t检验")
    print(f"其中 {significant_count} 个变量显示出显著差异 (p < 0.05)")
    
    if significant_count > 0:
        print("\n显著差异的变量:")
        for _, row in results_df[results_df['significant']].iterrows():
            print(f"  - {row['Variable']}: p = {row['p_value']:.4f}, Cohen's d = {row['cohens_d']:.3f}")

if __name__ == "__main__":
    main()

