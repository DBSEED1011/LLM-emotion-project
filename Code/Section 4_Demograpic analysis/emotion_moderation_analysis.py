#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析group如何调节不同特质对情绪变量的影响
对6个情绪变量进行调节效应分析和绘图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    try:
        df = pd.read_excel('data_with_grouping_variables.xlsx')
        print(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def setup_group_order():
    """设置group顺序和标签"""
    group_mapping = {
        'human': 'Human',
        'gpt3.5': 'GPT-3.5', 
        'o3': 'o3-mini',
        'V3': 'DeepSeek-V3',
        'R1': 'DeepSeek-R1'
    }
    
    group_order = ['human', 'gpt3.5', 'o3', 'V3', 'R1']
    group_labels = [group_mapping[g] for g in group_order]
    
    return group_order, group_labels

def analyze_moderation_effect(df, grouping_var, outcome_var):
    """分析调节效应"""
    print(f"\n" + "="*60)
    print(f"调节效应分析: {grouping_var} × group → {outcome_var}")
    print("="*60)
    
    # 检查变量是否存在
    if grouping_var not in df.columns or outcome_var not in df.columns:
        print(f"警告: 变量 '{grouping_var}' 或 '{outcome_var}' 不存在")
        return None
    
    # 移除缺失值
    analysis_df = df[[outcome_var, 'group', grouping_var]].dropna()
    print(f"分析样本量: {len(analysis_df)}")
    
    # 描述性统计
    print(f"\n{grouping_var}各水平下的{outcome_var}均值:")
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
            return {
                'significant_moderation': True,
                'interaction_p': interaction_effect,
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

def create_combined_plot(df, outcome_var, outcome_label):
    """创建合并的调节效应图表"""
    print(f"开始创建{outcome_var}的合并图表...")
    
    # 设置group顺序
    group_order, group_labels = setup_group_order()
    
    # 定义分组变量和标签
    grouping_vars = {
        'gender': 'Gender',
        'SVO': 'Social Value Orientation', 
        'CESD_2g': 'Depression Tendency',
        'AQ_2g': 'Autism Spectrum Quotient',
        'age_2g': 'Age',
        'JSI_2g': 'Justice Sensitivity',
        'ERS_2g': 'Emotional Reactivity'
    }
    
    # 创建大图
    fig, axes = plt.subplots(2, 4, figsize=(16, 11))
    axes = axes.flatten()
    
    # 设置颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (var, label) in enumerate(grouping_vars.items()):
        if i >= 7:  # 只处理前7个变量
            break
            
        ax = axes[i]
        
        # 计算每个group和trait水平的outcome均值
        mean_data = df.groupby([var, 'group'])[outcome_var].mean().reset_index()
        
        # 重新排序group
        mean_data['group'] = pd.Categorical(mean_data['group'], categories=group_order, ordered=True)
        mean_data = mean_data.sort_values('group')
        
        # 获取trait水平
        trait_levels = sorted(mean_data[var].unique())
        if pd.isna(trait_levels).any():
            trait_levels = [x for x in trait_levels if pd.notna(x)]
        
        # 为每个trait水平创建条形图
        x_pos = np.arange(len(group_order))
        width = 0.35
        
        for j, level in enumerate(trait_levels):
            if pd.notna(level):
                level_data = mean_data[mean_data[var] == level]
                
                # 确保所有group都有数据
                values = []
                for group in group_order:
                    group_data = level_data[level_data['group'] == group]
                    if len(group_data) > 0:
                        values.append(group_data[outcome_var].iloc[0])
                    else:
                        values.append(0)
                
                # 绘制条形图
                offset = (j - 0.5) * width
                bars = ax.bar(x_pos + offset, values, width, 
                             label=str(level), 
                             color=colors[j % len(colors)],
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=6)
        
        # 设置图表属性
        ax.set_title(f'{label}', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel(f'{outcome_label}', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        
        # 根据outcome变量设置y轴范围
        if 'AA_' in outcome_var or 'AC_' in outcome_var:
            ax.set_ylim(-100, 100)  # AA和AC变量范围-100到+100
        elif 'EmoFDBK_' in outcome_var:
            ax.set_ylim(-200, 200)  # EmoFDBK变量范围-200到+200
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper right')
        
        # 添加背景色区分human和LLM
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='Human')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue', label='LLMs')
    
    # 隐藏第8个子图（因为只有7个变量）
    axes[7].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # 保存图表
    filename = f'combined_moderation_{outcome_var}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"合并图表已保存: {filename}")
    
    plt.show()

def run_emotion_analysis(df):
    """运行所有情绪变量的分析"""
    print("\n" + "="*80)
    print("开始所有情绪变量的调节效应分析")
    print("="*80)
    
    # 定义情绪变量和标签
    emotion_vars = {
        'AA_valence': 'After-allocation valence',
        'AA_arousal': 'After-allocation arousal',
        'AC_valence': 'After-choice valence', 
        'AC_arousal': 'After-choice arousal',
        'EmoFDBK_valence': 'Emotional outcome valence',
        'EmoFDBK_arousal': 'Emotional outcome arousal'
    }
    
    # 所有分组变量
    grouping_vars = ['gender', 'SVO', 'CESD_2g', 'AQ_2g', 'age_2g', 'JSI_2g', 'ERS_2g']
    
    all_results = {}
    
    for emotion_var, emotion_label in emotion_vars.items():
        print(f"\n{'='*20} 分析 {emotion_var} {'='*20}")
        
        emotion_results = {}
        
        # 对每个分组变量进行调节效应分析
        for var in grouping_vars:
            if var in df.columns:
                result = analyze_moderation_effect(df, var, emotion_var)
                if result:
                    emotion_results[var] = result
        
        all_results[emotion_var] = emotion_results
        
        # 创建合并图表
        create_combined_plot(df, emotion_var, emotion_label)
        
        print(f"\n{emotion_var} 分析完成!")
        print("-"*60)
    
    return all_results

def generate_emotion_summary_report(all_results):
    """生成情绪变量分析总结报告"""
    print("\n" + "="*80)
    print("情绪变量调节效应分析总结报告")
    print("="*80)
    
    # 创建总结DataFrame
    summary_data = []
    
    for emotion_var, emotion_results in all_results.items():
        for grouping_var, result in emotion_results.items():
            summary_data.append({
                'emotion_variable': emotion_var,
                'grouping_variable': grouping_var,
                'significant_moderation': result['significant_moderation'],
                'interaction_p_value': result['interaction_p']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 按情绪变量分组统计
    print("\n各情绪变量的调节效应统计:")
    for emotion_var in summary_df['emotion_variable'].unique():
        emotion_data = summary_df[summary_df['emotion_variable'] == emotion_var]
        significant_count = emotion_data['significant_moderation'].sum()
        total_count = len(emotion_data)
        print(f"{emotion_var}: {significant_count}/{total_count} 个特质变量发现显著调节效应")
    
    # 按分组变量统计
    print("\n各特质变量的调节效应统计:")
    for grouping_var in summary_df['grouping_variable'].unique():
        grouping_data = summary_df[summary_df['grouping_variable'] == grouping_var]
        significant_count = grouping_data['significant_moderation'].sum()
        total_count = len(grouping_data)
        print(f"{grouping_var}: {significant_count}/{total_count} 个情绪变量发现显著调节效应")
    
    # 保存总结报告
    try:
        summary_df.to_excel('emotion_moderation_analysis_summary.xlsx', index=False)
        print(f"\n总结报告已保存到: emotion_moderation_analysis_summary.xlsx")
    except Exception as e:
        print(f"保存总结报告失败: {e}")

def main():
    """主函数"""
    print("开始情绪变量调节效应分析...")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 运行所有情绪变量的分析
    all_results = run_emotion_analysis(df)
    
    # 生成总结报告
    generate_emotion_summary_report(all_results)
    
    print("\n情绪变量调节效应分析完成!")

if __name__ == "__main__":
    main()
