#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建合并的调节效应图表：7个特质的choice均值比较
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

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

def create_combined_plot(df):
    """创建合并的调节效应图表"""
    print("开始创建合并图表...")
    
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
        
        # 计算每个group和trait水平的choice均值
        mean_data = df.groupby([var, 'group'])['choice'].mean().reset_index()
        
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
                        values.append(group_data['choice'].iloc[0])
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
        #ax.set_xlabel('Group', fontsize=10)
        ax.set_ylabel('Punishment Probability', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8, loc='upper right')
        
        # 添加背景色区分human和LLM
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red', label='Human')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue', label='LLMs')
    
    # 隐藏第8个子图（因为只有7个变量）
    axes[7].set_visible(False)
    
    # 设置总标题
    #fig.suptitle('Moderating Role of Participant Group in Trait-Behavior Associations\nChoice Means by Trait Level and Group', 
                 #fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.3, wspace=0.3)
    
    # 保存图表
    plt.savefig('combined_choice_moderation_effects.png', dpi=300, bbox_inches='tight')
    print("合并图表已保存: combined_choice_moderation_effects.png")
    
    plt.show()

def create_individual_plots(df):
    """创建单独的图表（可选）"""
    print("\n创建单独的图表...")
    
    group_order, group_labels = setup_group_order()
    
    grouping_vars = {
        'gender': 'Gender',
        'SVO': 'Social Value Orientation', 
        'CESD_2g': 'Depressive Symptoms',
        'AQ_2g': 'Autism Spectrum Traits',
        'age_2g': 'Age',
        'JSI_2g': 'Joint Attention',
        'ERS_2g': 'Emotional Reactivity'
    }
    
    for var, label in grouping_vars.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 计算均值数据
        mean_data = df.groupby([var, 'group'])['choice'].mean().reset_index()
        mean_data['group'] = pd.Categorical(mean_data['group'], categories=group_order, ordered=True)
        mean_data = mean_data.sort_values('group')
        
        # 获取trait水平
        trait_levels = sorted(mean_data[var].unique())
        if pd.isna(trait_levels).any():
            trait_levels = [x for x in trait_levels if pd.notna(x)]
        
        # 创建条形图
        x_pos = np.arange(len(group_order))
        width = 0.35
        
        for j, level in enumerate(trait_levels):
            if pd.notna(level):
                level_data = mean_data[mean_data[var] == level]
                
                values = []
                for group in group_order:
                    group_data = level_data[level_data['group'] == group]
                    if len(group_data) > 0:
                        values.append(group_data['choice'].iloc[0])
                    else:
                        values.append(0)
                
                offset = (j - 0.5) * width
                bars = ax.bar(x_pos + offset, values, width, 
                             label=str(level), 
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 设置图表属性
        ax.set_title(f'{label} × Group Interaction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Choice Mean', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # 添加背景色
        ax.axvspan(-0.5, 0.5, alpha=0.1, color='red')
        ax.axvspan(0.5, 4.5, alpha=0.1, color='blue')
        
        plt.tight_layout()
        filename = f'individual_{var}_moderation.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"单独图表已保存: {filename}")
        plt.show()

def main():
    """主函数"""
    print("开始创建调节效应图表...")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 创建合并图表
    create_combined_plot(df)
    
    # 可选：创建单独图表
    # create_individual_plots(df)
    
    print("\n图表创建完成!")

if __name__ == "__main__":
    main()
