# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 分类映射
category_map = {
    1: 'Emotion',
    2: 'Fairness',
    3: 'Cost',
    4: 'Others'
}

# 设置文件路径
cot_path = '/Users/daiyiqing/Desktop/LLM情感机制/LLM_annotator/比较/Top1000_CoT_Comparison_Claude_vs_Midsummer.xlsx'
human_path = '/Users/daiyiqing/Desktop/LLM情感机制/LLM_annotator/比较/Top180_Comparison_Claude_vs_Midsummer.xlsx'

# 读取数据
cot_df = pd.read_excel(cot_path)
human_df = pd.read_excel(human_path)

# 清洗列名以防大小写或空格问题
cot_df.columns = cot_df.columns.str.strip().str.lower()
human_df.columns = human_df.columns.str.strip().str.lower()

# 添加 Category 列
cot_df['category'] = cot_df['word_sort'].map(category_map)
human_df['category'] = human_df['word_sort'].map(category_map)

# 确保分类顺序一致
categories = ['Emotion', 'Fairness', 'Cost', 'Others']
cot_counts = cot_df['category'].value_counts(normalize=True).reindex(categories).fillna(0)
human_counts = human_df['category'].value_counts(normalize=True).reindex(categories).fillna(0)

# 合并为比例 DataFrame
proportions = pd.DataFrame({
    "DeepSeek-R1's CoT": cot_counts,
    "Human Reasoning": human_counts
})

# %%
# 画图设置
fig, ax = plt.subplots(figsize=(7, 3))

# 坐标轴放在上面
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# 颜色设置
colors = ['#f26c6c','#2171b5', '#6baed6', '#c6dbef']   #'#9ecae1

# 堆叠柱子
bar_height = 0.2
y_positions = [0.7, 0.45]  # y=1: CoT, y=0: Human
left_cot = 0
left_human = 0

for i, category in enumerate(categories):  # 用固定顺序遍历
    cot_val = proportions.loc[category, "DeepSeek-R1's CoT"]
    human_val = proportions.loc[category, "Human Reasoning"]

    ax.barh(y_positions[0], cot_val, left=left_cot, height=bar_height, color=colors[i])
    ax.barh(y_positions[1], human_val, left=left_human, height=bar_height, color=colors[i])

    left_cot += cot_val
    left_human += human_val

# 设置 y 轴标签
ax.set_yticks(y_positions)
ax.set_yticklabels(["DeepSeek-R1's CoT", "Humans' Reasoning"])

# x轴设置
ax.set_xlim(0, 1)
ax.set_xlabel("Proportion", fontsize=13, labelpad=10)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# 图例设置
legend_elements = [
    Patch(facecolor=colors[0], label='Emotion'), 
    Patch(facecolor=colors[1], label='Fairness'),
    Patch(facecolor=colors[2], label='Cost'),
    Patch(facecolor=colors[3], label='Others')
]

ax.legend(
    handles=legend_elements,
    title="Word Category",
    loc='lower center',
    bbox_to_anchor=(0.5, -0.6),
    ncol=4,
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    handleheight=1.5,       # 控制颜色块高度
    handlelength=2.0,       # 控制颜色块长度（宽度）
    borderaxespad=0.5       # 控制图例和图间距（可选）
)


# 去除边框
ax.spines[['right', 'left', 'bottom']].set_visible(False)

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 确保分类顺序一致
categories = ['Emotion', 'Fairness', 'Cost', 'Others']

# raw counts
cot_counts_raw = cot_df['category'].value_counts().reindex(categories).fillna(0)
human_counts_raw = human_df['category'].value_counts().reindex(categories).fillna(0)

# 构建列联表
contingency = pd.DataFrame({
    "DeepSeek-R1's CoT": cot_counts_raw,
    "Human Reasoning": human_counts_raw
}, index=categories)

print("Contingency Table:\n", contingency, "\n")

# 卡方检验
chi2, p, dof, expected = chi2_contingency(contingency.T)

# 样本总量
N = contingency.to_numpy().sum()

# Cramér’s V
cramers_v = np.sqrt(chi2 / (N * (min(contingency.shape) - 1)))

print(f"Chi-square test: χ²({dof}, N={N}) = {chi2:.2f}, p = {p:.3e}")
print(f"Cramér's V = {cramers_v:.3f}\n")

# 计算标准化残差
residuals = (contingency.T - expected) / np.sqrt(expected)
residuals = residuals.T  # 转回原始顺序 (categories × groups)

print("Standardized residuals (>|2| = major contributor):\n")
print(residuals.round(2))

# %%
