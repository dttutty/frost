
import matplotlib
matplotlib.use('Agg')    # use the non-interactive Agg backend

import pandas as pd
import matplotlib.pyplot as plt

# 准备新的 Average AUC 数据
auc_data = {
    'Random Neg Sample': [0.823069, 0.855658, 0.950478, 0.991733, 0.989830, 0.981640, 0.814717, 0.803707],
    'Version 1':         [0.833712, 0.856796, 0.931074, 0.992868, 0.989966, 0.979884, 0.827656, 0.798005],
    'Version 2':         [0.821606, 0.860511, 0.958492, 0.991312, 0.973418, 0.954580, 0.772789, 0.760288],
}

index = [
    'LASTFM_APAN', 'LASTFM_TGN',
    'MOOC_APAN',   'MOOC_TGN',
    'REDDIT_APAN','REDDIT_TGN',
    'WIKI_APAN',   'WIKI_TGN'
]

# 构建 DataFrame
df_auc = pd.DataFrame(auc_data, index=index)

# 绘制柱状图
ax = df_auc.plot(kind='bar', rot=45)
ax.set_ylabel('Average AUC')
ax.set_xlabel('Dataset & Model')
ax.set_title('Average AUC Comparison Across Versions (New Data)')
plt.tight_layout()

# 保存图表以供本地查看
output_path = 'avg_auc_comparison_new.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
