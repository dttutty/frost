
import matplotlib
matplotlib.use('Agg')    # use the non-interactive Agg backend
import pandas as pd
import matplotlib.pyplot as plt

# 准备新的 Avg Total Time 数据
time_data = {
    'Random Neg Sample': [4.96, 6.77, 1.56, 3.29, 4.85, 5.20, 1.16, 2.64],
    'Version 1':         [4.83, 6.46, 1.48, 3.36, 5.02, 5.24, 1.19, 2.65],
    'Version 2':         [4.92, 7.06, 1.50, 3.30, 4.96, 5.31, 1.18, 2.70],
}

index = [
    'LASTFM_APAN', 'LASTFM_TGN',
    'MOOC_APAN',   'MOOC_TGN',
    'REDDIT_APAN','REDDIT_TGN',
    'WIKI_APAN',   'WIKI_TGN'
]

df_time = pd.DataFrame(time_data, index=index)

# 绘制柱状图
ax = df_time.plot(kind='bar', rot=45)
ax.set_ylabel('Avg Total Time (s)')
ax.set_xlabel('Dataset & Model')
ax.set_title('Avg Total Time Comparison Across Versions (New Data)')
plt.tight_layout()

# 保存图表以供本地查看
output_path = 'avg_time_comparison_new.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

