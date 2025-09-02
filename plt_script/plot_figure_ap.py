

import matplotlib
matplotlib.use('Agg')    # use the non-interactive Agg backend
import pandas as pd
import matplotlib.pyplot as plt

# Prepare the new Average AP data
ap_data = {
    'Random Neg Sample': [0.754989, 0.834034, 0.943569, 0.989180, 0.985128, 0.976484, 0.795468, 0.824644],
    'Version 1':         [0.778015, 0.833992, 0.918679, 0.990132, 0.986003, 0.974421, 0.807966, 0.818593],
    'Version 2':         [0.747783, 0.836691, 0.945641, 0.988672, 0.952634, 0.946480, 0.742347, 0.779391],
}

index = [
    'LASTFM_APAN', 'LASTFM_TGN',
    'MOOC_APAN',   'MOOC_TGN',
    'REDDIT_APAN','REDDIT_TGN',
    'WIKI_APAN',   'WIKI_TGN'
]

df_ap = pd.DataFrame(ap_data, index=index)

# Plotting
ax = df_ap.plot(kind='bar', rot=45)
ax.set_ylabel('Average AP')
ax.set_xlabel('Dataset & Model')
ax.set_title('Average AP Comparison Across Versions (New Data)')
plt.tight_layout()

# Save to file for local plotting
output_path = 'avg_ap_comparison_new.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
