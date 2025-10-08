# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')    # use the non-interactive Agg backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 1. Data loading and parsing logic remains the same.
csv_data = """
DATASET,TGN_Deduplication,Node_Feature,Edge_Feature,CPU_GPU_Mem_and_Mail,GPU_CPU_Memory
MOOC,Yes,3275/34359,22247/40407,3304/42642,459/7918
MOOC,No,3281/52275,22239/40397,3262/62688,459/7918
WIKI,Yes,N/A,11189/35345,2966/15172,1256/8075
WIKI,No,N/A,11195/35367,2996/24600,1256/8075
REDDIT,Yes,N/A,38283/69372,5297/43890,2442/7957
REDDIT,No,N/A,38275/69369,5305/62567,2442/7957
LASTFM,Yes,1885/25986,12061/56040,1921/29639,1087/7982
LASTFM,No,1885/68016,12059/56042,1940/70185,1087/7982
"""

df = pd.read_csv(StringIO(csv_data))

def parse_counts(value):
    # Parses a string 'unique/total' into a tuple of two integers.
    # Returns (0, 0) for 'N/A' or invalid formats.
    if pd.isna(value) or value == 'N/A':
        return 0, 0
    try:
        unique, total = map(int, value.split('/'))
        return unique, total
    except (ValueError, AttributeError):
        return 0, 0

# 2. Prepare for plotting a single figure.
# Increased figure width to accommodate all datasets.
fig, ax = plt.subplots(figsize=(22, 10))

datasets = df['DATASET'].unique()
features = df.columns[2:]
feature_labels = [col.replace('_', ' ').replace(' and ', ' & ') for col in features]

# Bar chart settings
n_features = len(features)
n_datasets = len(datasets)
# Define positions for bars and groups
bar_width = 0.8
group_gap = 2  # The space between each dataset group
group_width = n_features * bar_width + (n_features - 1) * 0.1

# 3. Iterate through each dataset to plot its group of bars.
all_tick_positions = []
all_tick_labels = []
for i, dataset_name in enumerate(datasets):
    data_yes = df[(df['DATASET'] == dataset_name) & (df['TGN_Deduplication'] == 'Yes')].iloc[0]
    data_no = df[(df['DATASET'] == dataset_name) & (df['TGN_Deduplication'] == 'No')].iloc[0]

    # Calculate the starting position for the current group
    group_start_pos = i * (n_features + group_gap)

    # Iterate through each feature to plot a merged bar
    for j, feature_name in enumerate(features):
        unique_yes, total_yes = parse_counts(data_yes[feature_name])
        unique_no, total_no = parse_counts(data_no[feature_name])

        # Skip plotting if there is no data for this feature (e.g., Node Feature for WIKI)
        if total_yes == 0 and total_no == 0:
            continue

        # Calculate key values
        avg_unique = (unique_yes + unique_no) / 2
        max_total = max(total_yes, total_no)
        
        bar_pos = group_start_pos + j
        all_tick_positions.append(bar_pos)
        all_tick_labels.append(feature_labels[j])

        # 4. Plot a single white bar with black border as the container.
        ax.bar(bar_pos, max_total, width=bar_width, color='white', edgecolor='black', linewidth=1.2)
        
        # 5. Add background fills and lines.
        # Hatched background for 'Total (w/ Dedup)'
        ax.bar(bar_pos, total_yes, width=bar_width, color='none', edgecolor='black', hatch='////', lw=0)

        # Solid black background for 'Avg. Unique'
        ax.bar(bar_pos, avg_unique, width=bar_width, color='black')

        # Draw a blue line for 'Total (w/o Dedup)' for clarity
        ax.hlines(y=total_no, xmin=bar_pos - bar_width/2, xmax=bar_pos + bar_width/2, 
                  colors='blue', linestyles='-', lw=2)

# 6. Customize the plot.
ax.set_ylabel('Count', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(bottom=0)

# Set up a two-tier x-axis: Major ticks for datasets, minor ticks for features.
# Major Ticks for Datasets
major_tick_positions = [i * (n_features + group_gap) + (n_features - 1) / 2 for i in range(n_datasets)]
ax.set_xticks(major_tick_positions)
ax.set_xticklabels(datasets, fontsize=16, fontweight='bold')
ax.tick_params(axis='x', which='major', length=0, pad=20) # Hide major tick lines, add padding

# Minor Ticks for Features
ax.set_xticks(all_tick_positions, minor=True)
ax.set_xticklabels(all_tick_labels, minor=True, rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='x', which='minor', length=5)


# 7. Create a custom, shared legend for the entire figure.
legend_elements = [Patch(facecolor='black', edgecolor='black', label='Avg. Unique'),
                   Patch(facecolor='white', edgecolor='black', hatch='////', label='Total (w/ Dedup)'),
                   Line2D([0], [0], color='blue', lw=2, label='Total (w/o Dedup)')]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12)


plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for legend and labels
plt.savefig('plot_merged_single_figure_filled.png', dpi=300, bbox_inches='tight')
# plt.show()
