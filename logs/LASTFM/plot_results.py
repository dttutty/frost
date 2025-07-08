import matplotlib
matplotlib.use('Agg')      # use the non-interactive Agg backend
import os
import re
import matplotlib.pyplot as plt
import argparse

def parse_log_files(log_directory, prefix):
    """
    Parses log files in a given directory with a specific prefix.

    Args:
        log_directory (str): The path to the directory containing the log files.
        prefix (str): The prefix of the log files to look for (e.g., 'TGN_', 'APAN_').

    Returns:
        tuple: A tuple containing three lists (batch_sizes, aps, aucs).
    """
    try:
        # We need to find files that start with the prefix and end with .log
        # The batch size is between the prefix and the extension.
        # Example: TGN_500.log -> prefix='TGN_', batch_size='500', extension='.log'
        log_files = [f for f in os.listdir(log_directory) if f.startswith(prefix) and f.endswith('.log')]
    except FileNotFoundError:
        print(f"Error: Directory not found at '{log_directory}'")
        return [], [], []

    results = []
    
    # This pattern is now more general to match different prefixes
    file_pattern = re.compile(rf'^{re.escape(prefix)}(\d+)\.log$')
    data_pattern = re.compile(r"test AP:(\d+\.\d+)\s+test AUC:(\d+\.\d+)")

    for filename in log_files:
        match = file_pattern.match(filename)
        if not match:
            continue
        
        batch_size = int(match.group(1))
        filepath = os.path.join(log_directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                data_match = data_pattern.search(content)
                if data_match:
                    ap = float(data_match.group(1))
                    auc = float(data_match.group(2))
                    results.append({'batch_size': batch_size, 'ap': ap, 'auc': auc})
                else:
                    print(f"Warning: Could not find AP/AUC data in {filename}")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    results.sort(key=lambda x: x['batch_size'])
    
    batch_sizes = [res['batch_size'] for res in results]
    aps = [res['ap'] for res in results]
    aucs = [res['auc'] for res in results]
    
    return batch_sizes, aps, aucs

def plot_multiple_metrics(all_results):
    """
    Plots metrics from multiple experiment sources on a single figure.

    Args:
        all_results (dict): A dictionary where keys are custom labels
                            and values are dictionaries containing parsed data.
    """
    if not all_results:
        print("No data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    linestyles = ['-', '--', '-.', ':']

    for i, (label, data) in enumerate(all_results.items()):
        batch_sizes = data['batch_sizes']
        aps = data['aps']
        aucs = data['aucs']
        
        if not batch_sizes:
            continue

        color = plt.cm.viridis(i / max(1, len(all_results) - 1)) if len(all_results) > 1 else 'blue'

        ax.plot(batch_sizes, aps, marker=markers[i % len(markers)], linestyle=linestyles[0], label=f'{label} - Test AP', color=color)
        ax.plot(batch_sizes, aucs, marker=markers[i % len(markers)], linestyle=linestyles[1], label=f'{label} - Test AUC', color=color, alpha=0.7)

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Score')
    ax.set_title('Test AP and AUC vs. Batch Size for Multiple Experiments')
    ax.legend(fontsize='small')
    
    plt.tight_layout()
    
    plt.savefig("comparison_metrics.png", dpi=300)
    print("\nComparison plot saved to comparison_metrics.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse and plot experiment results from multiple sources.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # --- 主要修改在这里：使用新的 --source 参数 ---
    parser.add_argument(
        '--source', 
        nargs=3, 
        action='append', 
        required=True, 
        metavar=('PREFIX', 'DIRECTORY', 'LABEL'),
        help="Specify a data source. This option can be used multiple times.\n"
             "  PREFIX:    The prefix of the log files (e.g., 'TGN_').\n"
             "  DIRECTORY: The path to the log directory.\n"
             "  LABEL:     A custom label for the plot legend (e.g., 'TGN-WIKI')."
    )
    
    args = parser.parse_args()

    all_results = {}
    print("Parsing log files...")

    # --- 主要修改在这里：循环处理所有 --source ---
    if args.source:
        for prefix, log_dir, label in args.source:
            print(f"\n--- Processing Source: LABEL='{label}' ---")
            print(f"  - Directory: {log_dir}")
            print(f"  - Prefix: {prefix}")
            
            batch_sizes, test_aps, test_aucs = parse_log_files(log_dir, prefix)
            
            if batch_sizes:
                all_results[label] = {
                    'batch_sizes': batch_sizes,
                    'aps': test_aps,
                    'aucs': test_aucs
                }
                for i in range(len(batch_sizes)):
                    print(f"    Batch Size: {batch_sizes[i]:<5} -> AP: {test_aps[i]:.6f}, AUC: {test_aucs[i]:.6f}")
    
    plot_multiple_metrics(all_results)
