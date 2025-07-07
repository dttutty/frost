import matplotlib
matplotlib.use('Agg')    # use the non-interactive Agg backend
import os
import re
import matplotlib.pyplot as plt

def parse_log_files(log_directory):
    """
    Parses log files in a given directory to extract batch size, AP, and AUC.

    Args:
        log_directory (str): The path to the directory containing the log files.

    Returns:
        tuple: A tuple containing three lists (batch_sizes, aps, aucs).
    """
    log_files = [f for f in os.listdir(log_directory) if f.startswith('TGN_') and f.endswith('.log')]
    
    results = []

    # Regular expression to find AP and AUC values
    # It looks for a line containing "test ap:" and "test auc:"
    # and captures the floating point numbers that follow.
    # e.g., "test ap:0.959265   test auc:0.969155"
    data_pattern = re.compile(r"test ap:(\d+\.\d+)\s+test auc:(\d+\.\d+)")

    for filename in log_files:
        # Extract batch size from the filename, e.g., "TGN_128.log" -> 128
        match = re.search(r'TGN_(\d+).log', filename)
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

    # Sort the results based on batch size for correct plotting
    results.sort(key=lambda x: x['batch_size'])
    
    # Unpack the sorted results into separate lists
    batch_sizes = [res['batch_size'] for res in results]
    aps = [res['ap'] for res in results]
    aucs = [res['auc'] for res in results]
    
    return batch_sizes, aps, aucs

def plot_metrics(batch_sizes, aps, aucs):
    """
    Plots the AP and AUC metrics against batch sizes.

    Args:
        batch_sizes (list): A list of batch sizes.
        aps (list): A list of corresponding AP scores.
        aucs (list): A list of corresponding AUC scores.
    """
    if not batch_sizes:
        print("No data to plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot AP and AUC curves
    ax.plot(batch_sizes, aps, marker='o', linestyle='-', label='Test AP')
    ax.plot(batch_sizes, aucs, marker='s', linestyle='--', label='Test AUC')

    # Set the x-axis to a logarithmic scale as batch sizes are powers of 2
    ax.set_xscale('log', base=2)
    
    # Set labels, title, and legend
    ax.set_xlabel('Batch Size (log scale)')
    ax.set_ylabel('Score')
    ax.set_title('Test AP and AUC vs. Batch Size')
    ax.legend()
    
    # Improve readability
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig("batch_size_vs_metrics.png", dpi=300)

    # Show the plot
    # plt.show()

if __name__ == '__main__':
    # Define the directory where the log files are located.
    # An empty string '' means the current directory.
    log_directory = '' 
    
    # Step 1: Parse the data from log files
    batch_sizes, test_aps, test_aucs = parse_log_files(log_directory)
    
    # Print the extracted data
    print("Extracted Data:")
    for i in range(len(batch_sizes)):
        print(f"Batch Size: {batch_sizes[i]:<5} -> AP: {test_aps[i]:.6f}, AUC: {test_aucs[i]:.6f}")
        
    # Step 2: Plot the parsed data
    plot_metrics(batch_sizes, test_aps, test_aucs)
