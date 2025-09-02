import os
import re
import glob
from collections import defaultdict
import numpy as np

def parse_log_files(log_directory: str):
    """
    Parses all .log files in a directory to extract and average test AP and AUC scores.

    Args:
        log_directory (str): The path to the directory containing log files.
    """
    # Check if the log directory exists
    if not os.path.isdir(log_directory):
        print(f"Error: Directory '{log_directory}' not found.")
        return

    # Use glob to find all files ending with .log in the specified directory
    log_files = glob.glob(os.path.join(log_directory, '*.log'))

    if not log_files:
        print(f"No .log files found in '{log_directory}'.")
        return

    # Compile the regular expression for efficiency.
    # This pattern looks for "test ap:", followed by optional whitespace,
    # then captures a floating-point number. It does the same for "test auc:".
    # It's robust to varying whitespace between the two metrics.
    pattern = re.compile(r"test ap:\s*(\d+\.\d+)\s+test auc:\s*(\d+\.\d+)")

    # A dictionary to store the results for each file
    results = defaultdict(lambda: {'ap_scores': [], 'auc_scores': []})

    # Process each log file found
    for file_path in sorted(log_files):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        # group(1) is the first captured value (ap), group(2) is the second (auc)
                        ap_score = float(match.group(1))
                        auc_score = float(match.group(2))
                        results[filename]['ap_scores'].append(ap_score)
                        results[filename]['auc_scores'].append(auc_score)
        except Exception as e:
            print(f"Could not read file {filename}: {e}")

    # --- Print the summary report ---
    print("-" * 80)
    print(f"{'Log File':<30} | {'Runs Found':<12} | {'Average AP':<15} | {'Average AUC':<15}")
    print("-" * 80)

    if not results:
        print("No results found in any log files.")
        return

    for filename, data in sorted(results.items()):
        ap_scores = data['ap_scores']
        auc_scores = data['auc_scores']
        
        # Check if any scores were actually found in the file
        if ap_scores:
            avg_ap = np.mean(ap_scores)
            avg_auc = np.mean(auc_scores)
            runs_found = len(ap_scores)
            
            # Print formatted results for the current file
            print(f"{filename:<30} | {runs_found:<12} | {avg_ap:<15.6f} | {avg_auc:<15.6f}")
        else:
            print(f"{filename:<30} | {'0':<12} | {'N/A':<15} | {'N/A':<15}")
            
    print("-" * 80)


if __name__ == "__main__":
    # The directory where your log files are located.
    # Assumes the script is run from the parent directory of 'logs'.
    LOGS_DIR = "logs_his"
    parse_log_files(LOGS_DIR)
