import glob
import os
import re
from collections import defaultdict

import numpy as np


def parse_log_files(log_directory: str):
    """
    Parses all .log files in a directory to extract and average test AP, AUC,
    and total time scores.

    Args:
        log_directory (str): The path to the directory containing log files.
    """
    # Check if the log directory exists
    if not os.path.isdir(log_directory):
        logger.debug(f"Error: Directory '{log_directory}' not found.")
        return

    # Use glob to find all files ending with .log in the specified directory
    log_files = glob.glob(os.path.join(log_directory, "*.log"))

    if not log_files:
        logger.debug(f"No .log files found in '{log_directory}'.")
        return

    # --- Compile regular expressions for efficiency ---

    # Pattern for AP and AUC scores
    ap_auc_pattern = re.compile(r"test ap:\s*(\d+\.\d+)\s+test auc:\s*(\d+\.\d+)")

    # New pattern for total time
    # This captures the floating point number after "total time:" and before "s"
    time_pattern = re.compile(r"total time:(\d+\.\d+)s")

    # A dictionary to store the results for each file
    # Added 'total_times' list to the default dictionary
    results = defaultdict(
        lambda: {"ap_scores": [], "auc_scores": [], "total_times": []}
    )

    # Process each log file found
    for file_path in sorted(log_files):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Check for AP/AUC match
                    ap_auc_match = ap_auc_pattern.search(line)
                    if ap_auc_match:
                        ap_score = float(ap_auc_match.group(1))
                        auc_score = float(ap_auc_match.group(2))
                        results[filename]["ap_scores"].append(ap_score)
                        results[filename]["auc_scores"].append(auc_score)

                    # Check for total time match on the same or different lines
                    time_match = time_pattern.search(line)
                    if time_match:
                        total_time = float(time_match.group(1))
                        results[filename]["total_times"].append(total_time)

        except Exception as e:
            logger.debug(f"Could not read file {filename}: {e}")

    # --- Print the summary report ---
    # Added a new column for Average Total Time
    logger.debug("-" * 105)
    logger.debug(
        f"{'Log File':<30} | {'AP/AUC Runs':<12} | {'Time Runs':<10} | {'Average AP':<15} | {'Average AUC':<15} | {'Avg Total Time (s)':<20}"
    )
    logger.debug("-" * 105)

    if not results:
        logger.debug("No results found in any log files.")
        return

    for filename, data in sorted(results.items()):
        ap_scores = data["ap_scores"]
        auc_scores = data["auc_scores"]
        total_times = data["total_times"]

        # Calculate averages if data was found
        avg_ap_str = f"{np.mean(ap_scores):<15.6f}" if ap_scores else f"{'N/A':<15}"
        avg_auc_str = f"{np.mean(auc_scores):<15.6f}" if auc_scores else f"{'N/A':<15}"
        avg_time_str = (
            f"{np.mean(total_times):<20.2f}" if total_times else f"{'N/A':<20}"
        )

        ap_runs_found = len(ap_scores)
        time_runs_found = len(total_times)

        # Print formatted results for the current file
        logger.debug(
            f"{filename:<30} | {ap_runs_found:<12} | {time_runs_found:<10} | {avg_ap_str} | {avg_auc_str} | {avg_time_str}"
        )

    logger.debug("-" * 105)


if __name__ == "__main__":
    LOGS_DIR = "logs_his"
    parse_log_files(LOGS_DIR)
