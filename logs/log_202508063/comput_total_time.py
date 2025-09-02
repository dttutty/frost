import os
import re

def calculate_time_per_file(log_directory):
    """
    Finds all 'total time' entries within each .log file, sums them up per file,
    and returns a dictionary with the results.

    Args:
        log_directory (str): The path to the directory containing the log files.

    Returns:
        dict: A dictionary mapping filenames to their calculated total time.
              Returns None if the directory is not found.
    """
    # Regex to find all occurrences of the 'total time' value.
    time_pattern = re.compile(r"total time:(\d+\.?\d*)s")

    # Check if the target directory exists.
    if not os.path.isdir(log_directory):
        print(f"错误: 找不到目录 '{log_directory}'。")
        return None

    results = {}
    print(f"正在扫描目录: '{log_directory}'...")

    # Iterate over all files in the specified directory.
    for filename in os.listdir(log_directory):
        # Process only files ending with the .log extension.
        if filename.endswith(".log"):
            file_path = os.path.join(log_directory, filename)
            file_total_time = 0.0
            try:
                # Open and read the file content.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Use findall to get all matches in the file.
                    found_times = time_pattern.findall(content)
                    # If any matches are found, sum them up.
                    if found_times:
                        for time_str in found_times:
                            file_total_time += float(time_str)
                # Store the calculated sum for this file.
                results[filename] = file_total_time
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                results[filename] = "读取错误"

    return results

# Main execution block
if __name__ == "__main__":
    # Set the name of the directory containing your log files.
    log_dir = "."

    file_times = calculate_time_per_file(log_dir)

    # Print the results in a formatted table if the calculation was successful.
    if file_times is not None:
        print("\n--- 每个日志文件的 Total Time 合计 (按文件名排序) ---\n")
        if not file_times:
            print(f"在 '{log_dir}' 目录中没有找到 .log 文件。")
        else:
            # Print a header for the results table.
            print(f"{'Log 文件名':<40} | {'合计时间 (s)':<20}")
            print("-" * 62)
            
            # *** 修改点: 对结果按文件名 (字典的键) 进行排序 ***
            sorted_items = sorted(file_times.items())
            
            # Iterate through the sorted list and print the results.
            for filename, total_time in sorted_items:
                if isinstance(total_time, float):
                    # Format float values to 2 decimal places.
                    print(f"{filename:<40} | {total_time:<20.2f}")
                else:
                    # Print error messages as is.
                    print(f"{filename:<40} | {str(total_time):<20}")
        print("\n-----------------------------------------------\n")
