import re


def calculate_average_total_time(file_path):
    """
    计算日志文件中'total time'的平均值。

    参数:
    file_path (str): 日志文件的路径。

    返回:
    float: 'total time'的平均值，如果未找到则返回None。
    """
    total_times = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                # 使用正则表达式匹配 'total time:X.XXs' 并提取时间
                match = re.search(r"total time:(\d+\.\d+)s", line)
                if match:
                    try:
                        time_value = float(match.group(1))
                        total_times.append(time_value)
                    except ValueError:
                        # 跳过无法转换为浮点数的值
                        continue
    except FileNotFoundError:
        logger.debug(f"错误: 文件 '{file_path}' 未找到。")
        return None
    except Exception as e:
        logger.debug(f"读取文件时发生错误: {e}")
        return None

    if total_times:
        average_time = sum(total_times) / len(total_times)
        logger.debug(average_time)
        return average_time
    else:
        logger.debug("未在文件中找到 'total time' 数据。")
        return None


# 示例用法:
# 假设您的日志数据保存在 'training_log.txt' 文件中
# 您可以将您提供的日志数据复制并粘贴到一个名为 'training_log.txt' 的文件中。
# 示例文件内容如下:
# Epoch 0:
# train loss:928308.9196  val ap:0.760580  val auc:0.804820
# total time:161.17s sample time:0.12s prep time:131.88s
# New best validation AP: 0.7606. Saving model to models/LASTFM_1753716876.1077034.pkl
# Epoch 1:
# train loss:817916.3062  val ap:0.788794  val auc:0.827378
# total time:177.44s sample time:0.11s prep time:144.69s
# New best validation AP: 0.7888. Saving model to models/LASTFM_1753716876.1077034.pkl
# ... (其余日志数据)

# log_file = 'training_log.txt' # 请将此路径替换为您的实际文件路径
avg_total_time = calculate_average_total_time(
    "~/Projects/lei_pipe/tgl_batch_plan_logs/LASTFM_TGN.log"
)

# if avg_total_time is not None:
#     logger.debug(f"平均 'total time': {avg_total_time:.2f} 秒")
