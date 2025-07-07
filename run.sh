#!/bin/bash
#
# run_all_experiments.sh - 批量运行所有实验任务。
# 这个脚本会依次执行每个任务，即使某个任务失败，也会继续执行下一个。
# 它还会记录每个任务的成功与失败状态。
#
# 使用方法：chmod +x run_all_experiments.sh && ./run_all_experiments.sh
#

# --- 配置区 ---
# 定义所有需要运行的脚本、数据集和模型
# 如果未来有新增，只需修改这里的列表即可
declare -a SCRIPTS=("run_single_dataset_his.sh")
declare -a DATASETS=("MOOC" "WIKI" "REDDIT" "LASTFM")
declare -a MODELS=("APAN" "TGN")

# 定义一个主日志文件，记录整个批量任务的流程
LOG_FILE="master_run_$(date +'%Y%m%d_%H%M%S').log"
SUCCESS_COUNT=0
FAILURE_COUNT=0
TOTAL_TASKS=$(( ${#SCRIPTS[@]} * ${#DATASETS[@]} * ${#MODELS[@]} ))
CURRENT_TASK=0

# --- 脚本开始 ---
# 记录脚本开始时间，并将所有输出（stdout和stderr）重定向到主日志文件和控制台
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================="
echo "自动化实验流程开始"
echo "开始时间: $(date)"
echo "总任务数: ${TOTAL_TASKS}"
echo "主日志文件: ${LOG_FILE}"
echo "================================================================="
echo

# 使用三层循环遍历所有任务组合
for script in "${SCRIPTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # 根据数据集分配固定的 GPU 对
        case ${dataset} in
            "MOOC")   gpus="0,1" ;;
            "WIKI")   gpus="2,3" ;;
            "REDDIT") gpus="4,5" ;;
            "LASTFM") gpus="6,7" ;;
            *)
                echo "[ERROR] 未知的数据集 '${dataset}'，跳过。"
                continue
                ;;
        esac

        for model in "${MODELS[@]}"; do
            ((CURRENT_TASK++))
            echo "-----------------------------------------------------------------"
            echo "--> [${CURRENT_TASK}/${TOTAL_TASKS}] 开始执行任务: ${script} | ${dataset} | ${model} | GPU ${gpus}"
            echo "--> 时间: $(date)"
            echo "-----------------------------------------------------------------"

            # 构造并执行命令
            # 使用 if 结构来判断命令是否成功执行
            # 即使命令失败（返回非零退出码），if 结构也会捕获它，然后执行 else 分支，而不会中止脚本
            if ./${script} "${dataset}" "${gpus}" "${model}"; then
                echo "✅ [成功] 任务 ${script} | ${dataset} | ${model} 已成功完成。"
                ((SUCCESS_COUNT++))
            else
                # "$?" 是一个特殊变量，它保存着上一条命令的退出码
                exit_code=$?
                echo "❌ [失败] 任务 ${script} | ${dataset} | ${model} 失败，退出码: ${exit_code}。将继续下一个任务。"
                ((FAILURE_COUNT++))
            fi

            echo
            # 在任务之间可以加入一个短暂的延时，给系统一点喘息时间
            echo "等待 5 秒后开始下一个任务..."
            sleep 5
            echo
        done
    done
done

echo "================================================================="
echo "所有实验任务已执行完毕"
echo "结束时间: $(date)"
echo "✅ 成功任务数: ${SUCCESS_COUNT}"
echo "❌ 失败任务数: ${FAILURE_COUNT}"
echo "================================================================="
