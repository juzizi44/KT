#!/usr/bin/env bash
# ============================================================
# 批量实验运行脚本（并行版）
# 数据集: MOOCRadar, XES3G5M
# 模型: Qwen2.5-32B-Instruct (本地 vLLM)
# 模式: moderate
# 版本: v1, v3
# few-shot: 8, 16
# fewshot_strategy: random, knowledge_graph
#
# 并行策略: 6 个并行任务同时跑，1 轮跑完所有参数组合
# ============================================================
set -o pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || exit 1

# 优先使用项目虚拟环境，避免系统 python 缺依赖
if [ -x "$BASE_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$BASE_DIR/venv/bin/python"
else
    PYTHON_BIN="python"
fi

LOG_FILE="$BASE_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# ---------- 本地 Qwen vLLM 服务 ----------
export QWEN_API_KEY="${QWEN_API_KEY:-token-abc123}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-http://localhost:6688/v1}"
# vLLM 暴露的模型 ID 是完整的路径名
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-/home/fsq/.cache/huggingface/hub/Qwen2.5-32B-Instruct/}"

echo "  Qwen API:  $QWEN_BASE_URL" | tee -a "$LOG_FILE"
echo "  Qwen Model: $QWEN_MODEL_NAME" | tee -a "$LOG_FILE"

# ---------- 运行单个实验 ----------
run_experiment() {
    local dataset=$1
    local mode=$2
    local model=$3
    local fewshot_num=$4
    local fewshot_strategy=$5
    local version=$6
    local train_split=$7
    local test_num=$8
    local kg_path=$9
    local out_file=${10}

    local kg_flag=""
    if [ "$fewshot_strategy" = "knowledge_graph" ] && [ -n "$kg_path" ]; then
        kg_flag="--knowledge_graph_path $kg_path"
    fi

    "$PYTHON_BIN" main.py \
        --model_type llm \
        --model_name "$model" \
        --data_path ./datasets \
        --data_mode "$mode" \
        --dataset_name "$dataset" \
        --log_path ./logs \
        --train_split "$train_split" \
        --is_shuffle \
        --fewshot_num "$fewshot_num" \
        --fewshot_strategy "$fewshot_strategy" \
        --eval_strategy simple \
        --test_num "$test_num" \
        --random_seed 42 \
        --skip_post_explain \
        --workers 20 \
        --version "$version" \
        $kg_flag \
        > "$out_file" 2>&1

    return $?
}

# ---------- 检测实验结果是否已存在 ----------
check_result_exists() {
    local dataset=$1
    local mode=$2
    local model=$3
    local fewshot_num=$4
    local fewshot_strategy=$5
    local version=$6
    local kg_path=$7

    local version_suffix=""
    [ "$version" = "v3" ] && version_suffix="_v3"

    local config_suffix=""
    if [ "$fewshot_strategy" = "knowledge_graph" ] && [ -n "$kg_path" ]; then
        local filename
        filename=$(basename "$kg_path")
        local graph_config="${filename%.json}"
        config_suffix="_${graph_config}"
    fi

    local result_file="results_${version}/${model}/${mode}/${fewshot_strategy}/${dataset}/llm_${model}_fsn${fewshot_num}_fss${fewshot_strategy}${config_suffix}_essimple${version_suffix}.json"
    local checkpoint_file="${result_file%.json}_checkpoint.json"

    # 有 checkpoint 视为未完成：强制进入任务，由 main.py 内部续跑。
    if [ -f "$checkpoint_file" ]; then
        return 1
    fi

    # 仅当结果文件存在且包含 final_metrics 才视为完成。
    [ -f "$result_file" ] || return 1
    grep -q '"final_metrics"' "$result_file"
}

# ============================================================
# 实验参数配置
# ============================================================

# 数据集特定的固定参数
declare -A TRAIN_SPLIT
declare -A TEST_NUM
TRAIN_SPLIT["MOOCRadar"]=0.8
TEST_NUM["MOOCRadar"]=20
TRAIN_SPLIT["XES3G5M"]=0.9
TEST_NUM["XES3G5M"]=5

# 知识图谱路径
declare -A KG_PATH
KG_PATH["MOOCRadar"]="./datasets/knowledge_graph/MOOCRadar_knowledge_graph.json"
KG_PATH["XES3G5M"]="./datasets/knowledge_graph/XES3G5M_knowledge_graph.json"

# 数据集列表
DATASETS=("MOOCRadar" "XES3G5M")

# 模式列表
MODES=("moderate" "sparse")

# 模型
MODELS=("qwen2.5-32b-instruct")

# few-shot 数量
FEWSHOT_NUMS=(4 8 16)

# few-shot 策略
FEWSHOT_STRATEGIES=("random" "knowledge_graph")

# 版本
VERSIONS=("v1" "v3")

# ============================================================
# 主循环 — 两个数据集全部合并，12 路分批并行
# ============================================================
echo "==========================================" | tee -a "$LOG_FILE"
echo " 实验开始: $(date)" | tee -a "$LOG_FILE"
echo " 并行策略: 12 路分批并行" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

TASK_COUNTER=0

# 收集所有数据集的待运行实验（合并到一个列表）
EXPERIMENTS=()
for dataset in "${DATASETS[@]}"; do
    train_split=${TRAIN_SPLIT[$dataset]}
    test_num=${TEST_NUM[$dataset]}
    kg_path=${KG_PATH[$dataset]}

    for mode in "${MODES[@]}"; do
        for model in "${MODELS[@]}"; do
            for version in "${VERSIONS[@]}"; do
                for fewshot_num in "${FEWSHOT_NUMS[@]}"; do
                    for fewshot_strategy in "${FEWSHOT_STRATEGIES[@]}"; do

                        if check_result_exists "$dataset" "$mode" "$model" "$fewshot_num" "$fewshot_strategy" "$version" "$kg_path"; then
                            echo "  [跳过] [${dataset}][${mode}][v${version}][fsn=${fewshot_num}][${fewshot_strategy}] 结果已存在" | tee -a "$LOG_FILE"
                            continue
                        fi

                        EXPERIMENTS+=("${dataset}|${mode}|${model}|${fewshot_num}|${fewshot_strategy}|${version}|${train_split}|${test_num}|${kg_path}")
                    done
                done
            done
        done
    done
done

total=${#EXPERIMENTS[@]}
echo "" | tee -a "$LOG_FILE"
echo " 总计待运行实验: ${total} 个" | tee -a "$LOG_FILE"

if [ $total -eq 0 ]; then
    echo " 全部实验均已存在，跳过" | tee -a "$LOG_FILE"
else
    # ---------- 12 路分批并行 ----------
    BATCH_SIZE=12
    ROUND=0

    for ((offset=0; offset<total; offset+=BATCH_SIZE)); do
        ROUND=$((ROUND + 1))
        end=$((offset + BATCH_SIZE < total ? offset + BATCH_SIZE : total))

        echo "" | tee -a "$LOG_FILE"
        echo "++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"
        echo " 第 ${ROUND} 轮并行 ($((offset+1)) ~ ${end} / ${total})" | tee -a "$LOG_FILE"
        echo "++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"

        LOG_FILES=()
        PID_LIST=()

        for ((i=offset; i<end; i++)); do
            IFS='|' read -r ds mode model fewshot_num fewshot_strategy version train_split_val test_num_val kg_path_val <<< "${EXPERIMENTS[$i]}"

            TASK_COUNTER=$((TASK_COUNTER + 1))
            log_file="${BASE_DIR}/.batch_${TASK_COUNTER}.log"
            LOG_FILES+=("$log_file")

            echo "  启动 [${ds}][${mode}][v${version}][fsn=${fewshot_num}][${fewshot_strategy}]" | tee -a "$LOG_FILE"

            run_experiment \
                "$ds" "$mode" "$model" \
                "$fewshot_num" "$fewshot_strategy" "$version" \
                "$train_split_val" "$test_num_val" "$kg_path_val" \
                "$log_file" &
            PID_LIST+=($!)
        done

        echo "  等待 ${#PID_LIST[@]} 个任务完成..." | tee -a "$LOG_FILE"

        for i in "${!PID_LIST[@]}"; do
            pid="${PID_LIST[$i]}"
            log_file="${LOG_FILES[$i]}"
            wait "$pid"
            rc=$?

            if [ $rc -eq 0 ]; then
                echo "  [完成] pid=${pid}, rc=0" | tee -a "$LOG_FILE"
            else
                echo "  [失败] pid=${pid}, rc=${rc}, 日志: ${log_file}" | tee -a "$LOG_FILE"
            fi
        done

        # 清理日志
        for log_file in "${LOG_FILES[@]}"; do
            [ -f "$log_file" ] && rm -f "$log_file"
        done
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo " 全部实验完成: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"