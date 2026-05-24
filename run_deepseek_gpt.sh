#!/usr/bin/env bash
# ============================================================
# 批量实验运行脚本（并行版）
# 数据集: MOOCRadar, XES3G5M
# 模型:
# 模式: sparse, moderate
# 版本: v1, v3
# few-shot: 4, 8, 16
# fewshot_strategy: random, knowledge_graph
#
# 并行策略: 6 个 API Key 全部利用，每轮 6 个并行
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

# ---------- 6 个 OpenAI 兼容 API Key（通过 api.bianxie.ai 代理） ----------
DEEPSEEK_API_KEYS=(
    "sk-OE2WnoJQdJC2t8iO2t60NEdMQaIHRekaiERj4fZIM8wGbP6n"
    "sk-cLmuoiOZKPDDbOxU7eRZcaDCw8XGMqWr6l0J7LKs1R04aGlE"
    "sk-BmPPtiVsiLcDUNKfCLm1RSYzEY2aeHtF5TvS0ZeaTCVApRVe"
    "sk-88TfuixidM9veLrXuLVMvTBDbfUIFIUACMTHrJP3uxknxnWa"
    "sk-CUJBNbOc4sa4RhBJK6VlnHoqfd2K4AbMlbQ4ZSb65vNND0d7"
    "sk-XHIyjzM5YXvd680LZFwn6BA688VXxpORbAeZQanG1iVHr94I"
)

# ---------- 费用耗尽检测 ----------
check_budget_in_log() {
    local log_file=$1
    if [ -f "$log_file" ] && grep -qiE \
        "insufficient.*(quota|balance|credit)|quota.*exhausted|余额不足|402|payment.required|insufficient_quota|credits.*insufficient|account.*balance|lack.*balance|over.*quota" \
        "$log_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

# ---------- 运行单个实验 ----------
# 返回值：0=成功, 1=失败(非费用原因), 2=费用耗尽
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
    local api_key=${10}
    local out_file=${11}

    local kg_flag=""
    if [ "$fewshot_strategy" = "knowledge_graph" ] && [ -n "$kg_path" ]; then
        kg_flag="--knowledge_graph_path $kg_path"
    fi

    OPENAI_API_KEY="$api_key" "$PYTHON_BIN" main.py \
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

    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        if check_budget_in_log "$out_file"; then
            return 2
        fi
    fi
    return $exit_code
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

# 模式列表 (sparse 和 moderate)
MODES=("sparse" "moderate")

# 模型列表 (GPT 启用，其他注释)
# MODELS=("gpt-3.5-turbo")
MODELS+=("deepseek-v4-flash")

# few-shot 数量 (2 种，不跑 fewshot=4)
FEWSHOT_NUMS=(4 8 16)

# few-shot 策略 (2 种)
FEWSHOT_STRATEGIES=("random" "knowledge_graph")

# 版本 (2 种)
VERSIONS=("v1" "v3")

# ============================================================
# 主循环 — 先收集所有待运行实验，再以 6 个并行跑两轮
# ============================================================
echo "==========================================" | tee -a "$LOG_FILE"
echo " 实验开始: $(date)" | tee -a "$LOG_FILE"
echo " 并行策略: 6 个 API Key 全部利用，每轮 6 个并行" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

TASK_COUNTER=0
OVERALL_BUDGET_FAIL=false

# ---------- 收集待运行实验 ----------
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

                        # 跳过结果已存在的实验
                        if check_result_exists "$dataset" "$mode" "$model" "$fewshot_num" "$fewshot_strategy" "$version" "$kg_path"; then
                            echo "  [跳过] [${dataset}][v${version}][fsn=${fewshot_num}][${fewshot_strategy}] 结果已存在" | tee -a "$LOG_FILE"
                            continue
                        fi

                        EXPERIMENTS+=("${dataset}|${mode}|${model}|${fewshot_num}|${fewshot_strategy}|${version}|${train_split}|${test_num}|${kg_path}")
                    done
                done
            done
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "共 ${#EXPERIMENTS[@]} 个实验待运行" | tee -a "$LOG_FILE"

# ---------- 每轮 6 个并行执行 ----------
BATCH_SIZE=6
total=${#EXPERIMENTS[@]}
ROUND=0

for ((offset=0; offset<total; offset+=BATCH_SIZE)); do
    if [ "$OVERALL_BUDGET_FAIL" = true ]; then
        echo "  [跳过] 剩余实验 (费用已耗尽)" | tee -a "$LOG_FILE"
        break
    fi

    ROUND=$((ROUND + 1))
    end=$((offset + BATCH_SIZE < total ? offset + BATCH_SIZE : total))

    echo "" | tee -a "$LOG_FILE"
    echo "++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"
    echo " 第 ${ROUND} 轮并行 ($((offset+1)) ~ ${end} / ${total})" | tee -a "$LOG_FILE"
    echo "++++++++++++++++++++++++++++++++++++++++++++++" | tee -a "$LOG_FILE"

    LOG_FILES=()
    PID_LIST=()

    batch_idx=0
    for ((i=offset; i<end; i++)); do
        IFS='|' read -r dataset mode model fewshot_num fewshot_strategy version train_split test_num kg_path <<< "${EXPERIMENTS[$i]}"

        api_key="${DEEPSEEK_API_KEYS[$batch_idx]}"

        TASK_COUNTER=$((TASK_COUNTER + 1))
        log_file="${BASE_DIR}/.batch_${TASK_COUNTER}.log"
        LOG_FILES+=("$log_file")

        echo "  启动 [${dataset}][${mode}][v${version}][fsn=${fewshot_num}][${fewshot_strategy}] -> key[${batch_idx}]" | tee -a "$LOG_FILE"

        run_experiment \
            "$dataset" "$mode" "$model" \
            "$fewshot_num" "$fewshot_strategy" "$version" \
            "$train_split" "$test_num" "$kg_path" \
            "$api_key" "$log_file" &
        PID_LIST+=($!)
        batch_idx=$((batch_idx + 1))

        # 错开启动时间，避免多进程同时 import pandas 导致锁冲突
        sleep 2
    done

    echo "  等待 ${#PID_LIST[@]} 个任务完成 (第 ${ROUND} 轮)..." | tee -a "$LOG_FILE"

    # 逐个 wait 收集结果
    BATCH_HAS_BUDGET_FAIL=false
    for i in "${!PID_LIST[@]}"; do
        pid="${PID_LIST[$i]}"
        log_file="${LOG_FILES[$i]}"
        wait "$pid"
        rc=$?

        if [ $rc -eq 0 ]; then
            echo "  [完成] pid=${pid}, rc=0" | tee -a "$LOG_FILE"
        elif [ $rc -eq 2 ]; then
            echo "  [FATAL] pid=${pid} 费用耗尽！日志: ${log_file}" | tee -a "$LOG_FILE"
            BATCH_HAS_BUDGET_FAIL=true
            OVERALL_BUDGET_FAIL=true
        else
            echo "  [失败] pid=${pid}, rc=${rc}, 日志: ${log_file}" | tee -a "$LOG_FILE"
        fi
    done

    if [ "$BATCH_HAS_BUDGET_FAIL" = true ]; then
        echo "  [FATAL] 当前轮次费用耗尽，终止后续实验！" | tee -a "$LOG_FILE"
        echo "  [FATAL] 如需恢复，请补充 API 额度后删除对应 .batch_*.log 文件再重新运行。" | tee -a "$LOG_FILE"
        for log_file in "${LOG_FILES[@]}"; do
            [ -f "$log_file" ] && cat "$log_file" >> "$LOG_FILE" && rm -f "$log_file"
        done
        echo "==========================================" | tee -a "$LOG_FILE"
        echo " 异常终止: $(date) (费用耗尽)" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        exit 2
    fi

    # 清理日志
    for log_file in "${LOG_FILES[@]}"; do
        [ -f "$log_file" ] && rm -f "$log_file"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo " 全部实验完成: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
