#!/bin/bash

# ============================================================
# 实验运行脚本
# 数据集: MOOCRadar
# 策略: knowledge_graph (correctness:sequence = 1:0)
# ============================================================

# 保持脚本在前台运行
exec > >(tee -a run.log) 2>&1

set -e  # 遇到错误立即退出

# 通用参数
MODEL_TYPE="llm"
MODEL_NAME="gpt-3.5-turbo"
DATA_PATH="./datasets"
DATASET_NAME="MOOCRadar"
LOG_PATH="./logs"
TRAIN_SPLIT="0.8"
EVAL_STRATEGY="simple"
TEST_NUM="10"
RANDOM_SEED="42"
KG_PATH="./datasets/moderate/MOOCRadar/knowledge_graph_correctness1.0_sequence0.0.json"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_separator() {
    echo "============================================================"
}

# 切换到脚本所在目录
cd "$(dirname "$0")"

log_separator
log_info "开始运行实验"
log_info "工作目录: $(pwd)"
log_separator

# ============================================================
# 实验 1: moderate 模式, fewshot_num=4
# ============================================================
log_info "实验 1: moderate 模式, fewshot_num=4"
log_separator

python -u main.py \
  --model_type ${MODEL_TYPE} \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --data_mode moderate \
  --dataset_name ${DATASET_NAME} \
  --log_path ${LOG_PATH} \
  --train_split ${TRAIN_SPLIT} \
  --is_shuffle \
  --fewshot_num 4 \
  --fewshot_strategy knowledge_graph \
  --knowledge_graph_path ${KG_PATH} \
  --eval_strategy ${EVAL_STRATEGY} \
  --test_num ${TEST_NUM} \
  --random_seed ${RANDOM_SEED} \
  --skip_post_explain

log_info "实验 1 完成"
log_separator

# ============================================================
# 实验 2: sparse 模式, fewshot_num=4
# ============================================================
log_info "实验 2: sparse 模式, fewshot_num=4"
log_separator

python -u main.py \
  --model_type ${MODEL_TYPE} \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --data_mode sparse \
  --dataset_name ${DATASET_NAME} \
  --log_path ${LOG_PATH} \
  --train_split ${TRAIN_SPLIT} \
  --is_shuffle \
  --fewshot_num 4 \
  --fewshot_strategy knowledge_graph \
  --knowledge_graph_path ${KG_PATH} \
  --eval_strategy ${EVAL_STRATEGY} \
  --test_num ${TEST_NUM} \
  --random_seed ${RANDOM_SEED} \
  --skip_post_explain

log_info "实验 2 完成"
log_separator

# ============================================================
# 实验 3: sparse 模式, fewshot_num=8
# ============================================================
log_info "实验 3: sparse 模式, fewshot_num=8"
log_separator

python -u main.py \
  --model_type ${MODEL_TYPE} \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --data_mode sparse \
  --dataset_name ${DATASET_NAME} \
  --log_path ${LOG_PATH} \
  --train_split ${TRAIN_SPLIT} \
  --is_shuffle \
  --fewshot_num 8 \
  --fewshot_strategy knowledge_graph \
  --knowledge_graph_path ${KG_PATH} \
  --eval_strategy ${EVAL_STRATEGY} \
  --test_num ${TEST_NUM} \
  --random_seed ${RANDOM_SEED} \
  --skip_post_explain

log_info "实验 3 完成"
log_separator

log_info "所有实验完成!"
log_info "结果保存在: results/${MODEL_NAME}/"

# 保持脚本不退出，防止 screen 闪退
log_info "按 Ctrl+C 退出"
tail -f /dev/null
