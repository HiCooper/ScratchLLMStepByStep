#!/bin/bash
# 生产预训练启动脚本（可多卡 DDP，也可单卡）：用法
#   bash scripts/pretrain_start.sh [extra args...]
# 说明：所有路径/超参均可通过 CLI 扁平参数覆盖（见 minigpt/config.py），
#       例：--model_emb_dim 512 --train_batch_size 8 --paths_output_dir models/checkpoints/pretrain_v2
set -euo pipefail
cd "$(dirname "$0")/.."

NPROC="${NPROC:-1}"
LOG_DIR="$(python3 -c 'from minigpt.config import PathConfig; import os; print(os.path.dirname(PathConfig.output_dir))' 2>/dev/null || echo models/checkpoints)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pretrain_start_$(date +%Y%m%d_%H%M%S).log"

if [ "$NPROC" -gt 1 ]; then
    echo "启动 $NPROC 卡 DDP 训练，日志: $LOG"
    nohup torchrun --nproc_per_node "$NPROC" -m minigpt.train.pretrainer "$@" > "$LOG" 2>&1 &
else
    echo "启动单卡训练，日志: $LOG"
    nohup python3 -m minigpt.train.pretrainer "$@" > "$LOG" 2>&1 &
fi
echo "pid=$!"
