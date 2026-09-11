#!/bin/bash
# 终端训练看板：每 5 秒刷新（Ctrl-C 退出）
# 用法：bash scripts/watch_training.sh [刷新秒数] [训练日志] [checkpoint目录]
set -euo pipefail
cd "$(dirname "$0")/.."
REFRESH="${1:-5}"
LOG="${2:-models/checkpoints/pretrain_v2_full.log}"
OUT="${3:-models/checkpoints/pretrain_v2_full}"
while true; do
  clear
  python3 scripts/train_dashboard.py --once --refresh "$REFRESH" --log "$LOG" --out-dir "$OUT" || true
  echo; echo "（每 ${REFRESH}s 刷新，Ctrl-C 退出；浏览器看板：python3 scripts/train_dashboard.py --serve --port 8099）"
  sleep "$REFRESH"
done
