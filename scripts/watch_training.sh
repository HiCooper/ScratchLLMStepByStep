#!/bin/bash
# 终端训练看板：每 5 秒刷新（Ctrl-C 退出）。SSH 里看进度用；浏览器版见 train_dashboard.py --serve。
# 用法：bash scripts/watch_training.sh [刷新秒数] [训练日志] [checkpoint目录]
#   不传日志/目录时**自动选最近在跑的那个 run**（以前默认写死 pretrain_v2_full，早就没有这个 run 了）。
set -euo pipefail
cd "$(dirname "$0")/.."
REFRESH="${1:-5}"

detect_run() {   # 取 models/checkpoints 下 mtime 最新的 pretrain_* 目录
  ls -dt models/checkpoints/pretrain_*/ 2>/dev/null | head -1 | sed 's#/$##'
}

LOG="${2:-}"
OUT="${3:-}"
if [ -z "$OUT" ]; then
  OUT="$(detect_run || true)"
fi
if [ -z "$LOG" ]; then
  if [ -n "$OUT" ]; then LOG="$OUT.log"; else LOG="models/checkpoints/train.log"; fi
fi
if [ -z "$OUT" ] || [ ! -d "$OUT" ]; then
  echo "❌ 找不到任何 models/checkpoints/pretrain_* 目录，请显式传入：" >&2
  echo "   bash scripts/watch_training.sh 5 <训练日志> <checkpoint目录>" >&2
  exit 1
fi
echo "监控 run：$OUT（日志 $LOG，每 ${REFRESH}s 刷新，Ctrl-C 退出）"

while true; do
  clear
  python3 scripts/train_dashboard.py --once --refresh "$REFRESH" --log "$LOG" --out-dir "$OUT" || true
  echo; echo "（每 ${REFRESH}s 刷新，Ctrl-C 退出；浏览器看板：python3 scripts/train_dashboard.py --serve --port 8099）"
  sleep "$REFRESH"
done
