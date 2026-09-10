#!/bin/bash
# checkpoint 空间守护：每个目录只保留最近 KEEP 个 checkpoint-*.pth（final.pt 永不删），
# 且只删除 mtime 早于 MIN_AGE 的文件（避免删掉正在写入的存档）。
# 用法：setsid nohup bash scripts/checkpoint_janitor.sh [间隔秒=60] [保留个数=2] > /tmp/janitor.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
INTERVAL="${1:-60}"
KEEP="${2:-2}"
MIN_AGE="${3:-120}"
echo "[janitor] start interval=${INTERVAL}s keep=${KEEP} min_age=${MIN_AGE}s"
while true; do
  for dir in models/checkpoints/*/; do
    mapfile -t files < <(find "$dir" -maxdepth 1 -name 'checkpoint-*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{print $2}')
    n=${#files[@]}
    if [ "$n" -gt "$KEEP" ]; then
      for f in "${files[@]:$KEEP}"; do
        if [ -n "$(find "$f" -mmin +$((MIN_AGE / 60 + 1)) 2>/dev/null)" ]; then
          sz=$(du -m "$f" 2>/dev/null | cut -f1)
          rm -f "$f" && echo "[janitor] $(date '+%T') removed $f (${sz}MB, kept $KEEP)"
        fi
      done
    fi
  done
  sleep "$INTERVAL"
done
