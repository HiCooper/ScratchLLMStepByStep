#!/bin/bash
# 等待预训练 final.pt 出现后自动执行下游流水线（避免 GPU 空转），并检测训练进程是否意外退出。
# 用法：nohup bash scripts/wait_and_run_downstream.sh > models/checkpoints/wait_downstream.log 2>&1 &
# 说明：本脚本刻意**不加 `set -e`**——它的职责是长时间守护/自愈，很多命令（pgrep 无匹配返回 1、
# 单次训练失败需要重试、清理失败需要忽略）本来就允许非零退出；加了 -e 会让守护进程本身
# 被一次瞬时失败带走，反而更不安全。因此只用 `set -uo pipefail` 兜住未定义变量与管道错误。
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=models/checkpoints/pretrain_v2_full/final.pt
PATTERN="minigpt[.]train[.]pretrainer"
echo "[$(date '+%F %T')] waiting for $BASE ..."
for i in $(seq 1 1440); do                 # 最多等 12 小时
  if [ -f "$BASE" ]; then
    sleep 20
    echo "[$(date '+%F %T')] found $BASE, start downstream"
    bash scripts/run_downstream.sh
    exit $?
  fi
  if [ $((i % 10)) -eq 0 ]; then           # 每 5 分钟检查训练进程存活
    if ! pgrep -f "$PATTERN" >/dev/null; then
      sleep 45
      if ! pgrep -f "$PATTERN" >/dev/null && [ ! -f "$BASE" ]; then
        echo "[$(date '+%F %T')] ERROR: 训练进程已退出且未产出 $BASE，watcher 退出（请查看 pretrain_v2_full.log）"
        exit 3
      fi
    fi
  fi
  sleep 30
done
echo "[$(date '+%F %T')] timeout waiting for $BASE"
exit 1
