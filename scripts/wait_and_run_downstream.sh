#!/bin/bash
# 通用下游接力：等预训练 final.pt 出现后自动跑下游流水线（避免 GPU 空转），并看住训练进程别静默退出。
#
# 用法：
#   OUT_DIR=models/checkpoints/pretrain_xxx setsid nohup bash scripts/wait_and_run_downstream.sh \
#     > models/checkpoints/pretrain_xxx.downstream.log 2>&1 &
#
# 环境变量（都有默认值；**没有**任何写死的 run 名）：
#   OUT_DIR    预训练产物目录（默认 models/checkpoints/pretrain_v2_full；BASE 未给时用它拼 final.pt）
#   BASE       起始基座 checkpoint（默认 $OUT_DIR/final.pt，缺失时回退 best.pt）
#   TAG        下游产物前缀，透传给 run_downstream.sh（默认从 OUT_DIR 推导，如 pretrain_v5 → v5）
#   TOK        分词器目录（透传）
#   TIMEOUT_SEC 最多等多久（默认 24h；计数单位是**本进程**的预算，重启会重新计时）
#
# 为什么不加 `set -e`：它的职责是长时间守护，很多命令（pgrep 无匹配返回 1 等）本来就允许非零退出；
# 加了 -e 会让守护进程被一次瞬时失败带走，反而更不安全。故只用 `set -uo pipefail`。
#
# 另一个同类脚本是 watch_v5_chain.sh（v5 生产链路专用，多一层"训练守护双缺席就重新拉起"的二级监督）。
# 本脚本只做通用接力：等文件 → 跑下游 → 报告训练是否意外退出，不负责拉起训练。
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-models/checkpoints/pretrain_v2_full}"
BASE="${BASE:-}"
TOK="${TOK:-models/tokenizer_v3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-86400}"
PATTERN="minigpt[.]train[.]pretrainer"

# TAG 不写死：从 OUT_DIR 末段推导（pretrain_v5 → v5，pretrain_ddp → ddp），
# 否则下游会把 sft_v2_* 之类的目录写串。显式传 TAG 时以传入值为准。
if [ -z "${TAG:-}" ]; then
  TAG="$(basename "$OUT_DIR")"
  TAG="${TAG#pretrain_}"
fi

if [ -z "$BASE" ]; then
  BASE_OVERRIDE=""
else
  BASE_OVERRIDE="$BASE"
fi

log() { echo "[$(date '+%F %T')] $*"; }
log "接力启动：等 ${BASE_OVERRIDE:-$OUT_DIR/final.pt}（最多 $((TIMEOUT_SEC / 3600))h）；下游 TAG=$TAG tokenizer=$TOK"

ticks=$((TIMEOUT_SEC / 30))
for i in $(seq 1 "$ticks"); do
  # 起跑点**在触发时**才决定：优先 final.pt，没有才用 best.pt。
  # （启动时就算死会在 final.pt 还没落盘时错误地锁定 best.pt）
  PICK="$BASE_OVERRIDE"
  if [ -z "$PICK" ]; then
    if [ -f "$OUT_DIR/final.pt" ]; then PICK="$OUT_DIR/final.pt"
    elif [ -f "$OUT_DIR/best.pt" ]; then PICK="$OUT_DIR/best.pt"; fi
  fi
  if [ -n "$PICK" ]; then
    sleep 20                                     # 等 final.pt 落盘与 config 注入完成
    log "检测到 $PICK，开始下游（TAG=$TAG）"
    BASE="$PICK" TAG="$TAG" TOK="$TOK" bash scripts/run_downstream.sh
    rc=$?
    log "下游结束 rc=$rc"
    exit "$rc"
  fi
  if [ $((i % 10)) -eq 0 ]; then                 # 每 5 分钟检查训练进程存活
    if ! pgrep -f "$PATTERN" >/dev/null; then
      sleep 45
      if ! pgrep -f "$PATTERN" >/dev/null && [ ! -f "$OUT_DIR/final.pt" ]; then
        log "❌ 训练进程已退出且未产出 $OUT_DIR/final.pt，接力退出（请查看 $OUT_DIR.log）"
        exit 3
      fi
    fi
  fi
  sleep 30
done
log "等待超时（$((TIMEOUT_SEC / 3600))h）退出：未等到 $OUT_DIR/final.pt"
exit 1
