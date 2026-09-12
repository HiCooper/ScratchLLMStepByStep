#!/bin/bash
# v5 链路守护（二级监督）：等基座 final.pt → 自动接下游；同时看住"训练守护"本身。
#
# 为什么要二级监督：进程链是 `watch_v5_chain.sh` → `train_pretrain_resilient.sh` → `python trainer`。
# resilient 脚本能重启 trainer，但如果 **resilient 脚本自己**被杀死（会话中断、误杀、OOM-killer），
# 就再没有任何东西会拉起它 —— 训练会静默停到天亮。本脚本因此同时负责：
#   ① 完成信号：final.pt 出现 → 跑下游链路；
#   ② 看门狗：training 守护与 trainer **双双缺席** 持续 3 分钟 → 判定真死，按原参数重新拉起。
#      （只在"双缺席"时动手：resilient 正常重启 trainer 时有 20s 空窗，那时守护还在，不会误触发）
#
# 用法：setsid nohup bash scripts/watch_v5_chain.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

BASE_DIR="${BASE_DIR:-models/checkpoints/pretrain_v5}"
LOG="${LOG:-models/checkpoints/v5_chain.log}"
PIDFILE="${PIDFILE:-models/checkpoints/v5_chain.pid}"
# pidfile 由脚本**自己**写：启动方的 `setsid nohup bash ... & echo $! > pidfile`
# 记到的是外层 `bash -c` 包装进程的 pid（实测就是这么错的），拿它去 kill 会杀错进程。
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
PATTERN="minigpt[.]train[.]pretrainer"
WRAPPER="train_pretrain""_resilient.sh"
MAX_TICKS="${MAX_TICKS:-5760}"          # 30s × 5760 = 48h
# 为什么默认给到 48h 而不是刚好覆盖一次训练：这个计数是**进程**的预算，不是训练的预算——
# 每次重启（会话中断、被误杀、二级监督重新拉起）都从头计时，而超时是**静默跳过整条下游**。
# 实测：基座 17.5h + 下游 2~4h ≈ 21h，24h 的默认值只剩 3h 余量，一次重启就吃掉。
DEAD_STREAK_LIMIT="${DEAD_STREAK_LIMIT:-6}"   # 连续 6 次（3 分钟）双缺席才判死

# 重新拉起训练所需的环境（与首次启动一致；resume 由 resilient 脚本自己找最新 checkpoint）
export OUT_DIR="$BASE_DIR"
export DATA_BIN="${DATA_BIN:-dataset/bins/pretrain_v5_full.bin}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-models/tokenizer_v3}"
export TARGET_STEPS="${TARGET_STEPS:-412000}"
export FALLBACK_CKPT="${FALLBACK_CKPT:-}"
export RESET_STEP="${RESET_STEP:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PRESET_ARGS="${PRESET_ARGS:---model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 2000 --train_eval_steps 2000 --train_save_steps 2000 --train_epochs 1 --train_torch_compile True}"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "watcher 启动：等待 $BASE_DIR/final.pt（最多 $((MAX_TICKS/120)) 小时）；二级监督已启用"
dead=0
for i in $(seq 1 "$MAX_TICKS"); do
  if [ -f "$BASE_DIR/final.pt" ]; then
    log "检测到 final.pt，开始下游链路"
    sleep 20                                     # 等 final.pt 落盘稳定
    bash scripts/run_v5_downstream.sh >> "$LOG" 2>&1
    rc=$?
    log "下游链路结束 rc=$rc"
    echo "done rc=$rc $(date '+%F %T')" > models/checkpoints/v5_chain.DONE
    exit 0
  fi

  wrapper_alive=0; trainer_alive=0
  pgrep -f "$WRAPPER" >/dev/null && wrapper_alive=1
  pgrep -f "$PATTERN" >/dev/null && trainer_alive=1

  if [ "$wrapper_alive" -eq 0 ] && [ "$trainer_alive" -eq 0 ]; then
    dead=$((dead + 1))
    if [ "$dead" -ge "$DEAD_STREAK_LIMIT" ]; then
      log "⚠️ 训练守护与 trainer 双双缺席 $((DEAD_STREAK_LIMIT * 30)) 秒，判定已死 → 重新拉起"
      setsid nohup bash scripts/train_pretrain_resilient.sh >> "$BASE_DIR.driver.log" 2>&1 < /dev/null &
      log "已重新拉起训练守护（resume 由 resilient 自动选择最新 checkpoint）"
      dead=0
      sleep 60                                    # 给它启动时间，避免立刻重复判定
    fi
  else
    dead=0
  fi

  # 每 10 分钟记一次进度
  if [ $((i % 20)) -eq 0 ]; then
    newest=$(ls -t "$BASE_DIR"/checkpoint-*.pth 2>/dev/null | head -1 || true)
    [ -n "$newest" ] && log "进度：$(basename "$newest")"
  fi
  sleep 30
done
log "等待超时（$((MAX_TICKS/120)) 小时）退出"
exit 3
