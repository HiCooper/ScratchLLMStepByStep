#!/bin/bash
# 一键恢复 v5 全链路（5 个守护）。**幂等**：已在跑的组件跳过，不会起第二份。
#
# 为什么需要它：进程内的监督（resilient 续训 + watch_v5_chain 二级监督）能扛住
# "进程被杀"，但扛不住**机器重启**——2026-09-12 21:47 这台机器重启了一次，
# 训练守护/trainer/janitor/watcher/看板**全部消失且无人拉起**，直到人工发现。
# 这个脚本把"从零恢复到完整链路"变成一条命令，并打印每个组件的状态。
#
# 用法：
#   bash scripts/resume_v5_chain.sh            # 恢复并打印状态
#   DRY_RUN=1 bash scripts/resume_v5_chain.sh  # 只看会做什么
#
# 可覆盖：RUN / BIN / TOK / PORT / TARGET_STEPS / PRESET_ARGS / KEEP
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-models/checkpoints/pretrain_v5}"
BIN="${BIN:-dataset/bins/pretrain_v5_full.bin}"
TOK="${TOK:-models/tokenizer_v3}"
PORT="${PORT:-8099}"
TARGET_STEPS="${TARGET_STEPS:-412000}"
KEEP="${KEEP:-2}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-4088}"     # bs8 × (ctx512-1)
PRESET_ARGS="${PRESET_ARGS:---model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 2000 --train_eval_steps 2000 --train_save_steps 2000 --train_epochs 1 --train_torch_compile True}"
LOG="${LOG:-$RUN.log}"
DRY_RUN="${DRY_RUN:-0}"

alive() { pgrep -f "$1" >/dev/null 2>&1; }
say() { echo "[resume $(date '+%F %T')] $*"; }

say "目标 run=$RUN  bin=$BIN  目标步数=$TARGET_STEPS"

# ── 0) 已完成就别动 ──
if [ -f "$RUN/final.pt" ] || [ -f models/checkpoints/v5_chain.DONE ]; then
  say "✅ 已检测到 final.pt 或 v5_chain.DONE —— 训练/交付已进入收尾，不重复启动训练"
  START_TRAIN=0
else
  START_TRAIN=1
fi

# ── 1) 训练守护（自愈续训；会自己从最新 checkpoint resume）──
if alive "scripts/train_pretrain_resilient[.]sh"; then
  say "⏭ 训练守护已在运行，跳过"
else
  newest=$(ls -t "$RUN"/checkpoint-*.pth 2>/dev/null | head -1 || true)
  if [ "$START_TRAIN" = "0" ]; then
    say "⏭ 跳过训练（已完成）"
  elif [ "$DRY_RUN" = "1" ]; then
    say "[dry-run] 将启动训练守护，resume=${newest:-<从头>}"
  else
    OUT_DIR="$RUN" LOG="$LOG" DATA_BIN="$BIN" TOKENIZER_DIR="$TOK" \
      TARGET_STEPS="$TARGET_STEPS" FALLBACK_CKPT="${FALLBACK_CKPT:-}" RESET_STEP="${RESET_STEP:-0}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      PRESET_ARGS="$PRESET_ARGS" \
      setsid nohup bash scripts/train_pretrain_resilient.sh > "$RUN.watchdog.log" 2>&1 < /dev/null &
    say "▶ 已启动训练守护（resume=${newest:-<从头>}，日志 $LOG）"
  fi
fi

# ── 2) checkpoint 轮转（防磁盘写满）──
if alive "checkpoint_janitor[.]sh"; then
  say "⏭ janitor 已在运行，跳过"
elif [ "$DRY_RUN" = "1" ]; then
  say "[dry-run] 将启动 janitor（保留 $KEEP 个）"
else
  setsid nohup bash scripts/checkpoint_janitor.sh 300 "$KEEP" 120 > models/checkpoints/janitor.log 2>&1 < /dev/null &
  say "▶ 已启动 janitor（每目录保留 $KEEP 个）"
fi

# ── 3) 链路 watcher（等 final.pt 接下游 + 二级监督训练守护）──
if alive "scripts/watch_v5_chain[.]sh"; then
  say "⏭ 链路 watcher 已在运行，跳过"
elif [ "$DRY_RUN" = "1" ]; then
  say "[dry-run] 将启动 watch_v5_chain.sh"
else
  BASE_DIR="$RUN" LOG=models/checkpoints/v5_chain.log DATA_BIN="$BIN" TOKENIZER_DIR="$TOK" \
    TARGET_STEPS="$TARGET_STEPS" PRESET_ARGS="$PRESET_ARGS" \
    setsid nohup bash scripts/watch_v5_chain.sh > /dev/null 2>&1 < /dev/null &
  sleep 2
  say "▶ 已启动链路 watcher（pidfile=$(cat models/checkpoints/v5_chain.pid 2>/dev/null || echo '?'))"
fi

# ── 4) 看板 ──
if alive "train_dashboard[.]py"; then
  say "⏭ 看板已在运行，跳过"
elif [ "$DRY_RUN" = "1" ]; then
  say "[dry-run] 将启动看板 :$PORT"
else
  setsid nohup python3 scripts/train_dashboard.py --serve --port "$PORT" --refresh 5 \
    --log "$LOG" --out-dir "$RUN" --target-step "$TARGET_STEPS" \
    --tokens-per-step "$TOKENS_PER_STEP" > /tmp/dashboard.log 2>&1 < /dev/null &
  say "▶ 已启动看板 :$PORT"
fi

[ "$DRY_RUN" = "1" ] && exit 0

# ── 5) 状态汇总（不猜，逐个核）──
sleep 8
echo
say "=== 状态 ==="
for pair in "训练守护:scripts/train_pretrain_resilient[.]sh" \
            "trainer:minigpt[.]train[.]pretrainer" \
            "janitor:checkpoint_janitor[.]sh" \
            "watcher:scripts/watch_v5_chain[.]sh" \
            "看板:train_dashboard[.]py"; do
  name="${pair%%:*}"; pat="${pair#*:}"
  if alive "$pat"; then echo "  ✅ $name"; else echo "  ❌ $name 未在运行"; fi
done
newest=$(ls -t "$RUN"/checkpoint-*.pth 2>/dev/null | head -1 || true)
echo "  最新 checkpoint: ${newest:-无}"
echo "  看板: $(curl -s -o /dev/null -w 'HTTP %{http_code}' --max-time 5 "http://127.0.0.1:$PORT/" || echo '无响应')  http://127.0.0.1:$PORT"
echo "  磁盘可用: $(df -h . | tail -1 | awk '{print $4}')"
