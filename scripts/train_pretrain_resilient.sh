#!/bin/bash
# 可自愈的长训守护：崩溃/被信号终止后自动从最近 checkpoint 续训，直到到达 MAX_STEPS。
# 用法（务必 setsid 脱离工具进程组，避免被误杀）：
#   setsid nohup bash scripts/train_pretrain_resilient.sh > models/checkpoints/pretrain_v2_full.watchdog.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=models/checkpoints/pretrain_v2_full
FALLBACK=models/checkpoints/pretrain_v1_512/final.pt
MAX_STEPS=211000
MAX_RESTARTS=20
LOG="$OUT.log"

log() { echo "[$(date '+%F %T')] $*"; }

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  if [ -f "$OUT/final.pt" ]; then
    log "final.pt 已存在，训练完成，退出"
    exit 0
  fi
  CKPT=$(ls -t "$OUT"/checkpoint-*.pth 2>/dev/null | head -1 || true)
  [ -z "$CKPT" ] && CKPT="$FALLBACK"
  log "第 $attempt 次启动，resume from $CKPT（target max_steps=$MAX_STEPS）"

  python3 -u -m minigpt.train.pretrainer \
    --data_tokenizer_dir models/tokenizer_v3 \
    --data_tokenized_bin dataset/bins/pretrain_v3_full.bin \
    --data_eval_ratio 0.001 \
    --model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
    --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 200 \
    --train_eval_steps 2000 --train_save_steps 4000 --train_max_steps "$MAX_STEPS" \
    --train_epochs 7 --train_torch_compile True \
    --paths_last_checkpoint_path "$CKPT" \
    --paths_output_dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  log "训练进程退出 rc=$rc"
  if [ -f "$OUT/final.pt" ]; then
    log "检测到 final.pt，判定完成，退出"
    exit 0
  fi
  sleep 20
done

log "达到最大重启次数，放弃（请检查 $LOG）"
exit 1
