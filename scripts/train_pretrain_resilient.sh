#!/bin/bash
# 可自愈的长训守护：崩溃/被信号终止后自动从最近 checkpoint 续训，直到到达 TARGET_STEPS。
# 环境自适应：所有关键参数都可用环境变量覆盖（pipeline.sh 会按硬件预设自动注入）。
#
# 用法（务必 setsid 脱离工具进程组，避免被误杀）：
#   setsid nohup bash scripts/train_pretrain_resilient.sh > <log> 2>&1 < /dev/null &
#
# 可覆盖的环境变量（默认值 = 本仓库 6GB 单卡生产配置）：
#   OUT_DIR        产物目录            (models/checkpoints/pretrain_v2_full)
#   LOG            训练日志            ($OUT_DIR.log)
#   FALLBACK_CKPT  无 checkpoint 时的起点 (models/checkpoints/pretrain_v1_512/final.pt，可为空)
#   DATA_BIN       语料 .bin           (dataset/bins/pretrain_v3_full.bin)
#   TOKENIZER_DIR  分词器目录          (models/tokenizer_v3)
#   PRESET_ARGS    模型/训练超参串      (见下)
#   TARGET_STEPS   目标优化步数         (211000)
#   NPROC          GPU 数（>1 走 torchrun DDP）
#   DRY_RUN=1      只打印将要执行的训练命令，不启动
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-models/checkpoints/pretrain_v2_full}"
LOG="${LOG:-$OUT_DIR.log}"
FALLBACK_CKPT="${FALLBACK_CKPT:-models/checkpoints/pretrain_v1_512/final.pt}"
DATA_BIN="${DATA_BIN:-dataset/bins/pretrain_v3_full.bin}"
TOKENIZER_DIR="${TOKENIZER_DIR:-models/tokenizer_v3}"
TARGET_STEPS="${TARGET_STEPS:-211000}"
NPROC="${NPROC:-1}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
PRESET_ARGS="${PRESET_ARGS:---model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 200 --train_eval_steps 2000 --train_save_steps 4000 --train_epochs 7 --train_torch_compile True}"

log() { echo "[$(date '+%F %T')] $*"; }

if [ "$NPROC" -gt 1 ]; then
  LAUNCH=(torchrun --nproc_per_node "$NPROC" -m minigpt.train.pretrainer)
else
  LAUNCH=(python3 -u -m minigpt.train.pretrainer)
fi

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  if [ -f "$OUT_DIR/final.pt" ]; then
    log "final.pt 已存在，训练完成，退出"
    exit 0
  fi
  CKPT=$(ls -t "$OUT_DIR"/checkpoint-*.pth 2>/dev/null | head -1 || true)
  if [ -z "$CKPT" ] && [ -n "$FALLBACK_CKPT" ] && [ -f "$FALLBACK_CKPT" ]; then
    CKPT="$FALLBACK_CKPT"
  fi
  log "第 $attempt 次启动：nproc=$NPROC resume=${CKPT:-<从头>} target_steps=$TARGET_STEPS"
  log "预设参数: $PRESET_ARGS"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] ${LAUNCH[*]} --data_tokenizer_dir $TOKENIZER_DIR --data_tokenized_bin $DATA_BIN \\"
    echo "  --data_eval_ratio 0.001 $PRESET_ARGS --train_max_steps $TARGET_STEPS \\"
    echo "  ${CKPT:+--paths_last_checkpoint_path $CKPT }--paths_output_dir $OUT_DIR"
    exit 0
  fi

  if [ -n "$CKPT" ]; then
    "${LAUNCH[@]}" \
      --data_tokenizer_dir "$TOKENIZER_DIR" --data_tokenized_bin "$DATA_BIN" \
      --data_eval_ratio 0.001 $PRESET_ARGS \
      --train_max_steps "$TARGET_STEPS" \
      --paths_last_checkpoint_path "$CKPT" \
      --paths_output_dir "$OUT_DIR" >> "$LOG" 2>&1
  else
    "${LAUNCH[@]}" \
      --data_tokenizer_dir "$TOKENIZER_DIR" --data_tokenized_bin "$DATA_BIN" \
      --data_eval_ratio 0.001 $PRESET_ARGS \
      --train_max_steps "$TARGET_STEPS" \
      --paths_output_dir "$OUT_DIR" >> "$LOG" 2>&1
  fi
  rc=$?
  log "训练进程退出 rc=$rc"
  if [ -f "$OUT_DIR/final.pt" ]; then
    log "检测到 final.pt，判定完成，退出"
    exit 0
  fi
  sleep 20
done

log "达到最大重启次数，放弃（请检查 $LOG）"
exit 1
