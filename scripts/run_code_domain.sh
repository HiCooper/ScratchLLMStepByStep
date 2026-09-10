#!/bin/bash
# 代码领域增量预训练 → 复跑下游 SFT/CoT → 生成报告（全自动接力）
#
# 流程：
#   0) parquet → jsonl（过滤 + 混入 15% 通用语料）→ 建 .bin        [CPU，可与当前 GPU 任务并行]
#   1) 等待当前下游（SFT/评测）跑完，避免抢 GPU
#   2) 从 pretrain_v2_full/final.pt 增量续训（lr 1e-4、1~2 epoch、自愈守护）
#   3) 以领域基座复跑 SFT/CoT（TAG=domain，产物与 v2 并存）
#   4) 生成 models/checkpoints/TRAINING_REPORT.md
#
# 用法（后台）：
#   setsid nohup bash scripts/run_code_domain.sh > models/checkpoints/domain_pipeline.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

SRC="${SRC:-dataset/IndustryCorpus2_computer_programming_code_high}"
JSONL="${JSONL:-dataset/domain/code_corpus.jsonl}"
BIN="${BIN:-dataset/bins/code_domain.bin}"
META="${BIN%.bin}.meta.json"
OUT_DIR="${OUT_DIR:-models/checkpoints/pretrain_domain_code}"
BASE_CKPT="${BASE_CKPT:-models/checkpoints/pretrain_v2_full/final.pt}"
TOK="${TOK:-models/tokenizer_v3}"
HOURS="${HOURS:-3.5}"          # 增量预训练时间预算（用于换算目标步数）
LR="${LR:-1e-4}"
MIN_CHARS="${MIN_CHARS:-300}"
MAX_CHARS="${MAX_CHARS:-8000}"
MAX_LINE="${MAX_LINE:-500}"
MIN_QUALITY="${MIN_QUALITY:-3.0}"
MIX_RATIO="${MIX_RATIO:-0.15}"

log() { echo "[$(date '+%F %T')] $*"; }

# ---------------- 0) 数据准备（CPU，可与 GPU 任务并行） ----------------
if [ ! -f "$JSONL" ]; then
  log "=== 0.1 parquet → jsonl（过滤 + 混入 $MIX_RATIO 通用语料）==="
  python3 scripts/parquet_to_jsonl.py --src "$SRC" --out "$JSONL" \
    --min-chars "$MIN_CHARS" --max-chars "$MAX_CHARS" --max-line-length "$MAX_LINE" \
    --min-quality "$MIN_QUALITY" \
    --mix-jsonl dataset/pretrain_t2t_mini.jsonl --mix-ratio "$MIX_RATIO" --mix-limit 300000 \
    --tokenizer "$TOK" || exit 1
else
  log "已存在 jsonl：$JSONL"
fi

if [ ! -f "$BIN" ]; then
  log "=== 0.2 建 .bin（tokenize，CPU）==="
  python3 scripts/build_pretrain_bin.py build --corpus-jsonl "$JSONL" \
    --tokenizer-dir "$TOK" --out-bin "$BIN" --max-lines 0 || exit 1
else
  log "已存在 bin：$BIN"
fi

TOKENS=$(python3 -c "import json;print(json.load(open('$META'))['tokens'])" 2>/dev/null || echo 0)
log "领域语料 tokens=$TOKENS"

# ---------------- 1) 等待 GPU 空闲（当前下游跑完） ----------------
log "=== 1) 等待当前下游 SFT/评测结束 ==="
for i in $(seq 1 1920); do  # 最多等 16 小时（下游 SFT+CoT+评测通常 3~4h）
  busy=$(ps -eo pid,args | grep -v grep | grep -cE 'minigpt\.train\.(sft_)?trainer|eval_thinking|evaluate_pretrain|scripts/generate\.py' || true)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  # 必须同时满足：无下游进程 + 接力 watcher 已退出 + 连续 3 次（90s）显存空闲，
  # 避免刚好落在两个阶段之间的 1s 间隙里误判空闲，导致与下游抢 GPU 而 OOM。
  if [ "$busy" = "0" ] && ! pgrep -f 'wait_and_run_downstream[.]sh' >/dev/null \
     && ! pgrep -f 'run_downstream[.]sh' >/dev/null && [ "${free:-0}" -ge 3000 ]; then
    idle_ok=$(( ${idle_ok:-0} + 1 ))
    [ "$idle_ok" -ge 3 ] && break
  else
    idle_ok=0
  fi
  sleep 30
done
log "GPU 空闲检测结束（busy=$busy, free_vram=${free}MB, idle_ok=${idle_ok:-0}）"

# ---------------- 2) 增量预训练（自愈守护） ----------------
STEPS_BY_TOKENS=$(( TOKENS / 4088 ))                       # bs8 × (ctx512-1)
STEPS_BY_TIME=$(python3 -c "print(int($HOURS*3600/0.17))")
TARGET_STEPS=$(( STEPS_BY_TOKENS < STEPS_BY_TIME ? STEPS_BY_TOKENS : STEPS_BY_TIME ))
[ "$TARGET_STEPS" -lt 200 ] && TARGET_STEPS=200
log "=== 2) 领域增量预训练：target_steps=$TARGET_STEPS（tokens=$TOKENS, 预算 ${HOURS}h, lr=$LR）==="

PRESET_ARGS="--model_emb_dim 512 --model_n_layers 10 --model_n_heads 8 --model_context_length 512 \
--train_batch_size 8 --train_learning_rate $LR --train_warmup_steps 100 \
--train_eval_steps 2000 --train_save_steps 4000 --train_epochs 3 --train_torch_compile True"

OUT_DIR="$OUT_DIR" LOG="$OUT_DIR.log" DATA_BIN="$BIN" TOKENIZER_DIR="$TOK" \
NPROC=1 TARGET_STEPS="$TARGET_STEPS" PRESET_ARGS="$PRESET_ARGS" FALLBACK_CKPT="$BASE_CKPT" \
  bash scripts/train_pretrain_resilient.sh || { log "领域预训练失败"; exit 1; }
log "领域预训练完成：$OUT_DIR/final.pt"

# ---------------- 3) 复跑下游（TAG=domain） ----------------
log "=== 3) 领域基座复跑 SFT/CoT（TAG=domain）==="
BASE="$OUT_DIR/final.pt" TAG=domain TOK="$TOK" \
PPL_CKPTS="models/checkpoints/pretrain_v1_512/final.pt models/checkpoints/pretrain_v2_full/final.pt $OUT_DIR/final.pt" \
  bash scripts/run_downstream.sh || { log "领域下游失败"; exit 1; }

# ---------------- 4) 报告 ----------------
log "=== 4) 生成训练报告 ==="
python3 scripts/report_training.py --out models/checkpoints/TRAINING_REPORT.md \
  --json models/checkpoints/training_report.json || true
log "=== 领域流水线全部完成 ==="
