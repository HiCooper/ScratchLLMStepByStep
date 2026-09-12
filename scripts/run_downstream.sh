#!/bin/bash
# 预训练完成后的下游流水线（自动接力，避免 GPU 空转）：
#   1) 指令 SFT（chat）  2) CoT easy  3) CoT hard（课程式）
#   4) 评测：调用 scripts/run_downstream_evals.sh（思考模式 easy/hard + 同切分 ppl + 问答样例）
#
# 参数（环境变量；BASE/TAG 缺省时**自动选最新产出的 run**，不再写死某个早已不存在的 run）：
#   BASE      起始基座 checkpoint（默认：models/checkpoints 下最新的 pretrain_*/final.pt，
#             没有 final.pt 就用最新 best.pt；都没有则报错退出，不会拿错误路径硬跑）
#   TOK       分词器目录（默认 models/tokenizer_v3）
#   TAG       产物前缀（默认从 BASE 的目录名推导：pretrain_v5 → v5；领域模型传 domain 避免覆盖）
#   PPL_CKPTS 参与 ppl 同切分对比的 checkpoint 列表（空格分隔；默认只含本次 BASE，
#             要对比领域续训模型再显式追加——**跨语料 ppl 不可比**，只能同 bin 同 --split val 比）
#   SFT_EPOCHS / COT_EASY_EPOCHS / COT_HARD_EPOCHS / SFT_LINES / COT_EASY_LINES / COT_HARD_LINES
#
# 用法（后台）：setsid nohup bash scripts/run_downstream.sh > models/checkpoints/downstream.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

detect_base() {   # 最新的 pretrain run：先找有 final.pt 的，再退到有 best.pt 的
  local d
  for d in $(ls -dt models/checkpoints/pretrain_*/ 2>/dev/null || true); do
    [ -f "${d}final.pt" ] && { echo "${d}final.pt"; return 0; }
  done
  for d in $(ls -dt models/checkpoints/pretrain_*/ 2>/dev/null || true); do
    [ -f "${d}best.pt" ] && { echo "${d}best.pt"; return 0; }
  done
  return 1
}

BASE="${BASE:-}"
if [ -z "$BASE" ]; then
  BASE="$(detect_base || true)"
  if [ -z "$BASE" ]; then
    echo "❌ 未指定 BASE，且 models/checkpoints 下找不到任何 pretrain_*/final.pt|best.pt。" >&2
    echo "   用法：BASE=models/checkpoints/<run>/final.pt TAG=<tag> bash scripts/run_downstream.sh" >&2
    exit 1
  fi
  echo "[downstream] 未指定 BASE，自动选用最新 run：$BASE"
fi
TOK="${TOK:-models/tokenizer_v3}"
if [ -z "${TAG:-}" ]; then
  TAG="$(basename "$(dirname "$BASE")")"
  TAG="${TAG#pretrain_}"
  echo "[downstream] 未指定 TAG，按 BASE 推导：$TAG"
fi
SFT_EPOCHS="${SFT_EPOCHS:-2}"
COT_EASY_EPOCHS="${COT_EASY_EPOCHS:-2}"
COT_HARD_EPOCHS="${COT_HARD_EPOCHS:-3}"
SFT_LINES="${SFT_LINES:-60000}"
COT_EASY_LINES="${COT_EASY_LINES:-60000}"
COT_HARD_LINES="${COT_HARD_LINES:-80000}"
# 默认只评本次用的基座：以前的默认值写死了 pretrain_v1_512/pretrain_v2_full 两个
# 早已不存在的路径，跑起来只会得到两条"文件不存在"的噪声。
PPL_CKPTS="${PPL_CKPTS:-$BASE}"

CHAT="models/checkpoints/sft_${TAG}_chat"
COT_EASY="models/checkpoints/sft_${TAG}_cot_easy"
COT_HARD="models/checkpoints/sft_${TAG}_cot_hard"

log() { echo "[$(date '+%F %T')] $*"; }

# 选权重：best.pt 优先（小模型后期过拟合，best 通常更好），否则 final.pt
pick() { if [ -f "$1/best.pt" ]; then echo "$1/best.pt"; else echo "$1/final.pt"; fi; }

log "=== 0) 前置检查（BASE=$BASE TAG=$TAG）==="
test -f "$BASE" || { log "缺少 $BASE，退出"; exit 1; }

log "=== 1) 指令 SFT（chat，$SFT_LINES 条 × $SFT_EPOCHS epochs） ==="
rm -rf "$CHAT"
python3 -u -m minigpt.train.sft_trainer \
  --pretrain "$BASE" --data_tokenizer_dir "$TOK" \
  --sft-jsonl dataset/sft/sft_data_zh.jsonl --data_max_lines "$SFT_LINES" --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1e-5 --train_warmup_steps 100 \
  --train_eval_steps 500 --train_save_steps 4000 --train_epochs "$SFT_EPOCHS" \
  --paths_output_dir "$CHAT"

log "=== 2) CoT easy（$COT_EASY_LINES 条 × $COT_EASY_EPOCHS epochs） ==="
rm -rf "$COT_EASY"
python3 -u -m minigpt.train.sft_trainer \
  --pretrain "$(pick "$CHAT")" --data_tokenizer_dir "$TOK" \
  --sft-jsonl dataset/sft/sft_cot_easy_em50_60k.jsonl --data_max_lines "$COT_EASY_LINES" --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1.5e-5 --train_warmup_steps 100 \
  --train_eval_steps 500 --train_save_steps 4000 --train_epochs "$COT_EASY_EPOCHS" \
  --paths_output_dir "$COT_EASY"

log "=== 3) CoT hard（$COT_HARD_LINES 条 × $COT_HARD_EPOCHS epochs） ==="
rm -rf "$COT_HARD"
python3 -u -m minigpt.train.sft_trainer \
  --pretrain "$(pick "$COT_EASY")" --data_tokenizer_dir "$TOK" \
  --sft-jsonl dataset/sft/sft_cot_hard_80k.jsonl --data_max_lines "$COT_HARD_LINES" --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1e-5 --train_warmup_steps 100 \
  --train_eval_steps 500 --train_save_steps 4000 --train_epochs "$COT_HARD_EPOCHS" \
  --paths_output_dir "$COT_HARD"

# 评测与出样阶段独立成脚本（可单独重跑，自动优先 best.pt）
TAG="$TAG" TOK="$TOK" PPL_CKPTS="$PPL_CKPTS" bash scripts/run_downstream_evals.sh

log "=== 全部完成（TAG=$TAG）==="
