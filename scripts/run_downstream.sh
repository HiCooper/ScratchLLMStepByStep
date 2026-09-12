#!/bin/bash
# 预训练完成后的下游流水线（自动接力，避免 GPU 空转）：
#   1) 指令 SFT（chat）  2) CoT easy  3) CoT hard（课程式）
#   4) 评测：调用 scripts/run_downstream_evals.sh（思考模式 easy/hard + 同切分 ppl + 问答样例）
#
# 参数（环境变量，均有默认值，保持与既有 v2 产物一致的命名）：
#   BASE      起始基座 checkpoint（默认 models/checkpoints/pretrain_v2_full/final.pt）
#   TOK       分词器目录（默认 models/tokenizer_v3）
#   TAG       产物前缀（默认 v2；领域模型可传 domain，避免覆盖既有结果）
#   PPL_CKPTS 参与 ppl 同切分对比的 checkpoint 列表（空格分隔）
#   SFT_EPOCHS / COT_EASY_EPOCHS / COT_HARD_EPOCHS / SFT_LINES / COT_EASY_LINES / COT_HARD_LINES
#
# 用法（后台）：setsid nohup bash scripts/run_downstream.sh > models/checkpoints/downstream.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-models/checkpoints/pretrain_v2_full/final.pt}"
TOK="${TOK:-models/tokenizer_v3}"
TAG="${TAG:-v2}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
COT_EASY_EPOCHS="${COT_EASY_EPOCHS:-2}"
COT_HARD_EPOCHS="${COT_HARD_EPOCHS:-3}"
SFT_LINES="${SFT_LINES:-60000}"
COT_EASY_LINES="${COT_EASY_LINES:-60000}"
COT_HARD_LINES="${COT_HARD_LINES:-80000}"
PPL_CKPTS="${PPL_CKPTS:-models/checkpoints/pretrain_v1_512/final.pt models/checkpoints/pretrain_v2_full/final.pt}"

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
