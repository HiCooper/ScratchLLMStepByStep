#!/bin/bash
# 预训练完成后的下游流水线（自动接力，避免 GPU 空转）：
#   1) 指令 SFT（chat）  2) CoT easy  3) CoT hard（课程式）
#   4) 评测：思考模式三策略(easy/hard) + 同切分 ppl 对比 + 问答样例
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
  --pretrain "$CHAT/final.pt" --data_tokenizer_dir "$TOK" \
  --sft-jsonl dataset/sft/sft_cot_easy_60k.jsonl --data_max_lines "$COT_EASY_LINES" --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1.5e-5 --train_warmup_steps 100 \
  --train_eval_steps 500 --train_save_steps 4000 --train_epochs "$COT_EASY_EPOCHS" \
  --paths_output_dir "$COT_EASY"

log "=== 3) CoT hard（$COT_HARD_LINES 条 × $COT_HARD_EPOCHS epochs） ==="
rm -rf "$COT_HARD"
python3 -u -m minigpt.train.sft_trainer \
  --pretrain "$COT_EASY/final.pt" --data_tokenizer_dir "$TOK" \
  --sft-jsonl dataset/sft/sft_cot_hard_80k.jsonl --data_max_lines "$COT_HARD_LINES" --data_max_len 512 \
  --train_batch_size 8 --train_learning_rate 1e-5 --train_warmup_steps 100 \
  --train_eval_steps 500 --train_save_steps 4000 --train_epochs "$COT_HARD_EPOCHS" \
  --paths_output_dir "$COT_HARD"

log "=== 4a) 思考模式评测：easy ==="
python3 -u scripts/eval_thinking.py --checkpoint "$COT_EASY/final.pt" --tokenizer-dir "$TOK" \
  --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl --n 60 \
  --strategies plain,single,two-phase --repetition-penalty 1.0 \
  --output "models/checkpoints/eval_${TAG}_cot_easy.json"

log "=== 4b) 思考模式评测：hard ==="
python3 -u scripts/eval_thinking.py --checkpoint "$COT_HARD/final.pt" --tokenizer-dir "$TOK" \
  --eval-jsonl dataset/sft/cot_eval_zh.jsonl --n 60 \
  --strategies plain,single,two-phase --repetition-penalty 1.0 \
  --output "models/checkpoints/eval_${TAG}_cot_hard.json"

log "=== 4c) 同切分 ppl 对比（$PPL_CKPTS） ==="
for ck in $PPL_CKPTS; do
  [ -f "$ck" ] || { log "跳过不存在的 $ck"; continue; }
  out="models/checkpoints/ppl_$(basename "$(dirname "$ck")").json"
  python3 -u scripts/evaluate_pretrain.py --checkpoint "$ck" --tokenizer-dir "$TOK" \
    --bin dataset/bins/pretrain_v3_full.bin --batch-size 8 --max-rows 512 --output "$out"
done

log "=== 4d) 问答样例（chat 与 CoT 模型） ==="
python3 -u scripts/generate.py --checkpoint "$CHAT/final.pt" --tokenizer-dir "$TOK" --chat \
  --prompt "你是谁" --prompt "如何保持身体健康？" --prompt "请解释什么是人工智能" \
  --max-new-tokens 90 --do-sample --temperature 0.7 --top-k 50 --top-p 0.9 \
  --repeat-penalty 1.2 --seed 5 --output-file "models/checkpoints/samples_${TAG}_chat.txt"
python3 -u scripts/generate.py --checkpoint "$COT_EASY/final.pt" --tokenizer-dir "$TOK" --chat \
  --thinking --prompt "请计算 27 ÷ 3 等于多少？" --prompt "小明有 6 支铅笔，又买了 8 支，一共多少支？" \
  --max-new-tokens 60 --thinking-max-tokens 60 --seed 5 \
  --output-file "models/checkpoints/samples_${TAG}_cot.txt"

log "=== 全部完成（TAG=$TAG）==="
