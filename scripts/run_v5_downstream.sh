#!/bin/bash
# v5 下游链路：基座 → SFT → CoT(easy/hard) → 评测，逐阶段记录结果并生成汇总。
#
# 设计依据（见 references/data-distribution.md）：
#   - SFT 抽 4 万条 × 2 epoch（10 万条 / 21.2M 监督 token 对 47.9M 模型偏多，易过拟合）
#   - CoT easy 用 --easy-max 50 重新生成训练集与留出集：题面空间 ≥ 样本数/3，避免"100% 是记忆"
#   - CoT 从 chat 模型出发；评测必须 --repetition-penalty 1.0（算术题惩罚重复数字会误判）
#
# 用法（长任务，托管后台跑）：
#   BASE=models/checkpoints/pretrain_v5 bash scripts/run_v5_downstream.sh
# 产物：models/checkpoints/downstream_v5/{*.log,SUMMARY.md}
set -uo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-models/checkpoints/pretrain_v5}"        # 基座 run 目录
OUT="${OUT:-models/checkpoints/downstream_v5}"        # 汇总/日志目录
TOK="${TOK:-models/tokenizer_v3}"
BASE_CKPT="$BASE/best.pt"; [ -f "$BASE_CKPT" ] || BASE_CKPT="$BASE/final.pt"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"
: > "$SUMMARY"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/driver.log"; }
stage() {  # stage <名字> <日志文件> <命令...>
  local name="$1"; shift; local lf="$1"; shift
  log "▶ $name 开始（日志 $lf）"
  if "$@" > "$lf" 2>&1; then
    log "✅ $name 完成"
    echo "- ✅ $name" >> "$SUMMARY"
  else
    log "❌ $name 失败（rc=$?），继续下一阶段；详见 $lf"
    echo "- ❌ $name —— 见 \`$lf\` 末尾" >> "$SUMMARY"
  fi
}
ckpt_of() { [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/final.pt"; }

[ -f "$BASE_CKPT" ] || { log "❌ 找不到基座 checkpoint（$BASE/best.pt|final.pt），退出"; exit 1; }
log "基座 checkpoint = $BASE_CKPT"
python3 -c "
import json;m=json.load(open('$BASE/metrics.json',encoding='utf-8')) if __import__('os').path.exists('$BASE/metrics.json') else {}
print('基座指标:', {k:m.get(k) for k in ('step','train_loss','eval_loss','perplexity','best_eval_loss','best_step')})" 2>/dev/null | tee -a "$OUT/driver.log"

# ── ① SFT 指令微调（4 万条 × 2 epoch；sft_data_zh 已打乱，取前 4 万行无前缀偏差）
SFT="$OUT/sft_v5_chat"
stage "SFT 指令微调" "$OUT/sft.log" \
  python3 -u -m minigpt.train.sft_trainer \
    --pretrain "$BASE_CKPT" --data_tokenizer_dir "$TOK" \
    --sft-jsonl dataset/sft/sft_data_zh.jsonl --data_max_lines 40000 --data_max_len 512 \
    --model_context_length 512 \
    --train_batch_size 8 --train_learning_rate 1e-5 --train_warmup_steps 50 \
    --train_epochs 2 --train_eval_steps 200 --train_save_steps 500 \
    --paths_output_dir "$SFT"
SFT_CKPT=$(ckpt_of "$SFT")

# ── ② CoT 数据：easy 用 --easy-max 50 重生成（先建池再切留出集，保证零重合）
stage "CoT easy 数据生成(easy-max 50)" "$OUT/cot_data.log" \
  python3 scripts/build_cot_sft.py --profile easy --easy-max 50 \
    --n-train 60000 --n-eval 200 \
    --out-train dataset/sft/sft_cot_easy_em50_60k.jsonl \
    --out-eval  dataset/sft/cot_eval_easy_em50_disjoint.jsonl

# ── ③ CoT easy 训练（从 chat 模型出发）
COTE="$OUT/sft_v5_cot_easy"
stage "CoT easy 训练" "$OUT/cot_easy.log" \
  python3 -u -m minigpt.train.sft_trainer \
    --pretrain "$SFT_CKPT" --data_tokenizer_dir "$TOK" \
    --sft-jsonl dataset/sft/sft_cot_easy_em50_60k.jsonl --data_max_lines 60000 --data_max_len 512 \
    --model_context_length 512 \
    --train_batch_size 8 --train_learning_rate 1.5e-5 --train_warmup_steps 50 \
    --train_epochs 2 --train_eval_steps 200 --train_save_steps 500 \
    --paths_output_dir "$COTE"

# ── ④ CoT hard 训练（同样从 chat 出发，便于与 easy 对比）
COTH="$OUT/sft_v5_cot_hard"
stage "CoT hard 训练" "$OUT/cot_hard.log" \
  python3 -u -m minigpt.train.sft_trainer \
    --pretrain "$SFT_CKPT" --data_tokenizer_dir "$TOK" \
    --sft-jsonl dataset/sft/sft_cot_hard_80k.jsonl --data_max_lines 80000 --data_max_len 512 \
    --model_context_length 512 \
    --train_batch_size 8 --train_learning_rate 1.5e-5 --train_warmup_steps 50 \
    --train_epochs 2 --train_eval_steps 200 --train_save_steps 500 \
    --paths_output_dir "$COTH"

# ── ⑤ 评测：基座 ppl（同口径 val）+ CoT 零重合留出集（分档）+ 对话质检
stage "基座 ppl（--split val）" "$OUT/eval_base_ppl.log" \
  python3 scripts/evaluate_pretrain.py --checkpoint "$BASE_CKPT" --tokenizer-dir "$TOK" \
    --bin dataset/bins/pretrain_v5_full.bin --split val --batch-size 8 \
    --output "$OUT/eval_base_ppl.json"

stage "CoT easy 准确率（零重合留出集，分档）" "$OUT/eval_cot_easy.log" \
  python3 scripts/eval_thinking.py --checkpoint "$(ckpt_of "$COTE")" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_easy_em50_disjoint.jsonl --n 200 \
    --strategies plain,single --repetition-penalty 1.0 --output "$OUT/eval_cot_easy.json"

stage "CoT hard 准确率（零重合留出集，分档）" "$OUT/eval_cot_hard.log" \
  python3 scripts/eval_thinking.py --checkpoint "$(ckpt_of "$COTH")" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_hard_disjoint.jsonl --n 200 \
    --strategies plain,single --repetition-penalty 1.0 --output "$OUT/eval_cot_hard.json"

stage "对话质检（chat_probe 51 场景）" "$OUT/eval_chat_probe.log" \
  python3 scripts/chat_probe.py --checkpoint "$SFT_CKPT" --tokenizer-dir "$TOK" \
    --output "$OUT/chat_probe_v5.txt"

# ── 汇总
log "全部阶段结束，汇总写入 $SUMMARY"
{
  echo ""
  echo "## 产物"
  echo "- 基座：\`$BASE_CKPT\`"
  echo "- SFT：\`$(ckpt_of "$SFT")\`"
  echo "- CoT easy：\`$(ckpt_of "$COTE")\`"
  echo "- CoT hard：\`$(ckpt_of "$COTH")\`"
  echo ""
  echo "## 关键指标"
  for f in eval_base_ppl eval_cot_easy eval_cot_hard; do
    [ -f "$OUT/$f.json" ] && echo "- $f: \`$(head -c 400 "$OUT/$f.json" | tr -d '\n')\`"
  done
  echo ""
  echo "## 各阶段 eval_loss（训练日志尾部）"
  for lf in sft cot_easy cot_hard; do
    echo "### $lf"; grep -E "eval_loss" "$OUT/$lf.log" 2>/dev/null | tail -3
  done
} >> "$SUMMARY" 2>&1
log "完成。SUMMARY: $SUMMARY"
