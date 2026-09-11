#!/bin/bash
# 下游「评测与出样」阶段（只读 checkpoint，不训练）：可单独重跑，也可被 run_downstream.sh 复用。
#
#   4a) 思考模式评测：easy 留出集（plain / single / two-phase）
#   4b) 思考模式评测：hard 留出集
#   4c) 同切分 ppl 对比（同一 .bin、同一 --max-rows，保证可比）
#   4d) 中文问答样例（chat 模型 + CoT 模型）
#
# 选权重约定：优先用 best.pt（eval_loss 历史最优），不存在才回退 final.pt。
#
# 参数（环境变量，均有默认值）：
#   TAG       产物前缀（默认 v2）；决定 sft_${TAG}_{chat,cot_easy,cot_hard} 三个目录
#   TOK       分词器目录（默认 models/tokenizer_v3）
#   PPL_CKPTS 参与 ppl 同切分对比的 checkpoint 列表（空格分隔）
#   N         思考模式评测题量（默认 60，= 数据集内全部留出题）
#
# 用法：setsid nohup bash scripts/run_downstream_evals.sh > models/checkpoints/evals_${TAG}.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:-v2}"
TOK="${TOK:-models/tokenizer_v3}"
N="${N:-60}"
PPL_CKPTS="${PPL_CKPTS:-models/checkpoints/pretrain_v1_512/final.pt models/checkpoints/pretrain_v2_full/final.pt}"
BIN="${BIN:-dataset/bins/pretrain_v3_full.bin}"

CHAT="models/checkpoints/sft_${TAG}_chat"
COT_EASY="models/checkpoints/sft_${TAG}_cot_easy"
COT_HARD="models/checkpoints/sft_${TAG}_cot_hard"

log() { echo "[$(date '+%F %T')] $*"; }

# 选权重：best.pt 优先（小模型后期过拟合，best 通常更好），否则 final.pt
pick() {
  local dir="$1"
  if [ -f "$dir/best.pt" ]; then echo "$dir/best.pt"
  elif [ -f "$dir/final.pt" ]; then echo "$dir/final.pt"
  else echo ""; fi
}

log "=== 4a) 思考模式评测：easy（TAG=$TAG, N=$N）==="
CK=$(pick "$COT_EASY")
if [ -n "$CK" ]; then
  log "使用权重：$CK"
  python3 -u scripts/eval_thinking.py --checkpoint "$CK" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl --n "$N" \
    --strategies plain,single,two-phase --repetition-penalty 1.0 \
    --output "models/checkpoints/eval_${TAG}_cot_easy.json" || log "4a 失败（继续）"
else
  log "跳过：$COT_EASY 下没有 best.pt / final.pt"
fi

log "=== 4b) 思考模式评测：hard（TAG=$TAG, N=$N）==="
CK=$(pick "$COT_HARD")
if [ -n "$CK" ]; then
  log "使用权重：$CK"
  python3 -u scripts/eval_thinking.py --checkpoint "$CK" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_zh.jsonl --n "$N" \
    --strategies plain,single,two-phase --repetition-penalty 1.0 \
    --output "models/checkpoints/eval_${TAG}_cot_hard.json" || log "4b 失败（继续）"
else
  log "跳过：$COT_HARD 下没有 best.pt / final.pt"
fi

log "=== 4c) 同切分 ppl 对比：$PPL_CKPTS ==="
for ck in $PPL_CKPTS; do
  [ -f "$ck" ] || { log "跳过不存在的 $ck"; continue; }
  name=$(basename "$(dirname "$ck")")
  # --split val：评估 bin 尾部连续、对齐文档边界的验证集（与训练同一口径）。
  # 旧写法 --max-rows 512 取的是 bin 最前面的窗口，若该 bin 参与过训练，
  # 得到的是训练 loss，会严重高估泛化（历史 README 结论即因此失真）。
  python3 -u scripts/evaluate_pretrain.py --checkpoint "$ck" --tokenizer-dir "$TOK" \
    --bin "$BIN" --split val --batch-size 8 \
    --output "models/checkpoints/ppl_${name}.json" || log "4c $name 失败（继续）"
done

log "=== 4d) 中文问答样例（TAG=$TAG）==="
CK=$(pick "$CHAT")
if [ -n "$CK" ]; then
  log "chat 权重：$CK"
  python3 -u scripts/generate.py --checkpoint "$CK" --tokenizer-dir "$TOK" --chat \
    --prompt "你是谁" --prompt "如何保持身体健康？" --prompt "请解释什么是人工智能" \
    --max-new-tokens 90 --do-sample --temperature 0.7 --top-k 50 --top-p 0.9 \
    --repeat-penalty 1.2 --seed 5 --output-file "models/checkpoints/samples_${TAG}_chat.txt" || log "4d chat 失败（继续）"
fi
CK=$(pick "$COT_EASY")
if [ -n "$CK" ]; then
  log "CoT 权重：$CK"
  python3 -u scripts/generate.py --checkpoint "$CK" --tokenizer-dir "$TOK" --chat \
    --thinking --prompt "请计算 27 ÷ 3 等于多少？" --prompt "小明有 6 支铅笔，又买了 8 支，一共多少支？" \
    --max-new-tokens 60 --thinking-max-tokens 60 --seed 5 \
    --output-file "models/checkpoints/samples_${TAG}_cot.txt" || log "4d cot 失败（继续）"
fi

log "=== 评测与出样完成（TAG=$TAG）==="
