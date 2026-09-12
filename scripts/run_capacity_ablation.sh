#!/bin/bash
# 容量消融：同 token 预算下把基座从 512/10/8（47.9M）扩到 768/12/12（109.6M），
# 验证 README 里"hard CoT 是容量墙"的假设（而不是"步数不够"）。
#
# 为什么要单独一个脚本：容量对比必须**控制 token 预算**（isotoken），否则"更大模型
# 训了更多 token"和"更大模型容量更强"两个因素混在一起，结论无法归因。
#
# 用法（需要单卡 >=16GB；<16GB 请用 BS/ACCUM 组合把有效 batch 降下来）：
#   setsid nohup env TAG=cap768 TOKENS=8.6e8 BS=16 ACCUM=1 \
#     bash scripts/run_capacity_ablation.sh > models/checkpoints/cap768.log 2>&1 &
#
# 产物：
#   models/checkpoints/pretrain_cap768_final.pt 附近的标准产物（checkpoint-*/final.pt/metrics.json）
#   models/checkpoints/sft_cap768_cot_easy/、sft_cap768_cot_hard/
#   models/checkpoints/eval_cap768_cot_{easy,hard}.json   <- 用**零重合**留出集评测
#
# ⚠️ 本仓库开发环境无 GPU（torch.cuda.is_available()=False），该脚本只做过 bash -n 语法检查
#    与 dry-run 命令行拼装验证，**未真实执行过训练**。
set -uo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-cap768}"
TOKENS="${TOKENS:-8.6e8}"        # 与 pretrain_v2_full 对齐的 isotoken 预算
BS="${BS:-16}"
ACCUM="${ACCUM:-1}"
CTX="${CTX:-1024}"
EMB="${EMB:-768}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-12}"
LR="${LR:-6e-4}"
TOK="${TOK:-models/tokenizer_v3}"
BIN="${BIN:-dataset/bins/pretrain_v4_full.bin}"
OUT_DIR="${OUT_DIR:-models/checkpoints/pretrain_${TAG}}"
EPOCHS="${EPOCHS:-16}"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[$(date '+%F %T')] $*"; }

if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  log "❌ 未检测到 CUDA。容量消融必须在 GPU 上跑；本脚本只做命令行拼装（DRY_RUN=1 可离线检查）。"
  [ "$DRY_RUN" = "1" ] || exit 2
fi

# token 预算 -> 优化步数（有效 batch = BS × ACCUM，每步 token = BS×ACCUM×CTX）
STEPS=$(python3 -c "print(int(float('$TOKENS')/($BS*$ACCUM*$CTX)))")
log "isotoken 预算=$TOKENS tokens  有效batch=$((BS*ACCUM))  ctx=$CTX  => 目标步数=$STEPS"
log "模型：${EMB}/${LAYERS}/${HEADS}  lr=$LR  epochs=$EPOCHS（超出预算时靠 max_steps 截断）"

PRESET_ARGS="--model_emb_dim $EMB --model_n_layers $LAYERS --model_n_heads $HEADS \
--model_context_length $CTX --data_tokenizer_dir $TOK --data_tokenized_bin $BIN \
--train_batch_size $BS --train_grad_accumulation_steps $ACCUM --train_learning_rate $LR \
--train_warmup_steps 500 --train_eval_steps 2000 --train_save_steps 4000 \
--train_epochs $EPOCHS --train_max_steps $STEPS --train_reset_step False \
--train_torch_compile True --train_mixed_precision_dtype bfloat16"

if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] 将执行："
  echo "  OUT_DIR=$OUT_DIR TARGET_STEPS=$STEPS PRESET_ARGS=\"$PRESET_ARGS\" \\"
  echo "    setsid nohup bash scripts/train_pretrain_resilient.sh > $OUT_DIR.log 2>&1 &"
  echo "  # 之后："
  echo "  bash scripts/run_downstream.sh    # SFT + 零重合 CoT 评测（TAG=$TAG）"
  echo "  bash scripts/run_domain_compare.sh"
  exit 0
fi

log "=== 1) 预训练（isotoken，崩溃自动续跑）==="
OUT_DIR="$OUT_DIR" LOG="$OUT_DIR.log" DATA_BIN="$BIN" TOKENIZER_DIR="$TOK" \
  TARGET_STEPS="$STEPS" PRESET_ARGS="$PRESET_ARGS" RESET_STEP=0 \
  setsid nohup bash scripts/train_pretrain_resilient.sh > "$OUT_DIR.watchdog.log" 2>&1 &
log "已后台启动，日志：$OUT_DIR.log；监控：python3 scripts/train_dashboard.py --once --log $OUT_DIR.log"

cat <<EOF

=== 2) 训练完成后跑下游（手动执行，或等 wait_and_run_downstream.sh）===
  TAG=$TAG bash scripts/run_downstream.sh
  TAG=$TAG bash scripts/run_domain_compare.sh

=== 3) 对比口径（务必用同一 --split val）===
  # 容量假设成立 => cap768 在 hard CoT 留出集上明显优于 512/10/8，且 easy 已饱和
  python3 scripts/eval_thinking.py --checkpoint models/checkpoints/sft_${TAG}_cot_hard/final.pt \\
      --tokenizer-dir $TOK --eval-jsonl dataset/sft/cot_eval_hard_disjoint.jsonl \\
      --n 200 --strategies plain,single --repetition-penalty 1.0 \\
      --output models/checkpoints/eval_${TAG}_cot_hard.json
  与 512/10/8 基线对比：models/checkpoints/eval_v2_cot_hard.json
EOF
