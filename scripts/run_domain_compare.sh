#!/bin/bash
# 领域自适应的"双口径"对比：同一批 checkpoint 分别在
#   (a) 通用语料 dataset/bins/pretrain_v3_full.bin —— 看是否发生灾难性遗忘
#   (b) 领域语料 dataset/bins/code_domain.bin     —— 看领域适配是否真的有效
# 上做同切分（同 --max-rows）ppl 评估，并刷新 TRAINING_REPORT.md。
#
# 产物：models/checkpoints/ppl_code_domain_<标签>.json、models/checkpoints/TRAINING_REPORT.md
#
# 用法：TAG=domain bash scripts/run_domain_compare.sh
set -uo pipefail
cd "$(dirname "$0")/.."

TOK="${TOK:-models/tokenizer_v3}"
TAG="${TAG:-domain}"
DOM_BIN="${DOM_BIN:-dataset/bins/code_domain.bin}"
MAX_ROWS="${MAX_ROWS:-512}"   # 保留兼容：新口径由 --split val 决定评估窗口
# 领域基座（本 run）与对照基座（增量续训的起点）
DOM_CKPT="${DOM_CKPT:-models/checkpoints/pretrain_${TAG}_code/final.pt}"
[ -f "$DOM_CKPT" ] || DOM_CKPT="models/checkpoints/pretrain_${TAG}/final.pt"
BASE_CKPT="${BASE_CKPT:-models/checkpoints/pretrain_v2_full/final.pt}"

log() { echo "[$(date '+%F %T')] $*"; }

run_one() {   # $1=checkpoint $2=bin $3=输出前缀
  [ -f "$1" ] || { log "跳过不存在的 $1"; return 0; }
  # 双口径对比都走 --split val（尾部连续 + 文档边界对齐），否则比的是训练集 loss
  python3 -u scripts/evaluate_pretrain.py --checkpoint "$1" --tokenizer-dir "$TOK" \
    --bin "$2" --split val --batch-size 8 \
    --output "models/checkpoints/ppl_$3.json" || log "评估失败：$1 / $2"
}

if [ -f "$DOM_CKPT" ]; then
  log "=== 领域语料同切分 ppl（$DOM_BIN）==="
  run_one "$DOM_CKPT" "$DOM_BIN" "code_domain_${TAG}"
  run_one "$BASE_CKPT" "$DOM_BIN" "code_domain_base"
else
  log "未找到领域基座 checkpoint（$DOM_CKPT），跳过领域口径"
fi

# 通用语料口径由 run_downstream.sh 的 4c 阶段产出（PPL_CKPTS 已含领域基座），这里不重复跑。

log "=== 刷新训练报告 ==="
python3 scripts/report_training.py --out models/checkpoints/TRAINING_REPORT.md \
  --json models/checkpoints/training_report.json || true
log "=== 完成 ==="
