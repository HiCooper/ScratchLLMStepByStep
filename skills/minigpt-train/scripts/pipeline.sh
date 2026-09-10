#!/bin/bash
# MiniGPT 一键流水线（agent 可直接调用，幂等）：
#   --check  仅环境体检（默认）
#   --smoke  约 5 分钟小模型冒烟：建小语料 → 30 步预训练 → 采样，验证整条链路
#   --full   完整链路：全量 .bin（缺则构建）→ 预训练(自愈+compile) → 自动 SFT/CoT/评测 + 看板
#
# 用法：bash skills/minigpt-train/scripts/pipeline.sh [--check|--smoke|--full] [--tokenizer DIR]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODE="--check"
TOKENIZER="models/tokenizer_v3"
while [ $# -gt 0 ]; do
  case "$1" in
    --check|--smoke|--full) MODE="$1" ;;
    --tokenizer) TOKENIZER="$2"; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

log() { echo "[$(date '+%F %T')] $*"; }

# 只用真实运行实体判断（排除 setsid/nohup 包装与自身）
alive() {
  local pat="$1"
  ps -eo pid,args | grep -v grep | while read -r pid cmd; do
    case "$cmd" in *setsid*|*nohup*) continue ;; esac
    case "$cmd" in *"$pat"*) echo "$pid"; break ;; esac
  done
}

log "=== 1) 环境体检 ==="
bash skills/minigpt-train/scripts/preflight.sh --quick || { log "体检存在阻塞项，请先按提示修复"; exit 1; }

if [ "$MODE" = "--check" ]; then
  log "check 模式结束。下一步：bash skills/minigpt-train/scripts/pipeline.sh --smoke|--full"
  exit 0
fi

if [ "$MODE" = "--smoke" ]; then
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  if [ "${FREE_MB:-0}" -lt 3000 ]; then
    log "GPU 空闲显存仅 ${FREE_MB}MB（<3000MB），冒烟会与现有训练抢占显存。"
    log "请先停掉训练或稍后重试；本仓库正在跑的预训练不会被本脚本影响。"
    exit 3
  fi
  SMOKE_BIN="dataset/bins/smoke.bin"
  if [ ! -f "$SMOKE_BIN" ]; then
    log "构建冒烟语料（head 2000 行）"
    head -n 2000 dataset/pretrain_t2t_mini.jsonl > /tmp/smoke_corpus.jsonl
    python3 scripts/build_pretrain_bin.py build --corpus-jsonl /tmp/smoke_corpus.jsonl \
      --tokenizer-dir "$TOKENIZER" --out-bin "$SMOKE_BIN" || exit 1
  fi
  log "==== 冒烟预训练（小模型 30 步，约 2-4 分钟）===="
  rm -rf models/checkpoints/smoke_skill
  python3 -u -m minigpt.train.pretrainer \
    --data_tokenizer_dir "$TOKENIZER" --data_tokenized_bin "$SMOKE_BIN" --data_eval_ratio 0.02 \
    --model_emb_dim 128 --model_n_layers 2 --model_n_heads 4 --model_context_length 256 \
    --train_batch_size 8 --train_learning_rate 1e-3 --train_warmup_steps 5 \
    --train_eval_steps 10 --train_save_steps 20 --train_max_steps 30 \
    --paths_output_dir models/checkpoints/smoke_skill 2>&1 | tail -12
  test -f models/checkpoints/smoke_skill/final.pt || { log "冒烟失败：未产出 final.pt"; exit 1; }
  log "==== 冒烟推理 ===="
  python3 scripts/generate.py --checkpoint models/checkpoints/smoke_skill/final.pt \
    --tokenizer-dir "$TOKENIZER" --prompt "从前有座山" --max-new-tokens 30 --do-sample \
    --temperature 0.8 --top-k 50 --seed 0 2>&1 | tail -3
  log "冒烟通过 ✅（产物 models/checkpoints/smoke_skill/）"
  exit 0
fi

# ---------------- full ----------------
FULL_BIN="dataset/bins/pretrain_v3_full.bin"
if [ ! -f "$FULL_BIN" ]; then
  log "==== 构建全量语料 .bin（约 5-15 分钟，CPU）===="
  python3 scripts/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
    --tokenizer-dir "$TOKENIZER" --out-bin "$FULL_BIN" --max-lines 0 || exit 1
else
  log "全量 .bin 已存在：$FULL_BIN"
fi

OUT="models/checkpoints/pretrain_v2_full"
if [ -f "$OUT/final.pt" ]; then
  log "预训练已完成（$OUT/final.pt），跳过训练，直接确保下游与看板在跑"
else
  if [ -n "$(alive 'scripts/train_pretrain_resilient.sh')" ]; then
    log "预训练守护已在运行，跳过启动"
  else
    log "==== 启动预训练（自愈守护 + 空间守护）===="
    setsid nohup bash scripts/train_pretrain_resilient.sh > "$OUT.watchdog.log" 2>&1 < /dev/null &
    setsid nohup bash scripts/checkpoint_janitor.sh 60 2 120 > /tmp/janitor.log 2>&1 < /dev/null &
  fi
fi

if [ -z "$(alive 'scripts/wait_and_run_downstream.sh')" ]; then
  log "==== 启动下游接力（final.pt → SFT → CoT → 评测）===="
  setsid nohup bash scripts/wait_and_run_downstream.sh > "$OUT.downstream.log" 2>&1 < /dev/null &
fi

if [ -z "$(alive 'scripts/train_dashboard.py')" ]; then
  log "==== 启动看板 :8099 ===="
  setsid nohup python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5 --grad-clip 1.0 \
    > /tmp/dashboard.log 2>&1 < /dev/null &
fi

sleep 20
log "==== 当前状态 ===="
python3 scripts/train_dashboard.py --once || true
cat <<'TIP'

后续监控（agent 应定期执行）：
  python3 scripts/train_dashboard.py --once                       # 终端快照
  curl -s http://127.0.0.1:8099/data | head -c 300                # 网页看板数据
  tail -n 20 models/checkpoints/pretrain_v2_full.log              # 训练日志（每个 eval 一行）
  ls -t models/checkpoints/pretrain_v2_full/checkpoint-*.pth | head -1   # 最新续训点
完成判据：models/checkpoints/pretrain_v2_full/final.pt 存在，且下游 run 产出 final.pt / metrics.json / sample.txt
TIP
