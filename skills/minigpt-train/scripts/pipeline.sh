#!/bin/bash
# MiniGPT 自适应一键流水线（agent 可直接调用，幂等）：
#   --check   仅环境体检（默认）
#   --smoke   小模型冒烟（无 GPU 也可跑；有 GPU 时约 2-4 分钟）
#   --full    完整链路：按硬件预设建语料 → 预训练(自愈+compile/DDP/CPU) → 自动 SFT/CoT/评测 + 看板
#
# 环境自适应：自动读取 preflight 的硬件预设（cpu / gpu-tiny / gpu-small / gpu-mid / gpu-large / multi-gpu），
#            据此决定模型尺寸、batch、ctx、AMP、torch.compile、NPROC 与目标步数。
#
# 用法：
#   bash skills/minigpt-train/scripts/pipeline.sh [--check|--smoke|--full]
#        [--hours 10]            # 训练时间预算（用于换算目标步数）
#        [--preset NAME]         # 手动覆盖预设（cpu|gpu-tiny|gpu-small|gpu-mid|gpu-large|multi-gpu）
#        [--corpus-lines N]      # 构建 .bin 时取前 N 行（0=全量；CPU 默认 20000）
#        [--nproc N]             # 多卡时指定卡数（默认取检测值；配合 --preset multi-gpu）
#        [--dry-run]             # 只打印将要执行的命令，不真正启动
# 说明：本脚本刻意**不加 `set -e`**——它的职责是长时间守护/自愈，很多命令（pgrep 无匹配返回 1、
# 单次训练失败需要重试、清理失败需要忽略）本来就允许非零退出；加了 -e 会让守护进程本身
# 被一次瞬时失败带走，反而更不安全。因此只用 `set -uo pipefail` 兜住未定义变量与管道错误。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODE="--check"; HOURS=10; PRESET_OVERRIDE=""; CORPUS_LINES=""; DRY_RUN=0; NPROC_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check|--smoke|--full) MODE="$1" ;;
    --hours) HOURS="$2"; shift ;;
    --preset) PRESET_OVERRIDE="$2"; shift ;;
    --corpus-lines) CORPUS_LINES="$2"; shift ;;
    --nproc) NPROC_ARG="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

log() { echo "[$(date '+%F %T')] $*"; }

alive() {  # 只看真实运行实体，排除 setsid/nohup 包装与自身
  local pat="$1"
  ps -eo pid,args | grep -v grep | while read -r pid cmd; do
    case "$cmd" in *setsid*|*nohup*) continue ;; esac
    case "$cmd" in *"$pat"*) echo "$pid"; break ;; esac
  done
}

# ---------------- 1) 体检（产出 preflight.json，含硬件预设） ----------------
log "=== 1) 环境体检 ==="
bash skills/minigpt-train/scripts/preflight.sh --quick >/tmp/preflight.out 2>&1 || {
  echo "体检存在阻塞项："; tail -n 25 /tmp/preflight.out; exit 1; }
tail -n 6 /tmp/preflight.out

# ---------------- 2) 解析硬件预设（可直接用 --preset 覆盖） ----------------
PLAN=$(python3 - "$PRESET_OVERRIDE" "$HOURS" "$NPROC_ARG" <<'PY'
import json, sys
override, hours, nproc_arg = sys.argv[1], float(sys.argv[2]), sys.argv[3]
p = json.load(open("models/checkpoints/preflight.json", encoding="utf-8"))
pr = json.loads(json.dumps(p["preset"]))

# 手动预设定义（与 preflight 的自动判定保持同一套档位）
DEFS = {
    "cpu":       {"model": {"model_emb_dim": 128, "model_n_layers": 2, "model_n_heads": 4, "model_context_length": 128},
                  "train": {"train_batch_size": 8, "train_mixed_precision_dtype": "none", "train_torch_compile": False,
                            "train_eval_steps": 50, "train_save_steps": 200},
                  "assumed": 3.0,
                  "reason": "手动指定 CPU 预设：仅用于机制验证（无 GPU 时的降级方案）",
                  "expect": "实测 0.12–0.17k tok/s（6 核）；1 小时 ~0.5M tokens，请勿期待生成质量"},
    "gpu-tiny":  {"model": {"model_emb_dim": 384, "model_n_layers": 8, "model_n_heads": 8, "model_context_length": 384},
                  "train": {"train_batch_size": 4, "train_mixed_precision_dtype": "float16", "train_torch_compile": True},
                  "assumed": 0.25, "reason": "手动指定：单卡 <5.5GB", "expect": "约 10–18k tok/s"},
    "gpu-small": {"model": {"model_emb_dim": 512, "model_n_layers": 10, "model_n_heads": 8, "model_context_length": 512},
                  "train": {"train_batch_size": 8, "train_mixed_precision_dtype": "float16", "train_torch_compile": True},
                  "assumed": 0.17, "reason": "手动指定：单卡 5.5–8GB（本仓库生产配置）", "expect": "约 22–30k tok/s"},
    "gpu-mid":   {"model": {"model_emb_dim": 512, "model_n_layers": 10, "model_n_heads": 8, "model_context_length": 1024},
                  "train": {"train_batch_size": 12, "train_mixed_precision_dtype": "float16", "train_torch_compile": True},
                  "assumed": 0.12, "reason": "手动指定：单卡 8–16GB", "expect": "约 30–50k tok/s"},
    "gpu-large": {"model": {"model_emb_dim": 768, "model_n_layers": 12, "model_n_heads": 12, "model_context_length": 1024},
                  "train": {"train_batch_size": 16, "train_mixed_precision_dtype": "float16", "train_torch_compile": True},
                  "assumed": 0.06, "reason": "手动指定：单卡 ≥16GB", "expect": "约 60k+ tok/s"},
}
if override:
    pr["name"] = override
    if override == "multi-gpu":
        n = int(nproc_arg) if nproc_arg else max(int(pr.get("gpu_count") or 0), 2)
        base = DEFS.get(pr.get("name") if pr.get("name") in DEFS else "gpu-small", DEFS["gpu-small"])
        pr["model"].update(base["model"]); pr["train"].update(base["train"])
        pr["train"]["train_learning_rate"] = round(6e-4 * (n ** 0.5), 6)
        pr["launch"] = {"nproc": n, "extra": []}
        pr["assumed_s_per_step"] = base["assumed"]
        pr["reason"] = f"手动指定 {n} 卡 DDP：每卡 batch 不变，lr 已按 sqrt({n}) 放大"
        pr["expect"] = f"吞吐≈单卡×{n}×0.85；有效 batch = 每卡 batch × {n} × grad_accum"
    elif override in DEFS:
        d = DEFS[override]
        pr["model"] = dict(d["model"]); pr["train"].update(d["train"])
        pr["assumed_s_per_step"] = d["assumed"]; pr["launch"] = {"nproc": 1, "extra": []}
        pr["reason"], pr["expect"] = d["reason"], d["expect"]
    else:
        raise SystemExit(f"未知预设: {override}")

steps = max(50, int(hours * 3600 / max(float(pr["assumed_s_per_step"]), 1e-6)))
nproc = int(pr["launch"].get("nproc", 1))
ctx = int(pr["model"].get("model_context_length", 512))
bs = int(pr["train"].get("train_batch_size", 8))
tokens = steps * bs * max(ctx - 1, 1) * nproc
model = " ".join(f"--{k} {v}" for k, v in pr["model"].items())
train = " ".join(f"--{k} {v}" for k, v in pr["train"].items())
print(json.dumps({"preset": pr["name"], "reason": pr["reason"], "expect": pr["expect"],
                  "nproc": nproc, "assumed": pr["assumed_s_per_step"], "steps": steps,
                  "model_args": model, "train_args": train, "tokens": tokens,
                  "vram": pr.get("vram_gb"), "gpu_count": p["preset"].get("gpu_count")}, ensure_ascii=False))
PY
)
echo "$PLAN" > /tmp/pipeline_plan.json
python3 - "$HOURS" <<'PY'
import json, sys
p = json.load(open('/tmp/pipeline_plan.json', encoding='utf-8'))
print(f"[预设] {p['preset']}  GPU={p['gpu_count']}x{p['vram']}GB  nproc={p['nproc']}")
print(f"        {p['reason']}")
print(f"        预期：{p['expect']}")
print(f"[计划] 预算={sys.argv[1]}h → 目标步数={p['steps']}，预计训练 tokens≈{p['tokens']/1e6:.1f}M")
PY
PRESET=$(python3 -c "import json;print(json.load(open('/tmp/pipeline_plan.json',encoding='utf-8'))['preset'])")
NPROC=$(python3 -c "import json;print(json.load(open('/tmp/pipeline_plan.json',encoding='utf-8'))['nproc'])")
STEPS=$(python3 -c "import json;print(json.load(open('/tmp/pipeline_plan.json',encoding='utf-8'))['steps'])")
MODEL_ARGS=$(python3 -c "import json;print(json.load(open('/tmp/pipeline_plan.json',encoding='utf-8'))['model_args'])")
TRAIN_ARGS=$(python3 -c "import json;print(json.load(open('/tmp/pipeline_plan.json',encoding='utf-8'))['train_args'])")

if [ "$MODE" = "--check" ]; then
  log "check 完成。可执行：--smoke（冒烟）| --full --hours 10（完整）| --dry-run（只看命令）"
  exit 0
fi

# ---------------- 3) 冒烟 ----------------
if [ "$MODE" = "--smoke" ]; then
  SMOKE_BIN="dataset/bins/smoke_${PRESET}.bin"
  SMOKE_TRAIN="--model_emb_dim 128 --model_n_layers 2 --model_n_heads 4 --model_context_length 128 --train_batch_size 8 --train_learning_rate 1e-3 --train_warmup_steps 5 --train_eval_steps 10 --train_save_steps 20 --train_max_steps 30 --train_torch_compile False --train_mixed_precision_dtype none"
  CMD_BUILD="python3 scripts/build_pretrain_bin.py build --corpus-jsonl /tmp/smoke_corpus.jsonl --tokenizer-dir models/tokenizer_v3 --out-bin $SMOKE_BIN"
  CMD_TRAIN="python3 -u -m minigpt.train.pretrainer --data_tokenizer_dir models/tokenizer_v3 --data_tokenized_bin $SMOKE_BIN --data_eval_ratio 0.02 $SMOKE_TRAIN --paths_output_dir models/checkpoints/smoke_skill"
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] head -n 2000 dataset/pretrain_t2t.jsonl > /tmp/smoke_corpus.jsonl"
    echo "  $CMD_BUILD"; echo "  $CMD_TRAIN"
    exit 0
  fi
  if [ ! -f "$SMOKE_BIN" ]; then
    log "构建冒烟语料（2000 行）"
    head -n 2000 dataset/pretrain_t2t.jsonl > /tmp/smoke_corpus.jsonl
    eval "$CMD_BUILD" || exit 1
  fi
  if [ "$PRESET" != "cpu" ]; then
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    if [ "${FREE_MB:-0}" -lt 2000 ]; then
      log "GPU 空闲显存仅 ${FREE_MB}MB（<2000MB），冒烟会与现有任务抢占；请稍后或先停训练"; exit 3
    fi
  fi
  log "==== 冒烟训练（30 步，环境=$PRESET）===="
  rm -rf models/checkpoints/smoke_skill
  eval "$CMD_TRAIN" 2>&1 | tail -12
  test -f models/checkpoints/smoke_skill/final.pt || { log "冒烟失败：未产出 final.pt"; exit 1; }
  python3 scripts/generate.py --checkpoint models/checkpoints/smoke_skill/final.pt \
    --tokenizer-dir models/tokenizer_v3 --prompt "从前有座山" --max-new-tokens 24 --seed 0 2>&1 | tail -2
  log "冒烟通过 ✅ 预设=$PRESET（产物 models/checkpoints/smoke_skill/）"
  exit 0
fi

# ---------------- 4) full ----------------
if [ -z "$CORPUS_LINES" ]; then
  if [ "$PRESET" = "cpu" ]; then CORPUS_LINES=20000; else CORPUS_LINES=0; fi
fi
if [ "$PRESET" = "cpu" ]; then
  FULL_BIN="dataset/bins/pretrain_cpu20k.bin"
  OUT_DIR="models/checkpoints/pretrain_cpu_skill"
elif [ "$PRESET" = "multi-gpu" ]; then
  FULL_BIN="dataset/bins/pretrain_v4_full.bin"
  OUT_DIR="models/checkpoints/pretrain_ddp"
else
  FULL_BIN="dataset/bins/pretrain_v4_full.bin"
  OUT_DIR="models/checkpoints/pretrain_${PRESET}"
fi
LOG_FILE="$OUT_DIR.log"
PRESET_ARGS="$MODEL_ARGS $TRAIN_ARGS"

# 若已有训练在跑：沿用它的产物目录与日志（避免监控到错误路径）
RUNNING_PID="$(alive 'scripts/train_pretrain_resilient.sh')"
if [ -n "$RUNNING_PID" ]; then
  RUN_OUT="$(tr '\0' ' ' < /proc/$RUNNING_PID/cmdline 2>/dev/null | grep -oE 'OUT_DIR=[^ ]+' | head -1 | cut -d= -f2 || true)"
  if [ -z "$RUN_OUT" ]; then
    RUN_OUT="$(ps -o args= -p "$RUNNING_PID" 2>/dev/null | grep -oE -- '--paths_output_dir [^ ]+' | head -1 | awk '{print $2}' || true)"
  fi
  if [ -n "$RUN_OUT" ] && [ -d "$RUN_OUT" ]; then
    log "检测到已在运行的训练，沿用其产物目录：$RUN_OUT"
    OUT_DIR="$RUN_OUT"; LOG_FILE="$OUT_DIR.log"
  fi
fi

if [ "$DRY_RUN" = "1" ]; then
  cat <<EOF
[dry-run] 预设=$PRESET nproc=$NPROC 目标步数=$STEPS
  1) 数据：python3 scripts/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t.jsonl --tokenizer-dir models/tokenizer_v3 --out-bin $FULL_BIN --max-lines $CORPUS_LINES
  2) 训练：OUT_DIR=$OUT_DIR LOG=$LOG_FILE DATA_BIN=$FULL_BIN NPROC=$NPROC TARGET_STEPS=$STEPS PRESET_ARGS="$PRESET_ARGS" setsid nohup bash scripts/train_pretrain_resilient.sh > $OUT_DIR.watchdog.log 2>&1 &
  3) 守护：setsid nohup bash scripts/checkpoint_janitor.sh 60 2 120 > /tmp/janitor.log 2>&1 &
  4) 下游：setsid nohup bash scripts/wait_and_run_downstream.sh > $OUT_DIR.downstream.log 2>&1 &
  5) 看板：setsid nohup python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5 --grad-clip 1.0 > /tmp/dashboard.log 2>&1 &
EOF
  exit 0
fi

if [ ! -f "$FULL_BIN" ]; then
  log "==== 构建语料 .bin（max-lines=$CORPUS_LINES）===="
  python3 scripts/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t.jsonl \
    --tokenizer-dir models/tokenizer_v3 --out-bin "$FULL_BIN" --max-lines "$CORPUS_LINES" || exit 1
else
  log "语料已存在：$FULL_BIN"
fi

if [ -f "$OUT_DIR/final.pt" ]; then
  log "训练已完成（$OUT_DIR/final.pt），跳过启动"
elif [ -n "$(alive 'scripts/train_pretrain_resilient.sh')" ]; then
  log "已有训练守护在运行，跳过启动（如需切换预设请先停止旧任务）"
else
  log "==== 启动训练（预设=$PRESET, nproc=$NPROC, target_steps=$STEPS）===="
  OUT_DIR="$OUT_DIR" LOG="$LOG_FILE" DATA_BIN="$FULL_BIN" NPROC="$NPROC" \
    TARGET_STEPS="$STEPS" PRESET_ARGS="$PRESET_ARGS" \
    setsid nohup bash scripts/train_pretrain_resilient.sh > "$OUT_DIR.watchdog.log" 2>&1 < /dev/null &
  setsid nohup bash scripts/checkpoint_janitor.sh 60 2 120 > /tmp/janitor.log 2>&1 < /dev/null &
fi

if [ -z "$(alive 'scripts/wait_and_run_downstream.sh')" ]; then
  setsid nohup bash scripts/wait_and_run_downstream.sh > "$OUT_DIR.downstream.log" 2>&1 < /dev/null &
fi
if [ -z "$(alive 'scripts/train_dashboard.py')" ]; then
  setsid nohup python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5 --grad-clip 1.0 \
    > /tmp/dashboard.log 2>&1 < /dev/null &
fi

sleep 15
log "==== 当前状态 ===="
python3 scripts/train_dashboard.py --once --log "$LOG_FILE" --out-dir "$OUT_DIR" 2>/dev/null || true
cat <<TIP

监控：python3 scripts/train_dashboard.py --once [--log $LOG_FILE --out-dir $OUT_DIR]
      curl -s http://127.0.0.1:8099/data | head -c 200
完成判据：$OUT_DIR/final.pt 存在；下游 run 产出 final.pt/metrics.json/sample.txt
提示：CPU 预设仅用于机制验证；多卡预设已按 sqrt(N) 放大 lr，batch 为每卡大小。
TIP
