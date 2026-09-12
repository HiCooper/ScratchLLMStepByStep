#!/bin/bash
# v5 下游链路：基座 → SFT → CoT(easy/hard) → 评测 → 汇总报告
#
# 产物布局刻意与 `scripts/report_training.py` 的收集规则对齐（否则报告收不到）：
#   models/checkpoints/pretrain_*/metrics.json    基座
#   models/checkpoints/sft_*/metrics.json         SFT / CoT run
#   models/checkpoints/ppl_*.json                 同切分 ppl（**仅本 run 的 v5 val**，跨语料不可比）
#   models/checkpoints/eval_*_cot_*.json          CoT 分档准确率
#   models/checkpoints/samples_*.txt              样例
# 日志与 SUMMARY 另放 downstream_v5/，避免与 run 目录混在一起。
#
# 设计依据（references/data-distribution.md）：
#   - SFT 抽 4 万条 × 2 epoch = 8 万样本 / **实测 14.5M 监督 token**（scripts/audit_sft_lengths.py；
#     旧注释写的"10 万条 / 21.2M"两处都估错了），对 47.9M 模型约 0.3 token/参数，量级合适
#   - CoT easy 用 --easy-max 50：生成器题池唯一题面 45200（easy-max 9 只有 1063）→ 平均重复 56×→1.33×
#     （这组数字说的是**生成器题池**，不是磁盘文件的唯一题面数；验收看 audit_dataset.py 的重合检查）
#   - 算术评测必须 --repetition-penalty 1.0
#
# 用法：BASE=models/checkpoints/pretrain_v5 bash scripts/run_v5_downstream.sh
set -uo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-models/checkpoints/pretrain_v5}"        # 基座 run 目录
# 收集根可覆盖：默认是仓库的 models/checkpoints；端到端测试会在沙箱里指到别处
CP="${CP:-models/checkpoints}"                        # report_training.py 的收集根
OUT="${OUT:-$CP/downstream_v5}"                       # 日志/汇总目录
TOK="${TOK:-models/tokenizer_v3}"
BASE_CKPT="$BASE/best.pt"; [ -f "$BASE_CKPT" ] || BASE_CKPT="$BASE/final.pt"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.md"; : > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/driver.log"; }
stage() {  # stage <名字> <日志文件> <命令...>
  local name="$1"; shift; local lf="$1"; shift
  log "▶ $name 开始（日志 $lf）"
  if "$@" > "$lf" 2>&1; then
    log "✅ $name 完成"; echo "- ✅ $name" >> "$SUMMARY"
  else
    log "❌ $name 失败（rc=$?），继续下一阶段；详见 $lf"; echo "- ❌ $name —— 见 \`$lf\` 末尾" >> "$SUMMARY"
  fi
}
ckpt_of() { [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/final.pt"; }

# SFT 阶段专用：失败（多半是变长 padding 导致显存峰值）就降 batch、加梯度累积重试一次。
# 基座跑 20 小时才轮到 SFT，不能让它因为一次 OOM 就整个交付缺一块。
SFT_BS="${SFT_BS:-8}"     # 首次尝试的 batch（失败自动降 4 + accum 2）
sft_stage() {  # sft_stage <名字> <日志> <输出目录> <公共参数串（不含 --train_batch_size）>
  local name="$1" lf="$2" out="$3" args="$4"
  log "▶ $name 开始（日志 $lf）"
  # batch 由本函数统一注入，调用方**不要**在 $args 里再写 --train_batch_size，
  # 否则重试时会出现同名参数重复（argparse 取最后一个，能跑但日志误导）
  if python3 -u -m minigpt.train.sft_trainer $args \
       --train_batch_size "$SFT_BS" --paths_output_dir "$out" > "$lf" 2>&1; then
    log "✅ $name 完成"; echo "- ✅ $name" >> "$SUMMARY"; return 0
  fi
  log "⚠️ $name 首次失败，降 batch 重试（bs$SFT_BS → bs4 + grad_accum 2，有效 batch 不变）"
  if python3 -u -m minigpt.train.sft_trainer $args \
       --train_batch_size 4 --train_grad_accumulation_steps 2 \
       --paths_output_dir "$out" > "${lf%.log}.retry.log" 2>&1; then
    log "✅ $name 完成（降 batch 后）"; echo "- ✅ $name（OOM 降 batch 重试成功）" >> "$SUMMARY"; return 0
  fi
  log "❌ $name 两次都失败；详见 $lf 与 ${lf%.log}.retry.log"
  echo "- ❌ $name（含降 batch 重试）—— 见 \`${lf%.log}.retry.log\` 末尾" >> "$SUMMARY"; return 1
}

[ -f "$BASE_CKPT" ] || { log "❌ 找不到基座 checkpoint（$BASE/best.pt|final.pt），退出"; exit 1; }
log "基座 checkpoint = $BASE_CKPT"

# ── ① SFT 指令微调（4 万条 × 2 epoch；sft_data_zh 已打乱，取前 4 万行无前缀偏差）
SFT="$CP/sft_v5_chat"
sft_stage "SFT 指令微调" "$OUT/sft.log" "$SFT" \
  "--pretrain $BASE_CKPT --data_tokenizer_dir $TOK --sft-jsonl dataset/sft/sft_data_zh.jsonl \
   --data_max_lines 40000 --data_max_len 512 --train_learning_rate 1e-5 \
   --train_warmup_steps 50 --train_epochs 2 --train_eval_steps 200 --train_save_steps 500"
SFT_CKPT=$(ckpt_of "$SFT")

# ── ② CoT 数据：easy 用 --easy-max 50（幂等：已存在就跳过）
if [ -s dataset/sft/sft_cot_easy_em50_60k.jsonl ] && [ -s dataset/sft/cot_eval_easy_em50_disjoint.jsonl ]; then
  log "⏭ CoT easy 数据已存在，跳过生成"; echo "- ⏭ CoT easy 数据（已存在，跳过）" >> "$SUMMARY"
else
  stage "CoT easy 数据生成(easy-max 50)" "$OUT/cot_data.log" \
    python3 scripts/build_cot_sft.py --profile easy --easy-max 50 \
      --n-train 60000 --n-eval 200 \
      --out-train dataset/sft/sft_cot_easy_em50_60k.jsonl \
      --out-eval  dataset/sft/cot_eval_easy_em50_disjoint.jsonl
fi

# ── ③ CoT easy（从 chat 模型出发）
COTE="$CP/sft_v5_cot_easy"
sft_stage "CoT easy 训练" "$OUT/cot_easy.log" "$COTE" \
  "--pretrain $SFT_CKPT --data_tokenizer_dir $TOK --sft-jsonl dataset/sft/sft_cot_easy_em50_60k.jsonl \
   --data_max_lines 60000 --data_max_len 512 --train_learning_rate 1.5e-5 \
   --train_warmup_steps 50 --train_epochs 2 --train_eval_steps 200 --train_save_steps 500"

# ── ④ CoT hard（同样从 chat 出发，便于与 easy 对比）
COTH="$CP/sft_v5_cot_hard"
sft_stage "CoT hard 训练" "$OUT/cot_hard.log" "$COTH" \
  "--pretrain $SFT_CKPT --data_tokenizer_dir $TOK --sft-jsonl dataset/sft/sft_cot_hard_80k.jsonl \
   --data_max_lines 80000 --data_max_len 512 --train_learning_rate 1.5e-5 \
   --train_warmup_steps 50 --train_epochs 2 --train_eval_steps 200 --train_save_steps 500"

# ── ⑤ 评测（产物命名要对上 report_training.py 的 glob）
stage "v5 基座 ppl（--split val）" "$OUT/ppl_v5.log" \
  python3 scripts/evaluate_pretrain.py --checkpoint "$BASE_CKPT" --tokenizer-dir "$TOK" \
    --bin dataset/bins/pretrain_v5_full.bin --split val --batch-size 8 \
    --output "$CP/ppl_v5_general.json"
# 刻意**不做** "v4/历史基线 ppl 对比"：v5 语料 = pretrain_t2t.jsonl 的随机 80% + cosmopedia 20%，
# 而 v4 = 该文件 100% ⇒ v5 的训练集已包含 ~80% 的 v4 文档（历史 v2 用的 mini 语料更是被 100% 包含）。
# 用 v5 模型去评 v4/v2 的 val 切分 = 在测它见过的东西，数字会虚假偏低，不能作为 A/B 证据。
# 有效口径只有两个：① 本 run 自己的 v5 val 切分（文档级零重叠）；② 另行用同等预算训一个 v4 模型做真 A/B。
log "口径说明：跳过 v4/历史 ppl 对比（v5 语料包含 v4 的 ~80% 文档，跨语料 ppl 不可比）"
echo "- ⏭ 跨语料 ppl 对比（v5 含 v4 的 ~80% 文档，不可比；有效对比见 SUMMARY 口径说明）" >> "$SUMMARY"
stage "CoT easy 分档准确率（零重合留出集）" "$OUT/eval_cot_easy.log" \
  python3 scripts/eval_thinking.py --checkpoint "$(ckpt_of "$COTE")" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_easy_em50_disjoint.jsonl --n 200 \
    --strategies plain,single --repetition-penalty 1.0 --output "$CP/eval_v5_cot_easy.json"
stage "CoT hard 分档准确率（零重合留出集）" "$OUT/eval_cot_hard.log" \
  python3 scripts/eval_thinking.py --checkpoint "$(ckpt_of "$COTH")" --tokenizer-dir "$TOK" \
    --eval-jsonl dataset/sft/cot_eval_hard_disjoint.jsonl --n 200 \
    --strategies plain,single --repetition-penalty 1.0 --output "$CP/eval_v5_cot_hard.json"
stage "对话质检（chat_probe 51 场景）" "$OUT/chat_probe.log" \
  python3 scripts/chat_probe.py --checkpoint "$SFT_CKPT" --tokenizer-dir "$TOK" \
    --output "$CP/samples_v5_chat_probe.txt"

# ── ⑥ 汇总报告（复用既有生成器，自动收集上面那些路径）
stage "汇总报告" "$OUT/report.log" \
  python3 scripts/report_training.py --out "$CP/TRAINING_REPORT_v5.md"

# ── ⑦ 历史基线对照已移进生成器本身（report_training.py §6）
#      原因：只有生成器手里有 eval 曲线，才能给出**同 token 预算**的可核对点
#      （211k 步 ≈ 8.6 亿 tokens vs v2 的 15.82），而不是只贴一行"口径不同、不可比"。
#      这里不再追加任何内容，避免出现两个 §6。

log "全部阶段结束"
{
  echo ""
  echo "## 产物路径"
  echo "| 阶段 | checkpoint |"
  echo "|---|---|"
  echo "| 基座 | \`$BASE_CKPT\` |"
  echo "| SFT | \`$(ckpt_of "$SFT")\` |"
  echo "| CoT easy | \`$(ckpt_of "$COTE")\` |"
  echo "| CoT hard | \`$(ckpt_of "$COTH")\` |"
  echo ""
  echo "报告：\`$CP/TRAINING_REPORT_v5.md\`"
  echo ""
  echo "## 评测口径说明（重要）"
  echo "- ✅ **只用 v5 val 切分**（\`pretrain_v5_full.bin --split val\`）：均匀分块+双端对齐文档边界，"
  echo "  与训练文档级零重叠，是本次唯一有效的泛化指标。"
  echo "- ❌ **不与 v4 / 历史 v2 的 ppl 做直接对比**：v5 语料 = \`pretrain_t2t.jsonl\` 随机 80% + cosmopedia 20%，"
  echo "  而 v4 = 该文件 100%、v2 用的是其子集 mini ⇒ v5 模型已见过它们 ~80–100% 的文档，"
  echo "  评它们的 val 会虚假偏低。要做真 A/B 需用**同等 token 预算另训一个 v4 模型**再同切分对比。"
  echo "评测：\`$CP/{ppl_v5_general,eval_v5_cot_easy,eval_v5_cot_hard}.json\` + 对话样例 \`$CP/samples_v5_chat_probe.txt\`"
  echo ""
  echo "## 各阶段 eval_loss（训练日志尾部）"
  for lf in sft cot_easy cot_hard; do
    echo "### $lf"; grep -E "eval_loss" "$OUT/$lf.log" 2>/dev/null | tail -3
  done
  echo ""
  echo "## 关键指标"
  for f in ppl_v5_general eval_v5_cot_easy eval_v5_cot_hard; do
    [ -f "$CP/$f.json" ] && echo "- $f: \`$(head -c 300 "$CP/$f.json" | tr -d '\n')\`"
  done
} >> "$SUMMARY" 2>&1

# ── ⑧ 交付自检（最后一步，且写进 SUMMARY 末尾——用户/agent 第一眼就能看到"齐了没有"）
# 前面的 stage 都是"失败也继续"，所以必须有一步把散落各阶段的产物合起来判一次总账：
# 少了 SFT 或评测文件时，日志里只是几行 ❌，很容易被忽略成"训练好了"。
verify_rc=0
log "▶ 交付自检（scripts/verify_v5_delivery.py）"
python3 scripts/verify_v5_delivery.py > "$OUT/verify.log" 2>&1 || verify_rc=$?
if [ "$verify_rc" -eq 0 ]; then
  log "✅ 交付自检全部通过"
  echo "- ✅ 交付自检全部通过（明细 \`$OUT/verify.log\`）" >> "$SUMMARY"
else
  log "❌ 交付自检未通过（rc=$verify_rc），详见 $OUT/verify.log"
  {
    echo "- ❌ 交付自检未通过，缺失/异常项："
    grep -E "❌" "$OUT/verify.log" | sed 's/^/    /' || echo "    （详见 verify.log）"
  } >> "$SUMMARY" 2>&1
fi

log "完成。SUMMARY: $SUMMARY（自检 rc=$verify_rc）"
exit "$verify_rc"
