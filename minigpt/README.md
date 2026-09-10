# minigpt 包使用说明（生产级）

本目录是 MiniGPT 的核心工程实现：模型、数据管线、训练器、指标记录、推理与评估。
所有入口共享 `minigpt/config.py` 的集中配置，支持扁平 CLI 覆盖，产物带配置快照、可续训、可复现。

```
minigpt/
├── config.py               # 配置（Model/Data/Train/Path + CLI 覆盖）
├── model/
│   ├── attention.py        # 多头注意力 / FlashAttention 版本（可选捕获注意力权重）
│   └── transformer.py      # GPTConfig / TransformerBlock / MiniGPT（含采样解码）
├── data/
│   ├── pretrain_dataset.py # jsonl→.bin(+meta) 管线、TokenBinDataset、旧版 Dataset
│   └── sft_dataset.py      # 指令数据 InstructionDataset / collate
└── train/
    ├── trainer.py          # 通用 Trainer（AMP/梯度累积/lr 调度/eval/save/resume/DDP）
    ├── metrics.py          # TensorBoard 指标（标量/直方图/图像/模型图/嵌入投影/文本）
    ├── pretrainer.py       # 预训练入口（python -m minigpt.train.pretrainer）
    └── sft_trainer.py      # SFT 入口（python -m minigpt.train.sft_trainer）
```

---

## 1. 快速开始

```bash
# 数据管线：jsonl → .bin + .meta.json
python scripts/build_pretrain_bin.py build \
    --corpus-jsonl dataset/pretrain_t2t_mini.jsonl \
    --tokenizer-dir models/tokenizer_v3 \
    --out-bin dataset/bins/pretrain_v3_600k.bin --max-lines 600000

# 预训练（单卡；多卡用 NPROC=2 bash scripts/pretrain_start.sh ...）
python3 -m minigpt.train.pretrainer \
    --paths_output_dir models/checkpoints/pretrain_v1_512

# SFT（在预训练 checkpoint 上微调）
python3 -m minigpt.train.sft_trainer \
    --pretrain models/checkpoints/pretrain_v1_512/final.pt \
    --paths_output_dir models/checkpoints/sft_v1_512

# 推理 / 评估
python scripts/generate.py --checkpoint models/checkpoints/sft_v1_512/final.pt \
    --tokenizer-dir models/tokenizer_v3 --chat --prompt "什么是AI？" \
    --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2
python scripts/evaluate_pretrain.py --checkpoint models/checkpoints/pretrain_v1_512/final.pt \
    --tokenizer-dir models/tokenizer_v3 --bin dataset/bins/pretrain_v3_600k.bin --max-rows 512
```

## 2. 配置（`minigpt/config.py`）

| dataclass | 关键字段 |
|---|---|
| `ModelConfig` | `emb_dim/n_layers/n_heads/context_length/vocab_size(0=由 tokenizer 推导)/drop_rate/qkv_bias/flash_attn/tie_word_embeddings/use_swiglu/use_checkpoint` |
| `DataConfig` | `tokenizer_dir/corpus_jsonl/content_key/max_lines/tokenized_bin/bin_meta/eval_ratio` |
| `TrainConfig` | `epochs/learning_rate/batch_size/weight_decay/grad_accumulation_steps/max_steps/warmup_steps/eval_steps/save_steps/grad_clip/mixed_precision_dtype/seed/num_workers` + **指标项（见 §4）** |
| `PathConfig` | `output_dir/last_checkpoint_path`（续训） |

任意入口均可扁平覆盖：`--model_emb_dim 512 --train_batch_size 8 --paths_output_dir ...`；
运行结束会把完整配置写入 `output_dir/config.json`。

## 3. 训练产物

```
output_dir/
├── config.json                # 本次运行完整配置
├── checkpoint-{step}.pth      # 周期 checkpoint：model/optimizer/scaler/RNG/config
├── best.pt                    # eval_loss 历史最优时的权重（--train_save_best False 可关）
├── final.pt                   # 主进程最终保存（同样含 config，可直接用于推理）
├── tensorboard/               # 指标事件（见下）
├── metrics.json               # 最终指标（step/epoch/train_loss/eval_loss/perplexity/best_eval_loss/best_step）
└── sample.txt                 # 训练后采样输出
```
续训：`--paths_last_checkpoint_path output_dir/checkpoint-2000.pth`（自动恢复优化器、混合精度缩放器与随机数状态）。

> **best.pt vs final.pt（生产经验）**：47.9M 的小模型在 SFT/CoT 后期几乎必然过拟合——实测 CoT easy
> `best_eval_loss=1.611`（约 step 5k）而 `final.pt` 已升到 `1.740`。因此训练器在每次 eval 创新低时额外写
> `best.pt`；做下游训练、思考模式评测、对外发布时**优先用 `best.pt`**（同结构、同 config，可直接替换 `final.pt`）。
> 每个 checkpoint 都含优化器 + RNG 状态（单文件 ~585MB），用 `scripts/checkpoint_janitor.sh` 控盘。

---

## 4. 训练过程指标（TensorBoard）

### 4.1 查看方式
```bash
tensorboard --logdir models/checkpoints/pretrain_v1_512/tensorboard --port 6006
# 浏览器打开 http://localhost:6006
```
> 需要 `tensorboard` 依赖（已在 `requirements.txt`）；SFT 运行日志在各自 `output_dir/tensorboard/`。

### 4.2 记录的指标一览

| 面板 | 标签（tag） | 记录时机 / 默认间隔 | 说明 |
|---|---|---|---|
| **标量 Scalars** | `train/loss`、`train/lr`、`train/grad_norm`、`train/tokens_per_sec` | 每个参数更新步 | 训练损失、当前学习率、梯度范数、吞吐 |
| | `eval/loss`、`eval/perplexity`、`train/loss_eval_window`、`train/lr_eval`、`eval/final_loss` | 每 `eval_steps` 步 / 训练结束 | 验证损失与困惑度（ppl） |
| **直方图 Histograms** | `weights/<参数名>`、`grads/<参数名>`、`buffers/<缓冲区名>` | 每 `log_hist_every` 步（默认 500） | 权重/梯度分布，用于观察消失/爆炸、量化前后分布 |
| **图像 Images** | `attention/layer0_head0` | 每 `log_attention_every` 步（默认 1000） | 第 0 层第 0 头在真实批次上的注意力热力图（`attention.py` 的 `capture_attention` 按需开启，默认关闭） |
| | `curves/loss` | 同注意力间隔 / 训练结束 | train/eval 损失曲线图 |
| **模型图 Graphs** | `add_graph`（GRAPH 面板） | 首个训练步一次（`--train_log_graph True`） | 由 `torch.jit.trace` 生成的 MiniGPT 计算图 |
| **嵌入投影 Projector** | `token_embedding` + `token_embedding/metadata` | 每 `log_embedding_every` 步（默认 2000）/ 训练结束 | 输入词嵌入的 3D PCA/UMAP 投影，metadata 为 token 文本 |
| **文本 Text** | `samples/generation/text_summary` | 每 `log_samples_every` 步（`0`=仅结束时）与训练结束 | 用当前权重对 `sample_prompts` 采样生成的结果 |

### 4.3 指标相关配置（`TrainConfig` / CLI 前缀 `--train_`）

| 参数 | 默认 | 作用 |
|---|---|---|
| `log_hist_every` | 500 | 权重/梯度直方图间隔；`0` 关闭 |
| `log_hist_max_numel` | 2,000,000 | 超大张量（如大词表 embedding）跳过直方图，避免事件文件膨胀 |
| `log_embedding_every` | 2000 | 投影面板间隔；`0` 关闭 |
| `projector_max_tokens` | 2000 | 写入投影面板的 token 向量数（均匀采样全词表，带 token 文本） |
| `log_attention_every` | 1000 | 注意力热力图间隔；`0` 关闭 |
| `log_graph` | False | 是否记录模型计算图 |
| `log_samples_every` | 0 | 周期性记录生成文本；`0` 表示只在训练结束时记录 |
| `sample_max_new_tokens` | 60 | 样本文本生成长度 |
| `sample_prompts` | `什么是AI？\|如何保持身体健康？\|从前有座山，山上有座庙` | 用 `\|` 分隔的采样提示词 |

示例（开启全部面板、加密采样间隔，便于快速自检）：
```bash
python3 -m minigpt.train.pretrainer \
    --model_emb_dim 64 --model_n_layers 2 --model_n_heads 4 --model_context_length 64 \
    --train_max_steps 6 --train_eval_steps 2 --train_log_hist_every 2 \
    --train_log_embedding_every 2 --train_log_attention_every 2 \
    --train_log_graph True --train_log_samples_every 2 \
    --paths_output_dir /tmp/metrics_smoke
```

### 4.4 实现与注意事项
- 实现位置：`minigpt/train/metrics.py` 的 `MetricsLogger`，由 `pretrainer.py` / `sft_trainer.py` 在主进程创建并挂到 `Trainer.metrics`；
  `Trainer` 在“参数更新步 / 评估点 / 训练结束”三处回调（`on_train_step`、`on_eval`、`on_final`）。
- **嵌入投影**：torch 2.13 + tensorboard 2.21 组合下 `SummaryWriter.add_embedding` 会静默不写事件，
  本实现改为按 projector 插件规范直接写 `TensorProto`（`<tag>` 与 `<tag>/metadata`），因此 Projector 面板可正常自动发现；
  首次打开 Projector 面板时 TensorBoard 需要几十秒做 PCA/UMAP 降维。
- **模型图**：`add_graph` 基于 `torch.jit.trace`，会对动态位置编码等产生 `TracerWarning`（不影响训练与图形展示），
  且 trace 以首个批次的序列长度为常量；如需精确图可改用 `torch.fx` 导出。
- **GPU 开销**：直方图/图像/投影按间隔触发，默认间隔下对吞吐影响 < 2%；若追求极致吞吐可把 `log_hist_every` 调大或设为 0。
- **事件文件体积**：直方图与投影是主要体积来源，长训练建议保留默认间隔或降低 `projector_max_tokens`。

### 4.5 指标自检（无 UI 快速验证）
```bash
python - <<'PY'
import glob, struct
from tensorboard.compat.proto import event_pb2
from tensorboard.backend.event_processing import event_accumulator as EA

def records(data):
    i = 0
    while i + 12 <= len(data):
        n = struct.unpack('<Q', data[i:i+8])[0]; i += 12
        yield data[i:i+n]; i += n + 4

path = sorted(glob.glob('models/checkpoints/pretrain_v1_512/tensorboard/events*'))[-1]
kinds, tags, graph = {}, set(), False
for raw in records(open(path, 'rb').read()):
    ev = event_pb2.Event(); ev.ParseFromString(raw)
    k = ev.WhichOneof('what'); kinds[k] = kinds.get(k, 0) + 1
    graph = graph or k == 'graph_def'
    if k == 'summary':
        tags |= {v.tag for v in ev.summary.value}
print('events:', kinds, '| graph:', graph)
print('panels:', [t for t in sorted(tags) if '/' in t][:12], '...')
ea = EA.EventAccumulator(path, size_guidance={'scalars':0,'histograms':0,'images':0,'tensors':0})
ea.Reload(); print({k: len(v) if isinstance(v, list) else v for k, v in ea.Tags().items()})
PY
```
实测输出（小规模自检）：`summary` 事件 130+、`graph_def` 存在；标量 8、直方图 31、图像 2、
`token_embedding(+metadata)` 投影张量与 `samples/generation/text_summary` 文本均存在。

---

## 5. 推理与评估

### 5.1 交互式聊天 / 续写（本仓库已训练产物，可直接复制运行）

```bash
# ① SFT 后：交互式问答聊天（推荐，会自动套用 chat 模板组装 <|im_start|>user…<|im_start|>assistant）
python3 scripts/generate.py \
  --checkpoint models/checkpoints/sft_v1_512/final.pt \
  --tokenizer-dir models/tokenizer_v3 \
  --max-new-tokens 90 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2 --seed 5 \
  --chat --interactive

# ② 未 SFT：预训练基座的交互式续写（不加 --chat，直接续写你输入的文本）
python3 scripts/generate.py \
  --checkpoint models/checkpoints/pretrain_v1_512/final.pt \
  --tokenizer-dir models/tokenizer_v3 \
  --max-new-tokens 90 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2 --seed 5 \
  --interactive

# ③ 对照：预训练基座也套 chat 模板（未经指令微调，回答更发散、有时跑题）
python3 scripts/generate.py \
  --checkpoint models/checkpoints/pretrain_v1_512/final.pt \
  --tokenizer-dir models/tokenizer_v3 \
  --max-new-tokens 90 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 --repeat-penalty 1.2 --seed 5 \
  --chat --interactive
```

会话要点：
- 运行后出现 `用户> ` 提示符，输入问题回车即可；空行会跳过；`Ctrl-D`（或 `Ctrl-C`）退出并打印 `[generate] bye`。
- **特殊 token 默认隐藏**：`<|im_end|>` 是 tokenizer_v3 的 eos（id=2），模型在回答结束时输出它并**提前停止生成**（回合正常收尾）。
  想看原始序列加 `--keep-special-tokens`。
- 想换风格：调 `--temperature`（越高越发散）、`--top-k/--top-p`（截断采样范围）、`--repeat-penalty`（抑制复读）、`--seed`（复现同一回答）。
- 非交互用法等价：`--prompt "问题"`（可多次）或 `--prompt-file prompts.txt`，加 `--output-file out.txt` 落盘。
- SFT 产物是 47.9M 小模型（3 epochs 预训练 + 4,900 步 SFT），短问答可用；长对话一致性有限，
  续训/扩 SFT 数据即可提升（见 §3 续训与 §1 快速开始）。

### 5.2 采样参数与评估

```bash
# 采样解码参数：do_sample / temperature / top_k / top_p / repetition_penalty
python scripts/generate.py --checkpoint <ckpt> --tokenizer-dir <tok> \
    --prompt "..." [--prompt-file f.txt | --interactive] [--chat] \
    --max-new-tokens 128 --do-sample --temperature 0.8 --top-k 50 --top-p 0.92 \
    --repeat-penalty 1.2 --seed 0 [--output-file out.txt]

# 验证集 loss / perplexity（--max-rows 限制窗口数以控制耗时；不传则评估全部窗口）
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --bin <bin> \
    --tokenizer-dir <tok> --batch-size 8 --max-rows 512 --output metrics_eval.json
```

### 5.3 吞吐优化（本机实测）

| 配置 | 单步耗时（512/10/8，bs8×ctx512） | 吞吐 |
|---|---|---|
| 默认（fp16 + GradScaler） | 0.229 s | 17.9k tok/s |
| cudnn.benchmark=True / deterministic=False | 0.228 s | 17.9k tok/s（无收益） |
| **`--train_torch_compile True`** | **0.138 s** | **29.7k tok/s（+66%）** |

用法与注意：
```bash
python3 -m minigpt.train.pretrainer --train_torch_compile True ...   # 预训练固定形状，收益最大
```
- `torch.compile` 在**固定序列长度**（预训练窗口恒为 ctx）下收益最大；SFT 的变长 padding 会触发反复重编译，
  故 `sft_trainer` 默认不启用（如需可显式 `--train_torch_compile True` 并配合固定 `max_len`）。
- 启用后 `Trainer` 内部会同时兼容 `DistributedDataParallel` 与 `OptimizedModule`（checkpoint 存取统一走 `_unwrap()`）。
- 计算图中存在 `position_ids.max().item()` 等导致的 graph break（dynamo 会警告），但实测仍显著提速。

### 5.4 续训（resume）注意事项
- checkpoint 内含 `model_state / optimizer_state / scaler_state / rng_state / config`，续训用
  `--paths_last_checkpoint_path <ckpt>` 即可完整恢复优化器与随机状态。
- 已修复：`torch.load(map_location=device)` 会把 RNG 的 `ByteTensor` 一并搬到 GPU，导致
  `set_rng_state` 报 `RNG state must be a torch.ByteTensor`；现在加载时统一 `.cpu()` 修正。

## 6. 测试

```bash
pytest tests/ -q      # 覆盖模型/注意力/数据管线/采样解码/配置/trainer
```

## 7. 环境提示
- RTX 20 系（Turing）不支持 FlashAttention-2：统一 `--model_flash_attn False`（默认）。
- 大词表 tokenizer（如 Qwen2.5 151k）在 6GB 卡上吞吐仅 ~2k tok/s（见根 README 实测表），
  本仓库默认使用 32k 中文 BPE（`models/tokenizer_v3`）；换用其它 tokenizer 只需改
  `--data_tokenizer_dir`，模型词表会自动跟随。

---

## 8. 思考模式（Chain-of-Thought / CoT）

### 8.1 它是什么
"思考模式"不是模型里的一个开关，而是**训练数据 + 训练目标 + 推理协议**三件套：
1. **数据**：SFT 样本的答案里显式包含推理过程，模型才"有得学"；
2. **训练**：只对 assistant 段计算损失（`InstructionDataset` + `-100` 掩码已支持）；
3. **推理**：约定思考段与答案的分隔标记，生成时按标记切分/截断/隐藏。

本仓库采用的协议（`minigpt/model/generation.py`）：
```
<|im_start|>assistant
让我逐步分析：
1. …
2. …
最终答案：<答案><|im_end|>
```

### 8.2 构造 CoT 数据（可验证任务）
```bash
# easy：1~2 位数单步/两步（小模型容量匹配，能真正学会）
python scripts/build_cot_sft.py --profile easy \
    --out-train dataset/sft/sft_cot_easy_zh.jsonl \
    --out-eval  dataset/sft/cot_eval_easy_zh.jsonl --n-train 20000 --n-eval 200

# hard：多位数四则运算 + 应用题（对模型容量要求高）
python scripts/build_cot_sft.py --profile hard \
    --out-train dataset/sft/sft_cot_zh.jsonl \
    --out-eval  dataset/sft/cot_eval_zh.jsonl --n-train 30000 --n-eval 200
```
数据自带 `mix-general`（默认 25%）混入通用指令，避免灾难性遗忘；评测集用不同随机种子与生成器偏移，**与训练题不重合**。

### 8.3 训练 CoT-SFT
```bash
# easy（推荐先跑这个：~15 分钟，可验证"思考模式"确实有效）
python3 -m minigpt.train.sft_trainer \
    --pretrain models/checkpoints/pretrain_v1_512/final.pt \
    --data_tokenizer_dir models/tokenizer_v3 \
    --sft-jsonl dataset/sft/sft_cot_easy_zh.jsonl --data_max_lines 20000 \
    --train_batch_size 8 --train_learning_rate 2e-5 --train_epochs 2 \
    --paths_output_dir models/checkpoints/sft_cot_easy_v1

# hard（在已有对话模型上继续加 CoT）
python3 -m minigpt.train.sft_trainer \
    --pretrain models/checkpoints/sft_v1_512/final.pt \
    --sft-jsonl dataset/sft/sft_cot_zh.jsonl --data_max_lines 30000 \
    --train_batch_size 8 --train_learning_rate 1e-5 --train_epochs 1 \
    --paths_output_dir models/checkpoints/sft_cot_v1
```

### 8.4 推理（思考模式开关）
```bash
python3 scripts/generate.py \
  --checkpoint models/checkpoints/sft_cot_easy_v1/final.pt \
  --tokenizer-dir models/tokenizer_v3 \
  --chat --prompt "请计算 27 ÷ 3 等于多少？" --thinking
```
实测输出：
```
模型: 【思考】
      1. 想 3 乘几等于 27
      2. 3 × 9 = 27，所以商是 9
      【回答】
      9
```
参数：
| 参数 | 默认 | 作用 |
|---|---|---|
| `--thinking` | 关 | 开启思考模式 |
| `--thinking-strategy` | `single` | `single`＝一次生成"思考+答案"（最稳，推荐）；`two-phase`＝先思考到 `最终答案：` 再作答（思考预算可控、适合长推理） |
| `--thinking-max-tokens` | 120 | 思考预算（two-phase 的思考段上限；single 时与 `--max-new-tokens` 相加为总预算） |
| `--hide-thinking` | 关 | 只显示最终答案，思考仍在内部生成（用于产品化输出） |

在 Python 里调用：`minigpt.model.generation.generate_with_thinking(...)`，返回 `thinking / answer / display / full / marker_hit`。

### 8.5 评测（留出集，可复现）
```bash
python3 scripts/eval_thinking.py \
  --checkpoint models/checkpoints/sft_cot_easy_v1/final.pt \
  --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl --n 60 \
  --strategies plain,single,two-phase --repetition-penalty 1.0 \
  --output result.json
```

### 8.6 实测结果（RTX 2060 6GB，60 题留出集，贪心解码）

**v1 基座（67.7M tokens 预训练）**

| 任务档次 | 模型 | plain（直接生成后取答案） | single（单通道 CoT） | two-phase |
|---|---|---|---|---|
| **easy**（1~2 位数单/两步） | 预训练基座 | **0.0%** | 1.7% | 1.7% |
| | CoT-SFT（20k 条 / 2 epochs） | **90.0%** | **90.0%** | **90.0%** |
| **hard**（多位数四则+应用题） | CoT-SFT（30k 条 / 1 epoch） | 1.7% | **3.3%** | 3.3% |

**v2 基座（248M tokens 全量语料，续训 211k 步 ≈ 8.6 亿训练 tokens）**

| 任务档次 | 模型 | plain | single | two-phase |
|---|---|---|---|---|
| **easy** | CoT-SFT v2（60k 条 / 2 epochs） | **100.0%** | **100.0%** | 98.3% |
| **hard** | CoT-SFT v2（80k 条 / 3 epochs） | 1.7% | 3.3% | 3.3% |

同切分 ppl 对比（同一 `dataset/bins/pretrain_v3_full.bin`、`--max-rows 512`）：
v1 `eval_loss=3.6323 / ppl=37.80` → v2 `eval_loss=3.1637 / ppl=23.66`（loss −12.9%，ppl −37.4%）。

**domain 基座（在 v2 上做代码领域增量续训：26,847 步 / 74 分钟 / lr=1e-4 / 混入 15% 通用语料）**

| 任务档次 | 模型 | plain | single | two-phase |
|---|---|---|---|---|
| **easy** | CoT-SFT domain（60k 条 / 2 epochs） | **100.0%** | **100.0%** | 98.3% |
| **hard** | CoT-SFT domain（80k 条 / 3 epochs） | 1.7% | 1.7% | 1.7% |

**双口径 ppl（同一 `--max-rows 512`，`scripts/run_domain_compare.sh` 产出）**

| 评估语料 | v2 基座 | domain 基座 | 变化 |
|---|---|---|---|
| 代码领域 `code_domain.bin` | 4.3919 / ppl 80.79 | **3.3576 / ppl 28.72** | **−64%**（领域适配有效）|
| 通用 `pretrain_v3_full.bin` | **3.1637 / ppl 23.66** | 3.5095 / ppl 33.43 | **+41%**（通用能力退化）|

结论与经验：
- **思考模式本身有效**：easy 任务从 0% → 90%（v1）→ **100%（v2 / domain）**，且推理过程可读、可截断、可隐藏。
- **更强基座 + 更多 CoT 数据直接转化为准确率**：同一切分 ppl 降 37%，easy 准确率从 90% 提到 100%。
- **领域自适应 ≠ 无损**：1 epoch、lr=1e-4、只混 15% 通用语料，能把代码领域 ppl 打掉 64%，同时把通用 ppl
  抬高 41%，下游对话 SFT 的 eval_loss 从 2.7795 升到 2.9087，`samples_domain_chat.txt` 里"你是谁"直接答
  "抱歉，我无法回答这个问题。但我可以告诉你关于计算机程序的语法和编程模型"。**做领域续训必须双口径验收**：
  `bash scripts/run_domain_compare.sh`；想两者兼得就降 lr（1e-5~3e-5）、提高混料比（30%+）、减少步数。
- **容量是硬约束**：hard 任务上模型能学会"分步格式"（SFT 训练 loss ≈ 1.4），但多位数乘加仍会算错，
  说明 47.9M 参数不足以支撑多位数推理；需按根 README「实测训练结果」的路线续训/放大模型，
  或采用 **工具调用范式**（模型只生成算式，如 `27/3`，由 Python 计算结果再回填）。
- **评测陷阱（重要）**：贪心做算术评测时 `repetition_penalty` 必须为 `1.0`。用 `1.2` 会惩罚重复出现的数字，
  使 easy 准确率从 90% 虚假跌到 61.7%（我们踩过并已修正评测口径）。
- **协议改进方向**：把 `最终答案：` 固定为 tokenizer 的特殊 token（并在 config 中声明），可让阶段切分 100% 稳定；
  目前依赖自由生成 + 文本标记匹配，`marker_hit=False` 时会自动回退到直接作答。


---

## 9. 训练看板（实时监控）

### 9.1 浏览器看板（每 5 秒自动刷新）
```bash
python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5
# 打开 http://127.0.0.1:8099
```
页面内容：进度条（step/target）、**loss 走势图（train+eval 双曲线，带坐标轴/网格/悬停显示 step 与数值）**、
**grad_norm 走势图**、速率（s/step、k tok/s）与 ETA、GPU 利用率/显存/温度、最近 checkpoint 列表、日志尾部；
训练重启次数也会显示（自愈守护每次重启 +1）。数据来自训练日志 + checkpoint 文件 + `nvidia-smi`，纯标准库实现，无需额外依赖。

参数：`--log <训练日志> --out-dir <checkpoint 目录> --target-step <目标步数> --tokens-per-step <每步tokens> --refresh <秒>`。

### 9.2 导出损失走势图 PNG（报告/README 用）
```bash
python3 scripts/train_dashboard.py --plot models/checkpoints/loss_curve.png
```
输出含 train/eval loss 双曲线 + 右侧 lr 曲线的 PNG（matplotlib，已装）。

### 9.3 终端看板
```bash
bash scripts/watch_training.sh            # 默认 5 秒刷新
# 或手动循环
while true; do python3 scripts/train_dashboard.py --once; sleep 5; done
```
终端输出示例（实测）：
```
[2026-09-10 13:13:13]  训练看板（每 5s 刷新）
进度: 63999/211000  (30.3%)  epoch=2  重启=0
最新: train 3.3981 | eval 3.3945 | lr 5.70e-04 | grad 0.614
eval损失走势(最近8点): █▆▄▇▅▄▂▁  ↓越低越好
速率: 0.170s/step  24.1k tok/s  ETA 6.9h
GPU: util 85% | mem 5107/6144MB | temp 63C
checkpoints: checkpoint-56000.pth(12:02:27), checkpoint-60000.pth(12:58:13), checkpoint-64000.pth(13:09:44)
```

### 9.4 与 TensorBoard 的分工
- **本看板**：轻量、只看关键训练信号（进度/损失/速率/ETA/GPU），适合盯盘。
- **TensorBoard**（完整指标：直方图/图像/模型图/嵌入投影/文本，见 §4）：
  ```bash
  tensorboard --logdir models/checkpoints/pretrain_v2_full/tensorboard --port 6006
  ```
