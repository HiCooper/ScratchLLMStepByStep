# AGENTS.md — 给自动化 agent 的仓库入口

本仓库是「从零手写并训练 MiniGPT」的教程 + 生产级训练工程。**你要训练/微调/评测 MiniGPT，先读这个文件**，
再按 `skills/minigpt-train/SKILL.md` 执行（DSH 会扫描 `skills/`、`.dsh/skills/`、`.agents/skills/`）。

## 最短路径（无需人工执行脚本）

```bash
cd "$(git rev-parse --show-toplevel)"

# 1) 环境体检（只读）：依赖/GPU/显存/磁盘/语料/分词器/服务/单测，输出 models/checkpoints/preflight.json
bash skills/minigpt-train/scripts/preflight.sh

# 2) 一键流水线（幂等；--smoke 约 5 分钟验证链路，--full 完整训练+SFT+CoT+评测）
bash skills/minigpt-train/scripts/pipeline.sh --smoke
bash skills/minigpt-train/scripts/pipeline.sh --full
```

## 环境自适应（先体检，再按预设跑）

`preflight.sh` 会检测硬件并打印/写入推荐预设（`models/checkpoints/preflight.json: preset`），
`pipeline.sh` 自动消费它；也可 `--preset cpu|gpu-tiny|gpu-small|gpu-mid|gpu-large|multi-gpu` 手动覆盖，
并用 `--hours <预算>` 自动换算目标步数：

| 环境 | 预设 | 模型 | 关键参数 | 预期 |
|---|---|---|---|---|
| 无 GPU（含 MPS 回退） | `cpu` | 128/2/4 | ctx128, fp32, 不开 compile, 2 万行小语料 | 0.12–0.17k tok/s，**仅机制验证** |
| 单卡 <5.5GB | `gpu-tiny` | 384/8/8 | bs4, fp16, compile | 10–18k tok/s |
| 单卡 5.5–8GB（本机） | `gpu-small` | 512/10/8 | bs8, ctx512, fp16, compile | 22–30k tok/s |
| 单卡 ≥8GB | `gpu-mid`/`gpu-large` | 512/10/8 或 768/12/12 | 更大 ctx/batch | 30–60k+ tok/s |
| ≥2 张 GPU | `multi-gpu` | 按单卡显存 | `NPROC=N`（torchrun DDP），lr×√N | ≈单卡×N×0.85 |

CPU 环境下 agent 的正确做法：跑 `--smoke`/`--full --preset cpu --hours 1` 验证链路，明确告知用户「无 GPU 只能验证机制」，
建议改用 CUDA 机器继续；多卡环境：`--nproc N` 或 `NPROC=N`，有效 batch=batch×N×accum，step 数按有效 batch 重算。

## 训练相关事实（决策前必读）

- 默认配置：**32k 词表（`models/tokenizer_v3`）+ 512/10/8 + ctx512 + fp16 + `torch.compile`** ≈ 22–30k tok/s（RTX2060 6GB 实测）。
- 切换词表只改 `--data_tokenizer_dir`；**词表 >65535 时 `.bin` 必须 uint32**（`build_pretrain_bin.py` 自动处理并写 `.meta.json`）。
- 语料：`dataset/pretrain_t2t_mini.jsonl`（127 万行）→ `dataset/bins/pretrain_v3_full.bin`（2.475 亿 tokens）。
- 训练入口：`python3 -m minigpt.train.pretrainer`（预训练）/ `python3 -m minigpt.train.sft_trainer`（SFT）；
  SFT/CoT 数据生成：`scripts/build_cot_sft.py --profile easy|hard`。
- 评测：`scripts/evaluate_pretrain.py`（loss/perplexity，注意 `--max-rows`）、`scripts/eval_thinking.py`（思考模式，`--repetition-penalty 1.0`）。
- 推理：`scripts/generate.py --chat [--thinking --thinking-strategy single]`。
- 领域语料（parquet，如 `dataset/IndustryCorpus2_*`）：`scripts/parquet_to_jsonl.py` 转换（可混入通用语料防遗忘）→ 建 bin → 增量续训；**不要**把预训练语料直接拿去做 SFT（见 SKILL.md §7.5）。

## 硬性约定

1. 长任务必须 `setsid nohup ... > <log> 2>&1 < /dev/null &`，禁止前台阻塞；用 `pgrep -af` / 日志 / `scripts/train_dashboard.py --once` 验证存活与进度。
2. 长训练必须同时运行 `scripts/checkpoint_janitor.sh`（每目录只留最近 N 个 checkpoint，单个 ~585MB），避免磁盘写满。
3. 续训用 `--paths_last_checkpoint_path <latest checkpoint-*.pth>`；推荐直接用 `scripts/train_pretrain_resilient.sh`（崩溃自动续跑）。
4. 不要覆盖用户数据；不要提交 `models/`、`dataset/`（已在 `.gitignore`）；代码改动跑 `pytest tests/ -q` 后提交。
5. 汇报格式：进度(step/总步数)、eval_loss 变化、吞吐与 ETA、产物路径、关键样例。

## 进一步阅读

- `skills/minigpt-train/SKILL.md`：完整流程、配置决策树、监控与汇报模板。
- `skills/minigpt-train/references/training-recipes.md`：超参、吞吐表、预算↔tokens 换算、多卡与续训。
- `skills/minigpt-train/references/troubleshooting.md`：OOM、uint16 溢出、RNG 续训、速度骤降、磁盘写满等排障。
- `minigpt/README.md`：指标面板（TensorBoard 标量/直方图/图像/模型图/嵌入投影）、看板、思考模式与实测结果。
