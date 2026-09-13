# AGENTS.md — 给自动化 agent 的仓库入口

本仓库是「从零手写并训练 MiniGPT」的教程 + 生产级训练工程。**你要训练/微调/评测 MiniGPT，先读这个文件**，
再按 `skills/minigpt-train/SKILL.md` 执行（DSH 扫描 `skills/`、`.dsh/skills/`、`.agents/skills/`；本仓库使用 `skills/`，不要写成 `.skills/`，否则 skill 不会被加载）。

## 最短路径（无需人工执行脚本）

```bash
cd "$(git rev-parse --show-toplevel)"

# 1) 环境体检（只读）：依赖/GPU/显存/磁盘/语料/分词器/服务/单测，输出 models/checkpoints/preflight.json
bash skills/minigpt-train/scripts/preflight.sh

# 1.5) 数据集体检（只读，建议每次改语料后/开训前跑）：jsonl 内容质量、.bin↔meta↔分词器对账、
#      训练/验证切分预览、SFT 泄漏与 CoT 零重合、parquet 下载完整性、token 预算。有 BLOCKER 返回 1
python scripts/data/audit_dataset.py            # --quick 跳过 1.2GB 语料扫描；--json 落盘报告

# 2) 一键流水线（幂等；--smoke 约 5 分钟验证链路，--full 完整训练+SFT+CoT+评测）
bash skills/minigpt-train/scripts/pipeline.sh --smoke
bash skills/minigpt-train/scripts/pipeline.sh --full
```

## 环境自适应（先体检，再按预设跑）

`preflight.sh` 按硬件自动选预设并写入 `models/checkpoints/preflight.json: preset`（`pipeline.sh` 自动消费，
也可 `--preset cpu|gpu-tiny|gpu-small|gpu-mid|gpu-large|multi-gpu` 覆盖）。
**预设矩阵、判定条件、吞吐表与预算换算是 `skills/minigpt-train/references/training-recipes.md §0` 的事，这里不重复。**

两条最容易做错的：

- **无 GPU（含 MPS 回退）**：只能做机制验证（实测 0.12–0.17k tok/s）。跑
  `pipeline.sh --full --preset cpu --hours 1` 验证链路即可，并明确告诉用户"无 GPU 只能验证机制，建议换 CUDA 机器"。
- **多卡**：`--nproc N` 或 `NPROC=N`；有效 batch = batch×N×accum，**步数要按有效 batch 重算**，lr×√N。

## 训练相关事实（决策前必读）

- **默认模型**：32k 词表（`models/tokenizer_v3`）+ 512/10/8 + ctx512 + fp16 + 嵌入共享 = 47.9M 参数。
  换词表只改 `--data_tokenizer_dir`；**词表 >65535 时 `.bin` 必须 uint32**（管线自动处理并写 `.meta.json`）。
- **数据**：基座基线 `dataset/bins/pretrain_v5_full.bin`（3.24G 字符 / 16.92 亿 tokens / 通用 80% + 结构化 20%）。
  **训练只吃 `.bin`**（`TokenBinDataset` memmap），不读 jsonl；**各阶段数据源、规模与生成命令见 `dataset/README.md`**。
- **入口**：`python3 -m minigpt.train.pretrainer` / `sft_trainer`；推理 `scripts/generate.py --chat [--thinking]`；
  评测 `scripts/evaluate_pretrain.py`（默认 `--split val`）、`scripts/eval_thinking.py`（含难度分档）。
  完整流程与监控见 `skills/minigpt-train/SKILL.md`。
- **领域续训**：三条抽样/配比规则 + 双口径验收见 `SKILL.md §7.5`；原理与实测证据见
  `references/data-distribution.md`。**不要把预训练语料直接拿去做 SFT**（会训成续写器）。

## 硬性约定

1. 长任务必须 `setsid nohup ... > <log> 2>&1 < /dev/null &`，禁止前台阻塞；用 `pgrep -af` / 日志 / `scripts/train_dashboard.py --once` 验证存活与进度。
2. 长训练必须同时运行 `scripts/checkpoint_janitor.sh`（每目录只留最近 N 个 checkpoint，单个 ~585MB），避免磁盘写满。
3. 续训用 `--paths_last_checkpoint_path <latest checkpoint-*.pth>`；推荐直接用 `scripts/train_pretrain_resilient.sh`（崩溃自动续跑）。
4. 不要覆盖用户数据；不要提交 `models/`、`dataset/`（已在 `.gitignore`）；代码改动跑 `pytest tests/ -q` 后提交。
5. **建 bin / 开训前先跑完整体检** `python scripts/data/audit_dataset.py`（别加 `--quick`，它会跳过语料扫描）：
   实测教训是一次脏行让建 bin 在 95%、65 分钟后崩溃且没落 meta，而体检一行就报出来了。
   `tokenize_jsonl_to_bin` 现在会跳过脏行并写进 `meta.bad_lines`，但先体检便宜得多。
6. 汇报格式：进度(step/总步数)、eval_loss 变化、吞吐与 ETA、产物路径、关键样例。

## 进一步阅读（**每个主题只有一处详述**，其余地方只给链接）

| 想知道什么 | 看哪里 |
|---|---|
| agent 怎么一步步跑、监控什么、怎么汇报 | `skills/minigpt-train/SKILL.md` |
| 预设矩阵 / 吞吐 / 预算换算 / 评测协议 | `skills/minigpt-train/references/training-recipes.md` |
| 各阶段数据源、规模、生成命令 | `dataset/README.md` |
| 数据分布的原理、实测证据、验收标准 | `skills/minigpt-train/references/data-distribution.md` |
| 出故障了（OOM / 慢 / 续训 / 磁盘） | `skills/minigpt-train/references/troubleshooting.md` |
| 包 API、配置项、训练产物、指标面板 | `minigpt/README.md` |
| 项目是什么、怎么装、实测结果 | 根 `README.md` |
