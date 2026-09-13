# scripts/ 目录地图

脚本按**角色**分目录；**生产链路（正在跑的那条链）全部留在顶层**，其余按用途归类。
每个脚本的详细说明只在 `skills/minigpt-train/SKILL.md` 与各 reference 里写一次，这里只做索引。

```
scripts/
├── （顶层）        生产链路 + 各阶段入口（训练/下游/评测/报告/自检）
├── data/           语料与数据的下载、清洗、配比、建 bin、体检
├── tools/          环境与产物校验、资源估算
└── experiments/    领域续训 / 容量消融等实验配方（不在主链路里）
```

## 顶层：生产链路与入口

| 脚本 | 做什么 | 用在哪 |
|---|---|---|
| `train_pretrain_resilient.sh` | 自愈长训：崩溃/被杀自动从最近 checkpoint 续训 | 训练（**正在运行**） |
| `checkpoint_janitor.sh` | 每个目录只留最近 N 个 checkpoint，防磁盘写满 | 训练（**正在运行**） |
| `train_dashboard.py` | 看板：网页 8099 / `--once` / `--plot` | 监控（**正在运行**） |
| `watch_v5_chain.sh` | v5 接力：等 `final.pt` → 跑下游；并二级监督训练守护 | 交付（**正在运行**） |
| `resume_v5_chain.sh` | 一键恢复全链路（幂等）；**机器重启后必须跑它** | 故障恢复 |
| `run_v5_downstream.sh` | v5 下游：SFT → CoT easy → CoT hard → 评测 → 报告 → 自检 | 交付 |
| `run_downstream.sh` / `run_downstream_evals.sh` | 通用下游 / 只跑评测出样（`pipeline.sh` 用） | 通用流水线 |
| `wait_and_run_downstream.sh` | 通用接力：等 `$OUT_DIR/final.pt` → 通用下游 | `pipeline.sh` |
| `pretrain_start.sh` | 最简启动器（`nohup` 直接起 trainer，无自愈） | 首次/DDP 冒烟 |
| `evaluate_pretrain.py` | 语言建模 loss / ppl（默认 `--split val`，与训练同口径） | 评测 |
| `eval_thinking.py` | 思考模式准确率 **+ 难度分档**（`by_difficulty`） | 评测 |
| `chat_probe.py` | 51 场景对话质检（greedy + sampled） | 评测 |
| `generate.py` | 推理 CLI（`--chat` / `--thinking`） | 人工体验 |
| `report_training.py` | 汇总全部 run/评测 → Markdown 报告（曲线/噪声/吞吐/ETA、分档、样例、历史基线） | 交付 |
| `verify_v5_delivery.py` | 交付自检（40+ 项，缺件 exit 1；`--allow-partial` 看进度） | 交付 |
| `watch_training.sh` | 终端版看板（SSH 用；不传参自动选最新 run） | 监控 |

## data/：语料与数据

| 脚本 | 做什么 |
|---|---|
| `download_data.sh` | 下载原始数据集 |
| `parquet_to_jsonl.py` | parquet → jsonl（`--mix-by chars`、`--shards`、footer 校验） |
| `mix_corpora.py` | 按**字符**配比混合多份 jsonl + manifest |
| `clean_long_docs.py` | 长文档结构化切块 + LaTeX/参考文献过滤 |
| `build_pretrain_bin.py` | jsonl → `.bin` + `.meta.json`（`--nproc` 并行，与串行逐字节一致） |
| `train_tokenizer.py` | 训练分词器（默认 `models/tokenizer_v3`） |
| `build_cot_sft.py` | 生成 CoT 训练/留出集（`--easy-max 50`，留出集零重合） |
| `audit_dataset.py` | 数据集体检：`.bin↔meta↔分词器` 对账、切分预览、SFT 泄漏、CoT 重合、token 预算 |
| `audit_sft_lengths.py` | SFT 真实规模：监督 token、截断路径分布、零监督样本 |

## tools/：校验与估算

| 脚本 | 做什么 |
|---|---|
| `check_env.py` | 依赖 / GPU / 磁盘 / 语料 / 服务体检（`preflight.sh` 的细项实现） |
| `estimate_resources.py` | 参数量、显存、单步时间与步数预算估算 |
| `validate_pretrain.py` | 预训练产物自洽性校验（config/权重/分词器） |
| `validate_ddp.py` | DDP 行为校验（多卡前先跑） |

## experiments/：实验配方（**不在交付链路里**）

`run_code_domain.sh`（代码领域续训）、`run_domain_compare.sh`（通用 vs 领域同切分对比）、
`run_capacity_ablation.sh`（512/10/8 vs 768/12/12 同 token 预算消融）。
三者都会调用顶层的 `run_downstream.sh` / `report_training.py`，属于"按需实验"，不影响主链路。

## 判断某个脚本还有没有在用

```bash
# 谁引用了它（文档/脚本/测试）
grep -rn "scripts/data/audit_dataset.py" --include=*.md --include=*.sh --include=*.py .
# 文档里的示例命令是否仍然有效（含 flag 是否为目标脚本所认识）
python3 -m pytest tests/test_doc_commands.py -q
```

`tests/test_doc_commands.py` 会拦住三类漂移：文档命令指向不存在的脚本、用了目标脚本不认识的
flag、以及"任何脚本在任何文档里都没被提到"（隐形脚本）。挪动脚本后跑它即可确认没有漏改。
