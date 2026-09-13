# 数据分布：事实基线、配方与验收

> 这份文档回答一个具体问题：**"训练语料够不够、分布对不对"**，以及**怎么在开训前就查出来**。
> 配套工具：`scripts/data/audit_dataset.py`（体检）、`scripts/data/mix_corpora.py`（配比）、
> `scripts/data/parquet_to_jsonl.py`（领域语料入库）。配方命令见 `SKILL.md §7.6`。

## 1. 为什么单列一份文档

本仓库历史上翻车三次，**没有一次是"数据量不够"，全是"分布不对"或"取错了那部分"**：

| # | 事故 | 根因 | 后果 |
|---|---|---|---|
| 1 | 通用语料按领域排序，取前缀只拿到一个领域 | `pretrain_t2t_mini.jsonl` 前 120 万行中文短文本、末尾约 7 万行英文长文本 | 用 `--max-lines` 训的模型评全量 val（含英文尾部）时 ppl 落差巨大，被误判为"模型差" |
| 2 | 领域续训的"防遗忘"重放几乎为零 | 混料池用"顺序读前 N 行"，100% 落在中文头部 | 通用 ppl 23.66 → 33.43（**+41%**），下游对话质量崩坏 |
| 3 | "混 15%"实际只有 1.18% | 混料比例按**条数**算，而两侧条目长度差 14.8 倍 | 同上；"提高混料比"这个结论当时是建立在错误的度量上 |

## 2. 事实基线

**各语料的行数 / 字符 / tokens 与来源见 `dataset/README.md`（唯一详述处）。** 这里只列与"分布"直接相关的实测结论：

| 结论 | 实测 |
|---|---|
| 通用语料的语言配比（**token 口径**） | 中文 **84.5%** / 英文 15.5%（按文档数会高估中文，因为英文文档更长） |
| 长度分布对 ctx512 是健康的 | 56% 的 token 落在 <500 字符的短文；>2k 只占 8.5% → 每窗平均跨 2.6 篇文档 |
| 语料**按领域排序** | 前 120 万行 80–100% 中文，末尾约 7 万行英文为主（尾部 5000 行 100% 英文） |
| 结构化成分（cosmopedia） | 中文 98%，p50 2,285 字符，`data_format` = 中学/大学教材、wikihow、故事 |
| 长文档（英文数学）的密度 | 清洗前每百万字符仅 **26** 个不同样本（一篇占 25–35 窗）；清洗切块后 **654** |

**排序性**是这份语料最关键的性质：任何"读前 N 行 / 取前缀"的做法都只会采到单一领域。

## 3. 核心发现：配比口径错了

历史 `code_corpus.jsonl` 自称"混入 15% 通用语料"，实测：

```
按条数 = 14,232 / 94,884 = 15.0%     ← 文档里写的
按字符 = 2.7M / 231.4M   =  1.18%    ← 模型真正看到的复放量
平均长度：混料 192 字符 vs 领域 2,835 字符（差 14.8 倍）
```

**结论**：当年"15% 混料没防住灾难性遗忘"的结论，其实是"**1.2% 混料**没防住"。
这也意味着单纯把比例调到 30%（同口径）仍然不够——必须先把口径改成字符/token。

## 4. 解决方案（三条规则）

1. **整文件均匀抽样**，禁止前缀。工具内建蓄水池抽样；`--shuffle-files` 防止"预算截断 = 领域截断"。
2. **配比按字符（≈token）**，且均值必须用**全文件扫描的精确值**。
   `--mix-ratio` 默认已是 chars；`mix_corpora.py` 的 manifest 同时记录探针值与精确值。
3. **长文档截断而非丢弃**（`--max-chars 0 --truncate-chars N`），并对每个产物打印组成画像
   （来源分片 / 语种 / 长度分档），让"取到了哪部分"可见。

## 5. 产出的资产与关键结论

**清单（路径 / 规模 / tokens）见 `dataset/README.md`。** 与"分布"有关的只有两点：

- **复放量对比**：历史领域续训的有效复放是 **1.2%**（按字符）；本次 70:30 与 finance 都是 **30.0%**，
  提高约 25 倍——这才是"防遗忘"真正起作用的原因。
- **构建方式**：`build_pretrain_bin.py build ... --nproc 6` 按行首切分并行分词，
  与单进程串行产物**逐字节一致**（真实语料 sha256 实测相同），3.24G 字符从 65 分钟降到 ~12 分钟；
  `--nproc>1` 与 `--max-lines` 互斥。

## 6. 验收清单（不达标不开训）

```bash
python scripts/data/audit_dataset.py       # 有 BLOCKER 直接返回 1
```

> **必须跑完整模式，不要用 `--quick`**：`--quick` 会跳过语料扫描。
> 实测教训：`pretrain_t2t.jsonl` 第 8,441,338 行的开引号被 0x02 控制符替换，
> 建 bin 跑到 95%、65 分钟后崩溃且**没写出 meta**；而 `audit_dataset.py`（完整模式）
> 一行就能报出 `JSON 非法 1`。现在 `tokenize_jsonl_to_bin` 会跳过脏行并写进
> `meta.bad_lines`，但**先体检仍然便宜得多**。

- [ ] 每个来源 manifest 的 `actual_share` ≈ 目标权重（±1%）；
- [ ] 产物打印的"语种占比 / 长度分档"符合预期（例如中文领域语料不应出现英文为主）；
- [ ] `.bin` 的 `meta.tokens` 与文件字节数精确对账（`audit_dataset.py` 会校验）；
- [ ] 训练后**双口径**：`bash scripts/experiments/run_domain_compare.sh` —— 领域 ppl 明显下降 **且** 通用 ppl 上升 ≤5%。

## 7. 语料选型（实测踩到的事实）

- **`IndustryCorpus2` 不是通用语料库**：`BAAI/IndustryCorpus2` 共 30 个行业子集 / 1718 GiB / 3206 文件，
  没有"通用"子集。想要通用基座请用 `gongjy/minimind_dataset:pretrain_t2t.jsonl`（7.9 GB，与旧 mini 同源同分布）。
- **数学语料是英文的**：`IndustryCorpus2_mathematics_statistics` 70 GiB 中 `chinese/*` 只有 3 个分片（0.34 GiB），
  `english/high` 有 89 个。想要中文数学/领域语料必须换源。
- 中文占比高的可选子集（`chinese/high`）：`subject_education_education` 18.6 GiB / 23 文件、
  `medicine_health_psychology_traditional_chinese_medicine` 9.3 GiB / 12 文件、
  `current_affairs_government_administration` 8.7 GiB / 11 文件、`finance_economics` 8.3 GiB / 11 文件。

## 8. 下次训练怎么用（可直接抄的命令）

```bash
# ① 通用基座（从零预训练）：全量 846.9 万行 → 约 16 亿 tokens
#    512/10/8 单遍 ≈ 33× 参数（已超 Chinchilla 20×），不再需要靠多 epoch 凑 token
python3 -m minigpt.train.pretrainer \
  --paths_output_dir models/checkpoints/pretrain_v4 \
  --data_tokenized_bin dataset/bins/pretrain_v4_full.bin \
  --train_batch_size 8 --train_learning_rate 6e-4 --train_warmup_steps 2000 \
  --train_epochs 1 --train_eval_steps 2000 --train_save_steps 4000 \
  --train_torch_compile True

# ② 领域增量续训（以代码为例，领域:通用 = 70:30，通用侧全文件均匀抽 → 覆盖英文尾部）
#    必须 --train_reset_step True；lr 用基座的 1/5~1/10
python3 -m minigpt.train.pretrainer \
  --paths_output_dir models/checkpoints/domain_code70 \
  --data_tokenized_bin dataset/bins/domain_code70_general30.bin \
  --paths_last_checkpoint_path models/checkpoints/pretrain_v4/final.pt \
  --train_reset_step True --train_epochs 2 --train_batch_size 8 \
  --train_learning_rate 1e-4 --train_eval_steps 1000 --train_save_steps 4000 \
  --train_torch_compile True

# ③ 若 ② 的通用 ppl 上升超过 5%：用 50:50 档退火一小段（低 lr、少步数）再验收
python3 -m minigpt.train.pretrainer \
  --paths_output_dir models/checkpoints/domain_code50_anneal \
  --data_tokenized_bin dataset/bins/domain_code50_general50.bin \
  --paths_last_checkpoint_path models/checkpoints/domain_code70/final.pt \
  --train_reset_step True --train_epochs 1 --train_batch_size 8 \
  --train_learning_rate 2e-5 --train_eval_steps 1000 --train_save_steps 4000

# ④ 双口径验收（缺一不可）
bash scripts/experiments/run_domain_compare.sh
```

中文领域同理，把 `--data_tokenized_bin` 换成 `dataset/bins/finance_corpus.bin`
（1.279 亿 tokens，中文 100%，已含 30% 通用复放）。

## 9. SFT/CoT 侧的分布问题（不合理 3）

CoT 阶段的问题不是"数据量不够"，而是**题面空间与训练样本数不匹配**、以及**评测口径看不出问题**。

### 9.1 题面空间 ↔ 训练样本数（实测）

`build_cot_sft.py` 的 easy 档用 `EASY_MAX` 控制操作数上界，唯一题面数如下：

| `--easy-max` | 唯一题面数 | 60k 训练集的平均重复次数 |
|---|---|---|
| 9（默认，历史用法） | **1,063** | **56×** ← 纯记忆 |
| 20 | 9,603 | 6.3× |
| 50 | 40,390 | 1.5× |
| 100 | 76,055 | 0.8×（训练集不再覆盖题面空间） |

**规则**：让 `唯一题面数 ≥ 训练样本数 / 3`（重复 ≤3×）。60k 训练集对应 `--easy-max 40~50`。
默认的 9 会让"100% 准确率"完全测不出泛化——这正是仓库里 `*_disjoint.jsonl` 要修的问题。
（数据集侧已修：链路用 `build_cot_sft.py --profile easy --easy-max 50` 生成训练集
`dataset/sft/sft_cot_easy_em50_60k.jsonl` 与零重合留出集 `cot_eval_easy_em50_disjoint.jsonl`。
表里的"唯一题面数"是**生成器题池**的规模，不等于磁盘上某个具体文件的唯一题面数——
旧文件被重生成过，直接数文件里的唯一题面会得到不同数字。）

### 9.2 评测必须分档

单一平均准确率会把"简单题全对 + 难题全错"平均成一个看不出问题的数字。
`eval_thinking.py` 现在按**规模×题型**分档输出（`by_difficulty`）：

```
档位             题数         plain      single
mixed_small        12         91.7%       75.0%
sub_big            14         21.4%       28.6%   ← 全档低于 50%，优先怀疑容量/数据分布
```

判读规则：某档题数 ≥5 且所有策略都 <50% → **先别加步数**，那是容量墙或该档数据缺失
（47.9M 参数在多位数乘加上就是这种情况：加同分布数据无用，要么扩模型，要么改成
"模型只出算式、Python 结算"的工具调用范式）。

### 9.3 已核对无问题的部分

- `sft_data_zh.jsonl` 的前 10 万行**没有**前缀偏差（文件已打乱：首尾 1 万行的 `type`/`data_source`
  分布一致，100% 中文）；
- `build_cot_sft.py` 的 `load_general()` 是"整文件读入后 shuffle 再取 n"，不受排序影响；
- CoT 训练集与 disjoint 留出集实测重合 **0/200**（`audit_dataset.py` 每次都会复核）。

## 10. 仍未解决

- 通用语料是**别人打包的混合集**，无法按领域再做配额（只有一个 jsonl，没有领域标签）。
  若要做"按领域分层抽样"，需要自带领域标签的语料（如 IndustryCorpus2 的分目录结构）。
- 语料内的**语言比例**只能观测、不能控制（除非换语料或做语言过滤）。
- `code_corpus.dedup.jsonl` 里仍嵌着历史遗留的 1.2% 通用混料；由它派生的 70:30 产物实际通用占比
  约 31%，在 ±1% 容差外一点，需要严格 70:30 时应先重建无混料的纯领域语料。
