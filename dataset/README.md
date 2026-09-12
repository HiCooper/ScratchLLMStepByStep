# dataset —— 各训练阶段的数据源

> 本目录**不进版本库**（`.gitignore`），只有这份 README 被跟踪（见文末说明）。
> 语料可重新下载/重建，命令都写在下面对应行里。
> 开训前务必跑一次 **完整体检**：`python scripts/audit_dataset.py`（有 BLOCKER 会返回 1）。

## 目录约定

```
dataset/
├── pretrain_t2t.jsonl              通用语料源（唯一，不再有 mini/全量双份）
├── chinese_cosmopedia/             结构化语料源（parquet，下载了 8/64 分片）
├── IndustryCorpus2_*/              领域语料源（parquet，保留以便换参数重切）
├── bins/                           *.bin + *.meta.json   ★ 训练唯一入口
├── domain/                         parquet → jsonl 的领域语料（中间产物）
├── mixed/                          按配比混合后的语料 + manifest（中间产物）
├── sft/                            SFT / CoT 训练集与留出集
├── debug/                          非训练数据：链路验证/冒烟用的小样本
└── smsspam.txt                     notebook 12 分类微调用，与本模型训练无关
```

命名规则：`*.bin` 一律带同名 `.meta.json`（dtype / 行数 / tokens / 词表 / 脏行数）；
`mixed/*` 一律带 `.manifest.json`（每个来源的权重、扫描行数、抽中条数、字符数、**实际占比**）。

### 数据怎么流动（**训练不读 jsonl**）

```
源语料 parquet/jsonl
   │  mix_corpora.py / parquet_to_jsonl.py / clean_long_docs.py   ← 决定"用哪部分、什么配比"
   ▼
dataset/{mixed,domain}/*.jsonl  +  manifest/report        ← 中间产物（可复现、可复核）
   │  build_pretrain_bin.py build --nproc 6                ← 一次性 tokenize（uint16/uint32 + meta）
   ▼
dataset/bins/*.bin  +  *.meta.json                        ← ★ 训练唯一入口
   │  python -m minigpt.train.pretrainer --data_tokenized_bin dataset/bins/xxx.bin
   ▼
TokenBinDataset(memmap) → Trainer
```

`pretrainer.py` / `sft_trainer.py` 都只通过 `TokenBinDataset` 读 `.bin`，**没有任何直接读 jsonl 的路径**。
jsonl 保留的意义：换 tokenizer、换 ctx、或想改配比时，重新 tokenize 即可，不必重新下载/清洗。

## 各训练阶段的数据源

| 训练阶段 | 直接引用的训练数据 | 原始数据集 | 数据规模 |
|---|---|---|---|
| **① 基座预训练** | `bins/pretrain_v5_full.bin` | `gongjy/minimind_dataset` → `pretrain_t2t.jsonl`（**80%** 字符）<br>`opencsg/chinese-cosmopedia` → `data/*.parquet`（**20%** 字符） | **1,691,885,203 tokens**<br>3.244G 字符 / 6,974,328 条 |
| **② 领域续训·代码** | `bins/domain_code70_general30.bin`<br>退火档 `bins/domain_code50_general50.bin` | `BAAI/IndustryCorpus2` → `computer_programming_code/{chinese,english}/high`（70%）<br>+ `gongjy/minimind_dataset`（30%，防遗忘复放） | 1.014 亿 tokens<br>退火档 1.029 亿 |
| **② 领域续训·中文金融** | `bins/finance_corpus.bin` | `BAAI/IndustryCorpus2` → `finance_economics/chinese/high`（70%）<br>+ `gongjy/minimind_dataset`（30%） | 1.279 亿 tokens |
| **② 长文档续训·英文数学** | `bins/math_clean.bin` | `BAAI/IndustryCorpus2_mathematics_statistics` → `english/high`(85 分片) + `chinese/high`(1) | 1.206 亿 tokens<br>（清洗切块后 196,090 条） |
| **③ SFT 指令微调** | `sft/sft_data_zh.jsonl` | `BelleGroup/train_3.5M_CN` | 10 万条 / 21.2M 监督 token |
| **④ CoT 训练** | `sft/sft_cot_easy_disjoint60k.jsonl`<br>`sft/sft_cot_hard_80k.jsonl` | 合成：`scripts/build_cot_sft.py`（算术题面）<br>+ 25% 混入 `BelleGroup` 通用指令 | 6 万条 + 8 万条 |
| **⑤ 评测·CoT** | `sft/cot_eval_easy_disjoint.jsonl`<br>`sft/cot_eval_hard_disjoint.jsonl` | 合成，与训练集**零重合**（实测 0/200） | 各 200 条 |
| **⑤ 评测·对话** | `../eval_sets/chat_scenarios_zh.jsonl` | 本仓库手工构造 | 51 场景 |
| **⑤ 评测·语言建模** | 上表各 `.bin` 的 `--split val` 切分 | 同各自来源 | 443–513 窗（22–51 万 tokens） |
| 链路验证（**非训练**） | `debug/pretrain_head.bin` | `pretrain_t2t.jsonl` 前 2000 行 | 2,000 行 / 280,992 tokens |
| notebook 12（**与本模型训练无关**） | `smsspam.txt` | SMS Spam Collection | 5,574 行 |

- ② 的三行是**互斥候选**：按是否有该领域的交付需求选一个，不是串联。
- `bins/pretrain_v4_full.bin`（纯通用、同 token 预算）保留作 **A/B 对照组**，用来隔离"加结构化成分"的效果；
  它不是训练必需项，`audit_dataset.py` 已把它排除在可用语料统计之外。

## 数据是怎么来的（可复现命令）

```bash
# 通用语料（7.9GB）
bash scripts/download_data.sh

# 结构化语料：下载 8/64 分片 → 转 jsonl（每片均摊 81M 字符）
python - <<'EOF'   # 取分片路径
from modelscope.hub.api import HubApi
fs=[f for f in HubApi().get_dataset_files('opencsg/chinese-cosmopedia', revision='master', page_size=100) if f.get('Size') and f['Path'].startswith('data/')]
print(' '.join(sorted(f['Path'] for f in fs)[:8]))
EOF
modelscope download --dataset opencsg/chinese-cosmopedia --local_dir dataset/chinese_cosmopedia --include <上面 8 个路径>
python scripts/parquet_to_jsonl.py --src dataset/chinese_cosmopedia \
  --out dataset/domain/structured_cosmopedia.jsonl --min-chars 300 --max-chars 0 \
  --truncate-chars 16000 --target-chars 650000000 --shuffle-files --seed 20260912

# 基座 v5：通用 80% + 结构化 20%（字符级配额，manifest 记录实际占比）
python scripts/mix_corpora.py --source dataset/pretrain_t2t.jsonl:0.80 \
  --source dataset/domain/structured_cosmopedia.jsonl:0.20 \
  --target-chars 3244000000 --seed 20260912 \
  --out dataset/mixed/pretrain_v5_general80_structured20.jsonl
python scripts/build_pretrain_bin.py build \
  --corpus-jsonl dataset/mixed/pretrain_v5_general80_structured20.jsonl \
  --tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/pretrain_v5_full.bin --nproc 6

# 领域语料（以代码为例）
python scripts/parquet_to_jsonl.py --src dataset/IndustryCorpus2_computer_programming_code_high \
  --out dataset/domain/code_corpus.jsonl --min-chars 300 --max-chars 8000 \
  --max-line-length 500 --min-quality 3.0 \
  --mix-jsonl dataset/pretrain_t2t.jsonl --mix-ratio 0.30 --mix-limit 300000

# 长文档（论文/教材）必须先清洗再进池
python scripts/clean_long_docs.py --src dataset/IndustryCorpus2_mathematics_statistics_high \
  --out dataset/domain/math_clean.jsonl --chunk-chars 2000 --target-chars 300000000 \
  --shuffle-files --seed 20260912
```

## 保留 / 删除规则

- **保留原始源**（parquet/jsonl）：它们是已保留产物的来源，删了就无法换参数重切
  → `pretrain_t2t.jsonl`、`chinese_cosmopedia/`、三个 `IndustryCorpus2_*/`
- **派生产物要标清楚**：`mixed/*` 由其它语料派生，统计 token 预算时**不能**与来源相加（`audit_dataset.py`
  已把它们排除在"可用语料"之外）。

## 抽样与配比（**三条铁律的细则见参考文档**）

⚠️ **本语料按领域排序**（前 120 万行中文短文、末尾几十万行英文为主），
所以任何"读前 N 行 / 取前缀"的做法都只会拿到一个领域——这是本仓库踩过三次的坑。

三条铁律（均匀抽样 / 按字符配比 / 长文档先切块）、实测证据与验收标准，见
**`../skills/minigpt-train/references/data-distribution.md`**（唯一详述处）。

---

> **关于本 README 被跟踪**：`.gitignore` 用 `/dataset/*` 忽略目录内容（不能写 `/dataset/`，
> 否则 git 不进入被忽略的目录，下面的否定规则会失效），再加 `!/dataset/README.md` 放行本文件。
> 它是**文档**而非数据。若不想跟踪，删掉那条例外即可。
