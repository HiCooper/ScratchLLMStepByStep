# MiniGPT 训练配方与容量换算

## 1. 硬件档位（实测于 RTX2060 6GB / 6 核 / 15GB，torch 2.13 fp16 + GradScaler）

| 模型 | 词表 | 单步(bs8,ctx512) | 吞吐 | 备注 |
|---|---|---|---|---|
| 512/10/8（47.9M） | 32k | 0.138s（compile）/ 0.229s（eager） | 29.7k / 17.9k tok/s | 本仓库生产配置 |
| 384/8/8（~30M） | 32k | ~0.09s（compile 估计） | ~35k tok/s | 更省显存，质量略低 |
| 768/12/12（~120M） | 32k | 需 batch≤4 + 累积 | ~10-14k tok/s | 6GB 可跑但慢 |
| 任意小型 | 151k（Qwen2.5） | ~2.0s | **~2.0k tok/s** | 6GB 上不可行；仅大显存使用 |

> 结论：6GB 卡上优先「32k 词表 + 512/10/8 + fp16 + torch.compile」，用数据量换质量。

## 2. 预算换算（agent 决策公式）

```
可用步数   = 时间预算(秒) / 实测s_per_step
看到 tokens = 可用步数 × batch_size × (context_length-1)
Chinchilla 参考 = 10~20 × 参数量
```

实测参考（本机 512/10/8, compile, bs8, ctx512 ≈ 4096 tok/步, 0.17s/步）：
- 1 小时 ≈ 21k 步 ≈ **8,650 万 tokens**
- 10 小时 ≈ 21 万步 ≈ **8.6 亿 tokens**（≈18×参数，接近 compute-optimal）
- 全量语料 `pretrain_v3_full.bin` = **2.475 亿 tokens**（127 万行），1 epoch ≈ 6.0 万步 ≈ 2.8 小时

## 3. 超参经验值

| 项 | 预训练 | SFT | CoT-SFT |
|---|---|---|---|
| lr | 6e-4 ~ 1e-3（cosine，warmup 200~500） | 1e-5 ~ 2e-5 | 1e-5 ~ 1.5e-5 |
| batch | 8（ctx512） | 8 | 8 |
| epochs | 1~5（受 tokens 预算限制） | 1~2 | 2~3 |
| ctx | 512 | 512（max_len 截断） | 512 |
| eval/save | 2000 / 4000 | 500 / 4000 | 500 / 4000 |
| 关键开关 | `--train_torch_compile True` | 变长不启用 compile | 变长不启用 |

## 4. 续训与守护

- 续训：`--paths_last_checkpoint_path <ckpt>`（checkpoint 内含 optimizer/scaler/RNG，可无缝续）。
- 自愈：`scripts/train_pretrain_resilient.sh`（被杀/崩溃自动从最新 checkpoint 续跑，最多 20 次）。
- 空间：`scripts/checkpoint_janitor.sh 60 2 120`（每目录留最近 2 个，单个 ~585MB）。
- 多卡：`NPROC=<n> bash scripts/pretrain_start.sh ...`，显存允许时同步放大 batch。

## 5. 词表/数据管线要点

- 数据：`python scripts/build_pretrain_bin.py build --corpus-jsonl ... --tokenizer-dir ... --out-bin ... --max-lines N`
  自动按词表选择 uint16/uint32 并写 `.meta.json`（`TokenBinDataset` 依赖它）。
- 换 tokenizer：只改 `--data_tokenizer_dir`，模型词表自动跟随；**词表 >65535 必须 uint32**。
- 语料顺序即切片顺序（窗口按文件顺序切分）；要混合不同来源请先自行 shuffle 行顺序。

## 6. 评测协议（可复现）

```bash
# loss/perplexity：固定同一份 bin 与 --max-rows，比较不同 checkpoint
python scripts/evaluate_pretrain.py --checkpoint <ckpt> --bin dataset/bins/pretrain_v3_full.bin --max-rows 512
# 思考模式：贪心 + rp=1.0，固定留出集，多策略对比
python scripts/eval_thinking.py --checkpoint <ckpt> --eval-jsonl dataset/sft/cot_eval_easy_zh.jsonl \
  --n 60 --strategies plain,single,two-phase --repetition-penalty 1.0
```
