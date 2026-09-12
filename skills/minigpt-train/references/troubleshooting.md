# MiniGPT 训练排障手册（症状 → 原因 → 处置）

| 症状 | 可能原因 | 处置 |
|---|---|---|
| `CUDA out of memory` | batch/ctx 过大；评测/指标与本任务抢显存 | 降 `--train_batch_size 8→4→2`；升 `--train_grad_accumulation_steps`；降 ctx 512→384；关指标 `--train_log_hist_every 0 --train_log_embedding_every 0 --train_log_attention_every 0`；确认无其它训练进程 |
| `OverflowError: Python integer ... out of bounds for uint16` | 词表 >65535（如 Qwen2.5 151k）却用 uint16 | 用管线自动 dtype（`build_pretrain_bin.py`）或显式 uint32；旧 bin 需重建 |
| 续训报 `RNG state must be a torch.ByteTensor` | `torch.load(map_location=device)` 把 RNG 张量搬到 GPU | 已在 `Trainer._load_from_checkpoint` 统一 `.cpu()` 修复；升级仓库即可 |
| 训练进程莫名消失、日志无 traceback | 被外部信号杀死（终端/工具进程组） | 用 `setsid nohup ... &` 启动；用 `scripts/train_pretrain_resilient.sh` 自愈；`pgrep -af minigpt.train.pretrainer` 核查 |
| 吞吐骤降（>0.3s/步） | 未启用 compile；eval 过密；指标过密；GPU 被占用 | 确认 `--train_torch_compile True`；`--train_eval_steps 2000`；关直方图/注意力面板；`nvidia-smi` 查占用 |
| `torch.compile` 报 complex operator 警告 | RoPE 使用复数运算，inductor 无法代码生成 | 属已知警告，实测仍提速；如异常则 `--train_torch_compile False` |
| SFT 反复重编译/变慢 | 变长序列导致 dynamo 重编译 | SFT 不启用 compile；或固定 `--data_max_len` |
| eval_loss 不降 / 上升 | lr 过大、warmup 太短、数据异常 | 降 lr（×0.3~0.5）、加 warmup、检查 `--data_tokenized_bin` 与 meta 是否匹配 |
| grad_norm 持续 > 裁剪阈值（1.0） | lr 偏高 / 难样本多 | 看板红虚线之上说明每步都被裁剪；降 lr 或增大 `--train_grad_clip`；伴随 loss 尖峰时必须降 lr |
| 算术/思考模式评测准确率异常低 | 贪心评测误用 repetition_penalty>1（惩罚重复数字） | `--repetition-penalty 1.0` 重测 |
| 生成结果尾部有 `<|im_end|>` | 它是 tokenizer_v3 的 eos（回合结束符），属预期 | 默认 CLI 已去特殊 token；调试用 `--keep-special-tokens` |
| 磁盘写满（WSL 下 C 盘爆） | checkpoint 每 4000 步写 ~585MB；WSL vhdx 只增不减 | 跑 `scripts/checkpoint_janitor.sh`；清理 Windows Temp；数据放 D/E 盘；必要时 `wsl --manage <distro> --set-sparse true` 或迁移 vhdx |
| 看板无数据/进度不动 | 日志路径不对、任务未启动、端口占用 | `python3 scripts/train_dashboard.py --once`；确认 `--log/--out-dir` 指向当前 run；`ss -ltn | grep 8099` |
| `pytest` 失败 | 依赖缺失或 API 变更 | `pip install -r requirements.txt`；`pytest tests/ -q -x` 看首个失败；再跑 `scripts/check_env.py` |
| 数据行数/长度与预期不符 | jsonl 每行是多条拼接的长段落 | 训练前用 `scripts/build_pretrain_bin.py info --bin <bin>` 核对 tokens 与 dtype |

## 快速自检命令

```bash
bash skills/minigpt-train/scripts/preflight.sh            # 环境+资产+服务
python3 -m pytest tests/ -q                               # 单测
python3 scripts/build_pretrain_bin.py info --bin dataset/bins/pretrain_v4_full.bin
python3 scripts/train_dashboard.py --once                 # 训练进度
tail -n 30 models/checkpoints/pretrain_v2_full.log
```
