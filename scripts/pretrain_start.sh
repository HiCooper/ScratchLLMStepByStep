#!/bin/bash
# 预训练启动脚本：使用 torchrun 以 DDP 方式在多卡上训练。
# 脚本位于 scripts/ 下，先切回项目根目录，保证 minigpt 包与相对路径可用。

cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=0,1
# 让 torchrun 启动的子进程能找到 minigpt 包（项目根目录加入 PYTHONPATH）
export PYTHONPATH=$(pwd):$PYTHONPATH

nohup torchrun --nproc_per_node 2 minigpt/train/pretrainer.py > /data2/minigpt/models/20241210/20250114.log 2>&1 &

