#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# 让 torchrun 启动的子进程能找到 minigpt 包（项目根目录加入 PYTHONPATH）
export PYTHONPATH=$(pwd):$PYTHONPATH

nohup torchrun --nproc_per_node 2 minigpt/train/pretrainer.py > /data2/minigpt/models/20241210/20250114.log 2>&1 &

