#!/bin/bash
# 从 ModelScope（魔搭）下载本教程所需数据，避免 HuggingFace 网络问题。
#   - pretrain_t2t.jsonl (~7.9GB)  中英混合文本（846.9 万行 / 3.24G 字符 ≈ 16.5 亿 tokens），
#     既用于训练分词器，也用于预训练。旧的 mini 版（1.2GB / 2.475 亿 tokens）已删除：
#     它是全量的真子集，留着只会让人不知道该训哪一份。
# 数据源：https://www.modelscope.cn/datasets/gongjy/minimind_dataset
#
# 用法：bash scripts/data/download_data.sh [目标目录]   （默认下载到 ./dataset）
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

DATASET="gongjy/minimind_dataset"
TARGET_DIR="${1:-dataset}"
FILES=("pretrain_t2t.jsonl")

mkdir -p "$TARGET_DIR"

# 方式一：modelscope 命令行（推荐，需先 pip install modelscope）
if command -v modelscope >/dev/null 2>&1; then
    echo "使用 modelscope CLI 下载 ..."
    modelscope download --dataset "$DATASET" --local_dir "$TARGET_DIR" "${FILES[@]}"
else
    # 方式二：wget 直连（若 resolve 链接失效，请到网页手动点「下载」按钮）
    echo "未检测到 modelscope，改用 wget 直连下载 ..."
    for f in "${FILES[@]}"; do
        wget -c "https://www.modelscope.cn/datasets/${DATASET}/resolve/master/${f}" -O "$TARGET_DIR/$f"
    done
fi

cat <<EOF

下载完成，数据位于 $TARGET_DIR/。后续步骤：
  1) 分词器：对 $TARGET_DIR/pretrain_t2t.jsonl 取子集训练（见 scripts/data/train_tokenizer.py，加 --max-lines 控制规模）
  2) 预训练：对 $TARGET_DIR/pretrain_t2t.jsonl 执行 build_pretrain_bin.py（content_key="text"）
     生成 .bin 后即可预训练（见 scripts/tools/validate_pretrain.py）
EOF
