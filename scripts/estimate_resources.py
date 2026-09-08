"""估算模型参数量与训练资源（参数量 / 显存 / 起步 GPU / 训练时长）。

对应 README「模型规模与资源估算」一节的公式，用于粗略评估训练资源。

用法示例：
    # 教程默认配置（134M）
    python scripts/estimate_resources.py

    # 自定义 + 指定 GPU 与数据量
    python scripts/estimate_resources.py --emb-dim 1024 --n-layers 24 --n-heads 16 \
        --gpu rtx4090 --tokens 2e8

说明：结果仅作量级参考——吞吐按「算力线性、参数量反比」从 RTX 2060 实测锚点外推，
小模型可能受显存带宽限制、大模型/多卡有通信损耗，实际值会有偏差。
"""
import argparse

# 相对吞吐（以 RTX 2060 为 1×，锚点：134M 模型 fp16 实测 ≈ 2700 token/s）
GPU_POWER = {
    "rtx1660": 1.0,
    "rtx2060": 1.0,
    "rtx3060": 3.0,
    "rtx4060": 3.0,
    "rtx3080": 6.0,
    "rtx3090": 10.0,
    "rtx4090": 10.0,
    "a100": 14.0,
    "h100": 30.0,
}
ANCHOR_PARAMS = 134e6   # 实测锚点：134M 模型
ANCHOR_TPS = 2700       # RTX 2060 fp16 实测 token/s

# 常见 GPU 显存档位（GB）
GPU_SIZES = [4, 6, 8, 12, 16, 24, 48, 80]


def estimate_params(vocab, emb, layers, swiglu, qkv_merged, tie):
    """参数量估算（默认架构：GELU + 独立/合并 QKV + 可选共享输出头）。"""
    attn = 4 * emb * emb                              # Q/K/V/O 共 4 个 emb×emb 矩阵
    ffn = (12 if swiglu else 8) * emb * emb           # SwiGLU 3 矩阵 / GELU 2 矩阵
    emb_layer = vocab * emb                            # 词嵌入
    out_head = 0 if tie else vocab * emb               # 输出头（共享则省掉）
    return emb_layer + layers * (attn + ffn) + out_head


def nearest_gpu(need_gb):
    for size in GPU_SIZES:
        if need_gb <= size:
            return size
    return "多卡/集群"


def fmt(n):
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return f"{n:.0f}"


def main():
    p = argparse.ArgumentParser(description="估算模型参数量与训练资源")
    p.add_argument("--vocab", type=int, default=32000, help="词表大小")
    p.add_argument("--emb-dim", type=int, default=768, help="隐藏维度 emb_dim")
    p.add_argument("--n-layers", type=int, default=12, help="层数")
    p.add_argument("--n-heads", type=int, default=12, help="注意力头数(仅展示，不影响参数量公式)")
    p.add_argument("--swiglu", action="store_true", help="前馈用 SwiGLU(每层 +4×emb²)")
    p.add_argument("--qkv-merged", action="store_true", help="合并 QKV 投影")
    p.add_argument("--tie", action="store_true", help="词嵌入与输出头共享权重(省 vocab×emb)")
    p.add_argument("--gpu", default="rtx2060", choices=list(GPU_POWER), help="目标 GPU")
    p.add_argument("--tokens", type=float, default=2e8, help="数据量(token 数)，默认 2 亿 ≈ 教程 1.2GB")
    args = p.parse_args()

    params = estimate_params(args.vocab, args.emb_dim, args.n_layers, args.swiglu, args.qkv_merged, args.tie)
    weight_mem = params * 16 / 1e9  # GB：16 字节/参数（fp16 模型2 + 梯度2 + fp32 主权重4 + 动量4 + 方差4）

    power = GPU_POWER[args.gpu]
    tps = ANCHOR_TPS * power * (ANCHOR_PARAMS / params)   # 吞吐随算力线性、随参数量反比
    epoch_hours = args.tokens / tps / 3600

    print("=" * 60)
    print("模型规模估算")
    print("=" * 60)
    print(f"配置          : vocab={args.vocab}, emb_dim={args.emb_dim}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}")
    print(f"前沿开关      : SwiGLU={args.swiglu}, QKV合并={args.qkv_merged}, 权重共享={args.tie}")
    print(f"参数量        : {fmt(params)}")

    print()
    print("=" * 60)
    print("训练显存估算 (fp16 混合精度 + AdamW)")
    print("=" * 60)
    print(f"权重+优化器+梯度 : {weight_mem:.2f}GB  (16 字节/参数)")
    print(f"激活显存(估)     : 约 {weight_mem:.2f}~{2*weight_mem:.2f}GB (随 batch×seq 增加)")
    start_gb = nearest_gpu(weight_mem * 2.0)
    comfy_gb = nearest_gpu(weight_mem * 4.0)
    print(f"建议起步 GPU     : 显存 >= {weight_mem*2:.1f}GB（约 {start_gb}GB 卡，小 batch 可跑）")
    print(f"建议舒适 GPU     : 显存 >= {weight_mem*4:.1f}GB（约 {comfy_gb}GB 卡）")

    print()
    print("=" * 60)
    print("训练时长估算")
    print("=" * 60)
    print(f"目标 GPU        : {args.gpu} (相对算力 {power:.1f}×)")
    print(f"参考吞吐        : ~{tps:.0f} token/s")
    print(f"数据量          : {fmt(args.tokens)} token")
    print(f"1 epoch 参考时长: ~{epoch_hours:.1f} 小时")

    print()
    print("注：吞吐按算力线性、参数量反比从 RTX 2060 实测锚点外推，仅作量级参考；")
    print("    小模型可能受显存带宽限制、多卡有通信损耗，实际值会有偏差。")


if __name__ == "__main__":
    main()
