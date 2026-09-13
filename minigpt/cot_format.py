"""思考模式（CoT）文本格式的**唯一**定义。

为什么要单独一个模块：这三个字符串以前在三个地方各写了一份——
`minigpt/model/generation.py`（推理，靠它切分思考/答案）、
`scripts/data/build_cot_sft.py`（造训练数据，靠它拼输出）、
`scripts/eval_thinking.py`（评测，靠它从生成结果里抠答案）。
任何一处改了（哪怕只是全角/半角冒号）都不会报错，只会**静默**让准确率掉到
"取最后一个数字"的兜底路径上，把评测结论变成噪声。所以集中定义 + 单测锁一致性。

本模块刻意**零重依赖**（不 import torch/transformers），这样纯数据的
`build_cot_sft.py` 也能安全引用。
"""

# 训练数据里 output 以它开头；推理时也会把它预填到 assistant 之后（强制进入思考格式）
THINK_PREFIX = "让我逐步分析：\n"
# 思考段与最终答案的分隔标记
ANSWER_MARK = "最终答案："
