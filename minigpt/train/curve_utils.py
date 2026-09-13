"""训练曲线的共享口径：从相邻评估点估速率 / 吞吐。

**唯一实现**的由来：看板（`scripts/train_dashboard.py`）与交付报告
（`scripts/report_training.py`）以前各写一份速率估计 —— 看板那份会剔除重启/停机造成的
巨大空档，报告那份只看最后两个评估点。于是同一次 v5 长训（机器 2026-09-12 重启过），
看板显示 ~23k tok/s / ETA ~7h，而交付报告写出 ~0.9k tok/s / ETA 168h：
报告的数字直接进交付物，却没有任何东西会报红。两者从此共用这里的一份实现。
"""
from __future__ import annotations

import statistics
import time


def rate_from_evals(evals, tokens_per_step, max_gap_factor=3.0):
    """从相邻评估点估计速率；**排除重启/停机造成的巨大空档**。

    真实事故：机器重启后第一次评估进来时，最后两个点的间隔含 5 小时停机
    （21:47 的 step 76000 → 23:0x 的 80000），看板立刻显示 "1.7k tok/s / ETA 219h"，
    要等下一个评估点才自愈。做法：取最近若干个区间的 s/step 中位数，
    明显偏离中位数（>3×）的区间视为停机造成的、丢弃。

    `evals`：按时间升序的序列，每项需有 `step` 与 `ts`（"YYYY-MM-DD HH:MM:SS"）。
    只有两个评估点时无从判断（没有中位数可依），只能照用——此时刚重启后的第一次
    估计仍可能偏大，属已知局限；调用方可据此要求至少 3 个点。

    返回 `{"s_per_step": ..., "tok_per_s": ...}`；点数不足或时间戳不可解析时返回 None。
    """
    def _ts(x):
        return time.mktime(time.strptime(x["ts"], "%Y-%m-%d %H:%M:%S"))

    pairs = []
    recent = evals[-6:]                     # 配对必须是 (x[i], x[i+1] identical slice 会配到自己)
    for a, b in zip(recent, recent[1:]):
        dsteps, dt = b["step"] - a["step"], _ts(b) - _ts(a)
        if dsteps > 0 and dt > 0:
            pairs.append(dt / dsteps)
    if not pairs:
        return None
    med = statistics.median(pairs)
    good = [p for p in pairs if p <= med * max_gap_factor]
    s = statistics.median(good or pairs)
    return {"s_per_step": s, "tok_per_s": tokens_per_step / s}
