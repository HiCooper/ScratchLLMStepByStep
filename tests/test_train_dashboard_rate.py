"""看板速率/ETA 的健壮性：**重启造成的停机空档不能被当成训练变慢**。

真实事故：机器重启后第一个评估点进来时，最后两点间隔含 1.7h 停机
（step 76000 @21:17 → 80000 @22:58），看板立刻显示 "1.7k tok/s / ETA 219h"，
要等下一个评估点（约 6 分钟后）才自愈——而这恰好是用户重启后第一眼看到的东西。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_dashboard import _rate_from_evals  # noqa: E402


def _ev(step, ts):
    return {"step": step, "ts": ts}


NORMAL = [_ev(72000, "2026-09-12 21:05:00"), _ev(74000, "2026-09-12 21:11:00"),
          _ev(76000, "2026-09-12 21:17:00")]


def test_continuous_points_give_expected_rate():
    r = _rate_from_evals(NORMAL, 4088)
    assert abs(r["s_per_step"] - 0.18) < 1e-6          # 6 分钟 / 2000 步
    assert abs(r["tok_per_s"] - 4088 / 0.18) < 1.0


def test_restart_gap_is_excluded():
    """带 1.7h 停机的那一段必须被丢弃，速率仍贴近正常值。"""
    restart = NORMAL + [_ev(80000, "2026-09-12 22:58:00")]
    r = _rate_from_evals(restart, 4088)
    assert abs(r["s_per_step"] - 0.18) < 1e-6, f"停机空档没被排除：{r}"
    # 反例：若照旧写法直接用最后两点，会得到 ~1.9 s/步（ETA 200h+）
    naive = (22 * 3600 + 58 * 60 - (21 * 3600 + 17 * 60)) / 4000
    assert naive > 1.0 and r["s_per_step"] < naive / 5


def test_single_point_returns_none():
    assert _rate_from_evals([_ev(1000, "2026-09-12 21:00:00")], 4088) is None
    assert _rate_from_evals([], 4088) is None
