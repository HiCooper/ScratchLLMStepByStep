#!/usr/bin/env python3
"""汇总训练/评测产物，生成 Markdown 对比报告（供 agent 与用户查看）。

自动收集：
  - 预训练 run：models/checkpoints/pretrain_*/metrics.json（eval_loss / perplexity）
  - 预训练曲线：pretrain_*/tensorboard 事件文件（eval_loss 全程曲线 / 吞吐 / ETA）
  - SFT/CoT run：models/checkpoints/sft_*/metrics.json
  - 思考模式评测：models/checkpoints/eval_v2_cot_{easy,hard}.json（plain/single/two-phase 准确率）
  - 同切分 ppl 对比：models/checkpoints/ppl_*.json
  - 问答样例：models/checkpoints/samples_v2_*.txt、各 run 的 sample.txt

用法：
  python3 scripts/report_training.py [--out models/checkpoints/TRAINING_REPORT.md] [--json /tmp/report.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from minigpt.config import estimate_params_from_config  # noqa: E402
from minigpt.train.curve_utils import rate_from_evals  # noqa: E402
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def read_text(path, limit=1200):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()[:limit]
    except Exception:  # noqa: BLE001
        return ""


def _pick(cfg: dict, name: str, default=None, section: str = "model"):
    """按 section 读配置项，兼容仓库里出现过的三种 config 结构：

    - nested 规范：`{"model": {"context_length": 512}}` ← `dump_run_config` 唯一在写的结构
    - nested 旧前缀：`{"model": {"model_context_length": 512}}`（早期 dump）
    - flat：`{"emb_dim": ...}` / `{"model_context_length": ...}`（老产物与测试夹具）

    以前只认后两种，于是**当下唯一在写的** nested 规范结构永远取不到值，静默回落到
    默认 512：ctx1024 的 run 报告里吞吐被算成实际值的 1/4，且与 §6 的 token 预算换算
    互相矛盾，而没有任何东西会报红。
    """
    if not isinstance(cfg, dict):
        return default
    sec = cfg.get(section)
    if isinstance(sec, dict):
        for key in (name, f"{section}_{name}"):
            if sec.get(key) is not None:
                return sec[key]
    for key in (name, f"{section}_{name}"):
        if cfg.get(key) is not None:
            return cfg[key]
    return default


def estimate_params(cfg: dict) -> str:
    """参数量展示（调用 config.py 的唯一实现，含 SwiGLU/8-3·d/tie/head-bias 开关）。

    旧实现自带一份公式且忽略 SwiGLU —— 对 SwiGLU run 会多报 50% 前馈参数。
    """
    try:
        return f"{estimate_params_from_config(cfg) / 1e6:.1f}M"
    except Exception:  # noqa: BLE001
        return "?"


TB_TAGS = ("eval/loss", "eval/perplexity", "train/loss", "train/tokens_per_sec")

# 文档记录的历史基线（同步自根 README「实测结果」表的 pretrain_v2_full 行；
# tests/test_report.py 会解析 README 校验这几个数字，防止两边漂移）。
# 用途：给出**同架构、同 token 预算**的参照点——v2 是 211k 步 × 4096 tokens/步 ≈ 8.6 亿 tokens，
# v5 在同一步数消耗的 token 完全一致，可直接比 loss/perplexity（语料与验证切分不同，见渲染处的说明）。
HISTORICAL_BASELINE = {
    "run": "pretrain_v2_full",
    "step": 211_000,
    "tokens": 0.86e9,
    "train_loss": 23.66,
    "perplexity": 15.82,
}


def _token_matched(curve: dict, step: int, tol: float = 0.03):
    """曲线上与目标步数**足够接近**（默认 ±3%）的评估点；否则返回 None。

    必须有容差：评估每 2000 步一次，只取"≤ 目标步数的最后一个点"会让只跑了 76k 步的
    run 被拿去和 211k 步的历史基线并列，等于偷换预算。±3%（±6330 步）刚好能命中
    210k/212k，又不会把 76k 误判成"同预算"。
    """
    pts = [p for p in (curve.get("eval") or []) if abs(p[0] - step) <= step * tol]
    return min(pts, key=lambda p: abs(p[0] - step)) if pts else None


# 事件目录超过这个体量就改用日志重建曲线。实测：开了 log_hist_every 的长训，
# 282k 步时事件目录已 2.0GB（单个文件 1.5GB），解析要几分钟；而日志里每个 eval 点
# 本来就有一行（step/train_loss/eval_loss/时间戳），秒级可读。
TB_MAX_BYTES = 200 * 1024 * 1024

_LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) lr=([\d.eE+-]+), train_loss: ([\d.]+), "
    r"eval_loss: ([\d.]+), grad_norm=([\d.]+), steps: (\d+)/(\d+)")


def _tb_bytes(run_dir: str) -> int:
    tb = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(tb):
        return 0
    try:
        return sum(os.path.getsize(os.path.join(tb, f)) for f in os.listdir(tb))
    except OSError:
        return 0


def _seconds_between(t0: str, t1: str) -> float:
    try:
        return (datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
                - datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        return 0.0


def curve_from_log(log_path: str, tokens_per_step: int = 8 * 511) -> dict:
    """从训练日志重建 eval 曲线（tensorboard 过大时的兜底，秒级）。

    日志行形如
      `2026-09-13 08:25:50 lr=0.00012, train_loss: 2.41, eval_loss: 2.38, grad_norm=0.5, steps: 282000/412000`

    吞吐走 `curve_utils.rate_from_evals`（与看板同一实现）：取最近区间的中位数并丢弃
    >3× 中位数的空档。以前这里是"最后两点相减"——机器 2026-09-12 重启后，一次 v5 长训
    的两个评估点之间夹着数小时停机，报告就写出 ~0.9k tok/s / ETA 168h（实际 ~23k/~7h），
    而这条日志分支正是长训的默认路径（事件目录 >200MB）。

    `tokens_per_step` 只影响**展示的吞吐**，不影响 ETA：ETA = 剩余步数 × s_per_step，
    与 tokens_per_step 无关（下面直接按 s_per_step 算，避免这个量被写错两次）。
    """
    ev, stamps, tail_train, total = [], [], None, None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_LINE.match(line.strip())
                if not m:
                    continue
                ts, _lr, tr, el, _gn, step, ttl = m.groups()
                ev.append((int(step), float(el)))
                tail_train = float(tr)
                total = int(ttl)
                stamps.append((int(step), ts))
    except OSError:
        return {"eval": [], "tail_tok_s": None, "tail_train_loss": None, "eta_hours": None,
                "n_eval": 0, "source": "log", "tokens_per_step": tokens_per_step}
    # 至少 3 个点：有 ≥2 个区间才谈得上"用中位数剔掉停机空档"，两个点时纯属照用
    # （rate_from_evals 的已知局限，报告这边宁可不给数也不给错数）。
    rate = rate_from_evals([{"step": s, "ts": t} for s, t in stamps[-6:]],
                           tokens_per_step) if len(stamps) >= 3 else None
    out = {"eval": ev, "tail_tok_s": (rate or {}).get("tok_per_s"),
           "tail_train_loss": tail_train, "eta_hours": None, "n_eval": len(ev),
           "source": "log", "tokens_per_step": tokens_per_step}
    if rate and ev and total and ev[-1][0] < total:
        out["max_steps"] = total
        out["eta_from_step"] = ev[-1][0]
        out["eta_hours"] = (total - ev[-1][0]) * rate["s_per_step"] / 3600.0
    return out


def read_tb(run_dir: str) -> dict:
    """读取 run 目录下 tensorboard 事件里的曲线；任何异常都降级为 {}（绝不因报告而失败）。

    metrics.json 只留最后一次评估，**全程 eval_loss 曲线只在 tensorboard 里**，
    而"曲线"是交付报告的必要内容，所以这里把它抽出来。
    """
    tb_dir = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(tb_dir):
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
        return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)]
                for t in TB_TAGS if t in tags}
    except Exception:  # noqa: BLE001 —— 缺包/坏事件文件都不该让报告生成失败
        return {}


def _curve_summary(run_dir: str, cfg: dict) -> dict:
    """eval_loss 曲线 + 吞吐 + ETA（ETA 需要 config.json 里的 max_steps）。"""
    tb = read_tb(run_dir)
    ev = tb.get("eval/loss") or []
    tps = [v for _, v in (tb.get("train/tokens_per_sec") or [])[-500:]]
    tr = [v for _, v in (tb.get("train/loss") or [])[-200:]]
    out = {"eval": ev, "tail_tok_s": (sorted(tps)[len(tps) // 2] if tps else None),
           "tail_train_loss": (sum(tr) / len(tr) if tr else None), "eta_hours": None,
           "n_eval": len(ev), "tokens_per_step": tokens_per_step_of(cfg)}
    try:
        max_steps = int(_pick(cfg, "max_steps", 0, "train") or 0)
        # 只有 max_steps 参与 ETA（tokens/step 在 tok/s 口径里自相消），见 tokens_per_step_of
        if not max_steps:
            return out
    except Exception:  # noqa: BLE001
        return out
    if out["tail_tok_s"] and ev and ev[-1][0] < max_steps:
        out["max_steps"] = max_steps
        out["eta_from_step"] = ev[-1][0]
        out["eta_hours"] = (max_steps - ev[-1][0]) / out["tail_tok_s"] \
            * out["tokens_per_step"] / 3600.0
    return out


def _spark(values, width: int = 48) -> str:
    """把 eval_loss 曲线压成一行 sparkline（▁▂▃▄▅▆▇█），让报告自解释。"""
    if len(values) < 2:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    pts = values[::step][:width]
    lo, hi = min(pts), max(pts)
    if hi - lo < 1e-9:
        return blocks[0] * len(pts)
    return "".join(blocks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in pts)


def _loss_noise(eval_rows) -> float | None:
    """验证集上 eval_loss 的 1σ 统计噪声 ≈ 1/√(验证窗口数)。

    口径与 `scripts/data/audit_dataset.py` 的 split 检查一致（那边是 1/√(val_tok/(ctx-1))，
    等价于 1/√(窗口数)），不要在两处各写一个公式。

    为什么值得写进报告：v5 的验证切分是 511 个窗口，噪声约 ±0.044——
    相邻两次 eval 差 0.02 时那是噪声不是收敛，不写清楚很容易被当成"还在稳定下降"。
    """
    try:
        n = float(eval_rows)
        return 1.0 / math.sqrt(n) if n > 0 else None
    except (TypeError, ValueError):
        return None


def tokens_per_step_of(cfg: dict, default: int = 4096) -> int:
    """一次参数更新吃掉的 token 数 = batch × (ctx-1) × 梯度累积步数（**单卡口径**）。

    - 口径与 `skills/minigpt-train/scripts/preflight.py` 的 `tokens_per_step` 对齐
      （那边再乘上 `launch.nproc`）。config.json 里没有 world_size，所以 DDP run 这里是
      单 rank 值：§1.5 展示的 tok/s 对多卡 run 会低报 world_size 倍，属已知局限，报错不说谎。
    - `ctx` 必须走 `_pick`：以前写死读 `model["model_context_length"]`，而实际 dump 的是
      `model["context_length"]`，于是任何 ctx1024 预设（24GB+/多卡档位会选它）的吞吐
      都被算成 1/4。
    - 注意 ETA **不受**这个值影响（tok/s 与 tokens/step 在 ETA 公式里相消），它只决定
      报告中"吞吐"那一栏的绝对数值以及 §6 的同 token 预算换算。
    """
    try:
        bs = int(_pick(cfg, "batch_size", 8, "train") or 8)
        ctx = int(_pick(cfg, "context_length", 512, "model") or 512)
        accum = max(1, int(_pick(cfg, "grad_accumulation_steps", 1, "train") or 1))
        return bs * max(ctx - 1, 1) * accum
    except (TypeError, ValueError):
        return default


def _pick_curve(run_dir: str, cfg: dict, log_path: str) -> dict:
    """优先 tensorboard；事件目录过大（长训记了直方图）就改用日志重建。

    实测教训：v5 跑到 282k 步时 tensorboard 已 2.0GB（单文件 1.5GB、绝大部分是直方图），
    用 EventAccumulator 解析要几分钟且吃内存；交付报告不该被"日志体积"拖住。
    """
    size = _tb_bytes(run_dir)
    if size and size <= TB_MAX_BYTES:
        c = _curve_summary(run_dir, cfg)
        if c.get("eval"):
            c["source"] = "tensorboard"
            c["tb_bytes"] = size
            return c
    c = curve_from_log(log_path, tokens_per_step_of(cfg))
    c["tb_bytes"] = size
    # 日志缺行（改名/被轮转/训练日志没留下 eval 行）时回到 tensorboard。
    # 这里**不能**再判 `size <= TB_MAX_BYTES`：能走到这一行，要么 tb 超阈、要么
    # tb 分支读不出标量——两种情况下该条件都为假，兜底永远不可达，结果是"事件目录
    # 超阈 + 日志缺失"时整条阶段曲线静默消失。宁可慢，也不要空。
    if not c["eval"] and size:
        c = _curve_summary(run_dir, cfg)
        c["source"] = "tensorboard"
        c["tb_bytes"] = size
    return c


def collect(cp: str | None = None):
    """扫描 checkpoints 目录并汇总（cp 可指定，便于测试）。"""
    cp = cp or os.path.join(ROOT, "models", "checkpoints")
    data = {"time": datetime.now().strftime("%F %T"), "pretrain": [], "sft": [],
            "thinking": {}, "ppl": {}, "samples": {}, "success_rate": {},
            "curves": {}, "sft_curves": {}}
    # 曲线独立于 metrics.json 扫描：metrics.json 只在训练结束才落盘，
    # 但 tensorboard 是训练过程中实时写的，这样报告在训练途中也能看到曲线。
    for run_dir in sorted(glob.glob(os.path.join(cp, "pretrain_*"))):
        if not os.path.isdir(run_dir):
            continue
        cfg = load(os.path.join(run_dir, "config.json")) or {}
        c = _pick_curve(run_dir, cfg, os.path.join(cp, os.path.basename(run_dir) + ".log"))
        if c["eval"]:
            data["curves"][os.path.basename(run_dir)] = c
    # SFT/CoT 阶段同样写了 tensorboard（sft_trainer 会建 SummaryWriter），
    # 交付要求里的"各阶段 eval_loss 曲线"因此可以逐阶段给出来，而不是只有基座。
    for run_dir in sorted(glob.glob(os.path.join(cp, "sft_*"))):
        if not os.path.isdir(run_dir):
            continue
        cfg = load(os.path.join(run_dir, "config.json")) or {}
        c = _pick_curve(run_dir, cfg, os.path.join(cp, os.path.basename(run_dir) + ".log"))
        if c["eval"]:
            data["sft_curves"][os.path.basename(run_dir)] = c
    for path in sorted(glob.glob(os.path.join(cp, "pretrain_*", "metrics.json"))):
        m = load(path) or {}
        cfg = load(os.path.join(os.path.dirname(path), "config.json")) or {}
        run = os.path.basename(os.path.dirname(path))
        data["pretrain"].append({"run": run, "step": m.get("step"), "epoch": m.get("epoch"),
                                 "train_loss": m.get("train_loss"), "eval_loss": m.get("eval_loss"),
                                 "perplexity": m.get("perplexity"),
                                 "best_eval_loss": m.get("best_eval_loss"),
                                 "best_step": m.get("best_step"),
                                 # 评估口径：不同切分/不同 bin 的 eval_loss 不可直接比较
                                 "eval_split": (m.get("data_split") or {}).get("split"),
                                 "eval_rows": (m.get("data_split") or {}).get("eval_rows"),
                                 "eval_blocks": (m.get("data_split") or {}).get("n_blocks"),
                                 "params": estimate_params(cfg),
                                 "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt")),
                                 "best": os.path.exists(os.path.join(os.path.dirname(path), "best.pt"))})
        # 把"验证集有多大、噪声多少"挂到 §1.5 的曲线上（曲线本身来自 tensorboard，与 metrics.json 独立）
        c = data["curves"].get(run)
        if c is not None:
            c["eval_rows"] = data["pretrain"][-1].get("eval_rows")
            c["noise_sigma"] = _loss_noise(c["eval_rows"])
    for path in sorted(glob.glob(os.path.join(cp, "sft_*", "metrics.json"))):
        m = load(path) or {}
        data["sft"].append({"run": os.path.basename(os.path.dirname(path)),
                            "step": m.get("step"), "train_loss": m.get("train_loss"),
                            "eval_loss": m.get("eval_loss"),
                            "best_eval_loss": m.get("best_eval_loss"),
                            "best_step": m.get("best_step"),
                            "final": os.path.exists(os.path.join(os.path.dirname(path), "final.pt")),
                            "best": os.path.exists(os.path.join(os.path.dirname(path), "best.pt"))})
    for path in sorted(glob.glob(os.path.join(cp, "eval_*_cot_*.json"))):
        js = load(path)
        if js:
            data["thinking"][os.path.basename(path)[:-5]] = js.get("summary", js)
    for path in sorted(glob.glob(os.path.join(cp, "ppl_*.json"))):
        js = load(path) or {}
        data["ppl"][os.path.basename(path)[4:-5]] = {k: js.get(k) for k in
                                                     ("eval_loss", "perplexity", "rows", "tokens",
                                                      "checkpoint", "bin", "max_rows")}
    for path in sorted(glob.glob(os.path.join(cp, "samples_*.txt")) + glob.glob(os.path.join(cp, "*", "sample.txt"))):
        data["samples"][os.path.relpath(path, cp)] = read_text(path)
    # 评测中的正确率明细（若存在 details 且不含 ok 字段，则统计 success_rate）
    for name, js in list(data["thinking"].items()):
        det = js.get("details") if isinstance(js, dict) else None
        if det:
            for strat in ("plain", "single", "two-phase"):
                vals = [d.get(f"{strat}_ok") for d in det if f"{strat}_ok" in d]
                if vals:
                    data["success_rate"][f"{name}:{strat}"] = round(sum(bool(v) for v in vals) / len(vals), 4)
    return data


def _best(r: dict) -> str:
    """best_eval_loss(@step)：老产物没有 best_step 字段时只显示 loss。"""
    v = r.get("best_eval_loss")
    if v is None:
        return "—"
    st = r.get("best_step")
    return f"{float(v):.4f}" if st is None else f"{float(v):.4f}@{st}"


def _split_label(r) -> str:
    """把 metrics.json 里记录的评估口径压缩成短标签（区分"训练集口径"与"留出集口径"）。"""
    name = r.get("eval_split")
    if not name:
        return "—(旧产物)"
    if name == "blocked-document-aligned":
        return f"留出集/{r.get('eval_blocks') or '-'}块"
    return str(name)


def _num(v, nd: int = 4) -> str:
    """统一的数值格式化：None/非数值安全降级为 —。"""
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _points_str(ev, max_n: int = 14) -> str:
    """把 (step, loss) 序列压成 `2k:5.116 → … → 412k:1.90` 一行；点数多时等距抽稀但保留首尾。"""
    if not ev:
        return "—"
    if len(ev) > max_n:
        step = (len(ev) - 1) / (max_n - 1)
        idx = sorted({int(round(i * step)) for i in range(max_n)})
        ev = [ev[i] for i in idx]
    return " → ".join(f"{s / 1000:.1f}k:{v:.4f}" if s >= 1000 else f"{s}:{v:.4f}" for s, v in ev)


def render(d: dict) -> str:
    L = [f"# MiniGPT 训练报告", "", f"生成时间：{d['time']}", ""]
    L += ["## 1. 预训练 run", "",
          "> 注：各 run 的 eval_loss 来自各自的验证切分（语料/比例/块数可能不同），**只有「评估口径」一致的 run 才可直接比较**；"
          "严格的同口径对比见 §2（同一 bin、同一 `--split val`）。",
          "`best.pt` 是 eval_loss 历史最优时的权重（小模型后期易过拟合，做下游/评测通常优于 `final.pt`）。", "",
          "| run | step | params | train_loss | eval_loss | perplexity | 评估口径 | best_eval_loss@step | final.pt | best.pt |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in d["pretrain"]:
        best = _best(r)
        L.append(f"| {r['run']} | {r['step']} | {r['params']} | "
                 f"{_num(r.get('train_loss'))} | **{_num(r.get('eval_loss'))}** | {_num(r.get('perplexity'), 2)} | "
                 f"{_split_label(r)} | "
                 f"{best} | {'✅' if r['final'] else '—'} | {'✅' if r.get('best') else '—'} |")
    if not d["pretrain"]:
        msg = "（训练进行中：`metrics.json` 在 run 结束时才落盘，实时曲线见 §1.5）" if d.get("curves") else "（暂无）"
        L.append(f"| {msg} | | | | | | | | | |")

    L += ["", "## 1.5 预训练 eval_loss 曲线与吞吐", "",
          "> 曲线来源随体量自动选择：小 run 直接读 tensorboard 标量；开了 `log_hist_every` 的长训"
          "（实测 286k 步时事件目录已 2.2GB、绝大部分是直方图）改用训练日志重建——日志每个 eval 点"
          "本来就有一行，秒级可读，避免交付报告被几 GB 直方图拖住。", ""]
    if d.get("curves"):
        for run, c in sorted(d["curves"].items()):
            ev = c["eval"]
            vals = [v for _, v in ev]
            lo_i = vals.index(min(vals))
            tps, eta = c.get("tail_tok_s"), c.get("eta_hours")
            thr = "—" if not tps else f"{tps / 1000:.1f}k tok/s"
            prog = f"{ev[-1][0]}" + (f"/{c['max_steps']}" if c.get("max_steps") else "")
            eta_s = "—" if eta is None else f"约 {eta:.1f} h（自 step {c.get('eta_from_step')} 起）"
            src = c.get("source") or "tensorboard"
            tb = c.get("tb_bytes") or 0
            src_txt = f"{src}" + (f"（事件目录 {tb / 1e9:.2f}GB）" if tb > 1e8 else "")
            L += [f"### {run} — 最新 step {prog}，{c['n_eval']} 个评估点（来源：{src_txt}）", ""]
            if len(vals) >= 2:
                L += [f"`{_spark(vals)}`（左早右晚，最低 {min(vals):.4f} @step {ev[lo_i][0]}）", ""]
            L += ["| 指标 | 值 |", "|---|---|",
                  f"| 曲线 step:loss | {_points_str(ev)} |",
                  f"| 末段吞吐（中位数，已剔除停机空档；DDP 为单卡口径） | {thr} |",
                  f"| 末段 train_loss（最近 200 步均值） | {_num(c.get('tail_train_loss'))} |",
                  f"| ETA | {eta_s} |"]
            if c.get("noise_sigma"):
                L.append(f"| 评估噪声（1σ，{c.get('eval_rows')} 个验证窗） | "
                         f"**±{c['noise_sigma']:.4f}** —— 相邻两点差得比它小就不必当成收敛 |")
            L.append("")
    else:
        L += ["（无 tensorboard 数据：老 run 未开 writer，或事件文件已清理）", ""]

    L += ["", "## 2. 同切分 perplexity 对比（同一 bin / 同一 --max-rows）", ""]
    if d["ppl"]:
        # 按评估语料分组：只有同 bin + 同窗口数的数字才可直接比较，不同语料混在一张表里会误导
        groups = {}
        for k, v in d["ppl"].items():
            groups.setdefault((v.get("bin") or "(未记录)", v.get("max_rows")), []).append((k, v))
        for (bin_path, max_rows), items in sorted(groups.items()):
            items.sort(key=lambda kv: (kv[1].get("eval_loss") is None, kv[1].get("eval_loss")))
            base = items[0][1].get("eval_loss") if items else None
            L += [f"**语料 `{bin_path}`（max-rows={max_rows}）**", "",
                  "| run | eval_loss | perplexity | 相对最优降低 | 评估窗口 |", "|---|---|---|---|---|"]
            for k, v in items:
                el = v.get("eval_loss")
                delta = "—" if (el is None or base in (None, 0)) else f"{(1 - el / base) * 100:+.1f}%"
                L.append(f"| {k} | {_num(el)} | {_num(v.get('perplexity'), 2)} | {delta} | {v.get('rows')} |")
            L.append("")
    else:
        L.append("（尚未产出：等待 run_downstream.sh 的 4c 阶段）")

    L += ["", "## 3. 下游 SFT / CoT run", "",
          "| run | step | train_loss | eval_loss | best_eval_loss@step | final.pt | best.pt |",
          "|---|---|---|---|---|---|---|"]
    for r in d["sft"]:
        best = _best(r)
        L.append(f"| {r['run']} | {r['step']} | {_num(r.get('train_loss'))} | {_num(r.get('eval_loss'))} | "
                 f"{best} | {'✅' if r['final'] else '—'} | {'✅' if r.get('best') else '—'} |")
    if not d["sft"]:
        L.append("| （暂无） | | | | | | |")

    L += ["", "## 3.5 SFT / CoT 阶段 eval_loss 曲线（tensorboard）", "",
          "> 曲线直接取自各 run 的 tensorboard，训练途中也会实时出现；"
          "点数多时等距抽稀（保留首尾）。这些 run 的验证集是各自 SFT 数据的 1% 切分，"
          "与基座的语言建模 val 不是一回事，**不要跨阶段比 loss 绝对值**。", ""]
    if d.get("sft_curves"):
        for run, c in sorted(d["sft_curves"].items()):
            ev = c["eval"]
            vals = [v for _, v in ev]
            lo_i = vals.index(min(vals))
            L += [f"**{run}** — 最新 step {ev[-1][0]:,}，{c['n_eval']} 个评估点"
                  f"（最低 {min(vals):.4f} @step {ev[lo_i][0]}）", ""]
            if len(vals) >= 2:
                L += [f"`{_spark(vals)}`（左早右晚）", ""]
            L += [f"| 曲线 step:loss | {_points_str(ev)} |", "|---|---|",
                  f"| 末段 train_loss（最近 200 步均值） | {_num(c.get('tail_train_loss'))} |", ""]
    else:
        L.append("（尚未产出：等待下游 SFT/CoT 阶段）")

    L += ["", "## 4. 思考模式准确率（留出集，贪心 + repetition_penalty=1.0）", ""]
    if d["thinking"]:
        groups = {}
        for name, js in d["thinking"].items():
            m = re.match(r"eval_(?P<tag>.+)_cot_(?P<set>easy|hard)$", name)
            tag = m.group("tag") if m else name
            key = m.group("set") if m else "all"
            groups.setdefault(key, []).append((tag, js))
        for set_name, rows in sorted(groups.items()):
            L += [f"### {set_name}", "", "| run(tag) | n | plain | single | two-phase |", "|---|---|---|---|---|"]
            for tag, js in sorted(rows):
                acc = js.get("accuracy", {}) if isinstance(js, dict) else {}
                L.append(f"| {tag} | {js.get('n')} | {acc.get('plain')} | {acc.get('single')} | "
                         f"{acc.get('two-phase')} |")
            L.append("")
    else:
        L.append("（尚未产出：等待 4a/4b 阶段）")

    L += ["", "## 4.5 CoT 分档准确率（`by_difficulty`）", "",
          "> 平均准确率会把「简单题全对、难题全错」抹平成一个中不溜的数字，看不出**容量墙**。"
          "档位由题面决定（`eval_thinking.py::bucket_of`），与 §4 出自同一份 json。", ""]
    bucketed = {n: js for n, js in d["thinking"].items()
                if isinstance(js, dict) and js.get("by_difficulty")}
    if bucketed:
        for name, js in sorted(bucketed.items()):
            bd = js["by_difficulty"]
            strats = sorted({k for v in bd.values() for k in v if k != "n"})
            L += [f"**{name}**", "",
                  "| 档位 | n | " + " | ".join(strats) + " |",
                  "|---|---|" + "---|" * len(strats)]
            for b, v in bd.items():
                cells = " | ".join("—" if v.get(s) is None else f"{v[s]:.1%}" for s in strats)
                L.append(f"| {b} | {v.get('n')} | {cells} |")
            L.append("")
    else:
        L.append("（尚未产出：等待 4a/4b 阶段，或评测未加 `--strategies`）")

    L += ["", "## 5. 问答 / 生成样例", ""]
    if d["samples"]:
        for name, text in d["samples"].items():
            L += [f"### {name}", "", "```", text, "```", ""]
    else:
        L.append("（尚未产出：等待 4d 阶段）")

    L += ["", "## 6. 历史基线对照（同架构 + 同 token 预算）", "",
          "> **这不是 A/B 结论**。两行是「同模型结构、同步数（211k 步 × 4096 tokens/步 ≈ 8.6 亿 tokens）、"
          "各自评各自语料的验证切分」。语料不同（v2 = 通用 mini 语料；v5 = 通用 80% + 教科书体 20%）"
          "⇒ 验证集的固有熵不同，ppl 不能当作「谁更强」的证据；ppl 只在**同一 bin、同一 `--split val`** 下才可比。",
          "> 它的价值：在完全相同的 token 预算上给出一个可核对的参照点（v2 是历史上唯一有据可查的同预算 run）。",
          "> 注：v5 语料包含 v4 的约 80% 文档，所以**不能用 v5 模型去评 v4/v2 的 val**——那是它见过的东西。",
          "", "| run | step | 训练 token | train_loss | eval_loss | perplexity | 口径 |",
          "|---|---|---|---|---|---|---|",
          f"| `{HISTORICAL_BASELINE['run']}`（历史，文档记录） | {HISTORICAL_BASELINE['step']:,} | "
          f"{HISTORICAL_BASELINE['tokens'] / 1e8:.1f} 亿 | {HISTORICAL_BASELINE['train_loss']} | "
          f"{math.log(HISTORICAL_BASELINE['perplexity']):.4f} | **{HISTORICAL_BASELINE['perplexity']}** | 它自己的 val |"]
    matched_any = False
    for run, c in sorted(d.get("curves", {}).items()):
        # 同 token 预算 ≠ 同 step：批量/ctx/累积一变，一步吃的 token 就不同。
        # 基线的 211k 步是按 4096 tokens/步 记的，所以这里先把"同预算"换算到本 run 的
        # 步数刻度（ctx1024 的 run 是 8192 tok/步 ⇒ 等价预算点约 105.5k 步）。
        # 以前两处都把基准步数写死成 211,000 且 tok 写死 ×4096，于是 ctx1024 的 run
        # 会在半步数上被拿去和历史基线并列、token 数还少报一半。
        tps = int(c.get("tokens_per_step") or 4096)
        target_step = int(round(HISTORICAL_BASELINE["step"] * 4096 / tps))
        pt = _token_matched(c, target_step)
        if pt is None:
            last = (c.get("eval") or [])[-1] if c.get("eval") else None
            cur = f"当前 {last[0]:,} 步，eval_loss {last[1]:.4f}" if last else "暂无评估点"
            L.append(f"| `{run}`（本次） | 尚未到 {target_step:,} 步（{cur}） "
                     "| — | — | — | — | — |")
            continue
        matched_any = True
        ppl = math.exp(min(pt[1], 80.0))
        delta = (ppl / HISTORICAL_BASELINE["perplexity"] - 1) * 100
        tok = pt[0] * tps
        L.append(f"| `{run}`（本次，同预算点） | {pt[0]:,} | {tok / 1e8:.2f} 亿 | — | {pt[1]:.4f} | "
                 f"**{ppl:.2f}** | 它自己的 val |")
        L += ["", f"同预算点对比：`{run}` perplexity **{ppl:.2f}** vs `{HISTORICAL_BASELINE['run']}` "
                  f"**{HISTORICAL_BASELINE['perplexity']}**（{delta:+.1f}%，负值=更低）。"
                  "如上行所述，这个差里同时含「语料变化」与「验证集变化」，不能单独归因于数据质量。"]
        if tps != 4096:
            L += [f"注：本 run 每步 {tps:,} token（历史基线按 4096 记），所以上表的"
                  f"{target_step:,} 步与基线的 {HISTORICAL_BASELINE['step']:,} 步是等 token 预算的。"]
    if not matched_any and d.get("curves"):
        L.append("")
        L.append("（同预算点还没到：基座跑到 211,000 步后本行会自动出现）")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "checkpoints", "TRAINING_REPORT.md"))
    ap.add_argument("--json", default=None, help="同时输出结构化 JSON（供 agent 解析）")
    ap.add_argument("--dir", default=None, help="checkpoints 目录（默认 models/checkpoints）")
    args = ap.parse_args()
    d = collect(args.dir)
    md = render(d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[report] saved -> {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"[report] json  -> {args.json}")


if __name__ == "__main__":
    main()
