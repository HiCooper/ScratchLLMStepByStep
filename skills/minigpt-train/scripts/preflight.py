#!/usr/bin/env python3
"""MiniGPT 环境体检（只读）：依赖 / GPU / 磁盘 / 语料 / 分词器 / 产物 / 测试 / 运行中服务。

输出：人类可读摘要 + JSON（默认 models/checkpoints/preflight.json）。
退出码：0 = 无阻塞项；1 = 存在 BLOCKER。
用法：python3 skills/minigpt-train/scripts/preflight.py [--no-tests] [--json PATH] [--quick]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
results: list[dict] = []


def add(level: str, name: str, detail: str, hint: str = ""):
    results.append({"level": level, "item": name, "detail": detail, "hint": hint})


def ok(name, detail=""):
    add("OK", name, detail)


def warn(name, detail="", hint=""):
    add("WARN", name, detail, hint)


def blocker(name, detail="", hint=""):
    add("BLOCKER", name, detail, hint)


def sh(cmd: str, timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def check_python_deps():
    add_py = sys.version.split()[0]
    if sys.version_info >= (3, 10):
        ok("python", f"{add_py} ({sys.executable})")
    else:
        blocker("python", f"{add_py} 过低", "需要 Python >= 3.10")
    deps = {"torch": "torch", "transformers": "transformers", "tokenizers": "tokenizers",
            "numpy": "numpy"}
    optional = {"tensorboard": "tensorboard", "matplotlib": "matplotlib", "pandas": "pandas"}
    import importlib
    for mod, pkg in deps.items():
        try:
            m = importlib.import_module(mod)
            extra = ""
            if mod == "torch":
                extra = f", cuda_available={m.cuda.is_available()}, cuda={m.version.cuda}"
            ok(f"dep:{pkg}", f"{getattr(m, '__version__', '?')}{extra}")
        except Exception as exc:  # noqa: BLE001
            blocker(f"dep:{pkg}", f"导入失败: {exc}", f"pip install {pkg}")
    for mod, pkg in optional.items():
        try:
            importlib.import_module(mod)
            ok(f"dep:{pkg}", "已安装")
        except Exception:  # noqa: BLE001
            warn(f"dep:{pkg}", "未安装", f"pip install {pkg}（TensorBoard/绘图/看板需要）")


def check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            warn("gpu", "CUDA 不可用，将以 CPU 训练（极慢）", "安装 CUDA 版 torch 或指定 --device cpu")
            return
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        free = None
        out = sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
        if out:
            free = int(out.splitlines()[0]) / 1024
        detail = f"{name}, {total:.1f}GB" + (f", free {free:.1f}GB" if free else "")
        if free is not None and free < 1.5:
            warn("gpu", detail, "显存接近占用上限：确认是否有训练在跑；新任务请先确认空闲显存")
        elif total < 5.5:
            warn("gpu", detail, "显存偏小：用 emb<=384/layers<=8/ctx<=512/batch<=4")
        elif total < 8:
            ok("gpu", detail + " → 建议 512/10/8 ctx512 bs8 fp16 + torch.compile")
        else:
            ok("gpu", detail + " → 可上 768/12/12 或多卡")
    except Exception as exc:  # noqa: BLE001
        warn("gpu", f"检测失败: {exc}")


def check_host():
    cpu = os.cpu_count() or 0
    ok("cpu", f"{cpu} 核")
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            mem_gb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1)) / 2**20
        (ok if mem_gb >= 12 else warn)("memory", f"{mem_gb:.1f}GB",
                                       "" if mem_gb >= 12 else "内存偏小：降 batch/ctx，或先只跑小语料")
    except Exception:  # noqa: BLE001
        pass
    return cpu, mem_gb


def check_disk():
    for path, label, threshold in [("/", "根分区", 20), ("/mnt/c", "Windows C:", 10)]:
        if not os.path.exists(path):
            continue
        try:
            st = os.statvfs(path)
            free_gb = st.f_bavail * st.f_frsize / 2**30
            if free_gb < threshold:
                warn(f"disk:{label}", f"可用 {free_gb:.1f}GB",
                     "长训练前请清理（scripts/checkpoint_janitor.sh）或把数据移到 D/E 盘")
            else:
                ok(f"disk:{label}", f"可用 {free_gb:.1f}GB")
        except Exception:  # noqa: BLE001
            pass


def check_assets():
    tok = os.path.join(ROOT, "models", "tokenizer_v3")
    if os.path.isdir(tok) and os.path.exists(os.path.join(tok, "tokenizer.json")):
        try:
            sys.path.insert(0, ROOT)
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(tok)
            ok("tokenizer", f"models/tokenizer_v3 vocab={len(t)} (bos={t.bos_token}, eos={t.eos_token})")
        except Exception as exc:  # noqa: BLE001
            warn("tokenizer", f"加载失败: {exc}")
    else:
        warn("tokenizer", "缺少 models/tokenizer_v3",
             "用 notebook 01 训练，或 python scripts/train_tokenizer.py --data dataset/pretrain_t2t.jsonl --output models/tokenizer_v3 --vocab-size 32000")
    # 项目只使用 tokenizer_v3；.bin 的 meta.vocab_size 必须与之相符（训练侧会强校验）
    for meta_name in ("pretrain_v4_full", "code_domain_dedup", "domain_code70_general30",
                      "domain_code50_general50", "finance_corpus", "math_clean"):
        meta_path = os.path.join(ROOT, "dataset", "bins", f"{meta_name}.meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            import json as _json
            with open(meta_path, encoding="utf-8") as f:
                bin_vocab = _json.load(f).get("vocab_size")
            if bin_vocab:
                ok("bin:vocab", f"{meta_name}.bin vocab_size={bin_vocab}")
        except Exception as exc:  # noqa: BLE001
            warn("bin:vocab", f"{meta_name} meta 读取失败: {exc}")

    corpus = os.path.join(ROOT, "dataset", "pretrain_t2t.jsonl")
    if os.path.exists(corpus):
        ok("corpus", f"dataset/pretrain_t2t.jsonl {os.path.getsize(corpus)/2**30:.2f}GB")
    else:
        warn("corpus", "缺少预训练语料", "bash scripts/download_data.sh")

    bins = sorted(glob.glob(os.path.join(ROOT, "dataset", "bins", "*.bin")))
    if bins:
        for b in bins:
            meta_p = os.path.splitext(b)[0] + ".meta.json"
            meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
            ok("bin", f"{os.path.basename(b)} {os.path.getsize(b)/2**20:.0f}MB "
                      f"tokens={meta.get('tokens', '?')} dtype={meta.get('dtype', 'uint16(默认)')}")
    else:
        warn("bin", "dataset/bins 下没有 .bin（预训练需要）",
             "python scripts/build_pretrain_bin.py build --corpus-jsonl dataset/pretrain_t2t.jsonl "
             "--tokenizer-dir models/tokenizer_v3 --out-bin dataset/bins/pretrain_v4_full.bin --max-lines 0")

    sfts = sorted(glob.glob(os.path.join(ROOT, "dataset", "sft", "*.jsonl")))
    if sfts:
        for s in sfts[:6]:
            n = sum(1 for _ in open(s, encoding="utf-8", errors="replace"))
            ok("sft-data", f"{os.path.basename(s)} {n} 行")
    else:
        warn("sft-data", "dataset/sft 下没有指令数据", "下载 deepctrl sft_data_zh.jsonl 或运行 build_cot_sft.py")

    runs = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", "*")))
    finals = [p for p in runs if os.path.exists(os.path.join(p, "final.pt"))]
    if finals:
        ok("checkpoints", f"已完成 run: {', '.join(os.path.basename(p) for p in finals)}")
    else:
        ok("checkpoints", "暂无 final.pt（首次训练会新建）")


def detect_preset(cpu, mem_gb):
    """按硬件给出可直接使用的训练预设（agent 无需人工调参）。

    返回 {"name", "reason", "model": {...}, "train": {...}, "launch": {...}, "expect"}
    """
    import platform
    import torch
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    vram = 0.0
    if gpu_count:
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30

    model = {"model_emb_dim": 512, "model_n_layers": 10, "model_n_heads": 8,
             "model_context_length": 512}
    train = {"train_batch_size": 8, "train_grad_accumulation_steps": 1,
             "train_learning_rate": 6e-4, "train_warmup_steps": 200,
             "train_mixed_precision_dtype": "float16", "train_torch_compile": True}
    launch = {"nproc": 1, "extra": []}

    if gpu_count == 0:
        name = "cpu"
        reason = "未检测到 CUDA GPU：可跑通但极慢，建议仅做小模型冒烟/小语料验证"
        model.update({"model_emb_dim": 128, "model_n_layers": 2, "model_n_heads": 4,
                      "model_context_length": 128})
        train.update({"train_batch_size": 8, "train_learning_rate": 1e-3,
                      "train_mixed_precision_dtype": "none", "train_torch_compile": False,
                      "train_eval_steps": 50, "train_save_steps": 200})
        expect = ("本机实测：128/2/4 ctx128 bs4 ≈ 0.17k tok/s、256/4/4 ctx256 bs2 ≈ 0.12k tok/s；"
                  "CPU 只适合冒烟/机制验证（小语料 + 数百步），想训可用模型请用 CUDA GPU")
        assumed = 3.0
    elif gpu_count >= 2:
        per = vram
        name = "multi-gpu"
        reason = f"检测到 {gpu_count} 张 GPU（每张约 {per:.1f}GB）：用 torchrun DDP，按 sqrt(N) 放大学习率"
        base_lr = 6e-4
        n = gpu_count
        if per < 8:
            model.update({"model_emb_dim": 384, "model_n_layers": 8, "model_context_length": 512})
            train["train_batch_size"] = 8
        elif per < 16:
            train["train_batch_size"] = 12
        else:
            model.update({"model_emb_dim": 768, "model_n_layers": 12, "model_n_heads": 12,
                          "model_context_length": 1024})
            train["train_batch_size"] = 16
        train["train_learning_rate"] = round(base_lr * (n ** 0.5), 6)
        launch = {"nproc": n,
                  "extra": ["--train_eval_steps", "2000", "--train_save_steps", "4000"]}
        expect = f"有效 batch = batch×{n}×accum；吞吐≈单卡×{n}×0.8-0.9（NCCL 同步开销）"
        assumed = 0.17
    elif vram < 5.5:
        name = "gpu-tiny"
        reason = f"单卡仅 {vram:.1f}GB：用最小可用配置，可临时降 ctx/emb"
        model.update({"model_emb_dim": 384, "model_n_layers": 8, "model_context_length": 384})
        train["train_batch_size"] = 4
        expect = "约 10-18k tok/s"
        assumed = 0.25
    elif vram < 8:
        name = "gpu-small"
        reason = f"单卡 {vram:.1f}GB：本仓库默认生产配置（512/10/8 ctx512 bs8 fp16 + compile）"
        expect = "约 22-30k tok/s（RTX2060 实测）"
        assumed = 0.17
    elif vram < 16:
        name = "gpu-mid"
        reason = f"单卡 {vram:.1f}GB：可放大 ctx/ batch"
        model["model_context_length"] = 1024
        train["train_batch_size"] = 12
        expect = "约 30-50k tok/s"
        assumed = 0.12
    else:
        name = "gpu-large"
        reason = f"单卡 {vram:.1f}GB：可上 768/12/12 + ctx1024"
        model.update({"model_emb_dim": 768, "model_n_layers": 12, "model_n_heads": 12,
                      "model_context_length": 1024})
        train["train_batch_size"] = 16
        expect = "约 60k+ tok/s"
        assumed = 0.06

    if mps:
        warn("gpu:mps", "检测到 Apple MPS，但本仓库 AMP/device 逻辑仅适配 CUDA/CPU",
             "将按 CPU 预设运行（速度慢），或改用 CUDA 机器")
    if mem_gb is not None and mem_gb < 10:
        train["train_batch_size"] = min(train["train_batch_size"], 4)
        reason += "；内存 <10GB，已下调 batch"
    # 估算用：单步耗时（秒）与每步 tokens（batch×ctx×nproc）
    ctx = model["model_context_length"]
    tps = train["train_batch_size"] * max(ctx - 1, 1) * max(launch.get("nproc", 1), 1)
    preset = {"name": name, "reason": reason, "gpu_count": gpu_count,
              "vram_gb": round(vram, 1), "model": model, "train": train,
              "launch": launch, "expect": expect,
              "assumed_s_per_step": assumed, "tokens_per_step": tps}
    add("PRESET", name, f"{reason} | 预期：{expect}")
    return preset


def check_tests(quick: bool):
    if quick:
        warn("tests", "已跳过（--quick）")
        return
    # 这里**不能**再写 `-q`：pyproject 的 addopts 已经是 `-q`，叠加成 `-qq` 后 pytest
    # 连 "213 passed" 汇总行都不打印（实测），加上末尾的 warnings summary 会把
    # `tail -2` 挤成 "-- Docs: ..."，于是门禁永远误报"pytest 输出异常"。
    # 正解：只依赖 addopts 的单个 -q，并从全量输出里正则匹配汇总行；
    # failed/error 一律升级为 BLOCKER（测试不过就不该开训）。
    out = sh(f"cd {ROOT} && python3 -m pytest tests/ 2>&1", timeout=900)
    m = re.search(r"^.*?(\d+ (?:passed|failed|error)\b.*)$", out, re.M)
    summary = m.group(1).strip() if m else ""
    if summary and "failed" not in summary and "error" not in summary:
        ok("tests", summary)
    elif summary:
        blocker("tests", summary, "cd 仓库根目录执行 pytest tests/ 查看失败用例")
    else:
        warn("tests", f"pytest 输出异常: {out[-200:]}",
             "cd 仓库根目录执行 pytest tests/ 查看详情")


def check_services():
    # 只认真实运行实体：排除受本命令启发的包装行（含 setsid/nohup）与自身进程
    rows = sh("ps -eo pid,args | grep -v grep").splitlines()

    def find(pat: str) -> str:
        import re as _re
        for line in rows:
            m = _re.match(r"\s*(\d+)\s+(.*)", line)
            if not m:
                continue
            pid, cmd = m.group(1), m.group(2)
            if pid == str(os.getpid()) or "setsid" in cmd or "nohup" in cmd:
                continue
            if _re.search(pat, cmd):
                return pid
        return ""

    for name, pat in [("trainer", r"minigpt\.train\.pretrainer"),
                      ("dashboard", r"train_dashboard\.py"),
                      ("janitor", r"checkpoint_janitor\.sh"),
                      ("resilient", r"train_pretrain_resilient\.sh"),
                      ("downstream", r"wait_and_run_downstream\.sh")]:
        alive = find(pat)
        (ok if alive else warn)("service", f"{name}: {'running pid=' + alive if alive else 'not running'}")
    for port in (8099, 6006):
        used = sh(f"(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -c ':{port} '")
        if used and used.strip() not in ("", "0"):
            ok("port", f"{port} 已监听")
        else:
            ok("port", f"{port} 空闲")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "models", "checkpoints", "preflight.json"))
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--quick", action="store_true", help="跳过 pytest")
    args = ap.parse_args()

    add("OK", "repo", ROOT)
    check_python_deps()
    check_gpu()
    cpu, mem = check_host()
    preset = detect_preset(cpu, mem)
    check_disk()
    check_assets()
    check_tests(args.quick or args.no_tests)
    check_services()

    icon = {"OK": "✅", "WARN": "⚠️ ", "BLOCKER": "❌", "PRESET": "🎯"}
    print(f"\n=== MiniGPT preflight @ {datetime.now():%F %T} ===")
    for r in results:
        line = f"{icon[r['level']]} {r['item']:<22} {r['detail']}"
        if r["hint"]:
            line += f"\n      ↳ {r['hint']}"
        print(line)
    blockers = [r for r in results if r["level"] == "BLOCKER"]
    warns = [r for r in results if r["level"] == "WARN"]
    print(f"\n汇总: OK={len(results)-len(blockers)-len(warns)} WARN={len(warns)} BLOCKER={len(blockers)}")
    print("结论:", "存在阻塞项，需先修复" if blockers else "环境可用，可继续训练/评测")

    payload = {"time": datetime.now().isoformat(timespec="seconds"), "repo": ROOT,
               "blockers": len(blockers), "warns": len(warns), "results": results,
               "preset": preset}
    try:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"JSON 报告: {args.json}")
    except OSError as exc:
        print(f"（JSON 写入失败: {exc}）")
    sys.exit(1 if blockers else 0)


if __name__ == "__main__":
    main()
