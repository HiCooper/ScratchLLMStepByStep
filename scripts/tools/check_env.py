"""打印当前训练环境的基本信息，便于排查环境/硬件问题。

覆盖训练最关心的几块：Python、PyTorch/CUDA/cuDNN、GPU（型号/显存/算力）、CPU、内存，
以及主要依赖包版本。

用法：
    python scripts/tools/check_env.py
"""
import platform
import sys


def pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "(未安装)"


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return [("GPU", "无可用 CUDA GPU")]
        rows = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            mem = props.total_memory / 1024 ** 3
            rows.append((f"GPU[{i}]", f"{name}  显存 {mem:.1f}GiB  sm_{props.major}{props.minor}"))
        return rows
    except Exception as e:
        return [("GPU", f"查询失败: {e}")]


def cpu_info():
    import os
    n = os.cpu_count() or "?"
    model = "?"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return [("CPU", f"{model}  x{n} 逻辑核")]


def mem_info():
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                if line.startswith("MemTotal") or line.startswith("MemAvailable"):
                    key, val = line.split(":")
                    info[key] = int(val.strip().split()[0]) // 1024  # MiB
        total = info.get("MemTotal", "?")
        avail = info.get("MemAvailable", "?")
        return [("内存", f"{total}MiB 总 / {avail}MiB 可用")]
    except Exception:
        return [("内存", "查询失败")]


def main():
    print("=" * 62)
    print("Python 环境")
    print("=" * 62)
    print(f"Python    : {platform.python_version()}  ({sys.executable})")
    print(f"平台      : {platform.platform()}")

    print()
    print("=" * 62)
    print("PyTorch / CUDA / GPU")
    print("=" * 62)
    tv = pkg_version("torch")
    print(f"torch     : {tv}")
    if tv != "(未安装)":
        import torch
        print(f"CUDA 可用 : {torch.cuda.is_available()}")
        print(f"CUDA 版本 : {torch.version.cuda}")
        print(f"cuDNN     : {torch.backends.cudnn.version()}")
        print(f"GPU 数量  : {torch.cuda.device_count()}")
        for k, v in gpu_info():
            print(f"  {k}   : {v}")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            sm = major * 10 + minor
            support = "支持 bf16 / flash-attn" if sm >= 80 else "不支持 bf16 / flash-attn（需 Ampere+，sm>=80）"
            print(f"算力      : sm_{major}{minor}  →  {support}")

    print()
    print("=" * 62)
    print("CPU / 内存")
    print("=" * 62)
    for k, v in cpu_info() + mem_info():
        print(f"{k}   : {v}")

    print()
    print("=" * 62)
    print("主要依赖包版本")
    print("=" * 62)
    for name in ("torch", "torchvision", "transformers", "tokenizers", "numpy",
                 "modelscope", "datasets", "pytest", "tensorboard", "matplotlib",
                 "pandas", "regex", "flash_attn"):
        print(f"  {name:<12}: {pkg_version(name)}")


if __name__ == "__main__":
    main()
