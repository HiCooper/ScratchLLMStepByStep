#!/bin/bash
# MiniGPT 环境体检入口（包装 preflight.py，便于 agent 直接调用）
# 用法： bash skills/minigpt-train/scripts/preflight.sh [--no-tests] [--quick] [--json PATH]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec python3 "$ROOT/skills/minigpt-train/scripts/preflight.py" "$@"
