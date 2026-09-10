# 把 WSL 发行版整体迁移到 D 盘（Ubuntu-24.04）

> 适用于本机：WSL 2.7.13，发行版 `Ubuntu-24.04`，当前 vhdx 位于
> `C:\Users\hicooper\AppData\Local\wsl\{d33b36e4-3af0-4503-bbb8-6a38ddcf1ccc}\ext4.vhdx`（约 177 GB）。
> 目标：迁移到 `D:\WSL\Ubuntu-24.04`，释放 C 盘空间。

## ⚠️ 影响
- `wsl --shutdown` 会**停止整个 WSL**：本仓库里正在跑的训练、看板、自愈守护、以及 DSH 会话都会中断。
- 训练可从最近 checkpoint 续训（见下方「迁移后恢复」），不会丢已训练的权重。
- 迁移本质是复制 177GB（**不是**瞬间移动）：D 盘需 ≥200GB 空闲，耗时通常 5–20 分钟；期间不要断电。

## 迁移前（可选，强烈建议）
先瘦身再迁移，复制量更小、C 盘回收更彻底：
```bash
# 1) 把大数据移出 vhdx（本仓库已做：38G 语料 → /mnt/d/wsl-data 并软链回 dataset/）
# 2) 开启稀疏磁盘，让删除的空间真正还给 C 盘
wsl.exe --manage Ubuntu-24.04 --set-sparse true --allow-unsafe
```
再确认：
```bash
wsl.exe -l -v            # STATE 应为 Running/Stopped，NAME=Ubuntu-24.04
df -h /mnt/c | tail -1   # 记下迁移前可用空间
```

## 执行迁移（Windows PowerShell 或 CMD）
```powershell
wsl --shutdown
wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04
wsl -l -v                      # 确认发行版仍在且可启动
wsl -d Ubuntu-24.04            # 进入
```
> 若提示需要新版本 WSL：`wsl --update` 后重试。若 `--move` 不可用，退化为
> `wsl --export Ubuntu-24.04 D:\WSL\ubuntu-backup.tar` → `wsl --unregister Ubuntu-24.04` → `wsl --import Ubuntu-24.04 D:\WSL\Ubuntu-24.04 D:\WSL\ubuntu-backup.tar`。

本仓库也提供了 PowerShell 辅助脚本（在 Windows 侧执行）：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\migrate_wsl_to_d.ps1
```

## 迁移后恢复（在 WSL 里执行）
```bash
cd ~/projects/ScratchLLMStepByStep

# 1) 训练（自愈守护：自动从最近 checkpoint 续训）
setsid nohup bash scripts/train_pretrain_resilient.sh > models/checkpoints/pretrain_v2_full.watchdog.log 2>&1 < /dev/null &

# 2) 空间守护（每目录只留最近 2 个 checkpoint）
setsid nohup bash scripts/checkpoint_janitor.sh 60 2 120 > /tmp/janitor.log 2>&1 < /dev/null &

# 3) 下游自动接力（等 final.pt 出现后跑 SFT/CoT/评测）
setsid nohup bash scripts/wait_and_run_downstream.sh > models/checkpoints/wait_downstream.log 2>&1 < /dev/null &

# 4) 训练看板
setsid nohup python3 scripts/train_dashboard.py --serve --port 8099 --refresh 5 --grad-clip 1.0 > /tmp/dashboard.log 2>&1 < /dev/null &
# 打开 http://127.0.0.1:8099
```
另需重启 DSH（本会话的 Web 服务运行在 WSL 内，随 `wsl --shutdown` 一起停止）。

## 验证清单
```bash
wsl.exe -l -v                                   # 发行版正常
df -h / | tail -1                               # 根分区容量正常
df -h /mnt/c | tail -1                          # C 盘应释放约 170GB（迁移后）
ls /mnt/d/WSL/Ubuntu-24.04/ext4.vhdx            # vhdx 已落在 D 盘
nvidia-smi                                      # GPU 可用
python3 scripts/train_dashboard.py --once       # 看板数据正常
```
