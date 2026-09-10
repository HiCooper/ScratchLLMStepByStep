# 把 WSL 发行版整体迁移到 D 盘（在 Windows PowerShell 中运行；建议管理员）
# 用法： powershell -ExecutionPolicy Bypass -File .\scripts\migrate_wsl_to_d.ps1 [-Distro Ubuntu-24.04] [-Target D:\WSL\Ubuntu-24.04]
param(
  [string]$Distro = "Ubuntu-24.04",
  [string]$Target = "D:\WSL\Ubuntu-24.04"
)
$ErrorActionPreference = "Stop"
Write-Host "== WSL 迁移 ==`n发行版: $Distro`n目标:   $Target`n"

Write-Host "[1/5] 当前状态"; wsl -l -v
Write-Host "[2/5] 检查 C 盘 vhdx 与 D 盘空间"
$vhdx = Get-ChildItem "$env:LOCALAPPDATA\wsl","$env:LOCALAPPDATA\Packages" -Recurse -Filter *.vhdx -ErrorAction SilentlyContinue |
  Sort-Object Length -Descending | Select-Object -First 1
if ($vhdx) { "{0:N1} GB  {1}" -f ($vhdx.Length/1GB), $vhdx.FullName }
$free = (Get-PSDrive D).Free; "{0:N1} GB free on D:" -f ($free/1GB)
if ($free -lt 220GB) { throw "D 盘空闲不足 220GB，终止" }

Write-Host "[3/5] 关闭 WSL（会中断其中的训练/会话）"
wsl --shutdown
Start-Sleep -Seconds 5

Write-Host "[4/5] 迁移到 $Target"
wsl --manage $Distro --move $Target

Write-Host "[5/5] 验证"
wsl -l -v
Get-ChildItem $Target -Recurse -Filter *.vhdx | ForEach-Object { "{0:N1} GB  {1}" -f ($_.Length/1GB), $_.FullName }
"C: free = {0:N1} GB" -f ((Get-PSDrive C).Free/1GB)
Write-Host "`n完成。进入 WSL 后按 scripts/migrate_wsl_to_d.md 的『迁移后恢复』重启训练/看板/守护。"
