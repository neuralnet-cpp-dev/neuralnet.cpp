# run_with_vram.ps1 — 运行一个程序并用 nvidia-smi 采样整卡显存峰值
# 用法: ./bench/run_with_vram.ps1 -Exe <exe> -Arguments <args[]> -Log <程序日志>
# 输出: <Log>.vram (每 200ms 一个 MiB 采样) + 峰值打印到 stdout
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Log
)

$vramFile = "$Log.vram"
$outFile = "$Log.stdout"
$errFile = "$Log.stderr"

# nvidia-smi 的解析：先按名字，失败则回落到 Windows 标准路径
# （SystemRoot\system32 —— 本机实测按名字解析得到的也是这一份）
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi) { $smi = Join-Path $env:SystemRoot 'system32\nvidia-smi.exe' }

# NVML 故障探测：nvidia-smi 初始化失败（如 "Failed to initialize NVML: Unknown Error"）
# 时回落到 PDH 计数器 "\GPU Adapter Memory(*)\Dedicated Usage"（整卡 dedicated 用量，
# 取全部 GPU 适配器最大值 ≈ 旧口径的 NVIDIA 整卡 memory.used；2026-11-12 实测本机
# NVML 不可用而 Vulkan/CUDA 正常时的等价采样）
$usePdh = $false
try {
    $probe = (& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if (-not ("$probe".Trim() -match '^\d+$')) { $usePdh = $true }
} catch { $usePdh = $true }
if ($usePdh) { Write-Host "[run_with_vram] nvidia-smi/NVML 不可用，改用 PDH GPU Adapter Memory 采样" }

# ⚠ 采样必须在**前台**做：早期版本用 Start-Job 轮询（job 的 PATH/会话与前台
# 不同 → 实测采到 0 条样本），改用 nvidia-smi 的 -lms 循环模式也受进程生命周期
# 影响。当前实现 = 被测程序后台跑 + 前台每 ~150ms 取一次 memory.used。
$proc = Start-Process -FilePath $Exe -ArgumentList $Arguments `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile `
    -NoNewWindow -PassThru

[System.Collections.Generic.List[int]] $samples = @()
while (-not $proc.HasExited) {
    if ($usePdh) {
        try {
            $cs = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
            $maxBytes = ($cs | Measure-Object -Property CookedValue -Maximum).Maximum
            if ($maxBytes -ne $null) { $samples.Add([int][math]::Round($maxBytes / 1MB)) }
        } catch { }
    } else {
        $v = (& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
        if ("$v".Trim() -match '^\d+$') { $samples.Add([int]"$v".Trim()) }
    }
    Start-Sleep -Milliseconds 150
}
$proc.WaitForExit()
$exitCode = $proc.ExitCode

# 日志合并回单一 $Log（保持调用方的既有约定）
$merged = @()
if (Test-Path $outFile) { $merged += Get-Content $outFile }
if (Test-Path $errFile) { $merged += Get-Content $errFile }
Set-Content -Path $Log -Value $merged
Set-Content -Path $vramFile -Value $samples
Remove-Item -Path $outFile, $errFile -Force -ErrorAction SilentlyContinue

$peak = ($samples | Measure-Object -Maximum).Maximum
# 峰值落盘（同名 .peak）：调用方从文件读峰值，避免解析 Write-Host 输出
Set-Content -Path "$Log.peak" -Value $peak
Write-Host "=== $Exe exit=$exitCode  VRAM samples=$($samples.Count)  peak=${peak} MiB ==="
