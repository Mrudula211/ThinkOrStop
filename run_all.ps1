<#
Think-or-Stop: full CIFAR-10 pipeline in one command.

    powershell -ExecutionPolicy Bypass -File .\run_all.ps1
    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Seeds 0 -Epochs 30

Steps: download CIFAR-10 (with retries) -> GPU check -> for each seed: train
baseline + multi-exit, evaluate both (synthetic corruptions + latency), plot
-> multi-seed report. Stops at the first failing step.
#>
param(
    [int[]]$Seeds = @(0, 1, 2),
    [int]$Epochs = 30,
    [int]$BatchSize = 128,
    [double]$Threshold = 0.9,
    [double]$MaxDrop = 0.01,
    [int]$DownloadRetries = 5
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Virtual environment not found: $py" }

function Invoke-Step([string]$Desc, [string[]]$CmdArgs) {
    Write-Host "`n==> $Desc" -ForegroundColor Cyan
    & $py @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "Step failed (exit $LASTEXITCODE): $Desc" }
}

# 1. Dataset download; retried because the CIFAR-10 server connection is flaky.
for ($i = 1; $i -le $DownloadRetries; $i++) {
    Write-Host "`n==> Download CIFAR-10 (attempt $i/$DownloadRetries)" -ForegroundColor Cyan
    & $py -m think_or_stop.download --dataset cifar10
    if ($LASTEXITCODE -eq 0) { break }
    if ($i -eq $DownloadRetries) { throw "CIFAR-10 download failed after $DownloadRetries attempts" }
    Start-Sleep -Seconds 10
}

# 2. GPU verification + one-batch forward/backward smoke test.
Invoke-Step "GPU check" @("-m", "think_or_stop.gpu_check", "--device", "cuda")

# 3. Train, evaluate and plot each seed.
$metricDirs = @()
foreach ($s in $Seeds) {
    foreach ($m in @("baseline", "multiexit")) {
        Invoke-Step "Train $m (seed $s)" @("-m", "think_or_stop.train", "--dataset", "cifar10", "--model", $m,
            "--device", "cuda", "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--seed", "$s")
    }
    Invoke-Step "Evaluate baseline (seed $s)" @("-m", "think_or_stop.evaluate",
        "--checkpoint", "checkpoints/cifar10_baseline_s$s/best.pt", "--device", "cuda", "--corruptions", "synthetic")
    Invoke-Step "Evaluate multi-exit (seed $s)" @("-m", "think_or_stop.evaluate",
        "--checkpoint", "checkpoints/cifar10_multiexit_s$s/best.pt", "--device", "cuda", "--threshold", "$Threshold",
        "--select-threshold", "--constraint", "relative", "--max-drop", "$MaxDrop", "--corruptions", "synthetic")
    Invoke-Step "Plots (seed $s)" @("-m", "think_or_stop.plots", "--metrics-dir", "results/metrics/cifar10_multiexit_s$s",
        "--baseline-dir", "results/metrics/cifar10_baseline_s$s")
    Invoke-Step "Plots baseline (seed $s)" @("-m", "think_or_stop.plots", "--metrics-dir", "results/metrics/cifar10_baseline_s$s")
    $metricDirs += "results/metrics/cifar10_baseline_s$s", "results/metrics/cifar10_multiexit_s$s"
}

# 4. Aggregate across seeds (mean +/- std).
Invoke-Step "Report" (@("-m", "think_or_stop.report", "--metrics-dirs") + $metricDirs + @("--out", "results/metrics/report_cifar10"))

Write-Host "`nDone. Report: results/metrics/report_cifar10.md | Figures: results/figures/ | Checkpoints: checkpoints/" -ForegroundColor Green
