#
# Sensory Perceiver Experiment Launcher (PowerShell)
#
# Usage:
#   .\run_experiments.ps1                       # Quick test
#   .\run_experiments.ps1 -Mode full            # Full benchmark
#   .\run_experiments.ps1 -Mode quick -Parallel # Parallel GPU usage
#   .\run_experiments.ps1 -Mode full -Push      # Full + auto-commit to GitHub
#

param(
    [ValidateSet("quick", "full", "monitor")]
    [string]$Mode = "quick",

    [switch]$Parallel,
    [switch]$Push,
    [int]$NumGPUs = 3
)

# Configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Sensory Perceiver Experiment Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Install Python 3.8+ and add to PATH." -ForegroundColor Red
    exit 1
}

# Check CUDA availability
Write-Host "Checking GPU availability..."
$gpuInfo = & python -c @"
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"@

Write-Host $gpuInfo
Write-Host ""

# Configuration summary
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Mode: $Mode"
Write-Host "  Parallel: $(if ($Parallel) { 'Yes' } else { 'No' })"
Write-Host "  Push to GitHub: $(if ($Push) { 'Yes' } else { 'No' })"
Write-Host "  GPUs: $NumGPUs"
Write-Host ""

# Estimate runtime
$runtimeEstimates = @{
    "quick" = "4 hours (IoT with 10 epochs)"
    "full"  = "40+ hours (3 datasets × 3 seeds × 50 epochs)"
}

if ($runtimeEstimates.ContainsKey($Mode)) {
    Write-Host "Estimated runtime: $($runtimeEstimates[$Mode])" -ForegroundColor Yellow
}

Write-Host ""

# Create results directory
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Build command
$cmdArgs = @(
    "experiment_orchestrator.py"
    "--mode", $Mode
    "--n-gpus", $NumGPUs
)

if ($Parallel) {
    $cmdArgs += "--parallel"
}

if ($Push) {
    $cmdArgs += "--push-to-github"
}

# Run orchestrator
Write-Host "Starting experiment orchestrator..." -ForegroundColor Cyan
& python @cmdArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "✗ Experiment failed. Check experiment_orchestrator.log" -ForegroundColor Red
    exit 1
}

# Final summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Experiments Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to: results/" -ForegroundColor Green
Write-Host "Monitor dashboard: monitor_dashboard.html" -ForegroundColor Green
Write-Host "Log file: experiment_orchestrator.log" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open monitor_dashboard.html in a browser to track progress"
Write-Host "  2. Results will be auto-pushed to GitHub if --Push was set"
Write-Host "  3. Check results/ for detailed metrics and checkpoints"
Write-Host ""
