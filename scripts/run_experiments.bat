@echo off
REM Sensory Perceiver Experiment Launcher (Windows)
REM
REM Usage:
REM   run_experiments.bat                  -- Quick test (1 experiment, 4h)
REM   run_experiments.bat full             -- Full benchmark (9 experiments, 40h)
REM   run_experiments.bat quick parallel   -- Quick test with parallel GPU usage
REM   run_experiments.bat full push        -- Full benchmark + auto-commit to GitHub
REM

setlocal enabledelayedexpansion

REM Colors for output
for /F %%A in ('copy /Z "%~f0" nul') do set "BS=%%A"

echo.
echo ========================================
echo  Sensory Perceiver Experiment Launcher
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

REM Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU devices: {torch.cuda.device_count()}')"

REM Parse arguments
set MODE=quick
set PARALLEL=
set PUSH_GITHUB=

if not "%1"=="" set MODE=%1
if not "%2"=="" (
    if "%2"=="parallel" set PARALLEL=--parallel
    if "%2"=="push" set PUSH_GITHUB=--push-to-github
)
if not "%3"=="" (
    if "%3"=="push" set PUSH_GITHUB=--push-to-github
)

echo Mode: %MODE%
if not "%PARALLEL%"=="" echo Parallel: Yes
if not "%PUSH_GITHUB%"=="" echo Push to GitHub: Yes
echo.

REM Create results directory
if not exist results mkdir results

REM Run orchestrator
echo Starting experiment orchestrator...
python -m src.experiments.experiment_orchestrator --mode %MODE% %PARALLEL% %PUSH_GITHUB%

if errorlevel 1 (
    echo.
    echo ERROR: Experiment failed. Check experiment_orchestrator.log
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Experiments Complete!
echo ========================================
echo.
echo Results saved to: results\
echo Monitor dashboard: monitor_dashboard.html
echo Log file: experiment_orchestrator.log
echo.
pause
