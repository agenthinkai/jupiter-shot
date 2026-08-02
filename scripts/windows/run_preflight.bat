@echo off
REM Jupiter Shot — Kuwait Laptop GPU Preflight
REM Run this first before any validation.
REM
REM Usage: scripts\windows\run_preflight.bat

setlocal

REM Navigate to repo root (two levels up from scripts\windows\)
cd /d "%~dp0..\.."

echo ============================================================
echo   Jupiter Shot - Kuwait Laptop GPU Preflight
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment (.venv) activated
) else (
    echo [WARN] No .venv found. Using system Python.
    echo        Run scripts\windows\setup_laptop_environment.ps1 first.
)

echo.
echo Running hardware preflight check...
echo.

python scripts\laptop_gpu_preflight.py --output-dir benchmarks\results\laptop
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo [PASS] Preflight complete. Proceed with validation.
    echo        Run: scripts\windows\run_all_laptop_validation.bat
) else if %EXIT_CODE% EQU 1 (
    echo [FAIL] CUDA is not available.
    echo        Ensure NVIDIA drivers are installed and the GPU is recognized.
    echo        Run: nvidia-smi to check GPU status.
) else (
    echo [FAIL] Insufficient VRAM for any training configuration.
    echo        Check benchmarks\results\laptop\preflight.txt for details.
)

echo.
echo Results saved to: benchmarks\results\laptop\preflight.json
echo                   benchmarks\results\laptop\preflight.txt
echo.
pause
exit /b %EXIT_CODE%
