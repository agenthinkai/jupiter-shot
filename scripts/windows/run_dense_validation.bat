@echo off
REM Jupiter Shot — Kuwait Laptop Dense CUDA Validation
REM
REM Stages:
REM   1. 10 diagnostic steps
REM   2. 100 steps if stable
REM   3. Prompts for 1000 steps
REM
REM Usage: scripts\windows\run_dense_validation.bat [config] [steps]
REM   config: laptop_dense_tiny | laptop_dense_small | laptop_dense_medium
REM   steps:  10 | 100 | 1000

setlocal

cd /d "%~dp0..\.."

REM Default config (auto-selected from preflight if not specified)
set CONFIG=%1
set STEPS=%2

if "%CONFIG%"=="" (
    REM Try to read recommended config from preflight
    if exist "benchmarks\results\laptop\preflight.json" (
        for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('benchmarks/results/laptop/preflight.json')); print(d.get('recommended_dense_config','laptop_dense_small'))"') do set CONFIG=%%i
        echo [AUTO] Selected config from preflight: %CONFIG%
    ) else (
        set CONFIG=laptop_dense_small
        echo [DEFAULT] Using config: %CONFIG%
        echo           Run run_preflight.bat first for auto-selection.
    )
)

if "%STEPS%"=="" set STEPS=100

echo ============================================================
echo   Jupiter Shot - Dense CUDA Validation
echo   Config: %CONFIG%
echo   Steps:  %STEPS%
echo ============================================================
echo.

if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

echo [STAGE 1] Running 10 diagnostic steps...
python scripts\run_laptop_dense.py --config %CONFIG% --steps 10
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Diagnostic steps failed. Check logs above.
    pause
    exit /b 1
)

echo.
echo [STAGE 1] Diagnostic PASSED.
echo.

if %STEPS% GTR 10 (
    echo [STAGE 2] Running %STEPS% steps...
    if %STEPS% GTR 100 (
        python scripts\run_laptop_dense.py --config %CONFIG% --steps %STEPS% --confirmed
    ) else (
        python scripts\run_laptop_dense.py --config %CONFIG% --steps %STEPS%
    )
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] Training failed. Check benchmarks\results\laptop\errors.jsonl
        pause
        exit /b 1
    )
)

echo.
echo [DONE] Dense validation complete.
echo Results: benchmarks\results\laptop\dense_summary.json
echo.
pause
exit /b 0
