@echo off
REM Jupiter Shot — Kuwait Laptop MoE CUDA Validation
REM
REM Usage: scripts\windows\run_moe_validation.bat [config] [steps]

setlocal
cd /d "%~dp0..\.."

set CONFIG=%1
set STEPS=%2

if "%CONFIG%"=="" (
    if exist "benchmarks\results\laptop\preflight.json" (
        for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('benchmarks/results/laptop/preflight.json')); print(d.get('recommended_moe_config','laptop_moe_small'))"') do set CONFIG=%%i
        echo [AUTO] Selected MoE config from preflight: %CONFIG%
    ) else (
        set CONFIG=laptop_moe_small
        echo [DEFAULT] Using config: %CONFIG%
    )
)

if "%STEPS%"=="" set STEPS=100

echo ============================================================
echo   Jupiter Shot - MoE CUDA Validation
echo   Config: %CONFIG%
echo   Steps:  %STEPS%
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

echo [STAGE 1] Running 10 MoE diagnostic steps...
python scripts\run_laptop_moe.py --config %CONFIG% --steps 10
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] MoE diagnostic failed.
    pause
    exit /b 1
)

echo.
echo [STAGE 1] MoE Diagnostic PASSED.
echo.

if %STEPS% GTR 10 (
    echo [STAGE 2] Running %STEPS% MoE steps...
    if %STEPS% GTR 100 (
        python scripts\run_laptop_moe.py --config %CONFIG% --steps %STEPS% --confirmed
    ) else (
        python scripts\run_laptop_moe.py --config %CONFIG% --steps %STEPS%
    )
    if %ERRORLEVEL% NEQ 0 (
        echo [FAIL] MoE training failed. Check benchmarks\results\laptop\errors.jsonl
        pause
        exit /b 1
    )
)

echo.
echo [DONE] MoE validation complete.
echo Results: benchmarks\results\laptop\moe_summary.json
echo.
pause
exit /b 0
