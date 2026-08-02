@echo off
REM Jupiter Shot — Kuwait Laptop Checkpoint Resume Test
REM
REM Usage: scripts\windows\run_resume_validation.bat [config]

setlocal
cd /d "%~dp0..\.."

set CONFIG=%1
if "%CONFIG%"=="" (
    if exist "benchmarks\results\laptop\preflight.json" (
        for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('benchmarks/results/laptop/preflight.json')); print(d.get('recommended_dense_config','laptop_dense_small'))"') do set CONFIG=%%i
    ) else (
        set CONFIG=laptop_dense_small
    )
)

echo ============================================================
echo   Jupiter Shot - Checkpoint Resume Test
echo   Config: %CONFIG%
echo ============================================================
echo.

if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

python scripts\run_laptop_resume_test.py --config %CONFIG% --steps 20
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
    echo [PASS] Resume test passed.
) else (
    echo [FAIL] Resume test failed. Check benchmarks\results\laptop\resume_test.json
)

echo Results: benchmarks\results\laptop\resume_test.json
echo.
pause
exit /b %EXIT_CODE%
