@echo off
REM Jupiter Shot — Kuwait Laptop Full Validation Suite
REM
REM Runs all validation stages in order:
REM   1. Preflight check
REM   2. Dense CUDA validation (10 diagnostic + 100 steps)
REM   3. MoE CUDA validation (10 diagnostic + 100 steps)
REM   4. Checkpoint resume test
REM   5. Metrics collection
REM   6. Draft report generation
REM
REM Usage: scripts\windows\run_all_laptop_validation.bat
REM
REM To run 1000 steps: scripts\windows\run_all_laptop_validation.bat 1000

setlocal
cd /d "%~dp0..\.."

set STEPS=%1
if "%STEPS%"=="" set STEPS=100

echo ============================================================
echo   Jupiter Shot — Kuwait Laptop Full Validation Suite
echo   Steps per model: %STEPS%
echo   Date: %DATE% %TIME%
echo ============================================================
echo.
echo This script will:
echo   1. Check your GPU hardware
echo   2. Run dense transformer training (%STEPS% steps)
echo   3. Run MoE training (%STEPS% steps)
echo   4. Test checkpoint save and resume
echo   5. Generate validation report
echo.
echo Estimated time: 15–60 minutes depending on GPU and step count.
echo.
set /p CONFIRM="Type YES to proceed: "
if /i not "%CONFIRM%"=="YES" (
    echo Aborted.
    exit /b 1
)

if exist "venv\Scripts\activate.bat" call venv\Scripts\activate.bat

set PASS_COUNT=0
set FAIL_COUNT=0
set LOG_FILE=logs\laptop\validation_run_%DATE:~-4,4%%DATE:~-7,2%%DATE:~-10,2%.log

mkdir logs\laptop 2>nul

echo. >> %LOG_FILE%
echo === Jupiter Shot Validation Run === >> %LOG_FILE%
echo Date: %DATE% %TIME% >> %LOG_FILE%
echo Steps: %STEPS% >> %LOG_FILE%

REM ── Stage 1: Preflight ────────────────────────────────────────────────────
echo.
echo [STAGE 1/6] Hardware Preflight...
python scripts\laptop_gpu_preflight.py --output-dir benchmarks\results\laptop >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Preflight failed. CUDA not available or insufficient VRAM.
    echo        Cannot proceed with GPU validation.
    echo        See: benchmarks\results\laptop\preflight.txt
    pause
    exit /b 1
)
echo [PASS] Preflight
set /a PASS_COUNT+=1

REM Read recommended configs
for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('benchmarks/results/laptop/preflight.json')); print(d.get('recommended_dense_config','laptop_dense_small'))"') do set DENSE_CONFIG=%%i
for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('benchmarks/results/laptop/preflight.json')); print(d.get('recommended_moe_config','laptop_moe_small'))"') do set MOE_CONFIG=%%i

echo     Dense config: %DENSE_CONFIG%
echo     MoE config:   %MOE_CONFIG%

REM ── Stage 2: Dense validation ─────────────────────────────────────────────
echo.
echo [STAGE 2/6] Dense CUDA Validation (%STEPS% steps, config: %DENSE_CONFIG%)...
if %STEPS% GTR 100 (
    python scripts\run_laptop_dense.py --config %DENSE_CONFIG% --steps %STEPS% --confirmed >> %LOG_FILE% 2>&1
) else (
    python scripts\run_laptop_dense.py --config %DENSE_CONFIG% --steps %STEPS% >> %LOG_FILE% 2>&1
)
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Dense validation failed.
    set /a FAIL_COUNT+=1
) else (
    echo [PASS] Dense validation
    set /a PASS_COUNT+=1
)

REM ── Stage 3: MoE validation ───────────────────────────────────────────────
echo.
echo [STAGE 3/6] MoE CUDA Validation (%STEPS% steps, config: %MOE_CONFIG%)...
if %STEPS% GTR 100 (
    python scripts\run_laptop_moe.py --config %MOE_CONFIG% --steps %STEPS% --confirmed >> %LOG_FILE% 2>&1
) else (
    python scripts\run_laptop_moe.py --config %MOE_CONFIG% --steps %STEPS% >> %LOG_FILE% 2>&1
)
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] MoE validation failed.
    set /a FAIL_COUNT+=1
) else (
    echo [PASS] MoE validation
    set /a PASS_COUNT+=1
)

REM ── Stage 4: Resume test ──────────────────────────────────────────────────
echo.
echo [STAGE 4/6] Checkpoint Resume Test...
python scripts\run_laptop_resume_test.py --config %DENSE_CONFIG% --steps 20 >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Resume test failed.
    set /a FAIL_COUNT+=1
) else (
    echo [PASS] Resume test
    set /a PASS_COUNT+=1
)

REM ── Stage 5: Collect metrics ──────────────────────────────────────────────
echo.
echo [STAGE 5/6] Collecting metrics...
python scripts\collect_laptop_metrics.py --results-dir benchmarks\results\laptop >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Metrics collection had errors (non-fatal).
) else (
    echo [PASS] Metrics collected
    set /a PASS_COUNT+=1
)

REM ── Stage 6: Generate report ──────────────────────────────────────────────
echo.
echo [STAGE 6/6] Generating validation draft report...
python scripts\generate_laptop_validation_draft.py >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Report generation had errors (non-fatal).
) else (
    echo [PASS] Report generated: docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md
    set /a PASS_COUNT+=1
)

REM ── Final summary ─────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   VALIDATION COMPLETE
echo   Passed: %PASS_COUNT%  Failed: %FAIL_COUNT%
echo ============================================================
echo.
echo Results:
echo   benchmarks\results\laptop\preflight.json
echo   benchmarks\results\laptop\dense_summary.json
echo   benchmarks\results\laptop\moe_summary.json
echo   benchmarks\results\laptop\resume_test.json
echo   docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md
echo   %LOG_FILE%
echo.
echo Next step: Share docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md
echo            with the Jupiter Shot team.
echo.
pause
exit /b %FAIL_COUNT%
