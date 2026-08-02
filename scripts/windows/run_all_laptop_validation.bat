@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Jupiter Shot — Kuwait Laptop GPU Validation
:: run_all_laptop_validation.bat
::
:: What this script validates:
::   - CUDA execution on a single NVIDIA GPU
::   - Dense transformer training (small config)
::   - Small MoE routing (8-expert prototype)
::   - Checkpoint save, interrupt, and resume
::   - Metrics collection and thermal monitoring
::
:: What this script does NOT validate:
::   - 8x A100 distributed training
::   - DeepSpeed NCCL multi-node communication
::   - Full 1.3B parameter training run
::   - 20T scalability
::
:: A successful run authorizes the next controlled validation stage.
:: It does NOT automatically authorize 47B MoE training.
::
:: Requirements:
::   - Windows 10/11 with NVIDIA GPU (CUDA-capable)
::   - NVIDIA drivers installed (CUDA toolkit NOT required — PyTorch bundles CUDA)
::   - Python 3.10 or 3.11 installed and on PATH
::   - Internet access for initial package download (~2 GB)
::   - 20 GB free disk space
::
:: This script creates an ISOLATED virtual environment (.venv) in the project
:: directory. It does NOT modify the system-wide Python installation or
:: install/modify the system-wide CUDA toolkit.
:: ============================================================================

echo.
echo ============================================================
echo  Jupiter Shot - Kuwait Laptop GPU Validation
echo ============================================================
echo.

:: ----------------------------------------------------------------------------
:: Step 0: Locate project root
:: ----------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." && cd /d "%CD%"
set "PROJECT_ROOT=%CD%"
popd

echo [INFO] Project root: %PROJECT_ROOT%
echo.

:: ----------------------------------------------------------------------------
:: Step 1: Check Python is available
:: ----------------------------------------------------------------------------
echo [STEP 1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo         Ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [OK]   %PYTHON_VERSION%
echo.

:: ----------------------------------------------------------------------------
:: Step 2: Create isolated virtual environment (does NOT touch system Python)
:: ----------------------------------------------------------------------------
set "VENV_DIR=%PROJECT_ROOT%\.venv"
echo [STEP 2/7] Setting up isolated virtual environment at .venv\...

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [OK]   Virtual environment already exists, reusing it.
) else (
    echo [INFO] Creating new virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Ensure python -m venv is available (Python 3.10+).
        pause
        exit /b 1
    )
    echo [OK]   Virtual environment created.
)

:: Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK]   Virtual environment activated.
echo.

:: ----------------------------------------------------------------------------
:: Step 3: Install pinned dependencies (isolated to .venv, NOT system-wide)
:: ----------------------------------------------------------------------------
echo [STEP 3/7] Installing pinned dependencies into virtual environment...
echo [INFO] This downloads ~2 GB on first run. Subsequent runs are fast.
echo.

python -m pip install --upgrade pip --quiet

:: Install PyTorch with bundled CUDA 12.1 (no system CUDA toolkit needed)
echo [INFO] Installing PyTorch 2.2.2 with bundled CUDA 12.1...
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch. Check internet connection.
    pause
    exit /b 1
)

echo [INFO] Installing project dependencies from requirements.txt...
pip install -r "%PROJECT_ROOT%\requirements.txt" --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt dependencies.
    pause
    exit /b 1
)

echo [OK]   All dependencies installed.
echo.

:: ----------------------------------------------------------------------------
:: Step 4: Verify PyTorch CUDA detection (STOP if CUDA not available)
:: ----------------------------------------------------------------------------
echo [STEP 4/7] Verifying PyTorch CUDA detection...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[OK]   CUDA available: ' + torch.cuda.get_device_name(0))"
if errorlevel 1 (
    echo.
    echo [FAIL] PyTorch cannot detect a CUDA-capable GPU.
    echo.
    echo        Possible causes:
    echo          1. No NVIDIA GPU in this machine
    echo          2. NVIDIA drivers not installed or outdated
    echo             Download: https://www.nvidia.com/Download/index.aspx
    echo          3. GPU does not support CUDA (must be compute capability 3.7+)
    echo.
    echo        This validation requires a CUDA-capable NVIDIA GPU on Windows.
    echo        The script cannot continue without CUDA.
    echo.
    pause
    exit /b 1
)
echo.

:: ----------------------------------------------------------------------------
:: Step 5: Run preflight (STOP if preflight fails)
:: ----------------------------------------------------------------------------
echo [STEP 5/7] Running hardware preflight check...
echo.

mkdir "%PROJECT_ROOT%\benchmarks\results\laptop" 2>nul
mkdir "%PROJECT_ROOT%\docs\generated" 2>nul

python "%PROJECT_ROOT%\scripts\laptop_gpu_preflight.py" --output-dir "%PROJECT_ROOT%\benchmarks\results\laptop"
if errorlevel 1 (
    echo.
    echo [FAIL] Preflight check failed.
    echo        Review the output above before proceeding.
    echo        Do not run validation on a system that fails preflight.
    echo.
    pause
    exit /b 1
)
echo.
echo [OK]   Preflight passed.
echo.

:: Read recommended configs from preflight output
for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json')); print(d.get('recommended_dense_config','laptop_dense_small'))" 2^>nul') do set DENSE_CONFIG=%%i
for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json')); print(d.get('recommended_moe_config','laptop_moe_small'))" 2^>nul') do set MOE_CONFIG=%%i
if "!DENSE_CONFIG!"=="" set DENSE_CONFIG=laptop_dense_small
if "!MOE_CONFIG!"=="" set MOE_CONFIG=laptop_moe_small

echo [INFO] Selected dense config: !DENSE_CONFIG!
echo [INFO] Selected MoE config:   !MOE_CONFIG!
echo.

:: ----------------------------------------------------------------------------
:: Step 6: Confirm before running longer tests
:: ----------------------------------------------------------------------------
echo [STEP 6/7] Confirmation required before running GPU training tests.
echo.
echo   The following tests will run:
echo     - Dense transformer training  (estimated: 20-60 minutes)
echo     - MoE routing validation      (estimated: 20-60 minutes)
echo     - Checkpoint resume test      (estimated: 5-15 minutes)
echo.
echo   Total estimated time: 45 minutes to 2 hours depending on GPU speed.
echo   GPU will run at high utilization. Ensure adequate cooling.
echo.
set /p CONFIRM="   Type YES to continue, or press Enter to cancel: "
if /i not "!CONFIRM!"=="YES" (
    echo.
    echo [INFO] Validation cancelled by user.
    echo        Run this script again when ready.
    pause
    exit /b 0
)
echo.

:: ----------------------------------------------------------------------------
:: Step 7: Run validation tests
:: ----------------------------------------------------------------------------
echo [STEP 7/7] Running GPU validation tests...
echo.

set PASS_COUNT=0
set FAIL_COUNT=0

:: --- Dense validation ---
echo [TEST 1/3] Dense transformer training (!DENSE_CONFIG!)...
python "%PROJECT_ROOT%\scripts\run_laptop_dense.py" --config !DENSE_CONFIG! --steps 100
if errorlevel 1 (
    echo [FAIL] Dense training validation failed. Review output above.
    set /a FAIL_COUNT+=1
) else (
    echo [OK]   Dense training validation complete.
    set /a PASS_COUNT+=1
)
echo.

:: --- MoE validation ---
echo [TEST 2/3] MoE routing validation (!MOE_CONFIG!)...
python "%PROJECT_ROOT%\scripts\run_laptop_moe.py" --config !MOE_CONFIG! --steps 100
if errorlevel 1 (
    echo [FAIL] MoE validation failed. Review output above.
    set /a FAIL_COUNT+=1
) else (
    echo [OK]   MoE validation complete.
    set /a PASS_COUNT+=1
)
echo.

:: --- Resume test ---
echo [TEST 3/3] Checkpoint resume test...
python "%PROJECT_ROOT%\scripts\run_laptop_resume_test.py" --config !DENSE_CONFIG! --steps 20
if errorlevel 1 (
    echo [FAIL] Checkpoint resume test failed. Review output above.
    set /a FAIL_COUNT+=1
) else (
    echo [OK]   Checkpoint resume test complete.
    set /a PASS_COUNT+=1
)
echo.

:: --- Collect metrics ---
python "%PROJECT_ROOT%\scripts\collect_laptop_metrics.py" --results-dir "%PROJECT_ROOT%\benchmarks\results\laptop" >nul 2>&1

:: --- Generate report ---
echo [REPORT] Generating validation report...
python "%PROJECT_ROOT%\scripts\generate_laptop_validation_draft.py"
if errorlevel 1 (
    echo [WARN] Report generation encountered an error.
    echo        Raw metrics are still in benchmarks\results\laptop\
) else (
    echo [OK]   Report generated.
)

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo.
echo ============================================================
echo  VALIDATION COMPLETE
echo  Passed: !PASS_COUNT!   Failed: !FAIL_COUNT!
echo ============================================================
echo.
echo  Generated report (send this to the team):
echo    %PROJECT_ROOT%\docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md
echo.
echo  Raw metrics:
echo    %PROJECT_ROOT%\benchmarks\results\laptop\
echo.
echo  IMPORTANT:
echo    A successful result authorizes the next controlled validation stage.
echo    It does NOT automatically authorize 47B MoE training.
echo    The team will review the report before deciding next steps.
echo.
if !FAIL_COUNT! GTR 0 (
    echo  [WARN] !FAIL_COUNT! test(s) failed. Review the output above.
)
pause
exit /b !FAIL_COUNT!
