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
::   - Internet access for initial package download (~2.5 GB for Blackwell/cu128)
::   - 20 GB free disk space
::
:: RTX 50-series / Blackwell note:
::   PyTorch 2.7.1+cu128 is required for sm_120 (RTX 5060/5070/5080/5090).
::   This script detects the GPU architecture and selects the correct wheel.
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
pushd "%SCRIPT_DIR%..\.."
set "PROJECT_ROOT=%CD%"
popd

echo [INFO] Project root: %PROJECT_ROOT%
echo.

:: ----------------------------------------------------------------------------
:: Step 1: Check Python version (must be 3.10 or 3.11)
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
echo [OK]   Detected: %PYTHON_VERSION%

:: Reject unsupported Python versions
python -c "import sys; v=sys.version_info; ok=(v.major==3 and v.minor in (10,11)); print('PYTHON_OK' if ok else 'PYTHON_UNSUPPORTED')" >"%TEMP%\py_check.txt" 2>&1
set /p PY_STATUS=<"%TEMP%\py_check.txt"
if /i not "!PY_STATUS!"=="PYTHON_OK" (
    echo.
    echo [ERROR] Unsupported Python version: %PYTHON_VERSION%
    echo         Jupiter Shot requires Python 3.10 or 3.11.
    echo         Python 3.12+ is not yet supported.
    echo         Python 3.9 and below are not supported.
    echo.
    echo         Download Python 3.11.9 from:
    echo           https://www.python.org/downloads/release/python-3119/
    echo.
    pause
    exit /b 1
)
echo [OK]   Python version is supported.
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
        echo         Ensure python -m venv is available ^(Python 3.10+^).
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
:: Step 3: Detect GPU architecture and install correct PyTorch wheel
:: ----------------------------------------------------------------------------
echo [STEP 3/7] Detecting GPU architecture and installing dependencies...
echo.

python -m pip install --upgrade pip --quiet

:: Detect compute capability to choose the right PyTorch wheel
echo [INFO] Detecting NVIDIA GPU compute capability...
python -c "import subprocess, re, sys; r=subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True); cc=r.stdout.strip().split('\n')[0].strip() if r.returncode==0 else ''; print(cc if cc else 'UNKNOWN')" >"%TEMP%\gpu_cc.txt" 2>&1
set /p GPU_CC=<"%TEMP%\gpu_cc.txt"
echo [INFO] Compute capability: !GPU_CC!

:: Determine if this is a Blackwell GPU (sm_120 = compute capability 12.0)
set "TORCH_WHEEL=cu121"
set "TORCH_VERSION=2.2.2"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
set "IS_BLACKWELL=0"

python -c "cc='!GPU_CC!'; parts=cc.split('.') if '.' in cc else [cc,'0']; major=int(parts[0]) if parts[0].isdigit() else 0; print('BLACKWELL' if major>=12 else 'LEGACY')" >"%TEMP%\arch_check.txt" 2>&1
set /p ARCH_TYPE=<"%TEMP%\arch_check.txt"

if /i "!ARCH_TYPE!"=="BLACKWELL" (
    echo [INFO] RTX 50-series / Blackwell GPU detected ^(sm_120^).
    echo [INFO] Selecting PyTorch 2.7.1 with CUDA 12.8 for sm_120 support.
    set "TORCH_WHEEL=cu128"
    set "TORCH_VERSION=2.7.1"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    set "IS_BLACKWELL=1"
) else (
    echo [INFO] Pre-Blackwell GPU detected. Using PyTorch 2.2.2 with CUDA 12.1.
)

echo.
echo [INFO] PyTorch version : !TORCH_VERSION!
echo [INFO] CUDA wheel      : !TORCH_WHEEL!
echo [INFO] Index URL       : !TORCH_INDEX!
echo.
echo [WARN] Download size is approximately 2.3-2.5 GB. This may take
echo        5-30 minutes depending on your internet connection.
echo        If the download stalls, see the manual fallback below.
echo.
echo        Manual fallback ^(if pip stalls^):
echo          pip install torch==!TORCH_VERSION! --index-url !TORCH_INDEX!
echo        Then re-run this script.
echo.
echo        If your antivirus blocks the download, temporarily disable
echo        real-time scanning for the .venv directory only.
echo.

:: Install PyTorch (no torchvision/torchaudio unless actually needed)
pip install torch==!TORCH_VERSION! --index-url !TORCH_INDEX!
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install PyTorch !TORCH_VERSION!+!TORCH_WHEEL!.
    echo.
    echo         Troubleshooting:
    echo           1. Check internet connection
    echo           2. Try manual download:
    echo              pip install torch==!TORCH_VERSION! --index-url !TORCH_INDEX!
    echo           3. If download stalls at a specific percentage, your antivirus
    echo              may be scanning the wheel. Temporarily disable it.
    echo           4. Ensure 20 GB free disk space ^(check with: dir C:\^)
    echo.
    pause
    exit /b 1
)

echo [INFO] Installing laptop-specific dependencies from requirements-laptop.txt...
if exist "%PROJECT_ROOT%\requirements-laptop.txt" (
    pip install -r "%PROJECT_ROOT%\requirements-laptop.txt"
) else (
    echo [WARN] requirements-laptop.txt not found, falling back to requirements.txt
    pip install -r "%PROJECT_ROOT%\requirements.txt"
)
if errorlevel 1 (
    echo [ERROR] Failed to install project dependencies.
    pause
    exit /b 1
)

echo [OK]   All dependencies installed.
echo.

:: ----------------------------------------------------------------------------
:: Step 4: Run real-kernel preflight (STOP if kernel test fails)
:: ----------------------------------------------------------------------------
echo [STEP 4/7] Running hardware preflight check ^(real CUDA kernel test^)...
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
    echo        If the error is "CUDA kernel launch failed" or "sm_120 not in arch list":
    echo          - Your PyTorch wheel does not support this GPU architecture.
    echo          - Re-run this script; it will select the correct wheel automatically.
    echo          - Or install manually:
    echo            pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    echo.
    pause
    exit /b 1
)
echo.
echo [OK]   Preflight passed ^(real CUDA kernel verified^).
echo.

:: Read recommended configs from preflight output
for /f "tokens=*" %%i in ('python -c "import json; d=json.load^(open^('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json'^)^); print^(d.get^('recommended_dense_config','laptop_dense_small'^)^)" 2^>nul') do set DENSE_CONFIG=%%i
for /f "tokens=*" %%i in ('python -c "import json; d=json.load^(open^('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json'^)^); print^(d.get^('recommended_moe_config','laptop_moe_small'^)^)" 2^>nul') do set MOE_CONFIG=%%i
if "!DENSE_CONFIG!"=="" set DENSE_CONFIG=laptop_dense_small
if "!MOE_CONFIG!"=="" set MOE_CONFIG=laptop_moe_small

echo [INFO] Selected dense config: !DENSE_CONFIG!
echo [INFO] Selected MoE config:   !MOE_CONFIG!
echo.

:: Force SMALL config for RTX 5060 / 8 GB VRAM
python -c "import json; d=json.load(open('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json')); vram=d.get('gpu',{}).get('vram_total_gb',0); print('SMALL' if float(vram)<10 else 'OK')" >"%TEMP%\vram_check.txt" 2>&1
set /p VRAM_TIER=<"%TEMP%\vram_check.txt"
if /i "!VRAM_TIER!"=="SMALL" (
    echo [INFO] 8 GB VRAM detected. Enforcing SMALL configuration.
    echo        First run: 10 diagnostic steps only.
    echo        Full run: 100 steps after diagnostics pass.
    echo        The 1.3B model will NOT be attempted.
    set DENSE_CONFIG=laptop_dense_small
    set MOE_CONFIG=laptop_moe_small
)
echo.

:: ----------------------------------------------------------------------------
:: Step 5: Confirm before running longer tests
:: ----------------------------------------------------------------------------
echo [STEP 5/7] Confirmation required before running GPU training tests.
echo.
echo   The following tests will run:
echo     - Dense transformer training  ^(estimated: 20-60 minutes^)
echo     - MoE routing validation      ^(estimated: 20-60 minutes^)
echo     - Checkpoint resume test      ^(estimated: 5-15 minutes^)
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
:: Step 6: Run 10 diagnostic steps first (SMALL / 8 GB VRAM safety)
:: ----------------------------------------------------------------------------
echo [STEP 6/7] Running 10 diagnostic steps ^(SMALL config safety check^)...
echo.

echo [DIAG 1/2] Dense diagnostic ^(10 steps^)...
python "%PROJECT_ROOT%\scripts\run_laptop_dense.py" --config !DENSE_CONFIG! --steps 10
if errorlevel 1 (
    echo [FAIL] Dense diagnostic failed. Do not proceed to full run.
    echo        Review output above for CUDA/memory errors.
    pause
    exit /b 1
)
echo [OK]   Dense diagnostic passed.
echo.

echo [DIAG 2/2] MoE diagnostic ^(10 steps^)...
python "%PROJECT_ROOT%\scripts\run_laptop_moe.py" --config !MOE_CONFIG! --steps 10
if errorlevel 1 (
    echo [FAIL] MoE diagnostic failed. Do not proceed to full run.
    pause
    exit /b 1
)
echo [OK]   MoE diagnostic passed.
echo.

:: ----------------------------------------------------------------------------
:: Step 7: Run 100-step validation tests
:: ----------------------------------------------------------------------------
echo [STEP 7/7] Running 100-step GPU validation tests...
echo.

set PASS_COUNT=0
set FAIL_COUNT=0

:: --- Dense validation ---
echo [TEST 1/3] Dense transformer training ^(!DENSE_CONFIG!^)...
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
echo [TEST 2/3] MoE routing validation ^(!MOE_CONFIG!^)...
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
echo  Generated report ^(send this to the team^):
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
    echo  [WARN] !FAIL_COUNT! test^(s^) failed. Review the output above.
)
pause
exit /b !FAIL_COUNT!
