@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Jupiter Shot — Kuwait Laptop GPU Validation
:: run_all_laptop_validation.bat
::
:: Usage:
::   scripts\windows\run_all_laptop_validation.bat [--data-mode MODE]
::
::   --data-mode real       Use Wikitext-2 real text (default).
::                          Halts with REAL_TEXT_DATA_UNAVAILABLE if the
::                          dataset or tokenizer cannot be downloaded.
::   --data-mode synthetic  Use random token IDs. Forces outcome=NOT_ACCEPTED.
::   --data-mode auto       Try real; fall back to synthetic silently.
::
:: What this script validates:
::   - CUDA execution on a single NVIDIA GPU
::   - Dense transformer training (small config)
::   - MoE routing validation (8-expert, top-2)
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
:: Parse command-line arguments
:: --data-mode real | synthetic | auto
:: ----------------------------------------------------------------------------
set "DATA_MODE=real"
set "ARG_PARSE_ERROR=0"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--data-mode" (
    if "%~2"=="" (
        echo [ERROR] --data-mode requires a value: real, synthetic, or auto
        set ARG_PARSE_ERROR=1
        goto args_done
    )
    set "DATA_MODE=%~2"
    shift
    shift
    goto parse_args
)
echo [WARN] Unknown argument: %~1 (ignored)
shift
goto parse_args

:args_done
if !ARG_PARSE_ERROR! EQU 1 (
    echo [ERROR] Argument parsing failed. Exiting.
    pause
    exit /b 1
)

:: Validate --data-mode value
if /i "!DATA_MODE!"=="real"      goto data_mode_ok
if /i "!DATA_MODE!"=="synthetic" goto data_mode_ok
if /i "!DATA_MODE!"=="auto"      goto data_mode_ok
echo [ERROR] Invalid --data-mode value: !DATA_MODE!
echo         Valid values: real, synthetic, auto
pause
exit /b 1

:data_mode_ok
echo [INFO] Data mode: !DATA_MODE!
if /i "!DATA_MODE!"=="synthetic" (
    echo [WARN] --data-mode synthetic: outcome will be forced to NOT_ACCEPTED.
    echo        Synthetic runs cannot produce a PASS result.
    echo        Use --data-mode real for a valid validation run.
)
echo.

:: ----------------------------------------------------------------------------
:: Step 1: Check Python version (must be 3.10 or 3.11)
:: ----------------------------------------------------------------------------
echo [STEP 1/8] Checking Python installation...
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
echo [STEP 2/8] Setting up isolated virtual environment at .venv\...

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
echo [STEP 3/8] Detecting GPU architecture and installing dependencies...
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

:: Check if the correct PyTorch version is already installed
python -c "import torch; v=torch.__version__; print('TORCH_OK' if '!TORCH_VERSION!' in v else 'TORCH_MISSING')" >"%TEMP%\torch_check.txt" 2>&1
set /p TORCH_STATUS=<"%TEMP%\torch_check.txt"

if /i "!TORCH_STATUS!"=="TORCH_OK" (
    echo [OK]   PyTorch !TORCH_VERSION! already installed. Skipping download.
    echo        ^(Kishore's .venv is preserved — PyTorch will NOT be re-downloaded.^)
) else (
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
)

echo [INFO] Installing real-text dependencies (transformers, datasets, tokenizers)...
echo        PyTorch is already handled above and will NOT be re-downloaded.
echo        Only packages missing from .venv will be installed.
echo.

:: Install real-text deps without touching torch.
:: --no-deps is NOT used here because transformers/datasets have legitimate deps.
:: We exclude torch explicitly so pip does not re-resolve or re-download it.
python -c "import transformers" >nul 2>&1
if errorlevel 1 (
    echo [INFO] transformers not found. Installing transformers==4.40.2...
    pip install transformers==4.40.2 --no-deps
    pip install tokenizers==0.19.1 regex safetensors huggingface-hub filelock packaging requests tqdm
) else (
    echo [OK]   transformers already installed. Skipping.
)

python -c "import datasets" >nul 2>&1
if errorlevel 1 (
    echo [INFO] datasets not found. Installing datasets==2.19.1...
    pip install datasets==2.19.1 --no-deps
    pip install multiprocess dill pyarrow xxhash aiohttp fsspec
) else (
    echo [OK]   datasets already installed. Skipping.
)

:: Install remaining non-torch project deps from requirements-laptop.txt
:: but exclude torch/torchvision/torchaudio lines to prevent re-download.
if exist "%PROJECT_ROOT%\requirements-laptop.txt" (
    python -c "
import re, pathlib, sys
lines = pathlib.Path(r'%PROJECT_ROOT%\requirements-laptop.txt').read_text().splitlines()
filtered = [l for l in lines if not re.match(r'\s*(torch|torchvision|torchaudio)', l, re.I)]
pathlib.Path(r'%TEMP%\reqs_no_torch.txt').write_text('\n'.join(filtered))
"
    pip install -r "%TEMP%\reqs_no_torch.txt" --quiet
) else (
    echo [WARN] requirements-laptop.txt not found. Skipping project deps.
)
if errorlevel 1 (
    echo [ERROR] Failed to install project dependencies.
    pause
    exit /b 1
)

echo [OK]   All dependencies installed. PyTorch was NOT re-downloaded.
echo.

:: ----------------------------------------------------------------------------
:: Step 4: Real-text data preflight (only for --data-mode real)
:: ----------------------------------------------------------------------------
echo [STEP 4/8] Real-text data availability check...
echo.

if /i "!DATA_MODE!"=="real" (
    echo [INFO] --data-mode real: verifying Wikitext-2 and tokenizer availability...
    echo        Dataset:   wikitext / wikitext-2-raw-v1 / train  ^(CC BY-SA 4.0^)
    echo        Tokenizer: EleutherAI/gpt-neox-20b               ^(Apache 2.0^)
    echo.

    python -c "
import sys
errors = []
# Check datasets package
try:
    from datasets import load_dataset
except ImportError:
    errors.append('datasets package not installed (pip install datasets==2.19.1)')
else:
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train',
                          streaming=True, trust_remote_code=False)
        sample = next(iter(ds))
        if not sample.get('text'):
            errors.append('Wikitext-2 loaded but returned empty text')
        else:
            print('  [OK] Wikitext-2 available: ' + repr(sample['text'][:60]))
    except Exception as e:
        errors.append(f'Wikitext-2 unavailable: {e}')
# Check transformers package
try:
    from transformers import AutoTokenizer
except ImportError:
    errors.append('transformers package not installed (pip install transformers==4.40.2)')
else:
    try:
        tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        ids = tok('Hello world', return_tensors='pt')
        print('  [OK] Tokenizer available: EleutherAI/gpt-neox-20b (vocab=' + str(tok.vocab_size) + ')')
    except Exception as e:
        errors.append(f'Tokenizer unavailable: {e}')
if errors:
    print()
    print('REAL_TEXT_DATA_UNAVAILABLE')
    for err in errors:
        print('  ERROR: ' + err)
    sys.exit(1)
else:
    print('  [OK] Real-text data preflight passed.')
    sys.exit(0)
"
    if errorlevel 1 (
        echo.
        echo [FAIL] REAL_TEXT_DATA_UNAVAILABLE
        echo.
        echo        The validation requires real text data but it cannot be loaded.
        echo        Do NOT silently fall back to synthetic mode.
        echo.
        echo        To fix:
        echo          1. Ensure internet access is available
        echo          2. Re-run: pip install datasets==2.19.1 transformers==4.40.2
        echo          3. If HuggingFace is blocked, configure HF_ENDPOINT:
        echo             set HF_ENDPOINT=https://huggingface.co
        echo          4. If you must run without real data, use:
        echo             run_all_laptop_validation.bat --data-mode synthetic
        echo             WARNING: synthetic runs produce outcome=NOT_ACCEPTED
        echo.
        pause
        exit /b 1
    )
) else if /i "!DATA_MODE!"=="synthetic" (
    echo [INFO] --data-mode synthetic: skipping real-text preflight.
    echo [WARN] Synthetic runs will produce outcome=NOT_ACCEPTED.
) else (
    echo [INFO] --data-mode auto: real-text will be attempted; synthetic fallback enabled.
)
echo.

:: ----------------------------------------------------------------------------
:: Step 5: Run real-kernel preflight (STOP if kernel test fails)
:: ----------------------------------------------------------------------------
echo [STEP 5/8] Running hardware preflight check ^(real CUDA kernel test^)...
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
for /f "tokens=*" %%i in ('python -c "import json; d=json.load^(open^('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json'^)^); print^(d.get^('recommended_moe_config','laptop_moe_8expert_8gb_safe'^)^)" 2^>nul') do set MOE_CONFIG=%%i
if "!DENSE_CONFIG!"=="" set DENSE_CONFIG=laptop_dense_small
if "!MOE_CONFIG!"==""  set MOE_CONFIG=laptop_moe_8expert_8gb_safe

echo [INFO] Selected dense config: !DENSE_CONFIG!
echo [INFO] Selected MoE config:   !MOE_CONFIG!
echo.

:: Force 8GB-safe config for RTX 5060 / 8 GB VRAM
python -c "import json; d=json.load(open('%PROJECT_ROOT:\=/%/benchmarks/results/laptop/preflight.json')); vram=d.get('gpu',{}).get('vram_total_gb',0); print('SMALL' if float(vram)<10 else 'OK')" >"%TEMP%\vram_check.txt" 2>&1
set /p VRAM_TIER=<"%TEMP%\vram_check.txt"
if /i "!VRAM_TIER!"=="SMALL" (
    echo [INFO] 8 GB VRAM detected. Enforcing 8GB-safe configuration.
    echo        Dense config: laptop_dense_small
    echo        MoE config:   laptop_moe_8expert_8gb_safe  ^(8 experts, top-2^)
    echo        First run: 10 diagnostic steps only.
    echo        Full run: 100 steps after diagnostics pass.
    echo        The 1.3B model will NOT be attempted.
    set DENSE_CONFIG=laptop_dense_small
    set MOE_CONFIG=laptop_moe_8expert_8gb_safe
)
echo.

:: ----------------------------------------------------------------------------
:: Step 6: Confirm before running longer tests
:: ----------------------------------------------------------------------------
echo [STEP 6/8] Confirmation required before running GPU training tests.
echo.
echo   Data mode:    !DATA_MODE!
echo   Dense config: !DENSE_CONFIG!
echo   MoE config:   !MOE_CONFIG!
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
:: Step 7: Run 10 diagnostic steps first (8GB-safe config)
:: ----------------------------------------------------------------------------
echo [STEP 7/8] Running 10 diagnostic steps ^(8GB-safe config check^)...
echo.

echo [DIAG 1/2] Dense diagnostic ^(10 steps, --data-mode !DATA_MODE!^)...
python "%PROJECT_ROOT%\scripts\run_laptop_dense.py" --config !DENSE_CONFIG! --steps 10 --data-mode !DATA_MODE!
if errorlevel 1 (
    echo [FAIL] Dense diagnostic failed. Do not proceed to full run.
    echo        Review output above for CUDA/memory errors.
    pause
    exit /b 1
)
echo [OK]   Dense diagnostic passed.
echo.

echo [DIAG 2/2] MoE diagnostic ^(10 steps, --data-mode !DATA_MODE!^)...
python "%PROJECT_ROOT%\scripts\run_laptop_moe.py" --config !MOE_CONFIG! --steps 10 --data-mode !DATA_MODE!
if errorlevel 1 (
    echo [FAIL] MoE diagnostic failed. Do not proceed to full run.
    pause
    exit /b 1
)
echo [OK]   MoE diagnostic passed.
echo.

:: ----------------------------------------------------------------------------
:: Step 8: Run 100-step validation tests
:: ----------------------------------------------------------------------------
echo [STEP 8/8] Running 100-step GPU validation tests...
echo.

set PASS_COUNT=0
set FAIL_COUNT=0

:: --- Dense validation ---
echo [TEST 1/3] Dense transformer training ^(!DENSE_CONFIG!, --data-mode !DATA_MODE!^)...
python "%PROJECT_ROOT%\scripts\run_laptop_dense.py" --config !DENSE_CONFIG! --steps 100 --data-mode !DATA_MODE!
if errorlevel 1 (
    echo [FAIL] Dense training validation failed. Review output above.
    set /a FAIL_COUNT+=1
) else (
    echo [OK]   Dense training validation complete.
    set /a PASS_COUNT+=1
)
echo.

:: --- MoE validation ---
echo [TEST 2/3] MoE routing validation ^(!MOE_CONFIG!, --data-mode !DATA_MODE!^)...
python "%PROJECT_ROOT%\scripts\run_laptop_moe.py" --config !MOE_CONFIG! --steps 100 --data-mode !DATA_MODE!
if errorlevel 1 (
    echo [FAIL] MoE validation failed. Review output above.
    set /a FAIL_COUNT+=1
) else (
    echo [OK]   MoE validation complete.
    set /a PASS_COUNT+=1
)
echo.

:: --- Resume test ---
echo [TEST 3/3] Checkpoint resume test ^(!DENSE_CONFIG!^)...
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
echo  Data mode: !DATA_MODE!
echo  Passed:    !PASS_COUNT!   Failed: !FAIL_COUNT!
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
