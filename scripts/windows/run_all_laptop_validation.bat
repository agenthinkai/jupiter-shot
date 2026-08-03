@echo off
:: Jupiter Shot — Laptop Validation Launcher (Run 7+)
:: ====================================================
::
:: This file is a MINIMAL LAUNCHER ONLY.
:: All orchestration logic lives in:
::   scripts/run_laptop_validation_pipeline.py
::
:: Usage:
::   run_all_laptop_validation.bat [--data-mode real|synthetic|auto] [--preflight-only] [--steps N]
::
:: For preflight review only (Kishore's first Run 7 step):
::   run_all_laptop_validation.bat --data-mode real --preflight-only
::
:: For full validation:
::   run_all_laptop_validation.bat --data-mode real
::
:: VRAM / Config selection:
::   For 8 GB VRAM (RTX 5060): uses laptop_moe_8expert_8gb_safe.yaml (small config)
::   For 16+ GB VRAM:          uses laptop_moe_run7.yaml (full config)
::   The Python orchestrator selects the config automatically based on available VRAM.
::
:: Troubleshooting:
::   - If pip install hangs or fails, check antivirus (anti-virus) software.
::     Windows Defender and third-party antivirus programs often block or slow
::     Python package downloads. Temporarily disable real-time protection during
::     the initial pip install, then re-enable it.
::   - If torch import fails after install, re-run this script (the venv check
::     will skip the download and retry the import).
::
:: Exit codes (from Python orchestrator):
::   0  PASS
::   1  NOT_ACCEPTED
::   2  NOT_EVALUABLE
::   3  EXECUTION_ERROR
::   4  SAFETY_STOP

setlocal enabledelayedexpansion

:: ── Locate repo root (parent of scripts\windows) ──────────────────────────
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

echo.
echo ============================================================
echo  Jupiter Shot - Laptop Validation Launcher (Run 7+)
echo ============================================================
echo  Repo root: %REPO_ROOT%
echo  Args:      %*
echo ============================================================
echo.

:: ── Virtual environment ────────────────────────────────────────────────────
set "VENV_DIR=%REPO_ROOT%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

:: ── Create venv if it does not exist ──────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo [Launcher] Creating virtual environment at .venv\ ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [Launcher] ERROR: Failed to create virtual environment.
        echo [Launcher] Ensure Python 3.10 or 3.11 is on PATH.
        pause
        exit /b 3
    )
    echo [Launcher] Virtual environment created.
) else (
    echo [Launcher] Virtual environment already exists.
)

:: ── Install PyTorch if not already present ────────────────────────────────
"%VENV_PYTHON%" -c "import torch; v=torch.__version__; assert '2.7' in v and 'cu128' in v, f'Wrong torch: {v}'" >nul 2>&1
if errorlevel 1 (
    echo [Launcher] Installing PyTorch 2.7.1+cu128 ^(RTX 50-series / Blackwell^) ...
    echo [Launcher] Download size: ~2.3 GB. This may take 5-30 minutes.
    "%VENV_PYTHON%" -m pip install --quiet ^
        torch==2.7.1+cu128 ^
        --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [Launcher] ERROR: PyTorch installation failed.
        echo [Launcher] Manual fallback:
        echo [Launcher]   .venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
        pause
        exit /b 3
    )
    echo [Launcher] PyTorch 2.7.1+cu128 installed.
) else (
    echo [Launcher] PyTorch 2.7.1+cu128 already installed.
)

:: ── Install remaining dependencies if any are missing ─────────────────────
"%VENV_PYTHON%" -c "import transformers, datasets, pandas, pyarrow, yaml" >nul 2>&1
if errorlevel 1 (
    echo [Launcher] Installing remaining dependencies from requirements-laptop.txt ...
    "%VENV_PYTHON%" -m pip install --quiet -r "%REPO_ROOT%\requirements-laptop.txt" ^
        --extra-index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 (
        echo [Launcher] ERROR: Dependency installation failed.
        pause
        exit /b 3
    )
    echo [Launcher] Dependencies installed.
) else (
    echo [Launcher] All dependencies already installed.
)

echo.
echo [Launcher] Delegating to Python orchestrator ...
echo.

:: ── Delegate ALL orchestration to the Python pipeline ─────────────────────
:: All arguments (%*) are forwarded verbatim.
"%VENV_PYTHON%" "%REPO_ROOT%\scripts\run_laptop_validation_pipeline.py" %*
set "PIPELINE_EXIT=%ERRORLEVEL%"

echo.
echo [Launcher] Pipeline exited with code %PIPELINE_EXIT%

if %PIPELINE_EXIT% EQU 0 echo [Launcher] Result: PASS
if %PIPELINE_EXIT% EQU 1 echo [Launcher] Result: NOT_ACCEPTED
if %PIPELINE_EXIT% EQU 2 echo [Launcher] Result: NOT_EVALUABLE
if %PIPELINE_EXIT% EQU 3 echo [Launcher] Result: EXECUTION_ERROR
if %PIPELINE_EXIT% EQU 4 echo [Launcher] Result: SAFETY_STOP

exit /b %PIPELINE_EXIT%
