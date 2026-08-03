@echo off
:: Jupiter Shot — Laptop Validation Launcher (Run 8)
:: ====================================================
::
:: This file is a MINIMAL LAUNCHER ONLY.
:: All orchestration logic lives in:
::   scripts/run_laptop_validation_pipeline.py
::
:: Usage:
::   run_all_laptop_validation.bat [--data-mode real|synthetic|auto] [--preflight-only] [--steps N]
::
:: For preflight review only:
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
::   - If pip install hangs or fails, check antivirus software.
::     Windows Defender and third-party antivirus programs often block or slow
::     Python package downloads. Temporarily disable real-time protection during
::     the initial pip install, then re-enable it.
::   - If torch import fails after install, re-run this script (the venv check
::     will skip the download and retry the import).
::
:: Exit-code contract (Run 8):
::   0  = PASS (complete success only)
::   1  = NOT_ACCEPTED
::   2  = NOT_EVALUABLE
::   3  = EXECUTION_ERROR or dependency installation failure
::   4  = SAFETY_STOP (Python pipeline safety stop)
::
:: ERRORLEVEL rules applied in this file:
::   - ERRORLEVEL is captured into !ERR! immediately after every critical command.
::   - No pause, echo, set, or pipe command appears between a critical command
::     and the capture of its ERRORLEVEL.
::   - All error paths use "endlocal & exit /b N" to prevent setlocal from
::     swallowing the exit code when invoked from a parent batch file.
::   - pause is NEVER used in error paths (it resets ERRORLEVEL to 0 when
::     stdin is redirected, breaking CI and parent-batch invocations).

setlocal enabledelayedexpansion

:: ── Locate repo root (parent of scripts\windows) ──────────────────────────
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_ROOT=%CD%"
popd

echo.
echo ============================================================
echo  Jupiter Shot - Laptop Validation Launcher (Run 8)
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
    set "ERR=!ERRORLEVEL!"
    if !ERR! NEQ 0 (
        echo [Launcher] ERROR: Failed to create virtual environment.
        echo [Launcher] Ensure Python 3.10 or 3.11 is on PATH.
        endlocal & exit /b 3
    )
    echo [Launcher] Virtual environment created.
) else (
    echo [Launcher] Virtual environment already exists.
)

:: ── Install PyTorch if not already present ────────────────────────────────
"%VENV_PYTHON%" -c "import torch; v=torch.__version__; assert '2.7' in v and 'cu128' in v, f'Wrong torch: {v}'" >nul 2>&1
set "ERR=!ERRORLEVEL!"
if !ERR! NEQ 0 (
    echo [Launcher] Installing PyTorch 2.7.1+cu128 ^(RTX 50-series / Blackwell^) ...
    echo [Launcher] Download size: ~2.3 GB. This may take 5-30 minutes.
    "%VENV_PYTHON%" -m pip install --quiet torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    set "ERR=!ERRORLEVEL!"
    if !ERR! NEQ 0 (
        echo [Launcher] ERROR: PyTorch installation failed. Exit code: !ERR!
        echo [Launcher] Manual fallback:
        echo [Launcher]   .venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
        endlocal & exit /b 3
    )
    echo [Launcher] PyTorch 2.7.1+cu128 installed.
) else (
    echo [Launcher] PyTorch 2.7.1+cu128 already installed.
)

:: ── Install remaining dependencies if any are missing ─────────────────────
:: Check only the packages that are NOT torch (torch is handled above).
:: This check avoids re-downloading torch on every run.
"%VENV_PYTHON%" -c "import transformers, datasets, pandas, pyarrow, yaml, pyarrow_hotfix" >nul 2>&1
set "ERR=!ERRORLEVEL!"
if !ERR! NEQ 0 (
    echo [Launcher] Installing remaining dependencies from requirements-laptop.txt ...
    "%VENV_PYTHON%" -m pip install --quiet -r "%REPO_ROOT%\requirements-laptop.txt" --extra-index-url https://download.pytorch.org/whl/cu128
    set "ERR=!ERRORLEVEL!"
    if !ERR! NEQ 0 (
        echo [Launcher] ERROR: Dependency installation failed. Exit code: !ERR!
        echo [Launcher] Run manually to see full error:
        echo [Launcher]   .venv\Scripts\python.exe -m pip install -r requirements-laptop.txt
        endlocal & exit /b 3
    )
    echo [Launcher] Dependencies installed.
) else (
    echo [Launcher] All dependencies already installed.
)

:: ── pip check: verify no broken requirements ──────────────────────────────
"%VENV_PYTHON%" -m pip check >nul 2>&1
set "ERR=!ERRORLEVEL!"
if !ERR! NEQ 0 (
    echo [Launcher] ERROR: pip check failed - broken requirements detected.
    echo [Launcher] Run for details: .venv\Scripts\python.exe -m pip check
    endlocal & exit /b 3
)
echo [Launcher] pip check passed.

echo.
echo [Launcher] Delegating to Python orchestrator ...
echo.

:: ── Delegate ALL orchestration to the Python pipeline ─────────────────────
:: All arguments (%*) are forwarded verbatim.
:: ERRORLEVEL is captured immediately after the Python call with no intervening
:: commands (no echo, no set, no pipe) to prevent clobbering.
"%VENV_PYTHON%" "%REPO_ROOT%\scripts\run_laptop_validation_pipeline.py" %*
set "PIPELINE_EXIT=!ERRORLEVEL!"

echo.
echo [Launcher] Pipeline exited with code !PIPELINE_EXIT!

if !PIPELINE_EXIT! EQU 0 echo [Launcher] Result: PASS
if !PIPELINE_EXIT! EQU 1 echo [Launcher] Result: NOT_ACCEPTED
if !PIPELINE_EXIT! EQU 2 echo [Launcher] Result: NOT_EVALUABLE
if !PIPELINE_EXIT! EQU 3 echo [Launcher] Result: EXECUTION_ERROR
if !PIPELINE_EXIT! EQU 4 echo [Launcher] Result: SAFETY_STOP

endlocal & exit /b %PIPELINE_EXIT%
