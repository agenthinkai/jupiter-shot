# Jupiter Shot — Kuwait Laptop Environment Setup
# PowerShell script for Windows GPU laptop setup
#
# RTX 50-series / Blackwell note:
#   This script auto-detects Blackwell GPUs (sm_120) and installs
#   PyTorch 2.7.1+cu128 instead of 2.2.2+cu121.
#
# Run as Administrator in PowerShell:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\scripts\windows\setup_laptop_environment.ps1
param(
    [string]$PythonVersion = "3.11",
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Jupiter Shot — Kuwait Laptop Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
Write-Host "[1/8] Checking prerequisites..." -ForegroundColor Yellow

# Check Python — must be 3.10 or 3.11
try {
    $pythonVerRaw = python --version 2>&1
    Write-Host "  Detected: $pythonVerRaw" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python not found on PATH." -ForegroundColor Red
    Write-Host "  Install Python 3.11.9 from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Red
    Write-Host "  Ensure 'Add Python to PATH' is checked during installation." -ForegroundColor Red
    exit 1
}

# Enforce Python 3.10 or 3.11 — reject all other versions
$pyVersionCheck = python -c "import sys; v=sys.version_info; ok=(v.major==3 and v.minor in (10,11)); print('OK' if ok else f'UNSUPPORTED_{v.major}.{v.minor}')" 2>&1
if ($pyVersionCheck -notmatch "^OK") {
    $detected = $pyVersionCheck -replace "UNSUPPORTED_", ""
    Write-Host "" -ForegroundColor Red
    Write-Host "  [ERROR] Unsupported Python version: $detected" -ForegroundColor Red
    Write-Host "  Jupiter Shot requires Python 3.10 or 3.11." -ForegroundColor Red
    Write-Host "  Python 3.12+ is not yet supported." -ForegroundColor Red
    Write-Host "  Python 3.9 and below are not supported." -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "  Download Python 3.11.9 (recommended):" -ForegroundColor Yellow
    Write-Host "    https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    exit 1
}
Write-Host "  Python version is supported." -ForegroundColor Green

# Check pip
try {
    $pipVer = pip --version 2>&1
    Write-Host "  pip: $pipVer" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] pip not found." -ForegroundColor Red
    exit 1
}

# Check git
try {
    $gitVer = git --version 2>&1
    Write-Host "  git: $gitVer" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] git not found. Install from https://git-scm.com" -ForegroundColor Red
    exit 1
}

# Check NVIDIA driver and detect compute capability
$isBlackwell = $false
$gpuCC = "unknown"
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host "  NVIDIA GPU: $nvidiaSmi" -ForegroundColor Green
    # Detect compute capability
    $ccRaw = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1
    $gpuCC = $ccRaw.Trim().Split("`n")[0].Trim()
    Write-Host "  Compute capability: $gpuCC" -ForegroundColor Green
    # Blackwell = compute capability >= 12.0
    $ccMajor = [int]($gpuCC.Split(".")[0])
    if ($ccMajor -ge 12) {
        $isBlackwell = $true
        Write-Host "  RTX 50-series / Blackwell GPU detected (sm_120)." -ForegroundColor Cyan
        Write-Host "  Will install PyTorch 2.7.1+cu128 for sm_120 support." -ForegroundColor Cyan
    }
} catch {
    Write-Host "  [WARNING] nvidia-smi not found. CUDA validation will fail." -ForegroundColor Yellow
    Write-Host "  Install NVIDIA drivers from https://www.nvidia.com/drivers" -ForegroundColor Yellow
}

# ── Step 2: Create virtual environment ───────────────────────────────────────
Write-Host ""
Write-Host "[2/8] Creating virtual environment..." -ForegroundColor Yellow

$venvPath = ".\.venv"
if (Test-Path $venvPath) {
    if ($Force) {
        Write-Host "  Removing existing .venv (--Force specified)..."
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "  Virtual environment already exists at .venv. Use -Force to recreate." -ForegroundColor Green
    }
}

if (-not (Test-Path $venvPath)) {
    python -m venv .venv
    Write-Host "  Created: $venvPath" -ForegroundColor Green
}

# Activate
& ".\.venv\Scripts\Activate.ps1"
Write-Host "  Activated virtual environment (.venv)" -ForegroundColor Green

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "  pip upgraded" -ForegroundColor Green

# ── Step 4: Select and install PyTorch (architecture-aware) ──────────────────
Write-Host ""
Write-Host "[4/8] Installing PyTorch (architecture-aware)..." -ForegroundColor Yellow

if ($isBlackwell) {
    $torchVersion = "2.7.1"
    $cudaVersion = "cu128"
    $torchIndex = "https://download.pytorch.org/whl/cu128"
    Write-Host "  Blackwell GPU: installing PyTorch $torchVersion+$cudaVersion" -ForegroundColor Cyan
} else {
    $torchVersion = "2.2.2"
    $cudaVersion = "cu121"
    $torchIndex = "https://download.pytorch.org/whl/cu121"
    Write-Host "  Pre-Blackwell GPU: installing PyTorch $torchVersion+$cudaVersion" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "  Download size: approximately 2.3-2.5 GB." -ForegroundColor Yellow
Write-Host "  This may take 5-30 minutes depending on your internet connection." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Manual fallback (if this step stalls):" -ForegroundColor Gray
Write-Host "    pip install torch==$torchVersion --index-url $torchIndex" -ForegroundColor Gray
Write-Host "  Then re-run this script with -Force." -ForegroundColor Gray
Write-Host ""
Write-Host "  Antivirus note: if the download stalls at a fixed percentage," -ForegroundColor Gray
Write-Host "  temporarily disable real-time scanning for the .venv directory." -ForegroundColor Gray
Write-Host ""

# Install PyTorch only (no torchvision/torchaudio — not needed for validation)
pip install "torch==$torchVersion" --index-url $torchIndex
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "  [ERROR] Failed to install PyTorch $torchVersion+$cudaVersion." -ForegroundColor Red
    Write-Host "  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "    1. Check internet connection" -ForegroundColor Yellow
    Write-Host "    2. Try manual install:" -ForegroundColor Yellow
    Write-Host "       pip install torch==$torchVersion --index-url $torchIndex" -ForegroundColor Yellow
    Write-Host "    3. Disable antivirus real-time scanning for .venv during install" -ForegroundColor Yellow
    Write-Host "    4. Ensure 20 GB free disk space" -ForegroundColor Yellow
    exit 1
}

# Verify arch list for Blackwell
if ($isBlackwell) {
    $sm120Check = python -c "import torch; print('True' if 'sm_120' in torch.cuda.get_arch_list() else 'False')" 2>&1
    if ($sm120Check.Trim() -ne "True") {
        Write-Host "  [WARN] sm_120 not in arch list. Preflight will fail." -ForegroundColor Yellow
        Write-Host "  Ensure you installed torch==2.7.1+cu128, not an older wheel." -ForegroundColor Yellow
    } else {
        Write-Host "  sm_120 support: confirmed" -ForegroundColor Green
    }
}

# ── Step 5: Install laptop dependencies ──────────────────────────────────────
Write-Host ""
Write-Host "[5/8] Installing laptop-specific dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements-laptop.txt") {
    pip install -r requirements-laptop.txt
    Write-Host "  Installed from requirements-laptop.txt" -ForegroundColor Green
} elseif (Test-Path "requirements.txt") {
    Write-Host "  [WARN] requirements-laptop.txt not found, falling back to requirements.txt" -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "  Installed from requirements.txt" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Neither requirements-laptop.txt nor requirements.txt found." -ForegroundColor Red
    exit 1
}

# ── Step 6: Install test dependencies ─────────────────────────────────────────────
Write-Host ""
Write-Host "[6/8] Installing test dependencies (pinned)..." -ForegroundColor Yellow
pip install pytest==8.2.1 pytest-cov==5.0.0
Write-Host "  Test dependencies installed" -ForegroundColor Green

# ── Step 7: Skip DeepSpeed (not needed for laptop validation) ────────────────
Write-Host ""
Write-Host "[7/8] DeepSpeed (skipped — not required for laptop validation)..." -ForegroundColor Gray
Write-Host "  DeepSpeed requires Linux + NCCL for multi-GPU training." -ForegroundColor Gray
Write-Host "  Single-GPU laptop validation does not need it." -ForegroundColor Gray

# ── Step 8: Verify installation ───────────────────────────────────────────────
Write-Host ""
Write-Host "[8/8] Verifying installation..." -ForegroundColor Yellow

python -c @"
import torch
import yaml
import numpy as np
import sys

print(f'  Python:     {sys.version.split()[0]}')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM:       {props.total_memory / 1024**3:.1f} GB')
    cc = torch.cuda.get_device_capability(0)
    print(f'  Compute:    {cc[0]}.{cc[1]}')
    print(f'  BF16:       {cc[0] >= 8}')
    print(f'  FP16:       {cc[0] >= 5}')
    arch_list = torch.cuda.get_arch_list()
    print(f'  Arch list:  {arch_list}')
print(f'  NumPy:      {np.__version__}')
print(f'  PyYAML:     {yaml.__version__}')
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete." -ForegroundColor Cyan
Write-Host "  Next step: scripts\windows\run_preflight.bat" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
