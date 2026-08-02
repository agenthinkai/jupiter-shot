# Jupiter Shot — Kuwait Laptop Environment Setup
# PowerShell script for Windows GPU laptop setup
# Run as Administrator in PowerShell

param(
    [string]$PythonVersion = "3.11",
    [string]$CudaVersion = "cu121",
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Jupiter Shot — Kuwait Laptop Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
Write-Host "[1/8] Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVer = python --version 2>&1
    Write-Host "  Python: $pythonVer" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python not found. Install Python $PythonVersion from https://python.org" -ForegroundColor Red
    exit 1
}

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

# Check NVIDIA driver
try {
    $nvidiaSmi = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host "  NVIDIA GPU: $nvidiaSmi" -ForegroundColor Green
} catch {
    Write-Host "  [WARNING] nvidia-smi not found. CUDA validation will fail." -ForegroundColor Yellow
    Write-Host "  Install NVIDIA drivers from https://www.nvidia.com/drivers" -ForegroundColor Yellow
}

# ── Step 2: Create virtual environment ───────────────────────────────────────
Write-Host ""
Write-Host "[2/8] Creating virtual environment..." -ForegroundColor Yellow

$venvPath = ".\venv"
if (Test-Path $venvPath) {
    if ($Force) {
        Write-Host "  Removing existing venv (--Force specified)..."
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "  Virtual environment already exists. Use -Force to recreate." -ForegroundColor Green
    }
}

if (-not (Test-Path $venvPath)) {
    python -m venv venv
    Write-Host "  Created: $venvPath" -ForegroundColor Green
}

# Activate
& ".\venv\Scripts\Activate.ps1"
Write-Host "  Activated virtual environment" -ForegroundColor Green

# ── Step 3: Upgrade pip ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/8] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "  pip upgraded" -ForegroundColor Green

# ── Step 4: Install PyTorch with CUDA ────────────────────────────────────────
Write-Host ""
Write-Host "[4/8] Installing PyTorch with CUDA ($CudaVersion)..." -ForegroundColor Yellow
Write-Host "  This may take 5–15 minutes depending on internet speed."

$torchIndex = "https://download.pytorch.org/whl/$CudaVersion"
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url $torchIndex

# Verify CUDA
$cudaCheck = python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>&1
Write-Host "  $cudaCheck" -ForegroundColor Green

# ── Step 5: Install core dependencies ────────────────────────────────────────
Write-Host ""
Write-Host "[5/8] Installing core dependencies..." -ForegroundColor Yellow
pip install `
    transformers==4.40.2 `
    tokenizers==0.19.1 `
    datasets==2.19.1 `
    accelerate==0.29.3 `
    pyyaml==6.0.1 `
    numpy==1.26.4 `
    tqdm==4.66.2 `
    psutil==5.9.8

Write-Host "  Core dependencies installed" -ForegroundColor Green

# ── Step 6: Install test dependencies ────────────────────────────────────────
Write-Host ""
Write-Host "[6/8] Installing test dependencies..." -ForegroundColor Yellow
pip install `
    pytest==8.1.1 `
    pytest-asyncio==0.23.6 `
    datasketch==1.6.4

Write-Host "  Test dependencies installed" -ForegroundColor Green

# ── Step 7: Optional — DeepSpeed ─────────────────────────────────────────────
Write-Host ""
Write-Host "[7/8] DeepSpeed (optional, skip if install fails)..." -ForegroundColor Yellow
Write-Host "  DeepSpeed on Windows requires Visual Studio Build Tools."
Write-Host "  Skipping DeepSpeed — single-GPU validation does not require it."
Write-Host "  To install: pip install deepspeed (requires MSVC)" -ForegroundColor Gray

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
print(f'  NumPy:      {np.__version__}')
print(f'  PyYAML:     {yaml.__version__}')
"@

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete. Run: scripts\windows\run_preflight.bat" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
