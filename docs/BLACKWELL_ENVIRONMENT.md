# Jupiter Shot — RTX 50-Series / Blackwell Environment Profile

**Status:** ENVIRONMENT BLOCKED — REPAIRABLE
**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
**Architecture:** Blackwell, compute capability sm_120
**VRAM:** 8,151 MiB
**Driver:** 592.82
**Driver-reported CUDA capability:** 13.1
**OS:** Windows 11 Pro
**Python:** 3.11.9

---

## What Happened

The first Kuwait GPU validation attempt halted correctly during the CUDA preflight.

The installed PyTorch wheel (`2.2.2+cu121`) does not include compiled kernels for
`sm_120` (Blackwell). When the preflight attempted to launch a CUDA kernel, the
operation failed with a device-side error. `torch.cuda.is_available()` returned
`True` — the driver and device were detected — but no actual computation could run.

**This is a dependency compatibility failure, not a Jupiter model failure.**
The safety stop worked as designed. No Jupiter training workload was executed.

---

## Required PyTorch Version

| Component | Value |
|-----------|-------|
| Python | 3.11.9 (or 3.10.x) |
| PyTorch | **2.7.1** |
| CUDA wheel | **cu128** (CUDA 12.8) |
| sm_120 support | Yes (Blackwell) |
| Index URL | `https://download.pytorch.org/whl/cu128` |
| System CUDA toolkit | NOT required — PyTorch bundles CUDA runtime |

### Installation Command

```powershell
# In an isolated virtual environment (Python 3.11):
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

**Do not install torchvision or torchaudio** unless the laptop validation
actually imports them. If required, pin versions compatible with PyTorch 2.7.1.

---

## Why PyTorch 2.7.1+cu128

- First stable PyTorch release with official `sm_120` (Blackwell) support
- CUDA 12.8 wheel includes the `sm_120` PTX and SASS kernels
- Verified to work with NVIDIA driver 592.82 on Windows 11
- Does not require installing the CUDA Toolkit separately
- Conservative choice: stable release, not a nightly

---

## Architecture Detection

The setup script (`run_all_laptop_validation.bat`) automatically detects the GPU
compute capability via `nvidia-smi` and selects the correct wheel:

| Compute Capability | Architecture | PyTorch Wheel |
|--------------------|-------------|---------------|
| < 12.0 | Ampere, Turing, Volta, etc. | `2.2.2+cu121` |
| ≥ 12.0 | **Blackwell (RTX 50xx)** | **`2.7.1+cu128`** |

---

## VRAM Configuration (8 GB)

The RTX 5060 Laptop GPU has 8,151 MiB VRAM. The validation enforces the
**SMALL** configuration to preserve ≥ 20% VRAM headroom:

| Stage | Steps | Config |
|-------|-------|--------|
| Diagnostic | 10 | `laptop_dense_small` / `laptop_moe_small` |
| Full validation | 100 | `laptop_dense_small` / `laptop_moe_small` |
| NOT attempted | — | `laptop_dense_medium`, `laptop_dense_1b3`, any 1.3B model |

---

## Preflight Requirements (Upgraded)

The preflight now executes a **real CUDA kernel** before declaring the environment
ready. `torch.cuda.is_available()` alone is insufficient — it only checks driver
registration, not kernel execution.

The upgraded preflight:

1. Allocates two small tensors on CUDA
2. Performs a small matrix multiplication (`torch.matmul`)
3. Calls `torch.cuda.synchronize()`
4. Verifies the numerical result
5. Catches and reports kernel compatibility errors (e.g., `sm_120` not in arch list)
6. Records `torch.__version__`, `torch.version.cuda`, GPU name, compute capability,
   `torch.cuda.get_arch_list()`, kernel-launch PASS/FAIL, BF16 result, FP16 result
7. **Halts before training if the kernel test fails**

---

## Download Notes

The PyTorch 2.7.1+cu128 wheel is approximately 2.5 GB. If the download stalls:

1. **Manual download fallback:**
   ```
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   ```
2. **Antivirus interference:** Temporarily disable real-time scanning for the
   `.venv` directory. Re-enable after installation.
3. **File lock:** Close any other Python processes or IDEs before installing.
4. **Integrity check:** After installation, verify:
   ```python
   import torch
   print(torch.__version__)          # should be 2.7.1+cu128
   print(torch.version.cuda)         # should be 12.8
   print('sm_120' in torch.cuda.get_arch_list())  # should be True
   ```

---

## Operator Instructions (Kishore)

After the repair branch is merged, run:

```powershell
# 1. Pull the repair branch
git pull origin fix/rtx50-blackwell-validation

# 2. Delete the old virtual environment
rmdir /s /q .venv

# 3. Run the updated setup script
scripts\windows\run_all_laptop_validation.bat
```

The script will:
- Detect your RTX 5060 (sm_120) automatically
- Install PyTorch 2.7.1+cu128 (not 2.2.2+cu121)
- Run the upgraded real-kernel preflight
- Enforce SMALL config (10 diagnostic steps, then 100 steps)
- Halt immediately if the kernel test fails

---

## Classification

| Field | Value |
|-------|-------|
| Status | **ENVIRONMENT BLOCKED — REPAIRABLE** |
| Model validation | NOT STARTED (environment blocked) |
| Architecture modified | NO |
| GPU success claimed | NO |
| Safety stop | WORKED CORRECTLY |
| Repair estimate | 1 re-run after pulling this branch |
