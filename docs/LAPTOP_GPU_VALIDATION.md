# Jupiter Shot — Kuwait Laptop GPU Validation Guide

> **Branch:** `fix/rtx50-blackwell-validation`
> **Operator:** Kishore
> **Purpose:** Execute Month 1 GPU gates on a single laptop GPU in Kuwait
> **Status:** ENVIRONMENT BLOCKED — REPAIRABLE
> **Blocker:** RTX 5060 Laptop GPU (Blackwell, sm_120) requires PyTorch 2.7.1+cu128. Installed version was 2.2.2+cu121 which does not include sm_120 kernels.
> **Resolution:** Run `scripts/windows/run_all_laptop_validation.bat` — it now auto-detects Blackwell and installs the correct wheel.
> **See also:** [docs/BLACKWELL_ENVIRONMENT.md](BLACKWELL_ENVIRONMENT.md)

---

## Overview

This guide walks Kishore through running the complete Jupiter Shot GPU validation on a Windows laptop with an NVIDIA GPU. The validation covers:

1. Hardware preflight (GPU detection, VRAM measurement, precision support, kernel test)
2. Dense transformer CUDA training (forward/backward, loss decrease, memory)
3. Sparse MoE training (routing stability, expert utilization, aux loss)
4. Checkpoint save and resume (integrity, continuity)
5. Automatic report generation

**Total estimated time:** 30–90 minutes depending on GPU and step count.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA GTX 1650 (4 GB VRAM) | RTX 3060+ (8 GB VRAM) — RTX 5060 Laptop GPU supported |
| VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB |
| CUDA | 11.8 | 12.8 (required for RTX 50-series / Blackwell) |
| Python | 3.10 | 3.11 |
| OS | Windows 10 | Windows 11 |
| Disk | 5 GB free | 20 GB free (PyTorch 2.7.1+cu128 is ~2.5 GB) |

> **RTX 50-series (Blackwell) users:** The setup script auto-detects your GPU and installs PyTorch 2.7.1+cu128. No manual action required. See [BLACKWELL_ENVIRONMENT.md](BLACKWELL_ENVIRONMENT.md) for details.

---

## Step 1 — Environment Setup (One-Time)

Open **PowerShell as Administrator** and run:

```powershell
# Navigate to repo root
cd C:\path\to\jupiter-shot

# Allow script execution (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup (auto-detects GPU architecture, installs correct PyTorch)
.\scripts\windows\setup_laptop_environment.ps1
```

The setup script will:
- Check Python version (3.10 or 3.11 required — 3.12+ not yet supported)
- Detect GPU compute capability (Blackwell sm_120 auto-detected)
- Install PyTorch 2.7.1+cu128 (Blackwell) or 2.2.2+cu121 (pre-Blackwell)
- Install all dependencies from `requirements-laptop.txt`
- Print a summary of your GPU, VRAM, and precision support

---

## Step 2 — Run Full Validation Suite

```batch
scripts\windows\run_all_laptop_validation.bat
```

This runs all stages automatically. Type `YES` when prompted.

**To run 1000 steps instead of 100:**
```batch
scripts\windows\run_all_laptop_validation.bat 1000
```

---

## Step 3 — Run Individual Stages (If Needed)

If the full suite fails at a specific stage, run stages individually:

```batch
REM Stage 1: Hardware check (includes CUDA kernel test)
scripts\windows\run_preflight.bat

REM Stage 2: Dense training
scripts\windows\run_dense_validation.bat laptop_dense_small 100

REM Stage 3: MoE training
scripts\windows\run_moe_validation.bat laptop_moe_small 100

REM Stage 4: Checkpoint resume
scripts\windows\run_resume_validation.bat laptop_dense_small
```

---

## Config Selection by VRAM

The preflight script auto-selects the appropriate config. Manual override:

| VRAM | Dense Config | MoE Config | Approx Params |
|------|-------------|-----------|---------------|
| 4 GB | `laptop_dense_tiny` | `laptop_moe_tiny` | 25M / 40M |
| 6–8 GB | `laptop_dense_small` | `laptop_moe_small` | 85M / 180M |
| 10–16 GB | `laptop_dense_medium` | `laptop_moe_medium` | 350M / 700M |

> **RTX 5060 (8 GB VRAM):** Uses `laptop_dense_small` / `laptop_moe_small`. The 1.3B model is not attempted on 8 GB VRAM.

---

## What the Scripts Measure

### Dense Validation (`run_laptop_dense.py`)

| Metric | Pass Criterion |
|--------|----------------|
| CUDA available | Required |
| CUDA kernel test | Real matmul must succeed (not just is_available()) |
| Forward pass | No exception |
| Backward pass | No exception |
| NaN/Inf count | 0 |
| Loss at step 1 | < 12.0 (near random init: ln(32000) ≈ 10.37) |
| Loss decrease | Last 10 steps mean < first 10 steps mean |
| OOM count | 0 |
| Peak VRAM | < 95% of total VRAM |

### MoE Validation (`run_laptop_moe.py`)

| Metric | Pass Criterion |
|--------|----------------|
| Router stability | No NaN in routing weights |
| Expert utilization | All experts receive > 1% of tokens |
| Utilization CV | < 1.0 (coefficient of variation) |
| Router entropy | > 1.0 bits |
| Aux loss | Decreasing or stable |
| Dropped tokens | < 5% |

### Resume Test (`run_laptop_resume_test.py`)

| Test | Pass Criterion |
|------|----------------|
| Checkpoint save | File written, size > 0 |
| Global step match | Loaded step == saved step |
| Loss at checkpoint | Delta < 1e-5 |
| Loss continuity | Resumed loss within 10% of reference |

---

## Output Files

All results are saved to `benchmarks/results/laptop/`:

```
benchmarks/results/laptop/
  preflight.json          ← Hardware report (includes kernel_test section)
  preflight.txt           ← Human-readable preflight summary
  dense_metrics.jsonl     ← Per-step dense metrics (one JSON per line)
  dense_summary.json      ← Dense training summary
  moe_metrics.jsonl       ← Per-step MoE metrics
  moe_summary.json        ← MoE training summary
  resume_test.json        ← Checkpoint resume test results
  aggregated_metrics.json ← Combined summary (after collect_metrics.py)
  errors.jsonl            ← Any errors encountered

docs/generated/
  LAPTOP_GPU_VALIDATION_DRAFT.md  ← Auto-generated report
```

---

## Sharing Results

After running all stages, share these files with the Jupiter Shot team:

1. `benchmarks/results/laptop/` — all JSON files
2. `docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md` — the draft report
3. `logs/laptop/validation_run_*.log` — full console log

**To generate the draft report manually:**
```batch
python scripts\generate_laptop_validation_draft.py
```

---

## Troubleshooting

### CUDA not available
```
[FAIL] CUDA not available.
```
- Run `nvidia-smi` in Command Prompt. If it fails, reinstall NVIDIA drivers.
- Ensure PyTorch was installed with the correct CUDA version.
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory (OOM)
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
- The script automatically retries with a smaller config.
- If it persists, run with `laptop_dense_tiny` / `laptop_moe_tiny` explicitly.
- Close other GPU-using applications (games, browsers with hardware acceleration).

### Loss is NaN from step 1
- This indicates a numerical stability issue. The script will report it as FAIL.
- Try `--precision fp32` (slower but more stable): edit the config YAML and set `precision: fp32`.

### Import errors
```
ModuleNotFoundError: No module named 'torch'
```
- Activate the virtual environment: `venv\Scripts\activate.bat`
- Or re-run setup: `.\scripts\windows\setup_laptop_environment.ps1 -Force`

### Windows Defender / Antivirus blocking scripts
- Right-click `run_all_laptop_validation.bat` → Properties → Unblock
- Or add the repo folder to Windows Defender exclusions.

### RTX 50-series / Blackwell GPU (sm_120)
```
[FAIL] CUDA kernel test failed: CUDA error: no kernel image is available
       for execution on the device
Environment is BLOCKED — REPAIRABLE
```
This is the expected error when PyTorch 2.2.2+cu121 is installed on a Blackwell GPU.
**Fix:** The updated `run_all_laptop_validation.bat` auto-detects Blackwell and installs PyTorch 2.7.1+cu128.

If you are running the setup manually:
```powershell
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
Then verify:
```python
import torch
print(torch.cuda.get_arch_list())  # Must include 'sm_120'
print(torch.cuda.is_available())   # Must be True
```
See [docs/BLACKWELL_ENVIRONMENT.md](BLACKWELL_ENVIRONMENT.md) for the full repair guide.

### PyTorch download stalls on Blackwell
The PyTorch 2.7.1+cu128 wheel is approximately 2.5 GB. If the download stalls:
1. Temporarily disable antivirus real-time scanning for the `.venv` directory.
2. Try the manual fallback: `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`
3. Ensure at least 20 GB free disk space.
4. Check your internet connection — the download requires sustained throughput.

---

## Known Limitations

These results are from a single laptop GPU and do not represent production-scale training:

- **No distributed training** — single GPU only; multi-node not tested
- **Synthetic data** — results use random token sequences, not real text
- **Thermal throttling** — laptop GPUs may throttle after 10–20 minutes; tokens/sec may decrease
- **Windows overhead** — ~1–1.5 GB VRAM used by Windows/CUDA before training starts
- **No DeepSpeed** — single-GPU validation does not use DeepSpeed ZeRO

These limitations are documented in the validation report. A PASS on laptop validation confirms:
- The model architecture is numerically stable
- The training loop runs without errors
- Checkpointing works correctly

It does **not** confirm:
- 8× A100 distributed training
- DeepSpeed NCCL multi-node communication
- Full 1.3B parameter training run
- 20T scalability

A successful laptop validation may authorize the next controlled validation stage. It does not automatically authorize 47B MoE training.

---

## After Validation

Once all stages pass, the team will:
1. Review `docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md`
2. Update `benchmarks/MONTH1_VALIDATION_RESULTS.md` with measured results
3. Update `docs/MONTH1_GO_NO_GO.md` with GPU gate status
4. Issue a GO/NO-GO for the next controlled validation stage (Stage B: 8× A100 distributed validation)

**Important:** A successful Kuwait laptop test does not automatically authorize 47B MoE training or the 10B-token training run. Those require Stage B distributed-GPU validation.

---

*Last updated: 2026-08-03 | Branch: fix/rtx50-blackwell-validation | Repair: RTX 5060 Blackwell (sm_120) compatibility*
