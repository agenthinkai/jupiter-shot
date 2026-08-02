# Kishore — GPU Operator Checklist
## Jupiter Shot Month 1 Validation | Kuwait Laptop

> **Print this page.** Check each box as you complete it.  
> **Full guide:** `docs/LAPTOP_GPU_VALIDATION.md`

---

## Before You Start

- [ ] Laptop plugged into power (not battery)
- [ ] Laptop on a hard flat surface (ventilation)
- [ ] All other GPU-using apps closed (games, Chrome hardware acceleration)
- [ ] At least 5 GB free disk space
- [ ] Internet connection available (for cloning repo and installing packages)
- [ ] Python 3.10 or 3.11 installed (`python --version`)
- [ ] NVIDIA GPU driver installed (`nvidia-smi` works in Command Prompt)

---

## Part 1 — Setup (One-Time, ~15 minutes)

- [ ] Open PowerShell **as Administrator**
- [ ] Navigate to repo: `cd C:\path\to\jupiter-shot`
- [ ] Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] Run: `.\scripts\windows\setup_laptop_environment.ps1`
- [ ] Wait for completion — last line should say "Setup complete"
- [ ] Note your GPU model and VRAM from the output: `GPU: _______ VRAM: _______ GB`

---

## Part 2 — Full Validation Run (~30–90 minutes)

- [ ] Open Command Prompt (not PowerShell) in the repo root
- [ ] Run: `scripts\windows\run_all_laptop_validation.bat`
- [ ] Type `YES` when prompted
- [ ] Watch the output — each stage prints `[PASS]` or `[FAIL]`

**Record results here:**

| Stage | Result | Notes |
|-------|--------|-------|
| Preflight | PASS / FAIL | |
| Dense validation | PASS / FAIL | |
| MoE validation | PASS / FAIL | |
| Resume test | PASS / FAIL | |
| Metrics collection | PASS / FAIL | |
| Report generation | PASS / FAIL | |

---

## Part 3 — Record Key Numbers

After the run completes, open `benchmarks\results\laptop\dense_summary.json` and fill in:

| Metric | Value |
|--------|-------|
| GPU model | |
| VRAM total (GB) | |
| Dense config used | |
| Dense steps completed | |
| Dense first loss | |
| Dense last loss | |
| Dense tokens/sec | |
| Dense peak VRAM (GB) | |
| Dense NaN/Inf count | |
| MoE config used | |
| MoE steps completed | |
| MoE first loss | |
| MoE last loss | |
| MoE avg aux loss (last 10) | |
| MoE avg router entropy (bits) | |
| MoE dropped token % | |
| Resume test passed | YES / NO |
| Checkpoint size (MB) | |
| Checkpoint save time (s) | |
| Checkpoint load time (s) | |

---

## Part 4 — Share Results

- [ ] Open `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md`
- [ ] Verify it contains your GPU model and results (not all dashes)
- [ ] Share these files with the team:
  - `benchmarks\results\laptop\` (all JSON files)
  - `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md`
  - `logs\laptop\validation_run_*.log`

---

## If Something Fails

| Error | Action |
|-------|--------|
| `CUDA not available` | Run `nvidia-smi` — if it fails, reinstall NVIDIA drivers |
| `Out of memory` | Script auto-retries smaller config; if still fails, note the error |
| `NaN loss from step 1` | Note it and report — do not retry with different settings |
| `ModuleNotFoundError` | Run `venv\Scripts\activate.bat` first |
| Script blocked by Windows | Right-click → Properties → Unblock |

**Do not modify any scripts or configs to make tests pass.** Record failures as-is.

---

## Questions?

Contact the Jupiter Shot team with:
1. The error message (copy-paste from console)
2. Your GPU model and VRAM
3. The file `benchmarks\results\laptop\preflight.json`

---

*Jupiter Shot | Branch: validation/kuwait-laptop-gpu | 2026-08-02*
