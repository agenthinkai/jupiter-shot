# Kishore — GPU Operator Checklist
## Jupiter Shot Month 1 Validation | Kuwait Laptop | Run 9

> **Print this page.** Check each box as you complete it.
> **Full guide:** `docs/LAPTOP_GPU_VALIDATION.md`
> **Branch:** `fix/rtx50-blackwell-validation`
> **Run 9 commit:** `094b3e4` (or latest follow-up — confirm with `git rev-parse HEAD`)

---

## Before You Start

- [ ] Laptop plugged into power (not battery)
- [ ] Laptop on a hard flat surface (ventilation)
- [ ] All other GPU-using apps closed (games, Chrome hardware acceleration)
- [ ] At least 20 GB free disk space (for PyTorch download, checkpoints, and logs)
- [ ] Internet connection available (for cloning repo and installing packages)
- [ ] Python 3.10 or 3.11 installed (`python --version`)
- [ ] NVIDIA GPU driver installed (`nvidia-smi` works in Command Prompt)

---

## Hardware Confirmation (Required Before Any Run)

Confirm your GPU matches the validated Kuwait laptop specification:

| Item | Required Value | Your Value |
|------|---------------|------------|
| GPU model | NVIDIA RTX 5060 | __________ |
| Architecture | Blackwell | __________ |
| Compute capability | sm_120 | __________ |
| PyTorch version | 2.7.1+cu128 | __________ |
| CUDA version | 12.8 | __________ |

> **STOP if your GPU is not RTX 5060.** Do not proceed with a different GPU model.
> Run `nvidia-smi` to confirm GPU model and driver version.

---

## Part 1 — Setup (One-Time, ~15 minutes)

- [ ] Open PowerShell **as Administrator**
- [ ] Navigate to repo: `cd C:\path\to\jupiter-shot`
- [ ] Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] Run: `.\scripts\windows\setup_laptop_environment.ps1`
- [ ] Wait for completion — last line should say "Setup complete"
- [ ] Note your GPU model and VRAM from the output: `GPU: _______ VRAM: _______ GB`

---

## Part 2 — Repository Verification (STEP 1)

Open **Command Prompt** (not PowerShell) in the repo root:

```bat
cd C:\path\to\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull origin fix/rtx50-blackwell-validation
git status
git rev-parse HEAD
```

- [ ] Commit hash confirmed: `094b3e4` (or latest follow-up) — record: `__________`
- [ ] Working tree is clean (`git status` shows "nothing to commit")
- [ ] No local repairs applied

---

## Part 3 — Optional Synthetic Diagnostic (STEP 2)

> **OPTIONAL DIAGNOSTIC ONLY — DOES NOT AUTHORIZE FULL VALIDATION**
> A passing result here does not authorize the full GPU run.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode synthetic --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

- [ ] Synthetic diagnostic result: EXIT_CODE = `_____` (optional, record only)

---

## Part 4 — Mandatory Real-Data Preflight (STEP 3)

> **MANDATORY — must return EXIT_CODE=0 before proceeding to Step 4.**
> If this step fails, stop immediately. Preserve all evidence. Make no local repairs.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

Required result: `EXIT_CODE=0` and all 14 preflight gates pass.

**Verify each preflight confirmation:**

- [ ] Wikitext-2 loads successfully
- [ ] GPT-NeoX tokenizer loads successfully
- [ ] `tokenizer.vocab_size` = 50,254
- [ ] `len(tokenizer)` = 50,277
- [ ] `tokenizer_max_token_id` = 50,276
- [ ] Model vocabulary = 50,277
- [ ] Vocabulary contract passes
- [ ] All 14 preflight gates pass
- [ ] EXIT_CODE = 0

**Record preflight result:** EXIT_CODE = `_____`

> **If EXIT_CODE ≠ 0:** Stop here. Do not run Step 4. Preserve `benchmarks\results\laptop\preflight.json` and contact the Jupiter Shot team.

---

## Part 5 — Full Real-Data Validation (STEP 4)

> **Only execute after Part 4 returns EXIT_CODE=0.**
> The command must not silently fall back to synthetic data.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real
echo EXIT_CODE=%ERRORLEVEL%
```

Watch the output — each stage prints `[PASS]` or `[FAIL]`.

**Record results here:**

| Stage | Result | Notes |
|-------|--------|-------|
| Preflight | PASS / FAIL | |
| Dense validation | PASS / FAIL | |
| MoE validation | PASS / FAIL | |
| Resume test | PASS / FAIL | |
| Metrics collection | PASS / FAIL | |
| Report generation | PASS / FAIL | |

**Full run exit code:** `_____`

---

## Part 6 — Record Key Numbers

After the run completes, open `benchmarks\results\laptop\dense_summary.json` and fill in:

| Metric | Value |
|--------|-------|
| GPU model | |
| VRAM total (GB) | |
| Data mode | real |
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
| MoE expert utilization (all 8 populated) | YES / NO |
| MoE utilization CV evaluated | YES / NO |
| moe_accepted | true / false |
| Resume test passed | YES / NO |
| Checkpoint size (MB) | |
| Checkpoint save time (s) | |
| Checkpoint load time (s) | |

---

## Part 7 — Share Results

- [ ] Open `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md`
- [ ] Verify it shows GPU = RTX 5060 (not RTX 5090)
- [ ] Verify it shows data mode = real
- [ ] Verify it shows the correct branch and commit
- [ ] Verify PASS/FAIL matches the JSON verdicts and exit codes
- [ ] Share these files with the team:
  - `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md` ← **primary report**
  - `benchmarks\results\laptop\` (all JSON files) ← raw metrics
  - `logs\laptop\validation_run_*.log`

> **Scope reminder:** This validation confirms CUDA execution, dense training, small MoE routing, checkpoint resume, and thermal controls on a single RTX 5060 Blackwell GPU. It does not validate 8× A100 distributed training or 20T scalability. A successful result authorizes only preparation for controlled Stage B distributed validation. It does **not** authorize 200B/500B pretraining, 20T training, or Azure spending without a separate approved Stage B plan.

---

## If Something Fails

| Error | Action |
|-------|--------|
| `CUDA not available` | Run `nvidia-smi` — if it fails, reinstall NVIDIA drivers |
| `Out of memory` | Script auto-retries smaller config; if still fails, note the error |
| `NaN loss from step 1` | Note it and report — do not retry with different settings |
| `ModuleNotFoundError` | Run `.venv\Scripts\activate.bat` first |
| Script blocked by Windows | Right-click → Properties → Unblock |
| Exit code 3 (EXECUTION_ERROR) | Software defect — preserve error output and report |
| Exit code 4 (SAFETY_STOP) | Hardware safety condition — check GPU temperature immediately |

**Do not modify any scripts or configs to make tests pass.** Record failures as-is.

---

## Questions?

Contact the Jupiter Shot team with:
1. The error message (copy-paste from console)
2. Your GPU model and VRAM (`nvidia-smi` output)
3. The file `benchmarks\results\laptop\preflight.json`
4. The exit code from `echo EXIT_CODE=%ERRORLEVEL%`

---

*Jupiter Shot | Branch: fix/rtx50-blackwell-validation | Run 9 | GPU: NVIDIA RTX 5060 Blackwell (sm_120)*
