# Jupiter Shot Month 1 Validation | Kuwait Laptop | Run 12

> **Print this page.** Check each box as you complete it.
> **Full guide:** `docs/LAPTOP_GPU_VALIDATION.md`
> **Operator package:** `docs/RUN12_OPERATOR_PACKAGE.md`
> **Branch:** `fix/rtx50-blackwell-validation`
> **Authorized commit:** `PLACEHOLDER_COMMIT` ← replaced after this doc commit

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
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation
git status
git rev-parse HEAD
```

**STOP if either condition fails. Do not apply local repairs. Do not proceed.**

- [ ] `git rev-parse HEAD` outputs exactly: `PLACEHOLDER_COMMIT` — record: `__________`
- [ ] `git status` shows: `nothing to commit, working tree clean`

> Do not use `git pull` without `--ff-only`.  
> Do not accept "or latest follow-up" — the authorized commit is exact.

---

## Part 3 — Real-Object CPU Regression Tests (STEP 2)

```bat
.venv\Scripts\python.exe -m pytest tests\test_run12_gate10b_real_object.py -v
```

- [ ] All tests pass
- [ ] Zero failures
- [ ] Zero errors

> **These are CPU regression tests.** They verify gate 10b logic on a real
> `MoETransformer` instance using CPU. They do **not** prove CUDA execution.
> CUDA execution is proven in Step 4 when the artifact records `device = cuda`.
>
> Do **not** use `python` or `python3`. Use `.venv\Scripts\python.exe` only.

---

## Part 3b — OPTIONAL Synthetic Diagnostic (Pre-Check Only)

> **OPTIONAL** — This step is a quick diagnostic only. It does **not** authorize the full GPU validation run.
> Synthetic mode does not authorize full validation. Run this only if you want to confirm the environment loads before committing to real data.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode synthetic --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

> A passing synthetic diagnostic does not authorize proceeding to Part 4.
> Real-data preflight (Part 4) is always required.

---

## Part 4 — Mandatory Real-Data Preflight (STEP 3)
> **MANDATORY — must return EXIT_CODE=0 before proceeding to Step 5.**
> If this step fails: STOP. Preserve all evidence. Make no local repairs. Do not run full validation.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

> Do **not** call `python scripts\run_laptop_validation_pipeline.py` directly.  
> Do **not** use `--data-mode synthetic` or `--data-mode auto`.

**Verify each preflight confirmation:**

- [ ] Wikitext-2 loads successfully
- [ ] GPT-NeoX tokenizer loads successfully
- [ ] `tokenizer.vocab_size` = 50,254
- [ ] `len(tokenizer)` = 50,277
- [ ] `tokenizer_max_token_id` = 50,276
- [ ] Model vocabulary = 50,277
- [ ] `vocabulary_contract_passed` = `true`
- [ ] All mandatory preflight gates pass
- [ ] Gate 10b on CPU: `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`
- [ ] Gate 10b on CUDA: `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`
- [ ] `aux_loss_semantics` = `WEIGHTED`
- [ ] Auxiliary loss is positive and finite
- [ ] Isolated aux-loss router gradients are finite and nonzero
- [ ] Total-loss router gradients are finite and nonzero
- [ ] EXIT_CODE = 0

**Record preflight result:** EXIT_CODE = `_____`

> **If EXIT_CODE ≠ 0:** Stop here. Do not run Step 5. Preserve all evidence in
> `benchmarks\results\laptop\` and contact the Jupiter Shot team.

---

## Part 5 — Full Real-Data Validation (STEP 4 and STEP 5)

> **Only execute after Part 4 returns EXIT_CODE=0.**
> The command must not silently fall back to synthetic data.

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real
echo EXIT_CODE=%ERRORLEVEL%
```

> Do **not** call `python scripts\run_laptop_validation_pipeline.py` directly.  
> Do **not** use `--data-mode synthetic` or `--data-mode auto`.

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

## Part 6 — Confirm Fresh Artifacts and Consistent run_id (STEP 6 and STEP 7)

The pipeline prints the `run_id` at start and end. Artifacts are written to:

```
benchmarks\results\laptop\<run_id>\
```

Open `moe_summary.json` inside the **current run directory** (not a generic path):

```bat
REM Replace <run_id> with the actual run_id printed by the pipeline
type benchmarks\results\laptop\<run_id>\moe_summary.json
```

> Do **not** inspect `artifacts\moe_summary.json` — that path may be stale.  
> Do **not** reuse an artifact from a prior run.

**Verify each artifact field:**

- [ ] `schema_version` = `"1.0"`
- [ ] `run_id` matches the pipeline summary
- [ ] `commit` = `PLACEHOLDER_COMMIT`
- [ ] `branch` = `fix/rtx50-blackwell-validation`
- [ ] `data_mode` = `real`
- [ ] `aux_loss_semantics` = `"WEIGHTED"`
- [ ] `aux_loss` is positive and finite
- [ ] `n_passed` = `20`
- [ ] `n_failed` = `0`
- [ ] `n_blocked` = `0`
- [ ] `n_skipped` = `0`
- [ ] `n_total` = `20`
- [ ] `status` = `"ok"`
- [ ] `device` = `cuda` ← CUDA execution confirmed
- [ ] Artifact timestamp belongs to Run 12 (not a prior run)

**CUDA execution is confirmed only when `device = cuda` appears in the artifact.**

---

## Part 7 — Record Key Numbers

After the run completes, open `benchmarks\results\laptop\<run_id>\dense_summary.json` and fill in:

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

## Part 8 — Share Results (STEP 8)

- [ ] Open `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md`
- [ ] Verify it shows GPU = RTX 5060 (not RTX 5090)
- [ ] Verify it shows data mode = real
- [ ] Verify it shows the correct branch and commit
- [ ] Verify PASS/FAIL matches the JSON verdicts and exit codes
- [ ] Share these files with the team:
  - `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md` ← **primary report**
  - `benchmarks\results\laptop\<run_id>\` (all JSON files) ← raw artifacts
  - `logs\laptop\validation_run_*.log`

> **Scope reminder:** This validation confirms CUDA execution, dense training,
> small MoE routing, checkpoint resume, and thermal controls on a single RTX 5060
> Blackwell GPU. It does not validate 8× A100 distributed training or 20T
> scalability. A successful result authorizes only preparation for controlled
> Stage B distributed validation. It does **not** authorize 200B/500B
> pretraining, 20T training, or Azure spending without a separate approved
> Stage B plan.

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
3. The file `benchmarks\results\laptop\<run_id>\preflight.json`
4. The exit code from `echo EXIT_CODE=%ERRORLEVEL%`

---

*Jupiter Shot | Branch: fix/rtx50-blackwell-validation | Run 12 | GPU: NVIDIA RTX 5060 Blackwell (sm_120)*
