# Jupiter Shot — Run 13 Operator Package

**Branch:** `fix/rtx50-blackwell-validation`  
**Authorized commit:** see `git log -1 --format="%H %s"` after pull  
**Prepared by:** Manus AI agent  
**Date:** 2026-08-04

---

## Run 12 Root Cause (Confirmed)

Run 12 failed with:

```
FileNotFoundError: .../training/configs/training/configs/laptop_moe_run7.yaml.yaml
```

**Root cause:** The pipeline passes the full config path
(`training/configs/laptop_moe_run7.yaml`) to each runner. Each runner
then prepended the configs directory and appended `.yaml` again,
producing a doubled path prefix and doubled `.yaml` suffix.

**Fix:** `training/config_path.py` — a shared resolver that accepts full
paths, relative paths, bare names, and `name.yaml` stems without ever
doubling the suffix. All three runners now use it.

---

## What Changed in Run 13

| File | Change |
|---|---|
| `training/config_path.py` | **NEW** — shared config path resolver |
| `scripts/run_laptop_dense.py` | Uses `resolve_config_path()` |
| `scripts/run_laptop_moe.py` | Uses `resolve_config_path()` |
| `scripts/run_laptop_resume_test.py` | Uses `resolve_config_path()` + new artifact fields |
| `scripts/run_laptop_validation_pipeline.py` | Dual gate 10b (CPU + CUDA), device evidence step, c19 strengthened |
| `tests/test_run13_config_resolver.py` | 24 config resolver contract tests |
| `tests/test_run13_subprocess_smoke.py` | 31 production subprocess smoke tests |
| `docs/RUNNER_INTERFACE_MANIFEST.md` | Runner CLI contract documentation |

---

## Pre-Flight Checklist (Windows — RTX 5060 Ti)

### Step 0: Sync and verify

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation

REM Verify commit
git log -1 --format="%H %s"
REM STOP if HEAD does not match the authorized commit
```

### Step 1: Regression tests

```bat
.venv\Scripts\python.exe -m pytest ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  -v
```

**Expected:** 31 + 24 + 26 + 25 = 106 passed, 0 failed, 0 skipped  
*(Counts: test_run13_subprocess_smoke=31, test_run13_config_resolver=24, test_run12_gate10b_real_object=26, test_run12_operator_package=25 — see `tests/TEST_MANIFEST.md`)*  
**STOP if any test fails or is skipped.**

### Step 2: Mandatory real-data preflight

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

**Expected:** EXIT_CODE=0  
**STOP if EXIT_CODE != 0.**

### Step 3: Full real-data validation

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real
echo EXIT_CODE=%ERRORLEVEL%
```

**Expected:** EXIT_CODE=0

### Step 4: Verify artifacts

```bat
REM Replace <run_id> with the run_id printed during the run
type benchmarks\results\laptop\<run_id>\moe_summary.json
type benchmarks\results\laptop\<run_id>\device_evidence.json
```

**Required fields in `moe_summary.json`:**
- `"aux_loss_semantics": "WEIGHTED"`
- `"n_skipped": 0`
- `"n_passed": 20`
- `"n_failed": 0`
- `"n_blocked": 0`

**Required fields in `device_evidence.json`:**
- `"device": "cuda"` — **STOP if this is "cpu"**
- `"compute_capability": "sm_120"` (expected for RTX 5060 Ti / Blackwell)
- `"gpu_name"` — should contain "5060"

**Gate 10b CPU result (expected):**
```
MoE aux-loss verification: 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total
```

**Gate 10b CUDA result (expected):**
```
MoE aux-loss verification: 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total
```

**STOP if gate 10b CUDA shows NOT_EVALUABLE** — this means CUDA was not
detected. Check CUDA drivers and `nvidia-smi`.

---

## OPTIONAL: Synthetic diagnostic (does not authorize full validation)

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode synthetic --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

This is a diagnostic tool only. A passing synthetic run **does not
authorize** proceeding to full validation or training. Use only to
isolate environment issues before running the mandatory real-data
preflight.

---

## Stop Conditions

Do **not** proceed to Azure / 8×A100 / 200B / 500B / 20T training until
**all** of the following are satisfied:

1. `git log -1` shows the authorized Run 13 commit
2. All 106 targeted regression tests pass with 0 skips (see `tests/TEST_MANIFEST.md`)
3. `scripts\windows\run_all_laptop_validation.bat --data-mode real` exits 0
4. `moe_summary.json` contains `"n_passed": 20, "n_failed": 0, "n_blocked": 0, "n_skipped": 0`
5. `device_evidence.json` contains `"device": "cuda"`
6. Gate 10b CUDA prints `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`

---

## Artifact Path Convention

All artifacts are written to:

```
benchmarks\results\laptop\<run_id>\
  moe_summary.json
  device_evidence.json
  preflight.json
  run_manifest.json
  errors.jsonl
```

where `<run_id>` is printed at the start of the run in the format
`YYYYMMDD_HHMMSS_UTC`.

---

## Known Pre-Existing Failures (not introduced by Run 13)

The full test suite has 11 pre-existing failures on the base commit
`31a5275`. These are not introduced by Run 13. Run 13 introduces zero
new failures (verified by stash-and-retest).
