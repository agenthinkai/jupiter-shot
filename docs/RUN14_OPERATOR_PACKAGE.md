# Jupiter Shot — Run 14 Operator Package

**Branch:** `fix/rtx50-blackwell-validation`  
**Authorized commit:** see `git log -1 --format="%H %s"` after pull  
**Prepared by:** Manus AI agent  
**Date:** 2026-08-04

---

## Run 14 Root Causes (Confirmed)

Three defects were identified after Run 13 (first real GPU training on RTX 5060):

### Defect 1 — Checkpoint filename doubled-suffix construction

**Symptom:** Checkpoint files were named `dense_laptop_dense_run7.yaml_step100.pt`
instead of `dense_laptop_dense_run7_step100.pt`.

**Root cause:** The runners called `f"{config_name}.yaml"` to build the checkpoint
stem, but `config_name` already contained the full path including `.yaml`. The
`safe_checkpoint_name()` helper added in Run 13 was not yet used in the checkpoint
filename construction path.

**Fix:** All three runners (`run_laptop_dense.py`, `run_laptop_moe.py`,
`run_laptop_resume_test.py`) now call `safe_checkpoint_name(config_name)` to
produce a clean stem. A `checkpoint_status` field is also written to each
runner's summary artifact.

### Defect 2 — Router metrics lost under gradient checkpointing

**Symptom:** `router_metrics` was an empty list `[]` in the MoE forward-pass
output when `gradient_checkpointing=True` and the model was in training mode.

**Root cause:** `torch.utils.checkpoint.checkpoint()` cannot return Python dicts
through its boundary — only tensors. The original code returned `(out, aux)` from
the checkpointed function and appended an empty dict `{}` as a placeholder.

**Fix:** A router-only `torch.no_grad()` probe pass runs immediately after the
checkpoint call, calling only `layer.moe_ffn.router(x.detach())` to obtain the
metrics without affecting the gradient graph. The probe result is appended to
`all_router_metrics`.

### Defect 3 — Incomplete MoE artifact contract

**Symptom:** `moe_summary.json` lacked fields required by downstream pipeline
validation: `aux_loss_semantics`, `aux_loss_contract`,
`router_metrics_available_under_gc`. The write was also non-atomic (partial
reads possible). Stale artifacts from a previous run were silently accepted.

**Fix:**
- `aux_loss_semantics: "WEIGHTED"` added to the initial summary dict.
- `aux_loss_contract` string documents the weighting semantics.
- `router_metrics_available_under_gc: True` confirms the Defect 2 fix.
- Atomic write: `.tmp` file written then renamed to prevent partial reads.
- Stale artifact rejection: warns and overwrites when `run_id` mismatches.

---

## What Changed in Run 14

| File | Change |
|---|---|
| `training/config_path.py` | Added `safe_checkpoint_name()` function |
| `scripts/run_laptop_dense.py` | Uses `safe_checkpoint_name()`, adds `checkpoint_status` |
| `scripts/run_laptop_moe.py` | Uses `safe_checkpoint_name()`, adds `checkpoint_status`, `aux_loss_semantics`, `aux_loss_contract`, `router_metrics_available_under_gc`, atomic write, stale artifact rejection |
| `scripts/run_laptop_resume_test.py` | Uses `safe_checkpoint_name()`, adds `requested_data_mode`, `resume_test_data_source`, `resume_uses_wikitext` |
| `training/models/moe.py` | Router-only no-grad probe after GC call to populate `router_metrics` |
| `scripts/run_laptop_validation_pipeline.py` | Preflight step 13 extended: verifies `router_metrics` non-empty under GC |
| `tests/test_run14_integration.py` | **NEW** — 15 integration tests (T01–T15) |
| `tests/TEST_MANIFEST.md` | **NEW** — authoritative test count manifest |
| `tests/test_run10_regression.py` | Updated `test_moe_aux_loss_zero_was_the_defect` to accept Run 14 probe pattern |
| `docs/RUN13_OPERATOR_PACKAGE.md` | Corrected hardcoded 112 → 106 test count |

---

## Test Count Manifest

See `tests/TEST_MANIFEST.md` for authoritative counts.

| File | Tests |
|------|------:|
| `tests/test_run12_gate10b_real_object.py` | 26 |
| `tests/test_run12_operator_package.py` | 25 |
| `tests/test_run13_config_resolver.py` | 24 |
| `tests/test_run13_subprocess_smoke.py` | 31 |
| `tests/test_run14_integration.py` | 15 |
| **Total (Run 12–14 targeted)** | **121** |

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

### Step 1: Regression tests (Run 12–14)

```bat
.venv\Scripts\python.exe -m pytest ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  -v
```

**Expected:** 26 + 25 + 24 + 31 + 15 = 121 passed, 0 failed, 0 skipped  
*(See `tests/TEST_MANIFEST.md` for per-file counts)*  
**STOP if any test fails or is skipped.**

### Step 2: Mandatory real-data preflight

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

**Expected:** EXIT_CODE=0  
**STOP if EXIT_CODE != 0.**

Preflight step 13 now includes a Run 14 GC router metrics check:
- It runs a training-mode forward pass with `gradient_checkpointing=True`
- Verifies `router_metrics` is non-empty (confirms the Defect 2 fix)
- Prints `[OK]` for both eval-mode and GC-mode checks

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
- `"aux_loss_contract"` — non-empty string mentioning `TopKRouter`
- `"router_metrics_available_under_gc": true`
- `"checkpoint_status": "saved"` (or `"failed: ..."` if checkpoint failed)
- `"n_passed": 20, "n_failed": 0, "n_blocked": 0, "n_skipped": 0`

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

### Step 5: Verify checkpoint filename is correct

```bat
REM Checkpoint file should be named: dense_laptop_dense_run7_step<N>.pt
REM NOT: dense_laptop_dense_run7.yaml_step<N>.pt
dir benchmarks\results\laptop\<run_id>\checkpoints\
```

**Expected:** No `.yaml` in the checkpoint filename stem.

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

1. `git log -1` shows the authorized Run 14 commit
2. All 121 targeted regression tests pass with 0 skips (see `tests/TEST_MANIFEST.md`)
3. `scripts\windows\run_all_laptop_validation.bat --data-mode real` exits 0
4. `moe_summary.json` contains `"n_passed": 20, "n_failed": 0, "n_blocked": 0, "n_skipped": 0`
5. `device_evidence.json` contains `"device": "cuda"`
6. Gate 10b CUDA prints `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`
7. `moe_summary.json` contains `"aux_loss_semantics": "WEIGHTED"`
8. `moe_summary.json` contains `"router_metrics_available_under_gc": true`
9. Checkpoint filename contains no `.yaml` in the stem

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

## Known Pre-Existing Failures (not introduced by Run 14)

The full test suite has 11 pre-existing failures on the base commit
`5106291`. These are not introduced by Run 14. Run 14 introduces zero
new failures (verified by full suite run: 11 failed, 713 passed, 16 skipped).

Pre-existing failures are in:
- `tests/test_mesh.py` (8 failures — async test infrastructure)
- `tests/test_models.py` (1 failure — parameter count)
- `tests/test_tokenizer_vocab.py` (2 failures — vocab resolution)

---

## What Run 14 Does NOT Authorize

A successful Run 14 on the RTX 5060 Ti validates:
- CUDA execution on Blackwell architecture
- Dense training with correct checkpoint filenames
- MoE routing with router metrics under gradient checkpointing
- Atomic artifact writes and stale artifact rejection
- Complete MoE artifact contract

It does **not** authorize:
- 8× A100 distributed training
- DeepSpeed / NCCL validation
- Full 1.3B parameter training run
- Proof of 20T scalability
- Azure deployment
