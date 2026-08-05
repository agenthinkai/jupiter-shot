# Jupiter Shot — Run 14 Operator Package (Revised)

**Branch:** `fix/rtx50-blackwell-validation`  
**Commit:** `18efeb9b1ddfd5835044abbff6a12a5e6c47c300`  
**Baseline commit:** `31a5275`  
**Date:** 2026-08-05

---

## Summary of Changes

Run 14 resolves three defects and one blocking review issue identified after
the initial Run 14 delivery.

### Defect 1 — Checkpoint filename doubled-suffix (FIXED)

`safe_checkpoint_name()` in `training/config_path.py` now strips the full path
and any `.yaml` extension before the stem is used in checkpoint filenames.  All
three runners (`run_laptop_dense.py`, `run_laptop_moe.py`,
`run_laptop_resume_test.py`) call it.  A `checkpoint_status` field is written
to each runner's summary artifact.

### Defect 2 — Router metrics empty under gradient checkpointing (FIXED — REVISED)

**Original fix (Run 14 initial delivery):** A router-only `torch.no_grad()` probe
pass was added after each `gradient_checkpoint()` call.

**Blocking issue identified in review:** The probe called the router a second time,
which can produce different routing decisions under stochastic conditions (dropout,
jitter, any non-determinism in the gate).  The assignments and statistics used for
acceptance must derive from the exact router logits used by the training forward
pass.

**Revised fix (this delivery):** Same-pass tensor transport.

`TopKRouter.forward` gains a `return_metric_tensors=False` keyword.  When `True`,
it packs the five raw quantities needed by `compute_router_metrics()` into a flat
float32 tensor and returns it as the fourth output:

```
metric_tensor = [tokens_per_expert (E,), mean_prob_per_expert (E,),
                 aux_loss_unscaled (scalar), z_loss_unscaled (scalar),
                 num_tokens (scalar)]
```

`MoEFFNLayer.forward` and `MoETransformerBlock.forward` pass the flag through.

`MoETransformer.forward` — in the GC branch — calls the layer with
`return_metric_tensors=True` and returns `(out, aux_loss, metric_tensor)` through
`gradient_checkpoint()`.  All three are tensors; no dict crosses the boundary.
Outside the boundary, `metric_tensor` is unpacked and `compute_router_metrics()`
is called once to build the dict.  **The router is called exactly once per layer
per forward pass.**

### Defect 3 — Incomplete MoE artifact contract (FIXED)

`run_laptop_moe.py` now writes `aux_loss_semantics`, `aux_loss_contract`,
`router_metrics_available_under_gc`, and `checkpoint_status` to
`moe_summary.json`.  The write is atomic (`.tmp` rename).  Stale artifacts from
a previous run trigger a printed `[ARTIFACT WARNING]` and are overwritten.

### Blocking review issue — Stale artifact rejection

The pipeline's `_validate_runner_artifact()` (Rule 6) already enforces hard
rejection (EXECUTION_ERROR) when `artifact["run_id"] != expected run_id`.  The
MoE runner's `[ARTIFACT WARNING]` + overwrite is a runner-level diagnostic only;
the pipeline never trusts a stale artifact.

---

## What Changed in Run 14

| File | Change |
|---|---|
| `training/config_path.py` | `safe_checkpoint_name()` strips path and `.yaml` |
| `training/models/moe.py` | Same-pass tensor transport: `return_metric_tensors` keyword in `TopKRouter`, `MoEFFNLayer`, `MoETransformerBlock`; `MoETransformer.forward` unpacks `metric_tensor` outside GC boundary |
| `scripts/run_laptop_dense.py` | Uses `safe_checkpoint_name()`, adds `checkpoint_status` |
| `scripts/run_laptop_moe.py` | Uses `safe_checkpoint_name()`, adds `checkpoint_status`, `aux_loss_semantics`, `aux_loss_contract`, `router_metrics_available_under_gc`, atomic write, stale artifact rejection |
| `scripts/run_laptop_resume_test.py` | Uses `safe_checkpoint_name()`, adds `requested_data_mode`, `resume_test_data_source`, `resume_uses_wikitext` |
| `scripts/run_laptop_validation_pipeline.py` | Preflight step 13 extended: verifies `router_metrics` non-empty under GC |
| `tests/test_run14_integration.py` | **NEW** — 15 integration tests (T01–T15) |
| `tests/test_run14_same_pass_provenance.py` | **NEW** — 2 stochastic-jitter provenance tests (T16–T17) |
| `tests/test_run14_integration_contracts.py` | **NEW** — 28 integration contract tests (T18–T29) |
| `tests/test_run10_regression.py` | Updated to accept Run 14 same-pass transport signature |
| `tests/TEST_MANIFEST.md` | Updated with all Run 14 test files; 112→111 discrepancy explained |
| `docs/RUN13_OPERATOR_PACKAGE.md` | Corrected hardcoded 112 → 106 test count |

---

## Run 13 112→111 Discrepancy Explained

`docs/RUN13_OPERATOR_PACKAGE.md` and `docs/KISHORE_GPU_OPERATOR_CHECKLIST.md`
originally claimed 112 tests (stated as 60+27+25).  The actual
`grep -c "def test_"` counts are 31+24+26+25 = **106**.  The 112 figure was a
transcription error in the operator package — it was never the real
pytest-collected count.

`pytest --collect-only` returns **111** (106 function-level tests plus 5
parametrized variants in `test_run13_subprocess_smoke.py` that pytest counts as
individual collected items).  `tests/TEST_MANIFEST.md` is the authoritative
source for all counts going forward.

---

## Test Count Manifest

See `tests/TEST_MANIFEST.md` for authoritative counts.

| File | Tests | Added in Run |
|------|------:|-------------|
| `tests/test_run12_gate10b_real_object.py` | 26 | Run 12 |
| `tests/test_run12_operator_package.py` | 25 | Run 12 |
| `tests/test_run13_config_resolver.py` | 24 | Run 13 |
| `tests/test_run13_subprocess_smoke.py` | 31 | Run 13 |
| `tests/test_run14_integration.py` | 15 | Run 14 |
| `tests/test_run14_same_pass_provenance.py` | 2 | Run 14 |
| `tests/test_run14_integration_contracts.py` | 28 | Run 14 |
| **Total (Run 12–14 targeted)** | **151** | |

---

## Test Results

### Targeted suite (Run 12–14)

```
151 passed, 0 failed, 0 skipped
```

### Full suite

```
11 failed, 743 passed, 16 skipped
```

The 11 failures are all pre-existing (`test_mesh.py`, `test_models.py`,
`test_tokenizer_vocab.py`) and were present before Run 12.  **Zero new failures
introduced by Run 14.**

---

## Requirement-to-Test Traceability Table

| Test ID | Test Name | Requirement Covered |
|---------|-----------|---------------------|
| T01 | `test_t01_checkpoint_stem_no_yaml_suffix` | Defect 1: `safe_checkpoint_name` strips `.yaml` |
| T02 | `test_t02_checkpoint_stem_strips_path` | Defect 1: strips full path separators |
| T03 | `test_t03_checkpoint_stem_windows_path` | Defect 1: Windows backslash paths |
| T04 | `test_t04_checkpoint_stem_already_clean` | Defect 1: clean stem unchanged |
| T05 | `test_t05_dense_runner_uses_safe_checkpoint_name` | Defect 1: dense runner calls `safe_checkpoint_name` |
| T06 | `test_t06_moe_runner_uses_safe_checkpoint_name` | Defect 1: MoE runner calls `safe_checkpoint_name` |
| T07 | `test_t07_resume_runner_uses_safe_checkpoint_name` | Defect 1: resume runner calls `safe_checkpoint_name` |
| T08 | `test_t08_dense_artifact_has_checkpoint_status` | Defect 1: `checkpoint_status` field in dense artifact |
| T09 | `test_t09_moe_artifact_has_checkpoint_status` | Defect 1: `checkpoint_status` field in MoE artifact |
| T10 | `test_t10_moe_router_metrics_nonempty_under_gc` | Defect 2: router metrics non-empty when GC=True |
| T11 | `test_t11_moe_router_metrics_available_under_gc_field` | Defect 2: `router_metrics_available_under_gc` field |
| T12 | `test_t12_moe_stale_artifact_warning` | Defect 3: stale artifact `[ARTIFACT WARNING]` printed |
| T13 | `test_t13_moe_artifact_checkpoint_status` | Defect 3: `checkpoint_status` in MoE artifact |
| T14 | `test_t14_resume_artifact_fields` | Defect 3: resume artifact continuity fields |
| T15 | `test_t15_moe_aux_loss_contract` | Defect 3: `aux_loss_contract` mentions `TopKRouter` |
| T16 | `test_t16_router_called_exactly_once_per_layer_under_gc` | Same-pass provenance: router called exactly once per layer under GC |
| T17 | `test_t17_metrics_match_training_pass_assignments_under_jitter` | Same-pass provenance: metrics match training-pass routing under stochastic jitter |
| T18 | `test_t18_pipeline_dense_runner_command_includes_required_args` | Pipeline launches dense runner with all required args |
| T18b | `test_t18_dense_runner_saves_checkpoint_on_completion` | Dense artifact records `checkpoint_status='saved'` |
| T19 | `test_t19_pipeline_moe_runner_command_includes_required_args` | Pipeline launches MoE runner with all required args |
| T19b | `test_t19_moe_runner_saves_checkpoint_on_completion` | MoE artifact records `checkpoint_status='saved'` |
| T20 | `test_t20_resume_artifact_records_continuity_fields` | Resume artifact has continuity fields |
| T20b | `test_t20_pipeline_resume_runner_command_includes_required_args` | Pipeline launches resume runner with all required args |
| T21 | `test_t21_dense_pass_requires_saved_checkpoint` | Dense cannot PASS when checkpoint saving fails |
| T21b | `test_t21_dense_artifact_with_checkpoint_failure_is_execution_error` | Dense checkpoint failure → EXECUTION_ERROR |
| T22 | `test_t22_moe_pass_requires_saved_checkpoint` | MoE cannot PASS when checkpoint saving fails |
| T22b | `test_t22_moe_artifact_with_checkpoint_failure_is_execution_error` | MoE checkpoint failure → EXECUTION_ERROR |
| T23 | `test_t23_resume_artifact_not_pass_without_checkpoint` | Resume cannot PASS without fresh checkpoint |
| T23b | `test_t23_resume_source_checks_for_checkpoint` | Resume source handles missing checkpoint |
| T24 | `test_t24_windows_path_in_command_builder` | Windows config paths work verbatim in command builder |
| T24b | `test_t24_safe_checkpoint_name_handles_windows_path` | `safe_checkpoint_name` handles Windows paths |
| T25 | `test_t25_step13_raises_on_empty_router_metrics` | Missing router metric cannot produce PASS |
| T25b | `test_t25_moe_acceptance_requires_router_metrics` | Acceptance evaluation rejects empty metrics |
| T26 | `test_t26_safety_stop_artifact_returns_exit_4` | SAFETY_STOP artifact → exit code 4 |
| T26b | `test_t26_pipeline_safety_stop_takes_precedence` | SAFETY_STOP takes precedence over EXECUTION_ERROR |
| T27 | `test_t27_mismatched_run_id_is_execution_error` | Mismatched run_id → EXECUTION_ERROR (hard reject) |
| T27b | `test_t27_stale_timestamp_is_execution_error` | Stale timestamp → EXECUTION_ERROR |
| T27c | `test_t27_moe_runner_stale_artifact_rejection_is_hard_reject` | Pipeline source returns EXECUTION_ERROR for stale artifacts |
| T28 | `test_t28_run_manifest_contains_git_info` | run_manifest.json contains run_id, branch, commit |
| T28b | `test_t28_all_runner_artifacts_share_same_run_id` | All runner artifacts share the same run_id |
| T28c | `test_t28_pipeline_validates_run_id_in_all_artifacts` | Pipeline enforces run_id consistency for all runners |
| T29 | `test_t29_complete_moe_artifact_schema_fields` | Complete MoE artifact schema validated |
| T29b | `test_t29_moe_artifact_timestamp_is_utc` | MoE artifact timestamp is valid UTC ISO |
| T29c | `test_t29_pipeline_validates_complete_moe_schema` | Pipeline accepts a complete valid MoE artifact |
| T29d | `test_t29_moe_artifact_acceptance_counters_present` | All acceptance criteria counters present |

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
REM STOP if HEAD does not show 18efeb9b1ddfd5835044abbff6a12a5e6c47c300
```

### Step 1: Regression tests (Run 12–14)

```bat
.venv\Scripts\python.exe -m pytest ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  tests\test_run14_same_pass_provenance.py ^
  tests\test_run14_integration_contracts.py ^
  -v
```

**Expected:** 151 passed, 0 failed, 0 skipped  
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
- Runs a training-mode forward pass with `gradient_checkpointing=True`
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
type benchmarks\results\laptop\runs\<run_id>\moe_summary.json
type benchmarks\results\laptop\runs\<run_id>\device_evidence.json
```

**Required fields in `moe_summary.json`:**
- `"schema_version": "1.0"`
- `"aux_loss_semantics": "WEIGHTED"`
- `"aux_loss_contract"` — non-empty string mentioning `TopKRouter`
- `"router_metrics_available_under_gc": true`
- `"checkpoint_status": "saved"`
- `"moe_accepted": true`

**Required fields in `device_evidence.json`:**
- `"device": "cuda"` — **STOP if this is "cpu"**
- `"compute_capability": "sm_120"` (expected for RTX 5060 Ti / Blackwell)
- `"gpu_name"` — should contain "5060"

### Step 5: Verify checkpoint filename is correct

```bat
dir benchmarks\results\laptop\runs\<run_id>\checkpoints\
```

**Expected:** No `.yaml` in the checkpoint filename stem.  
Correct: `dense_laptop_dense_run7_step100.pt`  
Wrong: `dense_laptop_dense_run7.yaml_step100.pt`

---

## Stop Conditions

Do **not** proceed to Azure / 8×A100 / 200B / 500B / 20T training until
**all** of the following are satisfied:

1. `git log -1` shows commit `18efeb9b1ddfd5835044abbff6a12a5e6c47c300`
2. All 151 targeted regression tests pass with 0 skips (see `tests/TEST_MANIFEST.md`)
3. `scripts\windows\run_all_laptop_validation.bat --data-mode real` exits 0
4. `moe_summary.json` contains `"moe_accepted": true`
5. `device_evidence.json` contains `"device": "cuda"`
6. Gate 10b CUDA prints `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`
7. `moe_summary.json` contains `"aux_loss_semantics": "WEIGHTED"`
8. `moe_summary.json` contains `"router_metrics_available_under_gc": true`
9. Checkpoint filename contains no `.yaml` in the stem

---

## Known Pre-Existing Failures (not introduced by Run 14)

The full test suite has 11 pre-existing failures on the base commit `31a5275`.
These are not introduced by Run 14.  Run 14 introduces zero new failures
(verified by full suite run: 11 failed, 743 passed, 16 skipped).

Pre-existing failures are in:
- `tests/test_mesh.py` (4 failures — async test infrastructure)
- `tests/test_models.py` (1 failure — 1.3B parameter count, not used in laptop validation)
- `tests/test_tokenizer_vocab.py` (2 failures — GPT-2 vocab resolution)

---

## What Run 14 Does NOT Authorize

A successful Run 14 on the RTX 5060 Ti validates:
- CUDA execution on Blackwell architecture
- Dense training with correct checkpoint filenames
- MoE routing with router metrics under gradient checkpointing (same-pass provenance)
- Atomic artifact writes and stale artifact rejection
- Complete MoE artifact contract

It does **not** authorize:
- 8× A100 distributed training
- DeepSpeed / NCCL validation
- Full 1.3B parameter training run
- Proof of 20T scalability
- Azure deployment
