# Jupiter Shot — Run 12 Operator Package

**Branch:** `fix/rtx50-blackwell-validation`  
**Prepared for:** Kishore (RTX 5060, Windows 11, PyTorch 2.7.1+cu128)  
**Authorized commit:** `PLACEHOLDER_COMMIT` ← replaced after this doc commit  
**Spec:** Run 12 — Gate 10b accounting invariant + aux_loss semantics contract

---

## What Changed in Run 12

| Item | Run 11 (broken) | Run 12 (fixed) |
|---|---|---|
| Defect 1: aux_loss_coeff path | `config.moe.aux_loss_coef` (AttributeError) | `config.router_aux_loss_coeff` |
| Defect 2: gradient_checkpointing path | `config.gradient_checkpointing` (always None) | `config.base.gradient_checkpointing` |
| Defect 3: MoE submodule probe | `("moe","mlp","ffn")` (never found) | `("moe_ffn","moe","mlp","ffn")` |
| c05 check | Sane-range check (`aux_loss < 1.0`) | Semantics label (`WEIGHTED`) |
| c12 check ID | `c12_gc_bool` | `c12_gc_enabled` |
| c17 check | Attribute existence only | Expert count verified == 8 |
| c20 check | Isolated grad nonzero | `total_loss.backward()` router grad nonzero |
| SKIPPED status | Not supported (mock checks counted as PASS) | Supported; `n_skipped` in return dict |
| Accounting invariant | Not enforced | `n_passed + n_failed + n_blocked + n_skipped == n_total` asserted |
| aux_loss semantics | Not documented | `aux_loss_semantics: WEIGHTED` in return dict |

---

## Authoritative Kishore Run 12 Sequence

### STEP 1 — Verify exact commit and clean tree

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation
git status
git rev-parse HEAD
```

**Required conditions — STOP if either fails:**

1. `git rev-parse HEAD` must output exactly: `PLACEHOLDER_COMMIT`
2. `git status` must show: `nothing to commit, working tree clean`

Do not proceed if HEAD differs from the authorized commit.  
Do not apply local repairs.  
Do not use `git pull` without `--ff-only`.

---

### STEP 2 — Run real-object CPU regression tests

```bat
.venv\Scripts\python.exe -m pytest tests\test_run12_gate10b_real_object.py -v
```

**Required result:** All tests pass. Zero failures. Zero errors.

> **Note:** These are real-object **CPU** regression tests. They verify gate 10b
> logic using a real `MoETransformer` instance on CPU. They do **not** prove
> CUDA execution. CUDA execution is proven in Steps 3–4 when the pipeline
> records `device = cuda` in the artifact.

Do not use `python` or `python3` — use `.venv\Scripts\python.exe` only.

---

### STEP 3 — Run mandatory real-data preflight

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real --preflight-only
echo EXIT_CODE=%ERRORLEVEL%
```

**Required preflight result — STOP if any condition fails:**

| Condition | Required |
|---|---|
| Exit code | `0` |
| All mandatory preflight gates | PASS |
| Gate 10b on CPU | 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total |
| Gate 10b on CUDA | 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total |
| `aux_loss_semantics` | `WEIGHTED` |
| Auxiliary loss | positive and finite |
| Isolated aux-loss router gradients | finite and nonzero |
| Total-loss router gradients | finite and nonzero |
| `vocabulary_contract_passed` | `true` |

Do **not** call `python scripts\run_laptop_validation_pipeline.py` directly.  
Do **not** use `--data-mode synthetic` or `--data-mode auto` for official validation.  
Do **not** proceed to Step 5 if exit code is not `0`.  
Preserve all evidence if exit code is not `0`.

---

### STEP 4 — Confirm preflight exit code and gate 10b results

Confirm the following from the preflight output before proceeding:

```
MoE aux-loss verification: 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total
```

Both the CPU and CUDA gate 10b runs must show this line.

CUDA execution is confirmed only when the artifact explicitly records `device = cuda`.

**Required CUDA evidence:**

| Field | Required value |
|---|---|
| GPU model | NVIDIA RTX 5060 |
| Compute capability | sm_120 |
| PyTorch version | 2.7.1+cu128 |
| CUDA runtime | 12.8 |
| Gate 10b artifact `device` field | `cuda` |
| CUDA aux-loss measurements | present |
| CUDA router-gradient measurements | present |

---

### STEP 5 — Run full real-data validation

> **Only execute after Step 3 returns EXIT_CODE=0.**

```bat
scripts\windows\run_all_laptop_validation.bat --data-mode real
echo EXIT_CODE=%ERRORLEVEL%
```

**Required full run result:**

1. Uses real Wikitext-2 data — no synthetic fallback.
2. Produces fresh dense metrics.
3. Produces fresh MoE metrics.
4. Produces fresh router metrics.
5. Produces a fresh MoE acceptance artifact.
6. Produces a fresh checkpoint-resume result.
7. Produces a fresh pipeline summary.
8. Returns exit code `0` for PASS.

Do **not** call `python scripts\run_laptop_validation_pipeline.py` directly.  
Do **not** use `--data-mode synthetic` or `--data-mode auto`.

---

### STEP 6 — Confirm dense, MoE, router and checkpoint results

Review the console output for `[PASS]` on each stage:

- Preflight
- Dense validation
- MoE validation
- Resume test
- Metrics collection
- Report generation

---

### STEP 7 — Confirm fresh artifacts and consistent run_id

Locate the Run 12 artifact directory by the current `run_id` from the pipeline summary:

```bat
REM The pipeline prints the run_id at start and end.
REM Artifacts are written to: benchmarks\results\laptop\<run_id>\
REM Example: benchmarks\results\laptop\run_20260804_143022_abc123\
```

Open `moe_summary.json` inside the **current run directory** (not a generic path):

```bat
REM Replace <run_id> with the actual run_id printed by the pipeline
type benchmarks\results\laptop\<run_id>\moe_summary.json
```

**Required artifact fields:**

| Field | Required value |
|---|---|
| `schema_version` | `"1.0"` |
| `run_id` | matches the pipeline summary |
| `commit` | `PLACEHOLDER_COMMIT` |
| `branch` | `fix/rtx50-blackwell-validation` |
| `data_mode` | `real` |
| `aux_loss_semantics` | `"WEIGHTED"` |
| `aux_loss` | positive and finite |
| `n_passed` | `20` |
| `n_failed` | `0` |
| `n_blocked` | `0` |
| `n_skipped` | `0` |
| `n_total` | `20` |
| `status` | `"ok"` |
| `device` | `cuda` (CUDA evidence) |
| artifact timestamp | belongs to Run 12 (not a prior run) |

Do **not** inspect `artifacts\moe_summary.json` (generic path — may be stale).  
Do **not** reuse an artifact from a prior run.

---

### STEP 8 — Return the complete evidence package

Share with the Jupiter Shot team:

- `docs\generated\LAPTOP_GPU_VALIDATION_DRAFT.md` — primary report
- `benchmarks\results\laptop\<run_id>\` — all JSON artifacts
- `logs\laptop\validation_run_*.log` — full console log

---

## Stop Conditions

Do **not** proceed to Azure, 8×A100, 200B, 500B, or 20T training unless:

1. Step 3 preflight returns exit code `0`.
2. Gate 10b prints `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total` on both CPU and CUDA.
3. The CUDA artifact explicitly records `device = cuda`.
4. `moe_summary.json` contains `"aux_loss_semantics": "WEIGHTED"`.
5. `moe_summary.json` contains `"n_skipped": 0`.
6. `aux_loss > 0` in `moe_summary.json`.
7. All real-object CPU regression tests pass with zero failures.

---

## aux_loss Weighting Contract

The `aux_loss` value returned by `MoETransformer.forward()` is **WEIGHTED**:

```
TopKRouter.forward():
    raw_load_balance_loss = mean(fraction_tokens × fraction_capacity)
    raw_z_loss = mean(log(sum(exp(router_logits)))^2)
    aux_loss = aux_loss_coeff × raw_load_balance_loss
             + z_loss_coeff × raw_z_loss   ← weighted INSIDE router

MoETransformer.forward():
    total_aux_loss += layer.moe_ffn(x)["aux_loss"]   ← no second multiplication
    total_loss = lm_loss + total_aux_loss             ← direct sum
```

`out["aux_loss"]` is already scaled by `router_aux_loss_coeff` (default 0.01).
The raw unscaled value is available as `router_metrics["aux_loss_unscaled"]`
in the non-checkpoint path only.

---

## Files Changed in Run 12

| File | Change |
|---|---|
| `scripts/run_laptop_validation_pipeline.py` | Gate 10b rewritten: 3 defects fixed, 20 named checks, SKIPPED status, accounting invariant, 10-field check records, `aux_loss_semantics` in return dict |
| `tests/test_run11_exit_code_matrix.py` | Mock tests updated: check IDs renamed, test_d semantics updated |
| `tests/test_run12_gate10b_real_object.py` | New real-object CPU regression tests |
| `tests/test_run12_operator_package.py` | New operator-package contract regression tests (15 checks) |
| `docs/RUN12_OPERATOR_PACKAGE.md` | This file — corrected operator sequence |
| `docs/KISHORE_GPU_OPERATOR_CHECKLIST.md` | Run 12 section updated with corrected commands |
| `docs/LAPTOP_GPU_VALIDATION.md` | Run 12 status updated |
| `benchmarks/MONTH1_VALIDATION_RESULTS.md` | Run 12 section corrected |
