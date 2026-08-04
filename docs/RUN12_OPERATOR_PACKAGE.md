# Jupiter Shot — Run 12 Operator Package

**Branch:** `fix/rtx50-blackwell-validation`  
**Prepared for:** Kishore (RTX 5060, Windows 11, PyTorch 2.7.1+cu128)  
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

## Exact Windows Commands

### Step 1 — Pull the latest branch

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull origin fix/rtx50-blackwell-validation
```

### Step 2 — Verify the environment

```bat
python -m pytest tests\test_run12_gate10b_real_object.py -v
```

Expected output (all 27 tests pass, 0 skipped on PyTorch environment):

```
======================== 27 passed, 0 warnings in XX.XXs ========================
```

### Step 3 — Run the full preflight pipeline

```bat
python scripts\run_laptop_validation_pipeline.py
```

Expected gate 10b output:

```
[Preflight 10b/14] MoE auxiliary-loss verification (Run 12 corrected) ...
  ok      [c01_step10_status]
  ok      [c02_aux_loss_key]
  ok      [c03_aux_loss_type]
  ok      [c04_aux_loss_positive]
  ok      [c05_aux_loss_semantics]
  ok      [c06_model_config]
  ok      [c07_aux_loss_coeff]
  ok      [c08_coeff_numeric]
  ok      [c09_coeff_positive]
  ok      [c10_config_base]
  ok      [c11_gc_attr]
  ok      [c12_gc_enabled]
  ok      [c13_gc_aux_nonzero]
  ok      [c14_model_layers]
  ok      [c15_moe_submodule]
  ok      [c16_router_attr]
  ok      [c17_expert_count]
  ok      [c18_aux_grad_fn]
  ok      [c19_isolated_router_grad]
  ok      [c20_total_loss_backward]
  MoE aux-loss verification: 20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total
```

### Step 4 — Run the full test suite

```bat
python -m pytest --tb=short -q
```

Expected: **613+ passed, 11 pre-existing failures** (test_mesh.py async tests,
test_parameter_count_1b3, test_tokenizer_vocab vocab-50277 tests). Zero new
failures introduced by Run 12.

### Step 5 — Confirm moe_summary.json after the run

```bat
type artifacts\moe_summary.json
```

Verify the following fields are present and correct:

| Field | Expected |
|---|---|
| `aux_loss_semantics` | `"WEIGHTED"` |
| `aux_loss_semantics_note` | (non-empty string describing weighting contract) |
| `n_passed` | `20` |
| `n_failed` | `0` |
| `n_blocked` | `0` |
| `n_skipped` | `0` |
| `n_total` | `20` |
| `status` | `"ok"` |

---

## Stop Conditions

Do **not** proceed to Azure, 8×A100, 200B, 500B, or 20T training unless all of
the following are true on Kishore's PyTorch environment (CPU **and** CUDA):

1. Gate 10b prints `20 PASS, 0 FAIL, 0 BLOCKED, 0 SKIPPED / 20 total`
2. `moe_summary.json` contains `"aux_loss_semantics": "WEIGHTED"`
3. `moe_summary.json` contains `"n_skipped": 0`
4. `aux_loss > 0` in `moe_summary.json`
5. All 27 real-object regression tests pass with zero skips

---

## aux_loss Weighting Contract

The `aux_loss` value returned by `MoETransformer.forward()` is **WEIGHTED**:

```
TopKRouter.forward():
    raw_load_balance_loss = mean(fraction_tokens × fraction_capacity)
    raw_z_loss = mean(log(sum(exp(router_logits)))^2)
    aux_loss = aux_loss_coeff × raw_load_balance_loss
             + z_loss_coeff × raw_z_loss   ← weighted inside router

MoETransformer.forward():
    total_aux_loss += layer.moe_ffn(x)["aux_loss"]  ← no second multiplication
    total_loss = lm_loss + total_aux_loss            ← direct sum
```

**Implication:** `aux_loss` in all artifacts is already scaled by `router_aux_loss_coeff`
(default 0.01). The raw unscaled value is available as `router_metrics["aux_loss_unscaled"]`
in the non-checkpoint path only.

---

## Files Changed in Run 12

| File | Change |
|---|---|
| `scripts/run_laptop_validation_pipeline.py` | Gate 10b rewritten: 3 defects fixed, 20 named checks, SKIPPED status, accounting invariant, 10-field check records, `aux_loss_semantics` in return dict |
| `tests/test_run11_exit_code_matrix.py` | Mock tests updated: check IDs renamed, test_d semantics updated (aux_loss=1.5 no longer raises) |
| `tests/test_run12_gate10b_real_object.py` | New tests added: 10-field structure, expert count, total_loss backward, aux_loss semantics, accounting invariant |
| `docs/architecture_20t/COST_EFFICIENCY_HYPOTHESIS.md` | New: 20T cost-efficiency hypothesis with 4 scenarios and 18 metrics |
| `benchmarks/MONTH1_VALIDATION_RESULTS.md` | Run 11 checker defects documented; Run 12 section added |
| `docs/RUN12_OPERATOR_PACKAGE.md` | This file |
