# Jupiter Shot — Run 15 Operator Package (Corrected v2)

**Branch:** `fix/rtx50-blackwell-validation`
**Authorized Node Count:** **207**
**Authorized SHA-256:** `e5cbd90497bcbe536536abeed8c452fd9b26f9af354ae47731bbd287c99596ee`

---

## Corrections in This Package (v2)

### Defect 4 — Windows UTF-8 reproducibility: bare text I/O calls

On Windows with cp1252 as the platform default encoding, `open()`, `read_text()`,
or `write_text()` calls without `encoding="utf-8"` raise `UnicodeDecodeError` when
reading YAML config files that contain U+2190 (←). Both `laptop_dense_run7.yaml`
and `laptop_moe_run7.yaml` contain this character.

**Affected files and call sites fixed:**

| File | Call sites fixed |
|---|---|
| `scripts/run_laptop_dense.py` | 7 (lines 188, 415, 443, 445, 512, 599, 616) |
| `scripts/run_laptop_resume_test.py` | 4 (lines 99, 228, 310, 326) |
| `training/checkpoint.py` | 5 (lines 109, 154, 211, 268, 326) |

Already clean (no changes): `run_laptop_moe.py`, `run_laptop_validation_pipeline.py`,
`config_loader.py`, `config_path.py`, `thermal_monitor.py`,
`generate_laptop_validation_draft.py`.

**Fix:** Added `encoding="utf-8"` to all 16 affected call sites. An AST-based audit
(`test_ast_audit_all_9_production_files`) now permanently guards against regression.

**New test file:** `tests/test_run15_utf8_reproducibility.py` — 23 nodes covering
R01–R13 requirements and 4 AST audit tests. Tests run without `PYTHONUTF8` or
`PYTHONIOENCODING` set.

### Manifest update

Two new test files added to the authorized manifest:
- `tests/test_run15_manifest_verifier.py` — 10 nodes (V01–V10)
- `tests/test_run15_utf8_reproducibility.py` — 23 nodes (R01–R13 + AST)

Authorized total: **174 → 207 nodes** (+33).

---

## Run 15 Repairs (Unchanged from Initial Package)

| # | Defect | Fix |
|---|---|---|
| 1 | `auxiliary_load_balancing_loss` silently dropped by prefix filter | `ROUTER_METRIC_KEYS` allowlist in `training/router_metrics.py`; explicit filter in `run_laptop_moe.py` |
| 2 | MoE artifact missing schema fields | `schema_version`, `timestamp`, `branch`, `commit`, `device`, `gpu_name` added; `validate_moe_artifact_schema()` called before every write |
| 3 | Thermal monitor polled once per training step | `ThermalMonitor` background thread (1 Hz, independent of step timing) in `training/thermal_monitor.py` |

No model code, routing logic, thermal thresholds, acceptance criteria, or training
configurations were changed. `strategy/jupiter-20t` was not touched.

---

## Authorized Test Manifest

| File | Functions | Collected Nodes |
|---|---|---|
| `test_run12_gate10b_real_object.py` | 26 | 26 |
| `test_run12_operator_package.py` | 25 | 25 |
| `test_run13_config_resolver.py` | 24 | 29 |
| `test_run13_subprocess_smoke.py` | 31 | 31 |
| `test_run14_integration.py` | 15 | 15 |
| `test_run14_same_pass_provenance.py` | 2 | 2 |
| `test_run14_integration_contracts.py` | 28 | 28 |
| `test_run15_regression.py` | 18 | 18 |
| `test_run15_manifest_verifier.py` | 10 | 10 |
| `test_run15_utf8_reproducibility.py` | 20 | 23 |
| **Total** | **199** | **207** |

`test_run13_config_resolver.py`: 24 functions → 29 nodes (5 parametrized expansions).
`test_run15_utf8_reproducibility.py`: 20 functions → 23 nodes (3 parametrized expansions).

---

## PACKAGE_ATTESTED Block

```
PACKAGE_ATTESTED
  package:                     RUN15_OPERATOR_PACKAGE_v2
  branch:                      fix/rtx50-blackwell-validation
  commit:                      679fba0a0bfae5317eddce89a09579e799d6b7c9
  parent_commit:               0592fbef138dbf4965e92b5f5be9538d637c0975
  git_tree:                    0366e4c0f352363e39c5726049253b0b04ecd493
  authorized_nodes:            207
  sha256:                      e5cbd90497bcbe536536abeed8c452fd9b26f9af354ae47731bbd287c99596ee
  defects_repaired:            4
  model_code_changed:          NO
  thresholds_changed:          NO
  acceptance_criteria_changed: NO
  training_configs_changed:    NO
  strategy_jupiter_20t_touched: NO
  pythonutf8_required:         NO
  pythonioencoding_required:   NO
END_PACKAGE_ATTESTED
```

---

## PHYSICAL_OPERATOR_ATTESTATION

This section must be completed by the person physically beside the laptop immediately before the full GPU run begins. Software, Manus, and execution agents must not complete this section. The full GPU run cannot begin unless every physical answer is YES.

```
PHYSICAL_OPERATOR_ATTESTATION
  Operator name:
  Date and time:
  Laptop location:
  Original power adapter connected:                        YES / NO
  Laptop on a hard stable surface:                         YES / NO
  Cooling vents unobstructed:                              YES / NO
  Manufacturer high-performance cooling profile enabled:   YES / NO
  Laptop will remain attended throughout GPU load:         YES / NO
  Operator is prepared to stop the run if required:        YES / NO
  Operator signature or typed confirmation:
END_PHYSICAL_OPERATOR_ATTESTATION
```

---

## Test Results

| Suite | Collected Nodes | Result |
|---|---|---|
| Run 12–14 (7 files) | 151 | 151 passed |
| Run 15 regression (T01–T18) | 18 | 18 passed |
| Run 15 manifest verifier (V01–V10) | 10 | 10 passed |
| **Run 15 UTF-8 reproducibility (R01–R13 + AST, new)** | **23** | **23 passed** |
| **Targeted total (10 files)** | **207** | **207 passed** |
| Full suite | 811 | 11 failed (pre-existing), 784 passed, 16 skipped |

Zero new failures introduced.

---

## Physical Preparation — Operator Attestation Required

Before starting the GPU workload, Kishore must confirm each of the following manually:

```
[ ] Original power adapter connected (not battery-only)
[ ] Laptop placed on a hard, flat surface
[ ] Cooling vents unobstructed on all sides
[ ] Normal high-performance cooling profile enabled
[ ] Laptop will remain attended throughout the entire run
```

Record this attestation in the run summary before proceeding to Step 8.

---

## Kishore's Preflight Sequence

### Step 1 — Sync

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation
git log -1
```

Expected: latest commit message contains `run15-utf8-windows-reproducibility`

```bat
git status
```

Expected: `nothing to commit, working tree clean`

**STOP if working tree is not clean.**

---

### Step 2 — Verify manifest (automated)

```bat
.venv\Scripts\python.exe scripts\verify_test_manifest.py
```

Expected output:
```
[VERIFY OK] Exact match: 207 nodes, SHA-256: e5cbd90497bcbe53...
[VERIFY OK] All checks passed.
```

Expected exit code: `0`

**STOP if exit code is not 0 or output contains `[VERIFY FAIL]`.**

---

### Step 3 — Run targeted suite WITHOUT PYTHONUTF8

```bat
set PYTHONUTF8=
set PYTHONIOENCODING=
.venv\Scripts\python.exe -m pytest ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  tests\test_run14_same_pass_provenance.py ^
  tests\test_run14_integration_contracts.py ^
  tests\test_run15_regression.py ^
  tests\test_run15_manifest_verifier.py ^
  tests\test_run15_utf8_reproducibility.py ^
  -v
```

Expected: `207 passed, 0 failed, 0 errors, 0 skipped`

**STOP if any test fails, errors, or is skipped.**

---

### Step 4 — Require exact 207-node match

The verifier enforces this automatically. No manual counting required.

**STOP if the verifier did not print `Exact match: 207 nodes`.**

---

### Step 5 — Confirm PYTHONUTF8 was not set

```bat
echo %PYTHONUTF8%
```

Expected: empty line (variable not set).

**STOP if PYTHONUTF8 is set to any value.**

---

### Step 6 — Regenerate manifest (optional, for attestation)

```bat
.venv\Scripts\python.exe scripts\generate_test_manifest.py
```

Expected:
```
[MANIFEST] collected_node_count: 207
[MANIFEST] node_list_sha256:      e5cbd90497bcbe536536abeed8c452fd9b26f9af354ae47731bbd287c99596ee
[MANIFEST] Generation complete. Exit 0.
```

If the SHA-256 differs, **STOP** and report the full output.

---

### Step 7 — Physical preparation attestation

Complete the operator attestation checklist above. Record it in the run summary.

---

### Step 8 — Run real-data preflight

```bat
.venv\Scripts\python.exe scripts\run_laptop_validation_pipeline.py ^
  --data-mode real --preflight-only
```

Expected exit code: `0`

**STOP if exit code is not 0.**

---

### Step 9 — Run full real-data validation

Only proceed if all previous gates passed.

```bat
.venv\Scripts\python.exe scripts\run_laptop_validation_pipeline.py ^
  --data-mode real
```

---

### Step 10 — Verify artifact schema after run

```bat
.venv\Scripts\python.exe -c "
import json, pathlib
moe = json.loads(pathlib.Path('benchmarks/results/laptop/moe_summary.json').read_text(encoding='utf-8'))
required = ['schema_version','timestamp','run_id','branch','commit','device','gpu_name',
            'aux_loss_semantics','aux_loss_contract','router_metrics_available_under_gc',
            'checkpoint_status','thermal_health']
missing = [k for k in required if k not in moe]
print('MISSING:', missing if missing else 'none')
print('schema_version:', moe.get('schema_version'))
print('outcome:', moe.get('outcome'))
print('thermal_health:', moe.get('thermal_health',{}).get('status'))
print('router_metrics_available_under_gc:', moe.get('router_metrics_available_under_gc'))
"
```

Expected: `MISSING: none`, `outcome: PASS`, `thermal_health: OK`

**STOP if any field is missing or outcome is not PASS.**

---

### Step 11 — Preserve evidence, make no local repair

If any gate fails, preserve all logs and artifacts exactly as produced.
Do not attempt local repair. Report the exact failure output.

---

## Stop Conditions Summary

| Condition | Action |
|---|---|
| Working tree not clean | STOP |
| Verifier exit code ≠ 0 | STOP |
| Verifier node count ≠ 207 | STOP |
| SHA-256 ≠ `e5cbd904...` | STOP |
| Any targeted test fails/errors/skips | STOP |
| `PYTHONUTF8` is set during Step 3 | STOP |
| Physical attestation not completed | STOP |
| Preflight exit code ≠ 0 | STOP |
| `outcome != "PASS"` in any runner artifact | STOP |
| `thermal_health.status != "OK"` | STOP |
| Any missing required artifact field | STOP |

---

## Count History

| Package | Claimed | Collected Nodes | Explanation |
|---|---|---|---|
| RUN13 initial | 112 | — | Transcription error (60+27+25=112 was wrong) |
| RUN13 corrected | 111 | — | pytest collected 111 (5 parametrized expansions) |
| RUN14 initial | 151 | 151 | Run 12–14 targeted suite |
| RUN15 initial | 169 | 174 | 5 parametrized expansions in test_run13_config_resolver |
| RUN15 corrected v1 | 174 | 174 | Correct per-file breakdown documented; automated verifier added |
| **RUN15 corrected v2** | **207** | **207** | +33 nodes: manifest verifier (10) + UTF-8 tests (23) |
