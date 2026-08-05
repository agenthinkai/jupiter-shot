# Jupiter Shot — Run 15 Corrected Operator Package
## Branch: `fix/rtx50-blackwell-validation`
## Authorized Commit: `dc79637`
## Authorized Node Count: **174**
## Authorized SHA-256: `43142de49d415c5bd8e606aeac51cccec6976eb53c3f2500752b38f0e604c0a7`

---

## Corrections from Run 15 Initial Package

The initial Run 15 package stated an authorized total of 169 collected nodes. The actual
deterministic collection is **174 nodes**. The discrepancy comes entirely from
`tests/test_run13_config_resolver.py`:

| File | Functions | Collected Nodes | Explanation |
|---|---|---|---|
| `test_run13_config_resolver.py` | 24 | **29** | Two parametrized tests expand: `test_no_double_yaml_suffix` → 3 variants; `TestAllFourInputForms` → 4 variants. 24 − 2 + 7 = **29** |

All other files have equal function and node counts (no parametrized expansion).

The initial package did not distinguish between function count and collected-node count.
This package eliminates manual test-count maintenance by providing an automated manifest
generator and verifier.

---

## Per-File Node Breakdown

| File | Functions | Collected Nodes |
|---|---|---|
| `test_run12_gate10b_real_object.py` | 26 | 26 |
| `test_run12_operator_package.py` | 25 | 25 |
| `test_run13_config_resolver.py` | 24 | **29** |
| `test_run13_subprocess_smoke.py` | 31 | 31 |
| `test_run14_integration.py` | 15 | 15 |
| `test_run14_same_pass_provenance.py` | 2 | 2 |
| `test_run14_integration_contracts.py` | 28 | 28 |
| `test_run15_regression.py` | 18 | 18 |
| **Total** | **169** | **174** |

---

## Automated Manifest Generator and Verifier

Manual test-count maintenance is eliminated. The manifest is generated once (by the
developer) and verified by Kishore before every run.

### Files

| File | Purpose |
|---|---|
| `scripts/generate_test_manifest.py` | Generates `run15_authorized_nodes.txt` and `run15_authorized_manifest.json` using the pytest Python API |
| `scripts/verify_test_manifest.py` | Verifies current collection against the authorized manifest; exits 0 on exact match, 1 on any discrepancy |
| `tests/run15_authorized_nodes.txt` | Sorted list of 174 authorized node IDs, one per line |
| `tests/run15_authorized_manifest.json` | Machine-readable manifest with SHA-256, function count, collected count, branch, commit |

### Manifest Contents

```json
{
  "schema_version": "1.0",
  "test_files": ["tests/test_run12_gate10b_real_object.py", "..."],
  "function_count": 169,
  "collected_node_count": 174,
  "node_list_sha256": "43142de49d415c5bd8e606aeac51cccec6976eb53c3f2500752b38f0e604c0a7",
  "generated_with": "scripts/generate_test_manifest.py",
  "generated_at": "2026-...",
  "branch": "fix/rtx50-blackwell-validation",
  "commit": "dc79637"
}
```

### Verifier Behavior

The verifier fails with exit code 1 on any of the following:

- Missing node (test removed or renamed)
- Unexpected node (test added without re-generating the manifest)
- Changed parametrized variant name
- Duplicate node
- Collection failure (import error, missing file)
- SHA-256 hash mismatch
- Count mismatch

The verifier normalizes Windows backslash paths to forward slashes before comparison.

---

## Verifier Regression Tests

`tests/test_run15_manifest_verifier.py` — 10 tests (V01–V10):

| Test | What It Proves |
|---|---|
| V01 | Accepts the exact 174-node collection |
| V02 | Detects a missing node |
| V03 | Detects an unexpected node |
| V04 | Detects a changed parametrized variant |
| V05 | Detects a duplicate node |
| V06 | Detects collection failure |
| V07 | Detects hash mismatch |
| V08 | Produces deterministic output (same SHA-256 on repeated calls) |
| V09 | Handles Windows paths (backslash normalization) |
| V10 | Returns nonzero exit code on every mismatch |

---

## Run 15 Repairs (Unchanged from Initial Package)

| # | Defect | Fix |
|---|---|---|
| 1 | `auxiliary_load_balancing_loss` silently dropped | `ROUTER_METRIC_KEYS` allowlist in `training/router_metrics.py`; explicit filter in `run_laptop_moe.py` |
| 2 | MoE artifact missing schema fields | `schema_version`, `timestamp`, `branch`, `commit`, `device`, `gpu_name` added; `validate_moe_artifact_schema()` called before every write |
| 3 | Thermal monitor polled once per training step | `ThermalMonitor` background thread (1 Hz, independent of step timing) in `training/thermal_monitor.py` |

No model code, routing logic, thermal thresholds, acceptance criteria, or training
configurations were changed. `strategy/jupiter-20t` was not touched.

---

## Test Results

| Suite | Collected Nodes | Result |
|---|---|---|
| Run 12–14 (7 files) | 151 | 151 passed |
| Run 15 regression (T01–T18) | 18 | 18 passed |
| **Run 15 manifest verifier (V01–V10, new)** | **10** | **10 passed** |
| **Targeted total (9 files)** | **184** | **184 passed** |
| Full suite | 798 | 11 failed (pre-existing), 771 passed, 16 skipped |

Zero new failures introduced.

---

## Physical Preparation — Operator Attestation Required

Before starting the GPU workload, Kishore must confirm each of the following manually.
These items are operator-attested and cannot be software-verified:

```
[ ] Original power adapter connected (not battery-only)
[ ] Laptop placed on a hard, flat surface
[ ] Cooling vents unobstructed on all sides
[ ] Normal high-performance cooling profile enabled
[ ] Laptop will remain attended throughout the entire run
```

Record this attestation in the run summary before proceeding to Step 7.

---

## Kishore's Preflight Sequence

### Step 1 — Verify commit

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation
git log -1
```

Expected output: `dc79637 feat(run15): metric allowlist, complete artifact schema, independent thermal monitor`

**STOP if commit hash does not match `dc79637`.**

---

### Step 2 — Verify clean repository

```bat
git status
```

Expected: `nothing to commit, working tree clean`

**STOP if any local modifications are present.**

---

### Step 3 — Run automated manifest verifier

```bat
.venv\Scripts\python.exe scripts\verify_test_manifest.py
```

Expected output:
```
[VERIFY] Loading authorized manifest...
[VERIFY] Authorized: 174 nodes, SHA-256: 43142de49d415c5b...
[VERIFY] Collecting current nodes from 8 files...
[VERIFY OK] Exact match: 174 nodes, SHA-256: 43142de49d415c5b...
[VERIFY OK] All checks passed.
```

Expected exit code: `0`

**STOP if exit code is not 0 or output contains `[VERIFY FAIL]`.**

---

### Step 4 — Require exact 174-node match

The verifier enforces this automatically. No manual counting required.

**STOP if the verifier did not print `Exact match: 174 nodes`.**

---

### Step 5 — Run the 174 authorized tests

```bat
.venv\Scripts\python.exe -m pytest ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  tests\test_run14_same_pass_provenance.py ^
  tests\test_run14_integration_contracts.py ^
  tests\test_run15_regression.py ^
  -v
```

---

### Step 6 — Require 174 passed, zero failed/errors/skipped

Expected final line: `174 passed`

**STOP if any test fails, errors, or is skipped.**

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

### Step 10 — Verify thermal monitor during the loaded run

During the run, `thermal_telemetry.jsonl` will be written to the output directory.
After the run, verify the artifact:

```bat
python -c "
import json, sys
a = json.load(open('results\moe_summary.json'))
required = ['schema_version','timestamp','run_id','branch','commit',
            'outcome','exit_code','data_mode','device','gpu_name',
            'aux_loss_semantics','router_metrics_available_under_gc',
            'checkpoint_status','thermal_monitor_healthy']
missing = [k for k in required if k not in a or a[k] is None]
if missing:
    print('MISSING FIELDS:', missing); sys.exit(1)
print('Schema OK')
print('outcome:', a['outcome'])
print('thermal_monitor_healthy:', a['thermal_monitor_healthy'])
print('thermal_peak_c:', a.get('thermal_peak_c'))
print('thermal_stop_triggered:', a.get('thermal_stop_triggered'))
"
```

Expected:
```
Schema OK
outcome: PASS
thermal_monitor_healthy: True
thermal_peak_c: <value below 90>
thermal_stop_triggered: False
```

**STOP if `thermal_monitor_healthy` is False or `thermal_stop_triggered` is True.**

---

### Step 11 — Preserve evidence, make no local repair

If any gate fails, preserve all logs and artifacts exactly as produced.
Do not attempt local repair. Report the exact failure output.

---

## Stop Conditions Summary

| Condition | Action |
|---|---|
| Commit hash ≠ `dc79637` | STOP |
| Working tree not clean | STOP |
| Verifier exit code ≠ 0 | STOP |
| Verifier node count ≠ 174 | STOP |
| Any targeted test fails/errors/skips | STOP |
| Physical attestation not completed | STOP |
| Preflight exit code ≠ 0 | STOP |
| `outcome != "PASS"` in any runner artifact | STOP |
| `thermal_monitor_healthy: false` | STOP |
| `thermal_stop_triggered: true` | STOP |
| Any missing required artifact field | STOP |
