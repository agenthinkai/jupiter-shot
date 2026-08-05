# Jupiter Shot — Run 15 Operator Package
## Branch: `fix/rtx50-blackwell-validation`
## Authorized Commit: `(fill in after git log -1 on your machine)`

---

## Summary

Run 15 delivers three production-path repairs that blocked the Run 14 PASS:

| # | Defect | Root Cause | Fix |
|---|---|---|---|
| 1 | `auxiliary_load_balancing_loss` silently dropped from acceptance evaluation | `run_laptop_moe.py` line 517 used `not k.startswith(("aux", ...))` which matched `auxiliary_load_balancing_loss` as well as the intended `aux_loss` | Replaced with explicit `ROUTER_METRIC_KEYS` allowlist in `training/router_metrics.py`; only exact key `TRAINING_LEVEL_EXCLUDE_KEY = "aux_loss"` is excluded |
| 2 | MoE artifact missing required schema fields | `moe_summary.json` lacked `schema_version`, `timestamp`, `branch`, `commit`, `device`, `gpu_name` | Added all required fields to initial summary dict; added `validate_moe_artifact_schema()` and `_write_artifact()` helpers; schema validated before every write |
| 3 | Thermal monitor polled once per training step (step-timed) | `_check_thermal()` called inside the training loop — a 60-second step means 60 seconds of undetected overtemperature | Replaced with `ThermalMonitor` background thread (1 Hz independent sampling, telemetry file, warning/stop events, singleton guard, health summary) |

---

## Files Changed

| File | Change |
|---|---|
| `training/router_metrics.py` | Added `ROUTER_METRIC_KEYS` allowlist and `TRAINING_LEVEL_EXCLUDE_KEY` constant |
| `training/thermal_monitor.py` | **New module**: `ThermalMonitor` class with background thread, 1 Hz sampling, `thermal_telemetry.jsonl`, warning/stop events, singleton guard |
| `scripts/run_laptop_moe.py` | Imports `ROUTER_METRIC_KEYS`, `TRAINING_LEVEL_EXCLUDE_KEY`; uses allowlist filter; adds `validate_moe_artifact_schema()`, `_write_artifact()`; integrates `ThermalMonitor`; records `thermal_monitor_healthy`, `thermal_peak_c`, `thermal_warn_samples`, `thermal_stop_triggered`, `thermal_total_samples` in artifact |
| `tests/test_run15_regression.py` | **New file**: 18 regression tests (T01–T18) |
| `tests/TEST_MANIFEST.md` | Updated to collected-node counts; Run 15 row added; 169-node total |

---

## Repair 1 Detail: Metric Filter Allowlist

**Root cause:** Line 517 of `run_laptop_moe.py` filtered per-step records with:
```python
{k: v for k, v in m.items()
 if not k.startswith(("step", "loss", "aux", "lr", ...))}
```
The `"aux"` prefix matched both `"aux_loss"` (correct to exclude — training-level scalar) and
`"auxiliary_load_balancing_loss"` (must survive — required acceptance key). The evaluator
received records missing this key and returned `NOT_EVALUABLE` instead of `PASS`.

**Fix:** `training/router_metrics.py` now exports:
```python
ROUTER_METRIC_KEYS: frozenset[str]   # all keys compute_router_metrics() produces
TRAINING_LEVEL_EXCLUDE_KEY: str      # = "aux_loss"
```

The filter in `run_laptop_moe.py` is now:
```python
{k: v for k, v in m.items()
 if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY}
```

**Proof (T05):** The test explicitly shows the old prefix filter removes
`auxiliary_load_balancing_loss` and the new allowlist retains it.

---

## Repair 2 Detail: Complete MoE Artifact Schema

**Root cause:** `moe_summary.json` was missing `schema_version`, `timestamp`, `branch`,
`commit`, `device`, `gpu_name`. The pipeline's `_validate_runner_artifact` requires
`schema_version` and `timestamp` to be present and well-formed.

**Fix:** The initial `summary` dict in `run_moe_validation()` now includes:
```python
"schema_version": "1.0",
"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
"run_id": _run_id,
"branch": branch,
"commit": commit,
"device": "cuda",
"gpu_name": torch.cuda.get_device_name(0),
```

`validate_moe_artifact_schema(artifact)` returns a list of error strings. An empty list means
the artifact is schema-valid. Called before every `_write_artifact()` invocation.

**Required fields for schema validity:**
`schema_version`, `timestamp`, `run_id`, `branch`, `commit`, `outcome`, `exit_code`,
`data_mode`, `device`, `gpu_name`, `aux_loss_semantics`, `router_metrics_available_under_gc`,
`checkpoint_status`

---

## Repair 3 Detail: Independent High-Frequency Thermal Safety Monitor

**Root cause:** `_check_thermal()` was called once per training step. On the RTX 5090 with
long steps (up to 60 s), a GPU could reach 90°C and sustain it for nearly a minute before
the next check. The monitor was not independent of training-step timing.

**Fix:** `training/thermal_monitor.py` implements `ThermalMonitor`:

```python
from training.thermal_monitor import ThermalMonitor

monitor = ThermalMonitor(run_dir=output_dir, run_id=run_id)
monitor.start()   # background thread starts, 1 Hz sampling
try:
    for step in range(1, max_steps + 1):
        if monitor.stop_requested:     # non-blocking poll
            # SAFETY_STOP path
            break
        if monitor.monitor_failed:
            # Non-PASS: monitor health required
            break
        ... training step ...
finally:
    monitor.stop()
    health = monitor.health_summary()
    # health["healthy"] == True iff monitor ran without failure
```

**Key properties:**
- Samples at 1 Hz regardless of step timing
- Writes `thermal_telemetry.jsonl` continuously (one JSON record per line)
- Emits `[THERMAL WARNING]` at >= 80°C
- Sets `stop_requested` event at >= 90°C
- Sets `monitor_failed` event if the thread dies unexpectedly
- Singleton guard: only one active monitor per process
- `stop()` joins the thread and clears the singleton
- `health_summary()` returns `healthy`, `peak_temperature_c`, `samples_at_or_above_warn_c`,
  `safety_stop_triggered`, `total_samples`

**PASS prevention:** If `monitor_failed` is True when `outcome == PASS`, the outcome is
overridden to `EXECUTION_ERROR`. A healthy monitor is required for PASS.

---

## Test Traceability Table

| Test ID | Class | What It Proves |
|---|---|---|
| T01 | `TestMetricFilterAllowlist` | `auxiliary_load_balancing_loss` survives the allowlist filter |
| T02 | `TestMetricFilterAllowlist` | Only exact key `aux_loss` is excluded; `auxiliary_load_balancing_loss` is not |
| T03 | `TestMetricFilterAllowlist` | All 11 required router metrics reach `aggregate_layer_metrics` |
| T04 | `TestMetricFilterAllowlist` | All 11 required router metrics appear in a fresh MoE artifact; evaluator does not return `NOT_EVALUABLE` |
| T05 | `TestMetricFilterAllowlist` | **Regression proof**: old prefix filter removed `auxiliary_load_balancing_loss`; new allowlist retains it |
| T06 | `TestMoEArtifactSchemaValidation` | Complete schema-valid artifact produces zero errors |
| T07 | `TestMoEArtifactSchemaValidation` | Missing `schema_version` produces validation error |
| T08 | `TestMoEArtifactSchemaValidation` | Missing `timestamp` produces validation error |
| T09 | `TestMoEArtifactSchemaValidation` | Naive (timezone-unaware) timestamp produces validation error |
| T10 | `TestMoEArtifactSchemaValidation` | `schema_version != '1.0'` produces validation error |
| T11 | `TestThermalMonitor` | 79°C: no warning, no stop |
| T12 | `TestThermalMonitor` | 80°C: warning triggered, no stop |
| T13 | `TestThermalMonitor` | 85°C: warning triggered, training may continue |
| T14 | `TestThermalMonitor` | 89°C: warning triggered, no stop |
| T15 | `TestThermalMonitor` | 90°C: SAFETY_STOP triggered |
| T16 | `TestThermalMonitor` | 91°C: SAFETY_STOP triggered |
| T17 | `TestThermalMonitor` | Monitor thread failure sets `monitor_failed=True`, `healthy=False` |
| T18 | `TestThermalMonitor` | After `stop()`, thread is dead and singleton is cleared |

---

## Test Results

| Suite | Collected Nodes | Result |
|---|---|---|
| Run 12–14 (existing, 7 files) | 151 | 151 passed |
| **Run 15 regression (new, T01–T18)** | **18** | **18 passed** |
| **Targeted total (8 files)** | **169** | **169 passed** |
| Full suite | 788 | 11 failed (pre-existing), 761 passed, 16 skipped |

**Zero new failures introduced by Run 15.**

Pre-existing failures (11): `test_mesh.py` (6), `test_models.py` (1),
`test_tokenizer_vocab.py` (2) — all unrelated to the targeted suite.

---

## Kishore's Preflight Commands

```bat
cd C:\Users\Kishore\jupiter-shot
git fetch origin
git checkout fix/rtx50-blackwell-validation
git pull --ff-only origin fix/rtx50-blackwell-validation
git log -1
REM Expected: commit hash ending in f27c173 or later

REM Step 1: Verify collected-node count
.venv\Scripts\python.exe -m pytest --collect-only -q ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  tests\test_run14_same_pass_provenance.py ^
  tests\test_run14_integration_contracts.py ^
  tests\test_run15_regression.py
REM Expected: 169 tests collected

REM Step 2: Run targeted suite
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
REM Expected: 169 passed, 0 failed, exit code 0
```

---

## Stop Conditions

| Condition | Action |
|---|---|
| Fewer than 169 nodes collected | STOP — manifest mismatch, do not proceed |
| Any targeted test fails | STOP — do not run the validation pipeline |
| `moe_summary.json` missing `schema_version` or `timestamp` | STOP — schema validation failed |
| `thermal_monitor_healthy: false` in artifact | STOP — monitor failure, not a valid run |
| `thermal_stop_triggered: true` in artifact | STOP — GPU overtemperature during run |
| `outcome != "PASS"` in any runner artifact | STOP — runner did not pass |
| `exit_code != 0` from pipeline | STOP — pipeline reported failure |

---

## Artifact Verification

After a successful validation run, verify `moe_summary.json`:

```bat
REM Required fields
python -c "
import json, sys
a = json.load(open('results/moe_summary.json'))
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
"
```

Expected output:
```
Schema OK
outcome: PASS
thermal_monitor_healthy: True
thermal_peak_c: <temperature below 90>
```
