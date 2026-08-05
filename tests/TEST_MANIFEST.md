# Jupiter Shot — Required Test Node Manifest

This file is the authoritative record of test counts for the targeted validation
test suite. All operator packages and preflight gates MUST reference this file
rather than hardcoding totals.

**Authorization uses pytest collected-node counts, not function counts.**

---

## Terminology

| Term | Definition |
|---|---|
| **Test functions** | Python functions named `test_*` (`grep -c "def test_"`) |
| **Collected nodes** | Actual pytest node IDs (`pytest --collect-only -q`). Parametrized tests expand into multiple nodes. This is the authoritative count. |

---

## Targeted Validation Test Files

| File | Test Functions | Collected Nodes | Added in Run |
|---|---:|---:|---|
| `tests/test_run12_gate10b_real_object.py` | 26 | 26 | Run 12 |
| `tests/test_run12_operator_package.py` | 25 | 25 | Run 12 |
| `tests/test_run13_config_resolver.py` | 24 | 24 | Run 13 |
| `tests/test_run13_subprocess_smoke.py` | 31 | 31 | Run 13 |
| `tests/test_run14_integration.py` | 15 | 15 | Run 14 |
| `tests/test_run14_same_pass_provenance.py` | 2 | 2 | Run 14 |
| `tests/test_run14_integration_contracts.py` | 12 | 28 | Run 14 |
| `tests/test_run15_regression.py` | 18 | 18 | Run 15 |
| **TOTAL (Run 12–15 targeted)** | **153** | **169** | |

> **Note on test_run14_integration_contracts.py:** 12 test functions expand to 28 collected
> nodes because several tests use `@pytest.mark.parametrize`. The collected count (28) is
> authoritative.

---

## Notes

- **112→111 discrepancy explained:** The `docs/RUN13_OPERATOR_PACKAGE.md` originally claimed
  112 tests (60+27+25 — a transcription error). Actual `grep -c "def test_"` counts are
  31+24+26+25 = **106**. `pytest --collect-only` returns **111** because 5 parametrized
  variants in `test_run13_subprocess_smoke.py` expand into individual collected items.

- **Run 14 adds 45 new test functions:** 15 + 2 + 28 = 45. Collected nodes: 15 + 2 + 28 = 45.

- **Run 15 adds 18 new test functions / 18 collected nodes.**

- **Run 15 targeted total: 169 collected nodes** (verified by `pytest --collect-only`).

- Pre-existing failures (11) are in `test_mesh.py`, `test_models.py`,
  `test_tokenizer_vocab.py` — unrelated to the targeted suite, present since before Run 12.

---

## Gate Requirements

Kishore must verify before authorizing a Run 15 PASS:

```bat
.venv\Scripts\python.exe -m pytest --collect-only -q ^
  tests\test_run12_gate10b_real_object.py ^
  tests\test_run12_operator_package.py ^
  tests\test_run13_config_resolver.py ^
  tests\test_run13_subprocess_smoke.py ^
  tests\test_run14_integration.py ^
  tests\test_run14_same_pass_provenance.py ^
  tests\test_run14_integration_contracts.py ^
  tests\test_run15_regression.py
```

Expected: **169 tests collected**

Then run:

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

Required result:
- 169 passed
- 0 failed
- 0 errors
- 0 skips
- Pytest exit code 0

---

## Full Suite Baseline (Run 15)

| Category | Count |
|---|---|
| Passed | 761 |
| Failed (pre-existing) | 11 |
| Skipped | 16 |
