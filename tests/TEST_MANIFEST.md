# Jupiter Shot — Test Manifest

This file is the authoritative record of test counts for the targeted validation
test suite.  All operator packages and preflight gates MUST reference this file
rather than hardcoding totals.

## Targeted Validation Test Files

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

## Notes

- **112→111 discrepancy explained:** The `docs/RUN13_OPERATOR_PACKAGE.md` and
  `docs/KISHORE_GPU_OPERATOR_CHECKLIST.md` originally claimed 112 tests
  (stated as 60+27+25).  The actual `grep -c "def test_"` counts are
  31+24+26+25 = **106**.  The 112 figure was a transcription error in the
  operator package — it was never the real pytest-collected count.
  `pytest --collect-only` returns **111** (106 function-level tests plus 5
  parametrized variants in `test_run13_subprocess_smoke.py` that pytest counts
  as individual collected items).  This manifest uses function-level counts
  (grep) for consistency.

- **Run 12–13 targeted total: 106** (verified by `grep -c "def test_"`).

- **Run 14 adds 45 new tests:** 15 (integration) + 2 (same-pass provenance) +
  28 (integration contracts) = 45.  New targeted total: **151**.

- Pre-existing failures (11) are in `test_mesh.py`, `test_models.py`,
  `test_tokenizer_vocab.py` — these are unrelated to the targeted suite and
  have been present since before Run 12.

- Counts are verified by `grep -c "def test_"` on each file.

## Verification Command

```bash
# Verify counts match this manifest
for f in tests/test_run12_gate10b_real_object.py \
         tests/test_run12_operator_package.py \
         tests/test_run13_config_resolver.py \
         tests/test_run13_subprocess_smoke.py \
         tests/test_run14_integration.py \
         tests/test_run14_same_pass_provenance.py \
         tests/test_run14_integration_contracts.py; do
  echo "$f: $(grep -c 'def test_' $f)"
done
# Expected: 26, 25, 24, 31, 15, 2, 28  (total 151)
```
