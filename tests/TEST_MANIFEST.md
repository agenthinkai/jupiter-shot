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
| **Total (Run 12–14 targeted)** | **121** | |

## Notes

- The total was incorrectly stated as 112 in `docs/RUN13_OPERATOR_PACKAGE.md`
  and `docs/KISHORE_GPU_OPERATOR_CHECKLIST.md`.  The correct Run 12–13 total
  is **106**.  With the 15 Run 14 tests added, the new total is **121**.
- Pre-existing failures (11) are in `test_mesh.py`, `test_models.py`,
  `test_tokenizer_vocab.py` — these are unrelated to the targeted suite.
- Counts are verified by `grep -c "def test_"` on each file.

## Verification Command

```bash
# Verify counts match this manifest
for f in tests/test_run12_gate10b_real_object.py \
         tests/test_run12_operator_package.py \
         tests/test_run13_config_resolver.py \
         tests/test_run13_subprocess_smoke.py \
         tests/test_run14_integration.py; do
  echo "$f: $(grep -c 'def test_' $f)"
done
```
