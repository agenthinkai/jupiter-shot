# Jupiter Shot — Test Manifest
## Run 15 Corrected Package
## Branch: `fix/rtx50-blackwell-validation` | Commit: `dc79637`

---

## Authoritative Manifest Files

The authoritative node-ID manifest is machine-generated and machine-verified.
Do not edit these files manually.

| File | Purpose |
|---|---|
| `tests/run15_authorized_nodes.txt` | Sorted list of 174 authorized node IDs |
| `tests/run15_authorized_manifest.json` | JSON manifest with SHA-256, counts, branch, commit |

**Authorized SHA-256:** `43142de49d415c5bd8e606aeac51cccec6976eb53c3f2500752b38f0e604c0a7`

To verify: `python3 scripts/verify_test_manifest.py` (exit 0 = exact match)

---

## Per-File Node Breakdown

| File | Functions | Collected Nodes | Notes |
|---|---|---|---|
| `test_run12_gate10b_real_object.py` | 26 | 26 | No parametrized expansion |
| `test_run12_operator_package.py` | 25 | 25 | No parametrized expansion |
| `test_run13_config_resolver.py` | 24 | **29** | `test_no_double_yaml_suffix`: 3 variants; `TestAllFourInputForms`: 4 variants → 24−2+7=29 |
| `test_run13_subprocess_smoke.py` | 31 | 31 | No parametrized expansion |
| `test_run14_integration.py` | 15 | 15 | No parametrized expansion |
| `test_run14_same_pass_provenance.py` | 2 | 2 | No parametrized expansion |
| `test_run14_integration_contracts.py` | 28 | 28 | No parametrized expansion |
| `test_run15_regression.py` | 18 | 18 | No parametrized expansion |
| **Targeted total (8 files)** | **169** | **174** | |

---

## Verifier Tests

| File | Functions | Collected Nodes | Notes |
|---|---|---|---|
| `test_run15_manifest_verifier.py` | 10 | 10 | V01–V10: verifier regression tests |

The verifier test file is **not** part of the 174-node authorized suite. It tests the
verifier itself and is run separately.

---

## 112→111→169→174 Count History

| Package | Claimed | Actual | Root Cause |
|---|---|---|---|
| Run 13 initial | 112 | 111 | Transcription error (60+27+25=112 vs actual 31+24+26+25=106 functions; pytest counts 111 nodes due to 5 parametrized expansions in `test_run13_config_resolver.py`) |
| Run 14 initial | 106 | 111 | Manifest used `grep -c "def test_"` (function count) instead of `pytest --collect-only` (node count) |
| Run 15 initial | 169 | 174 | Same error: function count (169) reported instead of collected-node count (174); `test_run13_config_resolver.py` has 24 functions but 29 collected nodes |
| **Run 15 corrected** | **174** | **174** | Machine-generated manifest; automated verifier eliminates manual counting |

---

## Full Suite Baseline (pre-existing failures)

| Category | Count |
|---|---|
| Pre-existing failures | 11 |
| Pre-existing failures (files) | `test_mesh.py` (6), `test_models.py` (1), `test_tokenizer_vocab.py` (2), other (2) |
| Passed | 771 |
| Skipped | 16 |
| Total collected | 798 |

No new failures were introduced by Run 12, 13, 14, or 15.
