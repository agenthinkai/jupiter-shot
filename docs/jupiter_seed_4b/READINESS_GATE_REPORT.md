# Jupiter Seed 4B: Dataset Readiness Gate Report

**Dataset version:** 2.0.0
**Approval status:** PENDING INDEPENDENT AUDIT AND HUMAN REVIEW

## Honest Dataset Size

| Metric | Value |
| :--- | :--- |
| Authorized target | 850 |
| Actual distinct examples | 101 |
| Shortfall | 749 |

## Gate Results

| Gate | Name | Status | Details |
| :--- | :--- | :--- | :--- |
| 1 | Schema Integrity | ✓ PASS | All 101 records have required fields |
| 2 | Provenance Integrity | ✓ PASS | Provenance fields present. Warnings: 0 |
| 3 | UTF-8 Portability | ✓ PASS | All text I/O uses explicit encoding in training/ and tests/. Arabic bytes fail cp1252 as expected. |
| 4 | Language Contract | ✓ PASS | All 101 records satisfy independently recomputed language contracts |
| 5 | Raw Duplication | ✓ PASS | No raw cross-split duplicates detected |
| 6 | Canonical Duplication | ✓ PASS | No canonical cross-split duplicates, no duplicate scenario briefs, no artificial markers |
| 7 | Semantic Leakage | ✓ PASS | No blocking semantic leakage. Warnings: 0 |
| 8 | Content-Family Split Isolation | ✓ PASS | All content families are assigned to exactly one split |
| 9 | Review-Representation Identity | ✓ PASS | JSONL, Markdown, and CSV all contain the same 50 IDs. All reviewer judgment fields are empty. |
| 10 | Benchmark Manifest Integrity | ✓ PASS | Manifest version 2.0.0, approval PENDING. |
| 11 | Test-Suite Result | ✓ PASS | collected=141 passed=137 failed=0 errors=0 skipped=4 exit_code=0 |
| 12 | Human-Review Status | ⏳ NOT_READY | Human review pending for 101 records. No software approval detected. Status: PENDING INDEPENDENT AUDIT AND HUMAN REVIEW |

## Detailed Gate Results

### Gate 1: Schema Integrity — PASS

All 101 records have required fields

### Gate 2: Provenance Integrity — PASS

Provenance fields present. Warnings: 0

### Gate 3: UTF-8 Portability — PASS

All text I/O uses explicit encoding in training/ and tests/. Arabic bytes fail cp1252 as expected.

### Gate 4: Language Contract — PASS

All 101 records satisfy independently recomputed language contracts

### Gate 5: Raw Duplication — PASS

No raw cross-split duplicates detected

### Gate 6: Canonical Duplication — PASS

No canonical cross-split duplicates, no duplicate scenario briefs, no artificial markers

### Gate 7: Semantic Leakage — PASS

No blocking semantic leakage. Warnings: 0

### Gate 8: Content-Family Split Isolation — PASS

All content families are assigned to exactly one split

### Gate 9: Review-Representation Identity — PASS

JSONL, Markdown, and CSV all contain the same 50 IDs. All reviewer judgment fields are empty.

### Gate 10: Benchmark Manifest Integrity — PASS

Manifest version 2.0.0, approval PENDING.

### Gate 11: Test-Suite Result — PASS

collected=141 passed=137 failed=0 errors=0 skipped=4 exit_code=0

**Metadata:**
- `interpreter`: `/home/ubuntu/jupiter-shot/.venv/bin/python`
- `python_version`: `3.12.3`
- `pytest_version`: `pytest 8.3.5`
- `test_manifest_hash`: `3df9a6d7be1a49fbb9338e6c8f640ae51fae9d7d9d08480816154ce29f4c9838`
- `authorized_files`: `['test_adversarial_fixtures.py', 'test_repair_audit.py', 'test_integrity_audit.py', 'test_gold_dataset.py', 'test_diagnostic_pipeline.py', 'test_foundation_audit.py']`
- `found_files`: `['test_adversarial_fixtures.py', 'test_repair_audit.py', 'test_integrity_audit.py', 'test_gold_dataset.py', 'test_diagnostic_pipeline.py', 'test_foundation_audit.py']`
- `collected`: `141`
- `passed`: `137`
- `failed`: `0`
- `errors`: `0`
- `skipped`: `4`
- `exit_code`: `0`
- `start_ts`: `2026-08-06T19:15:11.306060Z`
- `end_ts`: `2026-08-06T19:15:12.573547Z`

### Gate 12: Human-Review Status — NOT_READY

Human review pending for 101 records. No software approval detected. Status: PENDING INDEPENDENT AUDIT AND HUMAN REVIEW

---

## Final Verdict

**MECHANICALLY READY FOR INDEPENDENT AUDIT — HUMAN REVIEW NOT YET AUTHORIZED**

AI-ASSISTED INTERNAL CONTENT AUDIT — NOT HUMAN APPROVAL
