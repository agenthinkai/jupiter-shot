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
| 11 | Test-Suite Result | ✓ PASS | collected=252 passed=248 failed=0 errors=0 skipped=4 exit_code=0 |
| 12 | Human-Review Status | ⏳ NOT_READY | Human review pending for 101 records. No software approval detected. Status: PENDING INDEPENDENT AUDIT AND HUMAN REVIEW |
| 13 | Corpus Integrity | ✓ PASS | Corpus intact: 101 records, ordered_sha256=123fbdaf47a1e6be..., content_commit=2e36f6b977a8... |
| 14 | Content-Risk | ✗ REVIEW_REQUIRED | 0 BLOCKED, 3 REVIEW_REQUIRED finding(s). Records must appear in human-review queue before training/release authorization. |
| 15 | Review-Queue Coverage | ✓ PASS | All 3 REVIEW_REQUIRED record(s) present in queue. No reviewer judgment fields populated. |

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

collected=252 passed=248 failed=0 errors=0 skipped=4 exit_code=0

**Metadata:**
- `interpreter`: `/home/ubuntu/jupiter-shot/.venv_v23/bin/python`
- `python_version`: `3.12.3`
- `pytest_version`: `pytest 8.3.5`
- `test_manifest_hash`: `f3578c32bbae49102f722a180f713296fb15934eeae5c1dea75a650f4152d203`
- `authorized_files`: `['test_adversarial_fixtures.py', 'test_gate_adversarial.py', 'test_repair_audit.py', 'test_integrity_audit.py', 'test_gold_dataset.py', 'test_diagnostic_pipeline.py', 'test_foundation_audit.py', 'test_v22_regression.py', 'test_v23_regex.py', 'test_v23_regression.py']`
- `found_files`: `['test_adversarial_fixtures.py', 'test_gate_adversarial.py', 'test_repair_audit.py', 'test_integrity_audit.py', 'test_gold_dataset.py', 'test_diagnostic_pipeline.py', 'test_foundation_audit.py', 'test_v22_regression.py', 'test_v23_regex.py', 'test_v23_regression.py']`
- `collected`: `252`
- `passed`: `248`
- `failed`: `0`
- `errors`: `0`
- `skipped`: `4`
- `exit_code`: `0`
- `start_ts`: `2026-08-08T12:39:19.700540Z`
- `end_ts`: `2026-08-08T12:39:20.967439Z`

### Gate 12: Human-Review Status — NOT_READY

Human review pending for 101 records. No software approval detected. Status: PENDING INDEPENDENT AUDIT AND HUMAN REVIEW

### Gate 13: Corpus Integrity — PASS

Corpus intact: 101 records, ordered_sha256=123fbdaf47a1e6be..., content_commit=2e36f6b977a8...

**Metadata:**
- `ordered_corpus_sha256`: `123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0`
- `split_assignment_sha256`: `fd150b1799475d455e4c0935b015fc671fb542cc906732df8cb0b81d21c42451`
- `family_assignment_sha256`: `a5ad9cff5bdb981a0996fb3f08d8402c98800867cd43034ca318d7094327bbe7`
- `dataset_content_commit`: `2e36f6b977a8af052fced5a532c1168dc1988b6f`
- `record_count`: `101`
- `split_counts`: `{'train': 68, 'valid': 16, 'eval': 17}`

### Gate 14: Content-Risk — REVIEW_REQUIRED

0 BLOCKED, 3 REVIEW_REQUIRED finding(s). Records must appear in human-review queue before training/release authorization.

**Metadata:**
- `blocked_count`: `0`
- `review_required_count`: `3`
- `review_required_records`: `['seed4b-train-0032', 'seed4b-valid-0005', 'seed4b-train-0038']`
- `findings`: `[{'example_id': 'seed4b-train-0032', 'field': 'response', 'rule': "'specific current requirements should be verified[^\\\\.]{0,60}\\\\.'", 'excerpt': "ation employment commitments. Specific current requirements should be verified against CITRA's published regulations.", 'severity': 'REVIEW_REQUIRED'}, {'example_id': 'seed4b-train-0038', 'field': 'response', 'rule': "'يُنصح بمراجعة[^\\\\.]{0,50}\\\\.'", 'excerpt': 'وني وحماية بيانات المستهلكين. يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على أحدث اللوائح.', 'severity': 'REVIEW_REQUIRED'}, {'example_id': 'seed4b-valid-0005', 'field': 'response', 'rule': "'يُنصح بمراجعة[^\\\\.]{0,50}\\\\.'", 'excerpt': 'لا يجوز للمشغّل الاستئثار به. يُنصح بمراجعة المعيار الكامل والهيئة الشرعية للتطبيق الفعلي.', 'severity': 'REVIEW_REQUIRED'}]`

**Warnings:**
- REVIEW_REQUIRED | seed4b-train-0032 [response] | 'specific current requirements should be | "ation employment commitments. Specific current requirements should be verified against CITRA's published regulations."
- REVIEW_REQUIRED | seed4b-train-0038 [response] | 'يُنصح بمراجعة[^\\.]{0,50}\\.' | 'وني وحماية بيانات المستهلكين. يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على أحدث اللوائح.'
- REVIEW_REQUIRED | seed4b-valid-0005 [response] | 'يُنصح بمراجعة[^\\.]{0,50}\\.' | 'لا يجوز للمشغّل الاستئثار به. يُنصح بمراجعة المعيار الكامل والهيئة الشرعية للتطبيق الفعلي.'

### Gate 15: Review-Queue Coverage — PASS

All 3 REVIEW_REQUIRED record(s) present in queue. No reviewer judgment fields populated.

**Metadata:**
- `review_required_ids`: `['seed4b-train-0032', 'seed4b-train-0038', 'seed4b-valid-0005']`
- `queue_coverage`: `COMPLETE`

---

## Final Verdict

**MECHANICALLY_BLOCKED — 3 REVIEW_REQUIRED RECORD(S) PENDING HUMAN JUDGMENT: ['seed4b-train-0032', 'seed4b-train-0038', 'seed4b-valid-0005']**

AI-ASSISTED INTERNAL CONTENT AUDIT — NOT HUMAN APPROVAL
