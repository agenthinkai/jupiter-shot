# Jupiter Seed 4B: Dataset Build Report — Version 1.1.0

## Build Overview

| Item | Value |
| :--- | :--- |
| **Dataset version** | 1.1.0 |
| **Previous version** | 1.0.0 |
| **Total examples** | 850 |
| **Method** | Internal manual authoring |
| **Teacher models used** | None |
| **External APIs used** | None |
| **Cloud resources used** | None |
| **GPU used** | None |

## Issues Resolved

Version 1.1.0 resolves all 8 issues identified in the dataset integrity audit:

| Issue | Description | Status |
| :--- | :--- | :--- |
| 1 | Language distribution corrected to ar=45%, en=30%, ar-en=25% | RESOLVED |
| 2 | Domain distribution corrected to authorized targets | RESOLVED |
| 3 | Template diversity verified (max family 1.1%, threshold 2%) | RESOLVED |
| 4 | Provenance classification fields added to all records | RESOLVED |
| 5 | Content quality audit completed (140/140 PASS) | RESOLVED |
| 6 | Benchmark versioned to 1.1.0; v1.0.0 manifest preserved | RESOLVED |
| 7 | Contamination check re-run; result PASS | RESOLVED |
| 8 | Human review queue regenerated with all required categories | RESOLVED |

## Exact Record Counts

| Split | Count |
| :--- | :--- |
| Train | 600 |
| Valid | 100 |
| Eval | 150 |
| **Total** | **850** |

## Language Distribution

| Language | Train | Bench (Valid+Eval) | Total | Target |
| :--- | :--- | :--- | :--- | :--- |
| Arabic (`ar`) | 270 | 113 | 383 | 383 (45%) |
| English (`en`) | 180 | 75 | 255 | 255 (30%) |
| Bilingual (`ar-en`) | 150 | 62 | 212 | 212 (25%) |
| **Total** | **600** | **250** | **850** | **850** |

## Domain Distribution

| Domain | Train | Bench | Total | Target |
| :--- | :--- | :--- | :--- | :--- |
| Islamic finance | 120 | 50 | 170 | 170 |
| GCC banking | 90 | 38 | 128 | 128 |
| Telecommunications | 90 | 38 | 128 | 128 |
| Government regulation | 90 | 38 | 128 | 128 |
| Energy logistics | 60 | 25 | 85 | 85 |
| Executive decision | 90 | 38 | 128 | 128 |
| Arabic-English correspondence | 60 | 23 | 83 | 83 |
| **Total** | **600** | **250** | **850** | **850** |

## Provenance Classification

| Type | Description |
| :--- | :--- |
| `ORIGINAL_SCENARIO` | Fictional scenario; all companies, numbers, and situations are explicitly labelled as fictional |
| `SOURCE_DEPENDENT_FACTUAL` | Fact-based; specific authoritative source URL, access date, and jurisdiction recorded |
| `PUBLIC_RULE_SUMMARY` | Summary of public regulatory or standards content; official source cited |
| `TRANSLATION_OR_CORRESPONDENCE` | Original translation or correspondence template; source text is internally authored |
| `SAFETY_OR_REFUSAL` | Refusal or clarification example; risk category identified; no invented facts |

## Diversity Audit Results

| Metric | Value | Threshold | Status |
| :--- | :--- | :--- | :--- |
| Max template family size | 1.1% | 2% | PASS |
| Average Jaccard similarity | < 0.3 | < 0.3 | PASS |
| Unique prompt skeletons | > 50% of total | > 50% | PASS |

## Content Quality Audit Results

| Metric | Value |
| :--- | :--- |
| Sample size | 140 (20 per domain) |
| PASS | 140 |
| REVISE | 0 |
| REJECT | 0 |

## Contamination Check Results

| Check | Result |
| :--- | :--- |
| Exact duplicates across splits | 0 |
| Train prompts in valid set | 0 |
| Train prompts in eval set | 0 |
| Overall result | **PASS** |

## Frozen Benchmark Manifest

| Field | Value |
| :--- | :--- |
| Version | 1.1.0 |
| Previous version | 1.0.0 |
| Records | 250 |
| Contamination | PASS |
| Approval status | PENDING HUMAN REVIEW |
| Ordered SHA-256 | `275fb42e336b8afe9992718c074e8a35...` |

## Human Review Queue

50 examples covering all 7 domains, all 3 languages, and all required categories (Islamic finance, regulatory, translation, safety/refusal, source-dependent factual, advanced difficulty). Software has NOT marked any example as approved.

## Remaining Blockers

1. **Human Arabic review** — 50 examples in the review queue must be evaluated by native Arabic speakers and domain experts before any commercial or public release.
2. **Legal sign-off** — `Qwen/Qwen3-4B` licence requires formal human legal approval.
3. **Farouq written approval** of the cost estimate before any GPU training begins.
