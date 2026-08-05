# Jupiter Seed 4B: Dataset Diversity Audit

**Label:** AI-ASSISTED INTERNAL DIVERSITY AUDIT — NOT HUMAN APPROVAL

This audit uses deterministic, CPU-compatible methods only. No embedding models, paid APIs, or GPU were used.

---

## 1. Uniqueness Summary

| Metric | Value |
| :--- | :--- |
| Total Records | 850 |
| Unique Normalized Prompts | 850 |
| Unique Normalized Responses | 80 |
| Unique Prompt Skeletons | 160 |
| Unique Response Skeletons | 80 |
| Duplicate Prompt Count | 0 |
| Duplicate Response Count | 770 |

## 2. Template Family Analysis

| Metric | Value |
| :--- | :--- |
| Total Families | 160 |
| Max Family Size | 9 |
| Max Family % | 1.06% |
| Families Exceeding 2% | 0 |
| **Template Family Check** | **PASS** |

## 3. N-gram Overlap (3-gram Jaccard, sample)

| Metric | Value |
| :--- | :--- |
| Sample Size | 200 |
| Average Jaccard Similarity | 0.0078 |
| High-Similarity Pairs (≥50%) | 34 |

## 4. Near-Duplicate Detection

| Metric | Value |
| :--- | :--- |
| Sample Size | 300 |
| Near-Duplicate Pairs (≥0.7) | 31 |

## 5. TF-IDF Similarity Clusters (sample)

| Metric | Value |
| :--- | :--- |
| Sample Size | 150 |
| High-Similarity Pairs (≥0.8) | 24 |

## 6. Opening/Closing Phrase Repetition

| Max Opening Repetition | 53 |
| :--- | :--- |
| Max Closing Repetition | 103 |

**Top 5 Opening Phrases:**

- `i do not have access` — 53 occurrences
- `يمكن معالجة هذا الاعتراض من` — 13 occurrences
- `تخضع المشتريات الحكومية في الكويت` — 13 occurrences
- `option investment expected roi timeline` — 13 occurrences
- `to provide a meaningful analysis` — 13 occurrences

## 7. Per-Domain Diversity

| Domain | Count | Unique Skeletons | Diversity Ratio |
| :--- | :--- | :--- | :--- |
| arabic_english_correspondence | 83 | 18 | 0.217 |
| energy_logistics | 85 | 16 | 0.188 |
| executive_decision | 128 | 20 | 0.156 |
| gcc_banking | 128 | 22 | 0.172 |
| government_regulation | 128 | 20 | 0.156 |
| islamic_finance | 170 | 40 | 0.235 |
| telecommunications | 128 | 24 | 0.188 |

## 8. Per-Language Diversity

| Language | Count | Unique Skeletons | Diversity Ratio |
| :--- | :--- | :--- | :--- |
| ar | 383 | 143 | 0.373 |
| ar-en | 212 | 116 | 0.547 |
| en | 255 | 127 | 0.498 |

## 9. Per-Task-Type Diversity

| Task Type | Count | Unique Skeletons | Diversity Ratio |
| :--- | :--- | :--- | :--- |
| arabic_english_translation | 61 | 59 | 0.967 |
| decision_comparison | 67 | 61 | 0.91 |
| document_summarization | 67 | 61 | 0.91 |
| executive_briefing | 67 | 61 | 0.91 |
| formal_correspondence | 61 | 59 | 0.967 |
| hallucination_resistance | 53 | 51 | 0.962 |
| islamic_finance_terminology | 58 | 56 | 0.966 |
| objection_handling | 65 | 60 | 0.923 |
| question_answering | 67 | 61 | 0.91 |
| refusal_missing_info | 53 | 51 | 0.962 |
| regulatory_explanation | 59 | 57 | 0.966 |
| request_for_clarification | 53 | 51 | 0.962 |
| risk_identification | 65 | 60 | 0.923 |
| structured_extraction | 54 | 52 | 0.963 |

## 10. Verdict

| Check | Result |
| :--- | :--- |
| No template family exceeds 2% | PASS |
| Unique prompt skeletons > 50% of total | REVISE |
| Average Jaccard similarity < 0.3 | PASS |

**AI-ASSISTED INTERNAL DIVERSITY AUDIT — NOT HUMAN APPROVAL**
