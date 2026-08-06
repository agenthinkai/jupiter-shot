# Jupiter Seed 4B: Content Quality Audit

**Label:** AI-ASSISTED INTERNAL CONTENT AUDIT — NOT HUMAN APPROVAL

**Defect 6 fix:** All fallback PASS logic removed. PASS requires explicit satisfaction of every mandatory condition.

## 1. Audit Summary

- **Total Sampled:** 86
- **PASS:** 84
- **REVISE:** 2
- **REJECT:** 0
- **BLOCKED/NOT_EVALUABLE:** 0

**Verdict:** All REVISE, REJECT, and BLOCKED decisions must be resolved before the dataset is declared ready.

## 2. Detailed Record Scores

| Example ID | Domain | Lang | Task | Verdict | Factual | GCC | Halluc. | Structural |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `seed4b-train-0012` | islamic_finance | ar | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0005` | islamic_finance | ar | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0020` | islamic_finance | ar | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0022` | islamic_finance | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0016` | islamic_finance | ar | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0024` | islamic_finance | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0004` | islamic_finance | en | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0001` | islamic_finance | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0014` | islamic_finance | en | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0011` | islamic_finance | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0006` | islamic_finance | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0018` | islamic_finance | en | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0009` | islamic_finance | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0006` | islamic_finance | ar-en | structured_extraction | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0010` | islamic_finance | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0002` | islamic_finance | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0008` | islamic_finance | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0023` | islamic_finance | ar-en | structured_extraction | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0019` | islamic_finance | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0005` | islamic_finance | ar | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0026` | gcc_banking | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0011` | gcc_banking | ar | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0031` | gcc_banking | ar | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0014` | gcc_banking | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0028` | gcc_banking | ar | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0030` | gcc_banking | ar | structured_extraction | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0013` | gcc_banking | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0010` | gcc_banking | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0025` | gcc_banking | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0027` | gcc_banking | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0029` | gcc_banking | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0012` | gcc_banking | ar-en | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0038` | telecommunications | ar | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0033` | telecommunications | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0035` | telecommunications | ar | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0000` | telecommunications | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0034` | telecommunications | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0037` | telecommunications | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0002` | telecommunications | en | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-valid-0015` | telecommunications | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0032` | telecommunications | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0036` | telecommunications | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0001` | telecommunications | ar-en | formal_correspondence | **REVISE** | PASS | PASS | PASS | PASS |
| `seed4b-train-0041` | government_regulation | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0043` | government_regulation | ar | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0039` | government_regulation | ar | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0004` | government_regulation | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0003` | government_regulation | en | regulatory_explanation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0040` | government_regulation | en | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0045` | government_regulation | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0042` | government_regulation | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0006` | government_regulation | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0044` | government_regulation | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0005` | government_regulation | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0050` | energy_logistics | ar | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0008` | energy_logistics | ar | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0047` | energy_logistics | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0049` | energy_logistics | ar | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0046` | energy_logistics | en | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0051` | energy_logistics | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0007` | energy_logistics | en | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0048` | energy_logistics | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0009` | energy_logistics | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0011` | executive_decision | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0059` | executive_decision | ar | question_answering | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0053` | executive_decision | ar | executive_briefing | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0055` | executive_decision | ar | objection_handling | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0054` | executive_decision | en | risk_identification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0052` | executive_decision | en | decision_comparison | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0058` | executive_decision | en | hallucination_resistance | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0010` | executive_decision | en | decision_comparison | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0056` | executive_decision | en | request_for_clarification | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0012` | executive_decision | ar-en | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0057` | executive_decision | ar-en | structured_extraction | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0060` | arabic_english_correspondence | ar | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0016` | arabic_english_correspondence | ar | refusal_missing_info | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0063` | arabic_english_correspondence | ar | document_summarization | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0013` | arabic_english_correspondence | ar | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0066` | arabic_english_correspondence | ar | refusal_missing_info | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0014` | arabic_english_correspondence | en | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0061` | arabic_english_correspondence | en | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0065` | arabic_english_correspondence | en | objection_handling | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0062` | arabic_english_correspondence | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-eval-0015` | arabic_english_correspondence | ar-en | arabic_english_translation | **PASS** | PASS | PASS | PASS | PASS |
| `seed4b-train-0067` | arabic_english_correspondence | ar-en | arabic_english_translation | **REVISE** | PASS | PASS | PASS | PASS |
| `seed4b-train-0064` | arabic_english_correspondence | ar-en | formal_correspondence | **PASS** | PASS | PASS | PASS | PASS |
