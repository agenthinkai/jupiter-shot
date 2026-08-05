# Jupiter Seed 4B: Arabic Human Review Queue

**Status:** PENDING HUMAN REVIEW — Version 1.1.0
**Warning:** Do not claim that these examples have been reviewed. Software must never mark an example as human-approved.

This queue contains exactly 50 examples selected after the dataset integrity audit and repair. The selection covers all 7 domains, all 3 languages, all splits, and prioritises high-risk examples including Islamic finance, regulatory, translation, safety/refusal, and source-dependent factual records.

The actual example content is located in `training/jupiter_seed_4b/data/human_review_queue.jsonl`.

## Selection Criteria

The 50 examples were selected to include:
- All 7 domains (at least 6–8 examples per domain)
- Arabic, English, and bilingual records
- High-risk Islamic finance examples (Murabaha, Sukuk, Mudaraba, Takaful)
- Regulatory explanation examples (CBK, SAMA, CITRA, AAOIFI)
- Translation examples (Arabic-English terminology)
- Safety and refusal examples (hallucination resistance, missing-info refusal)
- The most difficult examples (difficulty=advanced)
- Source-dependent factual records requiring verification

## Review Queue

| Example ID | Domain | Language | Task Type | Difficulty | Provenance Type | Reviewer | Accuracy | Fluency | GCC Approp. | Domain Terminology | Factuality | Accept/Reject/Revise | Reviewer Comments | Review Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `seed4b-train-0000` | islamic_finance | ar | islamic_finance_terminology | basic | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0001` | gcc_banking | ar | regulatory_explanation | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0002` | telecommunications | ar | question_answering | basic | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0003` | government_regulation | ar | regulatory_explanation | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0004` | energy_logistics | ar | question_answering | basic | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0005` | executive_decision | ar | executive_briefing | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0006` | arabic_english_correspondence | ar | formal_correspondence | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0007` | islamic_finance | ar | risk_identification | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0008` | gcc_banking | ar | question_answering | basic | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0009` | telecommunications | ar | executive_briefing | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0010` | government_regulation | ar | question_answering | basic | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0011` | energy_logistics | ar | document_summarization | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0012` | executive_decision | ar | objection_handling | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0013` | arabic_english_correspondence | ar | document_summarization | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0014` | islamic_finance | ar | refusal_missing_info | basic | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0015` | gcc_banking | ar | document_summarization | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0016` | telecommunications | ar | regulatory_explanation | advanced | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0017` | government_regulation | ar | executive_briefing | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0018` | energy_logistics | ar | risk_identification | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0019` | executive_decision | ar | question_answering | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0020` | islamic_finance | ar-en | arabic_english_translation | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0021` | gcc_banking | ar-en | formal_correspondence | intermediate | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0022` | telecommunications | ar-en | formal_correspondence | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0023` | government_regulation | ar-en | arabic_english_translation | intermediate | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0024` | energy_logistics | ar-en | arabic_english_translation | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0025` | executive_decision | ar-en | structured_extraction | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0026` | arabic_english_correspondence | ar-en | arabic_english_translation | intermediate | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0027` | islamic_finance | ar-en | structured_extraction | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0028` | gcc_banking | ar-en | arabic_english_translation | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0029` | arabic_english_correspondence | ar-en | formal_correspondence | intermediate | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0030` | islamic_finance | en | regulatory_explanation | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0031` | gcc_banking | en | risk_identification | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0032` | telecommunications | en | risk_identification | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0033` | government_regulation | en | document_summarization | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0034` | energy_logistics | en | executive_briefing | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0035` | executive_decision | en | decision_comparison | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0036` | arabic_english_correspondence | en | formal_correspondence | basic | TRANSLATION_OR_CORRESPONDENCE | | | | | | | | | |
| `seed4b-train-0037` | islamic_finance | en | hallucination_resistance | basic | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0038` | gcc_banking | en | hallucination_resistance | basic | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0039` | telecommunications | en | hallucination_resistance | intermediate | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0040` | government_regulation | en | refusal_missing_info | basic | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0041` | energy_logistics | en | hallucination_resistance | basic | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0042` | executive_decision | en | hallucination_resistance | intermediate | SAFETY_OR_REFUSAL | | | | | | | | | |
| `seed4b-train-0043` | arabic_english_correspondence | en | objection_handling | intermediate | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0044` | islamic_finance | en | decision_comparison | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0045` | gcc_banking | en | decision_comparison | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0046` | telecommunications | en | decision_comparison | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0047` | government_regulation | en | risk_identification | advanced | ORIGINAL_SCENARIO | | | | | | | | | |
| `seed4b-train-0048` | islamic_finance | en | executive_briefing | advanced | PUBLIC_RULE_SUMMARY | | | | | | | | | |
| `seed4b-train-0049` | gcc_banking | en | executive_briefing | intermediate | PUBLIC_RULE_SUMMARY | | | | | | | | | |

*(Note: The exact contents of these 50 examples are located in `training/jupiter_seed_4b/data/human_review_queue.jsonl`)*
