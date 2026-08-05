# Dataset Card: Jupiter Seed 4B Gold Dataset

**Status:** DRAFT (Internal Use Only)

## Dataset Summary

The Jupiter Seed 4B Gold Dataset is a highly curated, provenance-controlled dataset designed to adapt Qwen3-4B to GCC enterprise terminology and Arabic-English code-switching. It contains exactly 850 internally authored examples, strictly isolated across three splits.

## Splits

- **Train:** 600 examples
- **Valid:** 100 examples (Frozen)
- **Eval:** 150 examples (Frozen)

No example or materially equivalent paraphrase appears in more than one split.

## Domain Distribution

The dataset targets the following GCC enterprise domains:
- Islamic finance and Sharia-compliant products: 20%
- GCC banking and credit analysis: 15%
- Telecommunications and infrastructure: 15%
- Government and regulatory documents: 15%
- Energy and logistics: 10%
- Executive decision analysis: 15%
- Arabic-English enterprise correspondence: 10%

## Language Distribution

- Arabic (ar): ~45%
- English (en): ~30%
- Bilingual (ar-en): ~25%

## Dataset Structure

Every record adheres to a strict JSONL schema enforcing data provenance:

| Field | Type | Description |
| :--- | :--- | :--- |
| `example_id` | String | Unique identifier (e.g., `seed4b-train-0001`). |
| `split` | String | `train`, `valid`, or `eval`. |
| `prompt` | String | The instruction provided to the model. |
| `response` | String | The expected output (internally authored). |
| `language` | String | `ar`, `en`, or `ar-en`. |
| `country_or_region` | String | E.g., `GCC`, `Kuwait`, `Saudi Arabia`. |
| `domain` | String | Enterprise domain. |
| `task_type` | String | E.g., `question_answering`, `document_summarization`. |
| `difficulty` | String | `basic`, `intermediate`, or `advanced`. |
| `source_type` | String | E.g., `internally_authored`. |
| `source_name` | String | E.g., `AgenThink Jupiter Seed 4B Dataset`. |
| `source_url` | String | Optional URL if a public fact was used. |
| `licence_or_permission` | String | Legal status. |
| `creation_method` | String | E.g., `manual_authoring`. |
| `author_type` | String | E.g., `internal_human_author`. |
| `factuality_review_status` | String | E.g., `pending_human`. |
| `arabic_review_status` | String | E.g., `pending_human`. |
| `human_review_required` | Boolean | True for all newly authored examples. |
| `sensitive_data_status` | String | Must be `clean`. |
| `contamination_group` | String | Used for split isolation checks. |
| `created_at` | String | ISO 8601 timestamp. |
| `modified_at` | String | ISO 8601 timestamp. |
| `inclusion_reason` | String | Why the example was included. |
| `rejection_reason` | String | Null unless rejected during review. |
| `content_sha256` | String | Hash of prompt + response. |
| `schema_version` | String | `1.0`. |

## Provenance and Exclusions

- **100% Provenance Coverage:** All 850 examples have complete metadata.
- **No Teacher Models:** No outputs from Qwen, DeepSeek, OpenAI, or other models were used.
- **Strict Exclusions:** The dataset strictly excludes any confidential client data (e.g., Warba, InvestGB), PII, copyrighted material, and scraped private content.
