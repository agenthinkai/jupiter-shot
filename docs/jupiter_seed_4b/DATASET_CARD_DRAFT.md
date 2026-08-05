# Dataset Card: Jupiter Seed 4B Diagnostic Set

**Status:** DRAFT (Internal Use Only)

## Dataset Summary

The Jupiter Seed 4B Diagnostic Set is a minimal, highly curated dataset designed exclusively to validate the technical training pipeline (QLoRA) and evaluation scripts. It is limited to a maximum of 1,000 examples.

## Dataset Structure

Every record in the dataset adheres to a strict JSONL schema enforcing data provenance:

| Field | Type | Description |
| :--- | :--- | :--- |
| `example_id` | String | Unique UUID. |
| `source` | String | Origin of the data (e.g., "internal_authoring"). |
| `licence_status` | String | Legal status (e.g., "owned", "public_domain"). |
| `author` | String | Name of the human author or creator. |
| `creation_method` | String | How the data was created (e.g., "manual_translation"). |
| `language` | String | `ar`, `en`, or `ar-en`. |
| `domain` | String | Enterprise domain (e.g., `islamic_finance`). |
| `review_status` | String | Status in the review pipeline. |
| `instruction` | String | The prompt provided to the model. |
| `response` | String | The expected output. |

## Curation Rationale

The data is specifically tailored to test the model's ability to adapt to GCC enterprise terminology (Banking, Telecom, Islamic Finance) and Arabic-English code-switching, without relying on massive data volume.

## Provenance and Exclusions

- **100% Provenance Coverage:** The `dataset_loader.py` script enforces that no example is loaded without complete metadata.
- **Strict Exclusions:** The dataset strictly excludes any data from Warba, InvestGB, or other clients, as well as scraped private content and unauthorized teacher-model outputs.
