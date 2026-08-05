# Model Card: Jupiter Seed 4B Diagnostic

**Status:** DRAFT (Not for public release)

## Model Details

- **Model Name:** Jupiter Seed 4B Diagnostic
- **Base Model:** `Qwen/Qwen3-4B`
- **Architecture:** Dense Decoder-only (4.0B parameters)
- **Adaptation Method:** QLoRA (4-bit base, 16-rank adapters)
- **Language(s):** Arabic (MSA and GCC dialects), English
- **License:** PROVISIONAL (Pending legal review; base model is Apache 2.0)

## Intended Use

**Primary Use Cases:**
- Internal diagnostic testing of the AgenThink training and evaluation pipeline.
- Validation of GCC-specific enterprise terminology (Banking, Islamic Finance, Telecom).
- MeshPilot inference testing on constrained hardware.

**Prohibited Uses:**
- Commercial deployment or public redistribution.
- Generation of synthetic data for training other models.
- Any use violating the base model's license or AgenThink's acceptable use policy.

## Training Data

The diagnostic model was fine-tuned on a highly constrained dataset (500–1,000 examples) comprising:
- Internally authored prompts and answers.
- Clearly licensed public data.
- Public GCC regulatory documents.

**Data Exclusions:** No confidential client data, personal data, or unauthorized teacher-model outputs were used in this diagnostic run.

## Evaluation Results

*(To be populated after the diagnostic run using `DIAGNOSTIC_RESULTS_TEMPLATE.md`)*

## Limitations and Bias

- This is a diagnostic model trained on a minimal dataset; it is not expected to generalize well beyond the specific test cases.
- It is not a 20T frontier model and is not GPT-4 competitive.
- AI-assisted Arabic review was utilized; the model may still exhibit cultural or linguistic biases that require comprehensive human review before any commercial release.
