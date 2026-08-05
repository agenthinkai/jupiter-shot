# Jupiter Seed 4B: Dataset Design

The dataset for Jupiter Seed 4B is designed in three progressive stages. Each stage must clear specific quality gates before proceeding to the next.

## Stage A: Gold Seed Set

The foundation of the dataset relies on a highly curated, human-reviewed core.

**Target Size:** 1,000–2,000 human-reviewed examples.

**Purpose:**
This stage establishes the baseline quality standards for the model. It defines the required proficiency in both Arabic and English, sets the specific terminology for the target enterprise domains (e.g., GCC banking, telecom), establishes the desired enterprise writing style, and rigorously defines safe response behaviors.

## Stage B: Synthetic Distillation Set

This stage scales the dataset using controlled synthetic generation from approved teacher models.

**Initial Controlled Target:** 100,000 examples.

**Constraint:** Do not begin with 500,000 examples. The generation must be strictly limited to the initial 100,000 target.

**Progression Gate:**
Before any further expansion is authorized, the initial 100,000 examples must successfully pass all quality, duplication, provenance, and contamination gates.

## Stage C: Preference Pairs

This stage involves creating datasets for preference optimization (e.g., DPO/RLHF) by generating accepted and rejected response pairs based on explicit quality rubrics.

### Required Quality Controls

The generation and curation of preference pairs must adhere to the following strict quality controls:

| Control Mechanism | Description |
| :--- | :--- |
| **Exact deduplication** | Removal of identical data entries. |
| **Near-duplicate detection** | Identification and removal of highly similar entries to prevent overfitting. |
| **Language identification** | Verification that the text matches the intended language (Arabic/English). |
| **Arabic-script validation** | Ensuring correct encoding and rendering of Arabic characters. |
| **Personal-information scanning** | Automated removal of any PII to comply with data privacy policies. |
| **Citation checks** | Verifying that generated facts are properly cited where required. |
| **Factuality checks** | Validation of factual claims, particularly in specialized domains like finance or regulation. |
| **Teacher-disagreement analysis** | Reviewing instances where multiple teacher models produce conflicting outputs. |
| **Toxicity review** | Screening for offensive, harmful, or inappropriate language. |
| **Unsafe-content review** | Ensuring compliance with enterprise safety standards. |
| **Domain-balance reporting** | Monitoring the distribution of data across the target enterprise domains to ensure balanced training. |
| **Human sampling** | Periodic human review of random samples to verify automated quality controls. |
| **Evaluation contamination checks** | Final verification that no preference pair data exists in the frozen evaluation benchmark. |
