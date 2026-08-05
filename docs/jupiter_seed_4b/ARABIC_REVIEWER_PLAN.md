# Jupiter Seed 4B: Arabic Reviewer Plan

This document outlines the human-review requirements necessary to guarantee the Arabic quality of Jupiter Seed 4B. Claims of Arabic quality are strictly prohibited without documented native-speaker evaluation.

**No reviewer may be represented as confirmed unless a named human has accepted the role.**

## Required Reviewer Roles

1. **Lead Linguist (MSA & Gulf Dialects):** Responsible for final arbitration on grammar, dialect accuracy, and code-switching.
2. **Domain Expert (Islamic Finance & Banking):** Responsible for verifying technical accuracy and Sharia-compliant terminology.
3. **Domain Expert (Telecom & Energy/Government):** Responsible for verifying industry-specific terminology and regulatory language.
4. **Safety & Cultural Assessor:** Responsible for screening outputs for cultural appropriateness and safety within the GCC context.

## Coverage Requirements

The review team must evaluate model outputs and dataset samples across the following dimensions:

- **Modern Standard Arabic (MSA) coverage:** Ensuring formal, grammatically correct Arabic suitable for official documents.
- **GCC dialect (Gulf/Khaliji) coverage:** Ensuring natural tone and vocabulary for regional business communications.
- **Islamic-finance terminology coverage:** Exactness in translating and utilizing Sharia-compliant financial concepts.
- **Banking, telecom, government, energy, and legal-document coverage:** Accuracy in standard industry and regulatory language.
- **Arabic-English code switching:** Natural and accurate transitioning between Arabic and English within a single context.

## Reviewer Governance

- **Conflict-of-interest and confidentiality declarations:** All reviewers must sign NDAs and declare any conflicts of interest before accessing the evaluation datasets or model outputs.

## Scoring Rubric and Process

- **Scoring rubric:** Examples will be scored on a 1-5 scale across three axes: Linguistic Accuracy, Domain Accuracy, and Cultural Safety.
- **Reviewer-agreement measurement:** A subset of examples (e.g., 10%) will be independently scored by at least two reviewers. Cohen's kappa or a similar metric will be used to measure inter-reviewer agreement.
- **Rejection and escalation rules:** 
  - Any score of 1 or 2 on any axis results in immediate rejection of the example.
  - Disagreements between reviewers (a score delta > 1) will be escalated to the Lead Linguist or relevant Domain Expert for final arbitration.

## Estimated Schedule and Human Cost

- **Estimated schedule:** 2 - 3 weeks for Stage A (1,000 - 2,000 examples).
- **Human cost:** Assuming 3-5 reviewers processing 100-150 examples per day, compensated at standard GCC market rates for specialized linguistic/domain consulting.
