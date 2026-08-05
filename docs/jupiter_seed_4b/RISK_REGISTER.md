# Jupiter Seed 4B: Risk Register

This document tracks the primary risks associated with the Jupiter Seed 4B distillation sprint and outlines the conditions under which the project must be halted.

## Critical Stop Conditions

The project must be stopped immediately, and a report generated, if any of the following conditions are met:

1. **Licensing Failure:** No suitable student model licence is approved by legal review.
2. **Distillation Rights:** The legal right to distill from the chosen teacher models remains unclear or is explicitly prohibited.
3. **Data Contamination:** Evaluation data cannot be legally or technically separated from the training data, compromising the benchmark.
4. **Resource Unavailability:** Qualified human resources for Arabic language review are unavailable.
5. **Budget Exceedance:** The proposed diagnostic run exceeds the maximum authorized ceiling of USD 500.
6. **Performance Regression:** The fine-tuned student model fails to improve against the frozen baseline on the evaluation benchmark.
7. **Capability Degradation:** English language capability degrades beyond the pre-agreed acceptable threshold during the Arabic/domain specialization process.
8. **Quantization Failure:** The quantization process (to INT8 or INT4) destroys required model quality, making it unsuitable for deployment.
9. **Provenance Failure:** Data provenance cannot be accurately reproduced or verified according to the Data Provenance Policy.
10. **Data Breach:** Customer-confidential data or unauthorized PII inadvertently enters the training corpus.
11. **Unauthorized Expenditure:** Paid APIs or cloud resources are required or consumed during this foundation sprint without explicit prior authorization.

## Monitored Risks

| Risk Area | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Teacher Quality** | The chosen teacher models may produce hallucinations or biased outputs that are distilled into the student. | Rigorous Stage C quality controls, including factuality checks and human sampling. |
| **Domain Overfitting** | The model may become too specialized in GCC banking/telecom, losing general reasoning capabilities. | Maintain a balanced dataset (Domain-balance reporting) and monitor English/general reasoning benchmarks. |
| **Hardware Availability** | Spot instance preemption or lack of A100 availability may delay training. | Implement robust checkpoint/resume functionality; ensure scripts are hardware-agnostic where possible. |
| **Mesh Integration** | The quantized model may not perform efficiently on CPU architectures via MeshPilot. | Conduct early benchmarking of CPU inference during the diagnostic and pilot phases. |
