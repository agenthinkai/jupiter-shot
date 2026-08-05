# Jupiter Seed 4B: Model Selection Scorecard

This scorecard provides a reproducible framework for evaluating student and teacher model candidates. It utilizes weighted criteria to ensure all critical dimensions of the Jupiter Seed 4B product objective are met.

**Important:** Legal approval is a mandatory gate. A high numerical score on this scorecard cannot override an unacceptable or unapproved license.

## Weighted Evaluation Criteria

| Criterion | Weight | Description |
| :--- | :--- | :--- |
| **Licence and distillation clarity** | 25% | Explicitness of commercial use, modification, and synthetic data training permissions. |
| **Arabic capability** | 20% | Demonstrated performance on Arabic benchmarks, native vocabulary size, and cultural alignment. |
| **Enterprise-task quality** | 15% | Reasoning ability, document comprehension, and structured output generation (JSON/Code). |
| **Deployment efficiency** | 15% | Parameter count, VRAM requirements, and inference speed for private deployment. |
| **Ecosystem compatibility** | 10% | Support within PyTorch, Hugging Face Transformers, vLLM, and llama.cpp. |
| **Quantization suitability** | 10% | Demonstrated quality retention when quantized to INT8 and INT4 formats. |
| **Provenance and security risk** | 5% | Transparency of training data, absence of backdoors, and safety of custom execution code (`trust_remote_code`). |

## Scoring Methodology

Each candidate is scored from 1 (Poor) to 5 (Excellent) across the seven criteria. The scores are then multiplied by their respective weights to generate a final composite score (maximum 5.0).

**Example Scorecard Entry (Template)**

| Candidate | Licence (25%) | Arabic (20%) | Enterprise (15%) | Deployment (15%) | Ecosystem (10%) | Quantization (10%) | Security (5%) | Total Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Model Name* | [1-5] | [1-5] | [1-5] | [1-5] | [1-5] | [1-5] | [1-5] | **[Weighted Sum]** |

*Note: Final scoring will be conducted once the candidate list is legally approved.*
