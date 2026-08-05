# Jupiter Seed 4B: Model Selection Scorecard

This scorecard provides a reproducible framework for evaluating student and teacher model candidates. It utilizes weighted criteria to ensure all critical dimensions of the Jupiter Seed 4B product objective are met.

**Important:** Legal approval is a mandatory gate. A high numerical score on this scorecard cannot override an unacceptable or unapproved license.

## Weighted Evaluation Criteria

| Criterion | Weight | Description |
| :--- | :--- | :--- |
| **Licence and distillation clarity** | 25% | Explicitness of commercial use, modification, and synthetic data training permissions. |
| **Arabic capability** | 20% | Demonstrated performance on Arabic benchmarks, native vocabulary size, and cultural alignment. |
| **GCC enterprise suitability** | 15% | Reasoning ability, document comprehension, and structured output generation (JSON/Code). |
| **Deployment practicality** | 15% | Parameter count, VRAM requirements, and inference speed for private deployment. |
| **Ecosystem maturity** | 10% | Support within PyTorch, Hugging Face Transformers, vLLM, and llama.cpp. |
| **Quantization readiness** | 10% | Demonstrated quality retention when quantized to INT8 and INT4 formats. |
| **Provenance and security** | 5% | Transparency of training data, absence of backdoors, and safety of custom execution code (`trust_remote_code`). |

## Scoring Methodology

Each candidate is scored from 1 (Poor) to 5 (Excellent) across the seven criteria. The scores are then multiplied by their respective weights to generate a final composite score (maximum 5.0).

### Student Candidate Scores

| Candidate | Licence (25%) | Arabic (20%) | Enterprise (15%) | Deployment (15%) | Ecosystem (10%) | Quantization (10%) | Security (5%) | Total Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen3-4B` | 5 (Apache 2.0) | 4 (Strong) | 4 | 5 (~4GB VRAM) | 5 (vLLM/GGUF) | 5 | 4 (Disclosed) | **4.45** |
| `Qwen/Qwen2.5-3B` | 5 (Apache 2.0) | 4 (Strong) | 3 | 5 (~4GB VRAM) | 5 (vLLM/GGUF) | 5 | 3 (Partial) | **4.30** |
| `FreedomIntelligence/AceGPT-7B` | 5 (Apache 2.0) | 5 (Native) | 3 | 4 (~6GB VRAM) | 4 (LLaMA-based) | 4 | 4 (Disclosed) | **4.30** |

### Teacher Candidate Scores

| Candidate | Licence (25%) | Arabic (20%) | Enterprise (15%) | Deployment (15%) | Ecosystem (10%) | Quantization (10%) | Security (5%) | Total Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-72B-Instruct` | 3 (Qianwen) | 5 (Exceptional) | 5 | 3 (2x A100) | 5 (vLLM/GGUF) | 5 | 3 (Partial) | **4.10** |
| `deepseek-ai/DeepSeek-V3-Base` | 4 (DeepSeek) | 4 (High) | 5 | 1 (8x H100) | 3 (Emerging) | 3 (FP8) | 4 (Disclosed) | **3.50** |
| `inceptionai/jais-family-13b-chat` | 5 (Apache 2.0) | 5 (Native) | 3 | 4 (1x A100) | 5 (vLLM/GGUF) | 4 | 4 (Disclosed) | **4.40** |
| `MoonshotAI/Kimi-K3` | 2 (Modified MIT)| 2 (Unknown) | 5 | 1 (Massive) | 2 (Unknown) | 2 (MXFP4) | 2 (Unknown) | **2.55** |

## Provisional Recommendations

**ALL RECOMMENDATIONS ARE PROVISIONAL — SUBJECT TO HUMAN LEGAL REVIEW**

1. **Preferred student candidate:** `Qwen/Qwen3-4B` (Highest score, hits exact 4B target, Apache 2.0, strong multilingual base).
2. **Backup student candidate:** `FreedomIntelligence/AceGPT-7B` (Native Arabic focus, Apache 2.0, slightly larger footprint).
3. **Preferred teacher candidate:** `Qwen/Qwen2.5-72B-Instruct` (Exceptional Arabic and reasoning, manageable deployment for generation).
4. **Backup teacher candidate:** `deepseek-ai/DeepSeek-V3-Base` (Top-tier reasoning, explicit distillation terms, but requires massive hardware).
5. **Arabic-specialist evaluator or teacher candidate:** `inceptionai/jais-family-13b-chat` (Native Arabic-centric design, highly efficient deployment).

## Unresolved Blockers

**Unresolved Legal Blockers:**
- `Qwen2.5-72B-Instruct` utilizes the Qianwen License; commercial use and distillation terms require specific legal interpretation and potential approval from Alibaba Cloud.
- `DeepSeek-V3` Model License explicitly claims derivative rights over synthetic data distillation; the exact impact on Jupiter Seed 4B's commercialization plan must be assessed.
- `Kimi-K3` utilizes a Modified MIT license with unknown specific restrictions.

**Unresolved Technical Blockers:**
- `DeepSeek-V3` requires significant hardware (8x H100) for inference, which may complicate the synthetic data generation pipeline.
- `Kimi-K3` requires specialized hardware and software for MXFP4 quantization and KDA attention, making it currently impractical for standard generation pipelines.
