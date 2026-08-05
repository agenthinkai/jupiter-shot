# Jupiter Seed 4B: License and Distillation Matrix

**Verdict: PENDING HUMAN LEGAL REVIEW**

This document evaluates potential student and teacher models for the Jupiter Seed 4B distillation sprint. No candidate may be used until its exact version is marked **APPROVED** following human legal review. Do not infer legal permission from the phrase "open model." Software or automated agents must not mark legal approval.

## Student Candidates

| Exact Model ID | Exact Release/Version | Parameter Count | Active Parameters (MoE) | Context Length | Architecture | Model-Weight Licence | Code Licence | Distillation Status | Output-Training Status | Redistribution Status | Commercial-Use Status | Arabic Evaluation Evidence | English Evaluation Evidence | VRAM Required for Self-Hosting | Proposed Inference Precision | Estimated Teacher-Generation Cost | vLLM Support | Transformers Support | llama.cpp/GGUF Support | Known Custom-Code Requirements | Security Implications of `trust_remote_code` | Final Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-3B` | v2.5 | 3.09B | N/A (Dense) | 32,768 | Dense Decoder-only | Apache 2.0 | Apache 2.0 | Allowed | Allowed | Allowed | Allowed | Strong multilingual pre-training baseline | Proven strong English baseline | ~4GB (INT4) | INT4 | N/A (Student) | Yes | Yes | Yes | None | Low (standard architecture) | PENDING LEGAL REVIEW |
| `Qwen/Qwen2.5-7B` | v2.5 | 7.62B | N/A (Dense) | 131,072 | Dense Decoder-only | Apache 2.0 | Apache 2.0 | Allowed | Allowed | Allowed | Allowed | Excellent multilingual performance | High English reasoning | ~6GB (INT4) | INT4 | N/A (Student) | Yes | Yes | Yes | None | Low | PENDING LEGAL REVIEW |
| `FreedomIntelligence/AceGPT-7B` | v1.0 | 7B | N/A (Dense) | 4,096 | LLaMA-2 based | Apache 2.0 | Apache 2.0 | Allowed | Allowed | Allowed | Allowed | Tailored specifically for Arabic | Adequate English | ~6GB (INT4) | INT4 | N/A (Student) | Yes | Yes | Yes | Requires LLaMA architecture support | Low | PENDING LEGAL REVIEW |

## Teacher Candidates

| Exact Model ID | Exact Release/Version | Parameter Count | Active Parameters (MoE) | Context Length | Architecture | Model-Weight Licence | Code Licence | Distillation Status | Output-Training Status | Redistribution Status | Commercial-Use Status | Arabic Evaluation Evidence | English Evaluation Evidence | VRAM Required for Self-Hosting | Proposed Inference Precision | Estimated Teacher-Generation Cost | vLLM Support | Transformers Support | llama.cpp/GGUF Support | Known Custom-Code Requirements | Security Implications of `trust_remote_code` | Final Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deepseek-ai/DeepSeek-V3-Base` | V3 | 671B | 37B | 131,072 | MoE Decoder-only | DeepSeek Model License | MIT | Explicitly covers derivative models (weights, outputs, synthetic data) | Defined as derivative work | Requires identical license terms | Allowed | High multilingual capability | Top-tier reasoning | ~400GB (FP8) | FP8 / MXFP4 | High (Requires massive GPU cluster) | Yes | Yes | Partial (V3 support emerging) | MoE specific routing | Medium (complex architecture) | PENDING LEGAL REVIEW |
| `Qwen/Qwen2.5-72B-Instruct` | v2.5 | 72B | N/A (Dense) | 131,072 | Dense Decoder-only | Qianwen License | Apache 2.0 | Restricted (requires review) | Restricted | Allowed | Conditional | Exceptional Arabic performance | Top-tier English | ~40GB (INT4) | INT4 | Medium (Requires 2-4x A100) | Yes | Yes | Yes | None | Low | PENDING LEGAL REVIEW |
| `MoonshotAI/Kimi-K3` | K3 | 2.8T | Sparse | 1,000,000 | MoE with KDA | Modified MIT License | Modified MIT | Requires review | Requires review | Requires review | Requires review | Unknown / TBD | Top-tier | Extremely High (MXFP4) | MXFP4 | Extremely High | Unknown | Unknown | Unknown | Kimi Delta Attention (KDA) | High (Novel attention mechanism) | PENDING LEGAL REVIEW |
| `inceptionai/jais-family-13b-chat` | v1.0 | 13B | N/A (Dense) | 4,096 | GPT-3 based | Apache 2.0 | Apache 2.0 | Allowed | Allowed | Allowed | Allowed | Native Arabic-centric design | Good English | ~10GB (INT4) | INT4 | Low (1x A100) | Yes | Yes | Yes | ALiBi positional embeddings | Low | PENDING LEGAL REVIEW |

## License URLs
- **Qwen2.5 (Apache 2.0):** https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE
- **Qwen2.5-72B (Qianwen):** https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE
- **DeepSeek-V3 Model:** https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL
- **DeepSeek-V3 Code:** https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE
- **AceGPT:** https://github.com/FreedomIntelligence/AceGPT/blob/main/LICENSE
- **Jais-13b:** https://huggingface.co/inceptionai/jais-family-13b-chat/blob/main/LICENSE
- **Kimi-K3:** https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE

## DeepSeek-V3 License Note
Do not describe DeepSeek-V3 simply as "MIT licence." While the *code* is licensed under MIT, the *model weights* are governed by the DeepSeek Model License. This license explicitly defines derivative models to include models created through weight transfer, activations, outputs, and synthetic-data distillation. Any model trained on DeepSeek-V3 outputs is considered a derivative work and must adhere to the distribution and licensing obligations specified in the Model License.
