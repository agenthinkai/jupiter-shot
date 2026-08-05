# Jupiter Seed 4B: Legal Evidence Pack

This document provides primary-source evidence regarding the licensing terms for the student and teacher candidates under consideration for Jupiter Seed 4B.

**This document does not provide legal conclusions. It presents evidence and specific questions for qualified human counsel.**

---

## 1. Qwen/Qwen3-4B (Student Candidate)

**Repository Code Licence:** Apache 2.0
**Model-Weight Licence:** Apache 2.0
**Source:** [Qwen3-4B Official Repository](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE)

**Evidence:**
The repository and model weights are licensed under the standard Apache License, Version 2.0.

**Questions for Counsel:**
- Are there any specific attribution requirements when deploying a fine-tuned version of this model commercially?
- Does the Apache 2.0 license pose any restrictions on integrating this model into the proprietary AgenThink Mesh?

---

## 2. deepseek-ai/DeepSeek-V3-Base (Teacher Candidate)

**Repository Code Licence:** MIT
**Model-Weight Licence:** DeepSeek Model License
**Source (Code):** [DeepSeek-V3 LICENSE-CODE](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE)
**Source (Weights):** [DeepSeek-V3 LICENSE-MODEL](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL)

**Evidence (Output-Use & Distillation Rights):**
> "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.

**Evidence (Derivative-Model Obligations):**
> 4.a. Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model...

**Questions for Counsel:**
- Since generating synthetic data for training Jupiter Seed 4B explicitly creates a "Derivative of the Model," does this compel AgenThink to release Jupiter Seed 4B under the DeepSeek Model License?
- Does the requirement to include DeepSeek's use-based restrictions (Attachment A) conflict with our intended commercialization plan or the proposed Jupiter Seed license?

---

## 3. Qwen/Qwen2.5-72B-Instruct (Teacher Candidate)

**Repository Code Licence:** Apache 2.0
**Model-Weight Licence:** Qianwen License
**Source:** [Qwen2.5-72B-Instruct LICENSE](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE)

**Evidence:**
The model weights are governed by the bespoke Tongyi Qianwen License Agreement.

**Questions for Counsel:**
- Does the Qianwen License explicitly permit or restrict the use of model outputs for training (distillation) of a competing or commercial model?
- What are the specific commercial redistribution rights and obligations if we use this model to generate the synthetic distillation dataset?

---

## 4. FreedomIntelligence/AceGPT-7B (Student Candidate)

**Repository Code Licence:** Apache 2.0
**Model-Weight Licence:** Apache 2.0
**Source:** [AceGPT Official Repository](https://github.com/FreedomIntelligence/AceGPT/blob/main/LICENSE)

**Evidence:**
The repository is licensed under Apache 2.0. However, AceGPT is built upon LLaMA-2.

**Questions for Counsel:**
- Because AceGPT is a derivative of LLaMA-2, does the LLaMA-2 Community License (which has commercial user limits and acceptable use policies) pass through and govern our use of AceGPT, despite the Apache 2.0 tag on the repository?

---

## 5. inceptionai/jais-family-13b-chat (Teacher/Evaluator Candidate)

**Repository Code Licence:** Apache 2.0
**Model-Weight Licence:** Apache 2.0
**Source:** [jais-family-13b-chat Official Repository](https://huggingface.co/inceptionai/jais-family-13b-chat/blob/main/LICENSE)

**Evidence:**
Licensed under Apache 2.0.

**Questions for Counsel:**
- Confirm that using Jais-13b-chat outputs to train Jupiter Seed 4B (distillation) is fully permissible under this license without triggering copyleft obligations.

---

## 6. MoonshotAI/Kimi-K3 (Teacher Candidate)

**Repository Code Licence:** Modified MIT
**Model-Weight Licence:** Modified MIT
**Source:** [Kimi-K3 Official Repository](https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE)

**Evidence:**
Licensed under a "Modified MIT" license.

**Questions for Counsel:**
- What specific modifications have been made to the standard MIT license?
- Do these modifications restrict commercial use, distillation, or output-training?
