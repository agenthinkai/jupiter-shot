# Jupiter Shot — Founding Partner Brief

**Classification:** Confidential — Not for Distribution
**Version:** Month 1 Validation Draft
**Date:** 2026-08-02

---

## What We Are Building

Jupiter Shot is an open-architecture, sovereign-deployable sparse intelligence system. The target is a 20-trillion-parameter Mixture-of-Experts model trained on a curated, license-clear corpus, served through a distributed Mesh that routes requests to the most cost-effective capable tier.

The word "sovereign" is used precisely. It means: the model weights, training infrastructure, serving infrastructure, and audit logs can reside entirely within a jurisdiction's physical and legal borders, under that jurisdiction's control, without dependency on any foreign cloud provider, API, or licensing arrangement.

This is not a claim that Jupiter Shot will match GPT-4 or Gemini at launch. It is a claim that the architecture is designed from the start to support sovereign deployment, and that the path from the current 1.3B baseline to 20T is documented, staged, and gated on measured evidence.

---

## Current Status

Jupiter Shot is at **Month 1, Code-Complete, Hardware-Validation-Pending**.

| Deliverable | Status |
|-------------|--------|
| 1.3B dense baseline model | Code complete |
| Sparse MoE prototype (8 experts) | Code complete |
| Data pipeline (streaming, deduplication, tokenization) | Code complete |
| Checkpointing and fault tolerance | Code complete |
| Evaluation harness | Code complete |
| Scaling and cost estimator | Code complete |
| INT8/INT4 quantization foundation | Code complete |
| OpenAI-compatible inference server | Code complete |
| Mesh node agent, registry, router | Code complete |
| Compliance audit log foundation | Code complete |
| Test suite (76 passed, 0 failed) | Complete |
| CPU smoke test (20 steps, 0 NaN/Inf) | Complete |
| GPU validation (Gates 4–6) | **Pending** |

The GPU validation gates are the only remaining Month 1 items. They require access to a CUDA-capable GPU. The validation scripts are written and ready to run.

---

## The Founding Partner Opportunity

Founding Partners join before GPU validation is complete. This is the highest-risk and highest-upside position.

**What Founding Partners receive:**

1. **Named attribution** in the model card, repository, and all public communications for the life of the project
2. **Architecture co-design rights** — the ability to propose and vote on architectural decisions at each stage gate
3. **Sovereign deployment priority** — first access to deployment packages for their jurisdiction
4. **Training data inclusion** — the ability to contribute jurisdiction-specific datasets (subject to quality and license review)
5. **Governance seat** — one seat on the Jupiter Shot Technical Advisory Board

**What Founding Partners commit:**

1. **Compute contribution** — access to GPU compute for Month 1 GPU validation and Month 2 training (minimum: 8× A100 80GB for 30 days, or equivalent cloud credits)
2. **Domain expertise** — active participation in dataset curation and evaluation design for their jurisdiction
3. **Confidentiality** — this brief and all technical materials are confidential until the project's public launch

**What Founding Partners do not receive:**

- Equity in any entity (this is a technical collaboration, not a financial instrument)
- Exclusivity in any jurisdiction (multiple partners per jurisdiction are permitted)
- Any guarantee of model performance at any stage

---

## Why Now

The window for founding partnership closes when GPU validation is complete and the project moves to public announcement. After that point, the project will accept compute partners and dataset contributors under standard terms, without the co-design rights and named attribution that Founding Partners receive.

The founding partner window is estimated to close within 30–60 days.

---

## Technical Due Diligence

The complete technical foundation is available for review at:

**https://github.com/agenthinkai/jupiter-shot**

The repository contains:
- Full model architecture code (dense and MoE)
- Training scripts with DeepSpeed ZeRO integration
- Data pipeline with license-documented dataset registry
- Checkpointing with integrity verification
- Evaluation harness
- Scaling simulator with hardware cost estimates
- Architecture documents (ARCHITECTURE, SCALING_ROADMAP, HARDWARE_FEASIBILITY, DATA_GOVERNANCE, RISK_REGISTER)
- Month 1 validation results (CPU gates complete, GPU gates pending)

There are no black boxes. Every architectural decision is documented and justified. The risk register (docs/architecture_20t/RISK_REGISTER.md) explicitly lists what is known to be unsolved.

---

## Contact

To discuss founding partnership, contact the AgenThinkMesh team through the project repository or through the AgenThinkMesh platform at https://agenthinkmesh.ai.

---

*This brief does not constitute an offer of securities, investment advice, or any financial instrument. It is a technical collaboration proposal.*
