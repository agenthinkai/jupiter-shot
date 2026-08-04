# Jupiter Shot — 20T Cost-Efficiency Hypothesis

**Status:** RESEARCH AND PLANNING DOCUMENT — NOT A PERFORMANCE CLAIM  
**Date:** 2026-08-04  
**Version:** 1.0 (Run 12 preparation)

---

## Hypothesis Statement

> Jupiter Shot is exploring whether sparse expert activation, quantization and CPU/GPU Mesh orchestration can deliver up to **50× better intelligence-per-dollar** than conventional hyperscale deployment.

This is a working hypothesis. It has not been independently verified. No independently reviewed measurements currently support the 50× figure. The figure is a design target, not a measured outcome.

**Prohibited claim:** "We can build a 20T OpenAI-class model for one-fiftieth of OpenAI's cost."  
OpenAI has not published a directly comparable, audited per-model training cost that can serve as a denominator. Any comparison against OpenAI's total infrastructure investment (e.g., the $500B Stargate program) would be comparing a per-model cost against a multi-year, multi-model infrastructure program — these are not comparable denominators.

---

## Three Distinct Systems

This document distinguishes three architectures that are sometimes conflated. They are not interchangeable.

| System | Description | Can be called "20T LLM"? |
|---|---|---|
| **System 1** | A single jointly pretrained 20T sparse MoE model | Yes |
| **System 2** | A 20T-addressable distributed intelligence mesh composed of independently trained models and experts | **No** — this is a mesh, not a single jointly trained model |
| **System 3** | A smaller Jupiter model plus retrieval, tools and Decision Twins | **No** — this is an augmented smaller model |

Jupiter Shot's near-term work targets System 3 (practical) with a path toward System 2 (mesh). System 1 is included as a cost baseline only.

---

## Cost Model — Four Scenarios

All figures are **ESTIMATED** unless marked otherwise. Every assumption is stated explicitly.

### Shared Assumptions

| Parameter | Value | Status | Source |
|---|---|---|---|
| Training tokens | 2T | TARGET | Chinchilla scaling law for ~1B active params |
| Sequence length | 2,048 | TARGET | Current config |
| BF16 training precision | Yes | TARGET | Current implementation |
| Checkpoint storage per run | 50 GB | ESTIMATED | 2.6 GB weights × 10 checkpoints + optimizer states |
| Data acquisition and cleaning | $50,000 | ESTIMATED | Open-weight datasets + cleaning pipeline labor |
| Engineering labor (6 months, 3 engineers) | $450,000 | ESTIMATED | $75K/month fully loaded × 3 × 6 |
| Failed-run allowance | 20% of compute cost | ESTIMATED | Industry standard for research-stage training |
| Evaluation and safety testing | $30,000 | ESTIMATED | Internal + external red-teaming |

---

### Scenario A — Conventional Jointly Pretrained 20T Sparse MoE

**System type:** System 1 (single jointly pretrained model)

| Parameter | Value | Status | Source |
|---|---|---|---|
| Total parameters | 20T | SCENARIO DEFINITION | — |
| Active parameters per token | ~3T (15% sparsity) | ESTIMATED | Typical MoE top-2 routing at this scale |
| Training tokens | 2T | ESTIMATED | Chinchilla-optimal for active params |
| Training FLOPs | ~1.2 × 10²⁵ | ESTIMATED | 6 × active_params × tokens |
| Training precision | BF16 | ASSUMED | Industry standard |
| GPU type | H100 80GB SXM | ASSUMED | Best available for this scale |
| GPU count | 16,384 | ESTIMATED | Based on H100 FLOP/s and utilization |
| GPU utilization assumption | 45% | ESTIMATED | Large-scale MoE communication overhead |
| Training duration | ~90 days | ESTIMATED | At 45% MFU on 16K H100s |
| GPU-hours | 35,389,440 | ESTIMATED | 16,384 GPUs × 24h × 90 days |
| Cloud rate (H100, reserved) | $2.50/GPU-hour | ESTIMATED | AWS/Azure reserved pricing 2026 |
| **GPU compute cost** | **$88.5M** | **ESTIMATED** | 35.4M GPU-hours × $2.50 |
| Storage (training data + checkpoints) | $500,000 | ESTIMATED | 10 PB × $0.05/GB/month × 1 month |
| Networking (inter-node) | $2,000,000 | ESTIMATED | InfiniBand fabric at scale |
| Checkpoint storage (long-term) | $100,000 | ESTIMATED | 500 TB × $0.02/GB/month × 12 months |
| Data acquisition and cleaning | $500,000 | ESTIMATED | Web-scale data at 20T param scale |
| Engineering labor | $3,000,000 | ESTIMATED | 20 engineers × 12 months |
| Failed-run allowance (20%) | $17,700,000 | ESTIMATED | 20% of GPU compute |
| Evaluation and safety testing | $500,000 | ESTIMATED | External red-teaming at this scale |
| Quantization cost | $200,000 | ESTIMATED | INT8/INT4 post-training quantization |
| **Total training cost (base)** | **~$112M** | **ESTIMATED** | Sum of above |
| **Total training cost (low)** | **~$75M** | **ESTIMATED** | Optimistic utilization, partner compute |
| **Total training cost (high)** | **~$200M** | **ESTIMATED** | Pessimistic utilization, spot interruptions |
| Confidence level | Low | — | No precedent for 20T MoE joint training |
| Annual inference cost (10M queries/day, GPU) | $15,000,000 | ESTIMATED | 3T active params, H100 serving |
| Annual inference cost (CPU-only, INT4) | Not feasible | ESTIMATED | 3T active params exceed CPU memory |

---

### Scenario B — Jupiter Jointly Trained Sparse MoE Using Partner Compute

**System type:** System 1 (single jointly pretrained model, smaller scale)  
**Note:** Azure credits and contributed compute are treated as financing sources, not as zero economic cost.

| Parameter | Value | Status | Source |
|---|---|---|---|
| Total parameters | 1B–10B | TARGET | Current Jupiter Shot trajectory |
| Active parameters per token | 290M–1.4B | TARGET | Current MoE configs |
| Training tokens | 100B–2T | TARGET | Chinchilla scaling |
| Training FLOPs | 1.7 × 10²⁰ – 1.7 × 10²² | ESTIMATED | 6 × active_params × tokens |
| Training precision | BF16 | TARGET | Current implementation |
| GPU type | RTX 5060 (laptop) / A100 (cloud) | TARGET | Kishore's hardware + Azure |
| GPU count | 1–8 | TARGET | Current hardware plan |
| GPU utilization assumption | 55% | ESTIMATED | Small-scale, well-tuned |
| Training duration | 2–30 days | ESTIMATED | Depends on scale |
| GPU-hours | 48 – 5,760 | ESTIMATED | 1–8 GPUs × duration |
| Cloud rate (A100, Azure credits) | $3.00/GPU-hour | ESTIMATED | Azure pay-as-you-go 2026 |
| **Economic GPU compute cost** | **$144 – $17,280** | **ESTIMATED** | Credits are not free; economic cost applies |
| Storage | $500 – $5,000 | ESTIMATED | Small-scale |
| Engineering labor | $450,000 | ESTIMATED | 3 engineers × 6 months |
| Failed-run allowance (20%) | $29 – $3,456 | ESTIMATED | 20% of GPU compute |
| Evaluation and safety testing | $30,000 | ESTIMATED | Internal |
| **Total training cost (base)** | **~$480,000 – $500,000** | **ESTIMATED** | Labor-dominated at this scale |
| **Total training cost (low)** | **~$300,000** | **ESTIMATED** | Faster iteration, more partner compute |
| **Total training cost (high)** | **~$800,000** | **ESTIMATED** | More failed runs, slower iteration |
| Confidence level | Medium | — | Based on current trajectory |
| Annual inference cost (1M queries/day, GPU) | $50,000 – $200,000 | ESTIMATED | Depends on model size |
| Annual inference cost (CPU-only, INT4) | $5,000 – $30,000 | ESTIMATED | Feasible at 1B–10B scale |

---

### Scenario C — 20T-Addressable Mesh Using Open-Weight and Independently Trained Experts

**System type:** System 2 (mesh — NOT a single jointly pretrained 20T LLM)

This system is not a single model. It is a distributed mesh that routes requests to independently trained open-weight models and specialized experts. The "20T-addressable" figure refers to the total addressable parameter space across all mesh nodes, not to a single jointly trained model.

| Parameter | Value | Status | Source |
|---|---|---|---|
| Total addressable parameters | ~20T | TARGET | Sum across all mesh nodes |
| Parameters per mesh node | 7B–70B | ESTIMATED | Open-weight models (Llama, Mistral, etc.) |
| Mesh nodes | 100–300 | TARGET | Heterogeneous fleet |
| Active parameters per request | 7B–70B (single node) | ESTIMATED | Single-node routing for most requests |
| Training cost (new models) | $0 (open-weight reuse) | ESTIMATED | Reusing existing open-weight models |
| Fine-tuning cost per expert | $1,000 – $10,000 | ESTIMATED | LoRA/QLoRA on domain data |
| Total fine-tuning cost (100 experts) | $100,000 – $1,000,000 | ESTIMATED | — |
| Mesh orchestration engineering | $300,000 | ESTIMATED | 2 engineers × 6 months |
| MeshPilot router development | $150,000 | ESTIMATED | 1 engineer × 6 months |
| Infrastructure (CPU inference fleet) | $50,000/year | ESTIMATED | 20 CPU nodes × $200/month |
| GPU escalation fleet | $100,000/year | ESTIMATED | 4 GPU nodes × $2,000/month |
| **Total build cost (base)** | **~$600,000 – $1,500,000** | **ESTIMATED** | — |
| **Total build cost (low)** | **~$400,000** | **ESTIMATED** | Aggressive open-weight reuse |
| **Total build cost (high)** | **~$3,000,000** | **ESTIMATED** | More custom training |
| Confidence level | Medium | — | Open-weight reuse is proven; mesh routing is not |
| Annual inference cost (1M queries/day) | $50,000 – $150,000 | ESTIMATED | Mostly CPU, occasional GPU escalation |

---

### Scenario D — Practical Near-Term Jupiter System (Smaller Models + MeshPilot + Decision Twins)

**System type:** System 3 (augmented smaller model — NOT a 20T LLM)

This is the current Jupiter Shot implementation path. It uses a 1B–10B parameter model augmented with retrieval, tools and Decision Twins to deliver enterprise-grade outputs at a fraction of the cost of a large monolithic model.

| Parameter | Value | Status | Source |
|---|---|---|---|
| Core model parameters | 1B–10B | TARGET | Current Jupiter Shot trajectory |
| Active parameters per token | 290M–1.4B | TARGET | MoE routing |
| Retrieval augmentation | Yes (RAG) | TARGET | Decision Twin evidence register |
| Tool use | Yes | TARGET | MeshPilot tool calls |
| Decision Twin overlay | Yes | TARGET | Current AgenThinkMesh implementation |
| Training cost | $480,000 – $500,000 | ESTIMATED | Scenario B |
| Decision Twin development | $200,000 | ESTIMATED | 1 engineer × 8 months |
| RAG pipeline development | $100,000 | ESTIMATED | 1 engineer × 4 months |
| **Total build cost (base)** | **~$800,000** | **ESTIMATED** | — |
| **Total build cost (low)** | **~$500,000** | **ESTIMATED** | — |
| **Total build cost (high)** | **~$1,500,000** | **ESTIMATED** | — |
| Confidence level | High | — | Current implementation path |
| Annual inference cost (1M queries/day, CPU) | $10,000 – $50,000 | ESTIMATED | Small model, CPU-first |
| Annual inference cost (1M queries/day, GPU) | $30,000 – $100,000 | ESTIMATED | GPU escalation for complex queries |

---

## Scenario Comparison

| Metric | Scenario A (20T joint) | Scenario B (Jupiter joint) | Scenario C (Mesh) | Scenario D (Practical) |
|---|---|---|---|---|
| Total training cost (base) | ~$112M | ~$480K–$500K | ~$600K–$1.5M | ~$800K |
| Active params per token | ~3T | 290M–1.4B | 7B–70B (node) | 290M–1.4B |
| CPU inference feasible | No | Yes (small models) | Yes | Yes |
| GPU inference cost/year | $15M | $50K–$200K | $100K | $30K–$100K |
| Jointly pretrained | Yes | Yes | **No** | **No** |
| Status | Baseline only | Target | Target | **Current path** |

---

## 50× Efficiency Metrics

The following metrics define how the 50× hypothesis would be measured. All are currently **TARGET** or **UNKNOWN** — none are **MEASURED**.

| # | Metric | Status | Target | Current |
|---|---|---|---|---|
| 1 | Quality-adjusted output per dollar | UNKNOWN | 50× Scenario A | Not measured |
| 2 | Benchmark score per GPU-hour | UNKNOWN | TBD | Not measured |
| 3 | Useful tokens per dollar | UNKNOWN | TBD | Not measured |
| 4 | Cost per accepted enterprise decision | UNKNOWN | < $0.10 | Not measured |
| 5 | Active parameters per request | TARGET | < 1.5B | 290M–1.4B (design) |
| 6 | % of requests served CPU-only | TARGET | > 80% | Not measured |
| 7 | % requiring GPU escalation | TARGET | < 20% | Not measured |
| 8 | GPU utilization | TARGET | > 55% | Not measured |
| 9 | Expert-cache hit rate | UNKNOWN | > 60% | Not measured |
| 10 | Energy per accepted response | UNKNOWN | TBD | Not measured |
| 11 | Cost per 1M input tokens | TARGET | < $0.10 | Not measured |
| 12 | Cost per 1M output tokens | TARGET | < $0.30 | Not measured |
| 13 | Latency by hardware tier (CPU) | TARGET | < 2s P95 | Not measured |
| 14 | Latency by hardware tier (GPU) | TARGET | < 200ms P95 | Not measured |
| 15 | Quality loss from INT8 | UNKNOWN | < 1% on MMLU | Not measured |
| 16 | Quality loss from INT4 | UNKNOWN | < 3% on MMLU | Not measured |
| 17 | Quality loss from INT2 (experimental) | UNKNOWN | < 10% on MMLU | Not measured |
| 18 | Total cost of ownership per deployed customer | UNKNOWN | TBD | Not measured |

**Pass condition:** The 50× hypothesis passes only if Jupiter achieves comparable or superior measured quality at no more than 2% of the chosen baseline's total cost. The baseline must be a single, audited, comparable deployment — not a multi-year infrastructure program.

---

## Claim Status Summary

| Claim | Status |
|---|---|
| Jupiter Shot uses sparse MoE with top-2 routing | MEASURED (CPU forward pass verified) |
| `aux_loss > 0` after gradient-checkpoint fix | MEASURED (Run 12 gate 10b, CPU) |
| `total_loss = lm_loss + aux_loss` | MEASURED (Run 12 real-object test, CPU) |
| Isolated aux-loss router gradients are nonzero | MEASURED (Run 12 real-object test, CPU) |
| Active parameters per token ~290M (moe_1b config) | MEASURED (analytical, verified) |
| Training cost for 1B–10B model ~$480K–$500K | ESTIMATED |
| CPU inference feasible at 1B–10B scale | ESTIMATED (not yet benchmarked) |
| 50× better intelligence-per-dollar than Scenario A | TARGET (not measured) |
| Jupiter can build a 20T model at 1/50th of OpenAI's cost | **PROHIBITED — not a valid comparison** |

---

## Stop Conditions

The following conditions must be met before proceeding to any scale beyond the current laptop/single-GPU validation:

1. Run 12 gate 10b passes on Kishore's RTX 5060 with CUDA (`aux_loss > 0` on GPU)
2. Full test suite passes on RTX 5060 environment
3. At least one of the 50× metrics (items 1–18 above) is **MEASURED** (not estimated)
4. A comparable, audited baseline cost is identified for the denominator

**Do not proceed to Azure, 8×A100, 200B, 500B or 20T training** until all four conditions are met.

---

## References and Assumptions

All cost estimates use publicly available 2026 cloud pricing. No proprietary pricing agreements are assumed. Azure credits are treated as financing sources with economic cost equal to their market rate.

| Source | Used for |
|---|---|
| AWS EC2 H100 pricing (2026) | Scenario A GPU rate |
| Azure A100 pay-as-you-go (2026) | Scenario B GPU rate |
| Chinchilla scaling laws (Hoffmann et al., 2022) | Training token estimates |
| Jupiter Shot `moe_1b` config | Active parameter counts |
| Run 12 real-object test results | Measured forward-pass values |
| Industry survey (2024–2026) | Failed-run allowance, engineering labor |
