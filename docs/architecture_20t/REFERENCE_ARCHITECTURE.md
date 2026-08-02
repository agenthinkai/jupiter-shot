# Jupiter Shot — 20-Trillion-Parameter Sovereign Intelligence Reference Architecture

> **Status:** Reference Design — Not yet implemented. Requires successful completion of Month 1 GPU validation gates before any work begins on this architecture.
>
> **Prerequisite:** Month 1 Go/No-Go must be FULL GO (all GPU gates passed) before this document transitions from reference to implementation.

---

## 1. Executive Summary

This document describes the reference architecture for a 20-trillion-parameter sparse Mixture-of-Experts (MoE) language model designed for sovereign deployment — meaning the model weights, training infrastructure, and inference serving are operated entirely within a defined jurisdiction without dependency on foreign cloud providers.

The architecture is structured as a **staged progression** from the 1.3B dense baseline (Month 1) through a series of validated milestones to the 20T target. No stage begins until the preceding stage passes its Go/No-Go gate.

The 20T target is a **long-horizon reference point**, not a near-term commitment. The purpose of this document is to establish architectural constraints and design decisions that must be made correctly at small scale (1.3B) to avoid expensive rework at large scale (20T).

---

## 2. Design Principles

**Principle 1 — Sparsity is the only path to 20T.** A dense 20T model requires approximately 40 TB of BF16 weights. This is not deployable on any current hardware configuration. The 20T figure refers to total parameters in a sparse MoE model where only a small fraction (active parameters per token) are computed per forward pass. The architecture targets 2T active parameters per token (10% activation density), which is comparable in compute cost to a 2T dense model.

**Principle 2 — Sovereignty constrains hardware choices.** The architecture must be deployable on hardware that can be physically located within the target jurisdiction. This rules out certain cloud-only accelerators and requires the design to work on procurable server-class hardware (H100 NVL, H200, or equivalent).

**Principle 3 — Staged validation gates prevent wasted capital.** Each stage must demonstrate measurable progress on loss, routing stability, and hardware efficiency before the next stage is funded. The architecture is designed so that each stage produces a deployable model, not just a research artifact.

**Principle 4 — The Mesh is the product, not the model.** The 20T model is one node in the AgenThinkMesh. The architecture must support multi-model routing, capability-based dispatch, and compliance logging from the first deployable stage.

---

## 3. Staged Architecture Progression

| Stage | Total Params | Active Params/Token | Hardware | Training Tokens | Status |
|-------|-------------|---------------------|----------|-----------------|--------|
| 1 (current) | 1.3B dense | 1.3B | 8× A100 80GB | 26B | Month 1 validation |
| 2 | 47B MoE (8 experts, top-2) | 7B | 32× A100 80GB | 100B | Requires Stage 1 GO |
| 3 | 200B MoE (16 experts, top-2) | 25B | 128× H100 80GB | 500B | Requires Stage 2 GO |
| 4 | 1T MoE (64 experts, top-4) | 62B | 512× H100 NVL | 2T | Requires Stage 3 GO |
| 5 | 5T MoE (128 experts, top-4) | 156B | 2,048× H100 NVL | 5T | Requires Stage 4 GO |
| 6 (target) | 20T MoE (512 experts, top-4) | 625B | 8,192× H100 NVL | 15T | Requires Stage 5 GO |

**Important caveats on Stage 6:**
- The 8,192× H100 NVL figure assumes 80GB per GPU. H200 (141GB) or future hardware would reduce node count.
- The 15T training token target is based on Chinchilla scaling laws applied to 625B active parameters. This is an estimate, not a guarantee.
- The $2.1B capital estimate for Stage 6 hardware is based on 2024 H100 pricing and does not account for future hardware cost reductions, alternative architectures, or procurement discounts.

---

## 4. 20T Model Architecture

### 4.1 Tokenizer

The 20T model uses the same GPT-NeoX-20B tokenizer as the 1.3B baseline (vocabulary size: 50,257) to ensure full backward compatibility of trained artifacts. The vocabulary is frozen at Stage 1 and cannot be changed in later stages without invalidating all prior checkpoints.

### 4.2 Transformer Block

Each transformer block consists of:

1. **Pre-norm RMSNorm** applied to the residual stream before both the attention and FFN sublayers.
2. **Multi-head attention** with Grouped Query Attention (GQA) at Stage 3 and beyond. GQA reduces the KV cache memory requirement by a factor of `num_kv_heads / num_attention_heads`, which is critical for long-context inference at 20T scale.
3. **Rotary Position Embeddings (RoPE)** with base frequency scaled for the target context length. Stage 1 uses RoPE base 10,000 with 2,048-token context. Stage 6 targets 128,000-token context, requiring RoPE base ≥ 500,000.
4. **Sparse MoE FFN sublayer** replacing the dense SwiGLU FFN. Each token is routed to `top_k` experts out of `num_experts` total. The routing decision is made by a learned linear router.

### 4.3 Expert Architecture

Each expert is a standard SwiGLU FFN:

```
expert(x) = (SiLU(W_gate · x) ⊙ (W_up · x)) · W_down
```

At 20T scale with 512 experts, each expert has approximately 39B parameters. The total model size is:

```
Non-expert params:  ~500B  (embeddings, attention, norms, output head)
Expert params:      ~19.5T (512 experts × 38.1B params/expert)
Total:              ~20T
Active per token:   ~625B  (top-4 routing: 4 experts × 38.1B + 500B non-expert)
```

### 4.4 Router Design

The router is a learned linear projection from the residual stream to expert logits, followed by a softmax and top-k selection:

```
router_logits = x · W_router  # [batch, seq, num_experts]
router_probs = softmax(router_logits)
top_k_indices = argtop_k(router_probs, k=top_k)
```

The router is trained with:
- **Auxiliary load-balancing loss** (coefficient: 0.01) to prevent expert collapse
- **Router z-loss** (coefficient: 0.001) to prevent logit explosion
- **Expert capacity factor** (1.25) to limit token overflow

At 20T scale, the router must be replicated across all nodes (it is a small matrix: `hidden_size × num_experts`). Router synchronization is a critical bottleneck and requires careful placement in the communication schedule.

### 4.5 Parallelism Strategy

| Stage | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel |
|-------|----------------|-------------------|-----------------|---------------|
| 1 (1.3B) | 1 | 1 | 1 | 8 |
| 2 (47B) | 4 | 2 | 8 | 2 |
| 3 (200B) | 8 | 4 | 16 | 2 |
| 4 (1T) | 8 | 8 | 64 | 1 |
| 5 (5T) | 8 | 16 | 128 | 1 |
| 6 (20T) | 8 | 32 | 512 | 1 |

**Expert parallelism** is the dominant parallelism dimension at large scale. Each expert is assigned to a dedicated set of GPUs. All-to-all communication is required to route tokens to the correct expert GPUs and back. This is the primary communication bottleneck at 20T scale.

### 4.6 Context Length Progression

| Stage | Context Length | RoPE Base | KV Cache per Token (BF16) |
|-------|---------------|-----------|--------------------------|
| 1 | 2,048 | 10,000 | 2 × layers × heads × head_dim × 2 bytes |
| 2 | 4,096 | 10,000 | — |
| 3 | 8,192 | 50,000 | — |
| 4 | 32,768 | 200,000 | — |
| 5 | 65,536 | 500,000 | — |
| 6 | 128,000 | 1,000,000 | ~1.6 TB for 20T model |

The 1.6 TB KV cache at Stage 6 for a single 128K-token context is not deployable on current hardware without KV cache compression (quantization, eviction, or offloading). This is a known open problem and is not solved in this reference architecture.

---

## 5. Training Infrastructure

### 5.1 Network Topology

At 20T scale, the training cluster requires:
- **InfiniBand HDR (200 Gb/s) or NDR (400 Gb/s)** within each node group (8 GPUs)
- **Fat-tree topology** with at least 2:1 oversubscription at the spine layer
- **All-to-all bandwidth** of at least 100 GB/s aggregate for expert routing

The all-to-all communication for expert routing at 20T scale with 512 experts and a batch of 4M tokens per step requires approximately:

```
tokens_per_step = 4,000,000
bytes_per_token = hidden_size × 2 (BF16) = 8,192 × 2 = 16,384 bytes
total_all_to_all = tokens_per_step × bytes_per_token × 2 (send + receive)
                 = 4M × 16,384 × 2 = 131 TB per step
```

At 400 Gb/s (50 GB/s) per link, this requires approximately 2,620 seconds per step, which is clearly infeasible. In practice, expert parallelism is combined with tensor parallelism to reduce the all-to-all volume, and the computation is pipelined with communication. The actual communication overhead depends heavily on the specific hardware topology and software implementation.

**This is an unsolved engineering problem at 20T scale.** The reference architecture acknowledges this and defers the solution to Stage 4 (1T), where the communication patterns can be validated at smaller scale.

### 5.2 Fault Tolerance

At 8,192 GPUs, the mean time between failures (MTBF) of individual GPUs is approximately:

```
GPU MTBF (individual): ~100,000 hours
GPU MTBF (8,192 GPUs): 100,000 / 8,192 ≈ 12 hours
```

This means a GPU failure is expected approximately every 12 hours. The training system must:
1. Checkpoint every 30 minutes (not every 2 hours as in Stage 1)
2. Support elastic training (continue with N-1 GPUs after a failure)
3. Detect and exclude silent data corruption (NaN/Inf propagation)
4. Support spot instance preemption recovery in under 5 minutes

### 5.3 Data Pipeline at 20T Scale

The 15T training token target requires:
- **Data throughput**: 15T tokens / (estimated 90 days training) = 1.93T tokens/day = 22.3M tokens/second
- **Storage I/O**: At 2 bytes/token (BF16), 22.3M tokens/second = 44.6 GB/s sustained read throughput
- **Deduplication**: The training corpus must be deduplicated at scale before training begins. MinHash LSH deduplication of a 15T-token corpus requires approximately 150 TB of intermediate storage and 2–4 weeks of preprocessing time on a 100-node CPU cluster.

---

## 6. Inference Architecture

### 6.1 Serving Topology

At 20T scale, a single inference request requires activating 4 experts per token per layer. With 64 transformer layers, this means 256 expert activations per token. The experts are distributed across 512 expert-parallel shards, so a single forward pass requires all-to-all communication across all 512 shards.

The minimum serving configuration for a single 20T model instance is:
- **512 expert shards** × 8 GPUs per shard = 4,096 H100 GPUs
- **Latency per token** (estimated): 50–200ms depending on network topology
- **Throughput** (estimated): 10–50 tokens/second per serving instance

This makes the 20T model unsuitable for interactive use cases requiring sub-second latency. The Mesh architecture addresses this by routing latency-sensitive requests to smaller, faster models (1.3B or 47B) and routing deep-reasoning requests to the 20T model.

### 6.2 Quantization Strategy

The 20T model cannot be served in BF16 (40 TB of weights). The serving strategy is:
- **INT8 weight quantization**: 20 TB (2× reduction)
- **INT4 weight quantization**: 10 TB (4× reduction, quality loss must be validated)
- **FP8 (E4M3)**: 20 TB, supported on H100 with hardware acceleration

The reference architecture targets INT8 for production serving, with INT4 as a fallback for resource-constrained deployments. Quality validation of INT4 quantization at 20T scale is an open research problem.

---

## 7. Sovereign Deployment Constraints

### 7.1 Data Residency

All training data, model weights, and inference logs must remain within the designated jurisdiction. This requires:
- On-premises or sovereign-cloud object storage (no cross-border data transfer)
- Air-gapped checkpointing (no automatic sync to foreign cloud storage)
- Audit logs with cryptographic integrity protection (SHA-256 hash chains)

### 7.2 Supply Chain

The hardware supply chain for 8,192 H100 GPUs is subject to export controls. As of 2024, NVIDIA H100 and H200 GPUs are subject to US export restrictions to certain jurisdictions. The reference architecture must be validated against the applicable export control regime before procurement.

Alternative hardware options that may be available in restricted jurisdictions include:
- Huawei Ascend 910B (estimated performance: 60–70% of H100)
- Cambricon MLU370 (estimated performance: 30–40% of H100)
- Biren BR100 (estimated performance: 50–60% of H100)

The architecture is designed to be hardware-agnostic at the software level (PyTorch + DeepSpeed), but performance estimates in this document assume H100-class hardware.

### 7.3 Compliance Logging

The compliance logger (`mesh/compliance_logger.py`) provides SHA-256 integrity-protected audit logs. This is an **audit-log foundation** designed to support future compliance controls. It does not, by itself, establish GDPR compliance, SOC 2 compliance, or any other regulatory certification. Achieving regulatory compliance requires additional controls including:
- Data subject rights management (GDPR Articles 15–22)
- Access control and authentication (SOC 2 CC6)
- Incident response procedures (SOC 2 CC7)
- Third-party vendor management (SOC 2 CC9)
- Independent audit by a qualified assessor

---

## 8. Open Problems

The following problems must be solved before Stage 6 is feasible. They are listed here to ensure they are not forgotten during earlier stages.

| Problem | Severity | Current State |
|---------|----------|---------------|
| All-to-all communication at 512-expert scale | Critical | Unsolved at this scale |
| KV cache for 128K context at 20T | Critical | No known solution without compression |
| INT4 quality validation at 20T | High | Open research problem |
| Expert collapse at >128 experts | High | Observed at smaller scales, mitigations exist |
| Router load imbalance at inference time | High | Training-time balancing may not transfer |
| Checkpoint storage at 20T (40 TB/checkpoint) | High | Requires distributed checkpoint storage |
| GPU failure recovery at 8,192-GPU scale | High | Elastic training not yet validated at this scale |
| Export control compliance for hardware procurement | High | Jurisdiction-dependent |
| Training data quality at 15T tokens | Medium | Deduplication and quality filtering at scale |
| Tokenizer vocabulary adequacy for target languages | Medium | GPT-NeoX-20B tokenizer may be suboptimal |

---

## 9. What This Document Is Not

This document is a **reference architecture** — a set of design decisions and constraints that must be established early to avoid expensive rework. It is not:

- A commitment to build a 20T model
- A timeline or project plan
- A cost estimate with high confidence
- A solution to the open problems listed in Section 8
- A claim that the 20T model is technically feasible with current hardware

The 20T target is a **north star** that guides architectural decisions at smaller scales. Whether it is ever built depends on the success of earlier stages, the availability of hardware, and the business case at each stage.

---

*Last updated: 2026-08-02*
*Status: Reference design — awaiting Month 1 GPU validation*
