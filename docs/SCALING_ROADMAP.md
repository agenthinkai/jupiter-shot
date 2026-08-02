# Jupiter Shot — Scaling Roadmap

**Program:** Jupiter Shot  
**Phase:** Month 1 — Foundation and Architecture Validation  
**Status:** Projections (Stages 2–6 are estimates; Stage 1 results are measured)  
**Last Updated:** 2025-08-02

---

## Notation and Methodology

All FLOPs estimates use the Chinchilla approximation: **C ≈ 6 × N_active × D** where N_active is active parameters per token and D is training tokens. For MoE models, N_active << N_total.

GPU throughput estimates are based on measured Month 1 Gate C results (placeholder until measured; see MONTH1_GO_NO_GO.md). GPU costs use RunPod A100 80GB spot pricing (~$1.89/hr per GPU as of 2025) and H100 SXM5 pricing (~$3.50/hr per GPU).

**All projections are estimates. Replace with measured values as each stage completes.**

---

## Stage 1 — 1.3B Dense Baseline

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~1.3B | Measured |
| Active parameters per token | ~1.3B (dense) | Measured |
| Number of experts | N/A | N/A |
| Experts per token | N/A | N/A |
| Training tokens (Chinchilla optimal) | ~26B (20× params) | Chinchilla scaling law |
| Training tokens (Month 1 Gate C) | 100M–500M | Month 1 scope |
| Training FLOPs (Gate C, 500M tokens) | ~1.56 × 10¹⁸ | 6 × 1.3B × 500M |
| GPU type | A100 80GB | Month 1 hardware |
| GPU count | 8 | Month 1 hardware |
| Estimated throughput | ~35,000 tokens/sec (TBD) | Placeholder — measure at Gate C |
| Training duration (500M tokens) | ~4 hours (TBD) | Placeholder |
| Training duration (26B tokens) | ~9 days (TBD) | Placeholder |
| Estimated cost (500M tokens) | ~$60 (TBD) | Placeholder |
| Estimated cost (26B tokens) | ~$3,240 (TBD) | Placeholder |
| GPU memory per GPU (ZeRO-2) | ~12 GB | Estimated |
| Network bandwidth required | NVLink within node | Single node |
| Checkpoint size (BF16) | ~2.6 GB | 1.3B × 2 bytes |
| Inference memory (BF16) | ~2.6 GB | Model weights only |
| Key technical risks | Tokenizer compatibility, data pipeline stability | Month 1 |
| Advancement criteria | Gate B passes; loss decreases monotonically; checkpoint resume verified | — |

---

## Stage 2 — 1B–3B Sparse MoE Prototype

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~1B–3B | Design |
| Active parameters per token | ~300M–700M | Top-2 of 8 experts |
| Number of experts | 8 | Design |
| Experts per token | 2 | Top-2 routing |
| Training tokens (Chinchilla optimal for active params) | ~10B–14B | 20× active params |
| Training tokens (Month 1 prototype) | 100M–500M | Month 1 scope |
| Training FLOPs (500M tokens, 500M active) | ~1.5 × 10¹⁸ | 6 × 500M × 500M |
| GPU type | A100 80GB | Month 1 hardware |
| GPU count | 8 | Month 1 hardware |
| Estimated throughput | ~25,000–30,000 tokens/sec (TBD) | MoE overhead vs. dense |
| Training duration (500M tokens) | ~5–6 hours (TBD) | Placeholder |
| Estimated cost (500M tokens) | ~$75–90 (TBD) | Placeholder |
| GPU memory per GPU (ZeRO-2) | ~20–30 GB | Expert buffers + model |
| Network bandwidth required | NVLink (expert all-to-all) | Single node |
| Checkpoint size (BF16) | ~4–6 GB | 2–3B × 2 bytes |
| Inference memory (BF16) | ~4–6 GB | All experts loaded |
| Key technical risks | Router collapse, load imbalance, expert all-to-all latency | Month 1 |
| Advancement criteria | Expert utilization within 2× of uniform; routing loss converges; no token overflow > 5% | — |

---

## Stage 3 — 10B–30B Total-Parameter MoE

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~30B | Design target |
| Active parameters per token | ~3B–5B | ~10–17% activation rate |
| Number of experts | 32–64 | Design |
| Experts per token | 2 | Top-2 routing |
| Training tokens (Chinchilla optimal) | ~60B–100B | 20× active params |
| Training FLOPs (100B tokens, 4B active) | ~2.4 × 10²¹ | 6 × 4B × 100B |
| GPU type | A100 80GB or H100 80GB | Stage 3 hardware |
| GPU count | 32–64 | Multi-node required |
| Estimated throughput | ~80,000–120,000 tokens/sec | Projected |
| Training duration (100B tokens) | ~10–15 days | Projected |
| Estimated cost | ~$150,000–$250,000 | Projected at H100 pricing |
| GPU memory per GPU | ~40–60 GB | Tensor + expert parallelism |
| Network bandwidth required | 200 Gbps InfiniBand (inter-node) | Multi-node all-to-all |
| Checkpoint size (BF16) | ~60 GB | 30B × 2 bytes |
| Inference memory (BF16) | ~60 GB | 2× A100 minimum |
| Key technical risks | Multi-node expert all-to-all latency, checkpoint coordination, NVLink vs. InfiniBand bandwidth | — |
| Advancement criteria | Throughput ≥ 80K tokens/sec; expert utilization balanced; loss matches dense baseline at equivalent active params | — |

**Infrastructure requirements not available in Month 1:**
- Multi-node cluster with InfiniBand
- Shared distributed filesystem (Lustre or GPFS) for checkpoints
- Job scheduler (SLURM or Kubernetes)

---

## Stage 4 — 100B–500B Total-Parameter MoE

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~200B | Design target |
| Active parameters per token | ~20B | ~10% activation rate |
| Number of experts | 64–128 | Design |
| Experts per token | 2 | Top-2 routing |
| Training tokens (Chinchilla optimal) | ~400B | 20× active params |
| Training FLOPs (400B tokens, 20B active) | ~4.8 × 10²³ | 6 × 20B × 400B |
| GPU type | H100 SXM5 80GB | Stage 4 hardware |
| GPU count | 256–512 | Multi-node cluster |
| Estimated throughput | ~500,000–1M tokens/sec | Projected |
| Training duration (400B tokens) | ~5–10 days | Projected |
| Estimated cost | ~$2M–$5M | Projected at H100 pricing |
| GPU memory per GPU | ~60–80 GB | Pipeline + tensor + expert parallelism |
| Network bandwidth required | 400 Gbps InfiniBand HDR | Multi-cluster all-to-all |
| Checkpoint size (BF16) | ~400 GB | 200B × 2 bytes |
| Inference memory (BF16) | ~400 GB | 5× A100 minimum |
| Key technical risks | Pipeline bubble overhead, checkpoint I/O bottleneck, expert placement strategy | — |
| Advancement criteria | MFU ≥ 40%; expert routing stable at scale; checkpoint I/O < 5% of training time | — |

**Infrastructure requirements:**
- 256–512 H100 SXM5 cluster
- 400 Gbps InfiniBand HDR
- Petabyte-scale distributed storage
- Megatron-Core with 3D parallelism (tensor + pipeline + expert)

---

## Stage 5 — 1T–5T Total-Parameter MoE

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~1T | Design target |
| Active parameters per token | ~50B–100B | ~5–10% activation rate |
| Number of experts | 256–512 | Design |
| Experts per token | 2–4 | Top-2 to Top-4 routing |
| Training tokens (Chinchilla optimal) | ~1T–2T | 20× active params |
| Training FLOPs (1T tokens, 75B active) | ~4.5 × 10²⁵ | 6 × 75B × 1T |
| GPU type | H100 SXM5 or next-gen | Stage 5 hardware |
| GPU count | 2,048–8,192 | Multi-cluster |
| Estimated throughput | ~5M–10M tokens/sec | Projected |
| Training duration (1T tokens) | ~30–60 days | Projected |
| Estimated cost | ~$50M–$200M | Projected |
| GPU memory per GPU | ~80 GB | Expert sharding across nodes |
| Network bandwidth required | 800 Gbps InfiniBand NDR | Multi-cluster expert all-to-all |
| Checkpoint size (BF16) | ~2 TB | 1T × 2 bytes |
| Inference memory (BF16) | ~2 TB | Distributed inference required |
| Key technical risks | Expert placement across clusters, fault tolerance at scale, checkpoint coordination across 8K+ GPUs | — |
| Advancement criteria | Fault-tolerant training demonstrated; expert placement optimizer validated; MFU ≥ 35% | — |

**Infrastructure requirements:**
- Multi-cluster deployment (2–8 data centers)
- Custom expert placement optimizer
- Hierarchical checkpoint system (local + remote)
- Dedicated networking fabric

---

## Stage 6 — Up to 20T Total-Parameter MoE

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | ~20T | Long-term target |
| Active parameters per token | ~200B–500B | ~1–2.5% activation rate |
| Number of experts | 1,000–10,000 | Design |
| Experts per token | 2–8 | Hierarchical routing |
| Training tokens | ~4T–10T | 20× active params |
| Training FLOPs (4T tokens, 300B active) | ~7.2 × 10²⁷ | 6 × 300B × 4T |
| GPU type | Next-generation (H200, B200, or successor) | Stage 6 hardware |
| GPU count | 16,000–100,000 | Multi-cluster sovereign infrastructure |
| Estimated throughput | ~50M–200M tokens/sec | Projected |
| Training duration | ~60–180 days | Projected |
| Estimated cost | ~$500M–$5B | Projected |
| GPU memory per GPU | ~80–192 GB | Next-gen hardware |
| Network bandwidth required | Multi-Tbps custom fabric | Sovereign network |
| Checkpoint size (BF16) | ~40 TB | 20T × 2 bytes |
| Inference memory (BF16) | ~40 TB | Distributed inference across 500+ GPUs |
| Key technical risks | Expert placement at 10K+ expert scale, hierarchical routing latency, fault tolerance across 100K GPUs, data governance at 10T+ token scale | — |
| Advancement criteria | Demonstrated fault-tolerant training at 1T scale; hierarchical routing validated; sovereign infrastructure secured | — |

**Infrastructure requirements:**
- Sovereign multi-datacenter GPU cluster
- Custom interconnect fabric
- Petabyte-scale distributed storage with sub-second checkpoint I/O
- Hierarchical expert routing system
- Dedicated ML infrastructure team

**⚠️ Feasibility note:** Stage 6 requires infrastructure comparable to GPT-4 or Gemini Ultra training runs. This is a 3–5 year program requiring significant capital investment. Month 1 establishes the architectural foundation; Stages 3–6 require separate funding and infrastructure decisions.

---

## Scaling Summary Table

| Stage | Total Params | Active Params/Token | Experts | Training Tokens | GPU Count | Est. Cost | Timeline |
|-------|-------------|---------------------|---------|-----------------|-----------|-----------|----------|
| 1 | 1.3B | 1.3B | — | 26B (Chinchilla) | 8× A100 | ~$3K | Month 1 |
| 2 | 1B–3B | 300M–700M | 8 | 10B–14B | 8× A100 | ~$5K | Month 1 |
| 3 | 30B | 3B–5B | 32–64 | 60B–100B | 32–64× H100 | ~$200K | Month 3–4 |
| 4 | 200B | 20B | 64–128 | 400B | 256–512× H100 | ~$3M | Month 6–9 |
| 5 | 1T | 50B–100B | 256–512 | 1T–2T | 2K–8K× H100 | ~$100M | Year 2–3 |
| 6 | 20T | 200B–500B | 1K–10K | 4T–10T | 16K–100K | ~$1B+ | Year 4–5 |

---

## Month 1 Advancement Criteria

Month 1 is complete when:

1. Gate B (distributed smoke test on 2+ GPUs) passes
2. Gate C (100M–500M tokens) completes with measured throughput
3. MoE prototype trains without routing failure
4. Expert utilization is within 2× of uniform distribution
5. Checkpoint save and resume verified
6. Scaling projections calibrated to measured Gate C throughput

**Month 1 does NOT advance to Month 2 without explicit authorization.**

---

*All cost and duration estimates are projections based on scaling laws and current GPU pricing. Actual results will differ. Replace placeholders with measured values from Gate C.*
