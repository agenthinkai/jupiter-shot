# Jupiter Shot — Staged Training Plan

> **Status:** Reference plan — not yet active. Requires Month 1 GPU validation FULL GO before Stage 2 begins.

---

## Stage 1 — 1.3B Dense Baseline (Current)

**Objective:** Validate the full training stack on real hardware before committing to MoE complexity.

| Parameter | Value |
|-----------|-------|
| Model | Dense transformer, 1.3B parameters |
| Hardware | 8× A100 80GB |
| Training tokens | 26B (Chinchilla-optimal for 1.3B) |
| Batch size | 2M tokens (global) |
| Sequence length | 2,048 |
| Learning rate | 3e-4 with cosine decay |
| Warmup steps | 2,000 |
| Precision | BF16 |
| Parallelism | DDP (data parallel only) |
| Estimated duration | ~7 days |
| Estimated cost | ~$2,000 (spot A100) |
| Go/No-Go criteria | Loss < 2.5 at 26B tokens, zero NaN/Inf, checkpoint resume verified |

**Data mix (Stage 1):**
- The Pile (Apache 2.0 subset): 60%
- Wikipedia (CC-BY-SA): 20%
- GitHub Code (MIT/Apache): 15%
- ArXiv (CC-BY): 5%

---

## Stage 2 — 47B MoE Prototype

**Prerequisite:** Stage 1 Go/No-Go FULL GO

**Objective:** Validate MoE training at scale sufficient to observe routing dynamics and expert specialization.

| Parameter | Value |
|-----------|-------|
| Model | MoE, 47B total / 7B active |
| Experts | 8 total, top-2 routing |
| Hardware | 32× A100 80GB |
| Training tokens | 100B |
| Batch size | 4M tokens (global) |
| Sequence length | 4,096 |
| Learning rate | 1e-4 with cosine decay |
| Warmup steps | 5,000 |
| Precision | BF16 |
| Parallelism | TP=4, PP=2, EP=8, DP=2 |
| Estimated duration | ~21 days |
| Estimated cost | ~$25,000 (spot A100) |
| Go/No-Go criteria | Loss < 2.2 at 100B tokens, all experts active (>5% utilization), aux loss < 0.1 |

**Key validation targets for Stage 2:**
- Expert utilization coefficient of variation (CV) < 0.5 (balanced routing)
- Router entropy > 2.0 bits (diverse routing)
- No expert collapse (all experts receive > 1% of tokens)
- Communication overhead < 20% of compute time

---

## Stage 3 — 200B MoE

**Prerequisite:** Stage 2 Go/No-Go FULL GO

| Parameter | Value |
|-----------|-------|
| Model | MoE, 200B total / 25B active |
| Experts | 16 total, top-2 routing |
| Hardware | 128× H100 80GB |
| Training tokens | 500B |
| Estimated duration | ~45 days |
| Estimated cost | ~$500,000 |

**New challenges at Stage 3:**
- GQA (Grouped Query Attention) introduced to reduce KV cache
- Context length extended to 8,192 tokens
- Flash Attention 2 required for memory efficiency
- Pipeline parallelism (PP=4) introduces bubble overhead (~5–10%)

---

## Stage 4 — 1T MoE

**Prerequisite:** Stage 3 Go/No-Go FULL GO

| Parameter | Value |
|-----------|-------|
| Model | MoE, 1T total / 62B active |
| Experts | 64 total, top-4 routing |
| Hardware | 512× H100 NVL |
| Training tokens | 2T |
| Estimated duration | ~90 days |
| Estimated cost | ~$5,000,000 |

**New challenges at Stage 4:**
- Expert parallelism (EP=64) requires validated all-to-all implementation
- Context length extended to 32,768 tokens
- Checkpoint storage: ~2 TB per checkpoint, distributed checkpoint required
- GPU failure rate: ~1 failure every 2 days at 512 GPUs

---

## Stage 5 — 5T MoE

**Prerequisite:** Stage 4 Go/No-Go FULL GO

| Parameter | Value |
|-----------|-------|
| Model | MoE, 5T total / 156B active |
| Experts | 128 total, top-4 routing |
| Hardware | 2,048× H100 NVL |
| Training tokens | 5T |
| Estimated duration | ~180 days |
| Estimated cost | ~$50,000,000 |

---

## Stage 6 — 20T MoE (Target)

**Prerequisite:** Stage 5 Go/No-Go FULL GO

| Parameter | Value |
|-----------|-------|
| Model | MoE, 20T total / 625B active |
| Experts | 512 total, top-4 routing |
| Hardware | 8,192× H100 NVL (or equivalent) |
| Training tokens | 15T |
| Estimated duration | ~365 days |
| Estimated cost | ~$2,100,000,000 |

**This is a reference target, not a commitment.** The feasibility of Stage 6 depends on:
1. Successful completion of Stages 1–5
2. Resolution of the open problems listed in REFERENCE_ARCHITECTURE.md Section 8
3. Hardware availability and export control compliance
4. Capital availability

---

## Training Data Strategy

### Corpus Composition (Target for Stage 6)

| Source | Tokens | License | Notes |
|--------|--------|---------|-------|
| Web text (filtered) | 8T | Various | Requires aggressive quality filtering |
| Books | 1T | Various | Copyright compliance required |
| Code | 2T | MIT/Apache/BSD | High-quality reasoning signal |
| Scientific papers | 1T | CC-BY/open access | ArXiv, PubMed, Semantic Scholar |
| Wikipedia | 0.5T | CC-BY-SA | High quality, multilingual |
| Legal/regulatory | 0.5T | Public domain | Jurisdiction-specific |
| Synthetic (reasoning) | 2T | Proprietary | Generated from smaller models |
| **Total** | **15T** | | |

### Data Quality Pipeline

1. **Language identification**: fastText language classifier, filter to target languages
2. **Quality scoring**: perplexity filtering using a small reference model
3. **Deduplication**: MinHash LSH with 5-gram shingles, Jaccard threshold 0.8
4. **Toxicity filtering**: classifier-based, remove top 5% by toxicity score
5. **PII detection**: regex + NER-based, remove or redact personal information
6. **Domain classification**: route to domain-specific quality thresholds

---

## Evaluation Schedule

| Milestone | Benchmarks | Threshold |
|-----------|-----------|-----------|
| 10B tokens | HellaSwag, ARC-Easy | Loss < 3.0 |
| 50B tokens | + PIQA, WinoGrande | Loss < 2.5 |
| 100B tokens | + ARC-Challenge, MMLU | Loss < 2.2 |
| 500B tokens | + GSM8K, HumanEval | Loss < 2.0 |
| 1T tokens | + MATH, BBH | Loss < 1.9 |
| 5T tokens | Full benchmark suite | Loss < 1.8 |
| 15T tokens | Full benchmark suite | Loss < 1.7 |

---

*Last updated: 2026-08-02*
*Status: Reference plan — awaiting Month 1 GPU validation*
