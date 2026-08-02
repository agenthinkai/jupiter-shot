# Jupiter Shot — Risk Register

> **Status:** Active — updated at each stage gate.
> **Last reviewed:** 2026-08-02

---

## Risk Classification

| Severity | Definition |
|----------|-----------|
| Critical | Would prevent the project from proceeding to the next stage |
| High | Would significantly delay or increase cost of the next stage |
| Medium | Would require workaround or mitigation, manageable impact |
| Low | Acceptable risk, monitor only |

---

## Technical Risks

### T1 — Expert Collapse at Scale
**Severity:** Critical (Stage 3+)
**Description:** In sparse MoE models, the router may learn to send all tokens to a small subset of experts, causing the remaining experts to receive no gradient signal and become useless. This has been observed at 8-expert scale and is expected to worsen at 64+ experts.
**Mitigations:**
- Auxiliary load-balancing loss (implemented in `training/models/moe.py`)
- Router z-loss (implemented)
- Expert capacity factor with token dropping
- Jitter noise in router logits during training
**Residual risk:** Load-balancing loss coefficient must be tuned per stage. Too high → routing quality degrades. Too low → collapse risk. This requires empirical tuning at each scale.
**Status:** Mitigations implemented, not yet validated at >8 experts.

### T2 — All-to-All Communication Bottleneck
**Severity:** Critical (Stage 4+)
**Description:** Expert parallelism requires all-to-all communication to route tokens to expert GPUs. At 512 experts (Stage 6), the communication volume may exceed available bandwidth.
**Mitigations:**
- Expert parallelism combined with tensor parallelism to reduce all-to-all volume
- Overlap computation with communication (pipeline the all-to-all)
- Use of InfiniBand NDR (400 Gb/s) rather than Ethernet
**Residual risk:** The communication overhead at Stage 6 scale is not known. This must be measured at Stage 4.
**Status:** Not yet validated. Deferred to Stage 4.

### T3 — NaN/Inf Propagation
**Severity:** High (all stages)
**Description:** Numerical instability (NaN or Inf values in activations or gradients) can corrupt training and require restart from checkpoint. This is more likely with BF16 precision and large learning rates.
**Mitigations:**
- Gradient clipping (max norm 1.0, implemented in training scripts)
- Loss scaling for BF16 (handled by DeepSpeed)
- NaN/Inf detection and automatic checkpoint rollback (implemented in `training/train_dense.py`)
- Router z-loss to prevent logit explosion
**Residual risk:** Silent NaN propagation (NaN in a subset of parameters not detected by the loss) is possible. Periodic parameter norm monitoring is recommended.
**Status:** Detection implemented. Silent NaN monitoring not yet implemented.

### T4 — Checkpoint Storage at Scale
**Severity:** High (Stage 4+)
**Description:** At 20T parameters in BF16, a single checkpoint is ~40 TB. Saving and loading this at training speed (every 30 minutes) requires ~22 GB/s sustained write throughput.
**Mitigations:**
- Distributed checkpointing (each node saves its own shard)
- Asynchronous checkpointing (overlap with training)
- Checkpoint compression (not yet implemented)
**Residual risk:** Distributed checkpoint correctness is complex. A corrupted checkpoint at Stage 6 could cost weeks of training time.
**Status:** Single-node checkpointing implemented. Distributed checkpointing deferred to Stage 3.

### T5 — Tokenizer Vocabulary Inadequacy
**Severity:** Medium (jurisdiction-specific deployments)
**Description:** The GPT-NeoX-20B tokenizer is optimized for English text. Arabic, Chinese, and other non-Latin scripts are tokenized inefficiently (high token count per character), reducing effective context length and increasing training cost.
**Mitigations:**
- Evaluate tokenizer efficiency on target languages before Stage 2
- If inadequate, train a custom tokenizer before Stage 2 begins
**Residual risk:** Tokenizer cannot be changed after training begins. This decision must be made before Stage 2.
**Status:** Not yet evaluated for non-English targets.

### T6 — KV Cache at Long Context
**Severity:** High (Stage 4+)
**Description:** At 128K context length and 20T model scale, the KV cache per sequence is ~32 GB (see INFERENCE_ARCHITECTURE.md Section 5.1). This limits concurrent request capacity.
**Mitigations:**
- KV cache quantization (INT8: 16 GB per sequence)
- KV cache eviction (sliding window attention)
- KV cache offloading to CPU/NVMe
**Residual risk:** All mitigations involve quality trade-offs. The acceptable trade-off must be validated empirically.
**Status:** Not yet implemented. Deferred to Stage 4.

---

## Operational Risks

### O1 — Hardware Export Controls
**Severity:** Critical (sovereign deployments)
**Description:** NVIDIA H100 and H200 GPUs are subject to US export controls (EAR). Certain jurisdictions cannot legally import these GPUs. This could block hardware procurement for sovereign deployments.
**Mitigations:**
- Legal review of export control applicability before hardware procurement
- Evaluate alternative hardware (Huawei Ascend, Cambricon, Biren) for restricted jurisdictions
- Architecture designed to be hardware-agnostic at the software level
**Residual risk:** Alternative hardware has 30–70% lower performance than H100. Training time and cost estimates must be revised for alternative hardware.
**Status:** Legal review required before Stage 3 hardware procurement.

### O2 — GPU Failure Rate at Scale
**Severity:** High (Stage 4+)
**Description:** At 8,192 GPUs, the expected GPU failure rate is approximately 1 failure every 12 hours. Without elastic training, each failure requires a full restart from checkpoint.
**Mitigations:**
- Checkpoint every 30 minutes (not 2 hours)
- Elastic training (continue with N-1 GPUs after failure)
- Hot spare GPUs (10% spare capacity)
**Residual risk:** Elastic training is not yet implemented. Deferred to Stage 3.
**Status:** 30-minute checkpointing planned. Elastic training not yet implemented.

### O3 — Data License Compliance
**Severity:** High (all stages)
**Description:** Training data license compliance is complex and evolving. Datasets that are currently permissible may become restricted due to legal challenges or license changes.
**Mitigations:**
- Dataset registry with explicit license documentation (`training/dataset_registry.py`)
- Legal review of each dataset before inclusion
- Avoid datasets with ambiguous or disputed licenses
- Maintain audit trail of dataset versions used in each training run
**Residual risk:** Legal landscape for AI training data is evolving. Ongoing legal monitoring required.
**Status:** Dataset registry implemented. Legal review process not yet formalized.

### O4 — Key Personnel Dependency
**Severity:** High (all stages)
**Description:** The project currently depends on a small team. Loss of key personnel could significantly delay or block progress.
**Mitigations:**
- Comprehensive documentation (RUNBOOK, ARCHITECTURE, this risk register)
- Code quality standards that enable new contributors to onboard quickly
- Knowledge transfer sessions at each stage gate
**Residual risk:** Tacit knowledge cannot be fully documented. Some key personnel dependency is unavoidable.
**Status:** Documentation in progress.

### O5 — Regulatory Compliance Claims
**Severity:** High (all stages)
**Description:** Premature claims of GDPR or SOC 2 compliance could expose the project to legal liability if the claims are not substantiated.
**Mitigations:**
- All compliance language has been corrected to describe the audit-log foundation as supporting future compliance controls, not establishing compliance
- No compliance certifications claimed until independent audit is completed
**Residual risk:** Ongoing vigilance required to prevent compliance overclaiming in documentation, marketing, or partner communications.
**Status:** Corrected in codebase (2026-08-02). Monitor for regression.

---

## Risk Summary

| Risk | Severity | Stage | Status |
|------|----------|-------|--------|
| T1 Expert collapse | Critical | 3+ | Mitigated, not validated |
| T2 All-to-all bottleneck | Critical | 4+ | Not yet addressed |
| T3 NaN/Inf propagation | High | All | Mitigated |
| T4 Checkpoint storage | High | 4+ | Partial mitigation |
| T5 Tokenizer adequacy | Medium | 2+ | Not yet evaluated |
| T6 KV cache at long context | High | 4+ | Not yet addressed |
| O1 Export controls | Critical | 3+ | Legal review required |
| O2 GPU failure rate | High | 4+ | Partial mitigation |
| O3 Data license compliance | High | All | Mitigated |
| O4 Key personnel | High | All | Partially mitigated |
| O5 Compliance overclaiming | High | All | Corrected |

---

*Last updated: 2026-08-02*
*Next review: After Month 1 GPU validation*
