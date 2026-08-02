# Jupiter Shot — Hardware Feasibility Analysis

**Program:** Jupiter Shot  
**Phase:** Month 1  
**Last Updated:** 2025-08-02

---

## Month 1 Hardware: 8× NVIDIA A100 80GB

### What 8× A100 80GB Can Do

| Task | Feasible | Notes |
|------|----------|-------|
| 1.3B dense model training | ✅ Yes | ~12 GB/GPU with ZeRO-2; well within budget |
| 1B–3B MoE prototype training | ✅ Yes | ~20–30 GB/GPU with ZeRO-2 |
| Gate A (unit test, 1 GPU) | ✅ Yes | Tiny model, synthetic data |
| Gate B (2 GPUs, 10M tokens) | ✅ Yes | ~1 hour |
| Gate C (8 GPUs, 500M tokens) | ✅ Yes | ~4–6 hours |
| Gate D (8 GPUs, 10B tokens) | ⚠️ Possible | ~80–120 hours; requires authorization |
| Chinchilla-optimal 1.3B (26B tokens) | ⚠️ Possible | ~9 days; expensive on spot |
| 10B+ model training | ❌ No | Insufficient GPU count and memory |
| 200B model training | ❌ No | Requires 256+ H100s |
| 20T model training | ❌ No | Requires 16,000–100,000 next-gen GPUs |

### What 8× A100 80GB Cannot Do

**Gradient accumulation does not replace compute.** Gradient accumulation allows training with larger effective batch sizes by accumulating gradients over multiple micro-batches before updating weights. It reduces memory pressure but does not reduce the total FLOPs required to process a given number of tokens.

**Quantization does not reduce training requirements.** INT4/INT8 quantization reduces inference memory and latency. It does not reduce the FP32/BF16 compute required during training.

**8× A100 cannot train a 200B, 500B, or 20T model.** These require:
- 200B model: minimum 256× H100 SXM5 (tensor + pipeline parallelism)
- 500B model: minimum 512–1024× H100 SXM5
- 20T model: 16,000–100,000 next-generation GPUs

### Memory Budget Analysis

#### 1.3B Dense Model (BF16 + ZeRO-2, 8 GPUs)

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights (BF16) | 2.6 GB | 1.3B × 2 bytes |
| Gradients (BF16) | 2.6 GB | Same as weights |
| Optimizer states (FP32 Adam) | 10.4 GB | 1.3B × 8 bytes (m + v + param copy) |
| ZeRO-2 partition (8 GPUs) | /8 for optimizer | Optimizer sharded across GPUs |
| Per-GPU optimizer | ~1.3 GB | 10.4 GB / 8 |
| Activations (2048 seq, act. ckpt) | ~8–12 GB | Recomputed; depends on batch size |
| **Total per GPU (estimated)** | **~15–18 GB** | Well within 80 GB |

#### 1B–3B MoE Model (BF16 + ZeRO-2, 8 GPUs, 8 experts)

| Component | Memory | Notes |
|-----------|--------|-------|
| Model weights (BF16) | ~4–6 GB | 2–3B × 2 bytes |
| Gradients (BF16) | ~4–6 GB | Same as weights |
| Optimizer states (FP32) | ~16–24 GB | ZeRO-2 sharded |
| Per-GPU optimizer | ~2–3 GB | /8 |
| Expert communication buffers | ~2–4 GB | All-to-all buffers |
| Activations (act. ckpt) | ~10–15 GB | MoE adds routing overhead |
| **Total per GPU (estimated)** | **~20–30 GB** | Within 80 GB |

### Throughput Estimates (Placeholders — Replace at Gate C)

| Configuration | Estimated Tokens/Sec | Estimated GPU-Hours/1B Tokens |
|---------------|---------------------|-------------------------------|
| 1.3B dense, 8× A100, BF16, ZeRO-2 | ~35,000 (TBD) | ~8 GPU-hours (TBD) |
| 1B–3B MoE, 8× A100, BF16, ZeRO-2 | ~25,000 (TBD) | ~11 GPU-hours (TBD) |

**These are placeholders. Replace with measured Gate C values.**

---

## RunPod Spot Instance Considerations

RunPod spot instances provide significant cost savings (~60–70% vs. on-demand) but carry preemption risk. The training stack must handle preemption gracefully:

1. SIGTERM handler saves emergency checkpoint within 30 seconds
2. Training resumes from the last valid checkpoint on restart
3. Checkpoint every 1,000 steps (~15 minutes at estimated throughput)
4. Maximum data loss per preemption: ~15 minutes of compute

**Recommended checkpoint strategy for spot instances:**
- Save to local NVMe (fast, ~1–2 GB/min write speed)
- Asynchronously copy to object storage (S3/R2) after each checkpoint
- Keep 3 most recent checkpoints; delete older ones after verification

---

## Stage 3 Hardware Requirements (Not Month 1)

Stage 3 (10B–30B MoE) requires infrastructure not available in Month 1:

| Requirement | Specification | Availability |
|-------------|--------------|--------------|
| GPU count | 32–64× H100 SXM5 | RunPod on-demand or Lambda Labs |
| GPU memory | 80 GB each | H100 SXM5 |
| Interconnect | InfiniBand HDR (200 Gbps) | Required for multi-node all-to-all |
| Storage | 10+ TB NVMe (shared) | Lustre or NFS |
| Estimated cost | ~$200K–$500K | H100 at ~$3.50/hr |

**Month 1 cannot validate Stage 3 infrastructure.** Stage 3 requires a separate infrastructure procurement decision.

---

## Cost Projections (Month 1 Only)

| Test | Duration | GPU-Hours | Cost (A100 spot ~$1.89/GPU-hr) |
|------|----------|-----------|-------------------------------|
| Gate A (unit test) | ~10 min | 0.13 | ~$0.25 |
| Gate B (10M tokens, 2 GPUs) | ~1 hr | 2 | ~$3.78 |
| Gate C (500M tokens, 8 GPUs) | ~5 hrs | 40 | ~$75.60 |
| Gate D (10B tokens, 8 GPUs) | ~100 hrs | 800 | ~$1,512 |
| **Month 1 total (Gates A–C)** | — | ~42 | **~$80** |
| **Month 1 total (Gates A–D)** | — | ~842 | **~$1,590** |

**Gate D requires explicit authorization before launch.**

---

*All throughput and cost figures are estimates. Replace with measured values from Gate C.*
