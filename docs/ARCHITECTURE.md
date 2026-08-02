# Jupiter Shot — Architecture Document

**Program:** Jupiter Shot  
**Phase:** Month 1 — Foundation and Architecture Validation  
**Organization:** AgenThinkMesh  
**Status:** Month 1 Implementation  
**Last Updated:** 2025-08-02

---

## 1. Program Overview

Jupiter Shot is an open-source, sovereign, Mesh-orchestrated language model program targeting an eventual 20-trillion-parameter sparse Mixture-of-Experts (MoE) architecture. The program is structured as a staged progression: each stage must demonstrate measurable technical milestones before advancing to the next.

**Critical constraint:** Month 1 does not train a 20T model. Month 1 proves that the data pipeline, distributed training stack, MoE routing, checkpointing, evaluation, inference, and Mesh integration can scale progressively toward that objective.

---

## 2. Month 1 Architecture

### 2.1 Dense Baseline (Stage 1)

The 1.3B dense transformer serves as the technical foundation. It validates the training stack, data pipeline, checkpointing, and evaluation harness before introducing MoE complexity.

| Component | Specification |
|-----------|--------------|
| Total parameters | ~1.3B |
| Layers | 24 |
| Hidden dimension | 2048 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN expansion | 4× (SwiGLU: 2/3 × 4 × hidden) |
| Positional encoding | Rotary (RoPE) |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Objective | Causal language modeling |
| Vocabulary | Configurable (default: 32,000) |
| Sequence length | Configurable (default: 2048) |

**Parameter count breakdown (approximate):**
- Embedding: 32,000 × 2,048 = 65.5M
- Each transformer layer: ~106M (attention + FFN)
- 24 layers: ~2,548M — but with weight tying and actual SwiGLU sizing, net ~1.3B
- Output projection (tied to embedding): shared

### 2.2 Sparse MoE Prototype (Stage 2)

The MoE prototype replaces the FFN sublayer in each transformer block with a mixture of expert FFNs. Attention layers are shared (not replicated per expert).

| Component | Specification |
|-----------|--------------|
| Total parameters | ~1B–3B |
| Active parameters per token | ~300M–700M |
| Number of experts | 8 |
| Experts selected per token | Top-2 |
| Expert capacity factor | 1.25 (configurable) |
| Router type | Linear projection → softmax |
| Load-balancing loss | Auxiliary loss (Switch Transformer style) |
| Router-z loss | Supported (DeepSeekMoE style) |
| Shared expert | Optional (1 always-active expert) |
| Expert parallelism | Supported where hardware allows |

**MoE FFN structure:**
```
Router(x) → top-2 expert indices + weights
output = Σ weight_i × Expert_i(x)  for i in top-2
```

### 2.3 Training Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Orchestration                      │
│                  (DeepSpeed ZeRO-2/3 or FSDP)                  │
├─────────────────────────────────────────────────────────────────┤
│  Data Pipeline          │  Model              │  Checkpointing  │
│  ─────────────────      │  ────────────────   │  ─────────────  │
│  JSONL / Parquet        │  Dense Transformer  │  Model state    │
│  HF Streaming           │  MoE Transformer    │  Optimizer      │
│  Deduplication          │  Router             │  Scheduler      │
│  Tokenization           │  Expert layers      │  RNG state      │
│  Packing                │                     │  DataLoader pos │
├─────────────────────────────────────────────────────────────────┤
│  Evaluation             │  Quantization       │  Inference      │
│  ─────────────────      │  ────────────────   │  ─────────────  │
│  lm-eval-harness        │  GPTQ (INT4)        │  vLLM           │
│  HellaSwag              │  AWQ (INT4)         │  llama.cpp      │
│  MMLU                   │  GGUF               │  TGI (Docker)   │
│  HumanEval              │  FP16/BF16          │  REST API       │
├─────────────────────────────────────────────────────────────────┤
│                     Mesh Integration                            │
│  Node Agent │ Model Registry │ Routing Mock │ Compliance Logger │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Hardware Configuration (Month 1)

| Resource | Specification |
|----------|--------------|
| GPUs | 8× NVIDIA A100 80GB |
| GPU memory total | 640 GB |
| CUDA | 12.1 |
| Python | 3.10 |
| Interconnect | NVLink (assumed within node) |
| Instance type | RunPod spot |
| Preemption risk | High — checkpoint every 1,000 steps |

**Memory budget for 1.3B dense model (BF16 + ZeRO-2):**
- Model weights (BF16): 1.3B × 2 bytes = 2.6 GB
- Gradients (BF16): 2.6 GB
- Optimizer states (FP32 Adam): 1.3B × 8 bytes = 10.4 GB
- Activations (2048 seq, 24 layers, activation checkpointing): ~8–12 GB
- Total per GPU (ZeRO-2, 8 GPUs): ~8–12 GB — well within 80 GB

**Memory budget for 1B–3B MoE model (BF16 + ZeRO-2, 8 experts):**
- Model weights (BF16): ~3B × 2 bytes = 6 GB
- Optimizer states (FP32): ~24 GB
- Expert communication buffers: ~2–4 GB
- Total per GPU (ZeRO-2): ~20–30 GB — within 80 GB

---

## 3. Key Design Decisions

### 3.1 RMSNorm over LayerNorm
RMSNorm eliminates the mean-centering operation, reducing computation by ~40% with no measurable quality loss at this scale. Used in LLaMA, Mistral, and most modern open-source models.

### 3.2 SwiGLU over GELU FFN
SwiGLU (Swish-Gated Linear Unit) provides better training dynamics and final quality at equivalent parameter counts. The FFN dimension is scaled to 2/3 × 4 × hidden_dim to maintain parameter parity with a standard 4× FFN.

### 3.3 Rotary Positional Embeddings (RoPE)
RoPE encodes relative position information directly into attention computation, enabling length generalization and eliminating learned positional embeddings. Used in LLaMA, Mistral, Falcon, and most modern open-source models.

### 3.4 Top-2 MoE Routing
Top-2 routing (selecting 2 of 8 experts per token) provides a good balance between specialization and stability. Top-1 routing (Switch Transformer) is more efficient but less stable. Top-2 is the standard for production MoE models (Mixtral, DeepSeek-MoE).

### 3.5 DeepSpeed ZeRO over Megatron-LM for Month 1
Megatron-LM provides superior throughput at scale (1000B+ parameters) but requires significant infrastructure setup. DeepSpeed ZeRO-2/3 is simpler to configure, supports the same model architectures, and is appropriate for Month 1 scale. Megatron-Core will be introduced in Stage 3 (10B–30B).

### 3.6 Activation Checkpointing
Activation checkpointing (gradient checkpointing) trades compute for memory: activations are recomputed during the backward pass rather than stored. This reduces activation memory by ~10× at the cost of ~33% additional compute. Essential for training at sequence length 2048+ on A100s.

---

## 4. Data Architecture

### 4.1 Pipeline Stages

```
Raw Sources → Download/Stream → Deduplication → Tokenization → Packing → Training
```

### 4.2 Supported Dataset Sources

| Dataset | License | Commercial OK | Streaming | Size | Notes |
|---------|---------|---------------|-----------|------|-------|
| RedPajama-Data-1T | Apache 2.0 | Yes | Yes | ~1T tokens | HF: togethercomputer/RedPajama-Data-1T |
| RedPajama-Data-v2 | Apache 2.0 | Yes | Yes | ~30T tokens | HF: togethercomputer/RedPajama-Data-V2 |
| The Pile (EleutherAI) | Mixed | Partial | Yes | ~825GB | Some subsets non-commercial |
| OpenWebText2 | MIT | Yes | Yes | ~65GB | HF: the_pile_openwebtext2 |
| StarCoder (code) | BigCode OpenRAIL | Restricted | Yes | ~783GB | No military use |
| Dolma | AI2 ImpACT | Research | Yes | ~3T tokens | HF: allenai/dolma |

**Month 1 sample:** Use RedPajama-Data-1T streaming subset (Wikipedia + Books) for smoke tests. Full corpus requires distributed deduplication infrastructure (see DATA_GOVERNANCE.md).

### 4.3 Tokenizer

| Tokenizer | Source | Vocab Size | Notes |
|-----------|--------|------------|-------|
| GPT-NeoX-20B | EleutherAI (HF) | 50,257 | Default fallback; Apache 2.0 |
| Mistral-7B | Mistral AI (HF) | 32,000 | Requires HF auth; gated |
| LLaMA-2 | Meta (HF) | 32,000 | Requires Meta license + HF auth; gated |

Month 1 default: GPT-NeoX tokenizer (no authentication required).

---

## 5. Checkpointing Architecture

### 5.1 Checkpoint Contents

Every checkpoint saves:
1. Model state dict (all parameters)
2. Optimizer state (Adam moments, step count)
3. LR scheduler state
4. RNG states (Python, NumPy, PyTorch, CUDA)
5. DataLoader position (dataset index, shard, offset)
6. Global step and global token count
7. Distributed shard metadata (ZeRO partition map)
8. Checkpoint integrity hash (SHA-256)

### 5.2 Checkpoint Frequency

| Training Stage | Frequency | Rationale |
|----------------|-----------|-----------|
| Smoke test | Every 100 steps | Frequent for debugging |
| Gate B/C | Every 500 steps | Balance overhead vs. safety |
| Gate D (full) | Every 1,000 steps | ~15 min on 8× A100 |

### 5.3 Spot Instance Preemption Handling

RunPod spot instances can be preempted with ~30 seconds notice. The training loop:
1. Registers a SIGTERM handler
2. On SIGTERM: completes current step, saves emergency checkpoint, exits cleanly
3. On restart: detects latest valid checkpoint, resumes from exact position

---

## 6. Inference Architecture

### 6.1 Supported Backends

| Backend | Protocol | MoE Support | Quantization | Notes |
|---------|----------|-------------|--------------|-------|
| vLLM | OpenAI-compatible REST | Partial (custom models) | GPTQ, AWQ | Best throughput |
| llama.cpp | Custom REST | No (dense only) | GGUF (Q4, Q8) | CPU + GPU |
| TGI | OpenAI-compatible REST | Yes (Mixtral) | GPTQ, AWQ | Docker required |

### 6.2 Mesh Integration Layer

The Mesh node agent wraps any inference backend and provides:
- Model registration and discovery
- Request routing (mock in Month 1; hierarchical in future)
- Compliance logging (model version, latency, token counts)
- Health and readiness endpoints
- Telemetry export

---

## 7. Mesh Routing Architecture (Future)

The Jupiter hierarchical router operates at three levels:

**Level 1 — Domain Selection**  
Routes incoming requests to the appropriate domain model family (e.g., code, science, finance, general).

**Level 2 — Model/Expert-Family Selection**  
Within a domain, selects the appropriate model size or expert cluster based on task complexity and latency requirements.

**Level 3 — Token-Level MoE Routing**  
Inside the selected model, the standard MoE router assigns each token to the top-K experts.

**Important:** Level 1 and Level 2 routing between separate models is architecturally distinct from Level 3 token-level MoE routing. They are complementary layers of a hierarchical system, not equivalent mechanisms.

---

## 8. Quality Gates

| Gate | Scope | Pass Criteria |
|------|-------|---------------|
| A — Unit Test | Synthetic data, 1 GPU/CPU | Forward pass, backward pass, optimizer step complete without error |
| B — Distributed Smoke | 2+ GPUs, 10M tokens | Distributed training, checkpoint save/reload, continuous loss decrease |
| C — Scaling Validation | 100M–500M tokens | Measured throughput, stability, utilization, projected cost |
| D — Full Baseline | Up to 10B tokens | **Requires explicit authorization before launch** |

---

*This document reflects Month 1 implementation. Projections for Stages 3–6 are in SCALING_ROADMAP.md.*
