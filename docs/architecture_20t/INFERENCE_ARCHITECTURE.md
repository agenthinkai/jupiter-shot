# Jupiter Shot — Inference Architecture

> **Status:** Reference design — not yet implemented.

---

## 1. Overview

The inference architecture for Jupiter Shot is a **tiered serving system** where requests are routed to the most cost-effective model capable of satisfying the request's quality and latency requirements. The 20T model is the highest tier and handles only requests that require maximum reasoning depth.

---

## 2. Serving Tiers

| Tier | Model | Active Params | Latency Target | Cost/1M tokens | Use Case |
|------|-------|---------------|----------------|----------------|----------|
| T1 | 1.3B dense | 1.3B | < 50ms | ~$0.05 | Simple Q&A, classification, routing |
| T2 | 47B MoE | 7B active | < 200ms | ~$0.20 | General reasoning, summarization |
| T3 | 200B MoE | 25B active | < 500ms | ~$0.80 | Complex reasoning, code generation |
| T4 | 1T MoE | 62B active | < 2s | ~$3.00 | Deep analysis, research tasks |
| T5 | 20T MoE | 625B active | < 10s | ~$25.00 | Maximum reasoning depth |

The Mesh router (see `mesh/router.py`) selects the appropriate tier based on:
- Request complexity score (estimated from prompt length and keywords)
- Latency SLA specified by the caller
- Cost budget specified by the caller
- Current tier availability and queue depth

---

## 3. 20T Model Serving Requirements

### 3.1 Minimum Hardware Configuration

A single 20T model serving instance requires:

| Component | Specification | Count | Purpose |
|-----------|--------------|-------|---------|
| H100 NVL 94GB | NVIDIA H100 NVL | 4,096 | Expert shards (512 experts × 8 GPUs) |
| InfiniBand NDR | 400 Gb/s | Per node | Expert routing all-to-all |
| NVMe SSD | 100 TB | Distributed | Weight storage and fast loading |
| CPU servers | 64-core, 512GB RAM | 512 | CPU offload and orchestration |

**Note:** These are minimum estimates. Production deployments require redundancy (N+1 or N+2 for critical components), which increases hardware requirements by 10–20%.

### 3.2 Weight Loading Strategy

At 20T parameters in BF16, the total weight size is approximately 40 TB. Loading this from storage to GPU memory at startup is not feasible in a reasonable time. The serving system uses:

1. **Persistent weight caching**: Expert weights are kept in GPU memory across requests. This requires the full 4,096-GPU serving cluster to be dedicated to a single model instance.
2. **Lazy expert loading**: For low-traffic deployments, only the most frequently used experts are kept in GPU memory. Less-used experts are loaded from NVMe on demand (adds 50–200ms latency).
3. **Expert weight quantization**: INT8 quantization reduces weight size to 20 TB, fitting in 4,096 × 94GB = 384 TB of GPU memory with headroom.

### 3.3 Request Processing Pipeline

```
1. Request arrives at Mesh Router
2. Router scores complexity → selects T5 (20T)
3. Request queued in T5 serving queue
4. Tokenization (T1 model, < 1ms)
5. Prefill phase:
   a. Attention computation (tensor parallel across 8 GPUs per layer)
   b. Expert routing (all-to-all across 512 expert shards)
   c. Expert computation (parallel across assigned GPUs)
   d. All-to-all return
6. Decode phase (autoregressive, one token at a time):
   a. KV cache lookup
   b. Attention (fast, KV cache hit)
   c. Expert routing + computation
   d. Sampling
7. Response returned to caller
8. Compliance log written
```

---

## 4. Quantization Strategy

### 4.1 INT8 Weight-Only Quantization (Production Target)

INT8 weight-only quantization (W8A16) quantizes model weights to INT8 while keeping activations in BF16. This provides:
- 2× memory reduction (40 TB → 20 TB)
- Minimal quality degradation (< 0.5% on standard benchmarks at 1.3B scale; not yet validated at 20T scale)
- Hardware acceleration on H100 (INT8 Tensor Core support)

Implementation: `inference/quantization.py` provides the `quantize_model_int8()` function using bitsandbytes.

### 4.2 INT4 Weight Quantization (Resource-Constrained Fallback)

INT4 quantization (GPTQ or AWQ) provides:
- 4× memory reduction (40 TB → 10 TB)
- Measurable quality degradation (1–3% on standard benchmarks at 7B scale; not validated at 20T scale)
- Requires calibration dataset

**INT4 quality at 20T scale is an open research problem.** The reference architecture does not commit to INT4 for production serving until quality is validated at each stage.

### 4.3 FP8 (E4M3) Quantization

H100 GPUs support FP8 (E4M3) natively with hardware acceleration. FP8 provides:
- 2× memory reduction vs BF16 (same as INT8)
- Higher throughput than INT8 on H100 (FP8 Tensor Cores are faster)
- Better quality than INT8 in some configurations

FP8 quantization support is planned for Stage 3 and beyond.

---

## 5. KV Cache Management

### 5.1 KV Cache Size

For the 20T model with 64 transformer layers, 128 attention heads (GQA with 8 KV heads), and head dimension 128:

```
KV cache per token = 2 (K+V) × num_kv_heads × head_dim × num_layers × 2 bytes (BF16)
                   = 2 × 8 × 128 × 64 × 2
                   = 262,144 bytes = 256 KB per token
```

For a 128K-token context:
```
KV cache = 128,000 × 256 KB = 32 GB per sequence
```

This is manageable per sequence but requires careful memory management when serving multiple concurrent requests.

### 5.2 KV Cache Compression

For long-context serving (> 32K tokens), KV cache compression is required:
- **KV cache quantization** (INT8): 16 GB per 128K sequence
- **KV cache eviction** (sliding window): Evict oldest KV pairs, accept quality loss on long-range dependencies
- **KV cache offloading** (CPU/NVMe): Offload inactive KV pairs to CPU memory or NVMe, load on demand

The reference architecture does not specify a KV cache compression strategy for Stage 6. This is deferred to Stage 4 validation.

---

## 6. Inference Server

The inference server (`inference/server.py`) provides an OpenAI-compatible REST API:

```
POST /v1/completions
POST /v1/chat/completions
GET  /v1/models
GET  /health
```

The server supports:
- Streaming responses (Server-Sent Events)
- Batch inference (multiple requests in a single forward pass)
- Request cancellation
- Timeout handling

---

## 7. Latency Budget (Stage 6 Target)

For a 1,000-token prompt with 500-token response at 20T scale:

| Phase | Time | Notes |
|-------|------|-------|
| Tokenization | < 1ms | CPU, T1 model |
| Queue wait | 0–5s | Depends on load |
| Prefill (1,000 tokens) | ~2–5s | Dominated by all-to-all |
| Decode (500 tokens) | ~5–10s | 10–50 tokens/second |
| **Total** | **~7–15s** | P50 estimate |

This latency is acceptable for deep-reasoning tasks but not for interactive use. The Mesh router must not route interactive requests to T5.

---

*Last updated: 2026-08-02*
*Status: Reference design — awaiting Month 1 GPU validation*
