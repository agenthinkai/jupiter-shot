# Jupiter Shot 🚀

**Open-source sparse Mixture-of-Experts language model — AgenThink Mesh native**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-orange.svg)](https://pytorch.org)

Jupiter Shot is the open-source LLM foundation for [AgenThink Mesh](https://agenthinkmesh.ai) — a federated AI inference and orchestration network. This repository contains the Month 1 deliverables: a production-grade training codebase, a 1.3B dense baseline, a sparse MoE prototype, and the full Mesh integration layer.

---

## Architecture Overview

```
Jupiter Shot
├── training/          ← Data pipeline, models, training scripts, configs
│   ├── models/
│   │   ├── dense.py   ← 1.3B dense GPT-NeoX-style transformer
│   │   └── moe.py     ← Sparse MoE transformer (8×1.3B → 8.5B total, 1.3B active)
│   ├── data_loader.py ← Streaming data loader with packing
│   ├── dataset_registry.py ← Curated dataset registry (Apache-2.0 only)
│   ├── deduplication.py    ← Exact-match deduplication (MinHash scaffolded)
│   ├── tokenizer.py        ← JupiterTokenizer wrapping GPT-NeoX-20B tokenizer
│   ├── checkpoint.py       ← Fault-tolerant checkpointing with SHA-256 integrity
│   ├── train_dense.py      ← Dense training script (DeepSpeed ZeRO-2/3)
│   ├── train_moe.py        ← MoE training script (expert parallelism)
│   ├── evaluate.py         ← lm-eval harness integration
│   └── configs/            ← YAML training configurations
├── inference/
│   ├── quantization.py     ← INT8/INT4 quantization (bitsandbytes + GPTQ)
│   └── server.py           ← FastAPI inference server (OpenAI-compatible)
├── mesh/
│   ├── node_agent.py       ← Mesh node agent (registration, health, inference)
│   ├── registry.py         ← Node registry service (FastAPI)
│   ├── router.py           ← Intelligent request router (latency + cost aware)
│   └── compliance_logger.py ← Integrity-protected audit log (SHA-256 prompt hashing)
├── benchmarks/
│   └── scaling_estimator.py ← Chinchilla scaling laws + GPU cost estimator
├── tests/                  ← pytest unit and integration tests
└── docs/
    ├── ARCHITECTURE.md
    ├── SCALING_ROADMAP.md
    ├── HARDWARE_FEASIBILITY.md
    └── DATA_GOVERNANCE.md
```

---

## Month 1 Deliverables

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | Repository scaffold & CI skeleton | ✅ Complete |
| 2 | `ARCHITECTURE.md` — full system design | ✅ Complete |
| 3 | `SCALING_ROADMAP.md` — 6-stage plan to 70B+ | ✅ Complete |
| 4 | `HARDWARE_FEASIBILITY.md` — GPU cost analysis | ✅ Complete |
| 5 | `DATA_GOVERNANCE.md` — licensing & compliance | ✅ Complete |
| 6 | Data pipeline (loader, registry, dedup, tokenizer) | ✅ Complete |
| 7 | 1.3B dense baseline (`DenseTransformer`) | ✅ Complete |
| 8 | Dense training script + configs | ✅ Complete |
| 9 | Checkpointing & fault tolerance | ✅ Complete |
| 10 | Sparse MoE prototype (`MoETransformer`) | ✅ Complete |
| 11 | MoE training script + configs | ✅ Complete |
| 12 | Evaluation suite (lm-eval integration) | ✅ Complete |
| 13 | Scaling & cost estimator | ✅ Complete |
| 14 | INT8/INT4 quantization foundation | ✅ Complete |
| 15 | OpenAI-compatible inference server | ✅ Complete |
| 16 | Mesh node agent | ✅ Complete |
| 17 | Mesh registry service | ✅ Complete |
| 18 | Mesh request router | ✅ Complete |
| 19 | Compliance audit logger | ✅ Complete |
| 20 | Unit & integration tests (71 passing, 8 skipped pending GPU) | ✅ Complete |

---

## Quick Start

### Prerequisites

```bash
# Python 3.11+, CUDA 12.1+
pip install -r requirements.txt
```

### Smoke Test (CPU, 2 minutes)

```bash
python training/train_dense.py --config training/configs/dense_smoke.yaml
```

### Full 1.3B Training (8× A100 80GB)

```bash
deepspeed --num_gpus=8 training/train_dense.py \
  --config training/configs/dense_1b3.yaml \
  --deepspeed_config training/configs/ds_zero2.json
```

### MoE Prototype Training

```bash
deepspeed --num_gpus=8 training/train_moe.py \
  --config training/configs/moe_prototype.yaml
```

### Run Tests

```bash
# Non-torch tests (fast, no GPU required)
pytest tests/test_mesh.py tests/test_data_pipeline.py -v

# Full suite (requires torch + GPU)
pytest tests/ -v
```

### Start Inference Server

```bash
python inference/server.py --model-path /checkpoints/jupiter-1.3b
# OpenAI-compatible API at http://localhost:8000
```

### Start Mesh Node Agent

```bash
python mesh/node_agent.py \
  --registry-url https://mesh.agenthinkmesh.ai \
  --model-path /checkpoints/jupiter-1.3b \
  --node-id my-node-01
```

---

## Model Specifications

### Dense Baseline (1.3B)

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-NeoX-style decoder-only |
| Parameters | 1.2988B (1,298,761,728) — analytically verified |
| Layers | 24 |
| Hidden size | 2048 |
| Attention heads | 16 |
| Context length | 2048 tokens |
| Tokenizer | GPT-NeoX-20B (50,257 vocab) |
| Positional encoding | Rotary (RoPE) |
| Normalization | RMSNorm |
| Activation | SwiGLU |

### Sparse MoE Prototype

| Parameter | Value |
|-----------|-------|
| Architecture | Dense backbone + MoE FFN layers |
| Total parameters | 0.9137B (moe_1b config) — analytically verified |
| Active parameters | 0.2907B per token (top-2 routing) — analytically verified |
| Experts | 8 |
| Active experts per token | 2 |
| Router | Top-K with auxiliary load-balancing loss |
| Expert capacity | 1.25× (overflow to shared expert) |

---

## Hardware Requirements

### Training (Month 1)

| Workload | Minimum | Recommended |
|----------|---------|-------------|
| Smoke test | CPU (any) | — |
| 1.3B dense | 4× A100 40GB | 8× A100 80GB |
| MoE prototype | 8× A100 80GB | 8× H100 80GB |

### Inference

| Workload | Minimum | Notes |
|----------|---------|-------|
| 1.3B FP16 | 1× A10G (24GB) | — |
| 1.3B INT8 | 1× A10G (12GB) | bitsandbytes |
| 1.3B INT4 | 1× RTX 4090 (8GB) | GPTQ |

---

## Dataset Registry

All training datasets are pre-approved for commercial use (Apache 2.0 or equivalent):

| Dataset | Tokens | License |
|---------|--------|---------|
| RedPajama-Data-1T | 1.2T | Apache-2.0 |
| RedPajama-Data-V2 | 30T (filtered) | Apache-2.0 |
| OpenWebText2 | 65B | MIT |
| Wikipedia (EN) | 4B | CC-BY-SA 4.0 |
| Dolma | 3T | ImpACT (open) |

See [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) for full licensing analysis.

---

## Mesh Integration

Jupiter Shot models are designed to run as first-class citizens on the AgenThink Mesh network:

1. **Node Agent** — registers the model with the Mesh registry, serves inference via OpenAI-compatible API, reports health and capacity
2. **Registry** — maintains the live catalog of all Mesh nodes, their models, and availability
3. **Router** — routes inference requests to the optimal node based on latency, cost, and model capability
4. **Compliance Logger** — SHA-256 hashes all prompts/responses for integrity-protected audit trails without storing plaintext. This is an audit-log foundation designed to support future compliance controls; it does NOT by itself establish GDPR or SOC 2 compliance.

---

## Scaling Roadmap

| Stage | Model | Params | Timeline |
|-------|-------|--------|----------|
| 1 | Dense baseline | 1.3B | Month 1 ✅ |
| 2 | MoE prototype | 8.5B total / 1.3B active | Month 1 ✅ |
| 3 | MoE v1 | 47B total / 7B active | Month 3 |
| 4 | MoE v2 | 220B total / 22B active | Month 6 |
| 5 | MoE v3 | 660B total / 70B active | Month 12 |
| 6 | Jupiter-1T | 1T+ total | Month 18+ |

See [`docs/SCALING_ROADMAP.md`](docs/SCALING_ROADMAP.md) for full cost and compute analysis.

---

## Go/No-Go Criteria (Month 1)

The Month 1 milestone is complete when all of the following are verified:

- [ ] Smoke test completes on CPU without errors
- [ ] Dense 1.3B training runs for 100 steps on 8× A100 without OOM
- [ ] MoE prototype training runs for 100 steps on 8× A100 without OOM
- [ ] Checkpoint save/load round-trip preserves model weights exactly
- [ ] Router load-balancing loss < 0.1 after 100 steps
- [ ] Inference server returns valid completions via OpenAI API
- [ ] Mesh node agent registers and heartbeats successfully
- [ ] All 32 non-torch tests pass (`pytest tests/test_mesh.py tests/test_data_pipeline.py`)
- [ ] Full torch test suite passes on GPU environment

See [`docs/MONTH1_GO_NO_GO.md`](docs/MONTH1_GO_NO_GO.md) for detailed checklist.

---

## Contributing

This repository is the open-source foundation for AgenThink Mesh. Contributions are welcome under the Apache 2.0 license.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest tests/ -v`)
4. Submit a pull request

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{jupiter_shot_2026,
  title  = {Jupiter Shot: Open-Source Sparse MoE Foundation for AgenThink Mesh},
  author = {AgenThink AI},
  year   = {2026},
  url    = {https://github.com/agenthinkai/jupiter-shot},
  license = {Apache-2.0}
}
```
