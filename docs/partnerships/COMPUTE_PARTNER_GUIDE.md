# Jupiter Shot — Compute Partner Guide

**Classification:** Confidential — Not for Distribution
**Version:** Month 1 Validation Draft
**Date:** 2026-08-02

---

## Overview

Compute Partners provide GPU access for Jupiter Shot training and validation runs. This guide describes the technical requirements, contribution process, and what Compute Partners receive in return.

Compute Partnership is distinct from Founding Partnership. Compute Partners contribute hardware; Founding Partners contribute hardware plus co-design rights and governance participation. A Compute Partner may also be a Founding Partner if they meet both criteria.

---

## Minimum Compute Requirements by Stage

| Stage | Minimum Configuration | Duration | Estimated Cost |
|-------|----------------------|----------|----------------|
| Month 1 GPU Validation | 1× NVIDIA GPU (≥ 6GB VRAM) | 1–4 hours | < $5 |
| Stage 1 Full Training | 8× A100 80GB | ~2 days | ~$800 |
| Stage 2 Training | 32× A100 80GB | ~14 days | ~$21,000 |
| Stage 3 Training | 128× H100 80GB | ~17 days | ~$211,000 |
| Stage 4 Training | 512× H100 NVL | ~49 days | ~$3,000,000 |
| Stage 5 Training | 2,048× H100 NVL | ~89 days | ~$22,000,000 |
| Stage 6 Training | 8,192× H100 NVL | ~322 days | ~$316,000,000 |

Cost estimates use spot/reserved pricing. On-demand pricing is 2–3× higher.

---

## Month 1 GPU Validation — Immediate Opportunity

The most immediate compute need is Month 1 GPU validation. This requires a single GPU with at least 6GB VRAM and a CUDA-capable driver. A consumer gaming laptop with an NVIDIA RTX GPU is sufficient.

**What the validation runs:**

1. **Preflight** (~5 minutes): Hardware detection, CUDA version check, memory report
2. **Dense validation** (~30–60 minutes): 100–1,000 training steps on a VRAM-appropriate config
3. **MoE validation** (~30–60 minutes): 100–1,000 training steps on the 8-expert prototype
4. **Resume test** (~10 minutes): Checkpoint save, interruption simulation, resume verification

**How to run it:**

**Windows with NVIDIA GPU (recommended for Kuwait laptop validation):**
```bat
git clone https://github.com/agenthinkai/jupiter-shot.git
cd jupiter-shot
git checkout validation/kuwait-laptop-gpu

:: One-click runner — creates isolated .venv, installs pinned deps, runs all validation
scripts\windows\run_all_laptop_validation.bat
```

**Linux with NVIDIA GPU:**
```bash
git clone https://github.com/agenthinkai/jupiter-shot.git
cd jupiter-shot
git checkout validation/kuwait-laptop-gpu

# Create isolated virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install pinned dependencies (bundled CUDA — no system CUDA toolkit required)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run preflight (stops if CUDA not available)
python scripts/laptop_gpu_preflight.py

# Run validation
python scripts/run_laptop_dense.py
python scripts/run_laptop_moe.py
python scripts/run_laptop_resume_test.py

# Generate report
python scripts/generate_laptop_validation_draft.py
```

> **Note:** NVIDIA CUDA is not supported on modern macOS. Run validation only on Windows with NVIDIA GPU or Linux with NVIDIA GPU.

The validation scripts auto-detect VRAM and select the appropriate config (tiny/small/medium). The generated report is written to `docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md`. Raw metrics are stored under `benchmarks/results/laptop/`. Share the generated report with the Jupiter Shot team.

---

## Technical Requirements

### Supported Hardware

| GPU | VRAM | Config Selected | Notes |
|-----|------|-----------------|-------|
| RTX 3060 Laptop | 6GB | tiny | Minimum supported |
| RTX 3070/4060 | 8GB | small | Recommended for validation |
| RTX 3080/4070 | 10–12GB | medium | Full validation |
| RTX 4090 | 24GB | medium | Fastest validation |
| A100 40GB | 40GB | medium | Cloud/datacenter |
| A100 80GB | 80GB | medium | Preferred for Stage 1 |
| H100 80GB | 80GB | medium | Preferred for Stage 2+ |

### Software Requirements

- CUDA 11.8 or 12.1 (12.1 preferred)
- Python 3.10 or 3.11
- PyTorch 2.2.x (installed by setup script)
- 20GB free disk space (for checkpoints and logs)
- Internet access for initial package download

### Network Requirements for Multi-GPU Training

For Stage 1 (8× A100) and above:
- InfiniBand HDR (200 Gb/s) or NDR (400 Gb/s) preferred
- NVLink within nodes (NVSwitch for DGX systems)
- Minimum: 100 Gb/s Ethernet (expect 30–40% lower throughput vs InfiniBand)

---

## Data Residency

For Compute Partners in jurisdictions with data residency requirements:

1. **Training data** can be restricted to datasets that originate from or are approved for the partner's jurisdiction
2. **Model weights** can be stored exclusively on hardware within the partner's jurisdiction
3. **Checkpoint storage** can be configured to use local NVMe or jurisdiction-local object storage
4. **No data leaves the jurisdiction** during training — the training process is self-contained

The data pipeline (`training/data_loader.py`) and checkpoint system (`training/checkpoint.py`) are designed to support local-only operation without any external API calls.

---

## What Compute Partners Receive

1. **Named attribution** in the model card and repository for compute contributions above Stage 1 threshold
2. **Early access** to model weights for evaluation (non-commercial research use)
3. **Technical support** for deployment and fine-tuning within their jurisdiction
4. **Validation results** — all metrics from training runs on their hardware are shared back with the partner

---

## Contribution Process

1. **Express interest**: Contact the AgenThinkMesh team through https://agenthinkmesh.ai
2. **Technical review**: Share hardware specifications and jurisdiction requirements
3. **Agreement**: Sign the Compute Partner Agreement (standard terms, no exclusivity)
4. **Setup**: Follow the setup guide in `docs/LAPTOP_GPU_VALIDATION.md` or `docs/RUNBOOK.md`
5. **Run**: Execute the validation or training scripts
6. **Report**: Share the generated validation report with the team
7. **Attribution**: Named in the next model card update

---

## Frequently Asked Questions

**Q: Can I contribute cloud credits instead of physical hardware?**
A: Yes. AWS, GCP, Azure, and Lambda Labs credits are accepted. The team will manage the cloud instances.

**Q: Can I contribute a single GPU for validation only?**
A: Yes. Month 1 GPU validation is the most immediate need and can be done on a single consumer GPU.

**Q: Is there a minimum contribution threshold for attribution?**
A: Attribution requires completing at least one full validation run (Gates 4A + 4B) or contributing compute equivalent to Stage 1 training (~$800 in GPU-hours).

**Q: What happens to the model weights after training?**
A: Weights are stored on the Compute Partner's hardware (or cloud storage they control). The Jupiter Shot team receives a copy for evaluation. Weights are not published without explicit agreement from all Compute Partners who contributed to that training run.

**Q: Is this an open-source project?**
A: The code is open source (Apache 2.0). The model weights are not open source by default — the licensing terms for weights are determined at each stage gate based on the training data licenses and Compute Partner agreements.

---

*This guide does not constitute a financial instrument, investment offer, or guarantee of any kind.*
