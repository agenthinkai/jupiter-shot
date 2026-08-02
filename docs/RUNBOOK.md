# Jupiter Shot — Operations Runbook

**Version:** Month 1  
**Audience:** ML Engineers, DevOps, Mesh Operators

---

## 1. Environment Setup

### 1.1 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 22.04 | Ubuntu 22.04 LTS |
| Python | 3.11 | 3.11.x |
| CUDA | 12.1 | 12.3 |
| GPU (training) | 4× A100 40GB | 8× A100 80GB |
| GPU (inference) | 1× A10G 24GB | 1× A100 40GB |
| RAM | 256GB | 512GB |
| Storage | 2TB NVMe | 4TB NVMe RAID |
| Network | 100GbE | 400GbE InfiniBand |

### 1.2 Installation

```bash
# Clone repository
git clone https://github.com/agenthinkai/jupiter-shot.git
cd jupiter-shot

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 1.3 Environment Variables

```bash
# Required for HuggingFace datasets
export HF_TOKEN=hf_xxxx

# Required for Mesh node
export MESH_REGISTRY_URL=https://mesh.agenthinkmesh.ai
export MESH_NODE_SECRET=your_node_secret

# Optional: W&B logging
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=jupiter-shot
```

---

## 2. Training Procedures

### 2.1 Smoke Test (CPU, ~2 minutes)

```bash
python training/train_dense.py --config training/configs/dense_smoke.yaml
```

Expected output:
```
Step 1/10 | loss=10.832 | lr=1.0e-04 | tokens/s=1234
Step 2/10 | loss=10.791 | lr=1.0e-04 | tokens/s=1238
...
Step 10/10 | loss=10.234 | lr=1.0e-04 | tokens/s=1241
Smoke test complete. Loss decreased from 10.832 → 10.234.
```

### 2.2 Dense 1.3B Training (8× A100 80GB)

```bash
# Single node, 8 GPUs
deepspeed --num_gpus=8 training/train_dense.py \
  --config training/configs/dense_1b3.yaml \
  --output_dir /checkpoints/jupiter-1.3b

# Multi-node (2 nodes × 8 GPUs)
deepspeed --num_nodes=2 --num_gpus=8 \
  --hostfile hostfile.txt \
  training/train_dense.py \
  --config training/configs/dense_1b3.yaml \
  --output_dir /checkpoints/jupiter-1.3b
```

**Expected training time:** ~72 hours for 100B tokens on 8× A100 80GB.

**Expected GPU memory:** ~65GB per GPU with ZeRO-2, ~45GB with ZeRO-3.

### 2.3 MoE Prototype Training

```bash
deepspeed --num_gpus=8 training/train_moe.py \
  --config training/configs/moe_prototype.yaml \
  --output_dir /checkpoints/jupiter-moe-prototype
```

**Expected GPU memory:** ~70GB per GPU with expert parallelism across 8 GPUs.

### 2.4 Resuming from Checkpoint

```bash
python training/train_dense.py \
  --config training/configs/dense_1b3.yaml \
  --resume_from_checkpoint /checkpoints/jupiter-1.3b/step_010000
```

The training script automatically detects and resumes from the latest valid checkpoint if `--resume_from_checkpoint` is not specified and a checkpoint exists in `output_dir`.

### 2.5 Checkpoint Management

```bash
# List checkpoints
ls /checkpoints/jupiter-1.3b/step_*/metadata.json | xargs -I{} python3 -c \
  "import json; m=json.load(open('{}'));print(m['global_step'], m['loss'])"

# Manually rotate (keep last 5)
python3 -c "
from training.checkpoint import rotate_checkpoints
rotate_checkpoints('/checkpoints/jupiter-1.3b', keep_last_n=5)
"

# Verify checkpoint integrity
python3 -c "
from training.checkpoint import load_checkpoint
from training.models.dense import DenseConfig, DenseTransformer
cfg = DenseConfig.from_pretrained('/checkpoints/jupiter-1.3b/step_010000')
model = DenseTransformer(cfg)
meta = load_checkpoint('/checkpoints/jupiter-1.3b/step_010000', model)
print('Checkpoint valid. Step:', meta['global_step'])
"
```

---

## 3. Evaluation

### 3.1 Standard Benchmarks

```bash
python training/evaluate.py \
  --model_path /checkpoints/jupiter-1.3b/step_100000 \
  --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa \
  --output_dir /results/jupiter-1.3b-100k
```

### 3.2 Expected Month 1 Baseline Scores

| Benchmark | Random | GPT-2 (1.5B) | Target (1.3B) |
|-----------|--------|--------------|---------------|
| HellaSwag | 25.0% | 50.9% | ≥45% |
| ARC-Easy | 25.0% | 43.9% | ≥40% |
| ARC-Challenge | 25.0% | 29.0% | ≥27% |
| WinoGrande | 50.0% | 57.0% | ≥54% |
| PIQA | 50.0% | 70.8% | ≥65% |

> **Note:** These are indicative targets for a model trained on 100B tokens. Full performance requires training to Chinchilla-optimal token count (~26T tokens for 1.3B params).

---

## 4. Inference

### 4.1 Start Inference Server

```bash
python inference/server.py \
  --model-path /checkpoints/jupiter-1.3b/step_100000 \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization int8  # optional: int4, none
```

### 4.2 Test Inference

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jupiter-1.3b",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4.3 Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "model": "jupiter-1.3b", "gpu_memory_used_gb": 12.3}
```

---

## 5. Mesh Operations

### 5.1 Register Node with Mesh

```bash
python mesh/node_agent.py \
  --registry-url https://mesh.agenthinkmesh.ai \
  --model-path /checkpoints/jupiter-1.3b/step_100000 \
  --node-id $(hostname)-gpu0 \
  --max-concurrent-requests 4 \
  --port 8001
```

### 5.2 Start Mesh Registry (Self-Hosted)

```bash
uvicorn mesh.registry:app --host 0.0.0.0 --port 9000
```

### 5.3 Start Mesh Router

```bash
python mesh/router.py \
  --registry-url http://localhost:9000 \
  --host 0.0.0.0 \
  --port 8080
```

### 5.4 Check Node Health

```bash
# Via registry API
curl http://localhost:9000/nodes | python3 -m json.tool

# Via node agent directly
curl http://localhost:8001/health
```

---

## 6. Monitoring

### 6.1 Training Metrics (W&B)

Key metrics to monitor during training:

| Metric | Healthy Range | Alert Threshold |
|--------|---------------|-----------------|
| `train/loss` | Decreasing | Plateau > 1000 steps |
| `train/grad_norm` | 0.1 – 10.0 | > 100 (exploding) |
| `train/router_aux_loss` | 0.001 – 0.05 | > 0.1 (routing collapse) |
| `train/expert_load_imbalance` | < 0.3 | > 0.5 |
| `system/gpu_memory_used` | < 90% | > 95% |
| `system/tokens_per_second` | > 50k | < 20k |

### 6.2 Compliance Audit Log

```bash
# View recent audit entries
tail -f /var/log/jupiter-mesh/compliance.jsonl | python3 -m json.tool

# Count requests by model
cat /var/log/jupiter-mesh/compliance.jsonl | \
  python3 -c "import sys,json; [print(json.loads(l).get('model','?')) for l in sys.stdin]" | \
  sort | uniq -c | sort -rn
```

---

## 7. Troubleshooting

### 7.1 OOM During Training

**Symptom:** `CUDA out of memory` error.

**Solutions (in order):**
1. Reduce `per_device_train_batch_size` in config
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Switch from ZeRO-2 to ZeRO-3 in DeepSpeed config
4. Enable `gradient_checkpointing: true` in model config
5. Reduce `max_position_embeddings` (context length)

### 7.2 Router Load Imbalance

**Symptom:** `expert_load_imbalance_ratio > 0.5` in training logs.

**Solutions:**
1. Increase `router_aux_loss_coeff` (try 0.01 → 0.05)
2. Increase `router_z_loss_coeff` (try 0.001 → 0.01)
3. Check that `expert_capacity_factor` ≥ 1.25
4. Verify batch size is large enough (≥ 512 tokens per expert per step)

### 7.3 Checkpoint Integrity Failure

**Symptom:** `CheckpointIntegrityError: SHA-256 mismatch` on load.

**Solutions:**
1. The checkpoint file is corrupted — do not use it
2. Load from the previous checkpoint: `find_latest_checkpoint(output_dir, skip_latest=1)`
3. If all checkpoints are corrupted, restart training from scratch

### 7.4 Mesh Node Not Appearing in Registry

**Symptom:** Node registered but not visible via `/nodes` endpoint.

**Solutions:**
1. Check node heartbeat: `curl http://node-host:8001/health`
2. Verify `MESH_REGISTRY_URL` is correct and reachable
3. Check registry logs for authentication errors
4. Ensure node's `node_id` is unique across the mesh

---

## 8. Emergency Procedures

### 8.1 Kill Training Run

```bash
# Graceful shutdown (saves checkpoint)
kill -SIGTERM $(pgrep -f train_dense.py)

# Force kill (no checkpoint)
kill -SIGKILL $(pgrep -f train_dense.py)
```

### 8.2 Emergency Inference Rollback

```bash
# Stop current server
kill -SIGTERM $(pgrep -f "inference/server.py")

# Start with previous checkpoint
python inference/server.py \
  --model-path /checkpoints/jupiter-1.3b/step_090000  # previous step
```

### 8.3 Mesh Node Deregistration

```bash
curl -X DELETE http://mesh-registry:9000/nodes/my-node-id
```

---

*Last updated: Month 1 — August 2026*
