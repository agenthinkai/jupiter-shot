# Jupiter Seed 4B: Compute and Budget Gate

This document models three experiment sizes for the Jupiter Seed 4B distillation sprint. **No expenditure is authorized by this document.** Any budget allocation requires separate founder approval. Do not consume Azure credits during this sprint.

## 1. Diagnostic Run

**Objective:** Prove pipeline correctness and ensure all scripts execute without error.
**Dataset Size:** 1,000–5,000 examples.
**Training Steps:** Minimal (e.g., 100-500 steps).

| Resource | Estimate |
| :--- | :--- |
| **GPU Type** | 1x A100 (80GB) or equivalent |
| **GPU Count** | 1 |
| **Runtime** | < 2 hours |
| **GPU-Hours** | 2 |
| **Storage** | 50 GB |
| **Estimated Cost** | ~$5 - $10 |
| **Proposed Max Spend** | **USD 500** |
| **Shutdown Conditions** | OOM error, loss divergence (NaN), script failure |
| **Expected Checkpoints**| 2-3 |
| **Failure Allowance** | 3 restarts |
| **Restart Allowance** | Yes |

## 2. Pilot Run

**Objective:** Determine whether the distillation process yields measurable improvement on the held-out evaluation benchmark.
**Dataset Size:** Approximately 25,000 examples.

| Resource | Estimate |
| :--- | :--- |
| **GPU Type** | 4x A100 (80GB) |
| **GPU Count** | 4 |
| **Runtime** | 12 - 24 hours |
| **GPU-Hours** | 48 - 96 |
| **Storage** | 200 GB |
| **Estimated Cost** | ~$100 - $250 |
| **Proposed Max Spend** | Requires separate approval |
| **Shutdown Conditions** | Loss plateaus early, severe evaluation regression |
| **Expected Checkpoints**| Every 500 steps |
| **Failure Allowance** | 2 restarts |
| **Restart Allowance** | Yes |

## 3. Controlled Full Seed Run

**Objective:** Produce the first release candidate of Jupiter Seed 4B.
**Dataset Size:** Up to 100,000 examples.

| Resource | Estimate |
| :--- | :--- |
| **GPU Type** | 8x A100 (80GB) |
| **GPU Count** | 8 |
| **Runtime** | 48 - 72 hours |
| **GPU-Hours** | 384 - 576 |
| **Storage** | 500 GB |
| **Estimated Cost** | ~$800 - $1,500 |
| **Proposed Max Spend** | **USD 1,000 – USD 5,000** (Requires separate founder approval) |
| **Shutdown Conditions** | Validation loss increases significantly, catastrophic hardware failure |
| **Expected Checkpoints**| Every 1,000 steps |
| **Failure Allowance** | 1 restart |
| **Restart Allowance** | Yes, from latest valid checkpoint |
