# Jupiter Seed 4B: Diagnostic Experiment Design

**STATUS: DESIGN ONLY. DO NOT EXECUTE.**
**The diagnostic must not begin until Farouq provides explicit written approval after legal and Arabic-review gates are complete.**

This document outlines the first diagnostic experiment for Jupiter Seed 4B. The objective is strictly to validate the technical pipeline (data loading, training loop, checkpointing, and evaluation) without incurring significant compute costs.

## Experiment Parameters

- **Candidate student:** `Qwen/Qwen3-4B` (Provisional, pending legal approval)
- **Candidate teacher:** `Qwen/Qwen2.5-72B-Instruct` (Provisional, pending legal approval)
- **Maximum examples:** 1,000 (Sampled from the Stage A Gold Seed Set)
- **Maximum tokens:** ~2,000,000 (Assuming ~2k tokens per example)
- **Expected GPU type and hours:** 1x A100 (80GB) for < 2 hours
- **Hard spending limit:** USD 500

## Prerequisites

Before execution is authorized, the following gates must be cleared:
1. **Licence approval prerequisites:** The selected student and teacher models must be marked APPROVED in the `LICENSE_AND_DISTILLATION_MATRIX.md` by qualified human legal counsel.
2. **Arabic-review prerequisites:** The 1,000 examples used for this diagnostic must have passed the human review process defined in `ARABIC_REVIEWER_PLAN.md`.

## Execution Controls

- **Stop conditions:** 
  - Hardware failure (OOM).
  - Training loss divergence (NaN).
  - Spend approaches the $500 hard limit.
- **Data-provenance records:** Every example used must have a complete metadata record as defined in the `DATA_PROVENANCE_POLICY.md`.

## Evaluation & Criteria

- **Evaluation benchmarks:** A subset of the frozen evaluation benchmark (`EVALUATION_CONTRACT.md`) will be run to ensure the evaluation script executes correctly.
- **Success criteria:** 
  - The training script completes without error.
  - Checkpoints are successfully saved and can be resumed.
  - The evaluation script runs and outputs valid metrics.
- **Failure criteria:** 
  - Pipeline crashes.
  - Data provenance records are corrupted or incomplete.
  - Cost exceeds $500.
