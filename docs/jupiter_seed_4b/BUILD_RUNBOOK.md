# Jupiter Seed 4B Diagnostic: Build Runbook

**WARNING: DO NOT EXECUTE THESE COMMANDS UNTIL WRITTEN AUTHORIZATION IS PROVIDED.**

This runbook defines the exact execution sequence for the Jupiter Seed 4B diagnostic pipeline.

## 1. Environment Setup

Ensure the required dependencies are installed in a Python 3.10/3.11 environment:

```bash
pip install torch transformers peft trl bitsandbytes fastapi uvicorn pydantic pyyaml pytest
```

## 2. Dataset Provenance Loading

Validate the seed dataset (maximum 1,000 examples) and generate the provenance report. The pipeline will abort if any record violates the provenance policy.

```bash
python3 training/jupiter_seed_4b/dataset_loader.py \
    --dataset-path data/diagnostic_seed.jsonl \
    --output-path artifacts/provenance_report.json \
    --strict
```

## 3. AI-Assisted Arabic Review

Run the heuristic quality filter. This generates a list of examples requiring human review.

```bash
python3 training/jupiter_seed_4b/arabic_review.py \
    --dataset-path data/diagnostic_seed.jsonl \
    --output-path artifacts/arabic_review_results.jsonl \
    --top-n-human 50
```

## 4. Cost Estimation (Dry Run)

Generate the cost estimate. **This step must be completed and approved by Farouq before proceeding to step 5.**

```bash
python3 training/jupiter_seed_4b/training_runner.py \
    --config training/jupiter_seed_4b/configs/diagnostic_qlora.yaml \
    --dry-run
```

## 5. QLoRA Fine-Tuning

Execute the training run on the target hardware (e.g., RTX 5060 or 1x A100).

```bash
python3 training/jupiter_seed_4b/training_runner.py \
    --config training/jupiter_seed_4b/configs/diagnostic_qlora.yaml
```

## 6. Base vs. Adapted Evaluation

Evaluate both the untouched base model and the newly trained adapter on the frozen test set.

```bash
python3 training/jupiter_seed_4b/evaluate.py \
    --test-set benchmarks/jupiter_seed_4b/test_sets/diagnostic_test.jsonl \
    --base-model Qwen/Qwen3-4B \
    --adapter-path artifacts/seed4b_diagnostic/final_adapter \
    --output artifacts/evaluation_results.json \
    --device cuda
```

## 7. Quantization Export

Merge the adapter into the base weights and export to INT8 and GGUF (INT4) formats.

```bash
python3 training/jupiter_seed_4b/quantize.py \
    --adapter-path artifacts/seed4b_diagnostic/final_adapter \
    --output-dir artifacts/quantized \
    --formats int8 gguf
```

## 8. Local Inference Server

Launch the MeshPilot-compatible REST interface for local testing.

```bash
python3 training/jupiter_seed_4b/inference_server.py \
    --model-path artifacts/quantized/merged_fp16 \
    --backend transformers \
    --port 8080
```
