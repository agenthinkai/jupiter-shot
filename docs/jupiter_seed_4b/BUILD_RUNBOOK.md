# Jupiter Seed 4B Diagnostic: Build Runbook

**WARNING: DO NOT EXECUTE TRAINING COMMANDS UNTIL WRITTEN AUTHORIZATION IS PROVIDED.**

This runbook defines the exact execution sequence for the Jupiter Seed 4B diagnostic pipeline. Each section provides both Linux/macOS and Windows commands. Windows commands use `.venv\Scripts\python.exe` — do not use `python3` in Windows operator commands.

---

## 1. Environment Setup

**Linux/macOS:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft trl bitsandbytes fastapi uvicorn pydantic pyyaml pytest httpx
```

**Windows (Kishore):**
```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch transformers peft trl bitsandbytes fastapi uvicorn pydantic pyyaml pytest httpx
```

---

## 2. Build Gold Dataset

Generate all 850 examples across train (600), valid (100), and eval (150) splits.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/build_dataset.py \
    --output-dir training/jupiter_seed_4b/data \
    --seed 42
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\build_dataset.py ^
    --output-dir training\jupiter_seed_4b\data ^
    --seed 42
```

---

## 3. Contamination Check and Benchmark Manifest

Verify split isolation and generate the frozen benchmark manifest. The build fails if leakage is detected.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/contamination_check.py \
    --data-dir training/jupiter_seed_4b/data \
    --output benchmarks/jupiter_seed_4b/FROZEN_BENCHMARK_MANIFEST.json
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\contamination_check.py ^
    --data-dir training\jupiter_seed_4b\data ^
    --output benchmarks\jupiter_seed_4b\FROZEN_BENCHMARK_MANIFEST.json
```

---

## 4. Dataset Provenance Validation

Validate the training split and generate the provenance report. The pipeline will abort if any record violates the provenance policy.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/dataset_loader.py \
    --dataset-path training/jupiter_seed_4b/data/train.jsonl \
    --output-path artifacts/provenance_report.json \
    --strict
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\dataset_loader.py ^
    --dataset-path training\jupiter_seed_4b\data\train.jsonl ^
    --output-path artifacts\provenance_report.json ^
    --strict
```

---

## 5. AI-Assisted Arabic Review

Run the heuristic quality filter. This generates a list of examples requiring human review. AI review does NOT equal human approval.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/arabic_review.py \
    --dataset-path training/jupiter_seed_4b/data/train.jsonl \
    --output-path artifacts/arabic_review_results.jsonl \
    --top-n-human 50
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\arabic_review.py ^
    --dataset-path training\jupiter_seed_4b\data\train.jsonl ^
    --output-path artifacts\arabic_review_results.jsonl ^
    --top-n-human 50
```

---

## 6. Cost Estimation (Dry Run)

Generate the cost estimate. **This step must be completed and approved by Farouq in writing before proceeding to step 7.**

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/training_runner.py \
    --config training/jupiter_seed_4b/configs/diagnostic_qlora.yaml \
    --dry-run
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\training_runner.py ^
    --config training\jupiter_seed_4b\configs\diagnostic_qlora.yaml ^
    --dry-run
```

---

## 7. QLoRA Fine-Tuning (Requires Farouq Written Approval)

Execute the training run on the target hardware (RTX 5060 preferred, then Azure credits, then rented GPU).

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/training_runner.py \
    --config training/jupiter_seed_4b/configs/diagnostic_qlora.yaml
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\training_runner.py ^
    --config training\jupiter_seed_4b\configs\diagnostic_qlora.yaml
```

---

## 8. Base vs. Adapted Evaluation

Evaluate both the untouched base model and the trained adapter on the frozen test set.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/evaluate.py \
    --test-set benchmarks/jupiter_seed_4b/test_sets/diagnostic_test.jsonl \
    --base-model Qwen/Qwen3-4B \
    --adapter-path artifacts/seed4b_diagnostic/final_adapter \
    --output artifacts/evaluation_results.json \
    --device cuda
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\evaluate.py ^
    --test-set benchmarks\jupiter_seed_4b\test_sets\diagnostic_test.jsonl ^
    --base-model Qwen/Qwen3-4B ^
    --adapter-path artifacts\seed4b_diagnostic\final_adapter ^
    --output artifacts\evaluation_results.json ^
    --device cuda
```

---

## 9. Quantization Export

Merge the adapter into the base weights and export to INT8 and GGUF (INT4) formats.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/quantize.py \
    --adapter-path artifacts/seed4b_diagnostic/final_adapter \
    --output-dir artifacts/quantized \
    --formats int8 gguf
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\quantize.py ^
    --adapter-path artifacts\seed4b_diagnostic\final_adapter ^
    --output-dir artifacts\quantized ^
    --formats int8 gguf
```

---

## 10. Local Inference Server

Launch the MeshPilot-compatible REST interface for local testing.

**Linux/macOS:**
```bash
python3 training/jupiter_seed_4b/inference_server.py \
    --model-path artifacts/quantized/merged_fp16 \
    --backend transformers \
    --port 8080
```

**Windows (Kishore):**
```cmd
.venv\Scripts\python.exe training\jupiter_seed_4b\inference_server.py ^
    --model-path artifacts\quantized\merged_fp16 ^
    --backend transformers ^
    --port 8080
```
