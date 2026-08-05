# Jupiter Seed 4B: Technical Training Plan

This document outlines the technical process for training Jupiter Seed 4B. This is a design document only; no execution or resource provisioning is authorized during this foundation sprint.

## Training Process

The training pipeline must follow this strict sequence:

1. **Evaluate the untouched student baseline:** Establish the starting metrics on the frozen evaluation benchmark.
2. **Run supervised fine-tuning (SFT):** Utilize LoRA or QLoRA for efficient parameter updates.
3. **Compare with full fine-tuning:** Conduct a limited run of full-parameter fine-tuning to compare efficiency and quality against the LoRA approach.
4. **Preference optimization:** Apply preference optimization techniques (e.g., DPO) *only after* the supervised fine-tuning evaluation is complete and validated.
5. **Validate checkpoint save and resume:** Ensure the training loop can gracefully recover from interruptions.
6. **Quantize to INT8:** Perform 8-bit quantization and measure the resulting memory footprint.
7. **Quantize to INT4:** Perform 4-bit quantization for maximum deployment efficiency.
8. **Benchmark GPU inference:** Measure tokens-per-second and latency on target GPU hardware.
9. **Benchmark CPU inference:** Measure inference performance on CPU architectures utilizing MeshPilot.
10. **Measure quality regression:** Rigorously evaluate the INT8 and INT4 quantized models against the baseline to quantify any quality loss due to quantization.
11. **Generate a complete model card:** Document the final model specifications, training data provenance, evaluation results, and intended use cases.

## Preferred Software Stack

The training infrastructure will rely exclusively on open-source software:

- **Framework:** PyTorch
- **Transformers:** Hugging Face Transformers
- **Parameter-Efficient Fine-Tuning:** PEFT
- **Reinforcement Learning:** TRL
- **Distributed Training:** DeepSpeed or FSDP (where justified by model size or batch requirements)
- **Inference Backends:** vLLM, llama.cpp
- **Model Format:** GGUF
- **Quantization:** AutoGPTQ or AutoAWQ (where compatible with the chosen architecture)
- **Testing:** pytest
- **Evaluation:** lm-evaluation-harness (where appropriate for custom tasks)

## Script Requirements

Every script developed for this pipeline must adhere to the following standards:

| Requirement | Description |
| :--- | :--- |
| **CLI Support** | Must support `--help` for clear usage instructions. |
| **Type Hints** | Must use Python type hints for maintainability. |
| **Configuration** | Must accept explicit configuration files (e.g., YAML) rather than relying solely on command-line arguments. |
| **Determinism** | Must support deterministic seeds for reproducible runs. |
| **Artifacts** | Must produce structured output artifacts (e.g., JSON logs). |
| **Resilience** | Must support checkpoint resume functionality. |
| **Metadata Tracking** | Must record the Git branch, exact commit hash, model version, and a unique run ID for every execution. |
| **Data Integrity** | Must actively reject stale or malformed evidence/data inputs. |
