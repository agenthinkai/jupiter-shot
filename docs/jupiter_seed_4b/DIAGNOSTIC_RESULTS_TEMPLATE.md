# Jupiter Seed 4B: Diagnostic Results

**Run ID:** _______________________
**Date:** _________________________
**Hardware Used:** ________________

## 1. Pipeline Execution Status

| Component | Status (Pass/Fail) | Notes |
| :--- | :--- | :--- |
| Dataset Loader (Provenance 100%) | | |
| AI Arabic Review Execution | | |
| QLoRA Training (No NaN/OOM) | | |
| Checkpoint Save & Resume | | |
| Quantization (INT8 & GGUF) | | |
| Local Inference Server Start | | |

## 2. Evaluation Comparison

*Compare the adapted model against the untouched base model (`Qwen/Qwen3-4B`).*

| Dimension | Base Score | Adapted Score | Delta |
| :--- | :--- | :--- | :--- |
| Arabic Instruction Following | | | |
| English Instruction Following | | | |
| Arabic-English Translation | | | |
| GCC Banking Terminology | | | |
| Islamic Finance Terminology | | | |
| Telecom Scenarios | | | |
| Regulatory Summarization | | | |
| Enterprise Correspondence | | | |
| Hallucination Resistance | | | |
| Safety & Refusal Behavior | | | |

## 3. Performance Metrics

| Metric | Base Model (FP16) | Adapted (Merged FP16) | Adapted (INT8) | Adapted (INT4 GGUF) |
| :--- | :--- | :--- | :--- | :--- |
| Latency (ms) | | | | |
| Tokens per second | | | | |
| Peak VRAM (MB) | | | | |
| CPU RAM (MB) | | | | |
| Model Size (MB) | | | | |

## 4. Financial Ledger Summary

- **Authorized Budget:** USD 500.00
- **Actual Compute Spend:** USD _________
- **Status:** [ ] Under Budget   [ ] Over Budget (Explain below)

## 5. Conclusion

**Diagnostic Verdict:** [ ] SUCCESS   [ ] FAILURE
*(Success requires all pipeline components to pass, demonstrated domain improvement without severe general regression, and adherence to the budget).*
