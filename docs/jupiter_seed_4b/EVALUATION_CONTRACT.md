# Jupiter Seed 4B: Evaluation Contract

This document defines the strict evaluation protocols that must be established before generating the complete distillation dataset. An evaluation-first approach ensures that model improvements are measured against rigorous, held-out tasks rather than merely relying on training loss.

## Benchmark Construction

Before training begins, a frozen evaluation benchmark must be created in `benchmarks/jupiter_seed_4b/`. This benchmark will contain held-out tasks specifically designed to test the model's performance across critical enterprise domains.

### Required Task Coverage
The benchmark must include comprehensive evaluations for the following areas:
- Modern Standard Arabic and GCC business Arabic proficiency
- Arabic-English code-switching capabilities
- English enterprise writing standards
- Islamic-finance concepts and terminology
- GCC banking document comprehension
- Telecom operations analysis
- Energy and logistics sector reasoning
- Regulatory-document interpretation
- Numerical reasoning accuracy
- Evidence-grounded answer generation
- Refusal and uncertainty behavior (knowing when to decline or state uncertainty)
- Hallucination resistance metrics
- Decision-analysis quality

## Evaluation Protocol Definition

The evaluation framework must strictly adhere to the following defined parameters to ensure validity and reproducibility.

| Protocol Element | Definition Requirement |
| :--- | :--- |
| **Benchmark sources** | Exact origin of the evaluation data must be documented. |
| **Data licences** | Verification that the evaluation data can be legally used for benchmarking. |
| **Human-review protocol** | The specific process and rubrics used by human reviewers to grade outputs. |
| **Baseline models** | The specific, untouched student models and competitor models used for comparison. |
| **Metrics** | The exact statistical and qualitative metrics used to score performance. |
| **Pass thresholds** | The minimum required score to consider a model version successful. |
| **Failure thresholds** | The score at which a model version is deemed a regression or failure. |
| **Contamination checks** | Automated processes to guarantee evaluation data is entirely absent from the training set. |
| **Statistical uncertainty** | Methods for calculating confidence intervals and margin of error in the results. |
| **Reproducibility requirements** | Ensuring that the evaluation script produces identical results given the same model and seed. |

## Strict Prohibition on Training Loss Claims
It is strictly prohibited to claim model improvement based solely on a reduction in training loss. All claims of advancement must be substantiated by statistically significant improvements on the frozen evaluation benchmark.
