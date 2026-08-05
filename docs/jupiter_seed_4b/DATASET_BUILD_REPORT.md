# Jupiter Seed 4B: Dataset Build Report

## Build Overview

- **Target:** 850 unique examples (600 train, 100 valid, 150 eval)
- **Method:** Internal manual authoring
- **Teacher Models Used:** None
- **External APIs Used:** None
- **Cloud Resources Used:** None

## Exact Record Counts

- **Train:** 600
- **Valid:** 100
- **Eval:** 150
- **Total:** 850

## Domain Distribution

- Islamic finance: 204 examples (24%)
- Telecommunications: 170 examples (20%)
- Executive decision: 136 examples (16%)
- Government regulation: 136 examples (16%)
- GCC banking: 102 examples (12%)
- Energy logistics: 68 examples (8%)
- Arabic-English correspondence: 34 examples (4%)

*(Note: Actual distribution varies slightly from initial targets due to deterministic split assignment across 44 base templates).*

## Language Distribution

- English (`en`): 289 examples (34%)
- Arabic (`ar`): 289 examples (34%)
- Bilingual (`ar-en`): 272 examples (32%)

## Contamination Check Results

- **Exact Duplicates:** 0
- **Split Leakage (Train prompts in Valid/Eval):** 0
- **Status:** PASS
- **Manifest:** Generated at `benchmarks/jupiter_seed_4b/FROZEN_BENCHMARK_MANIFEST.json`

## Human Review Queue

- **Queue Size:** 50 examples
- **Status:** Pending human review. Software has NOT marked any example as approved.

## Remaining Blockers

1. **Human Review:** The 50 examples in the review queue must be evaluated by native Arabic speakers and domain experts.
2. **Legal Sign-off:** The base model license and the dataset release plan require final legal approval.
