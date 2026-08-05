# Jupiter Seed 4B: Arabic Reviewer Plan

This document outlines the human-review requirements necessary to guarantee the Arabic quality of Jupiter Seed 4B. Claims of Arabic quality are strictly prohibited without documented native-speaker evaluation.

## Human-Review Requirements

The review team must evaluate model outputs and dataset samples across the following dimensions:

- **Modern Standard Arabic (MSA):** Grammatical correctness, spelling, and formal tone.
- **Gulf/Khaliji business Arabic:** Regional nuances, appropriate professional terminology, and cultural context.
- **Islamic-finance terminology:** Exactness in translating and utilizing Sharia-compliant financial concepts.
- **Banking terminology:** Accuracy in standard banking and regulatory language.
- **Telecom terminology:** Correct usage of telecommunications and network operations terms.
- **Arabic-English code switching:** Natural and accurate transitioning between Arabic and English within a single context, common in GCC enterprise environments.
- **Safety and cultural quality:** Ensuring outputs are culturally sensitive, appropriate for the GCC region, and free from toxicity or bias.

## Resource Estimation and Workflow

| Metric | Estimate |
| :--- | :--- |
| **Number of reviewers** | 3 - 5 |
| **Required qualifications** | Native Arabic speakers (preferably GCC nationals), bilingual (English), with domain expertise in finance or telecom. |
| **Examples reviewed per person per day** | 100 - 150 (depending on length and complexity) |
| **Stage A completion time** | 2 - 3 weeks (for 1,000 - 2,000 examples) |
| **Compensation assumptions** | Market rate for specialized linguistic/domain consulting in the GCC. |

## Quality Assurance Process

- **Inter-reviewer agreement method:** A subset of examples (e.g., 10%) will be independently scored by at least two reviewers. Cohen's kappa or a similar metric will be used to measure agreement.
- **Escalation process:** Disagreements on terminology or safety will be escalated to a designated lead reviewer or domain expert for final arbitration.
- **Acceptance thresholds:** A minimum score of 4/5 on the established rubric is required for an example to be included in the Gold Seed Set (Stage A). Any example flagged for safety or cultural issues will be immediately rejected.
