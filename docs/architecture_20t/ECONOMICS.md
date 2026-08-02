# Jupiter Shot — Economics

> **Status:** Reference estimates — subject to significant uncertainty. All figures are based on 2024 hardware pricing and may change substantially.

---

## 1. Capital Requirements by Stage

| Stage | Model | Hardware | Training Cost | Total Capital |
|-------|-------|----------|---------------|---------------|
| 1 | 1.3B dense | 8× A100 | ~$2,000 | ~$10,000 |
| 2 | 47B MoE | 32× A100 | ~$25,000 | ~$100,000 |
| 3 | 200B MoE | 128× H100 | ~$500,000 | ~$2,000,000 |
| 4 | 1T MoE | 512× H100 NVL | ~$5,000,000 | ~$20,000,000 |
| 5 | 5T MoE | 2,048× H100 NVL | ~$50,000,000 | ~$200,000,000 |
| 6 | 20T MoE | 8,192× H100 NVL | ~$500,000,000 | ~$2,100,000,000 |

**Column definitions:**
- **Training Cost**: Estimated GPU compute cost for the training run only (spot pricing where available)
- **Total Capital**: Includes hardware procurement (if owned), data center, networking, staffing, and 12 months of operating costs

**Important caveats:**
1. These are order-of-magnitude estimates, not financial projections. Actual costs depend on hardware procurement strategy (owned vs. leased vs. cloud), negotiated pricing, and operational efficiency.
2. Stage 6 capital ($2.1B) assumes hardware ownership. Cloud-only deployment would cost significantly more per training run but requires less upfront capital.
3. Hardware costs are based on 2024 H100 pricing (~$30,000/GPU list price, ~$15,000–$20,000 negotiated). Future hardware (H200, B100, B200) may offer better price/performance.
4. The $2.1B figure is not a fundraising target. It is a reference point to illustrate the capital intensity of frontier model training.

---

## 2. Unit Economics (Inference)

### 2.1 Cost Per Token by Tier

| Tier | Model | Active Params | GPU-hours/1M tokens | Cost/1M tokens |
|------|-------|---------------|---------------------|----------------|
| T1 | 1.3B | 1.3B | ~0.01 | ~$0.05 |
| T2 | 47B MoE | 7B | ~0.05 | ~$0.20 |
| T3 | 200B MoE | 25B | ~0.20 | ~$0.80 |
| T4 | 1T MoE | 62B | ~0.75 | ~$3.00 |
| T5 | 20T MoE | 625B | ~6.25 | ~$25.00 |

**Assumptions:** $4/GPU-hour (H100 on-demand), 80% GPU utilization, 50% model FLOPs utilization (MFU).

### 2.2 Revenue Model

The AgenThinkMesh monetizes through:
1. **API access**: Per-token pricing for external developers
2. **Enterprise licensing**: Fixed monthly fee for dedicated capacity
3. **Sovereign deployment**: One-time licensing fee + annual support for on-premises deployments
4. **Fine-tuning services**: Custom model adaptation for specific domains

### 2.3 Break-Even Analysis (Stage 2)

For Stage 2 (47B MoE) to break even on training cost ($25,000):
- At $0.20/1M tokens: 125B tokens of API revenue
- At 1M tokens/day average usage: 125 days to break even
- At 10M tokens/day: 12.5 days to break even

Stage 2 break-even is achievable with modest API usage. Stage 6 break-even requires significant enterprise contracts.

---

## 3. Hardware Procurement Strategy

### 3.1 Stages 1–2: Cloud Spot Instances

For Stages 1 and 2, cloud spot instances (AWS p4d, GCP A3, Azure NDv4) are the most cost-effective option. Spot pricing is typically 60–70% below on-demand pricing.

**Risk**: Spot instance preemption. The checkpoint system (`training/checkpoint.py`) is designed to handle preemption with < 30-minute data loss.

### 3.2 Stages 3–4: Hybrid (Cloud + Owned)

For Stages 3 and 4, a hybrid approach is recommended:
- **Training**: Cloud reserved instances (1-year commitment, ~40% discount vs on-demand)
- **Inference**: Owned hardware for predictable workloads, cloud for burst

### 3.3 Stages 5–6: Sovereign Hardware Ownership

For sovereign deployment at Stage 5 and beyond, hardware ownership is required:
- Cloud providers cannot guarantee data residency at the required level
- Long-term total cost of ownership (TCO) favors ownership at this scale
- Hardware procurement requires 6–12 months lead time for large orders

---

## 4. Sensitivity Analysis

The Stage 6 capital estimate is sensitive to:

| Variable | Base Case | Optimistic | Pessimistic |
|----------|-----------|------------|-------------|
| GPU price (H100 NVL) | $25,000 | $15,000 | $35,000 |
| GPU count | 8,192 | 6,000 | 12,000 |
| Training duration | 365 days | 270 days | 540 days |
| Staff (100 engineers) | $20M/year | $15M/year | $30M/year |
| **Total capital** | **$2.1B** | **$1.1B** | **$4.2B** |

The 4× range between optimistic and pessimistic scenarios reflects the genuine uncertainty in frontier model training costs. Any business plan based on these figures should use the pessimistic scenario for financial planning.

---

## 5. What This Document Is Not

This document does not constitute:
- A financial projection or forecast
- An investment prospectus
- A commitment to raise capital
- A guarantee of any return on investment

The economics of frontier AI model training are highly uncertain. The figures in this document are reference estimates to support architectural decision-making, not financial planning.

---

*Last updated: 2026-08-02*
*Status: Reference estimates — not financial projections*
