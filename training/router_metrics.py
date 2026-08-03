"""
Jupiter Shot — Shared Router Metrics Schema
============================================

This module is the **single source of truth** for every key name, unit, and
mathematical definition used in MoE router metrics.  It is imported by:

  - training/models/moe.py          (TopKRouter.forward → emit_router_metrics)
  - scripts/run_laptop_moe.py       (aggregate_layer_metrics, acceptance eval)
  - scripts/collect_laptop_metrics.py
  - tests/test_runner_integration.py
  - tests/test_acceptance.py

Do NOT define router metric key names anywhere else.  If a key is missing from
the model output, the acceptance system must return NOT_EVALUABLE — never a
vacuous PASS.

──────────────────────────────────────────────────────────────────────────────
SCHEMA DEFINITIONS
──────────────────────────────────────────────────────────────────────────────

For 8 experts, top-2 routing, N tokens per batch:

  Total expert assignments = N × top_k  (each token selects 2 experts)
  Balanced assignment fraction per expert = top_k / num_experts = 2/8 = 0.25
  Balanced token routing fraction per expert = 1 / num_experts = 0.125

Key                          Type      Unit / Definition
─────────────────────────────────────────────────────────────────────────────
router_entropy               float     nats  — Shannon entropy of the mean
                                       routing probability distribution across
                                       all tokens.
                                       H = -Σ p_i · ln(p_i)
                                       where p_i = mean softmax prob for expert i
                                       over all tokens in the batch.
                                       Maximum (uniform): ln(num_experts)
                                       Minimum (collapsed): 0

expert_assignment_counts     list[int] Raw count of how many times each expert
                                       was selected (summed over top-k slots).
                                       Length = num_experts.
                                       Sum = N × top_k.

expert_assignment_fractions  list[float] Fraction of total assignments going to
                                       each expert.
                                       expert_assignment_fractions[i] =
                                         expert_assignment_counts[i] / (N × top_k)
                                       Balanced value = top_k / num_experts = 0.25
                                       for 8 experts, top-2.

token_routing_fractions      list[float] Fraction of tokens that were routed to
                                       each expert at least once.
                                       token_routing_fractions[i] =
                                         tokens_routed_to_expert_i / N
                                       Balanced value = top_k / num_experts = 0.25
                                       (same as assignment fraction for top-1;
                                       differs for top-k > 1 when a token can
                                       select the same expert twice, which is
                                       prevented by topk on distinct logits).

minimum_expert_fraction      float     min(expert_assignment_fractions)
maximum_expert_fraction      float     max(expert_assignment_fractions)

utilization_mean             float     mean(expert_assignment_fractions)
                                       Always equals top_k / num_experts.

utilization_std              float     std(expert_assignment_fractions, ddof=0)

utilization_cv               float     utilization_std / (utilization_mean + ε)
                                       Coefficient of variation.
                                       0 = perfectly balanced.
                                       Acceptance threshold: < 0.5

maximum_to_minimum_ratio     float     maximum_expert_fraction /
                                         (minimum_expert_fraction + ε)
                                       1.0 = perfectly balanced.

number_of_inactive_experts   int       Count of experts where
                                         expert_assignment_fractions[i] < 0.01
                                       (below 1% of total assignments).
                                       Acceptance threshold: 0

inactive_expert_indices      list[int] Indices of experts below the 1% threshold.

dropped_token_count          int       Tokens that exceeded expert capacity and
                                       were dropped (not processed by any expert).
                                       0 if capacity_factor is large enough.

dropped_token_fraction       float     dropped_token_count / N
                                       Acceptance threshold: < 0.01 (< 1%)

overflow_token_count         int       Tokens that hit the capacity buffer but
                                       were processed by a fallback mechanism.
                                       0 if no overflow buffer is used.

overflow_token_fraction      float     overflow_token_count / N

auxiliary_load_balancing_loss float    Switch Transformer auxiliary loss:
                                       aux_loss = num_experts × Σ(f_i × P_i)
                                       where f_i = expert_assignment_fractions[i]
                                             P_i = mean_prob_per_expert[i]
                                       Scaled by aux_loss_coeff before returning.

router_z_loss                float     z_loss = mean(log(Σ exp(logits))²)
                                       Scaled by z_loss_coeff before returning.
                                       Prevents logit explosion.

capacity_factor              float     Expert capacity multiplier.
                                       capacity = ceil(capacity_factor × N × top_k
                                                       / num_experts)

experts_selected_per_token   int       top_k value used in this forward pass.

number_of_experts            int       Total number of experts.
─────────────────────────────────────────────────────────────────────────────

MISSING METRIC POLICY
─────────────────────
If any REQUIRED key is absent from the metrics dict, the acceptance evaluator
must return outcome=NOT_EVALUABLE with an explicit error message listing the
missing keys.  It must never return PASS or NOT_ACCEPTED when data is missing.

REQUIRED keys for acceptance evaluation:
  router_entropy
  expert_assignment_fractions
  minimum_expert_fraction
  maximum_expert_fraction
  utilization_cv
  number_of_inactive_experts
  inactive_expert_indices
  dropped_token_fraction
  auxiliary_load_balancing_loss
  experts_selected_per_token
  number_of_experts
"""

from __future__ import annotations

import math
from typing import Any

# ── Key name constants ────────────────────────────────────────────────────────
# Import these in every consumer instead of writing string literals.

K_ROUTER_ENTROPY             = "router_entropy"
K_EXPERT_ASSIGNMENT_COUNTS   = "expert_assignment_counts"
K_EXPERT_ASSIGNMENT_FRACTIONS= "expert_assignment_fractions"
K_TOKEN_ROUTING_FRACTIONS    = "token_routing_fractions"
K_MINIMUM_EXPERT_FRACTION    = "minimum_expert_fraction"
K_MAXIMUM_EXPERT_FRACTION    = "maximum_expert_fraction"
K_UTILIZATION_MEAN           = "utilization_mean"
K_UTILIZATION_STD            = "utilization_std"
K_UTILIZATION_CV             = "utilization_cv"
K_MAX_MIN_RATIO              = "maximum_to_minimum_ratio"
K_NUM_INACTIVE_EXPERTS       = "number_of_inactive_experts"
K_INACTIVE_EXPERT_INDICES    = "inactive_expert_indices"
K_DROPPED_TOKEN_COUNT        = "dropped_token_count"
K_DROPPED_TOKEN_FRACTION     = "dropped_token_fraction"
K_OVERFLOW_TOKEN_COUNT       = "overflow_token_count"
K_OVERFLOW_TOKEN_FRACTION    = "overflow_token_fraction"
K_AUX_LOAD_BALANCING_LOSS    = "auxiliary_load_balancing_loss"
K_ROUTER_Z_LOSS              = "router_z_loss"
K_CAPACITY_FACTOR            = "capacity_factor"
K_EXPERTS_PER_TOKEN          = "experts_selected_per_token"
K_NUM_EXPERTS                = "number_of_experts"

# Keys required for acceptance evaluation
REQUIRED_ACCEPTANCE_KEYS: tuple[str, ...] = (
    K_ROUTER_ENTROPY,
    K_EXPERT_ASSIGNMENT_FRACTIONS,
    K_MINIMUM_EXPERT_FRACTION,
    K_MAXIMUM_EXPERT_FRACTION,
    K_UTILIZATION_CV,
    K_NUM_INACTIVE_EXPERTS,
    K_INACTIVE_EXPERT_INDICES,
    K_DROPPED_TOKEN_FRACTION,
    K_AUX_LOAD_BALANCING_LOSS,
    K_EXPERTS_PER_TOKEN,
    K_NUM_EXPERTS,
)

# ── Acceptance thresholds ─────────────────────────────────────────────────────

ACCEPT_MAX_DROPPED_TOKEN_FRACTION = 0.01   # < 1%
ACCEPT_MAX_UTILIZATION_CV         = 0.5    # CV < 0.5
ACCEPT_MIN_ROUTER_ENTROPY_NATS    = 1.0    # > 1.0 nats (not bits)
ACCEPT_MIN_EXPERT_FRACTION        = 0.01   # no expert below 1% of assignments
ACCEPT_MAX_INACTIVE_EXPERTS       = 0      # zero inactive experts

# ── Outcome codes ─────────────────────────────────────────────────────────────

OUTCOME_PASS             = "PASS"
OUTCOME_NOT_ACCEPTED     = "NOT_ACCEPTED"
OUTCOME_NOT_EVALUABLE    = "NOT_EVALUABLE"
OUTCOME_EXECUTION_ERROR  = "EXECUTION_ERROR"
OUTCOME_SAFETY_STOP      = "SAFETY_STOP"


# ── Metric computation ────────────────────────────────────────────────────────

def compute_router_metrics(
    *,
    expert_assignment_counts: list[int],
    num_tokens: int,
    num_experts: int,
    top_k: int,
    router_probs_mean: list[float],
    aux_loss_unscaled: float,
    aux_loss_coeff: float,
    z_loss_unscaled: float,
    z_loss_coeff: float,
    capacity_factor: float,
    dropped_token_count: int = 0,
    overflow_token_count: int = 0,
) -> dict[str, Any]:
    """
    Compute the full router metrics dict from raw routing statistics.

    This is the canonical computation function.  Call it from TopKRouter.forward
    and from any test that needs to construct a reference metrics dict.

    Args:
        expert_assignment_counts: Raw count of assignments per expert.
            Length must equal num_experts.
            Sum must equal num_tokens × top_k.
        num_tokens: Total tokens in the batch (batch_size × seq_len).
        num_experts: Total number of experts.
        top_k: Number of experts selected per token.
        router_probs_mean: Mean softmax probability per expert across all tokens.
            Length must equal num_experts.  Used for entropy computation.
        aux_loss_unscaled: Auxiliary load-balancing loss before scaling.
        aux_loss_coeff: Coefficient applied to aux_loss_unscaled.
        z_loss_unscaled: Router-z loss before scaling.
        z_loss_coeff: Coefficient applied to z_loss_unscaled.
        capacity_factor: Expert capacity multiplier.
        dropped_token_count: Tokens dropped due to capacity overflow.
        overflow_token_count: Tokens processed via overflow buffer.

    Returns:
        Dict with all keys defined in this module's schema.
    """
    if len(expert_assignment_counts) != num_experts:
        raise ValueError(
            f"expert_assignment_counts length {len(expert_assignment_counts)} "
            f"!= num_experts {num_experts}"
        )
    total_assignments = num_tokens * top_k
    fractions = [
        c / (total_assignments + 1e-9)
        for c in expert_assignment_counts
    ]
    # token_routing_fractions: for top-k routing without replacement,
    # this equals expert_assignment_fractions (each token selects each expert
    # at most once since topk operates on distinct logits).
    token_routing_fractions = fractions[:]

    min_frac = min(fractions)
    max_frac = max(fractions)
    mean_frac = sum(fractions) / num_experts
    variance = sum((f - mean_frac) ** 2 for f in fractions) / num_experts
    std_frac = math.sqrt(variance)
    cv = std_frac / (mean_frac + 1e-9)
    max_min_ratio = max_frac / (min_frac + 1e-9)

    inactive_indices = [i for i, f in enumerate(fractions) if f < ACCEPT_MIN_EXPERT_FRACTION]

    # Shannon entropy in nats from mean router probabilities
    entropy_nats = 0.0
    for p in router_probs_mean:
        if p > 1e-12:
            entropy_nats -= p * math.log(p)

    return {
        K_ROUTER_ENTROPY:              round(entropy_nats, 6),
        K_EXPERT_ASSIGNMENT_COUNTS:    [int(c) for c in expert_assignment_counts],
        K_EXPERT_ASSIGNMENT_FRACTIONS: [round(f, 6) for f in fractions],
        K_TOKEN_ROUTING_FRACTIONS:     [round(f, 6) for f in token_routing_fractions],
        K_MINIMUM_EXPERT_FRACTION:     round(min_frac, 6),
        K_MAXIMUM_EXPERT_FRACTION:     round(max_frac, 6),
        K_UTILIZATION_MEAN:            round(mean_frac, 6),
        K_UTILIZATION_STD:             round(std_frac, 6),
        K_UTILIZATION_CV:              round(cv, 6),
        K_MAX_MIN_RATIO:               round(max_min_ratio, 6),
        K_NUM_INACTIVE_EXPERTS:        len(inactive_indices),
        K_INACTIVE_EXPERT_INDICES:     inactive_indices,
        K_DROPPED_TOKEN_COUNT:         dropped_token_count,
        K_DROPPED_TOKEN_FRACTION:      round(dropped_token_count / (num_tokens + 1e-9), 6),
        K_OVERFLOW_TOKEN_COUNT:        overflow_token_count,
        K_OVERFLOW_TOKEN_FRACTION:     round(overflow_token_count / (num_tokens + 1e-9), 6),
        K_AUX_LOAD_BALANCING_LOSS:     round(aux_loss_unscaled * aux_loss_coeff, 8),
        K_ROUTER_Z_LOSS:               round(z_loss_unscaled * z_loss_coeff, 8),
        K_CAPACITY_FACTOR:             capacity_factor,
        K_EXPERTS_PER_TOKEN:           top_k,
        K_NUM_EXPERTS:                 num_experts,
    }


def aggregate_layer_metrics(layer_metrics_list: list[dict]) -> dict[str, Any]:
    """
    Aggregate per-layer router metrics from a forward pass into a single dict.

    Scalar metrics are averaged across layers.
    List metrics (fractions, counts) are summed element-wise then normalised.
    Missing layers (empty dicts) are skipped.

    Args:
        layer_metrics_list: List of per-layer metrics dicts from model forward.

    Returns:
        Aggregated metrics dict with the same schema, or empty dict if no
        valid layers are present.
    """
    valid = [m for m in layer_metrics_list if isinstance(m, dict) and m]
    if not valid:
        return {}

    n = len(valid)

    # Scalar keys: average
    scalar_keys = (
        K_ROUTER_ENTROPY,
        K_MINIMUM_EXPERT_FRACTION,
        K_MAXIMUM_EXPERT_FRACTION,
        K_UTILIZATION_MEAN,
        K_UTILIZATION_STD,
        K_UTILIZATION_CV,
        K_MAX_MIN_RATIO,
        K_DROPPED_TOKEN_FRACTION,
        K_OVERFLOW_TOKEN_FRACTION,
        K_AUX_LOAD_BALANCING_LOSS,
        K_ROUTER_Z_LOSS,
        K_CAPACITY_FACTOR,
        K_EXPERTS_PER_TOKEN,
        K_NUM_EXPERTS,
    )
    # Integer keys: sum
    int_keys = (
        K_NUM_INACTIVE_EXPERTS,
        K_DROPPED_TOKEN_COUNT,
        K_OVERFLOW_TOKEN_COUNT,
    )

    result: dict[str, Any] = {}

    for key in scalar_keys:
        vals = [m[key] for m in valid if key in m and isinstance(m[key], (int, float))]
        if vals:
            result[key] = round(sum(vals) / len(vals), 6)

    for key in int_keys:
        vals = [m[key] for m in valid if key in m and isinstance(m[key], int)]
        if vals:
            result[key] = sum(vals)

    # List keys: element-wise average
    for list_key in (K_EXPERT_ASSIGNMENT_FRACTIONS, K_TOKEN_ROUTING_FRACTIONS):
        lists = [m[list_key] for m in valid if list_key in m and isinstance(m[list_key], list)]
        if lists:
            num_exp = len(lists[0])
            result[list_key] = [
                round(sum(lst[i] for lst in lists if i < len(lst)) / len(lists), 6)
                for i in range(num_exp)
            ]

    # Counts: element-wise sum
    count_lists = [
        m[K_EXPERT_ASSIGNMENT_COUNTS]
        for m in valid
        if K_EXPERT_ASSIGNMENT_COUNTS in m and isinstance(m[K_EXPERT_ASSIGNMENT_COUNTS], list)
    ]
    if count_lists:
        num_exp = len(count_lists[0])
        result[K_EXPERT_ASSIGNMENT_COUNTS] = [
            sum(lst[i] for lst in count_lists if i < len(lst))
            for i in range(num_exp)
        ]

    # Inactive indices: union across layers
    all_inactive: set[int] = set()
    for m in valid:
        if K_INACTIVE_EXPERT_INDICES in m and isinstance(m[K_INACTIVE_EXPERT_INDICES], list):
            all_inactive.update(m[K_INACTIVE_EXPERT_INDICES])
    if all_inactive or any(K_INACTIVE_EXPERT_INDICES in m for m in valid):
        result[K_INACTIVE_EXPERT_INDICES] = sorted(all_inactive)
        result[K_NUM_INACTIVE_EXPERTS] = len(all_inactive)

    return result


def check_required_keys(metrics: dict) -> list[str]:
    """Return a list of required keys that are missing from metrics."""
    return [k for k in REQUIRED_ACCEPTANCE_KEYS if k not in metrics]


def evaluate_acceptance(
    aggregated_metrics: dict,
    window_description: str = "last-10-steps",
    inactive_threshold: float = ACCEPT_MIN_EXPERT_FRACTION,
) -> dict[str, Any]:
    """
    Evaluate MoE router acceptance against the defined thresholds.

    Returns a dict with:
      outcome: one of OUTCOME_* constants
      criteria: dict of per-criterion results
      missing_keys: list of keys that were absent (populated when NOT_EVALUABLE)
      errors: list of human-readable error strings
      window: description of the evaluation window
      thresholds: dict of thresholds used

    A missing required metric produces outcome=NOT_EVALUABLE.
    It never produces PASS or NOT_ACCEPTED.
    """
    missing = check_required_keys(aggregated_metrics)
    if missing:
        return {
            "outcome": OUTCOME_NOT_EVALUABLE,
            "criteria": {},
            "missing_keys": missing,
            "errors": [
                f"Required router metric '{k}' is absent. "
                f"Cannot evaluate acceptance without it."
                for k in missing
            ],
            "window": window_description,
            "thresholds": _threshold_dict(inactive_threshold),
        }

    num_experts = aggregated_metrics[K_NUM_EXPERTS]
    top_k = aggregated_metrics[K_EXPERTS_PER_TOKEN]
    fractions = aggregated_metrics[K_EXPERT_ASSIGNMENT_FRACTIONS]
    inactive_indices = aggregated_metrics[K_INACTIVE_EXPERT_INDICES]
    cv = aggregated_metrics[K_UTILIZATION_CV]
    entropy = aggregated_metrics[K_ROUTER_ENTROPY]
    dropped_frac = aggregated_metrics[K_DROPPED_TOKEN_FRACTION]
    num_inactive = aggregated_metrics[K_NUM_INACTIVE_EXPERTS]

    balanced_fraction = top_k / num_experts
    errors: list[str] = []

    # Per-criterion evaluation
    criteria: dict[str, Any] = {}

    # 1. No inactive experts
    inactive_pass = num_inactive == 0
    criteria["no_inactive_experts"] = {
        "pass": inactive_pass,
        "threshold": f"all experts >= {inactive_threshold * 100:.1f}% of assignments",
        "measured": {
            f"expert_{i}": round(fractions[i], 4)
            for i in range(len(fractions))
        },
        "inactive_experts": inactive_indices,
        "num_inactive": num_inactive,
    }
    if not inactive_pass:
        errors.append(
            f"Inactive experts detected: {inactive_indices} "
            f"(fractions: {[round(fractions[i], 4) for i in inactive_indices]}). "
            f"Threshold: >= {inactive_threshold * 100:.1f}% per expert."
        )

    # 2. Utilization CV
    cv_pass = cv < ACCEPT_MAX_UTILIZATION_CV
    criteria["utilization_cv_acceptable"] = {
        "pass": cv_pass,
        "threshold": f"CV < {ACCEPT_MAX_UTILIZATION_CV}",
        "measured_cv": round(cv, 4),
        "balanced_fraction": round(balanced_fraction, 4),
    }
    if not cv_pass:
        errors.append(
            f"Utilization CV {cv:.4f} exceeds threshold {ACCEPT_MAX_UTILIZATION_CV}."
        )

    # 3. Router entropy
    entropy_pass = entropy > ACCEPT_MIN_ROUTER_ENTROPY_NATS
    criteria["router_entropy_above_threshold"] = {
        "pass": entropy_pass,
        "threshold": f"> {ACCEPT_MIN_ROUTER_ENTROPY_NATS} nats",
        "measured_nats": round(entropy, 4),
        "max_possible_nats": round(math.log(num_experts), 4),
    }
    if not entropy_pass:
        errors.append(
            f"Router entropy {entropy:.4f} nats below threshold "
            f"{ACCEPT_MIN_ROUTER_ENTROPY_NATS} nats."
        )

    # 4. Dropped tokens
    dropped_pass = dropped_frac < ACCEPT_MAX_DROPPED_TOKEN_FRACTION
    criteria["dropped_tokens_below_threshold"] = {
        "pass": dropped_pass,
        "threshold": f"< {ACCEPT_MAX_DROPPED_TOKEN_FRACTION * 100:.1f}%",
        "measured_fraction": round(dropped_frac, 6),
    }
    if not dropped_pass:
        errors.append(
            f"Dropped token fraction {dropped_frac:.4%} exceeds threshold "
            f"{ACCEPT_MAX_DROPPED_TOKEN_FRACTION:.1%}."
        )

    all_pass = all(c["pass"] for c in criteria.values())
    outcome = OUTCOME_PASS if all_pass else OUTCOME_NOT_ACCEPTED

    return {
        "outcome": outcome,
        "criteria": criteria,
        "missing_keys": [],
        "errors": errors,
        "window": window_description,
        "thresholds": _threshold_dict(inactive_threshold),
    }


def _threshold_dict(inactive_threshold: float) -> dict:
    return {
        "max_dropped_token_fraction": ACCEPT_MAX_DROPPED_TOKEN_FRACTION,
        "max_utilization_cv": ACCEPT_MAX_UTILIZATION_CV,
        "min_router_entropy_nats": ACCEPT_MIN_ROUTER_ENTROPY_NATS,
        "min_expert_fraction": inactive_threshold,
        "max_inactive_experts": ACCEPT_MAX_INACTIVE_EXPERTS,
    }
