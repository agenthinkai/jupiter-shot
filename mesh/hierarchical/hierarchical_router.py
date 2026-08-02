"""
Jupiter Shot — Hierarchical Mesh Router

Routes incoming requests to the appropriate model tier based on:
- Estimated request complexity
- Latency SLA
- Cost budget
- Current tier availability

Architecture:
  Tier 1 (T1): 1.3B dense — fast, cheap, simple tasks
  Tier 2 (T2): 47B MoE — general reasoning
  Tier 3 (T3): 200B MoE — complex reasoning
  Tier 4 (T4): 1T MoE — deep analysis
  Tier 5 (T5): 20T MoE — maximum reasoning depth

The router is itself served by a T1 model to avoid circular dependency.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class ModelTier(str, Enum):
    T1 = "t1"   # 1.3B dense
    T2 = "t2"   # 47B MoE
    T3 = "t3"   # 200B MoE
    T4 = "t4"   # 1T MoE
    T5 = "t5"   # 20T MoE


class RoutingStrategy(str, Enum):
    COST_OPTIMAL = "cost_optimal"       # Cheapest tier that meets quality threshold
    LATENCY_OPTIMAL = "latency_optimal" # Fastest tier that meets quality threshold
    QUALITY_OPTIMAL = "quality_optimal" # Highest quality tier within budget
    EXPLICIT = "explicit"               # Caller specifies tier directly


# Tier metadata: (active_params, latency_ms_p50, cost_per_1m_tokens_usd)
TIER_METADATA: dict[ModelTier, dict] = {
    ModelTier.T1: {"active_params": 1_300_000_000, "latency_ms_p50": 50,   "cost_per_1m": 0.05,  "available": True},
    ModelTier.T2: {"active_params": 7_000_000_000, "latency_ms_p50": 200,  "cost_per_1m": 0.20,  "available": False},
    ModelTier.T3: {"active_params": 25_000_000_000,"latency_ms_p50": 500,  "cost_per_1m": 0.80,  "available": False},
    ModelTier.T4: {"active_params": 62_000_000_000,"latency_ms_p50": 2000, "cost_per_1m": 3.00,  "available": False},
    ModelTier.T5: {"active_params": 625_000_000_000,"latency_ms_p50": 10000,"cost_per_1m": 25.00, "available": False},
}

# Complexity thresholds for automatic tier selection
COMPLEXITY_TIER_MAP = [
    (0.0,  0.2,  ModelTier.T1),  # Simple: Q&A, classification, short summaries
    (0.2,  0.4,  ModelTier.T2),  # Moderate: general reasoning, longer summaries
    (0.4,  0.6,  ModelTier.T3),  # Complex: multi-step reasoning, code generation
    (0.6,  0.8,  ModelTier.T4),  # Deep: research, analysis, long-form generation
    (0.8,  1.0,  ModelTier.T5),  # Maximum: frontier reasoning tasks
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoutingRequest:
    request_id: str
    prompt: str
    max_tokens: int = 512
    strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMAL
    explicit_tier: Optional[ModelTier] = None
    max_latency_ms: Optional[int] = None       # Latency SLA (None = no constraint)
    max_cost_per_1m: Optional[float] = None    # Cost budget (None = no constraint)
    min_tier: Optional[ModelTier] = None       # Minimum tier (None = T1)
    metadata: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    request_id: str
    selected_tier: ModelTier
    routing_strategy: RoutingStrategy
    complexity_score: float
    estimated_latency_ms: int
    estimated_cost_usd: float
    reasoning: str
    fallback_tier: Optional[ModelTier]
    timestamp: float = field(default_factory=time.time)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["selected_tier"] = self.selected_tier.value
        d["routing_strategy"] = self.routing_strategy.value
        if self.fallback_tier:
            d["fallback_tier"] = self.fallback_tier.value
        return d


@dataclass
class TierHealth:
    tier: ModelTier
    available: bool
    queue_depth: int = 0
    p50_latency_ms: int = 0
    p99_latency_ms: int = 0
    error_rate_pct: float = 0.0
    last_updated: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Complexity estimator
# ---------------------------------------------------------------------------

class ComplexityEstimator:
    """
    Estimates request complexity as a float in [0, 1].

    This is a heuristic estimator. In production, this would be replaced
    by a T1 model that classifies request complexity.
    """

    # Keywords that increase complexity score
    HIGH_COMPLEXITY_KEYWORDS = [
        "analyze", "compare", "evaluate", "synthesize", "critique",
        "research", "explain in detail", "step by step", "prove",
        "derive", "implement", "debug", "optimize", "design",
        "architecture", "tradeoffs", "implications", "consequences",
    ]

    LOW_COMPLEXITY_KEYWORDS = [
        "what is", "who is", "when", "where", "yes or no",
        "translate", "summarize briefly", "list", "define",
    ]

    def estimate(self, request: RoutingRequest) -> float:
        """Return complexity score in [0, 1]."""
        prompt = request.prompt.lower()
        score = 0.3  # Base score

        # Length signal: longer prompts tend to be more complex
        token_estimate = len(prompt.split())
        if token_estimate > 500:
            score += 0.2
        elif token_estimate > 200:
            score += 0.1
        elif token_estimate < 50:
            score -= 0.1

        # Keyword signals
        for kw in self.HIGH_COMPLEXITY_KEYWORDS:
            if kw in prompt:
                score += 0.05
                break  # Only count once

        for kw in self.LOW_COMPLEXITY_KEYWORDS:
            if kw in prompt:
                score -= 0.05
                break

        # Max tokens signal: more tokens requested → more complex
        if request.max_tokens > 2000:
            score += 0.15
        elif request.max_tokens > 500:
            score += 0.05

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Hierarchical router
# ---------------------------------------------------------------------------

class HierarchicalRouter:
    """
    Routes requests to model tiers based on complexity, latency SLA, and cost budget.

    Thread-safe for concurrent use. Uses asyncio for non-blocking health checks.
    """

    def __init__(self) -> None:
        self._complexity_estimator = ComplexityEstimator()
        self._tier_health: dict[ModelTier, TierHealth] = {
            tier: TierHealth(
                tier=tier,
                available=meta["available"],
                p50_latency_ms=meta["latency_ms_p50"],
            )
            for tier, meta in TIER_METADATA.items()
        }
        self._routing_log: list[dict] = []
        self._lock = asyncio.Lock()

    def update_tier_health(self, tier: ModelTier, health: TierHealth) -> None:
        """Update health status for a tier (called by health check loop)."""
        self._tier_health[tier] = health

    def get_available_tiers(self) -> list[ModelTier]:
        """Return list of currently available tiers in ascending order."""
        return [
            tier for tier in ModelTier
            if self._tier_health[tier].available
        ]

    def _select_tier_by_complexity(self, complexity: float) -> ModelTier:
        """Map complexity score to tier."""
        for low, high, tier in COMPLEXITY_TIER_MAP:
            if low <= complexity < high:
                return tier
        return ModelTier.T5

    def _apply_constraints(
        self,
        preferred_tier: ModelTier,
        request: RoutingRequest,
        available_tiers: list[ModelTier],
    ) -> tuple[ModelTier, str]:
        """
        Apply latency, cost, and availability constraints to the preferred tier.
        Returns (selected_tier, reasoning).
        """
        tier_order = list(ModelTier)

        # If explicit tier requested, use it (if available)
        if request.strategy == RoutingStrategy.EXPLICIT and request.explicit_tier:
            if request.explicit_tier in available_tiers:
                return request.explicit_tier, f"Explicit tier {request.explicit_tier.value} requested and available"
            else:
                # Fall back to highest available tier below requested
                fallback = self._find_fallback(request.explicit_tier, available_tiers, direction="down")
                return fallback, f"Explicit tier {request.explicit_tier.value} unavailable, falling back to {fallback.value}"

        # Apply minimum tier constraint
        min_tier_idx = tier_order.index(request.min_tier) if request.min_tier else 0
        preferred_idx = tier_order.index(preferred_tier)
        effective_idx = max(min_tier_idx, preferred_idx)
        effective_tier = tier_order[effective_idx]

        # Apply latency constraint: if preferred tier is too slow, downgrade
        if request.max_latency_ms is not None:
            while effective_idx > 0:
                meta = TIER_METADATA[tier_order[effective_idx]]
                if meta["latency_ms_p50"] <= request.max_latency_ms:
                    break
                effective_idx -= 1
            effective_tier = tier_order[effective_idx]

        # Apply cost constraint: if preferred tier is too expensive, downgrade
        if request.max_cost_per_1m is not None:
            while effective_idx > 0:
                meta = TIER_METADATA[tier_order[effective_idx]]
                if meta["cost_per_1m"] <= request.max_cost_per_1m:
                    break
                effective_idx -= 1
            effective_tier = tier_order[effective_idx]

        # Check availability: if selected tier unavailable, find nearest available
        if effective_tier not in available_tiers:
            fallback = self._find_fallback(effective_tier, available_tiers, direction="down")
            return fallback, f"Tier {effective_tier.value} unavailable, routing to {fallback.value}"

        reasoning = (
            f"Complexity {preferred_tier.value} → constraints applied → {effective_tier.value}"
        )
        return effective_tier, reasoning

    def _find_fallback(
        self,
        preferred: ModelTier,
        available: list[ModelTier],
        direction: str = "down",
    ) -> ModelTier:
        """Find the nearest available tier in the given direction."""
        tier_order = list(ModelTier)
        idx = tier_order.index(preferred)
        if direction == "down":
            for i in range(idx, -1, -1):
                if tier_order[i] in available:
                    return tier_order[i]
        else:
            for i in range(idx, len(tier_order)):
                if tier_order[i] in available:
                    return tier_order[i]
        # Last resort: T1 (always available in reference architecture)
        return ModelTier.T1

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route a request to the appropriate tier.

        This is a synchronous method. For async use, wrap in asyncio.to_thread().
        """
        available_tiers = self.get_available_tiers()
        if not available_tiers:
            logger.warning("No tiers available, defaulting to T1")
            available_tiers = [ModelTier.T1]

        complexity = self._complexity_estimator.estimate(request)
        preferred_tier = self._select_tier_by_complexity(complexity)

        selected_tier, reasoning = self._apply_constraints(
            preferred_tier, request, available_tiers
        )

        meta = TIER_METADATA[selected_tier]
        health = self._tier_health[selected_tier]

        # Find fallback tier (next tier down that is available)
        tier_order = list(ModelTier)
        selected_idx = tier_order.index(selected_tier)
        fallback = None
        if selected_idx > 0:
            for i in range(selected_idx - 1, -1, -1):
                if tier_order[i] in available_tiers:
                    fallback = tier_order[i]
                    break

        # Estimate cost for this request
        estimated_tokens = len(request.prompt.split()) + request.max_tokens
        estimated_cost = (estimated_tokens / 1_000_000) * meta["cost_per_1m"]

        decision = RoutingDecision(
            request_id=request.request_id,
            selected_tier=selected_tier,
            routing_strategy=request.strategy,
            complexity_score=complexity,
            estimated_latency_ms=health.p50_latency_ms,
            estimated_cost_usd=estimated_cost,
            reasoning=reasoning,
            fallback_tier=fallback,
        )

        self._routing_log.append(decision.to_dict())
        logger.info(
            "Routed request %s → %s (complexity=%.2f, cost=$%.6f)",
            request.request_id, selected_tier.value, complexity, estimated_cost
        )
        return decision

    def get_routing_stats(self) -> dict:
        """Return aggregate routing statistics."""
        if not self._routing_log:
            return {"total_requests": 0}

        tier_counts: dict[str, int] = {}
        total_cost = 0.0
        for entry in self._routing_log:
            tier = entry["selected_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_cost += entry.get("estimated_cost_usd", 0.0)

        return {
            "total_requests": len(self._routing_log),
            "tier_distribution": tier_counts,
            "total_estimated_cost_usd": total_cost,
            "avg_cost_per_request_usd": total_cost / len(self._routing_log),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: Optional[HierarchicalRouter] = None


def get_router() -> HierarchicalRouter:
    """Get or create the module-level router singleton."""
    global _router
    if _router is None:
        _router = HierarchicalRouter()
    return _router


def route_request(
    prompt: str,
    max_tokens: int = 512,
    strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMAL,
    max_latency_ms: Optional[int] = None,
    max_cost_per_1m: Optional[float] = None,
    explicit_tier: Optional[ModelTier] = None,
) -> RoutingDecision:
    """Convenience function for routing a single request."""
    request = RoutingRequest(
        request_id=str(uuid.uuid4()),
        prompt=prompt,
        max_tokens=max_tokens,
        strategy=strategy if explicit_tier is None else RoutingStrategy.EXPLICIT,
        explicit_tier=explicit_tier,
        max_latency_ms=max_latency_ms,
        max_cost_per_1m=max_cost_per_1m,
    )
    return get_router().route(request)
