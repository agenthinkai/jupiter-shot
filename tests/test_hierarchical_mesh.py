"""
Tests for the hierarchical Mesh components:
- HierarchicalRouter
- ExpertRegistry
- ScalingSimulator
"""
import math
import sys
import time
import uuid
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh.hierarchical.hierarchical_router import (
    ComplexityEstimator,
    HierarchicalRouter,
    ModelTier,
    RoutingRequest,
    RoutingStrategy,
    TIER_METADATA,
    route_request,
)
from mesh.hierarchical.expert_registry import (
    ExpertEndpoint,
    ExpertRegistry,
    ExpertStatus,
    get_registry,
)
from benchmarks.simulators.scaling_simulator import (
    STAGES,
    GPU_SPECS,
    simulate_stage,
    compute_flops_per_token,
    compute_training_flops,
)


# ---------------------------------------------------------------------------
# ComplexityEstimator tests
# ---------------------------------------------------------------------------

class TestComplexityEstimator:
    def setup_method(self):
        self.estimator = ComplexityEstimator()

    def _make_request(self, prompt: str, max_tokens: int = 512) -> RoutingRequest:
        return RoutingRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            max_tokens=max_tokens,
        )

    def test_short_simple_prompt_low_complexity(self):
        req = self._make_request("What is the capital of France?", max_tokens=50)
        score = self.estimator.estimate(req)
        assert 0.0 <= score <= 0.5, f"Expected low complexity, got {score}"

    def test_long_complex_prompt_high_complexity(self):
        prompt = (
            "Analyze the tradeoffs between sparse mixture-of-experts architectures "
            "and dense transformer models for sovereign AI deployment. Consider "
            "compute efficiency, routing overhead, expert collapse risks, and "
            "implications for long-context reasoning. Provide a detailed comparison "
            "with specific recommendations for a 20-trillion-parameter system. " * 3
        )
        req = self._make_request(prompt, max_tokens=2000)
        score = self.estimator.estimate(req)
        assert score >= 0.35, f"Expected high complexity, got {score}"

    def test_score_in_valid_range(self):
        for prompt in ["hi", "x" * 10000, "analyze this " * 100]:
            req = self._make_request(prompt)
            score = self.estimator.estimate(req)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for prompt length {len(prompt)}"

    def test_high_max_tokens_increases_score(self):
        base_req = self._make_request("Explain quantum computing", max_tokens=100)
        high_req = self._make_request("Explain quantum computing", max_tokens=3000)
        base_score = self.estimator.estimate(base_req)
        high_score = self.estimator.estimate(high_req)
        assert high_score >= base_score, "Higher max_tokens should not decrease complexity score"


# ---------------------------------------------------------------------------
# HierarchicalRouter tests
# ---------------------------------------------------------------------------

class TestHierarchicalRouter:
    def setup_method(self):
        self.router = HierarchicalRouter()
        # Mark T1 as available (it's the only one available by default)
        # T1 is already available per TIER_METADATA

    def _make_request(self, prompt: str = "What is 2+2?", **kwargs) -> RoutingRequest:
        return RoutingRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            **kwargs,
        )

    def test_routes_simple_request_to_t1(self):
        req = self._make_request("What is the capital of France?", max_tokens=50)
        decision = self.router.route(req)
        assert decision.selected_tier == ModelTier.T1
        assert decision.complexity_score >= 0.0

    def test_decision_has_required_fields(self):
        req = self._make_request()
        decision = self.router.route(req)
        assert decision.request_id == req.request_id
        assert decision.selected_tier is not None
        assert decision.complexity_score >= 0.0
        assert decision.estimated_latency_ms > 0
        assert decision.estimated_cost_usd >= 0.0
        assert decision.reasoning != ""
        assert decision.decision_id != ""

    def test_explicit_tier_routing(self):
        req = self._make_request(
            strategy=RoutingStrategy.EXPLICIT,
            explicit_tier=ModelTier.T1,
        )
        decision = self.router.route(req)
        assert decision.selected_tier == ModelTier.T1

    def test_explicit_unavailable_tier_falls_back(self):
        # T5 is not available, should fall back to T1
        req = self._make_request(
            strategy=RoutingStrategy.EXPLICIT,
            explicit_tier=ModelTier.T5,
        )
        decision = self.router.route(req)
        # Should fall back to an available tier
        assert decision.selected_tier in self.router.get_available_tiers()

    def test_latency_constraint_respected(self):
        # With a very tight latency constraint, should route to T1 (fastest)
        req = self._make_request(
            max_latency_ms=100,  # Only T1 meets this
        )
        decision = self.router.route(req)
        assert decision.estimated_latency_ms <= 200  # T1 is ~50ms

    def test_cost_constraint_respected(self):
        # With a very tight cost constraint, should route to T1 (cheapest)
        req = self._make_request(
            max_cost_per_1m=0.10,  # Only T1 meets this at $0.05
        )
        decision = self.router.route(req)
        assert decision.selected_tier == ModelTier.T1

    def test_routing_stats_accumulate(self):
        for i in range(5):
            req = self._make_request(f"Request {i}")
            self.router.route(req)
        stats = self.router.get_routing_stats()
        assert stats["total_requests"] == 5
        assert "tier_distribution" in stats
        assert "total_estimated_cost_usd" in stats

    def test_decision_to_dict(self):
        req = self._make_request()
        decision = self.router.route(req)
        d = decision.to_dict()
        assert "selected_tier" in d
        assert "complexity_score" in d
        assert "estimated_cost_usd" in d
        assert isinstance(d["selected_tier"], str)  # Enum serialized to string

    def test_no_available_tiers_falls_back_to_t1(self):
        # Temporarily mark T1 as unavailable
        from mesh.hierarchical.hierarchical_router import TierHealth
        original_health = self.router._tier_health[ModelTier.T1]
        self.router._tier_health[ModelTier.T1] = TierHealth(
            tier=ModelTier.T1, available=False
        )
        try:
            req = self._make_request()
            decision = self.router.route(req)
            # Should still return a decision (T1 is the last resort)
            assert decision.selected_tier is not None
        finally:
            self.router._tier_health[ModelTier.T1] = original_health


# ---------------------------------------------------------------------------
# ExpertRegistry tests
# ---------------------------------------------------------------------------

class TestExpertRegistry:
    def setup_method(self):
        self.registry = ExpertRegistry()

    def _make_endpoint(
        self,
        expert_ids: list[int] = None,
        model_stage: int = 1,
        status: ExpertStatus = ExpertStatus.HEALTHY,
    ) -> ExpertEndpoint:
        return ExpertEndpoint(
            endpoint_id=str(uuid.uuid4()),
            host="10.0.0.1",
            port=8080,
            gpu_ids=[0, 1],
            expert_ids=expert_ids or [0, 1],
            model_stage=model_stage,
            status=status,
            vram_used_gb=40.0,
            vram_total_gb=80.0,
            last_heartbeat=time.time(),
        )

    def test_register_and_retrieve(self):
        ep = self._make_endpoint(expert_ids=[0, 1, 2])
        endpoint_id = self.registry.register(ep)
        assert endpoint_id == ep.endpoint_id

        result = self.registry.get_endpoint_for_expert(0, model_stage=1)
        assert result is not None
        assert result.endpoint_id == ep.endpoint_id

    def test_deregister_removes_endpoint(self):
        ep = self._make_endpoint(expert_ids=[5])
        self.registry.register(ep)
        assert self.registry.deregister(ep.endpoint_id)
        result = self.registry.get_endpoint_for_expert(5, model_stage=1)
        assert result is None

    def test_deregister_nonexistent_returns_false(self):
        assert not self.registry.deregister("nonexistent-id")

    def test_heartbeat_updates_health(self):
        ep = self._make_endpoint()
        self.registry.register(ep)
        self.registry.heartbeat(ep.endpoint_id, {
            "status": "healthy",
            "vram_used_gb": 60.0,
            "vram_total_gb": 80.0,
            "current_requests": 10,
            "p50_latency_ms": 45.0,
        })
        updated = self.registry._endpoints[ep.endpoint_id]
        assert updated.vram_used_gb == 60.0
        assert updated.p50_latency_ms == 45.0

    def test_stale_endpoint_marked_unavailable(self):
        ep = self._make_endpoint()
        ep.last_heartbeat = time.time() - 60  # 60 seconds ago (> 30s timeout)
        self.registry.register(ep)
        # Trigger stale check
        result = self.registry.get_endpoint_for_expert(ep.expert_ids[0], model_stage=1)
        assert result is None  # Stale endpoint should not be returned

    def test_coverage_report(self):
        # Register endpoints covering experts 0-3
        ep1 = self._make_endpoint(expert_ids=[0, 1])
        ep2 = self._make_endpoint(expert_ids=[2, 3])
        self.registry.register(ep1)
        self.registry.register(ep2)

        coverage = self.registry.get_coverage(model_stage=1, num_experts=8)
        assert coverage["covered_experts"] == 4
        assert coverage["coverage_pct"] == 50.0
        assert 4 in coverage["missing_experts"]

    def test_least_loaded_routing(self):
        # Two endpoints for same expert, different load
        ep1 = self._make_endpoint(expert_ids=[10])
        ep1.vram_used_gb = 70.0  # High load
        ep2 = self._make_endpoint(expert_ids=[10])
        ep2.vram_used_gb = 20.0  # Low load
        self.registry.register(ep1)
        self.registry.register(ep2)

        result = self.registry.get_endpoint_for_expert(10, model_stage=1, strategy="least_loaded")
        assert result is not None
        assert result.vram_used_gb == 20.0  # Should select lower load

    def test_registry_summary(self):
        ep1 = self._make_endpoint(expert_ids=[0], model_stage=1)
        ep2 = self._make_endpoint(expert_ids=[1], model_stage=2)
        self.registry.register(ep1)
        self.registry.register(ep2)

        summary = self.registry.get_registry_summary()
        assert summary["total_endpoints"] == 2
        assert 1 in summary["stages_registered"]
        assert 2 in summary["stages_registered"]

    def test_endpoint_load_factor(self):
        ep = self._make_endpoint()
        ep.vram_used_gb = 40.0
        ep.vram_total_gb = 80.0
        ep.current_requests = 50
        # load_factor = 0.7 * (40/80) + 0.3 * (50/100) = 0.35 + 0.15 = 0.50
        assert abs(ep.load_factor - 0.50) < 0.01

    def test_endpoint_to_dict(self):
        ep = self._make_endpoint()
        d = ep.to_dict()
        assert "endpoint_id" in d
        assert "status" in d
        assert isinstance(d["status"], str)  # Enum serialized
        assert "address" in d
        assert "load_factor" in d


# ---------------------------------------------------------------------------
# ScalingSimulator tests
# ---------------------------------------------------------------------------

class TestScalingSimulator:
    def test_all_stages_simulate_without_error(self):
        for stage_spec in STAGES:
            result = simulate_stage(stage_spec)
            assert result["stage"] == stage_spec.stage
            assert result["total_params"] > 0
            assert result["training_cost_usd"] > 0
            assert result["duration_days"] > 0

    def test_stage1_params_match_expected(self):
        stage1 = next(s for s in STAGES if s.stage == 1)
        result = simulate_stage(stage1)
        # 1.3B dense model
        assert 1_200_000_000 < result["total_params"] < 1_400_000_000

    def test_stage6_is_20t(self):
        stage6 = next(s for s in STAGES if s.stage == 6)
        result = simulate_stage(stage6)
        assert result["total_params"] == 20_000_000_000_000

    def test_moe_sparsity_correct(self):
        stage2 = next(s for s in STAGES if s.stage == 2)
        result = simulate_stage(stage2)
        assert result["is_moe"]
        assert result["sparsity_pct"] > 0
        # 47B total, 7B active → ~85% sparsity
        assert result["sparsity_pct"] > 80.0

    def test_dense_has_zero_sparsity(self):
        stage1 = next(s for s in STAGES if s.stage == 1)
        result = simulate_stage(stage1)
        assert not result["is_moe"]
        assert result["sparsity_pct"] == 0.0

    def test_flops_per_token_positive(self):
        for stage_spec in STAGES:
            flops = compute_flops_per_token(stage_spec)
            assert flops > 0, f"Stage {stage_spec.stage} has non-positive FLOPs per token"

    def test_total_training_flops_increases_with_stage(self):
        results = [simulate_stage(s) for s in STAGES]
        for i in range(1, len(results)):
            assert results[i]["total_flops"] > results[i-1]["total_flops"], \
                f"Stage {results[i]['stage']} total FLOPs should exceed stage {results[i-1]['stage']}"

    def test_gpu_price_override(self):
        stage1 = next(s for s in STAGES if s.stage == 1)
        result_default = simulate_stage(stage1)
        result_cheap = simulate_stage(stage1, gpu_price_override=1000.0)
        # Hardware cost should be lower with cheaper GPU price
        assert result_cheap["hardware_cost_usd"] < result_default["hardware_cost_usd"]

    def test_training_cost_positive_for_all_stages(self):
        for stage_spec in STAGES:
            result = simulate_stage(stage_spec)
            assert result["training_cost_usd"] > 0

    def test_stage1_training_cost_under_10k(self):
        stage1 = next(s for s in STAGES if s.stage == 1)
        result = simulate_stage(stage1)
        # Stage 1 should cost under $10,000 on spot pricing
        assert result["training_cost_usd"] < 10_000, \
            f"Stage 1 cost ${result['training_cost_usd']:.0f} exceeds $10,000"

    def test_prerequisite_chain(self):
        stage_map = {s.stage: s for s in STAGES}
        for stage_spec in STAGES:
            if stage_spec.prerequisite_stage:
                assert stage_spec.prerequisite_stage in stage_map, \
                    f"Stage {stage_spec.stage} prerequisite {stage_spec.prerequisite_stage} not found"
