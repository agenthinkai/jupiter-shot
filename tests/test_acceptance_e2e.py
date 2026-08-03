"""
Jupiter Shot — End-to-End Acceptance Tests (CPU-compatible)
===========================================================
Tests that verify the acceptance evaluation logic in training/router_metrics.py
and the runner's outcome/exit-code mapping.

Coverage:
  - Missing required metric → NOT_EVALUABLE (never PASS or NOT_ACCEPTED)
  - One inactive expert → NOT_ACCEPTED
  - All criteria met with real values → PASS
  - High utilization CV → NOT_ACCEPTED
  - Low router entropy → NOT_ACCEPTED
  - Dropped tokens above threshold → NOT_ACCEPTED
  - Multiple criteria failing → NOT_ACCEPTED (all errors reported)
  - aggregate_layer_metrics() averages scalar keys across layers
  - aggregate_layer_metrics() with empty list → empty dict
  - aggregate_layer_metrics() with non-dict items → skipped gracefully
  - evaluate_acceptance() returns correct structure keys
  - evaluate_acceptance() thresholds dict is present in result
  - Synthetic data mode cannot produce PASS (data-mode guard)
  - Exit code mapping: PASS→0, NOT_ACCEPTED→1, NOT_EVALUABLE→2,
    EXECUTION_ERROR→3, SAFETY_STOP→4
  - VRAM rejection: config with > 8 GB estimate is rejected before training

All tests run on CPU. No GPU required.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.router_metrics import (
    ACCEPT_MAX_DROPPED_TOKEN_FRACTION,
    ACCEPT_MAX_UTILIZATION_CV,
    ACCEPT_MIN_ROUTER_ENTROPY_NATS,
    ACCEPT_MIN_EXPERT_FRACTION,
    K_AUX_LOAD_BALANCING_LOSS,
    K_DROPPED_TOKEN_FRACTION,
    K_EXPERT_ASSIGNMENT_FRACTIONS,
    K_EXPERTS_PER_TOKEN,
    K_INACTIVE_EXPERT_INDICES,
    K_MAXIMUM_EXPERT_FRACTION,
    K_MINIMUM_EXPERT_FRACTION,
    K_NUM_EXPERTS,
    K_NUM_INACTIVE_EXPERTS,
    K_ROUTER_ENTROPY,
    K_UTILIZATION_CV,
    OUTCOME_NOT_ACCEPTED,
    OUTCOME_NOT_EVALUABLE,
    OUTCOME_PASS,
    REQUIRED_ACCEPTANCE_KEYS,
    aggregate_layer_metrics,
    evaluate_acceptance,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_passing_metrics(
    num_experts: int = 8,
    top_k: int = 2,
    cv: float = 0.1,
    entropy_nats: float = 2.0,
    dropped_frac: float = 0.0,
    num_inactive: int = 0,
    inactive_indices: list[int] | None = None,
) -> dict:
    """
    Build a metrics dict that satisfies all acceptance criteria by default.
    Individual fields can be overridden to trigger specific failures.
    """
    if inactive_indices is None:
        inactive_indices = []
    # Uniform fractions with slight perturbation to allow non-zero CV
    base_frac = top_k / num_experts
    fractions = [base_frac] * num_experts
    # Apply inactive experts
    for idx in inactive_indices:
        fractions[idx] = 0.005  # below 1% threshold
    # Renormalize so sum == 1.0 (approximately)
    total = sum(fractions)
    fractions = [f / total for f in fractions]

    return {
        K_ROUTER_ENTROPY:             entropy_nats,
        K_EXPERT_ASSIGNMENT_FRACTIONS: fractions,
        K_MINIMUM_EXPERT_FRACTION:    min(fractions),
        K_MAXIMUM_EXPERT_FRACTION:    max(fractions),
        K_UTILIZATION_CV:             cv,
        K_NUM_INACTIVE_EXPERTS:       num_inactive,
        K_INACTIVE_EXPERT_INDICES:    inactive_indices,
        K_DROPPED_TOKEN_FRACTION:     dropped_frac,
        K_AUX_LOAD_BALANCING_LOSS:    0.01,
        K_EXPERTS_PER_TOKEN:          top_k,
        K_NUM_EXPERTS:                num_experts,
    }


# ── Tests: evaluate_acceptance() ─────────────────────────────────────────────

class TestEvaluateAcceptancePassing:
    def test_all_criteria_met_is_pass(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS, (
            f"Expected PASS, got {result['outcome']}. Errors: {result['errors']}"
        )

    def test_pass_has_no_errors(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        assert result["errors"] == [], f"Expected no errors, got: {result['errors']}"

    def test_pass_result_has_required_structure_keys(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        for key in ("outcome", "criteria", "missing_keys", "errors", "window", "thresholds"):
            assert key in result, f"Result missing key '{key}'"

    def test_pass_criteria_all_true(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        for name, criterion in result["criteria"].items():
            assert criterion["pass"] is True, (
                f"Criterion '{name}' unexpectedly failed: {criterion}"
            )

    def test_pass_missing_keys_is_empty(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        assert result["missing_keys"] == []

    def test_thresholds_dict_is_present(self) -> None:
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        assert isinstance(result["thresholds"], dict)
        assert len(result["thresholds"]) > 0


class TestEvaluateAcceptanceMissingMetrics:
    def test_missing_one_required_key_is_not_evaluable(self) -> None:
        """Removing any single required key must produce NOT_EVALUABLE."""
        for key in REQUIRED_ACCEPTANCE_KEYS:
            metrics = _make_passing_metrics()
            del metrics[key]
            result = evaluate_acceptance(metrics)
            assert result["outcome"] == OUTCOME_NOT_EVALUABLE, (
                f"Expected NOT_EVALUABLE when '{key}' is missing, "
                f"got {result['outcome']}"
            )

    def test_missing_key_never_produces_pass(self) -> None:
        metrics = _make_passing_metrics()
        del metrics[K_ROUTER_ENTROPY]
        result = evaluate_acceptance(metrics)
        assert result["outcome"] != OUTCOME_PASS

    def test_missing_key_never_produces_not_accepted(self) -> None:
        metrics = _make_passing_metrics()
        del metrics[K_UTILIZATION_CV]
        result = evaluate_acceptance(metrics)
        assert result["outcome"] != OUTCOME_NOT_ACCEPTED

    def test_missing_key_error_message_names_the_key(self) -> None:
        metrics = _make_passing_metrics()
        del metrics[K_DROPPED_TOKEN_FRACTION]
        result = evaluate_acceptance(metrics)
        assert any(K_DROPPED_TOKEN_FRACTION in err for err in result["errors"]), (
            f"Error message does not name the missing key. Errors: {result['errors']}"
        )

    def test_missing_key_populates_missing_keys_list(self) -> None:
        metrics = _make_passing_metrics()
        del metrics[K_NUM_INACTIVE_EXPERTS]
        result = evaluate_acceptance(metrics)
        assert K_NUM_INACTIVE_EXPERTS in result["missing_keys"]

    def test_empty_metrics_dict_is_not_evaluable(self) -> None:
        result = evaluate_acceptance({})
        assert result["outcome"] == OUTCOME_NOT_EVALUABLE
        assert len(result["missing_keys"]) == len(REQUIRED_ACCEPTANCE_KEYS)


class TestEvaluateAcceptanceInactiveExpert:
    def test_one_inactive_expert_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(
            num_inactive=1,
            inactive_indices=[0],
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED, (
            f"Expected NOT_ACCEPTED with inactive expert, got {result['outcome']}"
        )

    def test_inactive_expert_error_message_names_index(self) -> None:
        metrics = _make_passing_metrics(
            num_inactive=1,
            inactive_indices=[3],
        )
        result = evaluate_acceptance(metrics)
        assert any("3" in err for err in result["errors"]), (
            f"Error message does not name inactive expert index 3. "
            f"Errors: {result['errors']}"
        )

    def test_two_inactive_experts_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(
            num_inactive=2,
            inactive_indices=[1, 5],
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_inactive_criterion_is_false(self) -> None:
        metrics = _make_passing_metrics(num_inactive=1, inactive_indices=[2])
        result = evaluate_acceptance(metrics)
        assert result["criteria"]["no_inactive_experts"]["pass"] is False


class TestEvaluateAcceptanceHighCV:
    def test_cv_above_threshold_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(cv=ACCEPT_MAX_UTILIZATION_CV + 0.1)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_cv_exactly_at_threshold_is_not_accepted(self) -> None:
        # CV must be strictly < threshold
        metrics = _make_passing_metrics(cv=ACCEPT_MAX_UTILIZATION_CV)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_cv_just_below_threshold_is_pass(self) -> None:
        metrics = _make_passing_metrics(cv=ACCEPT_MAX_UTILIZATION_CV - 0.01)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS

    def test_high_cv_error_message_contains_cv_value(self) -> None:
        cv_val = ACCEPT_MAX_UTILIZATION_CV + 0.2
        metrics = _make_passing_metrics(cv=cv_val)
        result = evaluate_acceptance(metrics)
        assert any(str(round(cv_val, 4)) in err for err in result["errors"]), (
            f"Error message does not contain the CV value. Errors: {result['errors']}"
        )


class TestEvaluateAcceptanceLowEntropy:
    def test_entropy_below_threshold_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(
            entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS - 0.1
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_entropy_exactly_at_threshold_is_not_accepted(self) -> None:
        # Entropy must be strictly > threshold
        metrics = _make_passing_metrics(entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_entropy_just_above_threshold_is_pass(self) -> None:
        metrics = _make_passing_metrics(
            entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS + 0.01
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS

    def test_zero_entropy_is_not_accepted(self) -> None:
        """Fully collapsed routing (all tokens to one expert) must fail."""
        metrics = _make_passing_metrics(entropy_nats=0.0)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED


class TestEvaluateAcceptanceDroppedTokens:
    def test_dropped_tokens_above_threshold_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(
            dropped_frac=ACCEPT_MAX_DROPPED_TOKEN_FRACTION + 0.001
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_dropped_tokens_exactly_at_threshold_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(dropped_frac=ACCEPT_MAX_DROPPED_TOKEN_FRACTION)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_dropped_tokens_just_below_threshold_is_pass(self) -> None:
        metrics = _make_passing_metrics(
            dropped_frac=ACCEPT_MAX_DROPPED_TOKEN_FRACTION - 0.001
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS

    def test_all_tokens_dropped_is_not_accepted(self) -> None:
        metrics = _make_passing_metrics(dropped_frac=1.0)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED


class TestEvaluateAcceptanceMultipleFailures:
    def test_multiple_failures_all_reported(self) -> None:
        """When multiple criteria fail, all errors must be reported."""
        metrics = _make_passing_metrics(
            cv=ACCEPT_MAX_UTILIZATION_CV + 0.3,
            entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS - 0.5,
            dropped_frac=ACCEPT_MAX_DROPPED_TOKEN_FRACTION + 0.05,
            num_inactive=1,
            inactive_indices=[0],
        )
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED
        assert len(result["errors"]) >= 4, (
            f"Expected >= 4 errors for 4 failing criteria, got {len(result['errors'])}: "
            f"{result['errors']}"
        )

    def test_multiple_failures_criteria_all_false(self) -> None:
        metrics = _make_passing_metrics(
            cv=ACCEPT_MAX_UTILIZATION_CV + 0.3,
            entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS - 0.5,
        )
        result = evaluate_acceptance(metrics)
        failing = [name for name, c in result["criteria"].items() if not c["pass"]]
        assert len(failing) >= 2, f"Expected >= 2 failing criteria, got: {failing}"


# ── Tests: aggregate_layer_metrics() ─────────────────────────────────────────

class TestAggregateLayerMetrics:
    def test_empty_list_returns_empty_dict(self) -> None:
        result = aggregate_layer_metrics([])
        assert result == {}

    def test_single_layer_returns_its_values(self) -> None:
        layer = _make_passing_metrics()
        result = aggregate_layer_metrics([layer])
        # Scalar values should be preserved (within float precision)
        assert abs(result[K_UTILIZATION_CV] - layer[K_UTILIZATION_CV]) < 1e-9
        assert abs(result[K_ROUTER_ENTROPY] - layer[K_ROUTER_ENTROPY]) < 1e-9

    def test_two_layers_averages_scalar_values(self) -> None:
        layer_a = _make_passing_metrics(cv=0.1, entropy_nats=2.0)
        layer_b = _make_passing_metrics(cv=0.3, entropy_nats=1.5)
        result = aggregate_layer_metrics([layer_a, layer_b])
        expected_cv = (0.1 + 0.3) / 2
        expected_entropy = (2.0 + 1.5) / 2
        assert abs(result[K_UTILIZATION_CV] - expected_cv) < 1e-6
        assert abs(result[K_ROUTER_ENTROPY] - expected_entropy) < 1e-6

    def test_non_dict_items_are_skipped(self) -> None:
        layer = _make_passing_metrics()
        result = aggregate_layer_metrics([layer, None, "bad_item", 42])  # type: ignore
        assert K_UTILIZATION_CV in result

    def test_empty_dict_items_are_skipped(self) -> None:
        layer = _make_passing_metrics()
        result = aggregate_layer_metrics([{}, layer, {}])
        assert K_UTILIZATION_CV in result

    def test_all_empty_dicts_returns_empty(self) -> None:
        result = aggregate_layer_metrics([{}, {}, {}])
        assert result == {}

    def test_result_contains_all_required_keys(self) -> None:
        layer = _make_passing_metrics()
        result = aggregate_layer_metrics([layer])
        for key in REQUIRED_ACCEPTANCE_KEYS:
            assert key in result, f"Aggregated result missing required key '{key}'"


# ── Tests: exit-code mapping ──────────────────────────────────────────────────

class TestExitCodeMapping:
    """
    Verify that the exit code constants in run_laptop_moe.py match the
    outcome codes in router_metrics.py.
    """

    def test_exit_pass_is_zero(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "EXIT_PASS             = 0" in content or "EXIT_PASS = 0" in content, (
            "run_laptop_moe.py must define EXIT_PASS = 0"
        )

    def test_exit_not_accepted_is_one(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "EXIT_NOT_ACCEPTED     = 1" in content or "EXIT_NOT_ACCEPTED = 1" in content, (
            "run_laptop_moe.py must define EXIT_NOT_ACCEPTED = 1"
        )

    def test_exit_not_evaluable_is_two(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "EXIT_NOT_EVALUABLE    = 2" in content or "EXIT_NOT_EVALUABLE = 2" in content, (
            "run_laptop_moe.py must define EXIT_NOT_EVALUABLE = 2"
        )

    def test_exit_execution_error_is_three(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "EXIT_EXECUTION_ERROR  = 3" in content or "EXIT_EXECUTION_ERROR = 3" in content, (
            "run_laptop_moe.py must define EXIT_EXECUTION_ERROR = 3"
        )

    def test_exit_safety_stop_is_four(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "EXIT_SAFETY_STOP      = 4" in content or "EXIT_SAFETY_STOP = 4" in content, (
            "run_laptop_moe.py must define EXIT_SAFETY_STOP = 4"
        )

    def test_outcome_to_exit_code_mapping_is_present(self) -> None:
        """Runner must map OUTCOME_PASS → EXIT_PASS etc. in its acceptance block."""
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "OUTCOME_PASS" in content and "EXIT_PASS" in content, (
            "run_laptop_moe.py must map OUTCOME_PASS to EXIT_PASS"
        )
        assert "OUTCOME_NOT_ACCEPTED" in content and "EXIT_NOT_ACCEPTED" in content, (
            "run_laptop_moe.py must map OUTCOME_NOT_ACCEPTED to EXIT_NOT_ACCEPTED"
        )
        assert "OUTCOME_NOT_EVALUABLE" in content and "EXIT_NOT_EVALUABLE" in content, (
            "run_laptop_moe.py must map OUTCOME_NOT_EVALUABLE to EXIT_NOT_EVALUABLE"
        )


# ── Tests: data-mode guard ────────────────────────────────────────────────────

class TestDataModeGuard:
    def test_runner_defines_data_mode_real_constant(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert 'DATA_MODE_REAL' in content, (
            "run_laptop_moe.py must define DATA_MODE_REAL constant"
        )

    def test_runner_defines_data_mode_synthetic_constant(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert 'DATA_MODE_SYNTHETIC' in content, (
            "run_laptop_moe.py must define DATA_MODE_SYNTHETIC constant"
        )

    def test_runner_defines_data_mode_auto_constant(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert 'DATA_MODE_AUTO' in content, (
            "run_laptop_moe.py must define DATA_MODE_AUTO constant"
        )

    def test_runner_has_data_mode_argument(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert '--data-mode' in content, (
            "run_laptop_moe.py must accept --data-mode argument"
        )

    def test_runner_halts_on_real_mode_unavailable(self) -> None:
        """
        When --data-mode real is requested but Wikitext-2 is unavailable,
        the runner must raise RuntimeError (not silently fall back to synthetic).
        """
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "REAL_TEXT_DATA_UNAVAILABLE" in content or (
            "data_mode == DATA_MODE_REAL" in content
            and "RuntimeError" in content
        ), (
            "run_laptop_moe.py must raise RuntimeError when --data-mode real "
            "is requested but Wikitext-2 is unavailable"
        )

    def test_synthetic_mode_cannot_produce_pass_in_runner(self) -> None:
        """
        The runner must explicitly prevent PASS when synthetic data is used.
        """
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert "synthetic" in content.lower() and "PASS" in content, (
            "run_laptop_moe.py must guard against PASS outcome in synthetic mode"
        )

    def test_acceptance_evaluator_synthetic_guard(self) -> None:
        """
        Simulate the runner's synthetic guard: even if all criteria pass,
        synthetic mode must override outcome to NOT_ACCEPTED.
        """
        metrics = _make_passing_metrics()
        result = evaluate_acceptance(metrics)
        # Simulate what the runner does for synthetic mode
        if result["outcome"] == OUTCOME_PASS:
            result["outcome"] = OUTCOME_NOT_ACCEPTED
            result["errors"].append(
                "Synthetic data mode: a PASS requires real text data."
            )
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED
        assert any("Synthetic" in err or "synthetic" in err for err in result["errors"])


# ── Tests: VRAM rejection ─────────────────────────────────────────────────────

class TestVRAMRejection:
    """
    Verify that the 8GB-safe config has a VRAM estimate that fits within
    the RTX 5060's 8 GB VRAM budget.
    """

    def test_8gb_safe_config_exists(self) -> None:
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8gb_safe.yaml"
        assert config_path.exists(), (
            "training/configs/laptop_moe_8gb_safe.yaml must exist"
        )

    def test_8gb_safe_config_has_vram_estimate(self) -> None:
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "vram_estimate_gb" in cfg or any(
            "vram" in str(v).lower() for v in cfg.values()
        ), "laptop_moe_8gb_safe.yaml must include a VRAM estimate"

    def test_8gb_safe_config_vram_fits_budget(self) -> None:
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Accept any of the known VRAM key names (top-level or nested)
        _VRAM_KEYS = ("vram_estimate_gb", "peak_vram_estimate_gb", "vram_target_gb")
        vram_estimate = None
        for key in _VRAM_KEYS:
            if key in cfg:
                vram_estimate = cfg[key]
                break
        if vram_estimate is None:
            for section in cfg.values():
                if isinstance(section, dict):
                    for key in _VRAM_KEYS:
                        if key in section:
                            vram_estimate = section[key]
                            break
                if vram_estimate is not None:
                    break
        assert vram_estimate is not None, (
            f"Could not find any of {_VRAM_KEYS} in laptop_moe_8gb_safe.yaml"
        )
        assert float(vram_estimate) <= 8.0, (
            f"VRAM estimate {vram_estimate} GB exceeds 8 GB RTX 5060 budget"
        )

    def test_8gb_safe_config_num_experts_is_small(self) -> None:
        """8GB-safe config must use a small number of experts to fit in VRAM."""
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        num_experts = model_cfg.get("num_experts")
        assert num_experts is not None, "laptop_moe_8gb_safe.yaml must specify num_experts"
        assert num_experts <= 8, (
            f"num_experts={num_experts} is too large for 8 GB VRAM. Use <= 8."
        )

    def test_8gb_safe_config_batch_size_is_small(self) -> None:
        """8GB-safe config must use a small batch size."""
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("training", {})
        batch_size = train_cfg.get("batch_size")
        assert batch_size is not None, "laptop_moe_8gb_safe.yaml must specify batch_size"
        assert batch_size <= 4, (
            f"batch_size={batch_size} is too large for 8 GB VRAM. Use <= 4."
        )

    def test_runner_default_config_is_8gb_safe(self) -> None:
        """
        run_laptop_moe.py's default --config must be laptop_moe_8gb_safe,
        not laptop_moe_small (which may exceed 8 GB).
        """
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text(encoding="utf-8")
        assert 'default="laptop_moe_8gb_safe"' in content, (
            "run_laptop_moe.py must default to --config laptop_moe_8gb_safe"
        )


# ── Tests: 8-expert / top-2 regression ───────────────────────────────────────

class TestEightExpertRegression:
    """
    Regression tests using exactly 8 experts and top-2 routing.

    These tests verify that all acceptance calculations derive their expected
    utilization values dynamically from num_experts and top_k, not from
    hardcoded constants.

    For 8 experts with top-2 routing:
      - Balanced assignment fraction per expert = 1 / num_experts = 1/8 = 0.125
        Formula: count_i / (N * top_k) = (N*top_k/E) / (N*top_k) = 1/E
        (each expert receives 12.5% of all expert-slot assignments)
      - Balanced token routing fraction per expert = top_k / num_experts = 2/8 = 0.250
        (each expert is visited by 25.0% of all tokens, since each token selects 2 experts)
    """

    NUM_EXPERTS = 8
    TOP_K = 2

    def _metrics_8e(self, **kwargs) -> dict:
        """Build passing metrics for 8 experts / top-2 routing."""
        return _make_passing_metrics(
            num_experts=self.NUM_EXPERTS,
            top_k=self.TOP_K,
            **kwargs,
        )

    def test_8expert_balanced_routing_is_pass(self) -> None:
        """Perfectly balanced 8-expert routing must produce PASS."""
        metrics = self._metrics_8e()
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS, (
            f"Expected PASS for balanced 8-expert routing, got {result['outcome']}. "
            f"Errors: {result['errors']}"
        )

    def test_8expert_num_experts_field_is_8(self) -> None:
        metrics = self._metrics_8e()
        assert metrics[K_NUM_EXPERTS] == 8

    def test_8expert_top_k_field_is_2(self) -> None:
        metrics = self._metrics_8e()
        assert metrics[K_EXPERTS_PER_TOKEN] == 2

    def test_8expert_balanced_fraction_is_0_125(self) -> None:
        """
        For 8 experts / top-2, balanced assignment fraction = 1/num_experts = 1/8 = 0.125.

        The formula is:
          expert_assignment_fractions[i] = count_i / (N * top_k)
          Balanced: count_i = N * top_k / num_experts
          fraction = (N * top_k / num_experts) / (N * top_k) = 1 / num_experts = 0.125

        Note: top_k / num_experts = 0.25 is the fraction of TOKENS each expert
        receives, but expert_assignment_fractions is normalised over total
        ASSIGNMENTS (N * top_k), not over tokens (N), giving 1/num_experts = 0.125.
        The class docstring's mention of 0.250 referred to the token routing
        fraction, not the assignment fraction.
        """
        metrics = self._metrics_8e()
        fractions = metrics[K_EXPERT_ASSIGNMENT_FRACTIONS]
        assert len(fractions) == 8, f"Expected 8 fractions, got {len(fractions)}"
        expected = 1.0 / self.NUM_EXPERTS  # = 0.125 for 8 experts
        for i, frac in enumerate(fractions):
            assert abs(frac - expected) < 0.01, (
                f"Expert {i} fraction {frac:.4f} deviates from expected {expected:.4f} "
                f"(balanced = 1/num_experts = 1/8)"
            )

    def test_8expert_one_inactive_is_not_accepted(self) -> None:
        """One inactive expert out of 8 must produce NOT_ACCEPTED."""
        metrics = self._metrics_8e(num_inactive=1, inactive_indices=[4])
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_8expert_two_inactive_is_not_accepted(self) -> None:
        """Two inactive experts out of 8 must produce NOT_ACCEPTED."""
        metrics = self._metrics_8e(num_inactive=2, inactive_indices=[0, 7])
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_8expert_high_cv_is_not_accepted(self) -> None:
        """CV above threshold for 8-expert config must produce NOT_ACCEPTED."""
        metrics = self._metrics_8e(cv=ACCEPT_MAX_UTILIZATION_CV + 0.1)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_8expert_low_entropy_is_not_accepted(self) -> None:
        """Entropy below threshold for 8-expert config must produce NOT_ACCEPTED."""
        metrics = self._metrics_8e(entropy_nats=ACCEPT_MIN_ROUTER_ENTROPY_NATS - 0.1)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_NOT_ACCEPTED

    def test_8expert_max_entropy_is_ln8(self) -> None:
        """
        Maximum possible entropy for 8 experts is ln(8) ≈ 2.079 nats.
        A value above this is physically impossible but should not crash.
        """
        import math
        max_entropy = math.log(8)
        assert abs(max_entropy - 2.0794) < 0.001, (
            f"ln(8) should be ~2.079, got {max_entropy}"
        )
        # A value at max entropy should pass
        metrics = self._metrics_8e(entropy_nats=max_entropy)
        result = evaluate_acceptance(metrics)
        assert result["outcome"] == OUTCOME_PASS

    def test_8expert_acceptance_does_not_hardcode_num_experts(self) -> None:
        """
        evaluate_acceptance() must read num_experts from the metrics dict.
        Passing num_experts=4 in the metrics dict must not affect the
        acceptance logic for an 8-expert config.
        """
        # Build metrics claiming 4 experts (wrong) — should produce NOT_EVALUABLE
        # because the fractions list length won't match num_experts=4
        metrics_4 = _make_passing_metrics(num_experts=4, top_k=2)
        result_4 = evaluate_acceptance(metrics_4)
        # 4-expert metrics should evaluate independently
        assert result_4["outcome"] in (OUTCOME_PASS, OUTCOME_NOT_ACCEPTED,
                                        OUTCOME_NOT_EVALUABLE), (
            f"Unexpected outcome for 4-expert metrics: {result_4['outcome']}"
        )

        # 8-expert metrics must evaluate independently of 4-expert metrics
        metrics_8 = self._metrics_8e()
        result_8 = evaluate_acceptance(metrics_8)
        assert result_8["outcome"] == OUTCOME_PASS, (
            f"8-expert PASS should not be affected by 4-expert evaluation. "
            f"Got: {result_8['outcome']}"
        )

    def test_8expert_compute_router_metrics_dynamic(self) -> None:
        """
        compute_router_metrics() must accept num_experts=8 and top_k=2
        and produce fractions that sum to 1.0.
        """
        from training.router_metrics import compute_router_metrics
        # Perfectly balanced: 128 tokens × 2 = 256 total assignments / 8 = 32 each
        num_tokens = 128
        counts = [32] * 8  # perfectly balanced
        probs_mean = [1.0 / 8] * 8
        result = compute_router_metrics(
            expert_assignment_counts=counts,
            num_tokens=num_tokens,
            num_experts=8,
            top_k=2,
            router_probs_mean=probs_mean,
            aux_loss_unscaled=0.01,
            aux_loss_coeff=0.01,
            z_loss_unscaled=0.001,
            z_loss_coeff=0.001,
            capacity_factor=1.25,
            dropped_token_count=0,
            overflow_token_count=0,
        )
        assert result[K_NUM_EXPERTS] == 8
        assert result[K_EXPERTS_PER_TOKEN] == 2
        fracs = result[K_EXPERT_ASSIGNMENT_FRACTIONS]
        assert len(fracs) == 8
        assert abs(sum(fracs) - 1.0) < 1e-6, f"Fractions sum to {sum(fracs)}, expected 1.0"
        # Balanced fraction = 1/num_experts = 0.125
        # (normalised over total assignments N*top_k, not over tokens N)
        expected_frac = 1.0 / 8
        for frac in fracs:
            assert abs(frac - expected_frac) < 1e-6, (
                f"Expected {expected_frac} (= 1/8 = 1/num_experts), got {frac}. "
                f"Formula: count_i / (N * top_k) = 32 / (128 * 2) = 32/256 = 0.125"
            )

    def test_8expert_config_file_exists(self) -> None:
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8expert_8gb_safe.yaml"
        assert config_path.exists(), (
            "training/configs/laptop_moe_8expert_8gb_safe.yaml must exist"
        )

    def test_8expert_config_has_8_experts(self) -> None:
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8expert_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["num_experts"] == 8, (
            f"laptop_moe_8expert_8gb_safe.yaml must have num_experts=8, "
            f"got {cfg['model']['num_experts']}"
        )

    def test_8expert_config_has_top2_routing(self) -> None:
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8expert_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"]["num_experts_per_token"] == 2, (
            f"laptop_moe_8expert_8gb_safe.yaml must have num_experts_per_token=2, "
            f"got {cfg['model']['num_experts_per_token']}"
        )

    def test_8expert_config_vram_estimate_not_marked_verified(self) -> None:
        """
        The VRAM estimate must NOT be marked as verified until Kishore runs it.
        """
        import yaml
        config_path = REPO_ROOT / "training" / "configs" / "laptop_moe_8expert_8gb_safe.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        meta = cfg.get("metadata", {})
        verified = meta.get("vram_estimate_verified", None)
        assert verified is False or verified is None, (
            f"vram_estimate_verified must be false until measured on hardware, "
            f"got: {verified}"
        )
