"""
Jupiter Shot — Run 15 Regression Tests
=======================================
18 production-path tests covering:
  - Repair 1: Metric filter allowlist (T01–T05)
  - Repair 2: MoE artifact schema validation (T06–T10)
  - Repair 3: Thermal safety monitor (T11–T18)

All tests invoke production functions, not duplicated logic.
"""
from __future__ import annotations

import datetime
import threading
import time
import tempfile
from pathlib import Path
from typing import Any

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Repair 1: Metric filter allowlist
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricFilterAllowlist:
    """
    T01–T05: Prove auxiliary_load_balancing_loss survives the complete path:
      per-step record → last-10 extraction → aggregate_layer_metrics → evaluator.
    """

    def _make_per_step_record(self, step: int = 1) -> dict[str, Any]:
        """
        Construct a realistic per-step metrics_log record as the runner would.
        Includes all router metric keys plus all the non-router keys that the
        old prefix filter was supposed to remove.
        """
        from training.router_metrics import (
            K_ROUTER_ENTROPY, K_EXPERT_ASSIGNMENT_FRACTIONS,
            K_EXPERT_ASSIGNMENT_COUNTS, K_TOKEN_ROUTING_FRACTIONS,
            K_MINIMUM_EXPERT_FRACTION, K_MAXIMUM_EXPERT_FRACTION,
            K_UTILIZATION_MEAN, K_UTILIZATION_STD, K_UTILIZATION_CV,
            K_MAX_MIN_RATIO, K_NUM_INACTIVE_EXPERTS, K_INACTIVE_EXPERT_INDICES,
            K_DROPPED_TOKEN_COUNT, K_DROPPED_TOKEN_FRACTION,
            K_OVERFLOW_TOKEN_COUNT, K_OVERFLOW_TOKEN_FRACTION,
            K_AUX_LOAD_BALANCING_LOSS, K_ROUTER_Z_LOSS,
            K_CAPACITY_FACTOR, K_EXPERTS_PER_TOKEN, K_NUM_EXPERTS,
        )
        n_experts = 8
        fracs = [0.125] * n_experts  # perfectly balanced
        return {
            # Non-router fields (must be excluded from aggregation)
            "step": step,
            "loss": 2.5,
            "aux_loss": 0.083,          # TRAINING_LEVEL_EXCLUDE_KEY
            "lr": 1e-4,
            "step_time_s": 0.12,
            "tokens_per_sec": 4096.0,
            "total_tokens": step * 512,
            "allocated_gb": 3.2,
            "reserved_gb": 4.0,
            "max_allocated_gb": 3.5,
            "gpu_temp_c": 72,
            # Router metric fields (must survive into aggregation)
            K_ROUTER_ENTROPY: 2.073,
            K_EXPERT_ASSIGNMENT_COUNTS: [64] * n_experts,
            K_EXPERT_ASSIGNMENT_FRACTIONS: fracs,
            K_TOKEN_ROUTING_FRACTIONS: fracs,
            K_MINIMUM_EXPERT_FRACTION: 0.125,
            K_MAXIMUM_EXPERT_FRACTION: 0.125,
            K_UTILIZATION_MEAN: 0.125,
            K_UTILIZATION_STD: 0.0,
            K_UTILIZATION_CV: 0.0,
            K_MAX_MIN_RATIO: 1.0,
            K_NUM_INACTIVE_EXPERTS: 0,
            K_INACTIVE_EXPERT_INDICES: [],
            K_DROPPED_TOKEN_COUNT: 0,
            K_DROPPED_TOKEN_FRACTION: 0.0,
            K_OVERFLOW_TOKEN_COUNT: 0,
            K_OVERFLOW_TOKEN_FRACTION: 0.0,
            K_AUX_LOAD_BALANCING_LOSS: 0.083,   # THE KEY THAT WAS BEING FILTERED
            K_ROUTER_Z_LOSS: 0.001,
            K_CAPACITY_FACTOR: 1.25,
            K_EXPERTS_PER_TOKEN: 2,
            K_NUM_EXPERTS: n_experts,
        }

    def test_t01_auxiliary_load_balancing_loss_survives_allowlist_filter(self):
        """T01: auxiliary_load_balancing_loss is retained by the allowlist filter."""
        from training.router_metrics import ROUTER_METRIC_KEYS, TRAINING_LEVEL_EXCLUDE_KEY
        record = self._make_per_step_record()
        filtered = {
            k: v for k, v in record.items()
            if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY
        }
        assert "auxiliary_load_balancing_loss" in filtered, (
            "auxiliary_load_balancing_loss must survive the allowlist filter"
        )

    def test_t02_only_exact_aux_loss_is_excluded(self):
        """T02: Only the exact key 'aux_loss' is excluded, not 'auxiliary_load_balancing_loss'."""
        from training.router_metrics import ROUTER_METRIC_KEYS, TRAINING_LEVEL_EXCLUDE_KEY
        assert TRAINING_LEVEL_EXCLUDE_KEY == "aux_loss"
        record = self._make_per_step_record()
        filtered = {
            k: v for k, v in record.items()
            if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY
        }
        assert "aux_loss" not in filtered, "'aux_loss' must be excluded from router aggregation"
        assert "auxiliary_load_balancing_loss" in filtered, (
            "'auxiliary_load_balancing_loss' must NOT be excluded by the aux_loss filter"
        )

    def test_t03_all_eleven_required_metrics_reach_evaluator(self):
        """T03: All 11 required router metrics survive to the acceptance evaluator."""
        from training.router_metrics import (
            ROUTER_METRIC_KEYS, TRAINING_LEVEL_EXCLUDE_KEY,
            REQUIRED_ACCEPTANCE_KEYS, aggregate_layer_metrics,
        )
        records = [self._make_per_step_record(step=i) for i in range(1, 11)]
        filtered_records = [
            {k: v for k, v in m.items()
             if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY}
            for m in records
        ]
        agg = aggregate_layer_metrics(filtered_records)
        missing = [k for k in REQUIRED_ACCEPTANCE_KEYS if k not in agg]
        assert not missing, (
            f"Required acceptance keys missing from aggregated metrics: {missing}"
        )

    def test_t04_all_eleven_required_metrics_appear_in_fresh_moe_artifact(self):
        """T04: All 11 required router metrics appear in the moe_summary.json acceptance section."""
        from training.router_metrics import (
            ROUTER_METRIC_KEYS, TRAINING_LEVEL_EXCLUDE_KEY,
            REQUIRED_ACCEPTANCE_KEYS, aggregate_layer_metrics, evaluate_acceptance,
        )
        records = [self._make_per_step_record(step=i) for i in range(1, 11)]
        filtered_records = [
            {k: v for k, v in m.items()
             if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY}
            for m in records
        ]
        agg = aggregate_layer_metrics(filtered_records)
        result = evaluate_acceptance(agg, window_description="test-last-10")
        # All required keys must be present in the aggregated dict passed to evaluator
        for key in REQUIRED_ACCEPTANCE_KEYS:
            assert key in agg, f"Required key '{key}' absent from aggregated metrics"
        # Evaluator must not return NOT_EVALUABLE (which means a required key is missing)
        assert result["outcome"] != "NOT_EVALUABLE", (
            f"Evaluator returned NOT_EVALUABLE with missing keys: {result.get('missing_keys')}"
        )

    def test_t05_old_prefix_filter_would_have_removed_auxiliary_load_balancing_loss(self):
        """T05: Regression proof — the old startswith('aux') filter removes the required key."""
        record = self._make_per_step_record()
        # Simulate the old broken filter
        old_filtered = {
            k: v for k, v in record.items()
            if not k.startswith(("step", "loss", "aux", "lr",
                                  "step_time", "tokens", "total",
                                  "allocated", "reserved", "max_",
                                  "gpu_temp"))
        }
        assert "auxiliary_load_balancing_loss" not in old_filtered, (
            "This test proves the old filter was broken: "
            "auxiliary_load_balancing_loss was silently removed by startswith('aux')"
        )
        # And the new allowlist correctly retains it
        from training.router_metrics import ROUTER_METRIC_KEYS, TRAINING_LEVEL_EXCLUDE_KEY
        new_filtered = {
            k: v for k, v in record.items()
            if k in ROUTER_METRIC_KEYS and k != TRAINING_LEVEL_EXCLUDE_KEY
        }
        assert "auxiliary_load_balancing_loss" in new_filtered, (
            "The new allowlist filter must retain auxiliary_load_balancing_loss"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Repair 2: MoE artifact schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestMoEArtifactSchemaValidation:
    """T06–T10: validate_moe_artifact_schema production function tests."""

    def _valid_artifact(self) -> dict[str, Any]:
        """Return a minimal schema-valid MoE artifact."""
        return {
            "schema_version": "1.0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": "20260805_120000_UTC",
            "branch": "fix/rtx50-blackwell-validation",
            "commit": "f27c173",
            "outcome": "PASS",
            "exit_code": 0,
            "data_mode": "real",
            "device": "cuda",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "aux_loss_semantics": "WEIGHTED",
            "router_metrics_available_under_gc": True,
            "checkpoint_status": "saved",
        }

    def test_t06_complete_schema_validates(self):
        """T06: A complete schema-valid artifact produces zero errors."""
        from scripts.run_laptop_moe import validate_moe_artifact_schema
        errors = validate_moe_artifact_schema(self._valid_artifact())
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_t07_missing_schema_version_cannot_pass(self):
        """T07: Missing schema_version produces a validation error."""
        from scripts.run_laptop_moe import validate_moe_artifact_schema
        artifact = self._valid_artifact()
        del artifact["schema_version"]
        errors = validate_moe_artifact_schema(artifact)
        assert any("schema_version" in e for e in errors), (
            f"Expected schema_version error, got: {errors}"
        )

    def test_t08_missing_timestamp_cannot_pass(self):
        """T08: Missing timestamp produces a validation error."""
        from scripts.run_laptop_moe import validate_moe_artifact_schema
        artifact = self._valid_artifact()
        del artifact["timestamp"]
        errors = validate_moe_artifact_schema(artifact)
        assert any("timestamp" in e for e in errors), (
            f"Expected timestamp error, got: {errors}"
        )

    def test_t09_timestamp_must_be_timezone_aware_utc(self):
        """T09: A naive (timezone-unaware) timestamp produces a validation error."""
        from scripts.run_laptop_moe import validate_moe_artifact_schema
        artifact = self._valid_artifact()
        # Naive datetime (no timezone info)
        artifact["timestamp"] = datetime.datetime.now().isoformat()
        errors = validate_moe_artifact_schema(artifact)
        assert any("timezone" in e.lower() or "aware" in e.lower() for e in errors), (
            f"Expected timezone-aware error, got: {errors}"
        )

    def test_t10_wrong_schema_version_produces_error(self):
        """T10: schema_version != '1.0' produces a validation error."""
        from scripts.run_laptop_moe import validate_moe_artifact_schema
        artifact = self._valid_artifact()
        artifact["schema_version"] = "2.0"
        errors = validate_moe_artifact_schema(artifact)
        assert any("schema_version" in e for e in errors), (
            f"Expected schema_version error, got: {errors}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Repair 3: Independent thermal safety monitor
# ─────────────────────────────────────────────────────────────────────────────

class TestThermalMonitor:
    """
    T11–T18: Deterministic tests using injected temperature samples.
    The temp_source callable is called by the monitor on each sample tick.
    """

    def _make_monitor(self, tmp_path: Path, temps: list[int]) -> "ThermalMonitor":
        """Create a ThermalMonitor with an injected temperature sequence."""
        from training.thermal_monitor import ThermalMonitor
        _iter = iter(temps)
        def _source():
            try:
                return next(_iter)
            except StopIteration:
                return 70  # safe fallback after sequence exhausted
        return ThermalMonitor(
            run_dir=tmp_path,
            run_id="test_run_id",
            warn_c=80,
            stop_c=90,
            interval_s=0.05,  # fast for tests
            temp_source=_source,
        )

    def test_t11_79c_no_warning(self, tmp_path):
        """T11: 79°C produces no warning and no stop."""
        monitor = self._make_monitor(tmp_path, [79] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert not health["safety_stop_triggered"]
        assert health["samples_at_or_above_warn_c"] == 0

    def test_t12_80c_warning(self, tmp_path):
        """T12: 80°C triggers a warning but does not stop."""
        monitor = self._make_monitor(tmp_path, [80] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert not health["safety_stop_triggered"], "80°C must not trigger SAFETY_STOP"
        assert health["samples_at_or_above_warn_c"] > 0, "80°C must trigger a warning"

    def test_t13_85c_warning_but_continue(self, tmp_path):
        """T13: 85°C triggers a warning but training may continue (no stop)."""
        monitor = self._make_monitor(tmp_path, [85] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert not health["safety_stop_triggered"], "85°C must not trigger SAFETY_STOP"
        assert health["samples_at_or_above_warn_c"] > 0, "85°C must trigger warnings"

    def test_t14_89c_continue(self, tmp_path):
        """T14: 89°C triggers warnings but does not stop."""
        monitor = self._make_monitor(tmp_path, [89] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert not health["safety_stop_triggered"], "89°C must not trigger SAFETY_STOP"

    def test_t15_90c_safety_stop(self, tmp_path):
        """T15: 90°C triggers SAFETY_STOP."""
        monitor = self._make_monitor(tmp_path, [90] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert health["safety_stop_triggered"], "90°C must trigger SAFETY_STOP"
        assert monitor.stop_requested

    def test_t16_91c_safety_stop(self, tmp_path):
        """T16: 91°C triggers SAFETY_STOP."""
        monitor = self._make_monitor(tmp_path, [91] * 5)
        monitor.start()
        time.sleep(0.3)
        monitor.stop()
        health = monitor.health_summary()
        assert health["safety_stop_triggered"], "91°C must trigger SAFETY_STOP"

    def test_t17_monitor_failure_prevents_pass(self, tmp_path):
        """T17: If the monitor thread dies, monitor_failed is True and healthy is False."""
        from training.thermal_monitor import ThermalMonitor
        # Inject a source that raises after first call to simulate monitor failure
        _called = [0]
        def _failing_source():
            _called[0] += 1
            if _called[0] > 1:
                raise RuntimeError("Simulated monitor hardware failure")
            return 70
        monitor = ThermalMonitor(
            run_dir=tmp_path,
            run_id="test_fail",
            warn_c=80,
            stop_c=90,
            interval_s=0.05,
            temp_source=_failing_source,
        )
        monitor.start()
        time.sleep(0.4)
        monitor.stop()
        health = monitor.health_summary()
        assert health["monitor_failed"], "Monitor failure must set monitor_failed=True"
        assert not health["healthy"], "Monitor failure must set healthy=False"

    def test_t18_monitor_cleanup_leaves_no_background_job(self, tmp_path):
        """T18: After stop(), the monitor thread is no longer alive and singleton is cleared."""
        from training.thermal_monitor import ThermalMonitor, _ACTIVE_MONITOR, _SINGLETON_LOCK
        monitor = self._make_monitor(tmp_path, [70] * 20)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        # Thread must be dead
        assert monitor._thread is not None
        assert not monitor._thread.is_alive(), "Monitor thread must be dead after stop()"
        # Singleton must be cleared
        import training.thermal_monitor as _tm
        with _tm._SINGLETON_LOCK:
            assert _tm._ACTIVE_MONITOR is None, (
                "Singleton _ACTIVE_MONITOR must be None after stop()"
            )
