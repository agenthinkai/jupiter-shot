"""
Jupiter Shot — Run 11 Exit-Code Contract Tests
================================================
Tests the _validate_runner_artifact() function directly with 16 scenarios,
and the step10b_moe_aux_loss_verification() function with 8 MoE aux-loss scenarios.

These are executable logic tests — they call the actual functions, not doc-string matching.

Run with:
    python3 -m pytest tests/test_run11_exit_code_matrix.py -v
"""
from __future__ import annotations

import datetime
import json
import math
import pathlib
import sys
import tempfile
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_laptop_validation_pipeline import (
    EXIT_EXECUTION_ERROR,
    EXIT_NOT_ACCEPTED,
    EXIT_NOT_EVALUABLE,
    EXIT_PASS,
    EXIT_SAFETY_STOP,
    _validate_runner_artifact,
    step10b_moe_aux_loss_verification,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"
RUN_ID = "test_run_20260804_120000_UTC"
RUN_START = datetime.datetime(2026, 8, 4, 12, 0, 0, tzinfo=datetime.timezone.utc)
ARTIFACT_TS = "2026-08-04T12:00:30+00:00"  # 30 seconds after run_start


def _write_artifact(tmp: pathlib.Path, data: dict) -> None:
    (tmp / "dense_summary.json").write_text(json.dumps(data))


def _valid_artifact(**overrides) -> dict:
    base = {
        "schema_version": SCHEMA_VERSION,
        "outcome": "PASS",
        "exit_code": EXIT_PASS,
        "run_id": RUN_ID,
        "timestamp": ARTIFACT_TS,
    }
    base.update(overrides)
    return base


# ── 16-scenario exit-code matrix ─────────────────────────────────────────────

class TestValidateRunnerArtifact:
    """16 scenarios for _validate_runner_artifact()."""

    def _run(self, tmp: pathlib.Path, raw: int) -> int:
        return _validate_runner_artifact(
            name="dense",
            run_dir=tmp,
            run_id=RUN_ID,
            raw_returncode=raw,
            run_start=RUN_START,
        )

    # Scenario 1: argparse exit 2, no artifact → EXECUTION_ERROR (3)
    def test_s01_argparse_exit2_no_artifact(self, tmp_path):
        result = self._run(tmp_path, raw=2)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S01: argparse exit 2 with no artifact must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 2: raw exit 0, no artifact → EXECUTION_ERROR (3)
    def test_s02_exit0_no_artifact(self, tmp_path):
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S02: exit 0 with no artifact must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 3: artifact exists but is malformed JSON → EXECUTION_ERROR (3)
    def test_s03_malformed_json(self, tmp_path):
        (tmp_path / "dense_summary.json").write_text("{not valid json")
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S03: malformed JSON must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 4: artifact missing 'outcome' field → EXECUTION_ERROR (3)
    def test_s04_missing_outcome_field(self, tmp_path):
        art = _valid_artifact()
        del art["outcome"]
        _write_artifact(tmp_path, art)
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S04: missing 'outcome' must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 5: artifact missing 'exit_code' field → EXECUTION_ERROR (3)
    def test_s05_missing_exit_code_field(self, tmp_path):
        art = _valid_artifact()
        del art["exit_code"]
        _write_artifact(tmp_path, art)
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S05: missing 'exit_code' must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 6: wrong schema_version → EXECUTION_ERROR (3)
    def test_s06_wrong_schema_version(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(schema_version="0.9"))
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S06: wrong schema_version must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 7: stale run_id (different from expected) → EXECUTION_ERROR (3)
    def test_s07_stale_run_id(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(run_id="old_run_20260803_000000_UTC"))
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S07: stale run_id must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 8: artifact timestamp > 1 hour before run_start → EXECUTION_ERROR (3)
    def test_s08_stale_timestamp(self, tmp_path):
        old_ts = "2026-08-04T10:00:00+00:00"  # 2 hours before run_start
        _write_artifact(tmp_path, _valid_artifact(timestamp=old_ts))
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S08: stale timestamp must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 9: outcome/exit_code inconsistency → EXECUTION_ERROR (3)
    def test_s09_outcome_exit_code_mismatch(self, tmp_path):
        # outcome=PASS but exit_code=1 (NOT_ACCEPTED)
        _write_artifact(tmp_path, _valid_artifact(outcome="PASS", exit_code=1))
        result = self._run(tmp_path, raw=1)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S09: outcome/exit_code mismatch must be EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 10: process returncode disagrees with artifact exit_code → trust artifact
    def test_s10_process_code_disagrees_with_artifact(self, tmp_path):
        # artifact says PASS (0) but process returned 1
        _write_artifact(tmp_path, _valid_artifact(outcome="PASS", exit_code=0))
        result = self._run(tmp_path, raw=1)
        # Per Rule 9: trust artifact over raw process code
        assert result == EXIT_PASS, (
            f"S10: when artifact says PASS but process returned 1, must trust artifact → PASS (0), got {result}"
        )

    # Scenario 11: valid PASS artifact, raw exit 0 → PASS (0)
    def test_s11_valid_pass(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="PASS", exit_code=0))
        result = self._run(tmp_path, raw=0)
        assert result == EXIT_PASS, (
            f"S11: valid PASS artifact must return PASS (0), got {result}"
        )

    # Scenario 12: valid NOT_ACCEPTED artifact, raw exit 1 → NOT_ACCEPTED (1)
    def test_s12_valid_not_accepted(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="NOT_ACCEPTED", exit_code=1))
        result = self._run(tmp_path, raw=1)
        assert result == EXIT_NOT_ACCEPTED, (
            f"S12: valid NOT_ACCEPTED artifact must return NOT_ACCEPTED (1), got {result}"
        )

    # Scenario 13: valid NOT_EVALUABLE artifact, raw exit 2 → NOT_EVALUABLE (2)
    # This is the ONLY case where exit 2 is valid — when the artifact explicitly says NOT_EVALUABLE
    def test_s13_valid_not_evaluable_from_artifact(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="NOT_EVALUABLE", exit_code=2))
        result = self._run(tmp_path, raw=2)
        assert result == EXIT_NOT_EVALUABLE, (
            f"S13: valid NOT_EVALUABLE artifact must return NOT_EVALUABLE (2), got {result}"
        )

    # Scenario 14: valid EXECUTION_ERROR artifact, raw exit 3 → EXECUTION_ERROR (3)
    def test_s14_valid_execution_error(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="EXECUTION_ERROR", exit_code=3))
        result = self._run(tmp_path, raw=3)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S14: valid EXECUTION_ERROR artifact must return EXECUTION_ERROR (3), got {result}"
        )

    # Scenario 15: valid SAFETY_STOP artifact, raw exit 4 → SAFETY_STOP (4)
    def test_s15_valid_safety_stop(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="SAFETY_STOP", exit_code=4))
        result = self._run(tmp_path, raw=4)
        assert result == EXIT_SAFETY_STOP, (
            f"S15: valid SAFETY_STOP artifact must return SAFETY_STOP (4), got {result}"
        )

    # Scenario 16: unknown outcome in artifact → EXECUTION_ERROR (3)
    def test_s16_unknown_outcome(self, tmp_path):
        _write_artifact(tmp_path, _valid_artifact(outcome="WEIRD_OUTCOME", exit_code=99))
        result = self._run(tmp_path, raw=99)
        assert result == EXIT_EXECUTION_ERROR, (
            f"S16: unknown outcome must be EXECUTION_ERROR (3), got {result}"
        )


# ── 8 MoE aux-loss regression tests ──────────────────────────────────────────

def _make_moe_model(
    has_config: bool = True,
    has_moe_config: bool = True,
    aux_loss_coef: float = 0.01,
    gradient_checkpointing: bool = True,
    has_layers: bool = True,
    has_router: bool = True,
    has_experts: bool = True,
) -> object:
    """Build a minimal mock MoE model for testing step10b."""
    model = types.SimpleNamespace()

    if has_config:
        cfg = types.SimpleNamespace()
        if has_moe_config:
            moe_cfg = types.SimpleNamespace()
            if aux_loss_coef is not None:
                moe_cfg.aux_loss_coef = aux_loss_coef
            cfg.moe = moe_cfg
        cfg.gradient_checkpointing = gradient_checkpointing
        model.config = cfg

    if has_layers:
        router = types.SimpleNamespace() if has_router else None
        experts = types.SimpleNamespace() if has_experts else None
        moe_sub = types.SimpleNamespace()
        if has_router:
            moe_sub.router = router
        if has_experts:
            moe_sub.experts = experts
        layer = types.SimpleNamespace(moe=moe_sub)
        model.layers = [layer]

    return model


class TestStep10bMoeAuxLoss:
    """8 scenarios for step10b_moe_aux_loss_verification()."""

    def _run(self, model, step10_result: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            return step10b_moe_aux_loss_verification(
                moe_model=model,
                step10_result=step10_result,
                errors_path=pathlib.Path(tmp) / "errors.jsonl",
            )

    def _good_step10(self, aux_loss: float = 0.042) -> dict:
        return {"status": "ok", "moe_cpu_loss": 10.5, "moe_cpu_aux_loss": aux_loss}

    # Test A: aux_loss > 0 with gradient_checkpointing=True → all 15 checks pass
    def test_a_aux_loss_positive_gc_true(self):
        model = _make_moe_model(gradient_checkpointing=True)
        result = self._run(model, self._good_step10(aux_loss=0.042))
        assert result["status"] == "ok"
        assert result["n_passed"] == result["n_total"]
        assert result["checks"]["c04_aux_loss_positive"]["passed"] is True
        assert result["checks"]["c15_gc_aux_nonzero"]["passed"] is True

    # Test B: aux_loss == 0.0 → c04 fails (Defect 3 not fixed)
    def test_b_aux_loss_zero_raises(self):
        model = _make_moe_model(gradient_checkpointing=True)
        with pytest.raises(Exception, match="Defect 3 NOT fixed|FAILED"):
            self._run(model, self._good_step10(aux_loss=0.0))

    # Test C: aux_loss < 0 → c04 fails
    def test_c_aux_loss_negative_raises(self):
        model = _make_moe_model(gradient_checkpointing=True)
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, self._good_step10(aux_loss=-0.001))

    # Test D: aux_loss >= 1.0 → c05 fails (dominating training loss)
    def test_d_aux_loss_too_large_raises(self):
        model = _make_moe_model(gradient_checkpointing=True)
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, self._good_step10(aux_loss=1.5))

    # Test E: step10 status != 'ok' → c01 fails
    def test_e_step10_failed_raises(self):
        model = _make_moe_model()
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, {"status": "FAILED", "error": "cuda OOM"})

    # Test F: moe_cpu_aux_loss key missing → c02 fails
    def test_f_aux_loss_key_missing_raises(self):
        model = _make_moe_model()
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, {"status": "ok", "moe_cpu_loss": 10.5})

    # Test G: aux_loss_coef == 0 → c09 fails
    def test_g_aux_loss_coef_zero_raises(self):
        model = _make_moe_model(aux_loss_coef=0.0)
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, self._good_step10(aux_loss=0.042))

    # Test H: gradient_checkpointing=True, aux_loss=0 → c15 fails
    def test_h_gc_true_aux_zero_raises(self):
        model = _make_moe_model(gradient_checkpointing=True)
        with pytest.raises(Exception, match="FAILED"):
            self._run(model, self._good_step10(aux_loss=0.0))
