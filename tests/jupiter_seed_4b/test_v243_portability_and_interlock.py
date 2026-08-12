"""V2.4.3 portability and completed training-authorization interlock tests.

CPU-only. No model download, CUDA initialization, network, cloud, paid API, or
training process is permitted. All authorization test artifacts are temporary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
CONFIG = TRAINING_DIR / "configs" / "diagnostic_qlora.yaml"
sys.path.insert(0, str(TRAINING_DIR))

import readiness_gate
import training_runner
from readiness_gate import PASS, gate_test_suite
from training_runner import (
    BASE_MODEL_ID,
    EXPECTED_CORPUS_SHA256,
    TrainingInterlockError,
    _authorization_binding_sha256,
    _resolve_commit,
    _review_package_sha256,
    authorize_non_dry_run,
    canonical_config_identity,
    canonicalize_compute_environment,
    check_training_authorization,
    run_fresh_readiness,
)


def configured_environment() -> dict:
    return {
        "device_class": "gpu",
        "execution_location": "local",
        "gpu_model": "nvidia_rtx_5060",
        "maximum_gpu_count": 1,
        "provider": None,
        "region": None,
        "approved_cost_ceiling_usd": 500.0,
    }


def readiness_payload(config_identity: dict, exit_code: int = 0) -> dict:
    return {
        "artifact_release_commit": _resolve_commit("HEAD"),
        "artifact_source_commit": _resolve_commit("HEAD^"),
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "review_package_sha256": _review_package_sha256(),
        "training_config_path": config_identity["path"],
        "training_config_sha256": config_identity["sha256"],
        "exit_code": exit_code,
    }


def valid_authorization(config_identity: dict, environment: dict, readiness_exit: int = 0) -> dict:
    readiness = readiness_payload(config_identity, readiness_exit)
    auth = {
        "schema_version": "2.0",
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "dataset_content_commit": "2e36f6b977a8af052fced5a532c1168dc1988b6f",
        "readiness_package_commit": _resolve_commit("HEAD"),
        "human_review_status": "COMPLETE",
        "accepted_count": 50,
        "revised_count": 0,
        "rejected_count": 0,
        "unresolved_count": 0,
        "legal_review_status": "APPROVED",
        "approved_model_id": BASE_MODEL_ID,
        "approval_scope": "internal test fixture only",
        "maximum_budget_usd": 500.0,
        "approved_by": "Named Human Authorizer",
        "approved_at_utc": "2026-08-12T00:00:00Z",
        "authorization_status": "APPROVED",
        "signature_or_typed_confirmation": "I Named Human Authorizer authorize the approved test fixture.",
        "approved_training_configuration": dict(config_identity),
        "approved_compute_environment": dict(environment),
        "approved_readiness": {**readiness, "authorization_binding_sha256": ""},
    }
    auth["approved_readiness"]["authorization_binding_sha256"] = _authorization_binding_sha256(auth)
    return auth


def write_auth(path: Path, auth: dict) -> None:
    path.write_text(json.dumps(auth, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_blocked_before_training(monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace) -> None:
    reached = []
    monkeypatch.setattr(training_runner, "parse_args", lambda: args)
    monkeypatch.setattr(training_runner, "run_training", lambda *a, **k: reached.append("training"))
    with pytest.raises(SystemExit) as exc:
        training_runner.main()
    assert exc.value.code == 1
    assert not reached, "run_training is the heavy-operation boundary and must not be reached"


class TestGate11ReplacementDecoding:
    def test_subprocess_replacement_decoding_handles_invalid_bytes_without_traceback(self) -> None:
        child = subprocess.run(
            [sys.executable, "-c", "import os; os.write(1, b'\\x97pytest output')"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        assert child.returncode == 0
        assert "\ufffd" in child.stdout
        assert "traceback" not in child.stderr.lower()

    def test_gate11_passes_replacement_text_and_preserves_structured_result(self) -> None:
        xml = '<?xml version="1.0"?><testsuite tests="2" failures="0" errors="0" skipped="0"/>'
        with tempfile.TemporaryDirectory() as raw:
            result_path = Path(raw) / "results.xml"
            result_path.write_text(xml, encoding="utf-8")

            class FakeTemporaryDirectory:
                def __enter__(self):
                    return raw
                def __exit__(self, *args):
                    return False

            fake_version = MagicMock(returncode=0, stdout="pytest \ufffd", stderr="")
            fake_run = MagicMock(returncode=0, stdout="\ufffd all passed", stderr="")
            with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}), \
                 patch("readiness_gate.subprocess.run", side_effect=[fake_version, fake_run]) as run, \
                 patch("readiness_gate.tempfile.TemporaryDirectory", return_value=FakeTemporaryDirectory()), \
                 patch("readiness_gate._check_test_manifest", return_value=[]), \
                 patch("readiness_gate._compute_test_manifest_hash", return_value=("hash", [])):
                result = gate_test_suite(REPO_ROOT)
        assert result.status == PASS
        assert result.metadata["exit_code"] == 0
        assert result.metadata["passed"] == 2
        for call in run.call_args_list:
            assert call.kwargs.get("encoding") == "utf-8"
            assert call.kwargs.get("errors") == "replace"


class TestV243AuthorizationInterlock:
    def _context(self, tmp: Path) -> tuple[Path, dict, dict, dict]:
        config_identity = canonical_config_identity(CONFIG)
        environment = canonicalize_compute_environment(configured_environment(), "requested")
        auth = valid_authorization(config_identity, environment)
        auth_path = tmp / "authorization.json"
        write_auth(auth_path, auth)
        return auth_path, config_identity, environment, auth

    def test_exact_config_and_environment_static_authorization_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, _ = self._context(Path(raw))
            returned = check_training_authorization(
                auth_path, BASE_MODEL_ID, 500.0, config, env, _resolve_commit("HEAD"),
            )
        assert returned["approved_training_configuration"] == config

    def test_config_hash_mismatch_blocks_and_alias_cannot_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            alias = Path(os.path.relpath(CONFIG, REPO_ROOT))
            assert canonical_config_identity(alias) == config
            auth["approved_training_configuration"]["sha256"] = "0" * 64
            write_auth(auth_path, auth)
            with pytest.raises(SystemExit):
                check_training_authorization(auth_path, BASE_MODEL_ID, 500.0, config, env, _resolve_commit("HEAD"))

    @pytest.mark.parametrize("mutation", [
        lambda env: env.update({"maximum_gpu_count": 2}),
        lambda env: env.update({"gpu_model": "nvidia_a100"}),
        lambda env: env.update({"provider": "example"}),
    ])
    def test_compute_environment_mismatch_blocks(self, mutation) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            mutation(auth["approved_compute_environment"])
            write_auth(auth_path, auth)
            with pytest.raises(SystemExit):
                check_training_authorization(auth_path, BASE_MODEL_ID, 500.0, config, env, _resolve_commit("HEAD"))

    @pytest.mark.parametrize("field", ["provider", "region", "unexpected_field"])
    def test_missing_or_extra_compute_environment_material_fields_block(self, field: str) -> None:
        raw = configured_environment()
        if field == "unexpected_field":
            raw[field] = "unexpected"
        else:
            raw.pop(field)
        with pytest.raises(TrainingInterlockError):
            canonicalize_compute_environment(raw, "requested")

    @pytest.mark.parametrize("exit_code", [1, 2, 3])
    def test_readiness_exit_not_zero_blocks_before_training(self, exit_code: int) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            readiness = readiness_payload(config, exit_code)
            auth["approved_readiness"] = {**readiness, "authorization_binding_sha256": ""}
            auth["approved_readiness"]["authorization_binding_sha256"] = _authorization_binding_sha256(auth)
            write_auth(auth_path, auth)
            with pytest.raises(TrainingInterlockError):
                authorize_non_dry_run(auth_path, CONFIG, env, BASE_MODEL_ID, 500.0,
                                      readiness_runner=lambda _: readiness)

    @pytest.mark.parametrize("field,value", [
        ("artifact_release_commit", "0" * 40),
        ("artifact_source_commit", "0" * 40),
        ("corpus_sha256", "0" * 64),
        ("training_config_sha256", "0" * 64),
    ])
    def test_stale_or_mismatched_readiness_binding_blocks(self, field: str, value: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            auth["approved_readiness"][field] = value
            auth["approved_readiness"]["authorization_binding_sha256"] = _authorization_binding_sha256(auth)
            write_auth(auth_path, auth)
            with pytest.raises(TrainingInterlockError):
                authorize_non_dry_run(auth_path, CONFIG, env, BASE_MODEL_ID, 500.0,
                                      readiness_runner=lambda _: readiness_payload(config, 0))

    @pytest.mark.parametrize("commit", ["", "abc1234", "not-a-commit", _resolve_commit("HEAD^")])
    def test_missing_short_malformed_or_source_readiness_package_commit_blocks(self, commit: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            auth["readiness_package_commit"] = commit
            write_auth(auth_path, auth)
            with pytest.raises(SystemExit):
                check_training_authorization(auth_path, BASE_MODEL_ID, 500.0, config, env, _resolve_commit("HEAD"))

    def test_unrelated_valid_commit_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            auth["readiness_package_commit"] = "2e36f6b977a8af052fced5a532c1168dc1988b6f"
            write_auth(auth_path, auth)
            with pytest.raises(SystemExit):
                check_training_authorization(auth_path, BASE_MODEL_ID, 500.0, config, env, _resolve_commit("HEAD"))

    def test_actual_fresh_readiness_baseline_returns_exit_2(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive readiness invocation inside Gate 11 subprocess")
        config = canonical_config_identity(CONFIG)
        result = run_fresh_readiness(config)
        assert result["exit_code"] == 2

    def test_main_rejection_is_before_training_boundary_for_invalid_compute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as raw:
            compute_path = Path(raw) / "bad-compute.json"
            compute_path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            args = SimpleNamespace(
                config=CONFIG, dataset_path=None, dry_run=False, seed=None,
                compute_environment=compute_path, training_authorization=Path(raw) / "missing-auth.json",
            )
            assert_blocked_before_training(monkeypatch, args)

    def test_main_rejection_is_before_training_boundary_for_baseline_exit_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive readiness invocation inside Gate 11 subprocess")
        with tempfile.TemporaryDirectory() as raw:
            auth_path, config, env, auth = self._context(Path(raw))
            compute_path = Path(raw) / "compute.json"
            compute_path.write_text(json.dumps(env), encoding="utf-8")
            # Bind authorization to the real baseline exit 2 so static authorization passes.
            baseline = readiness_payload(config, 2)
            auth["approved_readiness"] = {**baseline, "authorization_binding_sha256": ""}
            auth["approved_readiness"]["authorization_binding_sha256"] = _authorization_binding_sha256(auth)
            write_auth(auth_path, auth)
            args = SimpleNamespace(
                config=CONFIG, dataset_path=None, dry_run=False, seed=None,
                compute_environment=compute_path, training_authorization=auth_path,
            )
            assert_blocked_before_training(monkeypatch, args)
