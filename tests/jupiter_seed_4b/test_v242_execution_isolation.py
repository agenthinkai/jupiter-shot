"""Jupiter Seed 4B V2.4.2 production-path exit and runtime-isolation tests.

Every CLI failure test uses a pytest-owned temporary artifact directory. These
tests never modify frozen corpus records, committed reviewer artifacts, or docs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
DATA_DIR = TRAINING_DIR / "data"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"
SCRIPT = TRAINING_DIR / "readiness_gate.py"

sys.path.insert(0, str(TRAINING_DIR))
from readiness_gate import (
    EXIT_EXECUTION_ERROR,
    EXIT_HUMAN_REVIEW_REQUIRED,
    EXIT_MECHANICAL_FAILURE,
)


def repo_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout


def run_cli(extra: list[str], timeout: int = 240) -> tuple[subprocess.CompletedProcess[str], Path]:
    artifact_dir = Path(tempfile.mkdtemp(prefix="jupiter-v242-artifact-"))
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    # Top-level production invocations must not inherit pytest's marker. Nested
    # Gate 11 tests preserve it so their CLI children hit the recursion guard
    # rather than spawning another complete authorized suite.
    if not env.get("JUPITER_GATE11_SUBPROCESS"):
        env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--artifact-dir", str(artifact_dir), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=env,
    )
    return result, artifact_dir


def assert_execution_error(result: subprocess.CompletedProcess[str], artifact_dir: Path) -> dict:
    assert result.returncode == EXIT_EXECUTION_ERROR, (
        f"expected execution-error exit 3; got {result.returncode}\n"
        f"stdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )
    artifact_path = artifact_dir / "EXECUTION_ERROR_ARTIFACT.json"
    assert artifact_path.exists(), f"missing public artifact: {artifact_path}"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "outcome", "exit_code", "error_category", "error_type",
        "safe_message", "failed_operation", "timestamp_utc", "package_release_commit",
        "package_source_commit", "corpus_sha256", "training_authorized",
    }
    assert required.issubset(artifact), artifact
    assert artifact["outcome"] == "EXECUTION_ERROR"
    assert artifact["exit_code"] == 3
    assert artifact["training_authorized"] is False
    public_text = artifact_path.read_text(encoding="utf-8").lower()
    assert "traceback" not in public_text
    assert "full_traceback" not in public_text
    assert not (artifact_dir / "EXECUTION_ERROR_INTERNAL.json").exists()
    return artifact


def copy_data(tmp: Path) -> Path:
    destination = tmp / "data"
    shutil.copytree(DATA_DIR, destination)
    return destination


def copy_docs(tmp: Path) -> Path:
    destination = tmp / "docs"
    shutil.copytree(DOCS_DIR, destination)
    return destination


def copy_bench(tmp: Path) -> Path:
    destination = tmp / "bench"
    shutil.copytree(BENCHMARK_DIR, destination)
    return destination


class TestV242RequiredEvidenceExit3:
    def test_missing_data_directory_is_exit_3_with_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-data"
            result, artifact_dir = run_cli(["--data-dir", str(missing)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["error_category"] == "missing_required_input"
            assert artifact["failed_path"] == str(missing)

    @pytest.mark.parametrize("missing_name", ["train.jsonl", "valid.jsonl", "eval.jsonl", "human_review_queue.jsonl"])
    def test_missing_each_required_corpus_file_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            (data / missing_name).unlink()
            result, artifact_dir = run_cli(["--data-dir", str(data)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["failed_path"].endswith(missing_name)

    @pytest.mark.parametrize(
        "missing_name",
        ["REVIEW_RISK_FINDINGS.json", "REVIEWER_PACKAGE.jsonl", "REVIEWER_TEMPLATE.csv", "ARABIC_HUMAN_REVIEW_QUEUE.md"],
    )
    def test_missing_each_required_reviewer_representation_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            docs = copy_docs(Path(raw))
            (docs / missing_name).unlink()
            result, artifact_dir = run_cli(["--docs-dir", str(docs)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["failed_path"].endswith(missing_name)

    def test_missing_benchmark_directory_is_exit_3(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-benchmark"
            result, artifact_dir = run_cli(["--benchmark-dir", str(missing)])
            assert_execution_error(result, artifact_dir)

    @pytest.mark.parametrize("missing_name", ["FROZEN_CORPUS_MANIFEST.json", "FROZEN_BENCHMARK_MANIFEST.json"])
    def test_missing_each_required_benchmark_manifest_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bench = copy_bench(Path(raw))
            (bench / missing_name).unlink()
            result, artifact_dir = run_cli(["--benchmark-dir", str(bench)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["failed_path"].endswith(missing_name)

    @pytest.mark.parametrize("target", ["train.jsonl", "human_review_queue.jsonl"])
    def test_malformed_jsonl_is_exit_3(self, target: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            (data / target).write_text("{not-json}\n", encoding="utf-8")
            result, artifact_dir = run_cli(["--data-dir", str(data)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["error_category"] == "malformed_jsonl"

    @pytest.mark.parametrize("target", ["FROZEN_CORPUS_MANIFEST.json", "FROZEN_BENCHMARK_MANIFEST.json"])
    def test_malformed_benchmark_manifest_is_exit_3(self, target: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bench = copy_bench(Path(raw))
            (bench / target).write_text("{not-json}\n", encoding="utf-8")
            result, artifact_dir = run_cli(["--benchmark-dir", str(bench)])
            artifact = assert_execution_error(result, artifact_dir)
            assert artifact["error_category"] == "malformed_json"

    def test_unreadable_required_file_is_exit_3(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            target = data / "train.jsonl"
            target.chmod(0o000)
            try:
                result, artifact_dir = run_cli(["--data-dir", str(data)])
                assert_execution_error(result, artifact_dir)
            finally:
                target.chmod(0o644)

    def test_output_write_failure_is_exit_3_with_safe_stderr_fallback(self) -> None:
        if not Path("/dev/full").exists():
            pytest.skip("/dev/full is unavailable on this platform")
        result, artifact_dir = run_cli(["--output", "/dev/full"])
        assert_execution_error(result, artifact_dir)
        assert "traceback" not in result.stderr.lower()

    def test_error_artifact_write_failure_remains_exit_3(self) -> None:
        # /dev/null is a file, so creating a child directory must fail safely.
        env = dict(os.environ)
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        if not env.get("JUPITER_GATE11_SUBPROCESS"):
            env.pop("PYTEST_CURRENT_TEST", None)
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-data"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--data-dir", str(missing),
                 "--artifact-dir", "/dev/null/v242-artifacts"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, env=env,
            )
        assert result.returncode == EXIT_EXECUTION_ERROR
        assert "safe fallback" in result.stderr.lower()
        assert "traceback" not in result.stderr.lower()


class TestV242RuntimeIsolation:
    def test_baseline_returns_2_and_leaves_repository_unchanged(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive baseline invocation inside Gate 11 subprocess")
        before = repo_status()
        result, artifact_dir = run_cli([])
        after = repo_status()
        assert result.returncode == EXIT_HUMAN_REVIEW_REQUIRED, result.stderr[-1000:]
        assert before == after
        assert not artifact_dir.exists() or not list(artifact_dir.iterdir())

    def test_explicit_output_is_optional_and_isolated(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive baseline invocation inside Gate 11 subprocess")
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "readiness.md"
            result, _ = run_cli(["--output", str(report)])
            assert result.returncode == EXIT_HUMAN_REVIEW_REQUIRED
            assert report.exists()
            assert "HUMAN_REVIEW_REQUIRED" in report.read_text(encoding="utf-8")

    def test_mechanical_evidence_failure_is_exit_1_not_exit_3(self) -> None:
        # Valid evidence is loaded; an identity mismatch is therefore mechanical.
        with tempfile.TemporaryDirectory() as raw:
            docs = copy_docs(Path(raw))
            sidecar_path = docs / "REVIEW_RISK_FINDINGS.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["package_commit"] = "0" * 40
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            result, artifact_dir = run_cli(["--docs-dir", str(docs)])
            assert result.returncode == EXIT_MECHANICAL_FAILURE
            assert not artifact_dir.exists() or not list(artifact_dir.iterdir())

    def test_human_review_required_baseline_is_exit_2_not_exit_3(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive baseline invocation inside Gate 11 subprocess")
        result, _ = run_cli([])
        assert result.returncode == EXIT_HUMAN_REVIEW_REQUIRED
