"""Jupiter Seed 4B V2.4.2 production-path exit and runtime-isolation tests.

Every CLI failure test uses a pytest-owned temporary artifact directory. These
tests never modify frozen corpus records, committed reviewer artifacts, or docs.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CliRun:
    """Immutable CLI evidence retained after the helper deletes its owned directory."""
    result: subprocess.CompletedProcess[str]
    artifact_bytes: bytes | None
    owned_directory_removed: bool


def run_cli(
    extra: list[str],
    timeout: int = 240,
    temp_root: Path | None = None,
) -> CliRun:
    """Run the production CLI and deterministically remove only this invocation's temp tree."""
    temp_kwargs: dict[str, object] = {"prefix": "jupiter-v242-artifact-"}
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_kwargs["dir"] = str(temp_root)
    owned_dir: Path | None = None
    artifact_bytes: bytes | None = None
    with tempfile.TemporaryDirectory(**temp_kwargs) as raw:
        owned_dir = Path(raw)
        artifact_dir = owned_dir / "artifacts"
        env = dict(os.environ)
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        # Top-level production invocations must not inherit pytest's marker.
        if not env.get("JUPITER_GATE11_SUBPROCESS"):
            env.pop("PYTEST_CURRENT_TEST", None)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--artifact-dir", str(artifact_dir), *extra],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=env,
        )
        artifact_path = artifact_dir / "EXECUTION_ERROR_ARTIFACT.json"
        if artifact_path.exists():
            artifact_bytes = artifact_path.read_bytes()
    assert owned_dir is not None
    return CliRun(
        result=result,
        artifact_bytes=artifact_bytes,
        owned_directory_removed=not owned_dir.exists(),
    )


def assert_execution_error(run: CliRun) -> dict:
    result = run.result
    assert result.returncode == EXIT_EXECUTION_ERROR, (
        f"expected execution-error exit 3; got {result.returncode}\n"
        f"stdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
    )
    assert run.owned_directory_removed, "run_cli leaked its owned temporary directory"
    assert run.artifact_bytes is not None, "missing captured public execution artifact"
    artifact = json.loads(run.artifact_bytes.decode("utf-8"))
    required = {
        "schema_version", "outcome", "exit_code", "error_category", "error_type",
        "safe_message", "failed_operation", "timestamp_utc", "package_release_commit",
        "package_source_commit", "corpus_sha256", "training_authorized",
    }
    assert required.issubset(artifact), artifact
    assert artifact["outcome"] == "EXECUTION_ERROR"
    assert artifact["exit_code"] == 3
    assert artifact["training_authorized"] is False
    public_text = run.artifact_bytes.decode("utf-8").lower()
    assert "traceback" not in public_text
    assert "full_traceback" not in public_text
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


def _can_read(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.read(1)
        return True
    except OSError:
        return False


def _can_create_child(parent: Path) -> bool:
    probe = parent / f"probe-{uuid.uuid4().hex}"
    try:
        probe.mkdir()
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def _windows_icacls() -> str | None:
    if os.name != "nt":
        return None
    executable = shutil.which("icacls")
    if not executable:
        pytest.skip("Windows ACL test skipped: icacls is unavailable")
    return executable


def _windows_identity() -> str:
    result = subprocess.run(
        ["whoami"], capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    identity = result.stdout.strip()
    if result.returncode != 0 or not identity:
        pytest.skip("Windows ACL test skipped: unable to resolve exact current identity")
    return identity


def _icacls_run(icacls: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [icacls, *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _icacls_snapshot(icacls: str, path: Path, *, preflight: bool = False) -> str:
    result = _icacls_run(icacls, str(path))
    if result.returncode != 0:
        if preflight:
            pytest.skip("Windows ACL test skipped: unable to inspect pytest temporary-path ACL before denial")
        pytest.fail("Windows ACL test failure: unable to inspect ACL after DENY operation began")
    return result.stdout.replace("\r\n", "\n").strip()


def _has_explicit_deny(snapshot: str, identity: str) -> bool:
    """Detect the deny entry for the exact identity, not merely any DENY ACE."""
    folded_identity = identity.casefold()
    return any(
        folded_identity in line.casefold() and "(deny)" in line.casefold()
        for line in snapshot.splitlines()
    )


def _icacls_failure_text(result: subprocess.CompletedProcess[str]) -> bool:
    text = (result.stdout + "\n" + result.stderr).casefold()
    if "failed processing" in text and "failed processing 0 files" not in text:
        return True
    return any(token in text for token in ("access is denied", "error ", "unable to", "failed to"))


def _remove_exact_test_deny(icacls: str, path: Path, identity: str) -> str:
    """Remove only the exact deny ACE added by this test; never use /c, /save, or /restore."""
    first = _icacls_run(icacls, str(path), "/remove:d", identity)
    if first.returncode != 0 or _icacls_failure_text(first):
        # A single narrowly scoped retry is the safest recovery; no descriptor restore is attempted.
        retry = _icacls_run(icacls, str(path), "/remove:d", identity)
        if retry.returncode != 0 or _icacls_failure_text(retry):
            pytest.fail(
                "Windows ACL cleanup failed for exact test-owned DENY ACE: "
                f"first={first.stdout!r}/{first.stderr!r}; retry={retry.stdout!r}/{retry.stderr!r}"
            )
    after = _icacls_snapshot(icacls, path)
    if _has_explicit_deny(after, identity):
        pytest.fail("Windows ACL cleanup failed: exact test-owned DENY ACE remains after /remove:d")
    return after


@dataclass
class WindowsDenyLease:
    path: Path
    identity: str
    permission: str
    acl_before: str
    parent_acl_before: str
    acl_during: str = ""
    acl_after: str = ""


@contextmanager
def _windows_acl_deny(path: Path, permission: str):
    """Apply a narrow test-owned DENY ACE and guarantee verified cleanup after every path.

    File content denial uses ``RD`` (ReadData), not generic ``R``: RD denies content
    reads while preserving READ_CONTROL so the identity can inspect and repair the ACL.
    The helper never uses descriptor save/restore and never uses ``/c`` for cleanup.
    """
    icacls = _windows_icacls()
    assert icacls is not None
    identity = _windows_identity()
    before = _icacls_snapshot(icacls, path, preflight=True)
    parent_before = _icacls_snapshot(icacls, path.parent, preflight=True)
    lease = WindowsDenyLease(
        path=path, identity=identity, permission=permission,
        acl_before=before, parent_acl_before=parent_before,
    )
    deny_started = False
    try:
        deny_started = True
        applied = _icacls_run(icacls, str(path), "/deny", f"{identity}:({permission})")
        lease.acl_during = _icacls_snapshot(icacls, path)
        if applied.returncode != 0 or _icacls_failure_text(applied):
            if _has_explicit_deny(lease.acl_during, identity):
                pytest.fail("Windows ACL denial command reported failure after applying a DENY ACE")
            pytest.skip("Windows ACL test skipped: safe test-owned DENY ACE could not be applied")
        if not _has_explicit_deny(lease.acl_during, identity):
            pytest.skip("Windows ACL test skipped: test-owned DENY ACE was not observable")
        yield lease
    finally:
        if deny_started:
            current = _icacls_snapshot(icacls, path)
            if _has_explicit_deny(current, identity):
                lease.acl_after = _remove_exact_test_deny(icacls, path, identity)
            else:
                lease.acl_after = current
            parent_after = _icacls_snapshot(icacls, path.parent)
            if lease.acl_after != lease.acl_before:
                pytest.fail("Windows ACL cleanup changed the temporary file ACL beyond the exact test DENY ACE")
            if parent_after != lease.parent_acl_before:
                pytest.fail("Windows ACL cleanup changed a parent or inherited ACL outside the test-owned file ACE")


@contextmanager
def verified_unreadable_file(path: Path):
    """Make a temporary file actually unreadable, or explicitly skip."""
    if os.name == "nt":
        with _windows_acl_deny(path, "RD") as lease:
            if _can_read(path):
                pytest.skip("Windows ACL denial did not make the temporary file unreadable")
            assert _has_explicit_deny(lease.acl_during, lease.identity)
            yield
        if not _can_read(path):
            pytest.fail("Windows ACL cleanup failed: temporary file remains unreadable")
        return

    original_mode = path.stat().st_mode
    path.chmod(0)
    try:
        if _can_read(path):
            pytest.skip("POSIX environment can still read chmod(000) temporary file")
        yield
    finally:
        path.chmod(original_mode)


@contextmanager
def verified_unwritable_artifact_parent(parent: Path):
    """Make a temporary parent genuinely unwritable for child creation, or skip."""
    if os.name == "nt":
        with _windows_acl_deny(parent, "W") as lease:
            if _can_create_child(parent):
                pytest.skip("Windows ACL denial did not make temporary artifact parent unwritable")
            assert _has_explicit_deny(lease.acl_during, lease.identity)
            yield parent / "runtime-artifacts"
        if not _can_create_child(parent):
            pytest.fail("Windows ACL cleanup failed: temporary artifact parent remains unwritable")
        return

    original_mode = parent.stat().st_mode
    parent.chmod(0o500)
    try:
        if _can_create_child(parent):
            pytest.skip("POSIX environment can still create in chmod(0500) temporary directory")
        yield parent / "runtime-artifacts"
    finally:
        parent.chmod(original_mode)


class TestV242RequiredEvidenceExit3:
    def test_missing_data_directory_is_exit_3_with_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-data"
            run = run_cli(["--data-dir", str(missing)])
            artifact = assert_execution_error(run)
            assert artifact["error_category"] == "missing_required_input"
            assert artifact["failed_path"] == str(missing)

    @pytest.mark.parametrize("missing_name", ["train.jsonl", "valid.jsonl", "eval.jsonl", "human_review_queue.jsonl"])
    def test_missing_each_required_corpus_file_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            (data / missing_name).unlink()
            run = run_cli(["--data-dir", str(data)])
            artifact = assert_execution_error(run)
            assert artifact["failed_path"].endswith(missing_name)

    @pytest.mark.parametrize(
        "missing_name",
        ["REVIEW_RISK_FINDINGS.json", "REVIEWER_PACKAGE.jsonl", "REVIEWER_TEMPLATE.csv", "ARABIC_HUMAN_REVIEW_QUEUE.md"],
    )
    def test_missing_each_required_reviewer_representation_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            docs = copy_docs(Path(raw))
            (docs / missing_name).unlink()
            run = run_cli(["--docs-dir", str(docs)])
            artifact = assert_execution_error(run)
            assert artifact["failed_path"].endswith(missing_name)

    def test_missing_benchmark_directory_is_exit_3(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-benchmark"
            run = run_cli(["--benchmark-dir", str(missing)])
            assert_execution_error(run)

    @pytest.mark.parametrize("missing_name", ["FROZEN_CORPUS_MANIFEST.json", "FROZEN_BENCHMARK_MANIFEST.json"])
    def test_missing_each_required_benchmark_manifest_is_exit_3(self, missing_name: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bench = copy_bench(Path(raw))
            (bench / missing_name).unlink()
            run = run_cli(["--benchmark-dir", str(bench)])
            artifact = assert_execution_error(run)
            assert artifact["failed_path"].endswith(missing_name)

    @pytest.mark.parametrize("target", ["train.jsonl", "human_review_queue.jsonl"])
    def test_malformed_jsonl_is_exit_3(self, target: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            (data / target).write_text("{not-json}\n", encoding="utf-8")
            run = run_cli(["--data-dir", str(data)])
            artifact = assert_execution_error(run)
            assert artifact["error_category"] == "malformed_jsonl"

    @pytest.mark.parametrize("target", ["FROZEN_CORPUS_MANIFEST.json", "FROZEN_BENCHMARK_MANIFEST.json"])
    def test_malformed_benchmark_manifest_is_exit_3(self, target: str) -> None:
        with tempfile.TemporaryDirectory() as raw:
            bench = copy_bench(Path(raw))
            (bench / target).write_text("{not-json}\n", encoding="utf-8")
            run = run_cli(["--benchmark-dir", str(bench)])
            artifact = assert_execution_error(run)
            assert artifact["error_category"] == "malformed_json"

    def test_unreadable_required_file_is_exit_3(self) -> None:
        """Use verified Windows ACL or POSIX permission denial only inside pytest temp space."""
        with tempfile.TemporaryDirectory() as raw:
            data = copy_data(Path(raw))
            target = data / "train.jsonl"
            original_mode = target.stat().st_mode
            with verified_unreadable_file(target):
                assert not _can_read(target), "test setup must prove the file is actually unreadable"
                run = run_cli(["--data-dir", str(data)])
                artifact = assert_execution_error(run)
                assert artifact["error_category"] == "unreadable_required_input"
            # POSIX restoration is checked directly; Windows cleanup removes only the test-owned deny ACE.
            if os.name != "nt":
                assert target.stat().st_mode == original_mode

    def test_output_write_failure_is_exit_3_with_safe_stderr_fallback(self) -> None:
        if not Path("/dev/full").exists():
            pytest.skip("/dev/full is unavailable on this platform")
        run = run_cli(["--output", "/dev/full"])
        assert_execution_error(run)
        assert "traceback" not in run.result.stderr.lower()

    def test_error_artifact_write_failure_remains_exit_3(self) -> None:
        """Verify a real unwritable temporary destination; never depend on /dev or null paths."""
        env = dict(os.environ)
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        if not env.get("JUPITER_GATE11_SUBPROCESS"):
            env.pop("PYTEST_CURRENT_TEST", None)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            missing = root / "missing-data"
            parent = root / "unwritable-parent"
            parent.mkdir()
            with verified_unwritable_artifact_parent(parent) as artifact_dir:
                assert not _can_create_child(parent), "test setup must prove artifact parent is unwritable"
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--data-dir", str(missing),
                     "--artifact-dir", str(artifact_dir)],
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
        run = run_cli([])
        after = repo_status()
        assert run.result.returncode == EXIT_HUMAN_REVIEW_REQUIRED, run.result.stderr[-1000:]
        assert before == after
        assert run.owned_directory_removed
        assert run.artifact_bytes is None

    def test_explicit_output_is_optional_and_isolated(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive baseline invocation inside Gate 11 subprocess")
        with tempfile.TemporaryDirectory() as raw:
            report = Path(raw) / "readiness.md"
            run = run_cli(["--output", str(report)])
            assert run.result.returncode == EXIT_HUMAN_REVIEW_REQUIRED
            assert run.owned_directory_removed
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
            run = run_cli(["--docs-dir", str(docs)])
            assert run.result.returncode == EXIT_MECHANICAL_FAILURE
            assert run.owned_directory_removed
            assert run.artifact_bytes is None

    def test_human_review_required_baseline_is_exit_2_not_exit_3(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive baseline invocation inside Gate 11 subprocess")
        run = run_cli([])
        assert run.result.returncode == EXIT_HUMAN_REVIEW_REQUIRED
        assert run.owned_directory_removed
