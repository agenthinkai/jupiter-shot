"""V2.4.4 functional runtime-isolation and Windows ACL-cleanup regressions.

Windows ACL cases execute only on a genuine Windows host with ``icacls``. They
exercise the production helper on pytest-owned temporary paths; they never use
static source assertions as proof of ACL safety.
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
SCRIPT = TRAINING_DIR / "readiness_gate.py"
sys.path.insert(0, str(Path(__file__).parent))
import test_v242_execution_isolation as acl
sys.path.insert(0, str(TRAINING_DIR))
from readiness_gate import EXIT_EXECUTION_ERROR, EXIT_HUMAN_REVIEW_REQUIRED


def _tree_fingerprint(root: Path) -> list[str]:
    """Return a sorted non-Git tree fingerprint, including ignored cache paths."""
    items: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        suffix = "/" if path.is_dir() else ""
        items.append(relative.as_posix() + suffix)
    return sorted(items)


def _run_gate(extra: list[str] | None = None, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    if not env.get("JUPITER_GATE11_SUBPROCESS"):
        env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(extra or [])], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=env,
    )


def _require_windows_icacls() -> None:
    if os.name != "nt":
        pytest.skip("Windows ACL functional verification requires a genuine Windows host")
    if not shutil.which("icacls"):
        pytest.skip("Windows ACL functional verification requires icacls")


class TestGate11CacheIsolation:
    def test_gate11_cache_is_outside_repository_and_repeated_runs_preserve_tree(self) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive production Gate 11 invocation inside Gate 11 subprocess")
        cache = REPO_ROOT / ".pytest_cache"
        if cache.exists():
            pytest.skip("Clean-worktree cache proof is performed by isolated post-push verification")
        before = _tree_fingerprint(REPO_ROOT)
        first = _run_gate()
        middle = _tree_fingerprint(REPO_ROOT)
        second = _run_gate()
        after = _tree_fingerprint(REPO_ROOT)
        assert first.returncode == EXIT_HUMAN_REVIEW_REQUIRED, first.stderr[-1000:]
        assert second.returncode == EXIT_HUMAN_REVIEW_REQUIRED, second.stderr[-1000:]
        assert before == middle == after
        assert not cache.exists(), "Gate 11 must disable pytest cache provider for its subprocess"
        assert not list((REPO_ROOT / "docs").rglob(".pytest_cache"))
        assert not list((REPO_ROOT / "benchmarks").rglob(".pytest_cache"))
        assert not list((REPO_ROOT / "training" / "jupiter_seed_4b" / "data").rglob(".pytest_cache"))


class TestWindowsAclDenyCleanup:
    def test_deny_ace_is_effective_removed_and_temp_tree_is_deletable(self) -> None:
        _require_windows_icacls()
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-acl-"))
        target = root / "required.jsonl"
        target.write_text('{"safe": true}\n', encoding="utf-8")
        try:
            with acl._windows_acl_deny(target, "R") as lease:
                assert acl._has_explicit_deny(lease.acl_during, lease.identity)
                assert not acl._can_read(target)
            assert lease.cleanup_returncode == 0
            assert lease.acl_after == lease.acl_before
            assert acl._can_read(target)
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()

    def test_deny_ace_cleanup_runs_after_assertion_failure(self) -> None:
        _require_windows_icacls()
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-acl-"))
        target = root / "required.jsonl"
        target.write_text('{"safe": true}\n', encoding="utf-8")
        try:
            with pytest.raises(RuntimeError):
                with acl._windows_acl_deny(target, "R") as lease:
                    assert not acl._can_read(target)
                    raise RuntimeError("intentional production assertion failure")
            assert lease.cleanup_returncode == 0
            assert lease.acl_after == lease.acl_before
            assert acl._can_read(target)
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()

    def test_deny_ace_cleanup_runs_after_subprocess_exception(self) -> None:
        _require_windows_icacls()
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-acl-"))
        parent = root / "artifact-parent"
        parent.mkdir()
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                with acl._windows_acl_deny(parent, "W") as lease:
                    assert not acl._can_create_child(parent)
                    raise subprocess.TimeoutExpired(["readiness_gate.py"], 1)
            assert lease.cleanup_returncode == 0
            assert lease.acl_after == lease.acl_before
            assert acl._can_create_child(parent)
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()

    def test_repeated_windows_deny_runs_leave_no_orphaned_directories(self) -> None:
        _require_windows_icacls()
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-acl-repeat-"))
        try:
            for index in range(3):
                target = root / f"run-{index}"
                target.mkdir()
                with acl._windows_acl_deny(target, "W") as lease:
                    assert not acl._can_create_child(target)
                assert lease.cleanup_returncode == 0
                assert acl._can_create_child(target)
                shutil.rmtree(target, ignore_errors=False)
            assert not list(root.iterdir())
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()

    def test_windows_output_write_failure_is_safe_and_removes_deny_ace(self) -> None:
        _require_windows_icacls()
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-output-"))
        parent = root / "unwritable-output"
        artifact_dir = root / "runtime-artifacts"
        parent.mkdir()
        artifact_dir.mkdir()
        try:
            with acl.verified_unwritable_artifact_parent(parent):
                output = parent / "readiness.md"
                result = _run_gate(["--output", str(output), "--artifact-dir", str(artifact_dir)])
                assert result.returncode == EXIT_EXECUTION_ERROR
                assert "safe fallback" in result.stderr.lower()
                assert "traceback" not in result.stderr.lower()
                assert not output.exists(), "failed report write must not leave a partial output"
            assert acl._can_create_child(parent)
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()


class TestV244IsolationPreservation:
    def test_posix_permission_cleanup_remains_correct(self) -> None:
        if os.name == "nt":
            pytest.skip("POSIX permission restoration test")
        root = Path(tempfile.mkdtemp(prefix="jupiter-v244-posix-"))
        target = root / "data.jsonl"
        target.write_text('{"safe": true}\n', encoding="utf-8")
        original = target.stat().st_mode
        try:
            with acl.verified_unreadable_file(target):
                assert not acl._can_read(target)
            assert target.stat().st_mode == original
            assert acl._can_read(target)
        finally:
            shutil.rmtree(root, ignore_errors=False)
        assert not root.exists()

    def test_production_execution_error_keeps_public_artifact_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing-data"
            artifact_dir = Path(raw) / "runtime"
            result = _run_gate(["--data-dir", str(missing), "--artifact-dir", str(artifact_dir)])
            assert result.returncode == EXIT_EXECUTION_ERROR
            artifact = artifact_dir / "EXECUTION_ERROR_ARTIFACT.json"
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            assert payload["training_authorized"] is False
            assert "traceback" not in artifact.read_text(encoding="utf-8").lower()
            assert not (artifact_dir / "EXECUTION_ERROR_INTERNAL.json").exists()
