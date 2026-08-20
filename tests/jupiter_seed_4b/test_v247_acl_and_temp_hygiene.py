"""V2.4.7 — production-test hygiene regressions.

Windows ACL cases are collected and skipped outside genuine Windows environments.
All temporary-directory checks exercise the actual V2.4.2 ``run_cli`` helper.
"""
from __future__ import annotations

import ast
import inspect
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import test_v244_windows_acl_and_cache as v244

from test_v242_execution_isolation import (
    EXIT_EXECUTION_ERROR,
    _can_read,
    _has_explicit_deny,
    _icacls_snapshot,
    _windows_acl_deny,
    assert_execution_error,
    run_cli,
)


V242_PATH = Path(inspect.getsourcefile(run_cli)).resolve()
V244_PATH = V242_PATH.with_name("test_v244_windows_acl_and_cache.py")


def _children(root: Path) -> set[str]:
    return {child.name for child in root.iterdir()} if root.exists() else set()


def _run_cli_calls() -> list[ast.Call]:
    tree = ast.parse(V242_PATH.read_text(encoding="utf-8"))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_cli"
    ]


class TestRunCliOwnedTemporaryDirectory:
    def test_all_thirteen_call_sites_use_run_cli(self) -> None:
        # This is an AST count of real production-test paths, not a documentation count.
        assert len(_run_cli_calls()) == 13

    def test_successful_nonzero_exit_captures_artifact_then_removes_owned_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "owned"
        before = _children(root)
        run = run_cli(["--data-dir", str(tmp_path / "missing-data")], temp_root=root)
        artifact = assert_execution_error(run)
        assert artifact["error_category"] == "missing_required_input"
        assert run.owned_directory_removed
        assert _children(root) == before

    def test_mechanical_nonzero_exit_removes_owned_directory(self, tmp_path: Path) -> None:
        # A missing data directory returns execution error 3; this verifies nonzero exits do not leak.
        root = tmp_path / "owned"
        run = run_cli(["--data-dir", str(tmp_path / "missing-data")], temp_root=root)
        assert run.result.returncode == EXIT_EXECUTION_ERROR
        assert run.owned_directory_removed
        assert _children(root) == set()

    def test_subprocess_exception_removes_owned_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "owned"

        def raise_exception(*args, **kwargs):
            raise RuntimeError("injected subprocess exception")

        monkeypatch.setattr(subprocess, "run", raise_exception)
        with pytest.raises(RuntimeError, match="injected subprocess exception"):
            run_cli([], temp_root=root)
        assert _children(root) == set()

    def test_timeout_removes_owned_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "owned"

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="readiness_gate.py", timeout=1)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        with pytest.raises(subprocess.TimeoutExpired):
            run_cli([], timeout=1, temp_root=root)
        assert _children(root) == set()

    def test_skip_after_run_does_not_leak_owned_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "owned"

        def scenario() -> None:
            run = run_cli(["--data-dir", str(tmp_path / "missing-data")], temp_root=root)
            assert run.owned_directory_removed
            pytest.skip("injected post-run skip")

        with pytest.raises(pytest.skip.Exception):
            scenario()
        assert _children(root) == set()

    def test_unrelated_temporary_directory_is_untouched(self, tmp_path: Path) -> None:
        root = tmp_path / "owned"
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        marker = unrelated / "keep.txt"
        marker.write_text("do-not-delete", encoding="utf-8")
        run_cli(["--data-dir", str(tmp_path / "missing-data")], temp_root=root)
        assert marker.read_text(encoding="utf-8") == "do-not-delete"
        assert unrelated.exists()
        assert _children(root) == set()

    def test_five_repeated_cycles_have_zero_net_owned_directory_growth(self, tmp_path: Path) -> None:
        root = tmp_path / "owned"
        before = _children(root)
        for _ in range(5):
            run = run_cli(["--data-dir", str(tmp_path / "missing-data")], temp_root=root)
            assert run.owned_directory_removed
            assert_execution_error(run)
        assert _children(root) == before

    def test_helper_has_no_unmanaged_mkdtemp_lifecycle(self) -> None:
        source = inspect.getsource(run_cli)
        assert "TemporaryDirectory" in source
        assert "with tempfile.TemporaryDirectory" in source
        assert "mkdtemp" not in source


class TestLegacyV244RuntimeIsolation:
    def test_legacy_gate_helper_owns_artifacts_and_preserves_repository_tree(self, tmp_path: Path) -> None:
        if os.environ.get("JUPITER_GATE11_SUBPROCESS"):
            pytest.skip("Avoid recursive production Gate 11 invocation inside Gate 11 subprocess")
        cache = v244.REPO_ROOT / ".pytest_cache"
        if cache.exists():
            pytest.skip("Clean-worktree runtime isolation is verified in an isolated worktree")
        before = v244._tree_fingerprint(v244.REPO_ROOT)
        runtime_root = tmp_path / "legacy-gate-runtime"
        first = v244._run_gate(runtime_root=runtime_root)
        middle = v244._tree_fingerprint(v244.REPO_ROOT)
        second = v244._run_gate(runtime_root=runtime_root)
        after = v244._tree_fingerprint(v244.REPO_ROOT)
        assert first.returncode == v244.EXIT_HUMAN_REVIEW_REQUIRED, first.stderr[-1000:]
        assert second.returncode == v244.EXIT_HUMAN_REVIEW_REQUIRED, second.stderr[-1000:]
        assert before == middle == after
        assert runtime_root.exists() and not list(runtime_root.iterdir())
        source = inspect.getsource(v244._run_gate)
        assert "TemporaryDirectory" in source
        assert '"--artifact-dir"' in source
        assert "runtime_root" in source


class TestLegacyV244AclMigration:
    def test_legacy_file_scoped_acl_calls_use_rd_without_obsolete_cleanup_state(self) -> None:
        tree = ast.parse(V244_PATH.read_text(encoding="utf-8"))
        deny_permissions = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "_windows_acl_deny" or len(node.args) < 2:
                continue
            permission = node.args[1]
            if isinstance(permission, ast.Constant) and isinstance(permission.value, str):
                deny_permissions.append(permission.value)
        assert deny_permissions.count("RD") == 2
        assert "R" not in deny_permissions
        source = V244_PATH.read_text(encoding="utf-8")
        assert "cleanup_returncode" not in source
        assert "lease.acl_after == lease.acl_before" in source
        assert "acl._can_read(target)" in source
        assert "shutil.rmtree" in source


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL functional verification requires native Windows")
class TestWindowsFileScopedDenyLifecycle:
    def test_rd_deny_preserves_acl_inspection_and_blocks_content_read(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.txt"
        target.write_text("private", encoding="utf-8")
        with _windows_acl_deny(target, "RD") as lease:
            assert _has_explicit_deny(lease.acl_during, lease.identity)
            # Reading the ACL must still work after content denial; RD is narrower than generic R.
            assert _icacls_snapshot(shutil.which("icacls") or "icacls", target) == lease.acl_during
            assert not _can_read(target)
        assert _can_read(target)
        assert target.exists()

    def test_cleanup_runs_after_assertion_skip_and_exception_paths(self, tmp_path: Path) -> None:
        target = tmp_path / "payload.txt"
        target.write_text("private", encoding="utf-8")
        for mode in ("assertion", "skip", "exception"):
            if mode == "assertion":
                with pytest.raises(AssertionError):
                    with _windows_acl_deny(target, "RD"):
                        raise AssertionError("injected")
            elif mode == "skip":
                with pytest.raises(pytest.skip.Exception):
                    with _windows_acl_deny(target, "RD"):
                        pytest.skip("injected")
            else:
                with pytest.raises(RuntimeError):
                    with _windows_acl_deny(target, "RD"):
                        raise RuntimeError("injected")
            assert _can_read(target)
        assert target.exists()

    def test_cleanup_command_is_narrow_and_unmasked(self) -> None:
        source = inspect.getsource(_windows_acl_deny) + inspect.getsource(__import__("test_v242_execution_isolation")._remove_exact_test_deny)
        assert '"/remove:d", identity' in source
        assert '"/c"' not in source
        assert '"/save"' not in source
        assert '"/restore"' not in source
        assert "_icacls_failure_text" in source

    def test_five_repeated_file_cycles_leave_no_orphan_deny_or_directory(self, tmp_path: Path) -> None:
        for index in range(5):
            cycle = tmp_path / f"cycle-{index}"
            cycle.mkdir()
            target = cycle / "payload.txt"
            target.write_text("private", encoding="utf-8")
            with _windows_acl_deny(target, "RD") as lease:
                assert _has_explicit_deny(lease.acl_during, lease.identity)
                assert not _can_read(target)
            assert _can_read(target)
            shutil.rmtree(cycle)
            assert not cycle.exists()
