#!/usr/bin/env python3
"""
Jupiter Shot — Authorized Test Manifest Generator
==================================================
Run 15 Corrected Package

Generates the authoritative node-ID manifest for the Run 15 targeted test suite.
Must be re-run whenever a test file is added, removed, or parametrized variants change.

Uses the pytest Python API (pytest.main + plugin hook) to collect node IDs directly,
avoiding the need to parse subprocess text output. This is reliable across platforms.

Usage:
    python3 scripts/generate_test_manifest.py
    # Windows:
    .venv\\Scripts\\python.exe scripts\\generate_test_manifest.py

Outputs:
    tests/run15_authorized_nodes.txt      — sorted node IDs, one per line
    tests/run15_authorized_manifest.json  — machine-readable manifest with SHA-256

Exit codes:
    0 — manifest generated successfully
    1 — collection error (no nodes collected or pytest error)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Authorized test files (order is canonical) ───────────────────────────────
AUTHORIZED_TEST_FILES: list[str] = [
    "tests/test_run12_gate10b_real_object.py",
    "tests/test_run12_operator_package.py",
    "tests/test_run13_config_resolver.py",
    "tests/test_run13_subprocess_smoke.py",
    "tests/test_run14_integration.py",
    "tests/test_run14_same_pass_provenance.py",
    "tests/test_run14_integration_contracts.py",
    "tests/test_run15_regression.py",
    "tests/test_run15_manifest_verifier.py",
    "tests/test_run15_utf8_reproducibility.py",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_FILE = REPO_ROOT / "tests" / "run15_authorized_nodes.txt"
MANIFEST_FILE = REPO_ROOT / "tests" / "run15_authorized_manifest.json"
SCHEMA_VERSION = "1.0"


class _NodeCollectorPlugin:
    """Minimal pytest plugin that captures item.nodeid for every collected item."""
    def __init__(self):
        self.nodes: list[str] = []
        self.collection_error: bool = False

    def pytest_collection_finish(self, session):
        for item in session.items:
            # Normalize to forward slashes (Windows compatibility)
            self.nodes.append(item.nodeid.replace("\\", "/"))

    def pytest_exception_interact(self, node, call, report):
        if report.when == "collect":
            self.collection_error = True


def collect_nodes() -> list[str]:
    """
    Collect test node IDs using the pytest Python API.
    Returns a sorted list of normalized node IDs.
    Raises RuntimeError on collection failure or zero nodes.
    """
    import pytest as _pytest

    # Change to repo root so relative paths resolve correctly
    orig_dir = os.getcwd()
    os.chdir(str(REPO_ROOT))
    try:
        plugin = _NodeCollectorPlugin()
        ret = _pytest.main(
            ["--collect-only", "-q", "--no-header", "--tb=no"] + AUTHORIZED_TEST_FILES,
            plugins=[plugin],
        )
        # ExitCode.OK=0, ExitCode.NO_TESTS_COLLECTED=5
        if ret not in (0, 5) or plugin.collection_error:
            raise RuntimeError(
                f"pytest collection returned exit code {int(ret)}. "
                "Check that all authorized test files exist and can be imported."
            )
        if not plugin.nodes:
            raise RuntimeError(
                "No test nodes collected. Verify that authorized test files exist "
                "and contain test functions."
            )
        return sorted(plugin.nodes)
    finally:
        os.chdir(orig_dir)


def node_list_sha256(nodes: list[str]) -> str:
    """
    Compute SHA-256 of the sorted node-ID list.
    Each node ID is separated by a newline; the final newline is included.
    Paths are normalized to forward slashes before hashing.
    """
    content = "\n".join(nodes) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def count_functions(nodes: list[str]) -> int:
    """
    Count unique test function base names (ignoring parametrized variants).
    E.g. test_foo[a] and test_foo[b] count as one function.
    """
    seen: set[str] = set()
    for node in nodes:
        base = node.split("[")[0]
        seen.add(base)
    return len(seen)


def get_git_info() -> tuple[str, str]:
    """Return (branch, short_commit) from git, or ('unknown', 'unknown') on failure."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def generate() -> dict:
    """
    Collect nodes, write manifest files, and return the manifest dict.
    """
    print(f"[MANIFEST] Collecting nodes from {len(AUTHORIZED_TEST_FILES)} authorized files...")
    nodes = collect_nodes()
    sha = node_list_sha256(nodes)
    func_count = count_functions(nodes)
    branch, commit = get_git_info()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "test_files": AUTHORIZED_TEST_FILES,
        "function_count": func_count,
        "collected_node_count": len(nodes),
        "node_list_sha256": sha,
        "generated_with": "scripts/generate_test_manifest.py",
        "generated_at": now,
        "branch": branch,
        "commit": commit,
    }

    # Write sorted node list
    NODES_FILE.write_text("\n".join(nodes) + "\n", encoding="utf-8")
    print(f"[MANIFEST] Node list written: {NODES_FILE} ({len(nodes)} nodes)")

    # Write JSON manifest
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[MANIFEST] JSON manifest written: {MANIFEST_FILE}")
    print(f"[MANIFEST] collected_node_count: {len(nodes)}")
    print(f"[MANIFEST] function_count:        {func_count}")
    print(f"[MANIFEST] node_list_sha256:      {sha}")
    print(f"[MANIFEST] branch:                {branch}")
    print(f"[MANIFEST] commit:                {commit}")

    return manifest


if __name__ == "__main__":
    try:
        generate()
        print("[MANIFEST] Generation complete. Exit 0.")
        sys.exit(0)
    except Exception as e:
        print(f"[MANIFEST ERROR] {e}", file=sys.stderr)
        sys.exit(1)
