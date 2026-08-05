#!/usr/bin/env python3
"""
Jupiter Shot — Authorized Test Manifest Verifier
=================================================
Run 15 Corrected Package

Verifies that the current pytest collection exactly matches the authorized manifest.
Fails on any discrepancy: missing node, unexpected node, changed parametrized variant,
duplicate node, collection failure, or hash mismatch.

Uses the pytest Python API for reliable cross-platform node collection.

Usage:
    python3 scripts/verify_test_manifest.py
    # Windows:
    .venv\\Scripts\\python.exe scripts\\verify_test_manifest.py

Exit codes:
    0 — exact match: node set, count, and SHA-256 all match
    1 — mismatch or collection failure (details printed to stdout)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_FILE = REPO_ROOT / "tests" / "run15_authorized_nodes.txt"
MANIFEST_FILE = REPO_ROOT / "tests" / "run15_authorized_manifest.json"


def _normalize(node_id: str) -> str:
    """Normalize a node ID to forward slashes and strip whitespace."""
    return node_id.replace("\\", "/").strip()


def load_authorized_manifest() -> tuple[list[str], dict]:
    """
    Load the authorized node list and manifest JSON.
    Returns (sorted_nodes, manifest_dict).
    Raises FileNotFoundError if either file is missing.
    """
    if not NODES_FILE.exists():
        raise FileNotFoundError(
            f"Authorized node list not found: {NODES_FILE}\n"
            "Run scripts/generate_test_manifest.py first."
        )
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Authorized manifest not found: {MANIFEST_FILE}\n"
            "Run scripts/generate_test_manifest.py first."
        )
    authorized_nodes = [
        _normalize(line)
        for line in NODES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return authorized_nodes, manifest


class _NodeCollectorPlugin:
    """Minimal pytest plugin that captures item.nodeid for every collected item."""
    def __init__(self):
        self.nodes: list[str] = []
        self.collection_error: bool = False

    def pytest_collection_finish(self, session):
        for item in session.items:
            self.nodes.append(_normalize(item.nodeid))

    def pytest_exception_interact(self, node, call, report):
        if report.when == "collect":
            self.collection_error = True


def collect_current_nodes(test_files: list[str]) -> list[str]:
    """
    Collect current test node IDs using the pytest Python API.
    Returns sorted normalized node IDs.
    Raises RuntimeError on collection failure.
    """
    import pytest as _pytest

    orig_dir = os.getcwd()
    os.chdir(str(REPO_ROOT))
    try:
        plugin = _NodeCollectorPlugin()
        ret = _pytest.main(
            ["--collect-only", "-q", "--no-header", "--tb=no"] + test_files,
            plugins=[plugin],
        )
        if ret not in (0,) or plugin.collection_error:
            raise RuntimeError(
                f"pytest --collect-only failed with exit code {int(ret)}. "
                "This is a collection failure — verify that all authorized test files "
                "exist and can be imported."
            )
        if not plugin.nodes:
            raise RuntimeError(
                "No test nodes collected. Check that authorized test files exist."
            )
        return sorted(plugin.nodes)
    finally:
        os.chdir(orig_dir)


def node_list_sha256(nodes: list[str]) -> str:
    """Compute SHA-256 of the sorted node-ID list (same algorithm as generator)."""
    content = "\n".join(nodes) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify() -> bool:
    """
    Verify the current collection against the authorized manifest.
    Returns True on exact match, False on any discrepancy.
    Prints a detailed report to stdout.
    """
    print("[VERIFY] Loading authorized manifest...")
    authorized_nodes, manifest = load_authorized_manifest()
    authorized_set = set(authorized_nodes)
    authorized_count = manifest.get("collected_node_count", len(authorized_nodes))
    authorized_sha = manifest.get("node_list_sha256", "")

    print(f"[VERIFY] Authorized: {authorized_count} nodes, SHA-256: {authorized_sha[:16]}...")
    print(f"[VERIFY] Collecting current nodes from {len(manifest.get('test_files', []))} files...")

    try:
        current_nodes = collect_current_nodes(
            test_files=manifest.get("test_files", []),
        )
    except RuntimeError as e:
        print(f"[VERIFY FAIL] Collection failure:\n{e}")
        return False

    current_set = set(current_nodes)
    current_sha = node_list_sha256(current_nodes)

    failures: list[str] = []

    # Check for duplicate nodes in current collection
    if len(current_nodes) != len(current_set):
        from collections import Counter
        counts = Counter(current_nodes)
        dupes = [n for n, c in counts.items() if c > 1]
        failures.append(f"DUPLICATE NODES detected ({len(dupes)}): {dupes[:5]}")

    # Check for missing nodes
    missing = sorted(authorized_set - current_set)
    if missing:
        failures.append(
            f"MISSING NODES ({len(missing)}):\n"
            + "\n".join(f"  - {n}" for n in missing[:20])
            + ("\n  ... (truncated)" if len(missing) > 20 else "")
        )

    # Check for unexpected nodes
    unexpected = sorted(current_set - authorized_set)
    if unexpected:
        failures.append(
            f"UNEXPECTED NODES ({len(unexpected)}):\n"
            + "\n".join(f"  + {n}" for n in unexpected[:20])
            + ("\n  ... (truncated)" if len(unexpected) > 20 else "")
        )

    # Check count
    if len(current_nodes) != authorized_count:
        failures.append(
            f"COUNT MISMATCH: authorized={authorized_count}, "
            f"current={len(current_nodes)}"
        )

    # Check SHA-256
    if current_sha != authorized_sha:
        failures.append(
            f"SHA-256 MISMATCH:\n"
            f"  authorized: {authorized_sha}\n"
            f"  current:    {current_sha}"
        )

    if failures:
        print("[VERIFY FAIL] Manifest verification FAILED:")
        for f in failures:
            print(f"  {f}")
        return False

    print(f"[VERIFY OK] Exact match: {len(current_nodes)} nodes, SHA-256: {current_sha[:16]}...")
    print("[VERIFY OK] All checks passed.")
    return True


if __name__ == "__main__":
    try:
        ok = verify()
        sys.exit(0 if ok else 1)
    except FileNotFoundError as e:
        print(f"[VERIFY ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[VERIFY ERROR] Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)
