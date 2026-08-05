"""
Jupiter Shot — Run 15 Corrected Package
Manifest Verifier Regression Tests

10 tests proving that scripts/verify_test_manifest.py:
  V01 — Accepts the exact 174-node collection
  V02 — Detects a missing node
  V03 — Detects an unexpected node
  V04 — Detects a changed parametrized variant
  V05 — Detects a duplicate node
  V06 — Detects collection failure
  V07 — Detects hash mismatch
  V08 — Produces deterministic output (same SHA-256 on repeated calls)
  V09 — Handles Windows paths (backslash normalization)
  V10 — Returns a nonzero exit code on every mismatch
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Import the verifier module without executing __main__ ─────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIER_PATH = REPO_ROOT / "scripts" / "verify_test_manifest.py"
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_test_manifest.py"
NODES_FILE = REPO_ROOT / "tests" / "run15_authorized_nodes.txt"
MANIFEST_FILE = REPO_ROOT / "tests" / "run15_authorized_manifest.json"


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verifier = _load_module(VERIFIER_PATH, "verify_test_manifest")
generator = _load_module(GENERATOR_PATH, "generate_test_manifest")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_temp_manifest(
    nodes: list[str],
    test_files: list[str] | None = None,
    sha_override: str | None = None,
    count_override: int | None = None,
) -> tuple[Path, Path]:
    """Write a temporary nodes.txt and manifest.json; return (nodes_path, manifest_path)."""
    tmp = Path(tempfile.mkdtemp())
    nodes_path = tmp / "nodes.txt"
    manifest_path = tmp / "manifest.json"

    sorted_nodes = sorted(nodes)
    nodes_path.write_text("\n".join(sorted_nodes) + "\n", encoding="utf-8")

    sha = sha_override or generator.node_list_sha256(sorted_nodes)
    manifest = {
        "schema_version": "1.0",
        "test_files": test_files or generator.AUTHORIZED_TEST_FILES,
        "function_count": generator.count_functions(sorted_nodes),
        "collected_node_count": count_override if count_override is not None else len(sorted_nodes),
        "node_list_sha256": sha,
        "generated_with": "scripts/generate_test_manifest.py",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "branch": "fix/rtx50-blackwell-validation",
        "commit": "dc79637",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return nodes_path, manifest_path


def _run_verify_with_manifest(
    authorized_nodes: list[str],
    current_nodes: list[str],
    test_files: list[str] | None = None,
    sha_override: str | None = None,
    count_override: int | None = None,
) -> bool:
    """
    Patch NODES_FILE and MANIFEST_FILE in the verifier module, then call verify().
    Returns True if verification passed, False otherwise.
    """
    nodes_path, manifest_path = _make_temp_manifest(
        authorized_nodes,
        test_files=test_files,
        sha_override=sha_override,
        count_override=count_override,
    )

    def fake_collect(test_files):
        return sorted(n.replace("\\", "/") for n in current_nodes)

    with (
        patch.object(verifier, "NODES_FILE", nodes_path),
        patch.object(verifier, "MANIFEST_FILE", manifest_path),
        patch.object(verifier, "collect_current_nodes", side_effect=fake_collect),
    ):
        return verifier.verify()


# ── Load the real authorized nodes once ───────────────────────────────────────

@pytest.fixture(scope="module")
def authorized_nodes() -> list[str]:
    assert NODES_FILE.exists(), (
        f"Authorized node list not found: {NODES_FILE}. "
        "Run scripts/generate_test_manifest.py first."
    )
    nodes = [
        line.strip()
        for line in NODES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(nodes) == 174, f"Expected 174 authorized nodes, got {len(nodes)}"
    return nodes


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestManifestVerifier:

    def test_v01_accepts_exact_174_node_collection(self, authorized_nodes):
        """V01: Verifier accepts the exact 174-node collection."""
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes,
        )
        assert result is True, "Verifier should accept the exact authorized collection"

    def test_v02_detects_missing_node(self, authorized_nodes):
        """V02: Verifier detects a missing node."""
        # Remove the last node from the current collection
        missing_one = authorized_nodes[:-1]
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=missing_one,
        )
        assert result is False, "Verifier should fail when a node is missing"

    def test_v03_detects_unexpected_node(self, authorized_nodes):
        """V03: Verifier detects an unexpected (extra) node."""
        extra_node = "tests/test_run15_regression.py::TestMetricFilterAllowlist::test_unexpected_intruder"
        current_with_extra = authorized_nodes + [extra_node]
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=current_with_extra,
        )
        assert result is False, "Verifier should fail when an unexpected node is present"

    def test_v04_detects_changed_parametrized_variant(self, authorized_nodes):
        """V04: Verifier detects a changed parametrized variant name."""
        # Find a node from test_run13_config_resolver.py (which has parametrized tests)
        param_nodes = [n for n in authorized_nodes if "test_run13_config_resolver" in n]
        assert param_nodes, "Expected parametrized nodes in test_run13_config_resolver.py"

        # Replace the first parametrized node with a renamed variant
        original = param_nodes[0]
        # Modify the parametrized suffix or name
        if "[" in original:
            modified = original.rsplit("[", 1)[0] + "[changed_variant_name]"
        else:
            modified = original + "[new_param]"

        current_modified = [modified if n == original else n for n in authorized_nodes]
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=current_modified,
        )
        assert result is False, "Verifier should fail when a parametrized variant name changes"

    def test_v05_detects_duplicate_node(self, authorized_nodes):
        """V05: Verifier detects a duplicate node in the current collection."""
        # Duplicate the first node
        current_with_dupe = authorized_nodes + [authorized_nodes[0]]
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=current_with_dupe,
        )
        assert result is False, "Verifier should fail when a duplicate node is present"

    def test_v06_detects_collection_failure(self, authorized_nodes):
        """V06: Verifier detects collection failure (RuntimeError from collect_current_nodes)."""
        nodes_path, manifest_path = _make_temp_manifest(authorized_nodes)

        def failing_collect(test_files):
            raise RuntimeError("pytest --collect-only failed with exit code 2")

        with (
            patch.object(verifier, "NODES_FILE", nodes_path),
            patch.object(verifier, "MANIFEST_FILE", manifest_path),
            patch.object(verifier, "collect_current_nodes", side_effect=failing_collect),
        ):
            result = verifier.verify()

        assert result is False, "Verifier should fail on collection failure"

    def test_v07_detects_hash_mismatch(self, authorized_nodes):
        """V07: Verifier detects a SHA-256 hash mismatch."""
        # Provide the correct nodes but a tampered SHA in the manifest
        tampered_sha = "a" * 64  # wrong SHA
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes,
            sha_override=tampered_sha,
        )
        assert result is False, "Verifier should fail when the manifest SHA-256 is wrong"

    def test_v08_produces_deterministic_output(self, authorized_nodes):
        """V08: Verifier produces deterministic SHA-256 on repeated calls with same nodes."""
        sha1 = generator.node_list_sha256(authorized_nodes)
        sha2 = generator.node_list_sha256(authorized_nodes)
        sha3 = generator.node_list_sha256(sorted(authorized_nodes))
        assert sha1 == sha2 == sha3, (
            "node_list_sha256 must be deterministic: same nodes must always produce the same hash"
        )
        # Also verify the hash is non-trivial
        assert len(sha1) == 64, "SHA-256 hex digest must be 64 characters"
        assert sha1 != "0" * 64, "SHA-256 must not be all zeros"

    def test_v09_handles_windows_paths(self, authorized_nodes):
        """V09: Verifier normalizes Windows backslash paths to forward slashes."""
        # Convert all authorized nodes to Windows-style backslash paths
        windows_nodes = [n.replace("/", "\\") for n in authorized_nodes]

        # The verifier's _normalize function should handle this
        normalized = [verifier._normalize(n) for n in windows_nodes]
        assert normalized == authorized_nodes, (
            "Verifier must normalize Windows backslash paths to forward slashes"
        )

        # Also verify that the verifier passes when current collection uses backslashes
        result = _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=windows_nodes,  # backslash paths
        )
        assert result is True, (
            "Verifier should accept Windows-style backslash paths after normalization"
        )

    def test_v10_returns_nonzero_exit_code_on_every_mismatch(self, authorized_nodes):
        """V10: verify() returns False (nonzero exit) on every type of mismatch."""
        # Missing node → False
        assert _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes[:-1],
        ) is False, "Missing node must return False"

        # Extra node → False
        assert _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes + ["tests/extra.py::test_extra"],
        ) is False, "Extra node must return False"

        # Hash mismatch → False
        assert _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes,
            sha_override="b" * 64,
        ) is False, "Hash mismatch must return False"

        # Count mismatch (manifest says wrong count) → False
        assert _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes,
            count_override=999,
        ) is False, "Count mismatch must return False"

        # Exact match → True (sanity check)
        assert _run_verify_with_manifest(
            authorized_nodes=authorized_nodes,
            current_nodes=authorized_nodes,
        ) is True, "Exact match must return True"
