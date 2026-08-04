"""
tests/test_run13_config_resolver.py
====================================
Contract regression tests for training/config_path.py (shared config resolver).

Covers:
  - All five resolution rules (A, B, C, D, fallback)
  - Double-suffix prevention (the Run 12 root cause)
  - Missing-file detection and error message content
  - Runner interface: each runner's --config argument accepts all four input forms
  - Pipeline: full-path inputs (the form the pipeline uses) are resolved correctly
  - format_missing_error diagnostic block content

Run with:
    .venv\\Scripts\\python.exe -m pytest tests\\test_run13_config_resolver.py -v
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from training.config_path import (
    ConfigResolution,
    format_missing_error,
    resolve_config_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_repo(tmp_path: Path, yaml_names: list[str]) -> Path:
    """Create a minimal fake repo structure with YAML configs."""
    configs_dir = tmp_path / "training" / "configs"
    configs_dir.mkdir(parents=True)
    for name in yaml_names:
        (configs_dir / name).write_text("model:\n  hidden_size: 64\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Rule A — absolute path with .yaml suffix
# ---------------------------------------------------------------------------

class TestRuleA:
    """Absolute path with recognised suffix → use as-is, never append."""

    def test_absolute_yaml_suffix_rule_a(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        p = str(repo / "training" / "configs" / "laptop_dense_run7.yaml")
        res = resolve_config_path(p, repo_root=repo)
        assert res.resolution_rule == "A"
        assert res.suffix_added is False
        assert res.exists is True
        assert not str(res.resolved_path).endswith(".yaml.yaml")

    def test_absolute_yml_suffix_rule_a(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yml"])
        p = str(repo / "training" / "configs" / "laptop_dense_run7.yml")
        res = resolve_config_path(p, repo_root=repo)
        assert res.resolution_rule == "A"
        assert res.suffix_added is False
        assert not str(res.resolved_path).endswith(".yml.yaml")

    def test_absolute_path_no_double_suffix(self, tmp_path):
        """Core Run 12 regression: pipeline passes full path → must not double-suffix."""
        repo = _make_fake_repo(tmp_path, ["laptop_moe_run7.yaml"])
        full_path = str(repo / "training" / "configs" / "laptop_moe_run7.yaml")
        res = resolve_config_path(full_path, repo_root=repo)
        assert not str(res.resolved_path).endswith(".yaml.yaml"), (
            f"DOUBLE SUFFIX detected: {res.resolved_path}"
        )
        assert res.exists is True


# ---------------------------------------------------------------------------
# Rule B — relative path with .yaml suffix
# ---------------------------------------------------------------------------

class TestRuleB:
    """Relative path with recognised suffix → resolve relative to repo root."""

    def test_relative_yaml_suffix_rule_b(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("training/configs/laptop_dense_run7.yaml", repo_root=repo)
        assert res.resolution_rule == "B"
        assert res.suffix_added is False
        assert res.exists is True
        assert not str(res.resolved_path).endswith(".yaml.yaml")

    def test_relative_yml_suffix_rule_b(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yml"])
        res = resolve_config_path("training/configs/laptop_dense_run7.yml", repo_root=repo)
        assert res.resolution_rule == "B"
        assert res.suffix_added is False
        assert not str(res.resolved_path).endswith(".yml.yaml")

    def test_relative_path_no_double_suffix(self, tmp_path):
        """Relative path with .yaml suffix must never produce .yaml.yaml."""
        repo = _make_fake_repo(tmp_path, ["laptop_moe_run7.yaml"])
        res = resolve_config_path("training/configs/laptop_moe_run7.yaml", repo_root=repo)
        assert not str(res.resolved_path).endswith(".yaml.yaml")


# ---------------------------------------------------------------------------
# Rule C — bare name (no suffix, no path separators)
# ---------------------------------------------------------------------------

class TestRuleC:
    """Bare name → resolve under training/configs/ and append .yaml once."""

    def test_bare_name_rule_c(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("laptop_dense_run7", repo_root=repo)
        assert res.resolution_rule == "C"
        assert res.suffix_added is True
        assert res.exists is True
        assert res.resolved_path.name == "laptop_dense_run7.yaml"
        assert not str(res.resolved_path).endswith(".yaml.yaml")

    def test_bare_name_moe_rule_c(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_moe_run7.yaml"])
        res = resolve_config_path("laptop_moe_run7", repo_root=repo)
        assert res.resolution_rule == "C"
        assert res.suffix_added is True
        assert res.exists is True


# ---------------------------------------------------------------------------
# Rule D — relative path without suffix (has directory components)
# ---------------------------------------------------------------------------

class TestRuleD:
    """Relative path without suffix → resolve then append .yaml once."""

    def test_relative_no_suffix_rule_d(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("training/configs/laptop_dense_run7", repo_root=repo)
        assert res.resolution_rule == "D"
        assert res.suffix_added is True
        assert res.exists is True
        assert not str(res.resolved_path).endswith(".yaml.yaml")


# ---------------------------------------------------------------------------
# Double-suffix prevention (the Run 12 root cause)
# ---------------------------------------------------------------------------

class TestDoubleSuffixPrevention:
    """Exhaustive double-suffix regression tests."""

    @pytest.mark.parametrize("inp", [
        "training/configs/laptop_dense_run7.yaml",
        "training/configs/laptop_moe_run7.yaml",
        "training/configs/laptop_dense_run7.yml",
    ])
    def test_no_double_yaml_suffix(self, tmp_path, inp):
        repo = _make_fake_repo(tmp_path, [
            "laptop_dense_run7.yaml",
            "laptop_moe_run7.yaml",
            "laptop_dense_run7.yml",
        ])
        res = resolve_config_path(inp, repo_root=repo)
        assert not str(res.resolved_path).endswith(".yaml.yaml"), (
            f"Double suffix in: {res.resolved_path}"
        )
        assert not str(res.resolved_path).endswith(".yml.yaml"), (
            f"Double suffix in: {res.resolved_path}"
        )

    def test_absolute_path_no_double_suffix_parametric(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        full = str(repo / "training" / "configs" / "laptop_dense_run7.yaml")
        res = resolve_config_path(full, repo_root=repo)
        assert not str(res.resolved_path).endswith(".yaml.yaml")


# ---------------------------------------------------------------------------
# Missing file detection
# ---------------------------------------------------------------------------

class TestMissingFile:
    """Missing config → exists=False, format_missing_error includes diagnostics."""

    def test_missing_bare_name(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("nonexistent_config", repo_root=repo)
        assert res.exists is False

    def test_missing_relative_path(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("training/configs/nonexistent.yaml", repo_root=repo)
        assert res.exists is False

    def test_format_missing_error_contains_original_input(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("bad_config_name", repo_root=repo)
        msg = format_missing_error(res)
        assert "bad_config_name" in msg

    def test_format_missing_error_contains_resolved_path(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("bad_config_name", repo_root=repo)
        msg = format_missing_error(res)
        assert "resolved_path" in msg

    def test_format_missing_error_contains_alternatives(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml", "laptop_moe_run7.yaml"])
        res = resolve_config_path("bad_config_name", repo_root=repo)
        msg = format_missing_error(res)
        assert "laptop_dense_run7.yaml" in msg or "valid alternatives" in msg

    def test_format_missing_error_contains_execution_error(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("bad_config_name", repo_root=repo)
        msg = format_missing_error(res)
        assert "EXECUTION_ERROR" in msg


# ---------------------------------------------------------------------------
# Pipeline input form (full path as passed by run_laptop_validation_pipeline.py)
# ---------------------------------------------------------------------------

class TestPipelineInputForm:
    """The pipeline passes str(args.dense_config) which is a full path with .yaml suffix."""

    def test_pipeline_dense_config_path(self, tmp_path):
        """Simulate: pipeline passes training/configs/laptop_dense_run7.yaml as full path."""
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        # Pipeline default: str(REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml")
        full_path = str(repo / "training" / "configs" / "laptop_dense_run7.yaml")
        res = resolve_config_path(full_path, repo_root=repo)
        assert res.exists is True
        assert res.suffix_added is False
        assert not str(res.resolved_path).endswith(".yaml.yaml")

    def test_pipeline_moe_config_path(self, tmp_path):
        """Simulate: pipeline passes training/configs/laptop_moe_run7.yaml as full path."""
        repo = _make_fake_repo(tmp_path, ["laptop_moe_run7.yaml"])
        full_path = str(repo / "training" / "configs" / "laptop_moe_run7.yaml")
        res = resolve_config_path(full_path, repo_root=repo)
        assert res.exists is True
        assert res.suffix_added is False
        assert not str(res.resolved_path).endswith(".yaml.yaml")

    def test_pipeline_resume_uses_dense_config_path(self, tmp_path):
        """Simulate: pipeline passes dense config path to resume runner."""
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        full_path = str(repo / "training" / "configs" / "laptop_dense_run7.yaml")
        res = resolve_config_path(full_path, repo_root=repo)
        assert res.exists is True
        assert not str(res.resolved_path).endswith(".yaml.yaml")


# ---------------------------------------------------------------------------
# Runner interface: --config accepts all four forms
# ---------------------------------------------------------------------------

class TestRunnerInterface:
    """Verify that all four input forms produce a valid resolved path."""

    @pytest.mark.parametrize("config_input,yaml_file", [
        # Form 1: absolute path with .yaml
        (None, "laptop_dense_run7.yaml"),   # built dynamically in test
        # Form 2: relative path with .yaml
        ("training/configs/laptop_dense_run7.yaml", "laptop_dense_run7.yaml"),
        # Form 3: bare name
        ("laptop_dense_run7", "laptop_dense_run7.yaml"),
        # Form 4: relative path without suffix
        ("training/configs/laptop_dense_run7", "laptop_dense_run7.yaml"),
    ])
    def test_all_four_forms_resolve_to_same_file(self, tmp_path, config_input, yaml_file):
        repo = _make_fake_repo(tmp_path, [yaml_file])
        if config_input is None:
            # Form 1: absolute path
            config_input = str(repo / "training" / "configs" / yaml_file)
        res = resolve_config_path(config_input, repo_root=repo)
        assert res.exists is True, (
            f"Form {config_input!r} did not resolve to an existing file: {res.resolved_path}"
        )
        assert res.resolved_path.name == yaml_file
        assert not str(res.resolved_path).endswith(".yaml.yaml")


# ---------------------------------------------------------------------------
# ConfigResolution dataclass fields
# ---------------------------------------------------------------------------

class TestConfigResolutionFields:
    """ConfigResolution must expose all required fields."""

    def test_all_required_fields_present(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("laptop_dense_run7", repo_root=repo)
        assert hasattr(res, "original_input")
        assert hasattr(res, "resolved_path")
        assert hasattr(res, "suffix_added")
        assert hasattr(res, "exists")
        assert hasattr(res, "repository_root")
        assert hasattr(res, "resolution_rule")

    def test_original_input_preserved(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        inp = "laptop_dense_run7"
        res = resolve_config_path(inp, repo_root=repo)
        assert res.original_input == inp

    def test_resolution_rule_is_string(self, tmp_path):
        repo = _make_fake_repo(tmp_path, ["laptop_dense_run7.yaml"])
        res = resolve_config_path("laptop_dense_run7", repo_root=repo)
        assert isinstance(res.resolution_rule, str)
        assert res.resolution_rule in ("A", "B", "C", "D")
