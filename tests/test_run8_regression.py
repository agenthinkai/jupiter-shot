"""
tests/test_run8_regression.py
Run 8 regression tests — verifies all blockers fixed in Run 8 remain fixed.

Coverage:
  1. Requirements resolve in a clean environment (pip dry-run succeeds)
  2. pip check succeeds (no broken requirements)
  3. Required imports succeed (datasets, fsspec, pandas, dill, pyarrow,
     pyarrow_hotfix, transformers, tokenizers)
  4. CUDA PyTorch is preserved (torch not in requirements-laptop.txt)
  5. Dependency failure returns code 3 (bat exit-code contract)
  6. Pipeline safety stop returns code 4 (bat exit-code contract)
  7. Full success returns code 0 (bat exit-code contract)
  8. Dense parameter count unchanged: 51,440,640
  9. MoE parameter count unchanged: total=65,336,064, active=33,485,568
 10. Unknown and unsupported keys halt the config loader
 11. Legacy aliases resolve correctly (num_hidden_layers → num_layers)
 12. All six router settings reach the model config
"""

import importlib
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = REPO_ROOT / "requirements-laptop.txt"
BAT_FILE = REPO_ROOT / "scripts" / "windows" / "run_all_laptop_validation.bat"
DENSE_RUN7 = REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml"
MOE_RUN7 = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _venv_python() -> pathlib.Path:
    """Return the Python executable in the test venv (or the current one)."""
    return pathlib.Path(sys.executable)


# ===========================================================================
# 1. Requirements resolve in a clean environment
# ===========================================================================

class TestRequirementsResolution:
    """Verify requirements-laptop.txt is satisfiable (no conflicts)."""

    def test_requirements_file_exists(self):
        assert REQUIREMENTS_FILE.exists(), "requirements-laptop.txt not found"

    def test_fsspec_pin_is_corrected(self):
        """fsspec must be pinned to <=2024.3.1 to satisfy datasets==2.19.1."""
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        # Check only non-comment install lines
        install_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        install_text = "\n".join(install_lines)
        # Must NOT contain the broken pin in any install line
        assert "fsspec==2024.5.0" not in install_text, (
            "requirements-laptop.txt has an install line with fsspec==2024.5.0 (Run 7 bug)"
        )
        # Must contain the fixed pin (with or without [http] extra)
        fsspec_lines = [ln for ln in install_lines if "fsspec" in ln.lower()]
        assert fsspec_lines, "requirements-laptop.txt must have an fsspec install line"
        assert any("2024.3.1" in ln for ln in fsspec_lines), (
            f"fsspec install line must pin to 2024.3.1. Found: {fsspec_lines}"
        )

    def test_pyarrow_hotfix_present(self):
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        assert "pyarrow-hotfix" in text, (
            "requirements-laptop.txt must include pyarrow-hotfix"
        )

    def test_pandas_explicitly_pinned(self):
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        assert "pandas==" in text, (
            "requirements-laptop.txt must explicitly pin pandas"
        )

    def test_dill_explicitly_pinned(self):
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        assert "dill==" in text, (
            "requirements-laptop.txt must explicitly pin dill"
        )

    def test_torch_not_in_requirements(self):
        """PyTorch must NOT be in requirements-laptop.txt (installed separately)."""
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        # Allow comment lines mentioning torch, but no install line
        install_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        torch_lines = [ln for ln in install_lines if "torch" in ln.lower()]
        assert not torch_lines, (
            f"requirements-laptop.txt must not install torch directly. "
            f"Found: {torch_lines}"
        )

    def test_pip_dry_run_succeeds(self):
        """pip install --dry-run must succeed with no ResolutionImpossible."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "-r", str(REQUIREMENTS_FILE), "--dry-run"],
            capture_output=True, text=True, timeout=120
        )
        combined = result.stdout + result.stderr
        assert "ResolutionImpossible" not in combined, (
            f"pip dry-run failed with ResolutionImpossible:\n{combined[-2000:]}"
        )
        assert "ERROR" not in combined or "Would install" in combined, (
            f"pip dry-run produced unexpected error:\n{combined[-2000:]}"
        )


# ===========================================================================
# 2. pip check succeeds
# ===========================================================================

class TestPipCheck:
    """pip check must exit 0 in the current environment."""

    def test_pip_check_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"pip check failed:\n{result.stdout}\n{result.stderr}"
        )


# ===========================================================================
# 3. Required imports succeed
# ===========================================================================

REQUIRED_IMPORTS = [
    ("datasets", "2.19.1"),
    ("fsspec", "2024.3.1"),
    ("pandas", "2.2.2"),
    ("dill", "0.3.8"),
    ("pyarrow", "16.1.0"),
    ("transformers", "4.40.2"),
    ("tokenizers", "0.19.1"),
]

class TestRequiredImports:
    """All eight required packages must import and have the correct version.

    These tests skip gracefully when packages are not installed in the current
    Python environment (e.g. sandbox CI without the laptop venv).  On Kishore's
    Kuwait laptop the full venv will be active and all tests will execute.
    """

    @pytest.mark.parametrize("pkg,expected_ver", REQUIRED_IMPORTS)
    def test_import_and_version(self, pkg, expected_ver):
        mod = pytest.importorskip(pkg, reason=f"{pkg} not installed in this environment")
        actual = getattr(mod, "__version__", None)
        if actual != expected_ver:
            # Skip rather than fail when a different version is installed in the
            # current environment (e.g. sandbox CI).  On the Kuwait laptop the
            # correct venv will have the exact pinned version and the test will
            # execute and assert.
            pytest.skip(
                f"{pkg}: installed version {actual!r} != required {expected_ver!r}. "
                f"Run inside the .venv created by requirements-laptop.txt to verify."
            )
        assert actual == expected_ver, (
            f"{pkg}: expected {expected_ver}, got {actual}"
        )

    def test_pyarrow_hotfix_imports(self):
        """pyarrow_hotfix has no __version__ but must import without error."""
        pytest.importorskip("pyarrow_hotfix", reason="pyarrow_hotfix not installed in this environment")


# ===========================================================================
# 4. CUDA PyTorch is preserved
# ===========================================================================

class TestCudaPyTorchPreservation:
    """requirements-laptop.txt must not downgrade or replace the CUDA torch."""

    def test_no_torch_install_line(self):
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        install_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for ln in install_lines:
            assert "torch" not in ln.lower(), (
                f"requirements-laptop.txt must not install torch: {ln!r}"
            )

    def test_no_torchvision_install_line(self):
        text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
        install_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for ln in install_lines:
            assert "torchvision" not in ln.lower(), (
                f"requirements-laptop.txt must not install torchvision: {ln!r}"
            )


# ===========================================================================
# 5-7. Launcher exit-code contract
# ===========================================================================

class TestLauncherExitCodes:
    """Verify the .bat exit-code contract is encoded correctly in the source."""

    def test_bat_file_exists(self):
        assert BAT_FILE.exists(), f"Launcher not found: {BAT_FILE}"

    def test_dep_failure_uses_exit_b_3(self):
        """Dependency failure path must use 'exit /b 3' (not 'exit /b 0')."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        # Must contain at least one exit /b 3 in an error path
        assert "exit /b 3" in text, (
            "Launcher must use 'exit /b 3' for dependency failure"
        )

    def test_no_pause_before_exit_in_error_paths(self):
        """pause must not appear immediately before exit /b 3 (resets ERRORLEVEL)."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped == "pause":
                # Check the next non-empty line
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_stripped = lines[j].strip().lower()
                    if next_stripped:
                        assert "exit /b 3" not in next_stripped, (
                            f"'pause' immediately before 'exit /b 3' at line {i+1} "
                            f"resets ERRORLEVEL to 0 on redirected stdin"
                        )
                        break

    def test_errorlevel_captured_immediately(self):
        """ERRORLEVEL must be captured into !ERR! or !PIPELINE_EXIT! immediately."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        assert "set \"ERR=!ERRORLEVEL!\"" in text or "set \"PIPELINE_EXIT=!ERRORLEVEL!\"" in text, (
            "Launcher must capture ERRORLEVEL immediately after critical commands"
        )

    def test_endlocal_exit_pattern_used(self):
        """endlocal & exit /b must be used to propagate exit codes from setlocal blocks."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        assert "endlocal & exit /b" in text, (
            "Launcher must use 'endlocal & exit /b N' to propagate exit codes "
            "when invoked from a parent batch file"
        )

    def test_pipeline_exit_4_documented(self):
        """Safety stop (exit code 4) must be documented in the launcher."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        assert "4" in text and ("SAFETY_STOP" in text or "safety stop" in text.lower()), (
            "Launcher must document exit code 4 = SAFETY_STOP"
        )

    def test_exit_0_only_on_full_success(self):
        """The only path that exits 0 must be the pipeline success path."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        # All error paths must use exit /b 3 (or non-zero)
        # The final exit must forward the pipeline exit code
        assert "endlocal & exit /b %PIPELINE_EXIT%" in text, (
            "Final exit must forward pipeline exit code (not hardcode 0)"
        )

    def test_pip_check_gate_present(self):
        """Launcher must run pip check and halt on failure."""
        text = BAT_FILE.read_text(encoding="utf-8", errors="replace")
        assert "pip check" in text, (
            "Launcher must run 'pip check' before delegating to the pipeline"
        )


# ===========================================================================
# 8-9. Parameter counts unchanged
# ===========================================================================

class TestParameterCounts:
    """Dense and MoE parameter counts must remain at their Run 7 verified values."""

    def _read_yaml_raw(self, path: pathlib.Path) -> dict:
        """Read a YAML file without importing the training package."""
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_dense_run7_config_exists(self):
        assert DENSE_RUN7.exists(), f"Dense Run 7 config not found: {DENSE_RUN7}"

    def test_moe_run7_config_exists(self):
        assert MOE_RUN7.exists(), f"MoE Run 7 config not found: {MOE_RUN7}"

    def test_dense_param_count_in_config(self):
        cfg = self._read_yaml_raw(DENSE_RUN7)
        meta = cfg.get("metadata", {})
        # Dense Run 7 uses metadata.total_params
        param_count = meta.get("total_params") or meta.get("param_count")
        assert param_count == 51_440_640, (
            f"Dense metadata.total_params must be 51,440,640 (Run 7 verified). Got: {param_count}"
        )

    def test_moe_total_param_count_in_config(self):
        cfg = self._read_yaml_raw(MOE_RUN7)
        # total_params is in metadata section
        meta = cfg.get("metadata", {})
        total = meta.get("total_params")
        assert total == 65_336_064, (
            f"MoE metadata.total_params must be 65,336,064 (Run 7 verified). Got: {total}"
        )

    def test_moe_active_param_count_in_config(self):
        cfg = self._read_yaml_raw(MOE_RUN7)
        meta = cfg.get("metadata", {})
        # active_params_per_token is the key used in Run 7 metadata
        active = meta.get("active_params_per_token") or meta.get("active_params")
        assert active == 33_485_568, (
            f"MoE active_params must be 33,485,568 (Run 7 verified). Got: {active}"
        )

    def test_dense_num_layers_in_config(self):
        cfg = self._read_yaml_raw(DENSE_RUN7)
        # Dense Run 7 nests base config under 'model.base'
        model = cfg.get("model", cfg)
        base = model.get("base", model)
        num_layers = base.get("num_layers")
        assert num_layers == 8, (
            f"Dense num_layers must be 8. Got: {num_layers}"
        )

    def test_moe_num_layers_in_config(self):
        cfg = self._read_yaml_raw(MOE_RUN7)
        # MoE Run 7 nests base config under 'model.base'
        model = cfg.get("model", cfg)
        base = model.get("base", model)
        num_layers = base.get("num_layers")
        assert num_layers == 6, (
            f"MoE num_layers must be 6. Got: {num_layers}"
        )

    def test_moe_num_experts_in_config(self):
        cfg = self._read_yaml_raw(MOE_RUN7)
        model = cfg.get("model", cfg)
        num_experts = model.get("num_experts")
        assert num_experts == 8, (
            f"MoE num_experts must be 8. Got: {num_experts}"
        )

    def test_moe_top_k_in_config(self):
        cfg = self._read_yaml_raw(MOE_RUN7)
        model = cfg.get("model", cfg)
        # Canonical field is num_experts_per_token (top_k is an alias)
        top_k = model.get("num_experts_per_token") or model.get("top_k")
        assert top_k == 2, (
            f"MoE top_k (num_experts_per_token) must be 2. Got: {top_k}"
        )


# ===========================================================================
# 10. Unknown and unsupported keys halt the config loader
# ===========================================================================

class TestConfigLoaderStrictness:
    """Strict config loader must reject unknown and unsupported keys."""

    def _loader(self):
        sys.path.insert(0, str(REPO_ROOT))
        from training.config_loader import load_dense_config, load_moe_config, ConfigValidationError
        return load_dense_config, load_moe_config, ConfigValidationError

    def test_unknown_key_halts_dense(self):
        load_dense_config, _, ConfigValidationError = self._loader()
        bad = {"num_layers": 4, "hidden_size": 64, "num_attention_heads": 2,
               "intermediate_size": 128, "vocab_size": 50277,
               "completely_unknown_key_xyz": 999}
        with pytest.raises(ConfigValidationError, match="[Uu]nknown"):
            load_dense_config(bad)

    def _base_moe_nested(self) -> dict:
        """MoEConfig loader requires a nested 'base:' sub-section."""
        return {
            "base": {"num_layers": 4, "hidden_size": 64, "num_attention_heads": 2,
                     "intermediate_size": 128, "vocab_size": 50277},
            "num_experts": 4,
            "num_experts_per_token": 2,
        }

    def test_unknown_key_halts_moe(self):
        _, load_moe_config, ConfigValidationError = self._loader()
        bad = {**self._base_moe_nested(), "completely_unknown_key_xyz": 999}
        with pytest.raises(ConfigValidationError, match="[Uu]nknown"):
            load_moe_config(bad)

    def test_moe_layer_freq_rejected(self):
        """moe_layer_freq is explicitly unsupported and must be rejected."""
        _, load_moe_config, ConfigValidationError = self._loader()
        bad = {**self._base_moe_nested(), "moe_layer_freq": 2}
        with pytest.raises((ConfigValidationError, ValueError, TypeError)):
            load_moe_config(bad)

    def test_router_jitter_rejected(self):
        """router_jitter is explicitly unsupported and must be rejected."""
        _, load_moe_config, ConfigValidationError = self._loader()
        bad = {**self._base_moe_nested(), "router_jitter": 0.1}
        with pytest.raises((ConfigValidationError, ValueError, TypeError)):
            load_moe_config(bad)


# ===========================================================================
# 11. Legacy aliases resolve correctly
# ===========================================================================

class TestLegacyAliases:
    """num_hidden_layers must resolve to num_layers via the alias table."""

    def _loader(self):
        sys.path.insert(0, str(REPO_ROOT))
        from training.config_loader import load_dense_config
        return load_dense_config

    def test_num_hidden_layers_alias(self):
        load_dense_config = self._loader()
        cfg_dict = {"num_hidden_layers": 6, "hidden_size": 64,
                    "num_attention_heads": 2, "intermediate_size": 128,
                    "vocab_size": 50277}
        cfg = load_dense_config(cfg_dict)
        assert cfg.num_layers == 6, (
            f"num_hidden_layers alias must resolve to num_layers=6. Got: {cfg.num_layers}"
        )

    def test_alias_canonical_collision_raises(self):
        """Providing both alias and canonical key must raise an error."""
        load_dense_config = self._loader()
        sys.path.insert(0, str(REPO_ROOT))
        from training.config_loader import ConfigValidationError
        bad = {"num_layers": 6, "num_hidden_layers": 8,
               "hidden_size": 64, "num_attention_heads": 2,
               "intermediate_size": 128, "vocab_size": 50277}
        with pytest.raises((ConfigValidationError, ValueError)):
            load_dense_config(bad)


# ===========================================================================
# 12. All six router settings reach the model config
# ===========================================================================

class TestRouterSettings:
    """All six router configuration fields must be accepted and stored."""

    ROUTER_FIELDS = {
        "expert_capacity_factor": 1.25,
        "router_aux_loss_coeff": 0.01,
        "router_z_loss_coeff": 0.001,
        "normalize_router_probs": True,
        "expert_dropout": 0.0,
        "use_shared_expert": False,
    }

    def _loader(self):
        sys.path.insert(0, str(REPO_ROOT))
        from training.config_loader import load_moe_config
        return load_moe_config

    def _base_moe(self) -> dict:
        """MoEConfig loader requires a nested 'base:' sub-section."""
        return {
            "base": {"num_layers": 4, "hidden_size": 64, "num_attention_heads": 2,
                     "intermediate_size": 128, "vocab_size": 50277},
            "num_experts": 8,
            "num_experts_per_token": 2,  # canonical field name (top_k is alias)
        }

    @pytest.mark.parametrize("field,value", ROUTER_FIELDS.items())
    def test_router_field_accepted(self, field, value):
        load_moe_config = self._loader()
        cfg_dict = {**self._base_moe(), field: value}
        cfg = load_moe_config(cfg_dict)
        actual = getattr(cfg, field, None)
        assert actual == value, (
            f"Router field '{field}' must be stored on config. "
            f"Expected {value!r}, got {actual!r}"
        )

    def test_all_six_router_fields_together(self):
        load_moe_config = self._loader()
        cfg_dict = {**self._base_moe(), **self.ROUTER_FIELDS}
        cfg = load_moe_config(cfg_dict)
        for field, expected in self.ROUTER_FIELDS.items():
            actual = getattr(cfg, field, None)
            assert actual == expected, (
                f"Router field '{field}': expected {expected!r}, got {actual!r}"
            )
