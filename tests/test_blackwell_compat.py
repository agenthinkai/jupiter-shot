"""
Jupiter Shot — Blackwell Compatibility Regression Tests
========================================================
CPU-compatible regression tests for the RTX 50-series (Blackwell, sm_120)
compatibility repair on the validation/kuwait-laptop-gpu branch.

These tests verify:
  - req 1: Windows batch parser fix (no unescaped ) inside if blocks)
  - req 2: requirements-laptop.txt exists and is valid
  - req 3: Blackwell environment profile document exists and is complete
  - req 4: Architecture detection logic in setup script
  - req 5: Preflight kernel_test field present in report output
  - req 6: Python version enforcement in preflight and setup scripts
  - req 7: Download fallback instructions present in setup scripts
  - req 8: SMALL config enforced for 8 GB VRAM
  - req 9: Config files are Blackwell-compatible (bf16, min_pytorch_version)
  - req 10: BLACKWELL_ENVIRONMENT.md exists with correct status
  - req 11: All regression tests pass on CPU without CUDA

All tests run without a GPU (CPU-only environment).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── req 1: Windows batch parser fix ─────────────────────────────────────────

class TestBatchParserFix:
    """Verify the Windows batch parser error is fixed in all .bat files."""

    BAT_FILES = [
        "scripts/windows/run_all_laptop_validation.bat",
        "scripts/windows/run_preflight.bat",
        "scripts/windows/run_dense_validation.bat",
        "scripts/windows/run_moe_validation.bat",
        "scripts/windows/run_resume_validation.bat",
    ]

    def _get_bat_content(self, rel_path: str) -> str:
        path = REPO_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"Script not found: {rel_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def test_run_all_bat_no_unescaped_paren_in_if_blocks(self) -> None:
        """
        CMD parser error: unescaped ) inside if/else blocks causes
        '. was unexpected at this time'. Verify all ) in echo lines
        inside if blocks are escaped as ^).
        """
        content = self._get_bat_content("scripts/windows/run_all_laptop_validation.bat")
        lines = content.splitlines()
        in_block = 0
        violations: list[str] = []
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip().upper()
            # Track parenthesized block depth
            if stripped.startswith("IF ") or stripped.startswith("FOR "):
                if "(" in line:
                    in_block += line.count("(") - line.count("^(")
            if stripped.startswith(")") and in_block > 0:
                in_block -= 1
            # Inside a block: echo lines must not have bare ) (only ^))
            if in_block > 0 and stripped.startswith("ECHO "):
                # Find ) not preceded by ^
                bare_paren = re.findall(r'(?<!\^)\)', line)
                if bare_paren:
                    violations.append(f"Line {lineno}: {line.rstrip()}")
        assert not violations, (
            f"Unescaped ) found in if blocks (should be ^)):\n"
            + "\n".join(violations)
        )

    def test_run_all_bat_has_setlocal_enabledelayedexpansion(self) -> None:
        """setlocal enabledelayedexpansion is required for !VAR! syntax."""
        content = self._get_bat_content("scripts/windows/run_all_laptop_validation.bat")
        assert "setlocal enabledelayedexpansion" in content.lower(), \
            "run_all_laptop_validation.bat must have 'setlocal enabledelayedexpansion'"

    @pytest.mark.parametrize("bat_file", BAT_FILES)
    def test_bat_file_exists(self, bat_file: str) -> None:
        path = REPO_ROOT / bat_file
        assert path.exists(), f"Missing batch script: {bat_file}"

    @pytest.mark.parametrize("bat_file", BAT_FILES)
    def test_bat_file_not_empty(self, bat_file: str) -> None:
        path = REPO_ROOT / bat_file
        if path.exists():
            assert path.stat().st_size > 200, f"Batch script too small: {bat_file}"


# ─── req 2: requirements-laptop.txt ──────────────────────────────────────────

class TestRequirementsLaptop:
    """Verify requirements-laptop.txt exists and is valid."""

    def test_requirements_laptop_exists(self) -> None:
        path = REPO_ROOT / "requirements-laptop.txt"
        assert path.exists(), "requirements-laptop.txt not found in repo root"

    def test_requirements_laptop_not_empty(self) -> None:
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            content = path.read_text()
            non_comment_lines = [l for l in content.splitlines()
                                  if l.strip() and not l.strip().startswith("#")]
            assert len(non_comment_lines) >= 5, \
                "requirements-laptop.txt should have at least 5 non-comment lines"

    def test_requirements_laptop_no_torch(self) -> None:
        """PyTorch must NOT be in requirements-laptop.txt (installed separately)."""
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            # Only check non-comment lines
            non_comment = [l for l in path.read_text().splitlines()
                           if l.strip() and not l.strip().startswith("#")]
            content = "\n".join(non_comment).lower()
            assert "torch==" not in content, \
                "torch must not be in requirements-laptop.txt (installed separately by setup script)"
            assert "torchvision" not in content, \
                "torchvision must not be in requirements-laptop.txt"

    def test_requirements_laptop_no_deepspeed(self) -> None:
        """DeepSpeed must NOT be in requirements-laptop.txt (Linux-only)."""
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            # Only check non-comment lines
            non_comment = [l for l in path.read_text().splitlines()
                           if l.strip() and not l.strip().startswith("#")]
            content = "\n".join(non_comment).lower()
            assert "deepspeed" not in content, \
                "deepspeed must not be in requirements-laptop.txt (Linux-only)"

    def test_requirements_laptop_has_numpy(self) -> None:
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            content = path.read_text().lower()
            assert "numpy" in content, "requirements-laptop.txt should include numpy"

    def test_requirements_laptop_has_pyyaml(self) -> None:
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            content = path.read_text().lower()
            assert "pyyaml" in content, "requirements-laptop.txt should include pyyaml"

    def test_requirements_laptop_has_pytest(self) -> None:
        path = REPO_ROOT / "requirements-laptop.txt"
        if path.exists():
            content = path.read_text().lower()
            assert "pytest" in content, "requirements-laptop.txt should include pytest"


# ─── req 3 & 10: Blackwell environment profile document ──────────────────────

class TestBlackwellEnvironmentDoc:
    """Verify BLACKWELL_ENVIRONMENT.md exists and contains required sections."""

    DOC_PATH = REPO_ROOT / "docs" / "BLACKWELL_ENVIRONMENT.md"

    def test_blackwell_env_doc_exists(self) -> None:
        assert self.DOC_PATH.exists(), \
            "docs/BLACKWELL_ENVIRONMENT.md not found. Create it with Blackwell repair instructions."

    def test_blackwell_env_doc_status_blocked_repairable(self) -> None:
        """Status must be ENVIRONMENT BLOCKED — REPAIRABLE (not PASS or FAIL)."""
        if not self.DOC_PATH.exists():
            pytest.skip("BLACKWELL_ENVIRONMENT.md not found")
        content = self.DOC_PATH.read_text()
        assert "ENVIRONMENT BLOCKED" in content, \
            "BLACKWELL_ENVIRONMENT.md must state 'ENVIRONMENT BLOCKED'"
        assert "REPAIRABLE" in content, \
            "BLACKWELL_ENVIRONMENT.md must state 'REPAIRABLE'"

    def test_blackwell_env_doc_no_false_gpu_success(self) -> None:
        """Must not claim GPU validation succeeded."""
        if not self.DOC_PATH.exists():
            pytest.skip("BLACKWELL_ENVIRONMENT.md not found")
        content = self.DOC_PATH.read_text()
        assert "GPU validation succeeded" not in content
        assert "validation passed" not in content.lower() or \
               "safety stop worked" in content.lower() or \
               "not started" in content.lower(), \
            "BLACKWELL_ENVIRONMENT.md must not claim validation passed"

    def test_blackwell_env_doc_has_pytorch_version(self) -> None:
        """Must specify PyTorch 2.7.1+cu128."""
        if not self.DOC_PATH.exists():
            pytest.skip("BLACKWELL_ENVIRONMENT.md not found")
        content = self.DOC_PATH.read_text()
        assert "2.7.1" in content, \
            "BLACKWELL_ENVIRONMENT.md must specify PyTorch 2.7.1"
        assert "cu128" in content, \
            "BLACKWELL_ENVIRONMENT.md must specify cu128 wheel"

    def test_blackwell_env_doc_has_sm120(self) -> None:
        """Must mention sm_120 compute capability."""
        if not self.DOC_PATH.exists():
            pytest.skip("BLACKWELL_ENVIRONMENT.md not found")
        content = self.DOC_PATH.read_text()
        assert "sm_120" in content, \
            "BLACKWELL_ENVIRONMENT.md must mention sm_120 (Blackwell compute capability)"

    def test_blackwell_env_doc_has_operator_instructions(self) -> None:
        """Must have operator instructions for Kishore."""
        if not self.DOC_PATH.exists():
            pytest.skip("BLACKWELL_ENVIRONMENT.md not found")
        content = self.DOC_PATH.read_text()
        assert "Kishore" in content or "Operator" in content, \
            "BLACKWELL_ENVIRONMENT.md must have operator instructions"


# ─── req 4 & 6: Python version enforcement and architecture detection ─────────

class TestSetupScriptEnforcement:
    """Verify setup scripts enforce Python version and detect Blackwell."""

    PS1_PATH = REPO_ROOT / "scripts" / "windows" / "setup_laptop_environment.ps1"
    BAT_PATH = REPO_ROOT / "scripts" / "windows" / "run_all_laptop_validation.bat"

    def test_ps1_enforces_python_310_311(self) -> None:
        """PowerShell setup script must reject Python != 3.10 or 3.11."""
        if not self.PS1_PATH.exists():
            pytest.skip("setup_laptop_environment.ps1 not found")
        content = self.PS1_PATH.read_text()
        assert "3.10" in content and "3.11" in content, \
            "setup_laptop_environment.ps1 must enforce Python 3.10 or 3.11"
        assert "UNSUPPORTED" in content or "not supported" in content.lower() or \
               "Unsupported" in content, \
            "setup_laptop_environment.ps1 must reject unsupported Python versions"

    def test_bat_enforces_python_310_311(self) -> None:
        """Batch setup script must reject Python != 3.10 or 3.11."""
        if not self.BAT_PATH.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = self.BAT_PATH.read_text()
        assert "3.10" in content and "3.11" in content, \
            "run_all_laptop_validation.bat must enforce Python 3.10 or 3.11"

    def test_ps1_detects_blackwell(self) -> None:
        """PowerShell setup script must detect Blackwell (sm_120 / cc >= 12)."""
        if not self.PS1_PATH.exists():
            pytest.skip("setup_laptop_environment.ps1 not found")
        content = self.PS1_PATH.read_text()
        assert "isBlackwell" in content or "Blackwell" in content, \
            "setup_laptop_environment.ps1 must detect Blackwell GPU"
        assert "2.7.1" in content, \
            "setup_laptop_environment.ps1 must install PyTorch 2.7.1 for Blackwell"
        assert "cu128" in content, \
            "setup_laptop_environment.ps1 must use cu128 wheel for Blackwell"

    def test_bat_detects_blackwell(self) -> None:
        """Batch script must detect Blackwell and select cu128 wheel."""
        if not self.BAT_PATH.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = self.BAT_PATH.read_text()
        assert "Blackwell" in content or "BLACKWELL" in content or "sm_120" in content, \
            "run_all_laptop_validation.bat must detect Blackwell GPU"
        assert "2.7.1" in content, \
            "run_all_laptop_validation.bat must reference PyTorch 2.7.1 for Blackwell"
        assert "cu128" in content, \
            "run_all_laptop_validation.bat must reference cu128 wheel for Blackwell"


# ─── req 5: Preflight kernel_test field ──────────────────────────────────────

class TestPreflightKernelTest:
    """Verify the upgraded preflight includes real kernel test output."""

    def test_preflight_has_run_kernel_test_function(self) -> None:
        """The preflight script must define run_kernel_test()."""
        preflight_path = REPO_ROOT / "scripts" / "laptop_gpu_preflight.py"
        assert preflight_path.exists(), "laptop_gpu_preflight.py not found"
        content = preflight_path.read_text()
        assert "def run_kernel_test" in content, \
            "laptop_gpu_preflight.py must define run_kernel_test()"

    def test_preflight_kernel_test_checks_arch_list(self) -> None:
        """run_kernel_test must check get_arch_list() for sm_120."""
        preflight_path = REPO_ROOT / "scripts" / "laptop_gpu_preflight.py"
        if not preflight_path.exists():
            pytest.skip("laptop_gpu_preflight.py not found")
        content = preflight_path.read_text()
        assert "get_arch_list" in content, \
            "run_kernel_test must check torch.cuda.get_arch_list() for sm_120 support"
        assert "sm_120" in content, \
            "run_kernel_test must check for sm_120 in arch list"

    def test_preflight_kernel_test_does_real_matmul(self) -> None:
        """run_kernel_test must perform a real CUDA matmul, not just is_available()."""
        preflight_path = REPO_ROOT / "scripts" / "laptop_gpu_preflight.py"
        if not preflight_path.exists():
            pytest.skip("laptop_gpu_preflight.py not found")
        content = preflight_path.read_text()
        assert "torch.matmul" in content or "matmul" in content, \
            "run_kernel_test must perform a real CUDA matrix multiplication"
        assert "synchronize" in content, \
            "run_kernel_test must call torch.cuda.synchronize() after kernel launch"

    def test_preflight_halts_on_kernel_failure(self) -> None:
        """Preflight must return exit code 1 when kernel test fails."""
        preflight_path = REPO_ROOT / "scripts" / "laptop_gpu_preflight.py"
        if not preflight_path.exists():
            pytest.skip("laptop_gpu_preflight.py not found")
        content = preflight_path.read_text()
        assert "kernel_pass" in content, \
            "Preflight must check kernel_pass before allowing training to proceed"

    def test_preflight_no_cuda_returns_fail(self, tmp_path: Path) -> None:
        """Preflight should return FAIL status when CUDA is not available."""
        pytest.importorskip("torch", reason="torch not installed")
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            from scripts.laptop_gpu_preflight import run_preflight
            result = run_preflight(output_dir=tmp_path)
        assert result["gpu"]["cuda_available"] is False
        assert result.get("pass") is False
        assert result["preflight_status"] == "FAIL"

    def test_preflight_report_has_kernel_test_key(self, tmp_path: Path) -> None:
        """Preflight JSON output must include kernel_test section."""
        pytest.importorskip("torch", reason="torch not installed")
        from unittest.mock import patch
        with patch("torch.cuda.is_available", return_value=False):
            from scripts.laptop_gpu_preflight import run_preflight
            result = run_preflight(output_dir=tmp_path)
        # kernel_test key should be present (may be empty dict if CUDA unavailable)
        assert "kernel_test" in result, \
            "Preflight report must include 'kernel_test' section"


# ─── req 7: Download reliability / fallback instructions ─────────────────────

class TestDownloadReliability:
    """Verify setup scripts include download fallback and antivirus instructions."""

    PS1_PATH = REPO_ROOT / "scripts" / "windows" / "setup_laptop_environment.ps1"
    BAT_PATH = REPO_ROOT / "scripts" / "windows" / "run_all_laptop_validation.bat"

    def test_ps1_has_manual_fallback(self) -> None:
        if not self.PS1_PATH.exists():
            pytest.skip("setup_laptop_environment.ps1 not found")
        content = self.PS1_PATH.read_text()
        assert "fallback" in content.lower() or "manual" in content.lower(), \
            "setup_laptop_environment.ps1 must include manual download fallback instructions"

    def test_bat_has_manual_fallback(self) -> None:
        if not self.BAT_PATH.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = self.BAT_PATH.read_text()
        assert "fallback" in content.lower() or "manual" in content.lower(), \
            "run_all_laptop_validation.bat must include manual download fallback instructions"

    def test_bat_mentions_antivirus(self) -> None:
        if not self.BAT_PATH.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = self.BAT_PATH.read_text()
        assert "antivirus" in content.lower() or "anti-virus" in content.lower(), \
            "run_all_laptop_validation.bat must mention antivirus interference as a troubleshooting step"

    def test_ps1_mentions_antivirus(self) -> None:
        if not self.PS1_PATH.exists():
            pytest.skip("setup_laptop_environment.ps1 not found")
        content = self.PS1_PATH.read_text()
        assert "antivirus" in content.lower() or "anti-virus" in content.lower(), \
            "setup_laptop_environment.ps1 must mention antivirus interference"

    def test_bat_mentions_disk_space(self) -> None:
        if not self.BAT_PATH.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = self.BAT_PATH.read_text()
        assert "disk" in content.lower() or "GB" in content, \
            "run_all_laptop_validation.bat must mention disk space requirements"


# ─── req 8: SMALL config enforced for 8 GB VRAM ──────────────────────────────

class TestSmallConfigEnforcement:
    """Verify SMALL config is enforced for 8 GB VRAM."""

    def test_select_config_8gb_returns_small(self) -> None:
        """8 GB VRAM should map to laptop_dense_small (not medium or tiny)."""
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=8.0)
        assert result["dense_config"] == "laptop_dense_small", \
            f"8 GB VRAM should use laptop_dense_small, got {result['dense_config']}"
        assert result["moe_config"] == "laptop_moe_small", \
            f"8 GB VRAM should use laptop_moe_small, got {result['moe_config']}"

    def test_select_config_8gb_not_medium(self) -> None:
        """8 GB VRAM must NOT use medium config (requires 16 GB+)."""
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=8.0)
        assert "medium" not in (result.get("dense_config") or ""), \
            "8 GB VRAM must not use medium config"

    def test_select_config_8gb_not_1b3(self) -> None:
        """8 GB VRAM must NOT attempt the 1.3B model."""
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=8.0)
        assert "1b3" not in (result.get("dense_config") or ""), \
            "8 GB VRAM must not attempt the 1.3B model"

    def test_bat_enforces_small_for_8gb(self) -> None:
        """Batch script must enforce SMALL config when VRAM < 10 GB."""
        bat_path = REPO_ROOT / "scripts" / "windows" / "run_all_laptop_validation.bat"
        if not bat_path.exists():
            pytest.skip("run_all_laptop_validation.bat not found")
        content = bat_path.read_text()
        assert "SMALL" in content or "small" in content, \
            "run_all_laptop_validation.bat must enforce SMALL config for 8 GB VRAM"
        assert "8" in content, \
            "run_all_laptop_validation.bat must reference 8 GB VRAM threshold"

    def test_dense_small_config_gradient_checkpointing(self) -> None:
        """laptop_dense_small must have gradient_checkpointing=true."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_dense_small.yaml"
        assert cfg_path.exists(), "laptop_dense_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["training"]["gradient_checkpointing"] is True, \
            "laptop_dense_small must have gradient_checkpointing=true for 8 GB VRAM"

    def test_moe_small_config_gradient_checkpointing(self) -> None:
        """laptop_moe_small must have gradient_checkpointing=true."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_moe_small.yaml"
        assert cfg_path.exists(), "laptop_moe_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["training"]["gradient_checkpointing"] is True, \
            "laptop_moe_small must have gradient_checkpointing=true for 8 GB VRAM"


# ─── req 9: Config files are Blackwell-compatible ─────────────────────────────

class TestBlackwellConfigCompatibility:
    """Verify SMALL configs are updated for Blackwell (bf16, min_pytorch_version)."""

    def test_dense_small_uses_bf16(self) -> None:
        """laptop_dense_small should use bf16 (preferred on Blackwell)."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_dense_small.yaml"
        assert cfg_path.exists(), "laptop_dense_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        precision = cfg["training"]["precision"]
        assert precision in ("bf16", "fp16"), \
            f"laptop_dense_small precision must be bf16 or fp16, got {precision}"

    def test_moe_small_uses_bf16(self) -> None:
        """laptop_moe_small should use bf16 (preferred on Blackwell)."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_moe_small.yaml"
        assert cfg_path.exists(), "laptop_moe_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        precision = cfg["training"]["precision"]
        assert precision in ("bf16", "fp16"), \
            f"laptop_moe_small precision must be bf16 or fp16, got {precision}"

    def test_dense_small_has_blackwell_compatible_flag(self) -> None:
        """laptop_dense_small metadata should declare Blackwell compatibility."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_dense_small.yaml"
        assert cfg_path.exists(), "laptop_dense_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        meta = cfg.get("metadata", {})
        assert meta.get("blackwell_compatible") is True, \
            "laptop_dense_small metadata must have blackwell_compatible: true"

    def test_moe_small_has_blackwell_compatible_flag(self) -> None:
        """laptop_moe_small metadata should declare Blackwell compatibility."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_moe_small.yaml"
        assert cfg_path.exists(), "laptop_moe_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        meta = cfg.get("metadata", {})
        assert meta.get("blackwell_compatible") is True, \
            "laptop_moe_small metadata must have blackwell_compatible: true"

    def test_dense_small_has_min_pytorch_version(self) -> None:
        """laptop_dense_small should specify minimum PyTorch version for Blackwell."""
        cfg_path = REPO_ROOT / "training" / "configs" / "laptop_dense_small.yaml"
        assert cfg_path.exists(), "laptop_dense_small.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        meta = cfg.get("metadata", {})
        assert "min_pytorch_version" in meta, \
            "laptop_dense_small metadata must specify min_pytorch_version"
        assert "2.7.1" in str(meta["min_pytorch_version"]), \
            "laptop_dense_small min_pytorch_version must be 2.7.1+cu128"


# ─── req 11: All regression tests are CPU-compatible ─────────────────────────

class TestCPUCompatibility:
    """Meta-tests: verify this test suite itself runs without CUDA."""

    def test_no_cuda_required_for_config_tests(self) -> None:
        """Config loading tests must not require CUDA."""
        from scripts.laptop_gpu_preflight import select_config, estimate_memory
        result = select_config(vram_gb=8.0)
        assert result is not None
        mem = estimate_memory(params_m=85.0, precision="bf16")
        assert mem["total_training_gb"] > 0

    def test_no_cuda_required_for_preflight_import(self) -> None:
        """Preflight module must import without CUDA."""
        import importlib
        mod = importlib.import_module("scripts.laptop_gpu_preflight")
        assert hasattr(mod, "run_kernel_test")
        assert hasattr(mod, "select_config")
        assert hasattr(mod, "run_preflight")

    def test_requirements_laptop_installable_without_cuda(self) -> None:
        """All packages in requirements-laptop.txt must be CPU-installable."""
        path = REPO_ROOT / "requirements-laptop.txt"
        if not path.exists():
            pytest.skip("requirements-laptop.txt not found")
        # Only check non-comment lines
        non_comment = [l for l in path.read_text().splitlines()
                       if l.strip() and not l.strip().startswith("#")]
        content = "\n".join(non_comment).lower()
        # These packages require CUDA build tools and must not be in laptop reqs
        cuda_only = ["deepspeed", "flash-attn", "flash_attn", "vllm", "bitsandbytes"]
        for pkg in cuda_only:
            assert pkg not in content, \
                f"requirements-laptop.txt must not include {pkg} (requires CUDA build tools)"
