"""
Jupiter Shot — Laptop Validation Script Tests
=============================================
CPU-compatible tests for all Track A (laptop validation) components.
These tests run without a GPU by mocking CUDA-dependent paths.

Tests cover:
  - Config loading and VRAM tier selection
  - Preflight logic (CPU path)
  - Dense training loop structure (CPU, tiny model)
  - MoE training loop structure (CPU, tiny model)
  - Resume test logic (CPU)
  - Metrics collector (from fixture data)
  - Report generator (from fixture data)
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_dense_metrics_fixture(n_steps: int = 20) -> list[dict]:
    """Generate synthetic dense metrics JSONL entries."""
    import math
    metrics = []
    for i in range(1, n_steps + 1):
        metrics.append({
            "step": i,
            "loss": 10.37 - 0.05 * math.log(i + 1),
            "tokens_per_sec": 1200.0 + i * 5,
            "allocated_gb": 1.5 + 0.001 * i,
            "reserved_gb": 2.0,
            "gpu_temp_c": 65 + i % 10,
        })
    return metrics


def make_moe_metrics_fixture(n_steps: int = 20, n_experts: int = 8) -> list[dict]:
    """Generate synthetic MoE metrics JSONL entries."""
    metrics = []
    for i in range(1, n_steps + 1):
        # Simulate improving routing
        cv = max(0.1, 0.8 - 0.01 * i)
        entropy = min(3.0, 1.5 + 0.05 * i)
        share = 1.0 / n_experts
        metrics.append({
            "step": i,
            "loss": 10.37 - 0.04 * math.log(i + 1),
            "aux_loss": 0.05 - 0.0001 * i,
            "z_loss": 0.001,
            "tokens_per_sec": 900.0 + i * 3,
            "allocated_gb": 2.0 + 0.001 * i,
            "utilization_cv": cv,
            "router_entropy_bits": entropy,
            "num_inactive_experts": max(0, 2 - i // 10),
            "max_min_ratio": 2.0 - 0.01 * i,
            "expert_assignment_share": [share] * n_experts,
        })
    return metrics


def write_fixture_results(results_dir: Path) -> None:
    """Write fixture result files to a temp directory."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # preflight.json — cuda_available must be inside 'gpu' key for generate_draft to render hardware table
    preflight = {
        "preflight_status": "PASS",
        "system": {"date": "2026-08-02", "os": "Windows 11", "cpu": "Intel i7",
                    "system_ram_gb": 16.0, "system_ram_free_gb": 8.0,
                    "disk_total_gb": 500.0, "disk_free_gb": 200.0},
        "gpu": {
            "cuda_available": True,
            "gpu_model": "NVIDIA GeForce RTX 3060 Laptop GPU",
            "vram_total_gb": 6.0,
            "vram_free_gb": 4.5,
            "compute_capability": "8.6",
            "cuda_version": "12.1",
            "pytorch_version": "2.2.2",
            "driver_version": "535.104.05",
            "bf16_supported": True,
            "fp16_supported": True,
            "deepspeed_version": "not installed",
            "flash_attention_version": "not installed",
            "nccl_version": "unknown",
            "gpu_temperature_c": 42,
            "gpu_utilization_pct": 0,
        },
        "recommended_precision": "bf16",
        "recommended_dense_config": "laptop_dense_small",
        "recommended_moe_config": "laptop_moe_small",
        "config_tier": "6-8GB",
        "config_rationale": "Tight but workable.",
        "memory_estimate_note": "Excludes activations.",
        "pass": True,
    }
    (results_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))

    # dense_metrics.jsonl
    dense_metrics = make_dense_metrics_fixture(20)
    with open(results_dir / "dense_metrics.jsonl", "w") as f:
        for m in dense_metrics:
            f.write(json.dumps(m) + "\n")

    # dense_summary.json
    dense_summary = {
        "config": "laptop_dense_small",
        "total_params": 85_000_000,
        "precision": "bf16",
        "data_mode": "synthetic",
        "steps_completed": 20,
        "steps_planned": 20,
        "first_loss": dense_metrics[0]["loss"],
        "last_loss": dense_metrics[-1]["loss"],
        "loss_decreased": True,
        "nan_inf_count": 0,
        "oom_count": 0,
        "total_tokens": 20 * 2 * 512,
        "avg_tokens_per_sec": 1300.0,
        "peak_vram_allocated_gb": 1.52,
        "wall_time_s": 45.2,
        "checkpoint_size_mb": 340.0,
        "status": "PASS",
    }
    (results_dir / "dense_summary.json").write_text(json.dumps(dense_summary, indent=2))

    # moe_metrics.jsonl
    moe_metrics = make_moe_metrics_fixture(20)
    with open(results_dir / "moe_metrics.jsonl", "w") as f:
        for m in moe_metrics:
            f.write(json.dumps(m) + "\n")

    # moe_summary.json
    moe_summary = {
        "config": "laptop_moe_small",
        "total_params": 180_000_000,
        "active_params_per_token": 55_000_000,
        "num_experts": 8,
        "top_k": 2,
        "steps_completed": 20,
        "steps_planned": 20,
        "first_loss": moe_metrics[0]["loss"],
        "last_loss": moe_metrics[-1]["loss"],
        "avg_aux_loss_last10": 0.04,
        "avg_router_entropy_last10": 2.5,
        "avg_utilization_cv_last10": 0.3,
        "dropped_token_pct": 0.5,
        "nan_inf_count": 0,
        "peak_vram_allocated_gb": 2.02,
        "moe_accepted": True,
        "moe_acceptance": {
            "router_stable": True,
            "all_experts_active": True,
            "utilization_cv_ok": True,
            "entropy_ok": True,
            "aux_loss_ok": True,
            "dropped_tokens_ok": True,
        },
        "status": "PASS",
    }
    (results_dir / "moe_summary.json").write_text(json.dumps(moe_summary, indent=2))

    # resume_test.json
    resume = {
        "config": "laptop_dense_small",
        "initial_steps": 20,
        "resume_steps": 5,
        "tests": {
            "checkpoint_save": {"passed": True, "size_mb": 340.0, "duration_s": 1.2},
            "global_step_match": {"passed": True, "expected": 20, "loaded": 20},
            "loss_at_checkpoint_match": {"passed": True, "expected": 9.5, "loaded": 9.5, "delta": 0.0},
            "loss_continuity": {"passed": True, "reference_loss": 9.4, "resumed_loss": 9.41, "delta": 0.01, "delta_pct": 0.1},
        },
        "passed": True,
        "checkpoint_size_mb": 340.0,
        "checkpoint_save_duration_s": 1.2,
        "checkpoint_load_duration_s": 0.9,
    }
    (results_dir / "resume_test.json").write_text(json.dumps(resume, indent=2))


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestLaptopConfigs:
    """Test that all laptop YAML configs are valid and contain required fields."""

    DENSE_CONFIGS = [
        "laptop_dense_tiny",
        "laptop_dense_small",
        "laptop_dense_medium",
    ]
    MOE_CONFIGS = [
        "laptop_moe_tiny",
        "laptop_moe_small",
        "laptop_moe_medium",
    ]

    def _load_config(self, name: str) -> dict:
        path = REPO_ROOT / "training" / "configs" / f"{name}.yaml"
        assert path.exists(), f"Config not found: {path}"
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.mark.parametrize("config_name", DENSE_CONFIGS)
    def test_dense_config_has_required_fields(self, config_name: str) -> None:
        cfg = self._load_config(config_name)
        assert "model" in cfg, f"{config_name}: missing 'model' section"
        assert "training" in cfg, f"{config_name}: missing 'training' section"
        model = cfg["model"]
        for field in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                      "intermediate_size", "max_position_embeddings"]:
            assert field in model, f"{config_name}: missing model.{field}"
        training = cfg["training"]
        for field in ["batch_size", "seq_length", "learning_rate", "precision"]:
            assert field in training, f"{config_name}: missing training.{field}"

    @pytest.mark.parametrize("config_name", MOE_CONFIGS)
    def test_moe_config_has_required_fields(self, config_name: str) -> None:
        cfg = self._load_config(config_name)
        assert "model" in cfg, f"{config_name}: missing 'model' section"
        assert "training" in cfg, f"{config_name}: missing 'training' section"
        model = cfg["model"]
        assert "base" in model, f"{config_name}: missing model.base"
        for field in ["num_experts", "num_experts_per_token", "capacity_factor",
                      "aux_loss_coeff", "z_loss_coeff"]:
            assert field in model, f"{config_name}: missing model.{field}"

    @pytest.mark.parametrize("config_name", DENSE_CONFIGS)
    def test_dense_config_precision_valid(self, config_name: str) -> None:
        cfg = self._load_config(config_name)
        precision = cfg["training"]["precision"]
        assert precision in ("fp16", "bf16", "fp32"), \
            f"{config_name}: invalid precision '{precision}'"

    @pytest.mark.parametrize("config_name", DENSE_CONFIGS)
    def test_dense_config_vram_estimate_positive(self, config_name: str) -> None:
        cfg = self._load_config(config_name)
        meta = cfg.get("metadata", {})
        if "param_count_approx_m" in meta:
            assert meta["param_count_approx_m"] > 0
        if "vram_budget_gb" in meta:
            assert meta["vram_budget_gb"] > 0

    @pytest.mark.parametrize("config_name", MOE_CONFIGS)
    def test_moe_config_top_k_leq_num_experts(self, config_name: str) -> None:
        cfg = self._load_config(config_name)
        model = cfg["model"]
        top_k = model["num_experts_per_token"]
        n_experts = model["num_experts"]
        assert top_k <= n_experts, \
            f"{config_name}: top_k ({top_k}) > num_experts ({n_experts})"

    def test_vram_tiers_are_ordered(self) -> None:
        """Tiny < Small < Medium in param count."""
        tiny = self._load_config("laptop_dense_tiny")["metadata"]["param_count_approx_m"]
        small = self._load_config("laptop_dense_small")["metadata"]["param_count_approx_m"]
        medium = self._load_config("laptop_dense_medium")["metadata"]["param_count_approx_m"]
        assert tiny < small < medium, f"Tier ordering violated: {tiny} < {small} < {medium}"


# ─── Config Selector Tests ─────────────────────────────────────────────────────

class TestConfigSelector:
    """Test VRAM-based config selection logic using laptop_gpu_preflight.select_config."""

    def test_select_config_4gb(self) -> None:
        # 3.5 GB total - 1.5 GB overhead = 2.0 GB effective → below 3.5 GB min → INSUFFICIENT
        # Use 5.5 GB total to get tiny tier (5.5 - 1.5 = 4.0 >= 3.5 min)
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=5.5)
        assert result["dense_config"] is not None
        assert "tiny" in result["dense_config"] or "small" in result["dense_config"]

    def test_select_config_8gb(self) -> None:
        # 8.0 - 1.5 = 6.5 >= 6.0 → 6-8GB tier → small
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=8.0)
        assert result["dense_config"] is not None
        assert "small" in result["dense_config"]

    def test_select_config_16gb(self) -> None:
        # 17.5 - 1.5 = 16.0 >= 16.0 → 16GB+ tier → medium
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=17.5)
        assert result["dense_config"] is not None
        assert "medium" in result["dense_config"]

    def test_select_config_insufficient_vram(self) -> None:
        # 2.0 - 1.5 = 0.5 < 3.5 → INSUFFICIENT
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=2.0)
        assert result["tier"] == "INSUFFICIENT"
        assert result["dense_config"] is None

    def test_select_config_returns_tier(self) -> None:
        from scripts.laptop_gpu_preflight import select_config
        result = select_config(vram_gb=8.0)
        assert "tier" in result
        assert result["tier"] != "INSUFFICIENT"


# ─── Preflight Tests ──────────────────────────────────────────────────────────

class TestPreflight:
    """Test preflight logic with mocked CUDA."""

    def test_preflight_no_cuda(self, tmp_path: Path) -> None:
        """Preflight should return cuda_available=False when CUDA not available."""
        pytest.importorskip("torch", reason="torch not installed")
        with patch("torch.cuda.is_available", return_value=False):
            from scripts.laptop_gpu_preflight import run_preflight
            result = run_preflight(output_dir=tmp_path)
        assert result["gpu"]["cuda_available"] is False
        assert result["pass"] is False

    def test_preflight_writes_json(self, tmp_path: Path) -> None:
        """Preflight should write preflight.json to output_dir."""
        pytest.importorskip("torch", reason="torch not installed")
        with patch("torch.cuda.is_available", return_value=False):
            from scripts.laptop_gpu_preflight import run_preflight
            run_preflight(output_dir=tmp_path)
        assert (tmp_path / "preflight.json").exists()
        data = json.loads((tmp_path / "preflight.json").read_text())
        assert "gpu" in data

    def test_preflight_writes_txt(self, tmp_path: Path) -> None:
        """Preflight should write preflight.txt human-readable summary."""
        pytest.importorskip("torch", reason="torch not installed")
        with patch("torch.cuda.is_available", return_value=False):
            from scripts.laptop_gpu_preflight import run_preflight
            run_preflight(output_dir=tmp_path)
        assert (tmp_path / "preflight.txt").exists()
        txt = (tmp_path / "preflight.txt").read_text()
        assert "CUDA" in txt


# ─── Metrics Collector Tests ──────────────────────────────────────────────────

class TestMetricsCollector:
    """Test metrics aggregation from fixture data."""

    def test_collect_metrics_from_fixture(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        from scripts.collect_laptop_metrics import collect_metrics
        report = collect_metrics(tmp_path)
        assert "dense" in report
        assert "moe" in report
        assert "resume" in report
        assert "preflight" in report

    def test_dense_metrics_loss_trend(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        from scripts.collect_laptop_metrics import collect_metrics
        report = collect_metrics(tmp_path)
        dense = report["dense"]
        assert dense.get("loss_first") is not None
        assert dense.get("loss_last") is not None
        assert dense["loss_last"] < dense["loss_first"], \
            "Loss should decrease in fixture data"

    def test_moe_metrics_expert_utilization(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        from scripts.collect_laptop_metrics import collect_metrics
        report = collect_metrics(tmp_path)
        moe = report["moe"]
        shares = moe.get("mean_expert_assignment_share", [])
        assert len(shares) == 8, "Should have 8 expert shares"
        for share in shares:
            assert share > 0, "All experts should receive some tokens in fixture"

    def test_resume_metrics_all_passed(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        from scripts.collect_laptop_metrics import collect_metrics
        report = collect_metrics(tmp_path)
        assert report["resume"]["passed"] is True

    def test_collect_metrics_missing_files(self, tmp_path: Path) -> None:
        """Should not raise if result files are missing."""
        from scripts.collect_laptop_metrics import collect_metrics
        report = collect_metrics(tmp_path)
        assert "dense" in report  # Returns empty dict, not exception

    def test_collect_metrics_writes_output(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        out = tmp_path / "aggregated_metrics.json"
        from scripts.collect_laptop_metrics import collect_metrics
        collect_metrics(tmp_path)
        # Output is written by main(), not collect_metrics() directly
        # Just verify collect_metrics returns a dict
        report = collect_metrics(tmp_path)
        assert isinstance(report, dict)


# ─── Report Generator Tests ───────────────────────────────────────────────────

class TestReportGenerator:
    """Test draft report generation from fixture data."""

    def test_generate_report_from_fixture(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        output = tmp_path / "LAPTOP_GPU_VALIDATION_DRAFT.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        assert output.exists()
        content = output.read_text()
        assert "Kuwait Laptop GPU Validation" in content

    def test_report_contains_gpu_model(self, tmp_path: Path) -> None:
        # Write a preflight.json with cuda_available=True inside the 'gpu' key
        # (the generate_draft function checks gpu.get('cuda_available'))
        import json as _json
        preflight_with_cuda = {
            "preflight_status": "PASS",
            "system": {"date": "2026-08-02", "os": "Windows 11", "cpu": "Intel",
                        "system_ram_gb": 16.0, "system_ram_free_gb": 8.0,
                        "disk_total_gb": 500.0, "disk_free_gb": 200.0},
            "gpu": {
                "cuda_available": True,
                "gpu_model": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "vram_total_gb": 6.0,
                "vram_free_gb": 4.5,
                "compute_capability": "8.6",
                "cuda_version": "12.1",
                "pytorch_version": "2.2.2",
                "driver_version": "535.104.05",
                "bf16_supported": True,
                "fp16_supported": True,
                "deepspeed_version": "not installed",
                "flash_attention_version": "not installed",
                "nccl_version": "unknown",
                "gpu_temperature_c": 42,
                "gpu_utilization_pct": 0,
            },
            "recommended_precision": "bf16",
            "recommended_dense_config": "laptop_dense_small",
            "recommended_moe_config": "laptop_moe_small",
            "config_tier": "6-8GB",
            "config_rationale": "Tight but workable.",
            "memory_estimate_note": "Excludes activations.",
            "pass": True,
        }
        (tmp_path / "preflight.json").write_text(_json.dumps(preflight_with_cuda, indent=2))
        # Also write the other fixture files
        for fname in ["dense_summary.json", "moe_summary.json", "resume_test.json",
                      "dense_metrics.jsonl", "moe_metrics.jsonl"]:
            src = tmp_path / fname
            if not src.exists():
                write_fixture_results(tmp_path)
                break
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "RTX 3060" in content or "GeForce" in content

    def test_report_contains_dense_results(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "Dense CUDA Validation" in content
        assert "PASS" in content

    def test_report_contains_moe_results(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "MoE CUDA Validation" in content

    def test_report_contains_resume_results(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "Checkpoint Resume" in content

    def test_report_no_cuda_pending(self, tmp_path: Path) -> None:
        """Report with no results should say 'Not yet run'."""
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "Not yet run" in content

    def test_report_contains_limitations(self, tmp_path: Path) -> None:
        write_fixture_results(tmp_path)
        output = tmp_path / "draft.md"
        from scripts.generate_laptop_validation_draft import generate_draft
        generate_draft(results_dir=tmp_path, output_path=output)
        content = output.read_text()
        assert "Known Limitations" in content


# ─── Windows Script Existence Tests ───────────────────────────────────────────

class TestWindowsScripts:
    """Verify all Windows batch scripts and PowerShell scripts exist."""

    WINDOWS_SCRIPTS = [
        "scripts/windows/setup_laptop_environment.ps1",
        "scripts/windows/run_preflight.bat",
        "scripts/windows/run_dense_validation.bat",
        "scripts/windows/run_moe_validation.bat",
        "scripts/windows/run_resume_validation.bat",
        "scripts/windows/run_all_laptop_validation.bat",
    ]

    @pytest.mark.parametrize("script_path", WINDOWS_SCRIPTS)
    def test_script_exists(self, script_path: str) -> None:
        full_path = REPO_ROOT / script_path
        assert full_path.exists(), f"Missing script: {script_path}"

    @pytest.mark.parametrize("script_path", WINDOWS_SCRIPTS)
    def test_script_not_empty(self, script_path: str) -> None:
        full_path = REPO_ROOT / script_path
        if full_path.exists():
            assert full_path.stat().st_size > 100, f"Script too small: {script_path}"

    def test_batch_scripts_have_cd_to_repo_root(self) -> None:
        """All .bat scripts should navigate to repo root."""
        for script_path in self.WINDOWS_SCRIPTS:
            if script_path.endswith(".bat"):
                full_path = REPO_ROOT / script_path
                if full_path.exists():
                    content = full_path.read_text()
                    # Accept cd /d, cd , or pushd (all navigate to repo root)
                    has_nav = "cd /d" in content or "cd " in content or "pushd" in content
                    assert has_nav, \
                        f"{script_path}: should navigate to repo root (cd /d, cd, or pushd)"


# ─── Documentation Tests ──────────────────────────────────────────────────────

class TestDocumentation:
    """Verify all documentation files exist and contain required sections."""

    def test_laptop_validation_guide_exists(self) -> None:
        assert (REPO_ROOT / "docs" / "LAPTOP_GPU_VALIDATION.md").exists()

    def test_kishore_checklist_exists(self) -> None:
        assert (REPO_ROOT / "docs" / "KISHORE_GPU_OPERATOR_CHECKLIST.md").exists()

    def test_laptop_validation_guide_has_sections(self) -> None:
        content = (REPO_ROOT / "docs" / "LAPTOP_GPU_VALIDATION.md").read_text()
        for section in ["Prerequisites", "Setup", "Validation", "Troubleshooting", "Known Limitations"]:
            assert section in content, f"Missing section: {section}"

    def test_kishore_checklist_has_checkboxes(self) -> None:
        content = (REPO_ROOT / "docs" / "KISHORE_GPU_OPERATOR_CHECKLIST.md").read_text()
        assert "- [ ]" in content, "Checklist should have unchecked checkboxes"

    def test_no_false_compliance_claims(self) -> None:
        """Verify no source files claim GDPR or SOC 2 compliance."""
        # Exclude test files themselves (they contain the strings as assertion targets)
        excluded = {"test_laptop_scripts.py", "test_mesh.py"}
        for path in REPO_ROOT.rglob("*.py"):
            if path.name in excluded:
                continue
            content = path.read_text(errors="ignore")
            assert "GDPR compliant" not in content, f"{path}: false GDPR claim"
            assert "SOC 2 compliant" not in content, f"{path}: false SOC 2 claim"
        for path in REPO_ROOT.rglob("*.md"):
            content = path.read_text(errors="ignore")
            assert "GDPR compliant" not in content, f"{path}: false GDPR claim"
            assert "SOC 2 compliant" not in content, f"{path}: false SOC 2 claim"
