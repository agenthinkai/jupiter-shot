"""
Jupiter Seed 4B — Diagnostic Pipeline CPU Unit Tests
======================================================
Validates all pipeline components without GPU, model downloads,
paid APIs, or cloud resources.

GPU-dependent tests are explicitly marked as SKIPPED, not PASSED.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"

# Add training dir to path for imports
sys.path.insert(0, str(TRAINING_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_EXAMPLE = {
    "example_id": "test-001",
    "source": "internal_authoring",
    "source_url": None,
    "licence_status": "owned",
    "author": "AgenThink Team",
    "creation_method": "manual_authoring",
    "language": "ar-en",
    "domain": "islamic_finance",
    "review_status": "pending_human",
    "modification_history": [],
    "inclusion_reason": "Diagnostic test example",
    "instruction": "ما هو الفرق بين المرابحة والإجارة في التمويل الإسلامي؟",
    "response": (
        "المرابحة هي بيع بثمن مؤجل مع إفصاح عن الربح، "
        "بينما الإجارة هي عقد تأجير يمنح حق الانتفاع دون نقل الملكية."
    ),
}

PROHIBITED_EXAMPLE = {**VALID_EXAMPLE, "example_id": "test-002", "source": "warba_client_data"}
MISSING_FIELD_EXAMPLE = {**VALID_EXAMPLE, "example_id": "test-003", "instruction": ""}
UNKNOWN_DOMAIN_EXAMPLE = {**VALID_EXAMPLE, "example_id": "test-004", "domain": "unknown_domain"}


# ---------------------------------------------------------------------------
# Dataset Loader Tests
# ---------------------------------------------------------------------------

def test_valid_example_passes_validation() -> None:
    from dataset_loader import validate_record
    record = validate_record(VALID_EXAMPLE)
    assert record.example_id == "test-001"
    assert record.domain == "islamic_finance"


def test_prohibited_source_raises_error() -> None:
    from dataset_loader import validate_record, ProvenanceError
    with pytest.raises(ProvenanceError, match="Prohibited source"):
        validate_record(PROHIBITED_EXAMPLE)


def test_missing_instruction_raises_error() -> None:
    from dataset_loader import validate_record, ProvenanceError
    with pytest.raises(ProvenanceError, match="Missing required provenance field"):
        validate_record(MISSING_FIELD_EXAMPLE)


def test_unknown_domain_raises_error() -> None:
    from dataset_loader import validate_record, ProvenanceError
    with pytest.raises(ProvenanceError, match="Unknown domain"):
        validate_record(UNKNOWN_DOMAIN_EXAMPLE)


def test_load_dataset_from_file() -> None:
    from dataset_loader import load_dataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(VALID_EXAMPLE) + "\n")
        tmp_path = Path(f.name)
    try:
        records = load_dataset(tmp_path, strict=True)
        assert len(records) == 1
        assert records[0].example_id == "test-001"
    finally:
        tmp_path.unlink()


def test_provenance_report_generation() -> None:
    from dataset_loader import load_dataset, generate_provenance_report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(VALID_EXAMPLE) + "\n")
        tmp_path = Path(f.name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as rf:
        report_path = Path(rf.name)
    try:
        records = load_dataset(tmp_path, strict=True)
        generate_provenance_report(records, report_path)
        with report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["total_records"] == 1
        assert report["provenance_coverage_pct"] == 100.0
    finally:
        tmp_path.unlink()
        report_path.unlink()


# ---------------------------------------------------------------------------
# Arabic Review Tests
# ---------------------------------------------------------------------------

def test_arabic_review_safe_example() -> None:
    from arabic_review import review_example
    result = review_example(VALID_EXAMPLE)
    assert result.safety_score == 5
    assert result.rejection_reason is None
    assert result.msa_quality > 1  # Arabic text detected


def test_arabic_review_unsafe_example() -> None:
    from arabic_review import review_example
    unsafe = {**VALID_EXAMPLE, "response": "kill the enemy with a bomb"}
    result = review_example(unsafe)
    assert result.safety_score == 1
    assert result.rejection_reason == "UNSAFE_CONTENT_DETECTED"


def test_arabic_review_outputs_ai_note() -> None:
    from arabic_review import review_example
    result = review_example(VALID_EXAMPLE)
    assert "AI-ASSISTED REVIEW ONLY" in result.ai_review_note


def test_arabic_review_pipeline_writes_output() -> None:
    from arabic_review import run_review
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(VALID_EXAMPLE) + "\n")
        tmp_path = Path(f.name)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "review_results.jsonl"
        run_review(tmp_path, output_path, top_n_human=1)
        assert output_path.exists()
        with output_path.open(encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 1
    tmp_path.unlink()


# ---------------------------------------------------------------------------
# Training Runner Tests (CPU / dry-run only)
# ---------------------------------------------------------------------------

def test_training_config_loads() -> None:
    import yaml
    config_path = TRAINING_DIR / "configs" / "diagnostic_qlora.yaml"
    assert config_path.exists(), "Training config not found"
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["model_name_or_path"] == "Qwen/Qwen3-4B", (
        "Base model must be Qwen/Qwen3-4B — do not silently change it"
    )
    assert cfg["max_authorized_spend_usd"] == 500


def test_cost_estimator_within_budget() -> None:
    from training_runner import estimate_cost, load_config
    config_path = TRAINING_DIR / "configs" / "diagnostic_qlora.yaml"
    cfg = load_config(config_path)
    estimate = estimate_cost(cfg, num_examples=500)
    assert estimate["within_budget"], (
        f"Estimated cost USD {estimate['estimated_cost_usd']} exceeds authorized USD 500"
    )
    assert estimate["base_model"] == "Qwen/Qwen3-4B"


@pytest.mark.skip(reason="GPU required — not executed in CPU unit tests")
def test_training_runner_gpu() -> None:
    """Placeholder: GPU training test. Must be run manually on target hardware."""
    pass


# ---------------------------------------------------------------------------
# Evaluation Tests (CPU mock)
# ---------------------------------------------------------------------------

def test_evaluate_returns_mock_on_cpu() -> None:
    from evaluate import evaluate_model
    mock_examples = [
        {
            "instruction": "What is Murabaha?",
            "expected_answer": "Murabaha",
            "dimension": "islamic_finance_terminology",
        }
    ]
    result = evaluate_model("Qwen/Qwen3-4B", mock_examples, device="cpu")
    # evaluation_mode must be MOCK (updated from CPU_MOCK_DO_NOT_USE_AS_REAL_RESULTS)
    assert result["evaluation_mode"] == "MOCK"
    assert result["authoritative"] is False
    assert result["model_loaded"] is False
    assert "domain_scores" in result


# ---------------------------------------------------------------------------
# Inference Server Tests
# ---------------------------------------------------------------------------

def test_inference_server_creates_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inference_server import create_app

    original_cwd = Path.cwd()
    status_before = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    repository_artifacts = REPO_ROOT / "artifacts"
    assert not repository_artifacts.exists(), "test requires a clean repository artifact path"

    monkeypatch.chdir(tmp_path)
    app = create_app("mock_model_path", backend="transformers")
    compliance_log = tmp_path / "artifacts" / "compliance_log.jsonl"
    assert app is not None
    assert app.title == "Jupiter Seed 4B Inference Server"
    assert (tmp_path / "artifacts").is_dir()
    assert not repository_artifacts.exists()
    if compliance_log.exists():
        assert compliance_log.is_file()
        assert compliance_log.is_relative_to(tmp_path)

    monkeypatch.undo()
    status_after = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    assert Path.cwd() == original_cwd
    assert status_after == status_before
    assert not repository_artifacts.exists()


def test_inference_server_health_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inference_server import create_app
    from fastapi.testclient import TestClient

    original_cwd = Path.cwd()
    status_before = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    repository_artifacts = REPO_ROOT / "artifacts"
    assert not repository_artifacts.exists(), "test requires a clean repository artifact path"

    monkeypatch.chdir(tmp_path)
    try:
        app = create_app("mock_model_path", backend="transformers")
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "NOT production-ready" in data["disclaimer"]
        compliance_log = tmp_path / "artifacts" / "compliance_log.jsonl"
        assert (tmp_path / "artifacts").is_dir()
        assert not repository_artifacts.exists()
        if compliance_log.exists():
            assert compliance_log.is_file()
            assert compliance_log.is_relative_to(tmp_path)
    finally:
        monkeypatch.undo()

    status_after = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout
    assert Path.cwd() == original_cwd
    assert status_after == status_before
    assert not repository_artifacts.exists()


# ---------------------------------------------------------------------------
# Documentation Completeness Tests
# ---------------------------------------------------------------------------

REQUIRED_DOCS = [
    "BUILD_RUNBOOK.md",
    "MODEL_CARD_DRAFT.md",
    "DATASET_CARD_DRAFT.md",
    "DIAGNOSTIC_RESULTS_TEMPLATE.md",
    "COST_LEDGER.md",
    "RELEASE_GATE.md",
]


def test_all_required_docs_exist() -> None:
    for doc in REQUIRED_DOCS:
        path = DOCS_DIR / doc
        assert path.exists(), f"Required document missing: {doc}"


def test_release_gate_has_no_approvals() -> None:
    """Ensure no release gate checkboxes are pre-checked."""
    gate_path = DOCS_DIR / "RELEASE_GATE.md"
    content = gate_path.read_text(encoding="utf-8")
    assert "[x]" not in content.lower(), (
        "Release gate must not have pre-checked items — no release is authorized"
    )
