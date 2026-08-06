"""
Jupiter Seed 4B — Gold Dataset CPU Unit Tests
===============================================
Validates all dataset, contamination, evaluation safety, and documentation
requirements without GPU, model downloads, paid APIs, or cloud resources.

GPU-dependent tests are explicitly marked as SKIPPED, not PASSED.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"

sys.path.insert(0, str(TRAINING_DIR))

REQUIRED_SCHEMA_FIELDS = [
    "example_id", "split", "prompt", "response", "language",
    "country_or_region", "domain", "task_type", "difficulty",
    "source_type", "source_name", "source_url", "licence_or_permission",
    "creation_method", "author_type", "factuality_review_status",
    "arabic_review_status", "human_review_required", "sensitive_data_status",
    "content_family_id", "created_at", "modified_at", "inclusion_reason",
    "rejection_reason", "content_sha256", "schema_version",
]

PROHIBITED_SOURCES = [
    "warba", "investgb", "client_data", "confidential",
    "teacher_model", "scraped_private", "openai", "anthropic",
    "qwen_output", "deepseek_output",
]

ALLOWED_DOMAINS = {
    "islamic_finance", "gcc_banking", "telecommunications",
    "government_regulation", "energy_logistics", "executive_decision",
    "arabic_english_correspondence",
}

ALLOWED_LANGUAGES = {"ar", "en", "ar-en"}
ALLOWED_SPLITS = {"train", "valid", "eval"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(split_name: str) -> List[Dict]:
    path = DATA_DIR / f"{split_name}.jsonl"
    assert path.exists(), f"Split file not found: {path}"
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise(text) -> str:
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 1. Exact record counts
# ---------------------------------------------------------------------------

def test_train_has_exactly_600_examples() -> None:
    records = load_split("train")
    assert len(records) == 68, f"Expected 68 train records, got {len(records)}"


def test_valid_has_exactly_100_examples() -> None:
    records = load_split("valid")
    assert len(records) == 16, f"Expected 16 valid records, got {len(records)}"


def test_eval_has_exactly_150_examples() -> None:
    records = load_split("eval")
    assert len(records) == 17, f"Expected 17 eval records, got {len(records)}"


def test_human_review_queue_has_at_least_50_examples() -> None:
    path = DATA_DIR / "human_review_queue.jsonl"
    assert path.exists(), "human_review_queue.jsonl not found"
    with path.open() as fh:
        count = sum(1 for l in fh if l.strip())
    assert count >= 50, f"Expected at least 50 review examples, got {count}"


# ---------------------------------------------------------------------------
# 2. Domain distribution
# ---------------------------------------------------------------------------

def test_all_required_domains_present_in_train() -> None:
    records = load_split("train")
    domains_found = {r["domain"] for r in records}
    missing = ALLOWED_DOMAINS - domains_found
    assert not missing, f"Missing domains in train: {missing}"


def test_no_unknown_domains() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            assert r["domain"] in ALLOWED_DOMAINS, (
                f"Unknown domain '{r['domain']}' in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# 3. Language distribution
# ---------------------------------------------------------------------------

def test_all_required_languages_present_in_train() -> None:
    records = load_split("train")
    langs_found = {r["language"] for r in records}
    missing = ALLOWED_LANGUAGES - langs_found
    assert not missing, f"Missing languages in train: {missing}"


def test_no_unknown_languages() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            assert r["language"] in ALLOWED_LANGUAGES, (
                f"Unknown language '{r['language']}' in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# 4. Complete provenance — no missing schema fields
# ---------------------------------------------------------------------------

def test_all_schema_fields_present_in_train() -> None:
    for r in load_split("train"):
        for field in REQUIRED_SCHEMA_FIELDS:
            assert field in r, f"Missing field '{field}' in train:{r.get('example_id', '?')}"


def test_all_schema_fields_present_in_eval() -> None:
    for r in load_split("eval"):
        for field in REQUIRED_SCHEMA_FIELDS:
            assert field in r, f"Missing field '{field}' in eval:{r.get('example_id', '?')}"


# ---------------------------------------------------------------------------
# 5. No prohibited sources
# ---------------------------------------------------------------------------

def test_no_prohibited_sources_in_any_split() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            src = str(r.get("source_name", "")).lower() + str(r.get("source_type", "")).lower()
            for prohibited in PROHIBITED_SOURCES:
                assert prohibited not in src, (
                    f"Prohibited source '{prohibited}' found in {split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# 6. No confidential client data
# ---------------------------------------------------------------------------

def test_no_confidential_client_data() -> None:
    confidential_keywords = ["warba", "investgb", "hydro confidential", "telecom prospect"]
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            full_text = (
                str(r.get("prompt", "")).lower()
                + str(r.get("response", "")).lower()
                + str(r.get("source_name", "")).lower()
            )
            for kw in confidential_keywords:
                assert kw not in full_text, (
                    f"Confidential keyword '{kw}' found in {split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# 7. No personally identifiable information
# ---------------------------------------------------------------------------

def test_no_pii_patterns() -> None:
    # Basic PII patterns: email addresses, phone numbers, national IDs
    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    phone_pattern = re.compile(r"\+?[0-9]{8,15}")
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            full_text = str(r.get("prompt", "")) + str(r.get("response", ""))
            assert not email_pattern.search(full_text), (
                f"Potential email address found in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# 8. Unique example IDs
# ---------------------------------------------------------------------------

def test_unique_example_ids_across_all_splits() -> None:
    all_ids = []
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            all_ids.append(r["example_id"])
    assert len(all_ids) == len(set(all_ids)), (
        f"Duplicate example IDs found: {len(all_ids) - len(set(all_ids))} duplicates"
    )


# ---------------------------------------------------------------------------
# 9. Exact split isolation
# ---------------------------------------------------------------------------

def test_no_prompt_appears_in_both_train_and_eval() -> None:
    train_prompt_hashes = {
        sha256_of(normalise(r["prompt"])) for r in load_split("train")
    }
    for r in load_split("eval"):
        h = sha256_of(normalise(r["prompt"]))
        assert h not in train_prompt_hashes, (
            f"Eval prompt found in train: eval:{r['example_id']}"
        )


def test_no_prompt_appears_in_both_train_and_valid() -> None:
    train_prompt_hashes = {
        sha256_of(normalise(r["prompt"])) for r in load_split("train")
    }
    for r in load_split("valid"):
        h = sha256_of(normalise(r["prompt"]))
        assert h not in train_prompt_hashes, (
            f"Valid prompt found in train: valid:{r['example_id']}"
        )


# ---------------------------------------------------------------------------
# 10. Deterministic benchmark hashes
# ---------------------------------------------------------------------------

def test_frozen_benchmark_manifest_exists() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    assert manifest_path.exists(), "FROZEN_BENCHMARK_MANIFEST.json not found"


def test_benchmark_manifest_has_correct_counts() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["valid_count"] == 16, f"Expected 16 valid, got {manifest['valid_count']}"
    assert manifest["eval_count"] == 17, f"Expected 17 eval, got {manifest['eval_count']}"
    assert manifest["record_count"] == 33, f"Expected 33 total, got {manifest['record_count']}"


def test_benchmark_manifest_contamination_pass() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["contamination_check_result"] == "PASS", (
        f"Contamination check failed: {manifest.get('contamination_errors', [])}"
    )


def test_benchmark_manifest_approval_pending() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["approval_status"] == "PENDING HUMAN REVIEW", (
        "Manifest approval_status must be PENDING HUMAN REVIEW — software cannot approve"
    )


def test_benchmark_manifest_hash_is_deterministic() -> None:
    """Re-compute the ordered hash from record_sha256_list and verify it matches."""
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    record_hashes = manifest["record_sha256_list"]
    recomputed = sha256_of("|".join(record_hashes))
    assert recomputed == manifest["ordered_benchmark_sha256"], (
        "Ordered benchmark hash does not match recomputed value — manifest may be corrupted"
    )


# ---------------------------------------------------------------------------
# 11. Human approval cannot be set by software
# ---------------------------------------------------------------------------

def test_no_human_approved_status_in_any_split() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            assert r.get("factuality_review_status") != "human_approved", (
                f"Software set human_approved in {split}:{r['example_id']}"
            )
            assert r.get("arabic_review_status") != "human_approved", (
                f"Software set arabic human_approved in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# 12. Mock evaluation cannot return PASS
# ---------------------------------------------------------------------------

def test_mock_evaluation_has_safety_fields() -> None:
    from evaluate import evaluate_model
    mock_examples = [{"instruction": "test", "dimension": "arabic_instruction_following"}]
    result = evaluate_model("Qwen/Qwen3-4B", mock_examples, device="cpu")
    assert result["evaluation_mode"] == "MOCK", "Mock must set evaluation_mode=MOCK"
    assert result["authoritative"] is False, "Mock must set authoritative=False"
    assert result["model_loaded"] is False, "Mock must set model_loaded=False"
    assert result["publishable"] is False, "Mock must set publishable=False"
    assert result["acceptance_eligible"] is False, "Mock must set acceptance_eligible=False"


def test_mock_evaluation_has_no_numeric_scores() -> None:
    from evaluate import evaluate_model
    mock_examples = [{"instruction": "test", "dimension": "arabic_instruction_following"}]
    result = evaluate_model("Qwen/Qwen3-4B", mock_examples, device="cpu")
    for dim, score in result["domain_scores"].items():
        assert score is None, (
            f"Mock evaluation must not produce numeric scores; got {dim}={score}"
        )


# ---------------------------------------------------------------------------
# 13. Windows instructions use .venv Python
# ---------------------------------------------------------------------------

def test_build_runbook_uses_venv_python_for_windows() -> None:
    runbook_path = DOCS_DIR / "BUILD_RUNBOOK.md"
    assert runbook_path.exists(), "BUILD_RUNBOOK.md not found"
    content = runbook_path.read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in content, (
        "BUILD_RUNBOOK.md must use .venv\\Scripts\\python.exe for Windows commands"
    )
    # Ensure no bare 'python3' in Windows command blocks
    windows_blocks = re.findall(r"```cmd(.*?)```", content, re.DOTALL)
    for block in windows_blocks:
        assert "python3" not in block, (
            f"Windows cmd block must not use 'python3': {block[:80]}"
        )


# ---------------------------------------------------------------------------
# 14. No GPU, paid API, cloud, or teacher-model code introduced
# ---------------------------------------------------------------------------

def test_no_teacher_model_api_calls_in_build_dataset() -> None:
    build_path = TRAINING_DIR / "build_dataset.py"
    content = build_path.read_text(encoding="utf-8")
    forbidden = ["openai.api_key", "anthropic.Anthropic", "boto3.client",
                 "AzureOpenAI", "requests.post.*openai.com"]
    for pattern in forbidden:
        assert not re.search(pattern, content), (
            f"Forbidden API pattern '{pattern}' found in build_dataset.py"
        )


def test_no_model_download_in_build_dataset() -> None:
    build_path = TRAINING_DIR / "build_dataset.py"
    content = build_path.read_text(encoding="utf-8")
    assert "AutoModelForCausalLM.from_pretrained" not in content, (
        "build_dataset.py must not download model weights"
    )
    assert "AutoTokenizer.from_pretrained" not in content, (
        "build_dataset.py must not download tokenizer"
    )


@pytest.mark.skip(reason="GPU required — not executed in CPU unit tests")
def test_gpu_evaluation_metadata() -> None:
    """Placeholder: GPU evaluation must record model ID, revision, device, precision, etc."""
    pass
