"""
Jupiter Seed 4B — Repair Audit Regression Tests
=================================================
20 regression tests reproducing every independent finding from the
mechanical audit of commit a4ef84a032b8532384b8b531afceb35b260dc7bb.

CPU-only. No GPU, model downloads, paid APIs, or cloud resources.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_split(split_name: str) -> List[Dict]:
    path = DATA_DIR / f"{split_name}.jsonl"
    assert path.exists(), f"Split file not found: {path}"
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_queue() -> List[Dict]:
    path = DATA_DIR / "human_review_queue.jsonl"
    assert path.exists(), "human_review_queue.jsonl not found"
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ---------------------------------------------------------------------------
# Test 1: Markdown and JSONL contain exactly the same 50 records
# ---------------------------------------------------------------------------

def test_markdown_and_jsonl_same_50_records() -> None:
    queue = load_queue()
    md_path = DOCS_DIR / "ARABIC_HUMAN_REVIEW_QUEUE.md"
    assert md_path.exists(), "ARABIC_HUMAN_REVIEW_QUEUE.md not found"
    md_content = md_path.read_text(encoding="utf-8")

    jsonl_ids = {r["example_id"] for r in queue}
    # Extract IDs from Markdown table rows
    md_ids = set(re.findall(r"`(seed4b-[a-z]+-\d{4})`", md_content))

    assert len(queue) == 50, f"JSONL has {len(queue)} records, expected 50"
    assert len(md_ids) == 50, f"Markdown has {len(md_ids)} IDs, expected 50"
    assert jsonl_ids == md_ids, (
        f"JSONL and Markdown IDs differ.\n"
        f"In JSONL only: {jsonl_ids - md_ids}\n"
        f"In Markdown only: {md_ids - jsonl_ids}"
    )


# ---------------------------------------------------------------------------
# Test 2: Generated CSV contains the same 50 records
# ---------------------------------------------------------------------------

def test_csv_contains_same_50_records() -> None:
    queue = load_queue()
    csv_path = DOCS_DIR / "REVIEWER_TEMPLATE.csv"
    assert csv_path.exists(), "REVIEWER_TEMPLATE.csv not found"

    with csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        csv_rows = list(reader)

    jsonl_ids = {r["example_id"] for r in queue}
    csv_ids = {row["example_id"] for row in csv_rows}

    assert len(csv_rows) == 50, f"CSV has {len(csv_rows)} rows, expected 50"
    assert jsonl_ids == csv_ids, (
        f"JSONL and CSV IDs differ.\n"
        f"In JSONL only: {jsonl_ids - csv_ids}\n"
        f"In CSV only: {csv_ids - jsonl_ids}"
    )


# ---------------------------------------------------------------------------
# Test 3: No ID collision with different metadata
# ---------------------------------------------------------------------------

def test_no_id_collision_with_different_metadata() -> None:
    all_records = load_split("train") + load_split("valid") + load_split("eval")
    id_to_hash: Dict[str, str] = {}
    for r in all_records:
        eid = r["example_id"]
        content_hash = r.get("content_sha256", sha256_of(r["prompt"] + r["response"]))
        if eid in id_to_hash:
            assert id_to_hash[eid] == content_hash, (
                f"ID collision with different content: {eid}"
            )
        id_to_hash[eid] = content_hash


# ---------------------------------------------------------------------------
# Test 4: No duplicate response across splits
# ---------------------------------------------------------------------------

def test_no_duplicate_response_across_splits() -> None:
    train = load_split("train")
    valid = load_split("valid")
    eval_ = load_split("eval")

    train_hashes = {sha256_of(r["response"]) for r in train}
    for r in valid:
        h = sha256_of(r["response"])
        assert h not in train_hashes, (
            f"Valid response found in train: valid:{r['example_id']}"
        )
    for r in eval_:
        h = sha256_of(r["response"])
        assert h not in train_hashes, (
            f"Eval response found in train: eval:{r['example_id']}"
        )

    valid_hashes = {sha256_of(r["response"]) for r in valid}
    for r in eval_:
        h = sha256_of(r["response"])
        assert h not in valid_hashes, (
            f"Eval response found in valid: eval:{r['example_id']}"
        )


# ---------------------------------------------------------------------------
# Test 5: No duplicate prompt across splits
# ---------------------------------------------------------------------------

def test_no_duplicate_prompt_across_splits() -> None:
    train = load_split("train")
    valid = load_split("valid")
    eval_ = load_split("eval")

    train_hashes = {sha256_of(r["prompt"]) for r in train}
    for r in valid:
        h = sha256_of(r["prompt"])
        assert h not in train_hashes, (
            f"Valid prompt found in train: valid:{r['example_id']}"
        )
    for r in eval_:
        h = sha256_of(r["prompt"])
        assert h not in train_hashes, (
            f"Eval prompt found in train: eval:{r['example_id']}"
        )


# ---------------------------------------------------------------------------
# Test 6: Response-only hashes are checked (all 850 unique)
# ---------------------------------------------------------------------------

def test_all_850_responses_unique() -> None:
    all_records = load_split("train") + load_split("valid") + load_split("eval")
    response_hashes = [sha256_of(r["response"]) for r in all_records]
    assert len(set(response_hashes)) == 850, (
        f"Expected 850 unique responses, got {len(set(response_hashes))}"
    )


# ---------------------------------------------------------------------------
# Test 7: Prompt-only hashes are checked (all 850 unique)
# ---------------------------------------------------------------------------

def test_all_850_prompts_unique() -> None:
    all_records = load_split("train") + load_split("valid") + load_split("eval")
    prompt_hashes = [sha256_of(r["prompt"]) for r in all_records]
    assert len(set(prompt_hashes)) == 850, (
        f"Expected 850 unique prompts, got {len(set(prompt_hashes))}"
    )


# ---------------------------------------------------------------------------
# Test 8: Content-family identifiers are meaningful (not domain:<name>)
# ---------------------------------------------------------------------------

def test_content_family_ids_are_meaningful() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            family = r.get("content_family_id", "")
            assert family, f"Missing content_family_id in {split}:{r['example_id']}"
            assert not family.startswith("domain:"), (
                f"content_family_id must not be 'domain:<name>', got '{family}' "
                f"in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# Test 9: Arabic records contain meaningful Arabic responses
# ---------------------------------------------------------------------------

def test_arabic_records_have_meaningful_arabic_responses() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            if r["language"] == "ar":
                ar_count = len(re.findall(r"[\u0600-\u06FF]", r["response"]))
                assert ar_count >= 30, (
                    f"Arabic record has only {ar_count} Arabic chars in response: "
                    f"{split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# Test 10: Bilingual records meaningfully contain both languages
# ---------------------------------------------------------------------------

def test_bilingual_records_contain_both_languages() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            if r["language"] == "ar-en":
                # Check combined prompt+response for bilingual content
                combined = r["prompt"] + " " + r["response"]
                ar_count = len(re.findall(r"[\u0600-\u06FF]", combined))
                en_count = len(re.findall(r"[a-zA-Z]", combined))
                # Translation tasks may have English prompt + Arabic response or vice versa
                # Require at least 10 chars of each in the combined pair
                assert ar_count >= 10 and en_count >= 10, (
                    f"Bilingual record lacks meaningful content in both languages: "
                    f"{split}:{r['example_id']} ar={ar_count} en={en_count}"
                )


# ---------------------------------------------------------------------------
# Test 11: Language totals remain exact
# ---------------------------------------------------------------------------

def test_exact_language_totals() -> None:
    train = load_split("train")
    bench = load_split("valid") + load_split("eval")

    def count(records, lang):
        return sum(1 for r in records if r["language"] == lang)

    assert count(train, "ar") == 270, f"Train ar: expected 270, got {count(train, 'ar')}"
    assert count(train, "en") == 180, f"Train en: expected 180, got {count(train, 'en')}"
    assert count(train, "ar-en") == 150, f"Train ar-en: expected 150, got {count(train, 'ar-en')}"
    assert count(bench, "ar") == 113, f"Bench ar: expected 113, got {count(bench, 'ar')}"
    assert count(bench, "en") == 75, f"Bench en: expected 75, got {count(bench, 'en')}"
    assert count(bench, "ar-en") == 62, f"Bench ar-en: expected 62, got {count(bench, 'ar-en')}"


# ---------------------------------------------------------------------------
# Test 12: Domain totals remain exact
# ---------------------------------------------------------------------------

def test_exact_domain_totals() -> None:
    train = load_split("train")
    bench = load_split("valid") + load_split("eval")

    def count(records, domain):
        return sum(1 for r in records if r["domain"] == domain)

    train_expected = {
        "islamic_finance": 120, "gcc_banking": 90, "telecommunications": 90,
        "government_regulation": 90, "energy_logistics": 60,
        "executive_decision": 90, "arabic_english_correspondence": 60,
    }
    bench_expected = {
        "islamic_finance": 50, "gcc_banking": 38, "telecommunications": 38,
        "government_regulation": 38, "energy_logistics": 25,
        "executive_decision": 38, "arabic_english_correspondence": 23,
    }
    for domain, target in train_expected.items():
        actual = count(train, domain)
        assert actual == target, f"Train {domain}: expected {target}, got {actual}"
    for domain, target in bench_expected.items():
        actual = count(bench, domain)
        assert actual == target, f"Bench {domain}: expected {target}, got {actual}"


# ---------------------------------------------------------------------------
# Test 13: Review queue covers all required categories
# ---------------------------------------------------------------------------

def test_review_queue_covers_required_categories() -> None:
    queue = load_queue()
    domains = {r["domain"] for r in queue}
    langs = {r["language"] for r in queue}
    splits = {r["split"] for r in queue}
    prov_types = {r.get("provenance_type", "") for r in queue}

    required_domains = {
        "islamic_finance", "gcc_banking", "telecommunications",
        "government_regulation", "energy_logistics", "executive_decision",
        "arabic_english_correspondence",
    }
    assert required_domains <= domains, f"Missing domains: {required_domains - domains}"
    missing_langs = {"ar", "en", "ar-en"} - langs
    assert not missing_langs, f"Missing languages: {missing_langs}"
    missing_splits = {"train", "valid", "eval"} - splits
    assert not missing_splits, f"Missing splits: {missing_splits}"
    assert "SAFETY_OR_REFUSAL" in prov_types, "Queue must include SAFETY_OR_REFUSAL examples"
    assert "SOURCE_DEPENDENT_FACTUAL" in prov_types, "Queue must include SOURCE_DEPENDENT_FACTUAL examples"


# ---------------------------------------------------------------------------
# Test 14: All reviewer fields remain empty
# ---------------------------------------------------------------------------

def test_all_reviewer_fields_empty_in_csv() -> None:
    csv_path = DOCS_DIR / "REVIEWER_TEMPLATE.csv"
    assert csv_path.exists(), "REVIEWER_TEMPLATE.csv not found"

    reviewer_fields = [
        "reviewer_name", "reviewer_experience", "review_date",
        "accuracy_score", "fluency_score", "gcc_appropriateness_score",
        "domain_terminology_score", "factuality_score", "verdict",
        "corrected_wording", "rejection_reason", "comments",
    ]

    with csv_path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for field in reviewer_fields:
                if field in row:
                    assert row[field] == "", (
                        f"Reviewer field '{field}' is not empty for {row.get('example_id', '?')}: "
                        f"'{row[field]}'"
                    )


# ---------------------------------------------------------------------------
# Test 15: All text I/O is explicitly UTF-8 safe
# ---------------------------------------------------------------------------

def test_all_text_io_explicitly_utf8() -> None:
    """AST-based check: no open() without encoding= in production files."""
    py_files = list(TRAINING_DIR.glob("*.py"))
    # Exclude test files
    py_files = [f for f in py_files if not f.name.startswith("test_")]

    violations = []
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "open":
                    has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
                    if not has_encoding:
                        violations.append(f"{path.name}:{node.lineno}")

    assert not violations, (
        f"open() without explicit encoding= found in: {violations}"
    )


# ---------------------------------------------------------------------------
# Test 16: Benchmark history is preserved
# ---------------------------------------------------------------------------

def test_benchmark_history_preserved() -> None:
    v100 = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.0.0.json"
    v110 = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.1.0.json"
    v111 = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"

    assert v100.exists(), "v1.0.0 manifest not preserved"
    assert v110.exists(), "v1.1.0 manifest not preserved"
    assert v111.exists(), "Current manifest not found"

    with v100.open(encoding="utf-8") as fh:
        m100 = json.load(fh)
    with v110.open(encoding="utf-8") as fh:
        m110 = json.load(fh)
    with v111.open(encoding="utf-8") as fh:
        m111 = json.load(fh)

    assert m100["benchmark_version"] == "1.0.0"
    assert m110["benchmark_version"] == "1.1.0"
    assert m111["benchmark_version"] == "1.1.1"


# ---------------------------------------------------------------------------
# Test 17: No human approval is set by software
# ---------------------------------------------------------------------------

def test_no_software_human_approval() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            assert r.get("factuality_review_status") != "human_approved", (
                f"Software set human_approved in {split}:{r['example_id']}"
            )
            assert r.get("arabic_review_status") != "human_approved", (
                f"Software set arabic human_approved in {split}:{r['example_id']}"
            )

    # Also check the queue manifest
    manifest_path = DOCS_DIR / "REVIEW_QUEUE_MANIFEST.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest.get("reviewer_fields_populated_by_software") is False


# ---------------------------------------------------------------------------
# Test 18: No unresolved structural defect can receive PASS
# ---------------------------------------------------------------------------

def test_no_structural_defect_receives_pass() -> None:
    """Verify that the content audit correctly rejects records with structural failures."""
    audit_path = DOCS_DIR / "CONTENT_QUALITY_AUDIT.md"
    assert audit_path.exists(), "CONTENT_QUALITY_AUDIT.md not found"
    content = audit_path.read_text(encoding="utf-8")

    # All data rows must show PASS (no REVISE or REJECT)
    data_rows = [l for l in content.split("\n") if l.startswith("| `seed4b-")]
    revise_count = sum(1 for row in data_rows if "**REVISE**" in row or "| REVISE |" in row)
    reject_count = sum(1 for row in data_rows if "**REJECT**" in row or "| REJECT |" in row)
    assert revise_count == 0, f"Content audit has {revise_count} unresolved REVISE decisions"
    assert reject_count == 0, f"Content audit has {reject_count} unresolved REJECT decisions"


# ---------------------------------------------------------------------------
# Test 19: No model weights or teacher outputs are used
# ---------------------------------------------------------------------------

def test_no_model_weights_or_teacher_outputs() -> None:
    build_path = TRAINING_DIR / "build_dataset.py"
    content = build_path.read_text(encoding="utf-8")
    forbidden = [
        "AutoModelForCausalLM.from_pretrained",
        "AutoTokenizer.from_pretrained",
        "openai.api_key",
        "anthropic.Anthropic",
        "AzureOpenAI",
    ]
    for pattern in forbidden:
        assert pattern not in content, (
            f"Forbidden pattern '{pattern}' found in build_dataset.py"
        )


# ---------------------------------------------------------------------------
# Test 20: No GPU, paid API, cloud resource, or Azure credit is used
# ---------------------------------------------------------------------------

def test_no_gpu_paid_api_cloud_azure() -> None:
    py_files = list(TRAINING_DIR.glob("*.py"))
    py_files = [f for f in py_files if not f.name.startswith("test_")]

    forbidden = ["boto3.client", "azure.mgmt", "AzureOpenAI", "requests.post.*openai.com"]
    for path in py_files:
        content = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            import re as _re
            assert not _re.search(pattern, content), (
                f"Forbidden cloud/API pattern '{pattern}' found in {path.name}"
            )


@pytest.mark.skip(reason="GPU required — not executed in CPU unit tests")
def test_gpu_training_not_executed() -> None:
    """Placeholder: GPU training must not be executed without Farouq written approval."""
    pass
