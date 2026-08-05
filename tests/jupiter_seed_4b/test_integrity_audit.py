"""
Jupiter Seed 4B — Dataset Integrity Audit Test Suite
======================================================
Validates all requirements from the integrity audit sprint.
CPU-only. No GPU, model downloads, paid APIs, or cloud resources.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"

sys.path.insert(0, str(TRAINING_DIR))

ALLOWED_DOMAINS = {
    "islamic_finance", "gcc_banking", "telecommunications",
    "government_regulation", "energy_logistics", "executive_decision",
    "arabic_english_correspondence",
}

PROVENANCE_TYPES = {
    "ORIGINAL_SCENARIO", "SOURCE_DEPENDENT_FACTUAL", "PUBLIC_RULE_SUMMARY",
    "TRANSLATION_OR_CORRESPONDENCE", "SAFETY_OR_REFUSAL",
}


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


def skeleton(text: str) -> str:
    t = normalise(text)
    t = re.sub(r"\b\d[\d,\.]*\b", "NUM", t)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# 1. Exact record counts
# ---------------------------------------------------------------------------

def test_exactly_600_train_records() -> None:
    records = load_split("train")
    assert len(records) == 600, f"Expected 600 train records, got {len(records)}"


def test_exactly_100_valid_records() -> None:
    records = load_split("valid")
    assert len(records) == 100, f"Expected 100 valid records, got {len(records)}"


def test_exactly_150_eval_records() -> None:
    records = load_split("eval")
    assert len(records) == 150, f"Expected 150 eval records, got {len(records)}"


def test_total_850_unique_records() -> None:
    all_ids = []
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            all_ids.append(r["example_id"])
    assert len(all_ids) == 850, f"Expected 850 total records, got {len(all_ids)}"
    assert len(set(all_ids)) == 850, "Duplicate example IDs found"


# ---------------------------------------------------------------------------
# 2. Exact authorized language counts
# ---------------------------------------------------------------------------

def test_train_language_distribution() -> None:
    records = load_split("train")
    counts = {}
    for r in records:
        counts[r["language"]] = counts.get(r["language"], 0) + 1
    assert counts.get("ar", 0) == 270, f"Expected 270 ar in train, got {counts.get('ar', 0)}"
    assert counts.get("en", 0) == 180, f"Expected 180 en in train, got {counts.get('en', 0)}"
    assert counts.get("ar-en", 0) == 150, f"Expected 150 ar-en in train, got {counts.get('ar-en', 0)}"


def test_benchmark_language_distribution() -> None:
    bench = load_split("valid") + load_split("eval")
    counts = {}
    for r in bench:
        counts[r["language"]] = counts.get(r["language"], 0) + 1
    assert counts.get("ar", 0) == 113, f"Expected 113 ar in bench, got {counts.get('ar', 0)}"
    assert counts.get("en", 0) == 75, f"Expected 75 en in bench, got {counts.get('en', 0)}"
    assert counts.get("ar-en", 0) == 62, f"Expected 62 ar-en in bench, got {counts.get('ar-en', 0)}"


# ---------------------------------------------------------------------------
# 3. Authorized domain counts
# ---------------------------------------------------------------------------

def test_train_domain_distribution() -> None:
    records = load_split("train")
    counts = {}
    for r in records:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    expected = {
        "islamic_finance": 120, "gcc_banking": 90, "telecommunications": 90,
        "government_regulation": 90, "energy_logistics": 60,
        "executive_decision": 90, "arabic_english_correspondence": 60,
    }
    for domain, target in expected.items():
        assert counts.get(domain, 0) == target, (
            f"Expected {target} {domain} in train, got {counts.get(domain, 0)}"
        )


def test_benchmark_domain_distribution() -> None:
    bench = load_split("valid") + load_split("eval")
    counts = {}
    for r in bench:
        counts[r["domain"]] = counts.get(r["domain"], 0) + 1
    expected = {
        "islamic_finance": 50, "gcc_banking": 38, "telecommunications": 38,
        "government_regulation": 38, "energy_logistics": 25,
        "executive_decision": 38, "arabic_english_correspondence": 23,
    }
    for domain, target in expected.items():
        assert counts.get(domain, 0) == target, (
            f"Expected {target} {domain} in bench, got {counts.get(domain, 0)}"
        )


# ---------------------------------------------------------------------------
# 4. Every record has a provenance classification
# ---------------------------------------------------------------------------

def test_all_records_have_provenance_type() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            assert "provenance_type" in r, (
                f"Missing provenance_type in {split}:{r['example_id']}"
            )
            assert r["provenance_type"] in PROVENANCE_TYPES, (
                f"Unknown provenance_type '{r['provenance_type']}' in {split}:{r['example_id']}"
            )


# ---------------------------------------------------------------------------
# 5. Original scenarios are explicitly fictional
# ---------------------------------------------------------------------------

def test_original_scenarios_have_fictional_disclaimer() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            if r.get("provenance_type") == "ORIGINAL_SCENARIO":
                disclaimer = str(r.get("fictional_disclaimer", "")).lower()
                assert "fictional" in disclaimer, (
                    f"ORIGINAL_SCENARIO missing fictional_disclaimer in {split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# 6. Fact-dependent examples have specific source URLs
# ---------------------------------------------------------------------------

def test_source_dependent_factual_has_url() -> None:
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            if r.get("provenance_type") == "SOURCE_DEPENDENT_FACTUAL":
                url = r.get("source_url", "")
                assert url and url != "https://github.com/agenthinkai/jupiter-shot", (
                    f"SOURCE_DEPENDENT_FACTUAL must have a specific source URL, "
                    f"not the generic repo URL, in {split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# 7. No record relies solely on generic repo URL when factual_dependency=true
# ---------------------------------------------------------------------------

def test_no_factual_dependency_with_generic_url() -> None:
    generic_url = "https://github.com/agenthinkai/jupiter-shot"
    for split in ("train", "valid", "eval"):
        for r in load_split(split):
            if r.get("factual_dependency") is True:
                assert r.get("source_url", "") != generic_url, (
                    f"Record with factual_dependency=True must not use generic repo URL "
                    f"in {split}:{r['example_id']}"
                )


# ---------------------------------------------------------------------------
# 8. No template family exceeds 2%
# ---------------------------------------------------------------------------

def test_no_template_family_exceeds_2pct() -> None:
    all_records = []
    for split in ("train", "valid", "eval"):
        all_records.extend(load_split(split))
    total = len(all_records)
    families: Dict[str, int] = {}
    for r in all_records:
        skel = skeleton(r["prompt"])
        families[skel] = families.get(skel, 0) + 1
    max_pct = max(v / total * 100 for v in families.values())
    assert max_pct <= 2.0, (
        f"Template family exceeds 2%: max is {max_pct:.2f}%"
    )


# ---------------------------------------------------------------------------
# 9. No unresolved REVISE or REJECT audit result
# ---------------------------------------------------------------------------

def test_content_audit_has_no_revise_or_reject() -> None:
    audit_path = DOCS_DIR / "CONTENT_QUALITY_AUDIT.md"
    assert audit_path.exists(), "CONTENT_QUALITY_AUDIT.md not found"
    content = audit_path.read_text(encoding="utf-8")
    # Count REVISE and REJECT in data rows (not headers)
    data_rows = [l for l in content.split("\n") if l.startswith("| `seed4b-")]
    revise_count = sum(1 for row in data_rows if "REVISE" in row)
    reject_count = sum(1 for row in data_rows if "REJECT" in row)
    assert revise_count == 0, f"Content audit has {revise_count} unresolved REVISE decisions"
    assert reject_count == 0, f"Content audit has {reject_count} unresolved REJECT decisions"


# ---------------------------------------------------------------------------
# 10. Benchmark version is 1.1.0
# ---------------------------------------------------------------------------

def test_benchmark_version_is_1_1_0() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["benchmark_version"] == "1.1.0", (
        f"Expected benchmark version 1.1.0, got {manifest['benchmark_version']}"
    )


# ---------------------------------------------------------------------------
# 11. Version 1.0.0 history is preserved
# ---------------------------------------------------------------------------

def test_v1_0_0_manifest_preserved() -> None:
    v100_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.0.0.json"
    assert v100_path.exists(), "v1.0.0 manifest not preserved"
    with v100_path.open() as fh:
        v100 = json.load(fh)
    assert v100["benchmark_version"] == "1.0.0", "Preserved manifest must have version 1.0.0"


# ---------------------------------------------------------------------------
# 12. Frozen benchmark hashes are deterministic
# ---------------------------------------------------------------------------

def test_benchmark_hash_is_deterministic() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    recomputed = sha256_of("|".join(manifest["record_sha256_list"]))
    assert recomputed == manifest["ordered_benchmark_sha256"], (
        "Ordered benchmark hash does not match recomputed value"
    )


# ---------------------------------------------------------------------------
# 13. No material cross-split leakage
# ---------------------------------------------------------------------------

def test_no_train_to_eval_leakage() -> None:
    train_hashes = {sha256_of(normalise(r["prompt"])) for r in load_split("train")}
    for r in load_split("eval"):
        h = sha256_of(normalise(r["prompt"]))
        assert h not in train_hashes, (
            f"Eval prompt found in train: eval:{r['example_id']}"
        )


def test_no_train_to_valid_leakage() -> None:
    train_hashes = {sha256_of(normalise(r["prompt"])) for r in load_split("train")}
    for r in load_split("valid"):
        h = sha256_of(normalise(r["prompt"]))
        assert h not in train_hashes, (
            f"Valid prompt found in train: valid:{r['example_id']}"
        )


# ---------------------------------------------------------------------------
# 14. Human review queue contains exactly 50 records
# ---------------------------------------------------------------------------

def test_human_review_queue_has_exactly_50() -> None:
    path = DATA_DIR / "human_review_queue.jsonl"
    assert path.exists(), "human_review_queue.jsonl not found"
    with path.open() as fh:
        count = sum(1 for l in fh if l.strip())
    assert count == 50, f"Expected exactly 50 review examples, got {count}"


# ---------------------------------------------------------------------------
# 15. Software has not marked human approval
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


# ---------------------------------------------------------------------------
# 16. No teacher outputs were used
# ---------------------------------------------------------------------------

def test_no_teacher_outputs_in_build_dataset() -> None:
    build_path = TRAINING_DIR / "build_dataset.py"
    content = build_path.read_text(encoding="utf-8")
    forbidden = ["openai.api_key", "anthropic.Anthropic", "AzureOpenAI",
                 "AutoModelForCausalLM.from_pretrained", "boto3.client"]
    for pattern in forbidden:
        assert pattern not in content, (
            f"Forbidden pattern '{pattern}' found in build_dataset.py"
        )


# ---------------------------------------------------------------------------
# 17. No weights downloaded, no GPU, no paid API, no cloud
# ---------------------------------------------------------------------------

def test_no_model_download_in_diversity_audit() -> None:
    audit_path = TRAINING_DIR / "diversity_audit.py"
    content = audit_path.read_text(encoding="utf-8")
    assert "from_pretrained" not in content, "diversity_audit.py must not download model weights"
    assert "openai" not in content.lower(), "diversity_audit.py must not call paid APIs"


def test_benchmark_approval_still_pending() -> None:
    manifest_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    assert manifest["approval_status"] == "PENDING HUMAN REVIEW", (
        "Benchmark approval_status must remain PENDING HUMAN REVIEW"
    )


@pytest.mark.skip(reason="GPU required — not executed in CPU unit tests")
def test_gpu_training_not_executed() -> None:
    """Placeholder: GPU training must not be executed without Farouq written approval."""
    pass
