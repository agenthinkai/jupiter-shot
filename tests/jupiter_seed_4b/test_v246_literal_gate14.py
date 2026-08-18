"""V2.4.6 regression coverage for literal Gate 14 Arabic recall and effective review set."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training" / "jupiter_seed_4b"
DATA = TRAINING / "data"
sys.path.insert(0, str(TRAINING))

from readiness_gate import (  # noqa: E402
    GCC_TRIP_PHRASES,
    GCC_TRIP_RULE_IDS,
    derive_effective_review_set,
    gate_content_risk,
    gate_review_queue_coverage,
    load_all,
)

AR_IDS = {
    "seed4b-eval-0004",
    "seed4b-train-0015",
    "seed4b-train-0016",
    "seed4b-train-0028",
    "seed4b-train-0038",
    "seed4b-train-0039",
    "seed4b-valid-0005",
    "seed4b-valid-0014",
}
EN_IDS = {"seed4b-train-0032"}
ALL_FLAGGED = AR_IDS | EN_IDS
SUPPLEMENT_IDS = ["seed4b-train-0016", "seed4b-train-0028", "seed4b-valid-0014"]


def records():
    return load_all(DATA)


def gate14():
    return gate_content_risk(records())


def generate_temp_package(tmp_path: Path) -> Path:
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    result = subprocess.run(
        [sys.executable, str(TRAINING / "generate_review_package.py"),
         "--docs-dir", str(tmp_path), "--package-commit", source_commit],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path


def package_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestLiteralArabicRule:
    def test_exactly_eight_arabic_response_matches(self):
        g14 = gate14()
        findings = [f for f in g14.metadata["findings"] if f["rule_id"] == "GCC_TRIP_AR"]
        assert {f["example_id"] for f in findings} == AR_IDS
        assert {f["field"] for f in findings} == {"response"}
        assert len(findings) == 8

    def test_exactly_one_english_response_match(self):
        g14 = gate14()
        findings = [f for f in g14.metadata["findings"] if f["rule_id"] == "GCC_TRIP_EN"]
        assert {f["example_id"] for f in findings} == EN_IDS
        assert {f["field"] for f in findings} == {"response"}
        assert len(findings) == 1

    def test_nine_total_findings_and_no_prompt_only_match(self):
        g14 = gate14()
        assert set(g14.metadata["review_required_records"]) == ALL_FLAGGED
        assert g14.metadata["review_required_count"] == 9
        prompt_only = {"example_id": "prompt-only", "prompt": "يُنصح بمراجعة", "response": "A normal response."}
        assert all(f["example_id"] != "prompt-only" for f in gate_content_risk([prompt_only]).metadata["findings"])

    @pytest.mark.parametrize("text", [
        "يُنصح بمراجعة" + " أ" * 60 + "۔ لاحقاً",
        "تمهيد. يُنصح بمراجعة المتطلبات قبل المتابعة. خاتمة.",
        "يُنصح\n   بمراجعة المرجع الرسمي", 
        "يُنصح بمراجعة، ثم استكمال الإجراء", 
        "يُنصح بمراجعة. ثم نص إضافي طويل بعد العبارة.",
    ])
    def test_arabic_literal_phrase_is_position_and_punctuation_independent(self, text):
        assert GCC_TRIP_PHRASES[0].search(text)

    def test_unrelated_arabic_advice_is_not_a_false_positive(self):
        assert not GCC_TRIP_PHRASES[0].search("يُفضّل التحقق من اللائحة الرسمية قبل المتابعة.")

    def test_no_id_or_split_conditions_are_used(self):
        source = (TRAINING / "readiness_gate.py").read_text(encoding="utf-8")
        gate14_source = source[source.index("def _review_required_findings"):source.index("def gate_content_risk")]
        for prohibited in ("seed4b-", "allowlist"):
            assert prohibited not in gate14_source.lower()
        assert 'record.get("split"' not in gate14_source
        assert 'authority' not in gate14_source.lower()


class TestEffectiveReviewSet:
    def test_frozen_base_plus_three_supplements_equals_53(self):
        effective, policy = derive_effective_review_set(records(), DATA, gate14())
        assert policy["frozen_base_queue_count"] == 50
        assert policy["mandatory_risk_supplement_count"] == 3
        assert policy["effective_review_set_count"] == 53
        assert policy["supplemental_ids"] == SUPPLEMENT_IDS
        assert policy["review_required_count"] == 9
        assert policy["none_count"] == 44
        assert policy["clear_count"] == 0
        assert len({r["example_id"] for r in effective}) == 53

    def test_gate15_fails_on_missing_duplicate_or_unjustified_supplement(self):
        effective, _ = derive_effective_review_set(records(), DATA, gate14())
        ids = [r["example_id"] for r in effective]
        assert gate_review_queue_coverage(records(), DATA, gate14(), ids[:-1]).passed is False
        assert gate_review_queue_coverage(records(), DATA, gate14(), ids + [ids[-1]]).passed is False
        assert gate_review_queue_coverage(records(), DATA, gate14(), ids + ["seed4b-train-0000"]).passed is False

    def test_base_queue_is_byte_identical_after_derivation(self):
        before = (DATA / "human_review_queue.jsonl").read_bytes()
        derive_effective_review_set(records(), DATA, gate14())
        assert (DATA / "human_review_queue.jsonl").read_bytes() == before


class TestCrossFormatEffectivePackage:
    def test_all_representations_are_53_ids_and_9_44_0(self, tmp_path):
        docs = generate_temp_package(tmp_path)
        sidecar = json.loads((docs / "REVIEW_RISK_FINDINGS.json").read_text(encoding="utf-8"))
        jsonl_rows = package_rows(docs / "REVIEWER_PACKAGE.jsonl")
        with (docs / "REVIEWER_TEMPLATE.csv").open("r", encoding="utf-8-sig", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        markdown = (docs / "ARABIC_HUMAN_REVIEW_QUEUE.md").read_text(encoding="utf-8")
        jsonl_ids = {r["example_id"] for r in jsonl_rows}
        csv_ids = {r["example_id"] for r in csv_rows}
        markdown_ids = set(__import__("re").findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|", markdown.split("## All Queue Records", 1)[1], __import__("re").MULTILINE))
        assert len(jsonl_ids) == len(csv_ids) == len(markdown_ids) == 53
        assert jsonl_ids == csv_ids == markdown_ids
        assert sidecar["frozen_base_queue_count"] == 50
        assert sidecar["mandatory_risk_supplement_count"] == 3
        assert sidecar["effective_review_set_count"] == 53
        assert sidecar["review_required_count"] == 9
        assert sidecar["none_count"] == 44
        assert sidecar["clear_count"] == 0
        assert sidecar["supplemental_ids"] == SUPPLEMENT_IDS
        flagged = {r["example_id"] for r in jsonl_rows if r["content_risk_status"] == "REVIEW_REQUIRED"}
        assert flagged == ALL_FLAGGED
        assert sum(r["content_risk_status"] == "NONE" for r in jsonl_rows) == 44
        assert not any(r["content_risk_status"] == "CLEAR" for r in jsonl_rows)
        assert all(r["integrity_flags"] != "OK" for r in jsonl_rows if r["example_id"] in ALL_FLAGGED)
        human_fields = ["reviewer_name", "verdict", "comments", "corrected_wording", "rejection_reason"]
        for row in jsonl_rows:
            assert all(not str(row.get(field, "")).strip() for field in human_fields)

    def test_generation_order_is_deterministic(self, tmp_path):
        first = generate_temp_package(tmp_path / "a")
        second = generate_temp_package(tmp_path / "b")
        for name in ("REVIEWER_PACKAGE.jsonl", "REVIEWER_TEMPLATE.csv", "ARABIC_HUMAN_REVIEW_QUEUE.md"):
            a = (first / name).read_bytes()
            b = (second / name).read_bytes()
            assert a == b
        a = json.loads((first / "REVIEW_RISK_FINDINGS.json").read_text(encoding="utf-8"))
        b = json.loads((second / "REVIEW_RISK_FINDINGS.json").read_text(encoding="utf-8"))
        for key in ("frozen_base_queue_count", "mandatory_risk_supplement_count", "effective_review_set_count", "supplemental_ids", "review_required_ids"):
            assert a[key] == b[key]
        def normalize_findings(items):
            return [{k: v for k, v in item.items() if k != "generated_at_utc"} for item in items]
        assert normalize_findings(a["findings"]) == normalize_findings(b["findings"])
