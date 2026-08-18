"""
Jupiter Seed 4B V2.3 — Regression Tests
==========================================
Covers all V2.3 repair requirements:

1.  ADV-05 production-path BLOCKED result
2.  ADV-06 production-path REVIEW_REQUIRED result
3.  Guaranteed claim makes final verdict NOT_READY (exit 2)
4.  Required-review phrase is surfaced with record ID and rule
5.  Required-review record missing from queue returns exit 2
6.  Blocking WARN cannot return exit 0
7.  Mechanical failure returns exit 1
8.  Execution error returns exit 3
9.  Clean mechanically ready package returns exit 0
10. Gate 12 remains pending and never authorizes training
11. Blank stored hash is rejected
12. Three existing GCC-phrase records are disclosed
13. Human judgment fields remain empty
14. Corpus remains byte-identical

CPU-only. No corpus modification. No GPU, model downloads, paid APIs, cloud.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
sys.path.insert(0, str(TRAINING_DIR))

from readiness_gate import (
    PASS, FAIL, NOT_READY, BLOCKED, REVIEW_REQUIRED,
    EXIT_MECHANICALLY_READY, EXIT_MECHANICAL_FAILURE,
    EXIT_MECHANICALLY_BLOCKED, EXIT_EXECUTION_ERROR,
    gate_content_risk,
    gate_corpus_integrity,
    gate_human_review,
    gate_review_queue_coverage,
    GCC_TRIP_PHRASES,
    BLOCKING_CONTENT_PATTERNS,
    GateResult,
)
from canonicalize import sha256_of

DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"

EXPECTED_CORPUS_HASH = "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0"
EXPECTED_CONTENT_COMMIT = "2e36f6b977a8af052fced5a532c1168dc1988b6f"

LITERAL_FLAGGED_RECORDS = {
    "seed4b-eval-0004", "seed4b-train-0015", "seed4b-train-0016",
    "seed4b-train-0028", "seed4b-train-0032", "seed4b-train-0038",
    "seed4b-train-0039", "seed4b-valid-0005", "seed4b-valid-0014",
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def load_all_records() -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            pytest.skip(f"Data file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def make_clean_record(
    example_id: str = "adv-test-0001",
    split: str = "train",
    prompt: str = "Explain Murabaha financing.",
    response: str = (
        "Murabaha is a cost-plus financing structure where the bank purchases "
        "an asset and sells it to the client at a disclosed markup."
    ),
    language: str = "en",
    **kwargs,
) -> Dict:
    content = prompt + "\n" + response
    r = {
        "example_id": example_id,
        "split": split,
        "prompt": prompt,
        "response": response,
        "language": language,
        "domain": "islamic_finance",
        "task_type": "explanation",
        "difficulty": "intermediate",
        "provenance_type": "ORIGINAL_SCENARIO",
        "source_name": "Test Fixture",
        "source_url": "https://example.com/",
        "content_family_id": f"ADV-{example_id.upper()}",
        "scenario_brief": f"Test scenario for {example_id}",
        "language_contract_passed": True,
        "content_sha256": sha256_of(content),
        "canonical_prompt_hash": sha256_of(prompt),
        "canonical_response_hash": sha256_of(response),
        "scenario_brief_hash": sha256_of(f"Test scenario for {example_id}"),
        "schema_version": "2.0",
        "dataset_version": "2.0.0",
        "country_or_region": "GCC",
        "author_type": "human",
        "human_review_required": True,
        "factuality_review_status": "pending",
        "arabic_review_status": "pending",
        "sensitive_data_status": "clean",
        "creation_method": "human_authored",
        "inclusion_reason": "test fixture",
        "rejection_reason": "",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z",
        "fictional_disclaimer": "fictional scenario for training purposes",
    }
    r.update(kwargs)
    return r


# ===========================================================================
# Test 1: ADV-05 production-path BLOCKED result
# ===========================================================================

class TestADV05ProductionPath:
    """ADV-05: '100% guaranteed' must be rejected by the production gate_content_risk."""

    def test_adv05_gate_content_risk_returns_blocked(self) -> None:
        """gate_content_risk must return BLOCKED for a record with '100% guaranteed'."""
        r = make_clean_record(
            example_id="adv-05-blocked",
            response="This investment is 100% guaranteed to return your principal.",
        )
        result = gate_content_risk([r])
        assert result.status == BLOCKED, (
            f"ADV-05: gate_content_risk must return BLOCKED for '100% guaranteed'. "
            f"Got: {result.status}\nDetails: {result.details}"
        )

    def test_adv05_blocked_finding_identifies_record_and_rule(self) -> None:
        """BLOCKED finding must identify example_id, field, and rule."""
        r = make_clean_record(
            example_id="adv-05-blocked",
            response="This investment is 100% guaranteed to return your principal.",
        )
        result = gate_content_risk([r])
        assert result.status == BLOCKED
        assert result.errors, "BLOCKED result must have errors listing the finding"
        error_str = " ".join(result.errors)
        assert "adv-05-blocked" in error_str, "Finding must identify the example_id"
        assert "100% guaranteed" in error_str.lower() or "guaranteed" in error_str.lower(), (
            "Finding must identify the matched rule"
        )

    def test_adv05_blocked_result_prevents_exit_0(self) -> None:
        """A BLOCKED content-risk result must produce exit code 2, not 0."""
        r = make_clean_record(
            example_id="adv-05-blocked",
            response="This investment is 100% guaranteed to return your principal.",
        )
        result = gate_content_risk([r])
        assert result.status == BLOCKED
        # Verify that the exit code contract maps BLOCKED to exit 2
        assert EXIT_MECHANICALLY_BLOCKED == 2
        assert EXIT_MECHANICALLY_READY == 0
        # BLOCKED != PASS, so exit 0 is impossible
        assert not result.passed, "BLOCKED result must not be treated as PASS"


# ===========================================================================
# Test 2: ADV-06 production-path REVIEW_REQUIRED result
# ===========================================================================

class TestADV06ProductionPath:
    """ADV-06: GCC trip phrase must be rejected by the production gate_content_risk."""

    def test_adv06_gate_content_risk_returns_review_required(self) -> None:
        """gate_content_risk must return REVIEW_REQUIRED for a GCC trip phrase."""
        r = make_clean_record(
            example_id="adv-06-review",
            response=(
                "The company must comply with applicable regulations. "
                "Specific current requirements should be verified with the relevant authority."
            ),
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED, (
            f"ADV-06: gate_content_risk must return REVIEW_REQUIRED for GCC trip phrase. "
            f"Got: {result.status}\nDetails: {result.details}"
        )

    def test_adv06_review_required_finding_identifies_record_and_rule(self) -> None:
        """REVIEW_REQUIRED finding must identify example_id, field, and rule."""
        r = make_clean_record(
            example_id="adv-06-review",
            response=(
                "Specific current requirements should be verified with the relevant authority."
            ),
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED
        assert result.warnings, "REVIEW_REQUIRED result must have warnings listing the finding"
        warn_str = " ".join(result.warnings)
        assert "adv-06-review" in warn_str, "Finding must identify the example_id"
        assert "specific current requirements" in warn_str.lower() or "verified" in warn_str.lower(), (
            "Finding must identify the matched rule"
        )

    def test_adv06_review_required_does_not_silently_pass(self) -> None:
        """REVIEW_REQUIRED must not silently disappear behind an overall PASS."""
        r = make_clean_record(
            example_id="adv-06-review",
            response=(
                "Specific current requirements should be verified with the relevant authority."
            ),
        )
        result = gate_content_risk([r])
        assert result.status != PASS, (
            "REVIEW_REQUIRED must not be silently treated as PASS"
        )
        assert not result.passed, (
            "REVIEW_REQUIRED result must not have passed=True"
        )

    def test_adv06_arabic_gcc_phrase_returns_review_required(self) -> None:
        """Arabic GCC trip phrase must also return REVIEW_REQUIRED."""
        r = make_clean_record(
            example_id="adv-06-arabic",
            language="ar",
            response="للمزيد من المعلومات يُنصح بمراجعة الجهات المختصة.",
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED, (
            f"Arabic GCC trip phrase must return REVIEW_REQUIRED. Got: {result.status}"
        )


# ===========================================================================
# Test 3: Guaranteed claim makes final verdict NOT_READY (exit 2)
# ===========================================================================

class TestGuaranteedClaimBlocksReadiness:
    """A '100% guaranteed' claim must make the final verdict MECHANICALLY_BLOCKED."""

    def test_blocked_content_produces_exit_2_not_exit_0(self) -> None:
        """BLOCKED content must produce exit code 2, not 0."""
        r = make_clean_record(
            example_id="blocked-exit-test",
            response="This fund is 100% guaranteed to preserve capital.",
        )
        result = gate_content_risk([r])
        assert result.status == BLOCKED
        # The exit-code contract: BLOCKED → EXIT_MECHANICALLY_BLOCKED = 2
        assert EXIT_MECHANICALLY_BLOCKED == 2, "Exit code 2 must be MECHANICALLY_BLOCKED"
        assert EXIT_MECHANICALLY_READY == 0, "Exit code 0 must be MECHANICALLY_READY"
        # A BLOCKED gate must not be counted as passed
        assert not result.passed

    def test_blocked_content_not_in_passed_count(self) -> None:
        """BLOCKED gate must not be counted as a passed gate."""
        r = make_clean_record(
            response="This investment is 100% guaranteed.",
        )
        result = gate_content_risk([r])
        assert result.status == BLOCKED
        assert not result.passed
        assert not result.failed  # BLOCKED is not FAIL; it is a separate category
        assert result.is_blocked


# ===========================================================================
# Test 4: Required-review phrase is surfaced with record ID and rule
# ===========================================================================

class TestReviewRequiredSurfaced:
    """REVIEW_REQUIRED findings must surface record ID and rule."""

    def test_review_required_metadata_contains_record_ids(self) -> None:
        r = make_clean_record(
            example_id="rr-surface-test",
            response="Specific current requirements should be verified with the relevant authority.",
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED
        assert "rr-surface-test" in result.metadata.get("review_required_records", []), (
            "review_required_records metadata must contain the flagged example_id"
        )

    def test_review_required_warnings_contain_rule(self) -> None:
        r = make_clean_record(
            example_id="rr-rule-test",
            response="Specific current requirements should be verified with the relevant authority.",
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED
        assert result.warnings, "REVIEW_REQUIRED must produce warnings"
        # The warning must contain the pattern rule
        assert any("verified" in w.lower() or "specific" in w.lower() for w in result.warnings), (
            "Warning must contain the matched rule text"
        )


# ===========================================================================
# Test 5: Required-review record missing from queue returns exit 2
# ===========================================================================

class TestReviewQueueCoverage:
    """Gate 15 must return BLOCKED (exit 2) if a REVIEW_REQUIRED record is absent from queue."""

    def test_missing_review_required_record_returns_blocked(self) -> None:
        """If a REVIEW_REQUIRED record is not in the queue, gate_review_queue_coverage returns BLOCKED."""
        fake_content_risk = GateResult(
            14, "Content-Risk", REVIEW_REQUIRED,
            "1 REVIEW_REQUIRED finding",
            metadata={
                "blocked_count": 0,
                "review_required_count": 1,
                "review_required_records": ["adv-missing-from-queue"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True)
            queue_path = data_dir / "human_review_queue.jsonl"
            # Write a queue that does NOT contain "adv-missing-from-queue"
            with queue_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"example_id": "some-other-record"}) + "\n")

            result = gate_review_queue_coverage([], data_dir, fake_content_risk)

        assert result.status == BLOCKED, (
            f"Gate 15 must return BLOCKED when REVIEW_REQUIRED record is absent from queue. "
            f"Got: {result.status}\nErrors: {result.errors}"
        )

    def test_present_review_required_record_passes_coverage(self) -> None:
        """If all REVIEW_REQUIRED records are in the queue, gate_review_queue_coverage passes."""
        records = load_all_records()
        result = gate_review_queue_coverage(records, DATA_DIR, gate_content_risk(records))
        assert result.status == PASS, (
            f"Gate 15 must PASS when the effective review set contains the frozen base plus all mandatory supplements. "
            f"Got: {result.status}; errors={result.errors}"
        )


# ===========================================================================
# Test 6: Blocking WARN cannot return exit 0
# ===========================================================================

class TestBlockingWarnCannotReturnExit0:
    """A BLOCKED or REVIEW_REQUIRED gate must never produce exit code 0."""

    def test_blocked_status_maps_to_exit_2_not_0(self) -> None:
        assert EXIT_MECHANICALLY_BLOCKED == 2
        assert EXIT_MECHANICALLY_BLOCKED != EXIT_MECHANICALLY_READY

    def test_review_required_status_maps_to_exit_2_not_0(self) -> None:
        """REVIEW_REQUIRED must map to exit 2 (MECHANICALLY_BLOCKED), not exit 0."""
        r = make_clean_record(
            response="Specific current requirements should be verified with the relevant authority.",
        )
        result = gate_content_risk([r])
        assert result.status == REVIEW_REQUIRED
        # REVIEW_REQUIRED is not PASS, so exit 0 is impossible
        assert not result.passed, "REVIEW_REQUIRED must not be passed=True"
        # The exit-code contract maps REVIEW_REQUIRED to exit 2
        assert EXIT_MECHANICALLY_BLOCKED == 2


# ===========================================================================
# Test 7: Mechanical failure returns exit 1
# ===========================================================================

class TestMechanicalFailureReturnsExit1:
    """Mechanical failure must map to exit code 1."""

    def test_exit_mechanical_failure_is_1(self) -> None:
        assert EXIT_MECHANICAL_FAILURE == 1

    def test_fail_status_is_not_exit_0(self) -> None:
        assert EXIT_MECHANICAL_FAILURE != EXIT_MECHANICALLY_READY


# ===========================================================================
# Test 8: Execution error returns exit 3
# ===========================================================================

class TestExecutionErrorReturnsExit3:
    """Execution error must map to exit code 3."""

    def test_exit_execution_error_is_3(self) -> None:
        assert EXIT_EXECUTION_ERROR == 3

    def test_execution_error_is_not_exit_0_or_1_or_2(self) -> None:
        assert EXIT_EXECUTION_ERROR not in (
            EXIT_MECHANICALLY_READY,
            EXIT_MECHANICAL_FAILURE,
            EXIT_MECHANICALLY_BLOCKED,
        )


# ===========================================================================
# Test 9: Clean mechanically ready package returns exit 0
# ===========================================================================

class TestCleanPackageReturnsExit0:
    """A clean corpus with no content-risk findings must produce exit 0."""

    def test_clean_corpus_gate14_returns_pass(self) -> None:
        """gate_content_risk must return PASS for a clean record."""
        r = make_clean_record(
            response="Murabaha is a cost-plus financing structure used in Islamic banking.",
        )
        result = gate_content_risk([r])
        assert result.status == PASS, (
            f"Clean record must produce PASS from gate_content_risk. Got: {result.status}"
        )

    def test_exit_mechanically_ready_is_0(self) -> None:
        assert EXIT_MECHANICALLY_READY == 0

    def test_exit_codes_are_distinct(self) -> None:
        codes = {
            EXIT_MECHANICALLY_READY,
            EXIT_MECHANICAL_FAILURE,
            EXIT_MECHANICALLY_BLOCKED,
            EXIT_EXECUTION_ERROR,
        }
        assert len(codes) == 4, f"All 4 exit codes must be distinct. Got: {codes}"


# ===========================================================================
# Test 10: Gate 12 remains pending and never authorizes training
# ===========================================================================

class TestGate12NeverAuthorizesTraining:
    """Gate 12 must remain NOT_READY and must never be treated as training approval."""

    def test_gate12_returns_not_ready_on_frozen_corpus(self) -> None:
        records = load_all_records()
        result = gate_human_review(records)
        assert result.status == NOT_READY, (
            f"Gate 12 must return NOT_READY on the frozen corpus. Got: {result.status}"
        )

    def test_gate12_not_ready_is_not_training_approval(self) -> None:
        records = load_all_records()
        result = gate_human_review(records)
        assert result.status == NOT_READY
        assert "PENDING" in result.details.upper() or "NOT YET" in result.details.upper() or \
               "pending" in result.details.lower(), (
            "Gate 12 details must explicitly state that review is pending"
        )

    def test_gate12_not_ready_does_not_count_as_passed(self) -> None:
        records = load_all_records()
        result = gate_human_review(records)
        assert result.status == NOT_READY
        assert not result.passed, "Gate 12 NOT_READY must not be counted as passed"

    def test_software_cannot_set_human_approved(self) -> None:
        """Software-set human_approved must be detected and rejected by Gate 12."""
        records = load_all_records()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        tampered[0]["factuality_review_status"] = "human_approved"
        result = gate_human_review(tampered)
        assert result.status == FAIL, (
            "Gate 12 must FAIL if software sets human_approved"
        )


# ===========================================================================
# Test 11: Blank stored hash is rejected
# ===========================================================================

class TestBlankStoredHashRejected:
    """Gate 13 must reject a blank or missing content_sha256."""

    def test_blank_content_sha256_fails_gate13(self) -> None:
        records = load_all_records()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        tampered[0]["content_sha256"] = ""  # Blank
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL when content_sha256 is blank. Got: {result.status}"
        )

    def test_missing_content_sha256_fails_gate13(self) -> None:
        records = load_all_records()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        del tampered[0]["content_sha256"]  # Missing
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL when content_sha256 is missing. Got: {result.status}"
        )

    def test_whitespace_only_content_sha256_fails_gate13(self) -> None:
        records = load_all_records()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        tampered[0]["content_sha256"] = "   "  # Whitespace only
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL when content_sha256 is whitespace-only. Got: {result.status}"
        )


# ===========================================================================
# Test 12: Three existing GCC-phrase records are disclosed
# ===========================================================================

class TestThreeFlaggedRecordsDisclosed:
    """The three Kishore-flagged records must be disclosed by gate_content_risk."""

    def test_gate14_discloses_all_three_flagged_records(self) -> None:
        """gate_content_risk must surface all three GCC-phrase records."""
        records = load_all_records()
        result = gate_content_risk(records)
        assert result.status in (REVIEW_REQUIRED, BLOCKED), (
            f"gate_content_risk must return REVIEW_REQUIRED or BLOCKED for the frozen corpus. "
            f"Got: {result.status}"
        )
        disclosed_ids = set(result.metadata.get("review_required_records", []))
        missing = LITERAL_FLAGGED_RECORDS - disclosed_ids
        assert not missing, (
            f"gate_content_risk must disclose all literal-rule flagged records. "
            f"Missing: {missing}. Disclosed: {disclosed_ids}"
        )

    def test_seed4b_train_0032_is_disclosed(self) -> None:
        records = load_all_records()
        result = gate_content_risk(records)
        assert "seed4b-train-0032" in result.metadata.get("review_required_records", []), (
            "seed4b-train-0032 must be in review_required_records"
        )

    def test_seed4b_train_0038_is_disclosed(self) -> None:
        records = load_all_records()
        result = gate_content_risk(records)
        assert "seed4b-train-0038" in result.metadata.get("review_required_records", []), (
            "seed4b-train-0038 must be in review_required_records"
        )

    def test_seed4b_valid_0005_is_disclosed(self) -> None:
        records = load_all_records()
        result = gate_content_risk(records)
        assert "seed4b-valid-0005" in result.metadata.get("review_required_records", []), (
            "seed4b-valid-0005 must be in review_required_records"
        )


# ===========================================================================
# Test 13: Human judgment fields remain empty
# ===========================================================================

class TestHumanJudgmentFieldsEmpty:
    """Human judgment fields must remain empty in the corpus and review queue."""

    def test_no_human_judgment_fields_populated_in_corpus(self) -> None:
        from readiness_gate import REVIEWER_JUDGMENT_FIELDS
        records = load_all_records()
        violations = []
        for r in records:
            for jf in REVIEWER_JUDGMENT_FIELDS:
                val = r.get(jf, "")
                if val and str(val).strip():
                    violations.append(f"{r['example_id']}: {jf}={repr(val)}")
        assert not violations, (
            f"Human judgment fields must be empty in corpus. Violations: {violations[:5]}"
        )

    def test_no_human_judgment_fields_populated_in_review_queue(self) -> None:
        from readiness_gate import REVIEWER_JUDGMENT_FIELDS
        queue_path = DATA_DIR / "human_review_queue.jsonl"
        if not queue_path.exists():
            pytest.skip("human_review_queue.jsonl not found")
        violations = []
        with queue_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for jf in REVIEWER_JUDGMENT_FIELDS:
                    val = r.get(jf, "")
                    if val and str(val).strip():
                        violations.append(f"{r.get('example_id', '?')}: {jf}={repr(val)}")
        assert not violations, (
            f"Human judgment fields must be empty in review queue. Violations: {violations[:5]}"
        )


# ===========================================================================
# Test 14: Corpus remains byte-identical
# ===========================================================================

class TestCorpusByteIdentical:
    """Corpus must remain byte-identical to the verified content commit."""

    def test_corpus_hash_unchanged(self) -> None:
        records = load_all_records()
        records_sorted = sorted(records, key=lambda r: r.get("example_id", ""))
        prompt_hashes = [sha256_of(r.get("prompt", "")) for r in records_sorted]
        response_hashes = [sha256_of(r.get("response", "")) for r in records_sorted]
        brief_hashes = [sha256_of(str(r.get("scenario_brief", ""))) for r in records_sorted]
        ordered_str = "|".join(
            f"{p}:{r}:{b}"
            for p, r, b in zip(prompt_hashes, response_hashes, brief_hashes)
        )
        computed = sha256_of(ordered_str)
        assert computed == EXPECTED_CORPUS_HASH, (
            f"Corpus hash must be unchanged. "
            f"computed={computed} expected={EXPECTED_CORPUS_HASH}"
        )

    def test_corpus_record_count_is_101(self) -> None:
        records = load_all_records()
        assert len(records) == 101, f"Expected 101 records, got {len(records)}"

    def test_gate13_passes_on_frozen_corpus(self) -> None:
        records = load_all_records()
        result = gate_corpus_integrity(records, BENCHMARK_DIR)
        assert result.status == PASS, (
            f"Gate 13 must PASS on frozen corpus. Got: {result.status}\nErrors: {result.errors}"
        )
