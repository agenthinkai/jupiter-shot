"""
Jupiter Seed 4B V2.2 — Regression Tests
=========================================
Covers all V2.2 repair requirements:

Task 2: Gate 11 exit-code enforcement regression tests
Task 3: Authorized-test-manifest enforcement
Task 5: Gate 13 corpus-integrity gate
Task 6: All 12 adversarial cases rejected at gate level

CPU-only. No GPU, model downloads, paid APIs, or cloud resources.
Corpus is FROZEN — these tests use synthetic injection records only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
sys.path.insert(0, str(TRAINING_DIR))

from canonicalize import sha256_of, canonicalize, canonical_hash, has_artificial_marker
from readiness_gate import (
    AUTHORIZED_TEST_FILES,
    AUTHORIZED_MANIFEST_HASH,
    PASS,
    FAIL,
    NOT_READY,
    _check_test_manifest,
    gate_language_contract,
    gate_canonical_duplication,
    gate_family_isolation,
    gate_provenance,
    gate_raw_duplication,
    gate_corpus_integrity,
    recompute_language_contract,
)

DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"

EXPECTED_CORPUS_HASH = "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0"
EXPECTED_CONTENT_COMMIT = "2e36f6b977a8af052fced5a532c1168dc1988b6f"


# ─── Helper ───────────────────────────────────────────────────────────────

def make_record(
    example_id: str = "adv-train-0001",
    split: str = "train",
    prompt: str = "Explain Murabaha financing.",
    response: str = (
        "Murabaha is a cost-plus financing structure where the bank purchases "
        "an asset and sells it to the client at a disclosed markup, payable in instalments."
    ),
    language: str = "en",
    domain: str = "islamic_finance",
    provenance_type: str = "ORIGINAL_SCENARIO",
    content_family_id: str = "ADV-TRAIN-001",
    scenario_brief: str = "Explain Murabaha financing mechanism in Islamic banking context.",
    language_contract_passed: bool = True,
    source_url: str = "https://aaoifi.com/",
    source_name: str = "AAOIFI Standard",
    **kwargs,
) -> Dict:
    content = prompt + response
    record = {
        "example_id": example_id,
        "split": split,
        "prompt": prompt,
        "response": response,
        "language": language,
        "domain": domain,
        "task_type": "explanation",
        "difficulty": "intermediate",
        "provenance_type": provenance_type,
        "source_name": source_name,
        "source_url": source_url,
        "content_family_id": content_family_id,
        "scenario_brief": scenario_brief,
        "language_contract_passed": language_contract_passed,
        "content_sha256": sha256_of(content),
        "canonical_prompt_hash": canonical_hash(prompt),
        "canonical_response_hash": canonical_hash(response),
        "scenario_brief_hash": sha256_of(canonicalize(scenario_brief)),
        "schema_version": "2.0",
        "dataset_version": "2.0.0",
        "country_or_region": "GCC",
        "author_type": "human",
        "human_review_required": True,
        "factuality_review_status": "pending",
        "arabic_review_status": "pending",
        "sensitive_data_status": "clean",
        "creation_method": "human_authored",
        "inclusion_reason": "regression test fixture",
        "rejection_reason": "",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-01T00:00:00Z",
        "fictional_disclaimer": "fictional scenario for training purposes",
    }
    record.update(kwargs)
    return record


# ===========================================================================
# Task 2: Gate 11 exit-code enforcement regression tests
# ===========================================================================

class TestGate11ExitCodeEnforcement:
    """Regression tests proving Gate 11 fails on each required failure condition."""

    def _run_gate11_with_xml(self, xml_content: str, exit_code: int):
        """Helper: run gate_test_suite with a controlled XML result and exit code."""
        from readiness_gate import gate_test_suite
        import xml.etree.ElementTree as ET

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "results.xml"
            if xml_content is not None:
                xml_path.write_text(xml_content, encoding="utf-8")

            # Patch subprocess.run to return our controlled result
            # Patch the TemporaryDirectory to use our tmpdir
            class FakeTmpDir:
                def __enter__(self):
                    return tmpdir
                def __exit__(self, *args):
                    return False

            with patch("readiness_gate.subprocess.run") as mock_run, \
                 patch("readiness_gate.tempfile.TemporaryDirectory", return_value=FakeTmpDir()), \
                 patch("readiness_gate._check_test_manifest", return_value=[]), \
                 patch("readiness_gate._compute_test_manifest_hash",
                       return_value=("abc", AUTHORIZED_TEST_FILES)):

                mock_run.return_value = MagicMock(
                    returncode=exit_code,
                    stdout="test output",
                    stderr="",
                )
                result = gate_test_suite(REPO_ROOT)

        return result

    def test_gate11_fails_on_zero_collected_tests(self) -> None:
        """Gate 11 must FAIL when zero tests are collected."""
        xml_content = '<?xml version="1.0"?><testsuite tests="0" failures="0" errors="0" skipped="0"/>'
        result = self._run_gate11_with_xml(xml_content, exit_code=5)
        assert result.status == FAIL, f"Gate 11 must FAIL on zero collected. Got: {result.status}"

    def test_gate11_fails_on_nonzero_exit_code(self) -> None:
        """Gate 11 must FAIL when pytest exits nonzero even if XML shows passed tests."""
        xml_content = '<?xml version="1.0"?><testsuite tests="10" failures="0" errors="0" skipped="0"/>'
        result = self._run_gate11_with_xml(xml_content, exit_code=1)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL on nonzero exit code. Got: {result.status}"
        )

    def test_gate11_fails_on_missing_xml(self) -> None:
        """Gate 11 must FAIL when JUnit XML is missing."""
        result = self._run_gate11_with_xml(xml_content=None, exit_code=0)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL on missing JUnit XML. Got: {result.status}"
        )

    def test_gate11_fails_on_malformed_xml(self) -> None:
        """Gate 11 must FAIL when JUnit XML is malformed."""
        result = self._run_gate11_with_xml("NOT VALID XML <<<>>>", exit_code=0)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL on malformed JUnit XML. Got: {result.status}"
        )

    def test_gate11_fails_on_failed_tests(self) -> None:
        """Gate 11 must FAIL when any test failed."""
        xml_content = '<?xml version="1.0"?><testsuite tests="10" failures="2" errors="0" skipped="0"/>'
        result = self._run_gate11_with_xml(xml_content, exit_code=1)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL when tests failed. Got: {result.status}"
        )

    def test_gate11_fails_on_error_tests(self) -> None:
        """Gate 11 must FAIL when any test errored."""
        xml_content = '<?xml version="1.0"?><testsuite tests="10" failures="0" errors="1" skipped="0"/>'
        result = self._run_gate11_with_xml(xml_content, exit_code=1)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL when tests errored. Got: {result.status}"
        )

    def test_gate11_recursion_guard(self) -> None:
        """Gate 11 must FAIL when called from inside pytest (recursion guard)."""
        from readiness_gate import gate_test_suite
        import os

        # PYTEST_CURRENT_TEST is already set since we're running inside pytest
        assert os.environ.get("PYTEST_CURRENT_TEST"), (
            "This test must run inside pytest to verify the recursion guard"
        )
        result = gate_test_suite(REPO_ROOT)
        assert result.status == FAIL, (
            f"Gate 11 must FAIL inside pytest (recursion guard). Got: {result.status}"
        )


# ===========================================================================
# Task 3: Authorized-test-manifest enforcement
# ===========================================================================

class TestAuthorizedTestManifest:
    """Tests for the authorized-test-manifest enforcement in Gate 11."""

    def test_authorized_test_files_has_16_entries(self) -> None:
        """AUTHORIZED_TEST_FILES must contain exactly 16 entries (V2.4.6 adds literal Gate 14 coverage)."""
        assert len(AUTHORIZED_TEST_FILES) == 16, (
            f"Expected 16 authorized test files, got {len(AUTHORIZED_TEST_FILES)}: "
            f"{AUTHORIZED_TEST_FILES}"
        )

    def test_test_gate_adversarial_in_authorized_files(self) -> None:
        """test_gate_adversarial.py must be in AUTHORIZED_TEST_FILES."""
        assert "test_gate_adversarial.py" in AUTHORIZED_TEST_FILES, (
            "test_gate_adversarial.py must be in AUTHORIZED_TEST_FILES"
        )

    def test_v241_identity_suite_in_authorized_files(self) -> None:
        """The V2.4.1 fail-closed identity suite must be enforced by Gate 11."""
        assert "test_v241_package_identity.py" in AUTHORIZED_TEST_FILES, (
            "test_v241_package_identity.py must be in AUTHORIZED_TEST_FILES"
        )

    def test_v242_execution_suite_in_authorized_files(self) -> None:
        """The V2.4.2 runtime-isolation suite must be enforced by Gate 11."""
        assert "test_v242_execution_isolation.py" in AUTHORIZED_TEST_FILES, (
            "test_v242_execution_isolation.py must be in AUTHORIZED_TEST_FILES"
        )

    def test_v243_portability_and_interlock_suite_in_authorized_files(self) -> None:
        """The V2.4.3 portability/interlock suite must be enforced by Gate 11."""
        assert "test_v243_portability_and_interlock.py" in AUTHORIZED_TEST_FILES

    def test_v244_windows_acl_and_cache_suite_in_authorized_files(self) -> None:
        """The V2.4.4 ACL/cache suite must be enforced by Gate 11."""
        assert "test_v244_windows_acl_and_cache.py" in AUTHORIZED_TEST_FILES

    def test_v246_literal_gate14_suite_in_authorized_files(self) -> None:
        """The V2.4.6 literal Gate 14 suite must be enforced by Gate 11."""
        assert "test_v246_literal_gate14.py" in AUTHORIZED_TEST_FILES

    def test_no_duplicate_entries_in_authorized_files(self) -> None:
        """AUTHORIZED_TEST_FILES must not contain duplicate entries."""
        assert len(AUTHORIZED_TEST_FILES) == len(set(AUTHORIZED_TEST_FILES)), (
            f"Duplicate entries in AUTHORIZED_TEST_FILES: {AUTHORIZED_TEST_FILES}"
        )

    def test_authorized_manifest_hash_matches_file_list(self) -> None:
        """AUTHORIZED_MANIFEST_HASH must match SHA-256 of sorted AUTHORIZED_TEST_FILES."""
        expected = hashlib.sha256(
            "|".join(sorted(AUTHORIZED_TEST_FILES)).encode("utf-8")
        ).hexdigest()
        assert AUTHORIZED_MANIFEST_HASH == expected, (
            f"AUTHORIZED_MANIFEST_HASH mismatch: "
            f"computed={expected} stored={AUTHORIZED_MANIFEST_HASH}"
        )

    def test_all_authorized_files_exist_in_test_dir(self) -> None:
        """All 7 authorized test files must exist in tests/jupiter_seed_4b/."""
        test_dir = REPO_ROOT / "tests" / "jupiter_seed_4b"
        missing = [f for f in AUTHORIZED_TEST_FILES if not (test_dir / f).exists()]
        assert not missing, f"Missing authorized test files: {missing}"

    def test_no_unexpected_test_files_in_directory(self) -> None:
        """No unexpected test files must exist in tests/jupiter_seed_4b/."""
        test_dir = REPO_ROOT / "tests" / "jupiter_seed_4b"
        authorized_set = set(AUTHORIZED_TEST_FILES)
        actual_files = {p.name for p in test_dir.glob("test_*.py")}
        unexpected = sorted(actual_files - authorized_set)
        assert not unexpected, (
            f"Unexpected test files not in authorized manifest: {unexpected}"
        )

    def test_check_test_manifest_passes_on_current_dir(self) -> None:
        """_check_test_manifest must return no errors on the current test directory."""
        test_dir = REPO_ROOT / "tests" / "jupiter_seed_4b"
        errors = _check_test_manifest(test_dir)
        assert not errors, f"_check_test_manifest returned errors: {errors}"

    def test_check_test_manifest_fails_on_missing_file(self) -> None:
        """_check_test_manifest must fail when an authorized file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            # Create only 6 of the 7 authorized files
            for fname in AUTHORIZED_TEST_FILES[:-1]:
                (test_dir / fname).write_text("# placeholder", encoding="utf-8")
            errors = _check_test_manifest(test_dir)
        assert errors, "_check_test_manifest must return errors when a file is missing"
        assert any("Missing" in e or "missing" in e for e in errors), (
            f"Error must mention missing file. Errors: {errors}"
        )

    def test_check_test_manifest_fails_on_unexpected_file(self) -> None:
        """_check_test_manifest must fail when an unexpected test file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            # Create all authorized files
            for fname in AUTHORIZED_TEST_FILES:
                (test_dir / fname).write_text("# placeholder", encoding="utf-8")
            # Add an unexpected file
            (test_dir / "test_unauthorized_new.py").write_text("# unauthorized", encoding="utf-8")
            errors = _check_test_manifest(test_dir)
        assert errors, "_check_test_manifest must return errors for unexpected file"
        assert any("nexpected" in e for e in errors), (
            f"Error must mention unexpected file. Errors: {errors}"
        )


# ===========================================================================
# Task 5: Gate 13 corpus-integrity gate
# ===========================================================================

class TestGate13CorpusIntegrity:
    """Tests for the corpus-integrity gate."""

    def _load_all(self) -> List[Dict]:
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

    def test_corpus_integrity_passes_on_frozen_corpus(self) -> None:
        """Gate 13 must PASS on the frozen 101-record corpus."""
        records = self._load_all()
        result = gate_corpus_integrity(records, BENCHMARK_DIR)
        assert result.status == PASS, (
            f"Gate 13 must PASS on frozen corpus. Got: {result.status}\n"
            f"Errors: {result.errors}"
        )

    def test_corpus_hash_matches_expected(self) -> None:
        """Ordered corpus SHA-256 must match the verified expected hash."""
        records = self._load_all()
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
            f"Corpus hash mismatch: computed={computed} expected={EXPECTED_CORPUS_HASH}"
        )

    def test_corpus_record_count_is_101(self) -> None:
        """Corpus must contain exactly 101 records."""
        records = self._load_all()
        assert len(records) == 101, f"Expected 101 records, got {len(records)}"

    def test_corpus_split_counts(self) -> None:
        """Corpus must have train=68, valid=16, eval=17."""
        records = self._load_all()
        splits = {}
        for r in records:
            s = r.get("split", "unknown")
            splits[s] = splits.get(s, 0) + 1
        assert splits.get("train") == 68, f"Expected train=68, got {splits.get('train')}"
        assert splits.get("valid") == 16, f"Expected valid=16, got {splits.get('valid')}"
        assert splits.get("eval") == 17, f"Expected eval=17, got {splits.get('eval')}"

    def test_gate13_fails_on_modified_record(self) -> None:
        """Gate 13 must FAIL when a record's response is modified."""
        records = self._load_all()
        # Inject a tampered record
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        tampered[0]["response"] = "TAMPERED RESPONSE — this should fail corpus integrity"
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL when a record is modified. Got: {result.status}"
        )

    def test_gate13_fails_on_wrong_record_count(self) -> None:
        """Gate 13 must FAIL when record count is wrong."""
        records = self._load_all()
        # Remove one record
        truncated = records[:-1]
        result = gate_corpus_integrity(truncated, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL on wrong record count. Got: {result.status}"
        )

    def test_gate13_fails_on_wrong_split_assignment(self) -> None:
        """Gate 13 must FAIL when a record's split is changed."""
        records = self._load_all()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        original_split = tampered[0]["split"]
        tampered[0]["split"] = "eval" if original_split == "train" else "train"
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL when split assignment is changed. Got: {result.status}"
        )

    def test_gate13_fails_on_stored_hash_mismatch(self) -> None:
        """Gate 13 must FAIL when stored content_sha256 disagrees with recomputed."""
        records = self._load_all()
        tampered = list(records)
        tampered[0] = dict(tampered[0])
        # Use a hash that is definitely wrong (not the sha256 of prompt+newline+response)
        tampered[0]["content_sha256"] = "a" * 64  # Wrong hash
        result = gate_corpus_integrity(tampered, BENCHMARK_DIR)
        assert result.status == FAIL, (
            f"Gate 13 must FAIL on stored hash mismatch. Got: {result.status}"
        )

    def test_frozen_corpus_manifest_exists(self) -> None:
        """FROZEN_CORPUS_MANIFEST.json must exist."""
        assert (BENCHMARK_DIR / "FROZEN_CORPUS_MANIFEST.json").exists(), (
            "FROZEN_CORPUS_MANIFEST.json not found"
        )

    def test_frozen_corpus_manifest_hash_matches(self) -> None:
        """FROZEN_CORPUS_MANIFEST.json must record the correct corpus hash."""
        manifest_path = BENCHMARK_DIR / "FROZEN_CORPUS_MANIFEST.json"
        if not manifest_path.exists():
            pytest.skip("FROZEN_CORPUS_MANIFEST.json not found")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest["ordered_corpus_sha256"] == EXPECTED_CORPUS_HASH, (
            f"Manifest corpus hash mismatch: "
            f"manifest={manifest['ordered_corpus_sha256']} "
            f"expected={EXPECTED_CORPUS_HASH}"
        )
        assert manifest["dataset_content_commit"] == EXPECTED_CONTENT_COMMIT, (
            f"Manifest content commit mismatch: "
            f"manifest={manifest['dataset_content_commit']} "
            f"expected={EXPECTED_CONTENT_COMMIT}"
        )


# ===========================================================================
# Task 6: All 12 adversarial cases rejected at gate level
# ===========================================================================

class TestAll12AdversarialCasesRejected:
    """
    Proves all 12 adversarial injection cases are rejected by the applicable gate.
    These are gate-level tests (not just unit tests of the canonicalize engine).
    """

    def test_adv_01_english_response_labelled_arabic_rejected_by_gate4(self) -> None:
        r = make_record(
            language="ar",
            prompt="اشرح مفهوم تمويل المرابحة.",
            response="Murabaha is a cost-plus financing structure.",
            language_contract_passed=True,
        )
        result = gate_language_contract([r])
        assert result.status == FAIL, f"ADV-01 must be rejected by Gate 4. Got: {result.status}"

    def test_adv_02_artificial_marker_rejected_by_gate6(self) -> None:
        r = make_record(response="Murabaha is a cost-plus structure. [train-042]")
        assert has_artificial_marker(r["response"])
        result = gate_canonical_duplication([r])
        assert result.status == FAIL, f"ADV-02 must be rejected by Gate 6. Got: {result.status}"

    def test_adv_03_short_response_rejected_by_gate4(self) -> None:
        r = make_record(language="en", response="Yes.", language_contract_passed=True)
        result = gate_language_contract([r])
        assert result.status == FAIL, f"ADV-03 must be rejected by Gate 4. Got: {result.status}"

    def test_adv_04_placeholder_rejected_by_gate4(self) -> None:
        r = make_record(language="en", response="[placeholder]", language_contract_passed=True)
        result = gate_language_contract([r])
        assert result.status == FAIL, f"ADV-04 must be rejected by Gate 4. Got: {result.status}"

    def test_adv_05_100_percent_guaranteed_in_blocking_patterns(self) -> None:
        from readiness_gate import BLOCKING_CONTENT_PATTERNS
        import re
        phrase = "This investment is 100% guaranteed to return your principal."
        matched = any(p.search(phrase) for p, _ in BLOCKING_CONTENT_PATTERNS)
        assert matched, "ADV-05: '100% guaranteed' must be in BLOCKING_CONTENT_PATTERNS"

    def test_adv_06_gcc_trip_phrase_in_patterns(self) -> None:
        from readiness_gate import GCC_TRIP_PHRASES
        phrase = "specific current requirements should be verified with the relevant authority."
        matched = any(p.search(phrase) for p in GCC_TRIP_PHRASES)
        assert matched, "ADV-06: GCC trip phrase must be in GCC_TRIP_PHRASES"

    def test_adv_07_missing_provenance_rejected_by_gate2(self) -> None:
        r = make_record()
        del r["provenance_type"]
        result = gate_provenance([r])
        assert result.status == FAIL, f"ADV-07 must be rejected by Gate 2. Got: {result.status}"

    def test_adv_08_duplicate_scenario_briefs_rejected_by_gate6(self) -> None:
        brief = "Explain Murabaha financing mechanism in Islamic banking context."
        r1 = make_record(example_id="adv-t-0008a", content_family_id="FAM-008A", scenario_brief=brief)
        r2 = make_record(
            example_id="adv-t-0008b",
            content_family_id="FAM-008B",
            scenario_brief=brief,
            response="A completely different response.",
        )
        r2["scenario_brief_hash"] = sha256_of(canonicalize(brief))
        result = gate_canonical_duplication([r1, r2])
        assert result.status == FAIL, f"ADV-08 must be rejected by Gate 6. Got: {result.status}"

    def test_adv_09_canonical_cross_split_duplicate_rejected_by_gate6(self) -> None:
        r1 = make_record(
            example_id="adv-t-0009a", split="train",
            content_family_id="FAM-009A",
            response="Murabaha is a cost-plus financing structure used in Islamic banking.",
        )
        r2 = make_record(
            example_id="adv-e-0009b", split="eval",
            content_family_id="FAM-009B",
            response="Murabaha  is  a  cost-plus  financing  structure  used  in  Islamic  banking.",
            scenario_brief="A different scenario brief.",
        )
        r2["scenario_brief_hash"] = sha256_of(canonicalize(r2["scenario_brief"]))
        result = gate_canonical_duplication([r1, r2])
        assert result.status == FAIL, f"ADV-09 must be rejected by Gate 6. Got: {result.status}"

    def test_adv_10_cross_split_family_leakage_rejected_by_gate8(self) -> None:
        r1 = make_record(example_id="adv-t-0010a", split="train", content_family_id="SHARED-FAM")
        r2 = make_record(
            example_id="adv-e-0010b", split="eval",
            content_family_id="SHARED-FAM",
            response="A different response.",
            scenario_brief="A different scenario brief.",
        )
        result = gate_family_isolation([r1, r2])
        assert result.status == FAIL, f"ADV-10 must be rejected by Gate 8. Got: {result.status}"

    def test_adv_11_source_dependent_without_factual_dependency_flagged(self) -> None:
        r = make_record(provenance_type="SOURCE_DEPENDENT_FACTUAL")
        r.pop("factual_dependency", None)
        result = gate_provenance([r])
        assert result.status in (FAIL, "WARN"), (
            f"ADV-11 must be flagged by Gate 2. Got: {result.status}"
        )

    def test_adv_12_stored_true_recomputed_false_rejected_by_gate4(self) -> None:
        r = make_record(
            language="ar",
            prompt="اشرح مفهوم تمويل المرابحة.",
            response="This is an English-only response that violates the Arabic contract.",
            language_contract_passed=True,  # Stored True — must be caught
        )
        result = gate_language_contract([r])
        assert result.status == FAIL, (
            f"ADV-12 must be rejected by Gate 4 (stored=True, recomputed=False). "
            f"Got: {result.status}\nErrors: {result.errors}"
        )


# ===========================================================================
# Historical manifest verification
# ===========================================================================

class TestHistoricalManifests:
    """Verify all historical benchmark manifests are correct."""

    def test_v100_manifest_is_genuine(self) -> None:
        path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.0.0.json"
        assert path.exists(), "v1.0.0 manifest not found"
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d.get("benchmark_version") == "1.0.0", (
            f"v1.0.0 manifest has wrong version: {d.get('benchmark_version')}"
        )

    def test_v110_manifest_is_genuine(self) -> None:
        path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.1.0.json"
        assert path.exists(), "v1.1.0 manifest not found"
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d.get("benchmark_version") == "1.1.0", (
            f"v1.1.0 manifest has wrong version: {d.get('benchmark_version')}"
        )

    def test_v111_manifest_is_genuine_250_records(self) -> None:
        """v1.1.1 manifest must be the genuine historical 250-record manifest."""
        path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.1.1.json"
        assert path.exists(), "v1.1.1 manifest not found"
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d.get("benchmark_version") == "1.1.1", (
            f"v1.1.1 manifest has wrong version: {d.get('benchmark_version')}"
        )
        assert d.get("record_count") == 250, (
            f"v1.1.1 manifest must have 250 records (genuine historical), "
            f"got {d.get('record_count')}"
        )

    def test_v111_manifest_not_same_as_v200(self) -> None:
        """v1.1.1 manifest must not be identical to the current v2.0.0 manifest."""
        v111_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST_v1.1.1.json"
        v200_path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
        if not v111_path.exists() or not v200_path.exists():
            pytest.skip("Manifest files not found")
        v111_bytes = v111_path.read_bytes()
        v200_bytes = v200_path.read_bytes()
        assert v111_bytes != v200_bytes, (
            "v1.1.1 manifest must not be identical to v2.0.0 manifest"
        )

    def test_current_manifest_is_v200(self) -> None:
        """Current FROZEN_BENCHMARK_MANIFEST.json must be version 2.0.0."""
        path = BENCHMARK_DIR / "FROZEN_BENCHMARK_MANIFEST.json"
        assert path.exists(), "Current manifest not found"
        with path.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d.get("benchmark_version") == "2.0.0", (
            f"Current manifest must be v2.0.0, got {d.get('benchmark_version')}"
        )
        assert d.get("record_count") == 33, (
            f"Current v2.0.0 manifest must have 33 records, got {d.get('record_count')}"
        )

    def test_version_register_marks_v1x_superseded(self) -> None:
        """DATASET_VERSION_REGISTER.json must mark v1.x versions as SUPERSEDED NOT AUTHORIZED."""
        path = BENCHMARK_DIR / "DATASET_VERSION_REGISTER.json"
        assert path.exists(), "DATASET_VERSION_REGISTER.json not found"
        with path.open("r", encoding="utf-8") as fh:
            register = json.load(fh)
        versions = {v["version"]: v for v in register["versions"]}
        for ver in ("1.0.0", "1.1.0", "1.1.1"):
            assert versions[ver]["status"] == "SUPERSEDED", (
                f"Version {ver} must be SUPERSEDED"
            )
            assert "NOT AUTHORIZED" in versions[ver]["authorization"], (
                f"Version {ver} must be NOT AUTHORIZED FOR TRAINING"
            )
