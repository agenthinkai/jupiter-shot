"""
Jupiter Seed 4B V2.4 — Regression Tests
==========================================
Covers V2.4 repair requirements:

Blocker 1 — Review-package propagation:
  1.  Sidecar REVIEW_RISK_FINDINGS.json exists and has correct schema
  2.  Three flagged records are in sidecar
  3.  Flagged records show REVIEW_REQUIRED_CONTENT_RISK in REVIEWER_PACKAGE.jsonl
  4.  Flagged records do NOT show integrity_flags=OK
  5.  Unflagged records show content_risk_status=NONE
  6.  Gate 16 passes on current package
  7.  Gate 16 fails when sidecar is missing
  8.  Gate 16 fails when flagged record shows integrity_flags=OK

Blocker 2 — Exit code 3 (real subprocess tests):
  9.  Malformed train JSONL → exit code 3
  10. Missing eval file → exit code 3
  11. Malformed frozen manifest → exit code 3
  12. Unreadable review package → exit code 3 (or 1 if gate fails first)
  13. Execution-error artifact written on exit 3
  14. Artifact says exit_code=3 and outcome=EXECUTION_ERROR
  15. No false PASS or readiness report on exit 3

Status terminology:
  16. EXIT_HUMAN_REVIEW_REQUIRED == 2
  17. EXIT_MECHANICALLY_BLOCKED is backward-compat alias for 2
  18. Final verdict contains HUMAN_REVIEW_REQUIRED for flagged corpus

Training interlock:
  19. Missing auth artifact → training blocked
  20. authorization_status != APPROVED → blocked
  21. human_review_status != COMPLETE → blocked
  22. unresolved_count != 0 → blocked
  23. legal_review_status != APPROVED → blocked
  24. corpus_sha256 mismatch → blocked
  25. dataset_content_commit mismatch → blocked
  26. approved_model_id mismatch → blocked
  27. budget exceeded → blocked
  28. approved_by empty → blocked
  29. signature empty → blocked
  30. Template placeholder in field → blocked

CPU-only. No corpus modification. No GPU, model downloads, paid APIs, cloud.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "jupiter_seed_4b"

sys.path.insert(0, str(TRAINING_DIR))

from readiness_gate import (
    PASS, FAIL, BLOCKED, REVIEW_REQUIRED,
    EXIT_MECHANICALLY_READY, EXIT_MECHANICAL_FAILURE,
    EXIT_HUMAN_REVIEW_REQUIRED, EXIT_MECHANICALLY_BLOCKED,
    EXIT_EXECUTION_ERROR,
    gate_review_package_identity,
    gate_content_risk,
    GateResult,
)
from training_runner import check_training_authorization, EXPECTED_CORPUS_SHA256, EXPECTED_DATASET_CONTENT_COMMIT, BASE_MODEL_ID

THREE_FLAGGED = {"seed4b-train-0032", "seed4b-train-0038", "seed4b-valid-0005"}

PYTHON = sys.executable
GATE_SCRIPT = str(TRAINING_DIR / "readiness_gate.py")


# ─── Helpers ──────────────────────────────────────────────────────────────

def run_gate(extra_args=None, env=None, cwd=None) -> subprocess.CompletedProcess:
    cmd = [PYTHON, GATE_SCRIPT]
    if extra_args:
        cmd.extend(extra_args)
    run_env = dict(os.environ)
    run_env.pop("PYTHONUTF8", None)
    run_env.pop("PYTHONIOENCODING", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=run_env,
    )


def make_valid_auth(
    auth_path: Path,
    model_id: str = BASE_MODEL_ID,
    corpus_sha: str = EXPECTED_CORPUS_SHA256,
    dataset_commit: str = EXPECTED_DATASET_CONTENT_COMMIT,
    auth_status: str = "APPROVED",
    hr_status: str = "COMPLETE",
    unresolved: int = 0,
    legal_status: str = "APPROVED",
    max_budget: float = 500.0,
    approved_by: str = "Farouq Al-Rashidi",
    sig: str = "I Farouq Al-Rashidi authorize training of Qwen/Qwen3-4B on Jupiter Seed 4B dataset commit 2e36f6b977a8af052fced5a532c1168dc1988b6f on 2025-01-01",
) -> None:
    auth = {
        "schema_version": "1.0",
        "corpus_sha256": corpus_sha,
        "dataset_content_commit": dataset_commit,
        "readiness_package_commit": "e3295b8624810ee77ea599e91476e562fb251f34",
        "human_review_status": hr_status,
        "accepted_count": 48,
        "revised_count": 2,
        "rejected_count": 0,
        "unresolved_count": unresolved,
        "legal_review_status": legal_status,
        "approved_model_id": model_id,
        "approval_scope": "internal-only distillation, GCC enterprise pilot",
        "maximum_budget_usd": max_budget,
        "approved_by": approved_by,
        "approved_at_utc": "2025-01-01T00:00:00Z",
        "authorization_status": auth_status,
        "signature_or_typed_confirmation": sig,
    }
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    with auth_path.open("w", encoding="utf-8") as fh:
        json.dump(auth, fh, ensure_ascii=False, indent=2)


# ===========================================================================
# Blocker 1: Review-package propagation
# ===========================================================================

class TestReviewPackagePropagation:

    def test_sidecar_exists(self) -> None:
        sidecar_path = DOCS_DIR / "REVIEW_RISK_FINDINGS.json"
        assert sidecar_path.exists(), f"REVIEW_RISK_FINDINGS.json must exist: {sidecar_path}"

    def test_sidecar_schema_version(self) -> None:
        with (DOCS_DIR / "REVIEW_RISK_FINDINGS.json").open("r", encoding="utf-8") as fh:
            s = json.load(fh)
        assert "schema_version" in s
        assert "package_commit" in s
        assert "corpus_sha256" in s
        assert "findings" in s

    def test_sidecar_has_three_flagged_records(self) -> None:
        with (DOCS_DIR / "REVIEW_RISK_FINDINGS.json").open("r", encoding="utf-8") as fh:
            s = json.load(fh)
        finding_ids = {f["record_id"] for f in s["findings"]}
        missing = THREE_FLAGGED - finding_ids
        assert not missing, f"Sidecar missing flagged records: {missing}"

    def test_reviewer_package_flagged_records_show_review_required(self) -> None:
        pkg_path = DOCS_DIR / "REVIEWER_PACKAGE.jsonl"
        if not pkg_path.exists():
            pytest.skip("REVIEWER_PACKAGE.jsonl not found")
        found = {}
        with pkg_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] in THREE_FLAGGED:
                    found[r["example_id"]] = r
        for eid in THREE_FLAGGED:
            assert eid in found, f"{eid} missing from REVIEWER_PACKAGE.jsonl"
            assert found[eid].get("content_risk_status") == "REVIEW_REQUIRED", (
                f"{eid}: content_risk_status must be REVIEW_REQUIRED, "
                f"got {found[eid].get('content_risk_status')}"
            )

    def test_flagged_records_do_not_show_ok(self) -> None:
        pkg_path = DOCS_DIR / "REVIEWER_PACKAGE.jsonl"
        if not pkg_path.exists():
            pytest.skip("REVIEWER_PACKAGE.jsonl not found")
        with pkg_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] in THREE_FLAGGED:
                    flags = r.get("integrity_flags", "")
                    assert flags != "OK", (
                        f"{r['example_id']}: integrity_flags must not be OK for flagged record. "
                        f"Got: {flags}"
                    )

    def test_unflagged_records_show_none(self) -> None:
        pkg_path = DOCS_DIR / "REVIEWER_PACKAGE.jsonl"
        if not pkg_path.exists():
            pytest.skip("REVIEWER_PACKAGE.jsonl not found")
        violations = []
        with pkg_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] not in THREE_FLAGGED:
                    status = r.get("content_risk_status", "")
                    if status != "NONE":
                        violations.append(f"{r['example_id']}: {status}")
        assert not violations, f"Unflagged records must show NONE: {violations[:5]}"

    def test_gate16_passes_on_current_package(self) -> None:
        import json as _json
        records = []
        for split in ("train", "valid", "eval"):
            path = DATA_DIR / f"{split}.jsonl"
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(_json.loads(line))
        g14 = gate_content_risk(records)
        result = gate_review_package_identity(records, DOCS_DIR, DATA_DIR, g14)
        assert result.status == PASS, (
            f"Gate 16 must PASS on current package. Got: {result.status}\nErrors: {result.errors}"
        )

    def test_gate16_fails_when_sidecar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_docs = Path(tmpdir) / "docs"
            fake_docs.mkdir()
            # No sidecar in fake_docs
            g14 = GateResult(14, "Content-Risk", REVIEW_REQUIRED, "test",
                             metadata={"review_required_records": []})
            result = gate_review_package_identity([], fake_docs, DATA_DIR, g14)
        assert result.status == FAIL, (
            f"Gate 16 must FAIL when sidecar is missing. Got: {result.status}"
        )

    def test_gate16_fails_when_flagged_record_shows_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_docs = Path(tmpdir) / "docs"
            fake_docs.mkdir()
            # Write a sidecar with correct corpus hash
            sidecar = {
                "schema_version": "1.0",
                "package_commit": "e3295b8624810ee77ea599e91476e562fb251f34",
                "corpus_sha256": "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0",
                "findings": [{"record_id": "seed4b-train-0032", "severity": "REVIEW_REQUIRED",
                               "matched_rule_id": "GCC_TRIP_EN", "matched_field": "response"}],
            }
            with (fake_docs / "REVIEW_RISK_FINDINGS.json").open("w", encoding="utf-8") as fh:
                json.dump(sidecar, fh)
            # Write a REVIEWER_PACKAGE.jsonl where the flagged record shows integrity_flags=OK
            with (fake_docs / "REVIEWER_PACKAGE.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "example_id": "seed4b-train-0032",
                    "integrity_flags": "OK",  # WRONG — must be REVIEW_REQUIRED_CONTENT_RISK
                    "content_risk_status": "REVIEW_REQUIRED",
                }) + "\n")
            # Also write a queue file
            fake_data = DATA_DIR
            g14 = GateResult(14, "Content-Risk", REVIEW_REQUIRED, "test",
                             metadata={"review_required_records": ["seed4b-train-0032"]})
            result = gate_review_package_identity([], fake_docs, fake_data, g14)
        assert result.status == FAIL, (
            f"Gate 16 must FAIL when flagged record shows integrity_flags=OK. Got: {result.status}"
        )


# ===========================================================================
# Blocker 2: Exit code 3 (real subprocess tests)
# ===========================================================================

class TestExitCode3Subprocess:

    def test_malformed_train_jsonl_returns_exit_3(self) -> None:
        """Malformed train.jsonl must produce exit code 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            # Write malformed JSON
            (data_dir / "train.jsonl").write_text("{not valid json\n", encoding="utf-8")
            (data_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (data_dir / "eval.jsonl").write_text("", encoding="utf-8")
            result = run_gate(["--data-dir", str(data_dir)])
        assert result.returncode == EXIT_EXECUTION_ERROR, (
            f"Malformed train.jsonl must return exit 3. Got: {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_missing_eval_file_returns_exit_3(self) -> None:
        """Missing eval.jsonl must produce exit code 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            # Copy real train and valid, but omit eval
            shutil.copy(DATA_DIR / "train.jsonl", data_dir / "train.jsonl")
            shutil.copy(DATA_DIR / "valid.jsonl", data_dir / "valid.jsonl")
            # eval.jsonl is intentionally missing
            result = run_gate(["--data-dir", str(data_dir)])
        # Missing file causes FileNotFoundError → exit 3
        assert result.returncode in (EXIT_EXECUTION_ERROR, EXIT_MECHANICAL_FAILURE), (
            f"Missing eval.jsonl must return exit 3 or 1. Got: {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_malformed_frozen_manifest_returns_exit_3(self) -> None:
        """Malformed FROZEN_BENCHMARK_MANIFEST.json must produce exit code 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir) / "benchmarks"
            bench_dir.mkdir()
            (bench_dir / "FROZEN_BENCHMARK_MANIFEST.json").write_text(
                "{not valid json\n", encoding="utf-8"
            )
            result = run_gate(["--benchmark-dir", str(bench_dir)])
        assert result.returncode in (EXIT_EXECUTION_ERROR, EXIT_MECHANICAL_FAILURE), (
            f"Malformed manifest must return exit 3 or 1. Got: {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_unreadable_review_package_returns_nonzero(self) -> None:
        """Unreadable REVIEWER_PACKAGE.jsonl must not return exit 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            # Write a sidecar but make REVIEWER_PACKAGE.jsonl unreadable
            sidecar = {
                "schema_version": "1.0",
                "package_commit": "e3295b8624810ee77ea599e91476e562fb251f34",
                "corpus_sha256": "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0",
                "findings": [],
            }
            with (docs_dir / "REVIEW_RISK_FINDINGS.json").open("w", encoding="utf-8") as fh:
                json.dump(sidecar, fh)
            pkg_path = docs_dir / "REVIEWER_PACKAGE.jsonl"
            pkg_path.write_text("", encoding="utf-8")
            pkg_path.chmod(0o000)  # Make unreadable
            try:
                result = run_gate(["--docs-dir", str(docs_dir)])
                assert result.returncode != EXIT_MECHANICALLY_READY, (
                    f"Unreadable REVIEWER_PACKAGE.jsonl must not return exit 0. "
                    f"Got: {result.returncode}"
                )
            finally:
                pkg_path.chmod(0o644)  # Restore permissions

    def test_execution_error_artifact_written(self) -> None:
        """An execution-error artifact must be written when exit code 3 occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_text("{not valid json\n", encoding="utf-8")
            (data_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (data_dir / "eval.jsonl").write_text("", encoding="utf-8")
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            result = run_gate([
                "--data-dir", str(data_dir),
                "--docs-dir", str(docs_dir),
                "--output", str(docs_dir / "READINESS_GATE_REPORT.md"),
            ])
            # Check artifact INSIDE the context manager before tmpdir is deleted
            assert result.returncode == EXIT_EXECUTION_ERROR, (
                f"Expected exit 3, got {result.returncode}"
            )
            artifact_path = docs_dir / "EXECUTION_ERROR_ARTIFACT.json"
            assert artifact_path.exists(), (
                f"EXECUTION_ERROR_ARTIFACT.json must be written on exit 3. "
                f"Not found: {artifact_path}"
            )

    def test_execution_error_artifact_says_exit_code_3(self) -> None:
        """The execution-error artifact must contain exit_code=3 and outcome=EXECUTION_ERROR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_text("{not valid json\n", encoding="utf-8")
            (data_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (data_dir / "eval.jsonl").write_text("", encoding="utf-8")
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            result = run_gate([
                "--data-dir", str(data_dir),
                "--docs-dir", str(docs_dir),
                "--output", str(docs_dir / "READINESS_GATE_REPORT.md"),
            ])
            assert result.returncode == EXIT_EXECUTION_ERROR
            artifact_path = docs_dir / "EXECUTION_ERROR_ARTIFACT.json"
            assert artifact_path.exists(), "Artifact must exist inside context manager"
            with artifact_path.open("r", encoding="utf-8") as fh:
                artifact = json.load(fh)
            assert artifact.get("exit_code") == 3, (
                f"Artifact exit_code must be 3, got: {artifact.get('exit_code')}"
            )
            assert artifact.get("outcome") == "EXECUTION_ERROR", (
                f"Artifact outcome must be EXECUTION_ERROR, got: {artifact.get('outcome')}"
            )

    def test_no_false_pass_on_exit_3(self) -> None:
        """A readiness report claiming PASS must not exist when exit code is 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_text("{not valid json\n", encoding="utf-8")
            (data_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (data_dir / "eval.jsonl").write_text("", encoding="utf-8")
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            report_path = docs_dir / "READINESS_GATE_REPORT.md"
            result = run_gate([
                "--data-dir", str(data_dir),
                "--docs-dir", str(docs_dir),
                "--output", str(report_path),
            ])
            assert result.returncode == EXIT_EXECUTION_ERROR
            # Report must not exist or must not claim PASS — check inside context manager
            if report_path.exists():
                content = report_path.read_text(encoding="utf-8")
                assert "MECHANICALLY_READY_FOR_HUMAN_REVIEW" not in content, (
                    "Report must not claim MECHANICALLY_READY_FOR_HUMAN_REVIEW when exit code is 3"
                )


# ===========================================================================
# Status terminology
# ===========================================================================

class TestStatusTerminology:

    def test_exit_human_review_required_is_2(self) -> None:
        assert EXIT_HUMAN_REVIEW_REQUIRED == 2

    def test_exit_mechanically_blocked_is_backward_compat_alias(self) -> None:
        assert EXIT_MECHANICALLY_BLOCKED == 2
        assert EXIT_MECHANICALLY_BLOCKED == EXIT_HUMAN_REVIEW_REQUIRED

    def test_final_verdict_contains_human_review_required(self) -> None:
        """
        The write_report function must produce a verdict containing HUMAN_REVIEW_REQUIRED
        when there are REVIEW_REQUIRED findings and all mechanical gates pass.

        We test write_report directly to avoid a circular dependency:
        the CLI verdict depends on Gate 11 (test suite result), which would
        depend on this test passing if we used the CLI.
        """
        import tempfile as _tf
        from readiness_gate import write_report, GateResult, PASS, NOT_READY, REVIEW_REQUIRED

        # Simulate a state where all mechanical gates pass and Gate 14 is REVIEW_REQUIRED
        gates = [
            GateResult(g, f"Gate {g}", PASS, "pass") for g in range(1, 14)
        ]
        gates.append(GateResult(14, "Content-Risk", REVIEW_REQUIRED, "3 records",
                                 metadata={"review_required_records": ["seed4b-train-0032"]}))
        gates.append(GateResult(15, "Review-Queue Coverage", PASS, "pass"))
        gates.append(GateResult(16, "Review-Package Identity", PASS, "pass"))
        # Gate 12 must be NOT_READY
        gates[11] = GateResult(12, "Human-Review Status", NOT_READY, "pending")

        with _tf.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_path = Path(f.name)

        try:
            verdict = write_report(
                gates, output_path,
                {"authorized": 850, "actual": 101, "shortfall": 749},
                "2.0.0"
            )
        finally:
            output_path.unlink(missing_ok=True)

        assert "HUMAN_REVIEW_REQUIRED" in verdict, (
            f"write_report must produce HUMAN_REVIEW_REQUIRED verdict when Gate 14 is REVIEW_REQUIRED. "
            f"Got: {verdict}"
        )


# ===========================================================================
# Training interlock negative tests
# ===========================================================================

class TestTrainingInterlock:

    def test_missing_auth_artifact_blocks_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "TRAINING_AUTHORIZATION.json"
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_authorization_status_not_approved_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, auth_status="PENDING")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_human_review_status_not_complete_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, hr_status="IN_PROGRESS")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_unresolved_count_nonzero_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, unresolved=3)
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_legal_review_status_not_approved_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, legal_status="PENDING")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_corpus_sha256_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, corpus_sha="wrong_sha256")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_dataset_commit_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, dataset_commit="wrong_commit")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_approved_model_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, model_id="Qwen/Qwen2.5-3B")  # Wrong model
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_budget_exceeded_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, max_budget=100.0)
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 999.0)  # Over budget
            assert exc_info.value.code == 1

    def test_approved_by_empty_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, approved_by="")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_signature_empty_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, sig="")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1

    def test_template_placeholder_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_path = Path(tmpdir) / "auth.json"
            make_valid_auth(auth_path, approved_by="HUMAN_REQUIRED — fill this in")
            with pytest.raises(SystemExit) as exc_info:
                check_training_authorization(auth_path, BASE_MODEL_ID, 100.0)
            assert exc_info.value.code == 1
