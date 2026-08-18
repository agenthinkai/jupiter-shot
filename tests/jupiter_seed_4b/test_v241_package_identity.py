"""Jupiter Seed 4B V2.4.1 — fail-closed package-identity regression tests.

These tests exercise production helpers and Gate 16 against real generated
representations. They never modify the frozen corpus or launch training.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
DATA_DIR = TRAINING_DIR / "data"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
GENERATOR = TRAINING_DIR / "generate_review_package.py"
sys.path.insert(0, str(TRAINING_DIR))

import readiness_gate as rg

EXPECTED_FLAGGED = {
    "seed4b-eval-0004", "seed4b-train-0015", "seed4b-train-0016",
    "seed4b-train-0028", "seed4b-train-0032", "seed4b-train-0038",
    "seed4b-train-0039", "seed4b-valid-0005", "seed4b-valid-0014",
}
HUMAN_FIELDS = rg.REVIEWER_JUDGMENT_FIELDS


def load_records() -> list[dict]:
    records = []
    for split in ("train", "valid", "eval"):
        with (DATA_DIR / f"{split}.jsonl").open("r", encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    return records


def source_commit() -> str:
    return rg.resolve_git_commit("HEAD^")


def gate16(docs_dir: Path) -> rg.GateResult:
    return rg.gate_review_package_identity(
        load_records(), docs_dir, DATA_DIR, rg.gate_content_risk(load_records())
    )


def copy_docs(tmpdir: str) -> Path:
    destination = Path(tmpdir) / "docs"
    shutil.copytree(DOCS_DIR, destination)
    return destination


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestV241PackageIdentity:
    def test_production_gate16_passes_on_release_package(self) -> None:
        result = gate16(DOCS_DIR)
        assert result.status == rg.PASS, result.errors
        assert result.metadata["artifact_source_commit"] == source_commit()

    def test_production_generator_reproducibly_embeds_source_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "generated"
            result = subprocess.run(
                [
                    sys.executable, str(GENERATOR), "--docs-dir", str(out),
                    "--package-commit", source_commit(),
                ],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, result.stderr
            sidecar = json.loads((out / "REVIEW_RISK_FINDINGS.json").read_text(encoding="utf-8"))
            assert sidecar["package_commit"] == source_commit()
            package = load_jsonl(out / "REVIEWER_PACKAGE.jsonl")
            assert {r["package_commit"] for r in package} == {source_commit()}
            with (out / "REVIEWER_TEMPLATE.csv").open("r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert {r["package_commit"] for r in rows} == {source_commit()}
            markdown = (out / "ARABIC_HUMAN_REVIEW_QUEUE.md").read_text(encoding="utf-8")
            assert f"> Package source commit: `{source_commit()}`" in markdown

    def test_sidecar_stale_commit_fails_gate16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = copy_docs(tmpdir)
            path = docs / "REVIEW_RISK_FINDINGS.json"
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            sidecar["package_commit"] = "0" * 40
            path.write_text(json.dumps(sidecar), encoding="utf-8")
            result = gate16(docs)
        assert result.status == rg.FAIL
        assert any("Sidecar package_commit mismatch" in e for e in result.errors)

    @pytest.mark.parametrize("representation", ["jsonl", "csv", "markdown"])
    def test_stale_representation_commit_fails_gate16(self, representation: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = copy_docs(tmpdir)
            stale = "f" * 40
            if representation == "jsonl":
                rows = load_jsonl(docs / "REVIEWER_PACKAGE.jsonl")
                rows[0]["package_commit"] = stale
                with (docs / "REVIEWER_PACKAGE.jsonl").open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif representation == "csv":
                path = docs / "REVIEWER_TEMPLATE.csv"
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                    fields = fh.seek(0) or None
                fieldnames = list(rows[0])
                rows[0]["package_commit"] = stale
                with path.open("w", encoding="utf-8-sig", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                path = docs / "ARABIC_HUMAN_REVIEW_QUEUE.md"
                path.write_text(
                    path.read_text(encoding="utf-8").replace(source_commit(), stale, 1),
                    encoding="utf-8",
                )
            result = gate16(docs)
        assert result.status == rg.FAIL
        assert any("package_commit mismatch" in e or "Markdown package_commit mismatch" in e for e in result.errors)

    @pytest.mark.parametrize("bad_value", ["", "   ", "abcdef0", "not-a-git-hash"])
    def test_missing_or_malformed_sidecar_commit_fails_gate16(self, bad_value: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = copy_docs(tmpdir)
            path = docs / "REVIEW_RISK_FINDINGS.json"
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            sidecar["package_commit"] = bad_value
            path.write_text(json.dumps(sidecar), encoding="utf-8")
            result = gate16(docs)
        assert result.status == rg.FAIL
        assert any("missing, blank, or malformed" in e for e in result.errors)

    def test_git_resolution_nonzero_is_typed_failure(self) -> None:
        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="bad")
        with pytest.raises(rg.PackageIdentityResolutionError, match="nonzero"):
            rg.resolve_git_commit("HEAD", run_fn=fake_run)

    def test_git_resolution_empty_output_is_typed_failure(self) -> None:
        def fake_run(*args, **kwargs):
            return SimpleNamespace(returncode=0, stdout="\n", stderr="")
        with pytest.raises(rg.PackageIdentityResolutionError, match="empty"):
            rg.resolve_git_commit("HEAD", run_fn=fake_run)

    def test_git_resolution_unavailable_is_typed_failure(self) -> None:
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("git")
        with pytest.raises(rg.PackageIdentityResolutionError, match="unavailable"):
            rg.resolve_git_commit("HEAD", run_fn=fake_run)

    def test_git_resolution_timeout_is_typed_failure(self) -> None:
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 10)
        with pytest.raises(rg.PackageIdentityResolutionError, match="timed out"):
            rg.resolve_git_commit("HEAD", run_fn=fake_run)

    def test_invalid_repository_root_is_typed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(rg.PackageIdentityResolutionError, match="invalid repository root"):
                rg.resolve_git_commit("HEAD", repo_root=Path(tmpdir))

    def test_identity_resolution_failure_cannot_bypass_gate16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*args, **kwargs):
            raise rg.PackageIdentityResolutionError("forced resolver failure")
        monkeypatch.setattr(rg, "resolve_git_commit", fail)
        result = gate16(DOCS_DIR)
        assert result.status == rg.FAIL
        assert "forced resolver failure" in result.errors[0]

    def test_exact_status_counts_and_no_clear_alias(self) -> None:
        package = load_jsonl(DOCS_DIR / "REVIEWER_PACKAGE.jsonl")
        statuses = [r["content_risk_status"] for r in package]
        assert statuses.count("REVIEW_REQUIRED") == 9
        assert statuses.count("NONE") == 44
        assert "CLEAR" not in statuses
        assert {r["example_id"] for r in package if r["content_risk_status"] == "REVIEW_REQUIRED"} == EXPECTED_FLAGGED

    def test_cross_format_id_and_identity_parity(self) -> None:
        package = load_jsonl(DOCS_DIR / "REVIEWER_PACKAGE.jsonl")
        with (DOCS_DIR / "REVIEWER_TEMPLATE.csv").open("r", encoding="utf-8-sig", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        md = (DOCS_DIR / "ARABIC_HUMAN_REVIEW_QUEUE.md").read_text(encoding="utf-8")
        package_ids = {r["example_id"] for r in package}
        csv_ids = {r["example_id"] for r in csv_rows}
        assert len(package_ids) == len(csv_ids) == 53
        assert package_ids == csv_ids
        assert f"> Package source commit: `{source_commit()}`" in md
        assert {r["package_commit"] for r in package} == {source_commit()}
        assert {r["package_commit"] for r in csv_rows} == {source_commit()}

    def test_human_judgment_fields_are_empty(self) -> None:
        package = load_jsonl(DOCS_DIR / "REVIEWER_PACKAGE.jsonl")
        with (DOCS_DIR / "REVIEWER_TEMPLATE.csv").open("r", encoding="utf-8-sig", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        for rows in (package, csv_rows):
            for row in rows:
                for field in HUMAN_FIELDS:
                    assert not str(row.get(field, "")).strip(), f"{row['example_id']}:{field} populated"

    def test_tampering_flagged_risk_state_fails_gate16(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = copy_docs(tmpdir)
            rows = load_jsonl(docs / "REVIEWER_PACKAGE.jsonl")
            for row in rows:
                if row["example_id"] == "seed4b-train-0032":
                    row["integrity_flags"] = "OK"
            with (docs / "REVIEWER_PACKAGE.jsonl").open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            result = gate16(docs)
        assert result.status == rg.FAIL
        assert any("integrity_flags invalid" in e for e in result.errors)
