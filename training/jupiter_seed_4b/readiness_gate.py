#!/usr/bin/env python3
"""
Jupiter Seed 4B — Dataset Readiness Gate (12 Gates)
=====================================================
Evaluates the full dataset against 12 mandatory gates.
No gate defaults to PASS. Every gate must explicitly pass.

Gates:
  1.  Schema integrity
  2.  Provenance integrity
  3.  UTF-8 portability
  4.  Language contract
  5.  Raw duplication
  6.  Canonical duplication
  7.  Semantic leakage
  8.  Content-family split isolation
  9.  Review-representation identity
  10. Benchmark manifest integrity
  11. Test-suite result
  12. Human-review status

Usage:
    python3 readiness_gate.py --data-dir training/jupiter_seed_4b/data
                              --output docs/jupiter_seed_4b/READINESS_GATE_REPORT.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Import canonicalization engine
_TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(_TRAINING_DIR))
from canonicalize import (
    canonicalize,
    canonical_hash,
    has_artificial_marker,
    jaccard,
    run_deduplication,
    sha256_of,
    word_ngrams,
    JACCARD_BLOCK_THRESHOLD,
    JACCARD_WARN_THRESHOLD,
)

# ─── Gate result types ────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
NOT_READY = "NOT_READY"
WARN = "WARN"


class GateResult:
    def __init__(self, gate_id: int, name: str, status: str,
                 details: str, warnings: List[str] = None, errors: List[str] = None):
        self.gate_id = gate_id
        self.name = name
        self.status = status
        self.details = details
        self.warnings = warnings or []
        self.errors = errors or []

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def failed(self) -> bool:
        return self.status in (FAIL, NOT_READY)


# ─── Required schema fields ───────────────────────────────────────────────

REQUIRED_FIELDS = [
    "example_id", "split", "prompt", "response", "language",
    "domain", "task_type", "difficulty", "provenance_type",
    "source_name", "source_url", "content_family_id", "scenario_brief",
    "language_contract_passed", "content_sha256",
    "canonical_prompt_hash", "canonical_response_hash",
    "scenario_brief_hash", "schema_version", "dataset_version",
]

REVIEWER_JUDGMENT_FIELDS = [
    "reviewer_name", "reviewer_experience", "review_date",
    "accuracy_score", "fluency_score", "gcc_appropriateness_score",
    "domain_terminology_score", "factuality_score", "verdict",
    "corrected_wording", "rejection_reason", "comments",
]

# ─── Blocking content patterns ────────────────────────────────────────────

BLOCKING_CONTENT_PATTERNS = [
    (re.compile(r"100%\s+guaranteed", re.IGNORECASE), "100% guaranteed claim"),
    (re.compile(r"^\s*$"), "empty response"),
    (re.compile(r"^\s*\[placeholder\]\s*$", re.IGNORECASE), "placeholder text"),
    (re.compile(r"^\s*\[.*\]\s*$"), "response is only a placeholder bracket"),
]

GCC_TRIP_PHRASES = [
    re.compile(r"يُنصح بمراجعة[^\.]{0,50}\.$"),
    re.compile(r"specific current requirements should be verified[^\.]{0,60}\.$", re.IGNORECASE),
]


def load_split(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_all(data_dir: Path) -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        records.extend(load_split(data_dir / f"{split}.jsonl"))
    return records


# ─── Gate 1: Schema integrity ─────────────────────────────────────────────

def gate_schema(records: List[Dict]) -> GateResult:
    errors = []
    for r in records:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            errors.append(f"{r.get('example_id', '?')}: missing {missing}")
    if errors:
        return GateResult(1, "Schema Integrity", FAIL,
                          f"{len(errors)} records missing required fields",
                          errors=errors[:10])
    return GateResult(1, "Schema Integrity", PASS,
                      f"All {len(records)} records have required fields")


# ─── Gate 2: Provenance integrity ─────────────────────────────────────────

def gate_provenance(records: List[Dict]) -> GateResult:
    errors = []
    warnings = []
    for r in records:
        prov = r.get("provenance_type", "")
        if not prov:
            errors.append(f"{r['example_id']}: missing provenance_type")
        if not r.get("source_url"):
            errors.append(f"{r['example_id']}: missing source_url")
        if not r.get("scenario_brief", "").strip():
            errors.append(f"{r['example_id']}: empty scenario_brief")
        if prov == "SOURCE_DEPENDENT_FACTUAL" and not r.get("factual_dependency"):
            warnings.append(f"{r['example_id']}: SOURCE_DEPENDENT_FACTUAL but factual_dependency=False")
        if prov == "ORIGINAL_SCENARIO" and "fictional_disclaimer" not in r:
            errors.append(f"{r['example_id']}: ORIGINAL_SCENARIO missing fictional_disclaimer")

    if errors:
        return GateResult(2, "Provenance Integrity", FAIL,
                          f"{len(errors)} provenance errors",
                          warnings=warnings[:5], errors=errors[:10])
    status = WARN if warnings else PASS
    return GateResult(2, "Provenance Integrity", status,
                      f"Provenance fields present. Warnings: {len(warnings)}",
                      warnings=warnings[:5])


# ─── Gate 3: UTF-8 portability ────────────────────────────────────────────

def gate_utf8(data_dir: Path) -> GateResult:
    """Test that Arabic content in JSONL files cannot be read under cp1252."""
    errors = []
    warnings = []

    for split in ("train", "valid", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        # Try reading with cp1252 — should fail if Arabic is present
        try:
            content = path.read_bytes()
            # Check if file contains Arabic bytes (UTF-8 encoded Arabic is multi-byte)
            has_arabic = any(b >= 0xC0 for b in content)
            if has_arabic:
                try:
                    content.decode("cp1252")
                    # If cp1252 decoding succeeds, it means no multi-byte sequences
                    # (unlikely with Arabic, but check)
                    warnings.append(f"{split}.jsonl: cp1252 decode succeeded — verify Arabic content")
                except UnicodeDecodeError:
                    pass  # Expected: cp1252 cannot decode Arabic UTF-8 bytes
            else:
                warnings.append(f"{split}.jsonl: no multi-byte content found — may lack Arabic")
        except Exception as e:
            errors.append(f"{split}.jsonl: {e}")

    # Check that all production Python files use explicit encoding
    training_dir = data_dir.parent
    py_files = list(training_dir.glob("*.py"))
    for path in py_files:
        if path.name.startswith("test_"):
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                kws = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                is_open = (isinstance(func, ast.Name) and func.id == "open") or \
                          (isinstance(func, ast.Attribute) and func.attr in {"open", "read_text", "write_text"})
                if is_open and "encoding" not in kws:
                    errors.append(f"{path.name}:{node.lineno}: text I/O without explicit encoding")

    if errors:
        return GateResult(3, "UTF-8 Portability", FAIL,
                          f"{len(errors)} UTF-8 portability errors",
                          warnings=warnings, errors=errors[:10])
    status = WARN if warnings else PASS
    return GateResult(3, "UTF-8 Portability", status,
                      "All text I/O uses explicit encoding. Arabic bytes fail cp1252 as expected.",
                      warnings=warnings)


# ─── Gate 4: Language contract ────────────────────────────────────────────

def gate_language_contract(records: List[Dict]) -> GateResult:
    errors = []
    for r in records:
        if not r.get("language_contract_passed", False):
            errors.append(
                f"{r['example_id']}: language={r['language']} "
                f"ar_chars_response={r.get('arabic_character_count_response', 0)}"
            )
    if errors:
        return GateResult(4, "Language Contract", FAIL,
                          f"{len(errors)} language contract violations",
                          errors=errors[:10])
    return GateResult(4, "Language Contract", PASS,
                      f"All {len(records)} records satisfy language contracts")


# ─── Gate 5: Raw duplication ──────────────────────────────────────────────

def gate_raw_duplication(records: List[Dict]) -> GateResult:
    errors = []
    id_to_split = {r["example_id"]: r["split"] for r in records}

    # Raw response hashes
    resp_to_ids = defaultdict(list)
    for r in records:
        resp_to_ids[sha256_of(r["response"])].append(r["example_id"])
    cross_split_raw = [
        ids for ids in resp_to_ids.values()
        if len(ids) > 1 and len({id_to_split[i] for i in ids}) > 1
    ]
    if cross_split_raw:
        for ids in cross_split_raw[:5]:
            errors.append(f"Raw cross-split response duplicate: {ids}")

    # Raw prompt hashes
    prompt_to_ids = defaultdict(list)
    for r in records:
        prompt_to_ids[sha256_of(r["prompt"])].append(r["example_id"])
    cross_split_prompt = [
        ids for ids in prompt_to_ids.values()
        if len(ids) > 1 and len({id_to_split[i] for i in ids}) > 1
    ]
    if cross_split_prompt:
        for ids in cross_split_prompt[:5]:
            errors.append(f"Raw cross-split prompt duplicate: {ids}")

    if errors:
        return GateResult(5, "Raw Duplication", FAIL,
                          f"{len(cross_split_raw)} cross-split response dups, "
                          f"{len(cross_split_prompt)} cross-split prompt dups",
                          errors=errors[:10])
    return GateResult(5, "Raw Duplication", PASS,
                      "No raw cross-split duplicates detected")


# ─── Gate 6: Canonical duplication ───────────────────────────────────────

def gate_canonical_duplication(records: List[Dict]) -> GateResult:
    errors = []
    id_to_split = {r["example_id"]: r["split"] for r in records}

    # Canonical response hashes
    canon_resp_to_ids = defaultdict(list)
    for r in records:
        canon_resp_to_ids[canonical_hash(r["response"])].append(r["example_id"])
    cross_split_canon = [
        ids for ids in canon_resp_to_ids.values()
        if len(ids) > 1 and len({id_to_split[i] for i in ids}) > 1
    ]
    if cross_split_canon:
        for ids in cross_split_canon[:5]:
            errors.append(f"Canonical cross-split response duplicate: {ids}")

    # Canonical scenario brief duplicates
    brief_to_ids = defaultdict(list)
    for r in records:
        brief_to_ids[sha256_of(canonicalize(r.get("scenario_brief", "")))].append(r["example_id"])
    dup_briefs = [ids for ids in brief_to_ids.values() if len(ids) > 1]
    if dup_briefs:
        for ids in dup_briefs[:5]:
            errors.append(f"Duplicate scenario brief: {ids}")

    # Check for artificial markers
    marker_ids = [r["example_id"] for r in records
                  if has_artificial_marker(r.get("prompt", "")) or
                  has_artificial_marker(r.get("response", ""))]
    if marker_ids:
        errors.append(f"Artificial markers in {len(marker_ids)} records: {marker_ids[:5]}")

    if errors:
        return GateResult(6, "Canonical Duplication", FAIL,
                          f"{len(cross_split_canon)} canonical cross-split dups, "
                          f"{len(dup_briefs)} duplicate briefs, "
                          f"{len(marker_ids)} artificial markers",
                          errors=errors[:10])
    return GateResult(6, "Canonical Duplication", PASS,
                      "No canonical cross-split duplicates, no duplicate scenario briefs, "
                      "no artificial markers")


# ─── Gate 7: Semantic leakage ─────────────────────────────────────────────

def gate_semantic_leakage(records: List[Dict]) -> GateResult:
    """
    Check for high-similarity cross-split pairs using Jaccard on canonical word trigrams.
    Blocking threshold: JACCARD_BLOCK_THRESHOLD (0.70)
    Warning threshold: JACCARD_WARN_THRESHOLD (0.50)
    """
    errors = []
    warnings = []

    split_records = defaultdict(list)
    for r in records:
        split_records[r["split"]].append(r)

    splits = list(split_records.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            for r1 in split_records[s1]:
                c1 = canonicalize(r1.get("response", ""))
                if len(c1) < 20:
                    continue
                ng1 = word_ngrams(c1, 3)
                for r2 in split_records[s2]:
                    c2 = canonicalize(r2.get("response", ""))
                    if len(c2) < 20:
                        continue
                    ng2 = word_ngrams(c2, 3)
                    j = jaccard(ng1, ng2)
                    if j >= JACCARD_BLOCK_THRESHOLD:
                        errors.append(
                            f"Blocking semantic similarity ({j:.3f}): "
                            f"{r1['example_id']} ({s1}) vs {r2['example_id']} ({s2})"
                        )
                    elif j >= JACCARD_WARN_THRESHOLD:
                        warnings.append(
                            f"Warning semantic similarity ({j:.3f}): "
                            f"{r1['example_id']} ({s1}) vs {r2['example_id']} ({s2})"
                        )

    if errors:
        return GateResult(7, "Semantic Leakage", FAIL,
                          f"{len(errors)} blocking cross-split semantic pairs "
                          f"(threshold={JACCARD_BLOCK_THRESHOLD})",
                          warnings=warnings[:5], errors=errors[:10])
    status = WARN if warnings else PASS
    return GateResult(7, "Semantic Leakage", status,
                      f"No blocking semantic leakage. Warnings: {len(warnings)}",
                      warnings=warnings[:5])


# ─── Gate 8: Content-family split isolation ───────────────────────────────

def gate_family_isolation(records: List[Dict]) -> GateResult:
    errors = []
    family_to_splits = defaultdict(set)
    for r in records:
        family = r.get("content_family_id", "")
        if family:
            family_to_splits[family].add(r["split"])
    violations = {f: list(s) for f, s in family_to_splits.items() if len(s) > 1}
    if violations:
        for family, splits in list(violations.items())[:5]:
            errors.append(f"Family {family} appears in splits: {splits}")
        return GateResult(8, "Content-Family Split Isolation", FAIL,
                          f"{len(violations)} families cross split boundaries",
                          errors=errors)
    return GateResult(8, "Content-Family Split Isolation", PASS,
                      "All content families are assigned to exactly one split")


# ─── Gate 9: Review-representation identity ───────────────────────────────

def gate_review_identity(data_dir: Path, docs_dir: Path) -> GateResult:
    errors = []
    jsonl_path = data_dir / "human_review_queue.jsonl"
    md_path = docs_dir / "ARABIC_HUMAN_REVIEW_QUEUE.md"
    csv_path = docs_dir / "REVIEWER_TEMPLATE.csv"

    if not jsonl_path.exists():
        return GateResult(9, "Review-Representation Identity", FAIL,
                          "human_review_queue.jsonl not found")

    with jsonl_path.open("r", encoding="utf-8") as fh:
        queue = [json.loads(l) for l in fh if l.strip()]
    jsonl_ids = {r["example_id"] for r in queue}

    if md_path.exists():
        md_content = md_path.read_text(encoding="utf-8")
        md_ids = set(re.findall(r"`(seed4b-[a-z]+-\d{4})`", md_content))
        if jsonl_ids != md_ids:
            errors.append(
                f"JSONL has {len(jsonl_ids)} IDs, Markdown has {len(md_ids)} IDs. "
                f"Diff: {jsonl_ids.symmetric_difference(md_ids)}"
            )
    else:
        errors.append("ARABIC_HUMAN_REVIEW_QUEUE.md not found")

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            csv_rows = list(reader)
        csv_ids = {row["example_id"] for row in csv_rows}
        if jsonl_ids != csv_ids:
            errors.append(
                f"JSONL has {len(jsonl_ids)} IDs, CSV has {len(csv_ids)} IDs. "
                f"Diff: {jsonl_ids.symmetric_difference(csv_ids)}"
            )
        # Check reviewer fields are empty
        for row in csv_rows:
            for field in REVIEWER_JUDGMENT_FIELDS:
                if field in row and row[field].strip():
                    errors.append(f"Reviewer field '{field}' is not empty for {row.get('example_id', '?')}")
                    break
    else:
        errors.append("REVIEWER_TEMPLATE.csv not found")

    if errors:
        return GateResult(9, "Review-Representation Identity", FAIL,
                          f"{len(errors)} review representation errors",
                          errors=errors[:10])
    return GateResult(9, "Review-Representation Identity", PASS,
                      f"JSONL, Markdown, and CSV all contain the same {len(jsonl_ids)} IDs. "
                      "All reviewer judgment fields are empty.")


# ─── Gate 10: Benchmark manifest integrity ────────────────────────────────

def gate_benchmark_manifest(benchmark_dir: Path) -> GateResult:
    errors = []
    warnings = []
    manifest_path = benchmark_dir / "FROZEN_BENCHMARK_MANIFEST.json"

    if not manifest_path.exists():
        return GateResult(10, "Benchmark Manifest Integrity", FAIL,
                          "FROZEN_BENCHMARK_MANIFEST.json not found")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    version = manifest.get("benchmark_version", "")
    if not version.startswith("2."):
        errors.append(f"Benchmark version must be 2.x.x, got {version}")

    approval = manifest.get("approval_status", "")
    if "PENDING" not in approval.upper():
        errors.append(f"Approval status must be PENDING, got '{approval}'")

    contamination = manifest.get("contamination_result", "")
    if contamination and contamination != "PASS":
        errors.append(f"Contamination check result is not PASS: {contamination}")

    # Check history
    v100 = benchmark_dir / "FROZEN_BENCHMARK_MANIFEST_v1.0.0.json"
    v110 = benchmark_dir / "FROZEN_BENCHMARK_MANIFEST_v1.1.0.json"
    v111 = benchmark_dir / "FROZEN_BENCHMARK_MANIFEST_v1.1.1.json"
    for path in [v100, v110, v111]:
        if not path.exists():
            warnings.append(f"Historical manifest not found: {path.name}")

    if errors:
        return GateResult(10, "Benchmark Manifest Integrity", FAIL,
                          f"{len(errors)} manifest errors",
                          warnings=warnings, errors=errors)
    status = WARN if warnings else PASS
    return GateResult(10, "Benchmark Manifest Integrity", status,
                      f"Manifest version {version}, approval PENDING.",
                      warnings=warnings)


# ─── Gate 11: Test-suite result ───────────────────────────────────────────

def gate_test_suite(repo_root: Path) -> GateResult:
    """Run the CPU test suite and check for zero failures."""
    test_dir = repo_root / "tests" / "jupiter_seed_4b"
    if not test_dir.exists():
        return GateResult(11, "Test-Suite Result", FAIL, "Test directory not found")

    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", str(test_dir), "--tb=no", "-q",
             "--ignore", str(test_dir / "test_adversarial_fixtures.py")],
            capture_output=True, text=True, timeout=120,
            cwd=str(repo_root),
            env={**__import__("os").environ, "PYTHONUTF8": "", "PYTHONIOENCODING": ""},
        )
        output = result.stdout + result.stderr
        # Parse results
        passed = failed = errors_count = skipped = 0
        for line in output.split("\n"):
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+) error", line)
            if m:
                errors_count = int(m.group(1))
            m = re.search(r"(\d+) skipped", line)
            if m:
                skipped = int(m.group(1))

        if failed > 0 or errors_count > 0:
            return GateResult(11, "Test-Suite Result", FAIL,
                              f"passed={passed} failed={failed} errors={errors_count} skipped={skipped}",
                              errors=[output[-500:]])
        return GateResult(11, "Test-Suite Result", PASS,
                          f"passed={passed} failed=0 errors=0 skipped={skipped}")
    except subprocess.TimeoutExpired:
        return GateResult(11, "Test-Suite Result", FAIL, "Test suite timed out")
    except Exception as e:
        return GateResult(11, "Test-Suite Result", FAIL, str(e))


# ─── Gate 12: Human-review status ────────────────────────────────────────

def gate_human_review(records: List[Dict]) -> GateResult:
    errors = []
    for r in records:
        if r.get("factuality_review_status") == "human_approved":
            errors.append(f"Software set human_approved on {r['example_id']}")
        if r.get("arabic_review_status") == "human_approved":
            errors.append(f"Software set arabic human_approved on {r['example_id']}")

    if errors:
        return GateResult(12, "Human-Review Status", FAIL,
                          f"Software illegally set human approval on {len(errors)} records",
                          errors=errors[:10])

    pending = sum(1 for r in records if r.get("human_review_required", False))
    return GateResult(12, "Human-Review Status", NOT_READY,
                      f"Human review pending for {pending} records. "
                      "No software approval detected. "
                      "Status: PENDING INDEPENDENT AUDIT AND HUMAN REVIEW")


# ─── Report writer ────────────────────────────────────────────────────────

def write_report(gates: List[GateResult], output_path: Path,
                 honest_count: Dict, dataset_version: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_passed = all(g.passed for g in gates if g.gate_id != 12)
    gate12 = next((g for g in gates if g.gate_id == 12), None)
    final_verdict = (
        "JUPITER SEED 4B REVIEW PACKAGE REPAIRED — READY FOR INDEPENDENT HUMAN REVIEW"
        if all_passed and gate12 and gate12.status == NOT_READY
        else "JUPITER SEED 4B REVIEW PACKAGE NOT READY"
    )

    lines = [
        "# Jupiter Seed 4B: Dataset Readiness Gate Report",
        "",
        f"**Dataset version:** {dataset_version}",
        f"**Approval status:** PENDING INDEPENDENT AUDIT AND HUMAN REVIEW",
        "",
        "## Honest Dataset Size",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Authorized target | {honest_count.get('authorized', 'N/A')} |",
        f"| Actual distinct examples | {honest_count.get('actual', 'N/A')} |",
        f"| Shortfall | {honest_count.get('shortfall', 'N/A')} |",
        f"| Human authoring required | {honest_count.get('shortfall', 'N/A')} additional examples |",
        "",
        "## Gate Results",
        "",
        "| Gate | Name | Status | Details |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for g in gates:
        icon = "✓" if g.passed else ("⚠" if g.status == WARN else ("⏳" if g.status == NOT_READY else "✗"))
        lines.append(f"| {g.gate_id} | {g.name} | {icon} {g.status} | {g.details} |")

    lines += ["", "## Detailed Gate Results", ""]
    for g in gates:
        lines.append(f"### Gate {g.gate_id}: {g.name} — {g.status}")
        lines.append("")
        lines.append(g.details)
        if g.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for w in g.warnings:
                lines.append(f"- {w}")
        if g.errors:
            lines.append("")
            lines.append("**Errors:**")
            for e in g.errors:
                lines.append(f"- {e}")
        lines.append("")

    lines += [
        "---",
        "",
        f"## Final Verdict",
        "",
        f"**{final_verdict}**",
        "",
        "AI-ASSISTED INTERNAL CONTENT AUDIT — NOT HUMAN APPROVAL",
    ]

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Readiness gate report written to %s", output_path)
    return final_verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Dataset Readiness Gate")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("training/jupiter_seed_4b/data"))
    parser.add_argument("--docs-dir", type=Path,
                        default=Path("docs/jupiter_seed_4b"))
    parser.add_argument("--benchmark-dir", type=Path,
                        default=Path("benchmarks/jupiter_seed_4b"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/jupiter_seed_4b/READINESS_GATE_REPORT.md"))
    parser.add_argument("--repo-root", type=Path,
                        default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_all(args.data_dir)

    if not records:
        log.error("No records found in %s", args.data_dir)
        sys.exit(1)

    log.info("Loaded %d records from %s", len(records), args.data_dir)

    # Honest count
    authorized = 850
    actual = len(records)
    shortfall = max(0, authorized - actual)
    honest_count = {
        "authorized": authorized,
        "actual": actual,
        "shortfall": shortfall,
    }

    # Dataset version from first record
    dataset_version = records[0].get("dataset_version", "unknown")

    # Run all 12 gates
    gates = [
        gate_schema(records),
        gate_provenance(records),
        gate_utf8(args.data_dir),
        gate_language_contract(records),
        gate_raw_duplication(records),
        gate_canonical_duplication(records),
        gate_semantic_leakage(records),
        gate_family_isolation(records),
        gate_review_identity(args.data_dir, args.docs_dir),
        gate_benchmark_manifest(args.benchmark_dir),
        gate_test_suite(args.repo_root),
        gate_human_review(records),
    ]

    # Report
    verdict = write_report(gates, args.output, honest_count, dataset_version)

    # Summary
    passed = sum(1 for g in gates if g.passed)
    failed = sum(1 for g in gates if g.failed and g.gate_id != 12)
    not_ready = sum(1 for g in gates if g.status == NOT_READY)
    warned = sum(1 for g in gates if g.status == WARN)

    log.info("=== READINESS GATE SUMMARY ===")
    log.info("Gates passed: %d / 12", passed)
    log.info("Gates failed: %d", failed)
    log.info("Gates not_ready: %d", not_ready)
    log.info("Gates warned: %d", warned)
    log.info("Honest dataset size: %d / %d (shortfall: %d)", actual, authorized, shortfall)
    log.info("Final verdict: %s", verdict)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
