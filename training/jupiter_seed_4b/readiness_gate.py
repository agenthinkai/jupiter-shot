#!/usr/bin/env python3
"""
Jupiter Seed 4B — Dataset Readiness Gate (12 Gates) v2.1
=========================================================
Tooling-only repair: fixes Defects 1, 2, 4 from independent audit.

Defect 1 fix: Gate 11 uses sys.executable + JUnit XML, never python3/python.
Defect 2 fix: Gate 3 UTF-8 audit covers both training/ and tests/ directories.
Defect 4 fix: Gate 4 independently recomputes language contracts from text.

Corpus is FROZEN. This file does not modify any dataset records.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Import canonicalization engine
_TRAINING_DIR = Path(__file__).resolve().parent
# Deterministic repository root: training/jupiter_seed_4b/readiness_gate.py -> repo root
REPO_ROOT = _TRAINING_DIR.parents[1]
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
BLOCKED = "BLOCKED"           # V2.3: objectively prohibited content
REVIEW_REQUIRED = "REVIEW_REQUIRED"  # V2.3: culturally sensitive, needs human review

# V2.3 Exit-code contract
EXIT_MECHANICALLY_READY = 0    # all mechanical gates pass; Gate 12 pending human judgment
EXIT_MECHANICAL_FAILURE = 1    # failed integrity test, gate, contamination, schema, test, corpus
EXIT_HUMAN_REVIEW_REQUIRED = 2  # human review required before training/release (was MECHANICALLY_BLOCKED)
EXIT_MECHANICALLY_BLOCKED = EXIT_HUMAN_REVIEW_REQUIRED  # backward-compat alias
EXIT_EXECUTION_ERROR = 3       # dependency, subprocess, missing file, malformed artifact

# Authorized Seed 4B test files (must all be present and run)
# V2.2 Task 3: 8 files (test_gate_adversarial.py and test_v22_regression.py added)
AUTHORIZED_TEST_FILES = [
    "test_adversarial_fixtures.py",
    "test_gate_adversarial.py",
    "test_repair_audit.py",
    "test_integrity_audit.py",
    "test_gold_dataset.py",
    "test_diagnostic_pipeline.py",
    "test_foundation_audit.py",
    "test_v22_regression.py",
    "test_v23_regex.py",      # V2.3: regex regression tests
    "test_v23_regression.py", # V2.3: content-risk and exit-code regression tests
    "test_v24_regression.py", # V2.4: review-package propagation and training interlock
    "test_v241_package_identity.py", # V2.4.1: fail-closed package identity
]

# Canonical authorized manifest hash (SHA-256 of sorted filenames joined by '|')
# Recompute with: sha256("|".join(sorted(AUTHORIZED_TEST_FILES)).encode()).hexdigest()
import hashlib as _hashlib
AUTHORIZED_MANIFEST_HASH = _hashlib.sha256(
    "|".join(sorted(AUTHORIZED_TEST_FILES)).encode("utf-8")
).hexdigest()


class GateResult:
    def __init__(self, gate_id: int, name: str, status: str,
                 details: str, warnings: List[str] = None, errors: List[str] = None,
                 metadata: Dict = None):
        self.gate_id = gate_id
        self.name = name
        self.status = status
        self.details = details
        self.warnings = warnings or []
        self.errors = errors or []
        self.metadata = metadata or {}

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def failed(self) -> bool:
        return self.status in (FAIL, NOT_READY)

    @property
    def is_blocked(self) -> bool:
        return self.status == BLOCKED

    @property
    def is_review_required(self) -> bool:
        return self.status == REVIEW_REQUIRED


# ─── Package identity helpers (V2.4.1) ─────────────────────────────────────

GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class PackageIdentityResolutionError(RuntimeError):
    """A fail-closed failure while resolving a Git package identity."""


def is_valid_git_commit(value: Any) -> bool:
    """Return True only for a full 40-character hexadecimal Git object ID."""
    return isinstance(value, str) and GIT_COMMIT_RE.fullmatch(value.strip()) is not None


def resolve_git_commit(
    revision: str,
    repo_root: Optional[Path] = None,
    run_fn: Any = None,
) -> str:
    """Resolve `revision` to a full Git commit hash or raise a typed fail-closed error.

    The root is derived from this module by default and never from the caller's
    current working directory. Callers must not convert any error into an empty
    string, because an empty identity is not evidence of equality.
    """
    root = Path(repo_root if repo_root is not None else REPO_ROOT).resolve()
    if not root.is_dir() or not (root / '.git').exists():
        raise PackageIdentityResolutionError(f'invalid repository root: {root}')
    runner = run_fn or subprocess.run
    try:
        result = runner(
            ['git', 'rev-parse', '--verify', f'{revision}^{{commit}}'],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PackageIdentityResolutionError(f'git resolution timed out for {revision}') from exc
    except FileNotFoundError as exc:
        raise PackageIdentityResolutionError('git executable unavailable') from exc
    except OSError as exc:
        raise PackageIdentityResolutionError(f'git resolution OS error: {type(exc).__name__}') from exc
    except Exception as exc:
        raise PackageIdentityResolutionError(f'git resolution unexpected error: {type(exc).__name__}') from exc

    if getattr(result, 'returncode', None) != 0:
        raise PackageIdentityResolutionError(f'git rev-parse returned nonzero for {revision}')
    stdout = str(getattr(result, 'stdout', '') or '').strip()
    if not stdout:
        raise PackageIdentityResolutionError(f'git rev-parse returned empty output for {revision}')
    if not is_valid_git_commit(stdout):
        raise PackageIdentityResolutionError(f'git rev-parse returned malformed commit for {revision}')
    return stdout.lower()


def _read_jsonl_by_id(path: Path, label: str, errors: List[str]) -> Dict[str, Dict]:
    """Load a reviewer JSONL representation and fail closed on malformed identities."""
    rows: Dict[str, Dict] = {}
    if not path.exists():
        errors.append(f'{label} not found: {path}')
        return rows
    try:
        with path.open('r', encoding='utf-8') as fh:
            for number, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                eid = row.get('example_id', '')
                if not eid:
                    errors.append(f'{label}:{number}: missing example_id')
                elif eid in rows:
                    errors.append(f'{label}: duplicate example_id: {eid}')
                else:
                    rows[eid] = row
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f'cannot read {label}: {type(exc).__name__}')
    return rows


def _read_csv_by_id(path: Path, errors: List[str]) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    if not path.exists():
        errors.append(f'REVIEWER_TEMPLATE.csv not found: {path}')
        return rows
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            required = {'example_id', 'package_commit', 'content_risk_status', 'integrity_flags'}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                errors.append('REVIEWER_TEMPLATE.csv: required identity or risk columns missing')
                return rows
            for number, row in enumerate(reader, 2):
                eid = row.get('example_id', '')
                if not eid:
                    errors.append(f'REVIEWER_TEMPLATE.csv:{number}: missing example_id')
                elif eid in rows:
                    errors.append(f'REVIEWER_TEMPLATE.csv: duplicate example_id: {eid}')
                else:
                    rows[eid] = row
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f'cannot read REVIEWER_TEMPLATE.csv: {type(exc).__name__}')
    return rows


def _markdown_all_queue_ids(markdown: str) -> List[str]:
    """Extract IDs only from the All Queue Records table, avoiding the flagged summary."""
    marker = '## All Queue Records'
    if marker not in markdown:
        return []
    section = markdown.split(marker, 1)[1]
    return re.findall(r'^\\|\\s*\\d+\\s*\\|\\s*`([^`]+)`\\s*\\|', section, flags=re.MULTILINE)


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

# ─── Language contract helpers (Defect 4) ────────────────────────────────

def ar_char_count(text: str) -> int:
    return len(re.findall(r"[\u0600-\u06FF]", str(text)))


def en_char_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]", str(text)))


def recompute_language_contract(lang: str, prompt: str, response: str) -> Tuple[bool, Dict]:
    """
    Independently recompute language contract from raw text.
    Returns (passed: bool, evidence: dict).
    """
    ar_resp = ar_char_count(response)
    en_resp = en_char_count(response)
    combined_ar = ar_char_count(prompt + " " + response)
    combined_en = en_char_count(prompt + " " + response)
    resp_len = max(len(response.replace(" ", "")), 1)
    ar_ratio = ar_resp / resp_len

    evidence = {
        "ar_chars_response": ar_resp,
        "en_chars_response": en_resp,
        "ar_chars_combined": combined_ar,
        "en_chars_combined": combined_en,
        "ar_ratio_response": round(ar_ratio, 4),
    }

    if lang == "ar":
        passed = ar_resp >= 30 and ar_ratio >= 0.20
    elif lang == "en":
        passed = en_resp >= 30
    elif lang == "ar-en":
        passed = combined_ar >= 20 and combined_en >= 20
    else:
        passed = True  # Unknown language — not our contract to enforce

    evidence["recomputed_passed"] = passed
    return passed, evidence


# ─── Blocking content patterns ────────────────────────────────────────────

BLOCKING_CONTENT_PATTERNS = [
    (re.compile(r"100%\s+guaranteed", re.IGNORECASE), "100% guaranteed claim"),
    (re.compile(r"^\s*$"), "empty response"),
    (re.compile(r"^\s*\[placeholder\]\s*$", re.IGNORECASE), "placeholder text"),
    (re.compile(r"^\s*\[.*\]\s*$"), "response is only a placeholder bracket"),
]

# Module-level corpus constants (used by Gate 13, Gate 16, and test suite)
EXPECTED_CORPUS_HASH = "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0"

# V2.3: Corrected GCC_TRIP_PHRASES — removed erroneous $ end-anchor.
# Previous patterns used \.$  which is \. (literal period) + $ (end-of-string anchor).
# This meant phrases were only detected when they appeared at the very end of the string.
# Corrected: \. matches a sentence-ending period anywhere in the text.
# Regression tests in test_v23_regex.py verify anchor behaviour and punctuation boundaries.
GCC_TRIP_PHRASES = [
    re.compile(r"يُنصح بمراجعة[^\.]{0,50}\.", re.UNICODE),
    re.compile(r"specific current requirements should be verified[^\.]{0,60}\.", re.IGNORECASE),
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


# ─── Gate 3: UTF-8 portability (Defect 2 fix) ────────────────────────────

def _check_utf8_in_dir(py_dir: Path, label: str) -> List[str]:
    """
    AST-based check for text I/O without explicit encoding= in a directory.
    Covers: open(), Path.open(), .read_text(), .write_text()
    """
    violations = []
    for path in sorted(py_dir.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            kws = {kw.arg for kw in node.keywords if kw.arg is not None}
            # Detect: open(...), Path.open(...), .open(...)
            is_open_call = (
                (isinstance(func, ast.Name) and func.id == "open") or
                (isinstance(func, ast.Attribute) and func.attr == "open")
            )
            # Detect: .read_text(...), .write_text(...)
            is_text_method = (
                isinstance(func, ast.Attribute) and
                func.attr in {"read_text", "write_text"}
            )
            if (is_open_call or is_text_method) and "encoding" not in kws:
                violations.append(f"{label}/{path.name}:{node.lineno}")
    return violations


def gate_utf8(data_dir: Path) -> GateResult:
    """
    Defect 2 fix: covers both training/ and tests/ directories.
    Runs with PYTHONUTF8 and PYTHONIOENCODING unset.
    """
    errors = []
    warnings = []

    # Check JSONL files contain Arabic bytes that fail cp1252
    for split in ("train", "valid", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        content = path.read_bytes()
        has_arabic = any(b >= 0xC0 for b in content)
        if has_arabic:
            try:
                content.decode("cp1252")
                warnings.append(f"{split}.jsonl: cp1252 decode succeeded — verify Arabic content")
            except UnicodeDecodeError:
                pass  # Expected
        else:
            warnings.append(f"{split}.jsonl: no multi-byte content found — may lack Arabic")

    # Check production Python files (training/)
    training_dir = data_dir.parent
    errors.extend(_check_utf8_in_dir(training_dir, "training"))

    # Check test Python files (tests/jupiter_seed_4b/) — Defect 2 fix
    repo_root = training_dir.parent.parent
    tests_dir = repo_root / "tests" / "jupiter_seed_4b"
    if tests_dir.exists():
        errors.extend(_check_utf8_in_dir(tests_dir, "tests"))

    if errors:
        return GateResult(3, "UTF-8 Portability", FAIL,
                          f"{len(errors)} UTF-8 portability errors (training + tests)",
                          warnings=warnings, errors=errors[:20])
    status = WARN if warnings else PASS
    return GateResult(3, "UTF-8 Portability", status,
                      "All text I/O uses explicit encoding in training/ and tests/. "
                      "Arabic bytes fail cp1252 as expected.",
                      warnings=warnings)


# ─── Gate 4: Language contract (Defect 4 fix) ────────────────────────────

def gate_language_contract(records: List[Dict]) -> GateResult:
    """
    Defect 4 fix: independently recomputes language contracts from raw text.
    Does NOT trust stored language_contract_passed boolean.
    Fails when stored and recomputed values disagree.
    """
    errors = []
    disagreements = []

    for r in records:
        lang = r.get("language", "")
        prompt = r.get("prompt", "")
        response = r.get("response", "")
        stored = r.get("language_contract_passed", None)

        recomputed, evidence = recompute_language_contract(lang, prompt, response)

        # Fail if recomputed contract is violated
        if not recomputed:
            errors.append(
                f"{r['example_id']}: language={lang} "
                f"ar_resp={evidence['ar_chars_response']} "
                f"en_resp={evidence['en_chars_response']} "
                f"ar_ratio={evidence['ar_ratio_response']}"
            )

        # Fail if stored value disagrees with recomputed
        if stored is not None and bool(stored) != recomputed:
            disagreements.append(
                f"{r['example_id']}: stored={stored} recomputed={recomputed} "
                f"lang={lang} ar_resp={evidence['ar_chars_response']}"
            )

    all_errors = errors + disagreements
    if all_errors:
        return GateResult(4, "Language Contract", FAIL,
                          f"{len(errors)} contract violations, "
                          f"{len(disagreements)} stored/recomputed disagreements",
                          errors=all_errors[:10])
    return GateResult(4, "Language Contract", PASS,
                      f"All {len(records)} records satisfy independently recomputed language contracts")


# ─── Gate 5: Raw duplication ──────────────────────────────────────────────

def gate_raw_duplication(records: List[Dict]) -> GateResult:
    errors = []
    id_to_split = {r["example_id"]: r["split"] for r in records}

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

    brief_to_ids = defaultdict(list)
    for r in records:
        brief_to_ids[sha256_of(canonicalize(r.get("scenario_brief", "")))].append(r["example_id"])
    dup_briefs = [ids for ids in brief_to_ids.values() if len(ids) > 1]
    if dup_briefs:
        for ids in dup_briefs[:5]:
            errors.append(f"Duplicate scenario brief: {ids}")

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

    for fname in ["FROZEN_BENCHMARK_MANIFEST_v1.0.0.json",
                  "FROZEN_BENCHMARK_MANIFEST_v1.1.0.json",
                  "FROZEN_BENCHMARK_MANIFEST_v1.1.1.json"]:
        if not (benchmark_dir / fname).exists():
            warnings.append(f"Historical manifest not found: {fname}")

    if errors:
        return GateResult(10, "Benchmark Manifest Integrity", FAIL,
                          f"{len(errors)} manifest errors",
                          warnings=warnings, errors=errors)
    status = WARN if warnings else PASS
    return GateResult(10, "Benchmark Manifest Integrity", status,
                      f"Manifest version {version}, approval PENDING.",
                      warnings=warnings)


# ─── Gate 11: Test-suite result (Defect 1 fix) ───────────────────────────

def _compute_test_manifest_hash(test_dir: Path) -> Tuple[str, List[str]]:
    """Compute SHA-256 of the sorted list of authorized test file names."""
    found = []
    for fname in AUTHORIZED_TEST_FILES:
        p = test_dir / fname
        if p.exists():
            found.append(fname)
    manifest_str = "|".join(sorted(found))
    return hashlib.sha256(manifest_str.encode("utf-8")).hexdigest(), found


def _check_test_manifest(test_dir: Path) -> List[str]:
    """
    V2.2 Task 3: Full authorized-test-manifest enforcement.
    Returns a list of error strings. Empty list means all checks passed.
    Fails on:
      - missing authorized files
      - unexpected test files in the directory
      - duplicate entries in AUTHORIZED_TEST_FILES
      - manifest hash mismatch
    """
    errors = []

    # Check for duplicate entries in AUTHORIZED_TEST_FILES
    if len(AUTHORIZED_TEST_FILES) != len(set(AUTHORIZED_TEST_FILES)):
        from collections import Counter
        dupes = [f for f, c in Counter(AUTHORIZED_TEST_FILES).items() if c > 1]
        errors.append(f"Duplicate entries in AUTHORIZED_TEST_FILES: {dupes}")

    # Check all authorized files are present
    authorized_set = set(AUTHORIZED_TEST_FILES)
    missing = [f for f in AUTHORIZED_TEST_FILES if not (test_dir / f).exists()]
    if missing:
        errors.append(f"Missing authorized test files: {missing}")

    # Check for unexpected test files in the directory
    actual_files = {p.name for p in test_dir.glob("test_*.py")}
    unexpected = sorted(actual_files - authorized_set)
    if unexpected:
        errors.append(
            f"Unexpected test files not in authorized manifest: {unexpected}. "
            f"Add them to AUTHORIZED_TEST_FILES or remove them."
        )

    # Verify manifest hash matches AUTHORIZED_MANIFEST_HASH
    found = sorted([f for f in AUTHORIZED_TEST_FILES if (test_dir / f).exists()])
    computed_hash = hashlib.sha256("|".join(sorted(found)).encode("utf-8")).hexdigest()
    if computed_hash != AUTHORIZED_MANIFEST_HASH:
        errors.append(
            f"Manifest hash mismatch: computed={computed_hash} "
            f"expected={AUTHORIZED_MANIFEST_HASH}"
        )

    return errors


def gate_test_suite(repo_root: Path) -> GateResult:
    """
    Defect 1 fix:
    - Uses sys.executable, never 'python3' or 'python'
    - Runs pytest via sys.executable -m pytest
    - Uses JUnit XML for structured result parsing
    - Requires: exit_code=0, collected>0, passed>0, failed=0, errors=0
    - Includes test_adversarial_fixtures.py
    - Prevents recursive execution
    - Records interpreter path, Python version, pytest version
    - A missing/empty/malformed result file returns FAIL
    """
    # Prevent recursive execution: if we are already inside pytest, skip
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return GateResult(11, "Test-Suite Result", FAIL,
                          "Gate 11 cannot run inside pytest (recursive execution prevented)")

    test_dir = repo_root / "tests" / "jupiter_seed_4b"
    if not test_dir.exists():
        return GateResult(11, "Test-Suite Result", FAIL, "Test directory not found")

    # V2.2 Task 3: Full authorized-test-manifest enforcement
    manifest_errors = _check_test_manifest(test_dir)
    if manifest_errors:
        return GateResult(11, "Test-Suite Result", FAIL,
                          f"Authorized test manifest failed: {len(manifest_errors)} error(s)",
                          errors=manifest_errors)
    manifest_hash, found_files = _compute_test_manifest_hash(test_dir)

    # Record interpreter details
    interpreter = sys.executable
    python_version = sys.version.split()[0]

    # Get pytest version
    try:
        v_result = subprocess.run(
            [interpreter, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8",
        )
        pytest_version = v_result.stdout.strip() + v_result.stderr.strip()
    except Exception:
        pytest_version = "unknown"

    # Run pytest with JUnit XML output
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "results.xml"

        # Build clean environment: unset encoding overrides (Defect 2)
        env = dict(os.environ)
        env.pop("PYTHONUTF8", None)
        env.pop("PYTHONIOENCODING", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            result = subprocess.run(
                [interpreter, "-m", "pytest",
                 str(test_dir),
                 f"--junitxml={xml_path}",
                 "--tb=short",
                 "-q"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=180,
                cwd=str(repo_root),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return GateResult(11, "Test-Suite Result", FAIL,
                              "Test suite timed out after 180 seconds")
        except Exception as e:
            return GateResult(11, "Test-Suite Result", FAIL, f"subprocess error: {e}")

        end_ts = datetime.datetime.utcnow().isoformat() + "Z"
        exit_code = result.returncode

        # Parse JUnit XML (Defect 1: structured parsing, not text parsing)
        if not xml_path.exists() or xml_path.stat().st_size == 0:
            return GateResult(11, "Test-Suite Result", FAIL,
                              "JUnit XML result file missing or empty — cannot determine test outcome",
                              errors=[result.stdout[-500:] if result.stdout else "no output"])

        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            # pytest JUnit XML: <testsuites> or <testsuite>
            if root.tag == "testsuites":
                suites = list(root)
            else:
                suites = [root]

            collected = 0
            passed = 0
            failed_count = 0
            errors_count = 0
            skipped_count = 0

            for suite in suites:
                suite_tests = int(suite.get("tests", 0))
                suite_failures = int(suite.get("failures", 0))
                suite_errors = int(suite.get("errors", 0))
                suite_skipped = int(suite.get("skipped", 0))
                collected += suite_tests
                failed_count += suite_failures
                errors_count += suite_errors
                skipped_count += suite_skipped

            passed = collected - failed_count - errors_count - skipped_count

        except ET.ParseError as e:
            return GateResult(11, "Test-Suite Result", FAIL,
                              f"JUnit XML malformed: {e}",
                              errors=[result.stdout[-300:] if result.stdout else ""])

    # Defect 1: Gate 11 must NEVER pass when zero tests ran
    if collected == 0:
        return GateResult(11, "Test-Suite Result", FAIL,
                          "Zero tests collected — Gate 11 cannot PASS when no tests ran",
                          errors=[result.stdout[-300:] if result.stdout else "no output"])

    if passed == 0:
        return GateResult(11, "Test-Suite Result", FAIL,
                          f"Zero tests passed (collected={collected}) — Gate 11 cannot PASS",
                          errors=[result.stdout[-300:] if result.stdout else ""])

    metadata = {
        "interpreter": interpreter,
        "python_version": python_version,
        "pytest_version": pytest_version,
        "test_manifest_hash": manifest_hash,
        "authorized_files": AUTHORIZED_TEST_FILES,
        "found_files": found_files,
        "collected": collected,
        "passed": passed,
        "failed": failed_count,
        "errors": errors_count,
        "skipped": skipped_count,
        "exit_code": exit_code,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }

    if failed_count > 0 or errors_count > 0:
        return GateResult(11, "Test-Suite Result", FAIL,
                          f"collected={collected} passed={passed} failed={failed_count} "
                          f"errors={errors_count} skipped={skipped_count}",
                          errors=[result.stdout[-500:] if result.stdout else ""],
                          metadata=metadata)

    # V2.2 Task 2: enforce exit code — nonzero exit must FAIL even if XML parsed OK
    if exit_code != 0:
        return GateResult(11, "Test-Suite Result", FAIL,
                          f"pytest exit_code={exit_code} (nonzero) — Gate 11 cannot PASS",
                          errors=[result.stdout[-500:] if result.stdout else ""],
                          metadata=metadata)

    return GateResult(11, "Test-Suite Result", PASS,
                      f"collected={collected} passed={passed} failed=0 errors=0 "
                      f"skipped={skipped_count} exit_code=0",
                      metadata=metadata)


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


# ─── Gate 13: Corpus Integrity (V2.2 Task 5) ────────────────────────────

def gate_corpus_integrity(records: List[Dict], benchmark_dir: Path) -> GateResult:
    """
    V2.2 Task 5: Enforced corpus-integrity gate.
    Recomputes and verifies:
      - ordered corpus SHA-256
      - dataset content commit (from manifest)
      - record count
      - split counts
      - content-family assignment hash
      - split-assignment hash
      - per-record stored hashes against recomputed values
    Any stored-versus-recomputed disagreement fails readiness.
    Does NOT trust stored booleans or stored hashes without recomputation.
    """
    EXPECTED_CORPUS_HASH = "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0"
    EXPECTED_CONTENT_COMMIT = "2e36f6b977a8af052fced5a532c1168dc1988b6f"
    EXPECTED_RECORD_COUNT = 101
    EXPECTED_SPLITS = {"train": 68, "valid": 16, "eval": 17}

    errors = []
    warnings = []

    # 1. Record count
    if len(records) != EXPECTED_RECORD_COUNT:
        errors.append(
            f"Record count mismatch: expected={EXPECTED_RECORD_COUNT} actual={len(records)}"
        )

    # 2. Split counts
    actual_splits: Dict[str, int] = {}
    for r in records:
        s = r.get("split", "unknown")
        actual_splits[s] = actual_splits.get(s, 0) + 1
    for split_name, expected_count in EXPECTED_SPLITS.items():
        actual_count = actual_splits.get(split_name, 0)
        if actual_count != expected_count:
            errors.append(
                f"Split '{split_name}' count mismatch: expected={expected_count} actual={actual_count}"
            )

    # 3. Per-record stored hash vs recomputed
    # V2.3 Task 5A: blank or missing stored hash must be rejected (not silently skipped)
    hash_errors = []
    blank_hash_errors = []
    for r in records:
        eid = r.get("example_id", "?")
        stored_hash = r.get("content_sha256", "")
        # content_sha256 is sha256(prompt + "\n" + response) per build_dataset.py
        recomputed_hash = sha256_of(r.get("prompt", "") + "\n" + r.get("response", ""))

        # V2.3: Reject blank or missing stored hash
        if not stored_hash or not stored_hash.strip():
            blank_hash_errors.append(
                f"{eid}: content_sha256 is blank or missing — cannot verify integrity"
            )
        elif stored_hash != recomputed_hash:
            hash_errors.append(
                f"{eid}: stored={stored_hash[:16]}... recomputed={recomputed_hash[:16]}..."
            )
    if blank_hash_errors:
        errors.append(f"{len(blank_hash_errors)} record(s) have blank/missing content_sha256")
        errors.extend(blank_hash_errors[:5])
    if hash_errors:
        errors.append(f"{len(hash_errors)} per-record content hash mismatches")
        errors.extend(hash_errors[:5])

    # 4. Ordered corpus SHA-256
    records_sorted = sorted(records, key=lambda r: r.get("example_id", ""))
    prompt_hashes = [sha256_of(r.get("prompt", "")) for r in records_sorted]
    response_hashes = [sha256_of(r.get("response", "")) for r in records_sorted]
    brief_hashes = [sha256_of(r.get("scenario_brief", "") if isinstance(r.get("scenario_brief"), str) else str(r.get("scenario_brief", ""))) for r in records_sorted]
    ordered_str = "|".join(
        f"{p}:{r}:{b}"
        for p, r, b in zip(prompt_hashes, response_hashes, brief_hashes)
    )
    computed_corpus_hash = sha256_of(ordered_str)
    if computed_corpus_hash != EXPECTED_CORPUS_HASH:
        errors.append(
            f"Ordered corpus SHA-256 mismatch: "
            f"expected={EXPECTED_CORPUS_HASH} "
            f"computed={computed_corpus_hash}"
        )
    else:
        pass  # PASS

    # 5. Split-assignment hash
    split_str = "|".join(f"{r['example_id']}:{r['split']}" for r in records_sorted)
    computed_split_hash = sha256_of(split_str)

    # 6. Content-family assignment hash
    family_str = "|".join(
        f"{r['example_id']}:{r.get('content_family_id', '')}" for r in records_sorted
    )
    computed_family_hash = sha256_of(family_str)

    # 7. Verify against FROZEN_CORPUS_MANIFEST.json if it exists
    manifest_path = benchmark_dir / "FROZEN_CORPUS_MANIFEST.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            manifest_corpus_hash = manifest.get("ordered_corpus_sha256", "")
            if manifest_corpus_hash and manifest_corpus_hash != computed_corpus_hash:
                errors.append(
                    f"Corpus hash disagrees with FROZEN_CORPUS_MANIFEST.json: "
                    f"manifest={manifest_corpus_hash[:16]}... "
                    f"computed={computed_corpus_hash[:16]}..."
                )
            manifest_split_hash = manifest.get("split_assignment_sha256", "")
            if manifest_split_hash and manifest_split_hash != computed_split_hash:
                errors.append(
                    f"Split-assignment hash disagrees with manifest: "
                    f"manifest={manifest_split_hash[:16]}... "
                    f"computed={computed_split_hash[:16]}..."
                )
            manifest_family_hash = manifest.get("family_assignment_sha256", "")
            if manifest_family_hash and manifest_family_hash != computed_family_hash:
                errors.append(
                    f"Family-assignment hash disagrees with manifest: "
                    f"manifest={manifest_family_hash[:16]}... "
                    f"computed={computed_family_hash[:16]}..."
                )
            manifest_content_commit = manifest.get("dataset_content_commit", "")
            if manifest_content_commit and manifest_content_commit != EXPECTED_CONTENT_COMMIT:
                errors.append(
                    f"dataset_content_commit in manifest does not match expected: "
                    f"manifest={manifest_content_commit} "
                    f"expected={EXPECTED_CONTENT_COMMIT}"
                )
        except Exception as e:
            errors.append(f"Failed to read FROZEN_CORPUS_MANIFEST.json: {e}")
    else:
        warnings.append("FROZEN_CORPUS_MANIFEST.json not found — skipping manifest cross-check")

    if errors:
        return GateResult(13, "Corpus Integrity", FAIL,
                          f"{len(errors)} corpus integrity errors",
                          warnings=warnings, errors=errors[:15])

    return GateResult(13, "Corpus Integrity", PASS,
                      f"Corpus intact: {EXPECTED_RECORD_COUNT} records, "
                      f"ordered_sha256={computed_corpus_hash[:16]}..., "
                      f"content_commit={EXPECTED_CONTENT_COMMIT[:12]}...",
                      warnings=warnings,
                      metadata={
                          "ordered_corpus_sha256": computed_corpus_hash,
                          "split_assignment_sha256": computed_split_hash,
                          "family_assignment_sha256": computed_family_hash,
                          "dataset_content_commit": EXPECTED_CONTENT_COMMIT,
                          "record_count": len(records),
                          "split_counts": actual_splits,
                      })


# ─── Gate 14: Content-Risk Gate (V2.3) ──────────────────────────────────────

def gate_content_risk(records: List[Dict]) -> GateResult:
    """
    V2.3 Task 1: Production content-risk gate.
    Scans the actual corpus using BLOCKING_CONTENT_PATTERNS and GCC_TRIP_PHRASES.

    Severity A — BLOCKED:
      Any match on BLOCKING_CONTENT_PATTERNS (e.g. '100% guaranteed').
      Makes the final mechanical verdict NOT_READY.
      Prevents training authorization.

    Severity B — REVIEW_REQUIRED:
      Any match on GCC_TRIP_PHRASES (culturally sensitive phrases).
      Adds record to human-review attention list.
      Prevents training/release authorization.
      Does not silently disappear behind an overall PASS.

    Each finding records: example_id, field, rule, safe excerpt.
    """
    blocked_findings = []
    review_required_findings = []

    for r in records:
        eid = r.get("example_id", "?")
        for field in ("prompt", "response"):
            text = r.get(field, "")

            # Severity A: BLOCKED
            for pat, rule_name in BLOCKING_CONTENT_PATTERNS:
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    excerpt = text[start:end].replace("\n", " ")
                    blocked_findings.append({
                        "example_id": eid,
                        "field": field,
                        "rule": rule_name,
                        "excerpt": excerpt[:120],
                        "severity": "BLOCKED",
                    })

            # Severity B: REVIEW_REQUIRED
            for pat in GCC_TRIP_PHRASES:
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    excerpt = text[start:end].replace("\n", " ")
                    review_required_findings.append({
                        "example_id": eid,
                        "field": field,
                        "rule": repr(pat.pattern),
                        "excerpt": excerpt[:120],
                        "severity": "REVIEW_REQUIRED",
                    })

    all_findings = blocked_findings + review_required_findings

    if blocked_findings:
        return GateResult(
            14, "Content-Risk", BLOCKED,
            f"{len(blocked_findings)} BLOCKED finding(s) and "
            f"{len(review_required_findings)} REVIEW_REQUIRED finding(s). "
            f"Blocked records must not be used for training.",
            errors=[
                f"BLOCKED | {f['example_id']} [{f['field']}] | {f['rule']} | {repr(f['excerpt'])}"
                for f in blocked_findings
            ],
            warnings=[
                f"REVIEW_REQUIRED | {f['example_id']} [{f['field']}] | {f['rule'][:40]} | {repr(f['excerpt'])}"
                for f in review_required_findings
            ],
            metadata={
                "blocked_count": len(blocked_findings),
                "review_required_count": len(review_required_findings),
                "blocked_records": [f["example_id"] for f in blocked_findings],
                "review_required_records": list({f["example_id"] for f in review_required_findings}),
            },
        )

    if review_required_findings:
        return GateResult(
            14, "Content-Risk", REVIEW_REQUIRED,
            f"0 BLOCKED, {len(review_required_findings)} REVIEW_REQUIRED finding(s). "
            f"Records must appear in human-review queue before training/release authorization.",
            warnings=[
                f"REVIEW_REQUIRED | {f['example_id']} [{f['field']}] | {f['rule'][:40]} | {repr(f['excerpt'])}"
                for f in review_required_findings
            ],
            metadata={
                "blocked_count": 0,
                "review_required_count": len(review_required_findings),
                "review_required_records": list({f["example_id"] for f in review_required_findings}),
                "findings": review_required_findings,
            },
        )

    return GateResult(
        14, "Content-Risk", PASS,
        "No BLOCKED or REVIEW_REQUIRED content found in corpus.",
        metadata={"blocked_count": 0, "review_required_count": 0},
    )


# ─── Gate 15: Review-queue coverage (V2.3 Task 4) ─────────────────────────────

def gate_review_queue_coverage(
    records: List[Dict],
    data_dir: Path,
    content_risk_gate: GateResult,
) -> GateResult:
    """
    V2.3 Task 4: For every REVIEW_REQUIRED corpus record, verify it appears
    in the authoritative human-review queue JSONL.
    Returns exit-code 2 (MECHANICALLY_BLOCKED) if any required-review record
    is absent from the queue.
    Human judgment fields must remain empty.
    """
    # Identify REVIEW_REQUIRED records from Gate 14
    review_required_ids: set = set()
    if content_risk_gate.metadata:
        review_required_ids = set(
            content_risk_gate.metadata.get("review_required_records", [])
        )

    if not review_required_ids:
        return GateResult(
            15, "Review-Queue Coverage", PASS,
            "No REVIEW_REQUIRED records to verify.",
        )

    # Load the authoritative queue JSONL from the data directory
    queue_path = data_dir / "human_review_queue.jsonl"
    if not queue_path.exists():
        return GateResult(
            15, "Review-Queue Coverage", BLOCKED,
            "human_review_queue.jsonl not found — cannot verify coverage.",
            errors=[f"Queue file not found: {queue_path}"],
        )

    queue_ids: set = set()
    illegal_approvals = []
    with queue_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                qr = json.loads(line)
                queue_ids.add(qr.get("example_id", ""))
                # Check no human judgment fields are populated
                for jf in REVIEWER_JUDGMENT_FIELDS:
                    val = qr.get(jf, "")
                    if val and str(val).strip():
                        illegal_approvals.append(
                            f"{qr.get('example_id', '?')}: reviewer field '{jf}' is populated"
                        )
            except json.JSONDecodeError:
                pass

    missing = sorted(review_required_ids - queue_ids)
    errors = []
    warnings = []

    if missing:
        errors.append(
            f"{len(missing)} REVIEW_REQUIRED record(s) absent from human-review queue: {missing}"
        )

    if illegal_approvals:
        errors.append(
            f"{len(illegal_approvals)} reviewer judgment field(s) illegally populated by software"
        )
        errors.extend(illegal_approvals[:5])

    if errors:
        return GateResult(
            15, "Review-Queue Coverage", BLOCKED,
            f"Review-queue coverage failed: {len(errors)} error(s). Exit code 2 required.",
            errors=errors,
            warnings=warnings,
            metadata={
                "review_required_ids": sorted(review_required_ids),
                "queue_ids_found": sorted(queue_ids & review_required_ids),
                "missing_from_queue": missing,
            },
        )

    return GateResult(
        15, "Review-Queue Coverage", PASS,
        f"All {len(review_required_ids)} REVIEW_REQUIRED record(s) present in queue. "
        f"No reviewer judgment fields populated.",
        warnings=warnings,
        metadata={
            "review_required_ids": sorted(review_required_ids),
            "queue_coverage": "COMPLETE",
        },
    )


# ─── Gate 16: Review-Package Identity and Coverage (V2.4) ───────────────────────

def gate_review_package_identity(
    records: List[Dict],
    docs_dir: Path,
    data_dir: Path,
    content_risk_gate: GateResult,
) -> GateResult:
    """Fail-closed Gate 16 identity and representation verification (V2.4.1).

    Artifact `package_commit` uses the documented source-commit scheme: it must
    equal the direct parent of the artifact release commit (HEAD^). This avoids
    an impossible self-referential commit hash while guaranteeing that the
    artifacts were generated from the exact source tree that precedes them.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Identity resolution is independent and fail-closed. Never replace failure
    # with an empty string or permit a comparison bypass.
    try:
        release_commit = resolve_git_commit('HEAD')
        source_commit = resolve_git_commit('HEAD^')
    except PackageIdentityResolutionError as exc:
        return GateResult(
            16, 'Review-Package Identity', FAIL,
            f'Package identity resolution failed: {exc}', errors=[str(exc)]
        )

    sidecar_path = docs_dir / 'REVIEW_RISK_FINDINGS.json'
    try:
        with sidecar_path.open('r', encoding='utf-8') as fh:
            sidecar = json.load(fh)
    except FileNotFoundError:
        return GateResult(16, 'Review-Package Identity', FAIL,
                          'REVIEW_RISK_FINDINGS.json sidecar not found.',
                          errors=[f'Missing: {sidecar_path}'])
    except (OSError, json.JSONDecodeError) as exc:
        return GateResult(16, 'Review-Package Identity', FAIL,
                          f'Malformed or unreadable sidecar: {type(exc).__name__}',
                          errors=[type(exc).__name__])

    sidecar_commit = sidecar.get('package_commit')
    if not is_valid_git_commit(sidecar_commit):
        errors.append('Sidecar package_commit is missing, blank, or malformed')
    elif sidecar_commit.lower() != source_commit:
        errors.append(
            f'Sidecar package_commit mismatch: sidecar={sidecar_commit[:12]}... '
            f'expected artifact-source={source_commit[:12]}...'
        )
    if sidecar.get('corpus_sha256') != EXPECTED_CORPUS_HASH:
        errors.append('Sidecar corpus_sha256 mismatch')

    review_required_ids = set(content_risk_gate.metadata.get('review_required_records', []))
    sidecar_findings: Dict[str, Dict] = {}
    for finding in sidecar.get('findings', []):
        eid = finding.get('record_id', '')
        if not eid:
            errors.append('Sidecar finding missing record_id')
        elif eid in sidecar_findings:
            errors.append(f'Duplicate sidecar finding for record: {eid}')
        else:
            sidecar_findings[eid] = finding
    if set(sidecar_findings) != review_required_ids:
        errors.append(
            f'Sidecar finding IDs mismatch Gate 14: sidecar={sorted(sidecar_findings)} '
            f'gate14={sorted(review_required_ids)}'
        )

    queue_rows = _read_jsonl_by_id(data_dir / 'human_review_queue.jsonl', 'human_review_queue.jsonl', errors)
    package_rows = _read_jsonl_by_id(docs_dir / 'REVIEWER_PACKAGE.jsonl', 'REVIEWER_PACKAGE.jsonl', errors)
    csv_rows = _read_csv_by_id(docs_dir / 'REVIEWER_TEMPLATE.csv', errors)
    queue_ids = set(queue_rows)

    for label, rows in (('REVIEWER_PACKAGE.jsonl', package_rows), ('REVIEWER_TEMPLATE.csv', csv_rows)):
        if set(rows) != queue_ids:
            errors.append(
                f'{label} ID set mismatch: missing={sorted(queue_ids - set(rows))[:5]} '
                f'unexpected={sorted(set(rows) - queue_ids)[:5]}'
            )
        for eid, row in rows.items():
            commit = row.get('package_commit')
            if not is_valid_git_commit(commit):
                errors.append(f'{label}:{eid}: package_commit is missing, blank, or malformed')
            elif commit.lower() != source_commit:
                errors.append(f'{label}:{eid}: package_commit mismatch')

    markdown_path = docs_dir / 'ARABIC_HUMAN_REVIEW_QUEUE.md'
    try:
        markdown = markdown_path.read_text(encoding='utf-8')
    except OSError as exc:
        markdown = ''
        errors.append(f'cannot read ARABIC_HUMAN_REVIEW_QUEUE.md: {type(exc).__name__}')
    matches = re.findall(r'^> Package source commit:\s*`([0-9a-fA-F]{40})`\s*$', markdown, flags=re.MULTILINE)
    if len(matches) != 1:
        errors.append('Markdown package_commit is missing, malformed, or duplicated')
    elif matches[0].lower() != source_commit:
        errors.append('Markdown package_commit mismatch')
    markdown_ids = _markdown_all_queue_ids(markdown)
    if len(markdown_ids) != len(set(markdown_ids)) or set(markdown_ids) != queue_ids:
        errors.append('Markdown queue-ID set is missing, duplicated, or unexpected')

    # Validate risk-state and reason identity across sidecar, JSONL, CSV, and Markdown.
    for eid in review_required_ids:
        finding = sidecar_findings.get(eid, {})
        if eid not in package_rows or eid not in csv_rows:
            continue
        pkg, row = package_rows[eid], csv_rows[eid]
        for label, representation in (('REVIEWER_PACKAGE.jsonl', pkg), ('REVIEWER_TEMPLATE.csv', row)):
            if representation.get('content_risk_status') != REVIEW_REQUIRED:
                errors.append(f'{label}:{eid}: flagged record risk status must be REVIEW_REQUIRED')
            if str(representation.get('reviewer_attention_required')).lower() != 'true':
                errors.append(f'{label}:{eid}: flagged record attention must be true')
            if representation.get('integrity_flags') != 'REVIEW_REQUIRED_CONTENT_RISK':
                errors.append(f'{label}:{eid}: flagged record integrity_flags invalid')
            field_map = {
                'content_risk_rule_id': 'matched_rule_id',
                'content_risk_rule_description': 'matched_rule_description',
                'content_risk_field': 'matched_field',
                'content_risk_excerpt': 'safe_excerpt',
            }
            for out_field, sidecar_field in field_map.items():
                if representation.get(out_field) != finding.get(sidecar_field):
                    errors.append(f'{label}:{eid}: {out_field} disagrees with sidecar')
        if eid not in markdown or finding.get('matched_rule_id', '') not in markdown or finding.get('safe_excerpt', '')[:30] not in markdown:
            errors.append(f'Markdown missing risk disclosure for {eid}')

    for eid in queue_ids - review_required_ids:
        for label, rows in (('REVIEWER_PACKAGE.jsonl', package_rows), ('REVIEWER_TEMPLATE.csv', csv_rows)):
            row = rows.get(eid, {})
            if row.get('content_risk_status') != 'NONE':
                errors.append(f'{label}:{eid}: unflagged risk status must be NONE')
            if str(row.get('reviewer_attention_required')).lower() != 'false':
                errors.append(f'{label}:{eid}: unflagged attention must be false')

    for eid, row in package_rows.items():
        for field in REVIEWER_JUDGMENT_FIELDS:
            if str(row.get(field, '')).strip():
                errors.append(f'REVIEWER_PACKAGE.jsonl:{eid}: reviewer field {field} populated')
    for eid, row in csv_rows.items():
        for field in REVIEWER_JUDGMENT_FIELDS:
            if str(row.get(field, '')).strip():
                errors.append(f'REVIEWER_TEMPLATE.csv:{eid}: reviewer field {field} populated')

    if errors:
        return GateResult(
            16, 'Review-Package Identity', FAIL,
            f'{len(errors)} identity/coverage error(s) in reviewer-facing files.',
            errors=errors[:20], warnings=warnings,
            metadata={'artifact_release_commit': release_commit,
                      'artifact_source_commit': source_commit,
                      'review_required_ids': sorted(review_required_ids)},
        )
    return GateResult(
        16, 'Review-Package Identity', PASS,
        f'All reviewer-facing representations are consistent. '
        f'{len(review_required_ids)} flagged record(s) correctly disclosed. '
        f'Artifact source commit verified against HEAD^.',
        warnings=warnings,
        metadata={'artifact_release_commit': release_commit,
                  'artifact_source_commit': source_commit,
                  'sidecar_package_commit': sidecar_commit,
                  'review_required_ids': sorted(review_required_ids)},
    )


# ─── Report writer ────────────────────────────────────────────────────────

def write_report(gates: List[GateResult], output_path: Path,
                 honest_count: Dict, dataset_version: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Gate 12 (human review) is expected NOT_READY.
    # Gates 14/15 may be BLOCKED or REVIEW_REQUIRED (not a FAIL, but blocks authorization).
    # All other gates must PASS for mechanical readiness.
    gate12 = next((g for g in gates if g.gate_id == 12), None)
    gate14 = next((g for g in gates if g.gate_id == 14), None)
    gate15 = next((g for g in gates if g.gate_id == 15), None)

    has_fail = any(g.status == FAIL for g in gates)
    has_blocked = any(g.status == BLOCKED for g in gates)
    has_review_req = any(g.status == REVIEW_REQUIRED for g in gates)
    all_mechanical_pass = all(
        g.passed for g in gates
        if g.gate_id not in (12, 14, 15)  # 12=human-review, 14/15=content-risk
        # Gate 16 is a FAIL gate (not REVIEW_REQUIRED), so it is included in has_fail check
    )

    if has_fail:
        final_verdict = "MECHANICAL_FAILURE — HUMAN REVIEW MUST NOT BEGIN"
    elif has_blocked:
        final_verdict = "HUMAN_REVIEW_REQUIRED — BLOCKED CONTENT MUST BE RESOLVED BEFORE HUMAN REVIEW"
    elif has_review_req and all_mechanical_pass:
        rr_ids = gate14.metadata.get('review_required_records', []) if gate14 else []
        final_verdict = (
            f"HUMAN_REVIEW_REQUIRED — {len(rr_ids)} REVIEW_REQUIRED RECORD(S) PENDING HUMAN JUDGMENT: "
            f"{sorted(rr_ids)}"
        )
    elif all_mechanical_pass and gate12 and gate12.status == NOT_READY:
        final_verdict = "MECHANICALLY_READY_FOR_HUMAN_REVIEW — GATE 12 PENDING HUMAN JUDGMENT"
    else:
        final_verdict = "JUPITER SEED 4B NOT READY"

    lines = [
        "# Jupiter Seed 4B: Dataset Readiness Gate Report",
        "",
        f"**Dataset version:** {dataset_version}",
        f"**Approval status:** PENDING INDEPENDENT AUDIT AND HUMAN REVIEW",
        "",
        "## Honest Dataset Size",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Authorized target | {honest_count.get('authorized', 'N/A')} |",
        f"| Actual distinct examples | {honest_count.get('actual', 'N/A')} |",
        f"| Shortfall | {honest_count.get('shortfall', 'N/A')} |",
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
        if g.metadata:
            lines.append("")
            lines.append("**Metadata:**")
            for k, v in g.metadata.items():
                lines.append(f"- `{k}`: `{v}`")
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
        "## Final Verdict",
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
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Dataset Readiness Gate v2.1")
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


def _write_execution_error_artifact(
    stage: str,
    exc: Exception,
    exc_category: str,
    docs_dir: Path,
) -> None:
    """Write a machine-readable execution-error artifact before exiting with code 3."""
    import traceback as _tb
    try:
        import subprocess as _sp
        result = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5
        )
        commit = result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
    except Exception:
        commit = "UNKNOWN"

    artifact = {
        "outcome": "EXECUTION_ERROR",
        "exit_code": EXIT_EXECUTION_ERROR,
        "exception_category": exc_category,
        "safe_error_message": str(exc)[:500],
        "failing_stage": stage,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": commit,
    }
    # Internal diagnostic (full traceback, not reviewer-facing)
    internal = dict(artifact)
    internal["full_traceback"] = _tb.format_exc()[:4000]

    try:
        docs_dir.mkdir(parents=True, exist_ok=True)
        public_path = docs_dir / "EXECUTION_ERROR_ARTIFACT.json"
        with public_path.open("w", encoding="utf-8") as fh:
            json.dump(artifact, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        internal_path = docs_dir / "EXECUTION_ERROR_INTERNAL.json"
        with internal_path.open("w", encoding="utf-8") as fh:
            json.dump(internal, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        log.info("Execution-error artifact written to %s", public_path)
    except Exception as write_exc:
        log.error("Could not write execution-error artifact: %s", write_exc)


def main() -> None:
    # Parse args first so we know where to write the error artifact
    try:
        _args = parse_args()
        _docs_dir = _args.docs_dir
    except Exception:
        _docs_dir = Path("docs/jupiter_seed_4b")

    try:
        _main_inner()
    except SystemExit:
        raise  # Let sys.exit() pass through normally
    except json.JSONDecodeError as e:
        log.error("EXECUTION_ERROR: malformed JSON: %s", e)
        _write_execution_error_artifact("json_parsing", e, "JSONDecodeError", _docs_dir)
        sys.exit(EXIT_EXECUTION_ERROR)
    except FileNotFoundError as e:
        log.error("EXECUTION_ERROR: missing required file: %s", e)
        _write_execution_error_artifact("file_loading", e, "FileNotFoundError", _docs_dir)
        sys.exit(EXIT_EXECUTION_ERROR)
    except PermissionError as e:
        log.error("EXECUTION_ERROR: unreadable file: %s", e)
        _write_execution_error_artifact("file_loading", e, "PermissionError", _docs_dir)
        sys.exit(EXIT_EXECUTION_ERROR)
    except subprocess.SubprocessError as e:
        log.error("EXECUTION_ERROR: subprocess failure: %s", e)
        _write_execution_error_artifact("subprocess", e, "SubprocessError", _docs_dir)
        sys.exit(EXIT_EXECUTION_ERROR)
    except Exception as e:
        log.error("EXECUTION_ERROR: unexpected exception before valid verdict: %s", e)
        _write_execution_error_artifact("unexpected", e, type(e).__name__, _docs_dir)
        sys.exit(EXIT_EXECUTION_ERROR)


def _main_inner() -> None:
    args = parse_args()
    records = load_all(args.data_dir)

    if not records:
        log.error("No records found in %s", args.data_dir)
        sys.exit(EXIT_MECHANICAL_FAILURE)

    log.info("Loaded %d records from %s", len(records), args.data_dir)

    authorized = 850
    actual = len(records)
    shortfall = max(0, authorized - actual)
    honest_count = {
        "authorized": authorized,
        "actual": actual,
        "shortfall": shortfall,
    }

    dataset_version = records[0].get("dataset_version", "unknown")

    # Run all gates
    g_content_risk = gate_content_risk(records)  # V2.3 Gate 14
    g_review_coverage = gate_review_queue_coverage(  # V2.3 Gate 15
        records, args.data_dir, g_content_risk
    )
    g_review_pkg_identity = gate_review_package_identity(  # V2.4 Gate 16
        records, args.docs_dir, args.data_dir, g_content_risk
    )

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
        gate_corpus_integrity(records, args.benchmark_dir),  # V2.2 Gate 13
        g_content_risk,                                      # V2.3 Gate 14
        g_review_coverage,                                   # V2.3 Gate 15
        g_review_pkg_identity,                               # V2.4 Gate 16
    ]

    verdict = write_report(gates, args.output, honest_count, dataset_version)

    passed = sum(1 for g in gates if g.passed)
    failed = sum(1 for g in gates if g.status == FAIL)
    not_ready = sum(1 for g in gates if g.status == NOT_READY)
    warned = sum(1 for g in gates if g.status == WARN)
    blocked = sum(1 for g in gates if g.status == BLOCKED)
    review_req = sum(1 for g in gates if g.status == REVIEW_REQUIRED)

    log.info("=== READINESS GATE SUMMARY ===")
    log.info("Gates passed: %d / 16", passed)
    log.info("Gates failed: %d", failed)
    log.info("Gates blocked: %d", blocked)
    log.info("Gates review_required: %d", review_req)
    log.info("Gates not_ready: %d", not_ready)
    log.info("Gates warned: %d", warned)
    log.info("Honest dataset size: %d / %d (shortfall: %d)", actual, authorized, shortfall)
    log.info("Final verdict: %s", verdict)

    # V2.3 Task 3: Unambiguous 4-code exit contract
    # Exit 1: any mechanical gate FAIL
    if failed > 0:
        log.info("Exit code: %d (MECHANICAL_FAILURE)", EXIT_MECHANICAL_FAILURE)
        sys.exit(EXIT_MECHANICAL_FAILURE)

    # Exit 2: any BLOCKED or REVIEW_REQUIRED gate (content risk or queue coverage)
    if blocked > 0 or review_req > 0:
        log.info("Exit code: %d (HUMAN_REVIEW_REQUIRED)", EXIT_HUMAN_REVIEW_REQUIRED)
        sys.exit(EXIT_HUMAN_REVIEW_REQUIRED)

    # Exit 0: all mechanical gates pass; Gate 12 pending human judgment (expected NOT_READY)
    log.info("Exit code: %d (MECHANICALLY_READY_FOR_HUMAN_REVIEW)", EXIT_MECHANICALLY_READY)
    sys.exit(EXIT_MECHANICALLY_READY)


if __name__ == "__main__":
    main()
