"""
Jupiter Seed 4B — Review Package Generator
============================================
Generates all reviewer-facing artifacts from two frozen sources:
  1. training/jupiter_seed_4b/data/human_review_queue.jsonl  (frozen corpus)
  2. Gate 14 content-risk findings (computed at generation time)

Outputs:
  docs/jupiter_seed_4b/REVIEW_RISK_FINDINGS.json      (immutable sidecar)
  docs/jupiter_seed_4b/REVIEWER_PACKAGE.jsonl         (joined representation)
  docs/jupiter_seed_4b/REVIEWER_TEMPLATE.csv          (UTF-8 BOM for Excel)
  docs/jupiter_seed_4b/ARABIC_HUMAN_REVIEW_QUEUE.md   (Markdown index)

Rules:
  - Frozen corpus files are never modified.
  - Human judgment fields are never populated by software.
  - content_risk_status = REVIEW_REQUIRED for flagged records.
  - content_risk_status = CLEAR for unflagged records.
  - integrity_flags = REVIEW_REQUIRED_CONTENT_RISK for flagged records.
  - integrity_flags = OK for unflagged records.

Usage:
  python3 training/jupiter_seed_4b/generate_review_package.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
DOCS_DIR = REPO_ROOT / "docs" / "jupiter_seed_4b"
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"

sys.path.insert(0, str(TRAINING_DIR))
from readiness_gate import (
    gate_content_risk,
    REVIEW_REQUIRED,
    BLOCKED,
    GCC_TRIP_PHRASES,
    BLOCKING_CONTENT_PATTERNS,
    REVIEWER_JUDGMENT_FIELDS,
)

SIDECAR_SCHEMA_VERSION = "1.0"
EXPECTED_CORPUS_SHA256 = (
    "123fbdaf47a1e6befdb8f3c55ac5f9c5df01d2b473bbdec4417e3c3bfce5ccd0"
)

REVIEWER_FACING_FIELDS = [
    "content_risk_status",
    "content_risk_severity",
    "content_risk_rule_id",
    "content_risk_rule_description",
    "content_risk_field",
    "content_risk_excerpt",
    "reviewer_attention_required",
    "integrity_flags",
]


# ─── Helpers ──────────────────────────────────────────────────────────────

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def load_queue() -> List[Dict]:
    path = DATA_DIR / "human_review_queue.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"human_review_queue.jsonl not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {i} of human_review_queue.jsonl: {e}")
    return records


def load_all_corpus() -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{split}.jsonl not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def build_rule_id(pat_pattern: str, is_arabic: bool) -> str:
    if is_arabic:
        return "GCC_TRIP_AR"
    if "specific current requirements" in pat_pattern.lower():
        return "GCC_TRIP_EN"
    if "100%" in pat_pattern:
        return "BLOCKING_100PCT_GUARANTEED"
    if "placeholder" in pat_pattern.lower():
        return "BLOCKING_PLACEHOLDER"
    return "BLOCKING_CONTENT"


def build_rule_description(rule_id: str) -> str:
    descriptions = {
        "GCC_TRIP_AR": (
            "Arabic GCC trip phrase: يُنصح بمراجعة — "
            "advisory language that may imply the model is uncertain about current regulations. "
            "Requires human reviewer judgment on accuracy and appropriateness."
        ),
        "GCC_TRIP_EN": (
            "English GCC trip phrase: 'specific current requirements should be verified' — "
            "advisory language that may imply the model is uncertain about current regulations. "
            "Requires human reviewer judgment on accuracy and appropriateness."
        ),
        "BLOCKING_100PCT_GUARANTEED": (
            "Prohibited claim: '100% guaranteed' — "
            "objectively false financial guarantee claim. Must not be used for training."
        ),
        "BLOCKING_PLACEHOLDER": (
            "Placeholder text detected — response is not a real answer. "
            "Must not be used for training."
        ),
        "BLOCKING_CONTENT": (
            "Blocked content pattern detected. Must not be used for training."
        ),
    }
    return descriptions.get(rule_id, "Unknown rule")


def compute_content_risk_fields(
    record: Dict,
    findings_by_id: Dict[str, List[Dict]],
) -> Dict:
    """Return content-risk fields for a single record."""
    eid = record.get("example_id", "")
    findings = findings_by_id.get(eid, [])

    if not findings:
        return {
            "content_risk_status": "CLEAR",
            "content_risk_severity": "NONE",
            "content_risk_rule_id": "",
            "content_risk_rule_description": "",
            "content_risk_field": "",
            "content_risk_excerpt": "",
            "reviewer_attention_required": False,
            "integrity_flags": "OK",
        }

    # Use the first (most severe) finding
    f = findings[0]
    return {
        "content_risk_status": f["severity"],
        "content_risk_severity": f["severity"],
        "content_risk_rule_id": f["rule_id"],
        "content_risk_rule_description": f["rule_description"],
        "content_risk_field": f["field"],
        "content_risk_excerpt": f["excerpt"],
        "reviewer_attention_required": True,
        "integrity_flags": "REVIEW_REQUIRED_CONTENT_RISK",
    }


# ─── Main generation logic ─────────────────────────────────────────────────

def generate_sidecar(
    corpus_records: List[Dict],
    package_commit: str,
    output_path: Path,
) -> Dict[str, List[Dict]]:
    """
    Run Gate 14 on the corpus, build the REVIEW_RISK_FINDINGS.json sidecar,
    and return a dict mapping example_id → list of findings.
    """
    gate14 = gate_content_risk(corpus_records)

    findings_list = []
    findings_by_id: Dict[str, List[Dict]] = {}

    # Process all findings from Gate 14 metadata
    raw_findings = gate14.metadata.get("findings", [])

    # Also collect BLOCKED findings from errors
    for r in corpus_records:
        eid = r.get("example_id", "")
        for field in ("prompt", "response"):
            text = r.get(field, "")

            # BLOCKED patterns
            for pat, rule_name in BLOCKING_CONTENT_PATTERNS:
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    excerpt = text[start:end].replace("\n", " ")
                    rule_id = build_rule_id(pat.pattern, False)
                    finding = {
                        "record_id": eid,
                        "severity": "BLOCKED",
                        "status": "BLOCKED",
                        "source_gate": "gate_14_content_risk",
                        "matched_rule_id": rule_id,
                        "matched_rule_description": build_rule_description(rule_id),
                        "matched_field": field,
                        "safe_excerpt": excerpt[:120],
                        "reviewer_instruction": (
                            "This record contains objectively prohibited content. "
                            "It must NOT be used for training, validation, or evaluation. "
                            "Mark as REJECT."
                        ),
                        # Internal fields for cross-representation identity
                        "rule_id": rule_id,
                        "rule_description": build_rule_description(rule_id),
                        "field": field,
                        "excerpt": excerpt[:120],
                    }
                    findings_list.append(finding)
                    findings_by_id.setdefault(eid, []).append(finding)

            # REVIEW_REQUIRED patterns (GCC trip phrases)
            for i, pat in enumerate(GCC_TRIP_PHRASES):
                m = pat.search(text)
                if m:
                    start = max(0, m.start() - 30)
                    end = min(len(text), m.end() + 30)
                    excerpt = text[start:end].replace("\n", " ")
                    is_arabic = i == 0  # Pattern 0 is Arabic
                    rule_id = build_rule_id(pat.pattern, is_arabic)
                    finding = {
                        "record_id": eid,
                        "severity": "REVIEW_REQUIRED",
                        "status": "REVIEW_REQUIRED",
                        "source_gate": "gate_14_content_risk",
                        "matched_rule_id": rule_id,
                        "matched_rule_description": build_rule_description(rule_id),
                        "matched_field": field,
                        "safe_excerpt": excerpt[:120],
                        "reviewer_instruction": (
                            "This record contains a GCC regulatory advisory phrase. "
                            "Please verify that the advice is accurate, current, and appropriate "
                            "for the target audience. Mark ACCEPT if correct, REVISE if wording "
                            "needs adjustment, or REJECT if the advice is misleading."
                        ),
                        # Internal fields for cross-representation identity
                        "rule_id": rule_id,
                        "rule_description": build_rule_description(rule_id),
                        "field": field,
                        "excerpt": excerpt[:120],
                    }
                    findings_list.append(finding)
                    findings_by_id.setdefault(eid, []).append(finding)

    # Deduplicate by (record_id, rule_id, field)
    seen = set()
    deduped = []
    for f in findings_list:
        key = (f["record_id"], f["matched_rule_id"], f["matched_field"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    findings_list = deduped
    # Rebuild findings_by_id from deduped
    findings_by_id = {}
    for f in findings_list:
        findings_by_id.setdefault(f["record_id"], []).append(f)

    # Build sidecar document
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "package_commit": package_commit,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_findings": len(findings_list),
        "blocked_count": sum(1 for f in findings_list if f["severity"] == "BLOCKED"),
        "review_required_count": sum(
            1 for f in findings_list if f["severity"] == "REVIEW_REQUIRED"
        ),
        "findings": [
            {
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "package_commit": package_commit,
                "corpus_sha256": EXPECTED_CORPUS_SHA256,
                "record_id": f["record_id"],
                "severity": f["severity"],
                "status": f["status"],
                "source_gate": f["source_gate"],
                "matched_rule_id": f["matched_rule_id"],
                "matched_rule_description": f["matched_rule_description"],
                "matched_field": f["matched_field"],
                "safe_excerpt": f["safe_excerpt"],
                "reviewer_instruction": f["reviewer_instruction"],
                "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            for f in findings_list
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Sidecar written: {output_path} ({len(findings_list)} findings)")
    return findings_by_id


def generate_reviewer_package(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    output_path: Path,
) -> None:
    """Generate REVIEWER_PACKAGE.jsonl — joined representation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in queue_records:
            risk_fields = compute_content_risk_fields(r, findings_by_id)
            row = dict(r)
            row.update(risk_fields)
            # Ensure all human judgment fields are empty
            for jf in REVIEWER_JUDGMENT_FIELDS:
                row[jf] = ""
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Reviewer package written: {output_path} ({len(queue_records)} records)")


def generate_reviewer_csv(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    output_path: Path,
) -> None:
    """Generate REVIEWER_TEMPLATE.csv — UTF-8 BOM for Excel."""
    if not queue_records:
        return

    # Build column order: identity fields, content-risk fields, human judgment fields
    identity_cols = [
        "example_id", "split", "language", "domain", "task_type",
        "difficulty", "provenance_type", "content_family_id",
    ]
    content_cols = [
        "prompt", "response",
        "scenario_brief",
    ]
    risk_cols = REVIEWER_FACING_FIELDS
    judgment_cols = list(REVIEWER_JUDGMENT_FIELDS)

    all_cols = identity_cols + content_cols + risk_cols + judgment_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for r in queue_records:
            risk_fields = compute_content_risk_fields(r, findings_by_id)
            row = {col: r.get(col, "") for col in all_cols}
            row.update(risk_fields)
            # Ensure all human judgment fields are empty
            for jf in judgment_cols:
                row[jf] = ""
            writer.writerow(row)
    print(f"Reviewer CSV written: {output_path} ({len(queue_records)} records)")


def generate_markdown(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    output_path: Path,
) -> None:
    """Generate ARABIC_HUMAN_REVIEW_QUEUE.md — Markdown index with content-risk disclosure."""
    flagged_ids = set(findings_by_id.keys())

    lines = [
        "# Jupiter Seed 4B — Arabic Human Review Queue",
        "",
        "> **AI-ASSISTED CONTENT AUDIT — NOT HUMAN APPROVAL**",
        "> This document is generated from the frozen corpus and Gate 14 findings.",
        "> Human reviewers must independently judge every record.",
        "> Software has not populated any reviewer judgment field.",
        "",
        "## Coverage",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Total records | {len(queue_records)} |",
        f"| Flagged (REVIEW_REQUIRED) | {len(flagged_ids)} |",
        f"| Clear | {len(queue_records) - len(flagged_ids)} |",
        "",
        "## Flagged Records — Require Reviewer Attention",
        "",
        "The following records were flagged by Gate 14 (Content-Risk).",
        "Reviewers must explicitly judge each flagged record.",
        "",
        "| Example ID | Field | Rule | Severity | Excerpt |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for eid, findings in sorted(findings_by_id.items()):
        for f in findings:
            excerpt_safe = f["excerpt"].replace("|", "\\|")[:80]
            lines.append(
                f"| `{eid}` | `{f['field']}` | `{f['rule_id']}` "
                f"| **{f['severity']}** | {excerpt_safe} |"
            )

    lines += [
        "",
        "## All Queue Records",
        "",
        "| # | Example ID | Split | Lang | Domain | Risk Status | Attention | Prompt (preview) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for i, r in enumerate(queue_records, 1):
        eid = r.get("example_id", "")
        risk_fields = compute_content_risk_fields(r, findings_by_id)
        status = risk_fields["content_risk_status"]
        attention = "**YES**" if risk_fields["reviewer_attention_required"] else "no"
        prompt_preview = str(r.get("prompt", ""))[:60].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | `{eid}` | {r.get('split','')} | {r.get('language','')} "
            f"| {r.get('domain','')} | {status} | {attention} | {prompt_preview}… |"
        )

    lines += [
        "",
        "## Reviewer Instructions",
        "",
        "For each record, complete the `REVIEWER_TEMPLATE.csv` columns:",
        "- `reviewer_name`: your full name",
        "- `accuracy_score`: 1–5",
        "- `fluency_score`: 1–5",
        "- `gcc_appropriateness_score`: 1–5",
        "- `domain_terminology_score`: 1–5",
        "- `factuality_score`: 1–5",
        "- `verdict`: ACCEPT / REVISE / REJECT",
        "- `corrected_wording`: if REVISE",
        "- `rejection_reason`: if REJECT",
        "- `comments`: optional",
        "",
        "**Flagged records** (`REVIEW_REQUIRED`) require explicit judgment on the flagged phrase.",
        "",
        "Software must never populate these fields.",
        "",
        "---",
        "",
        "Generated by `generate_review_package.py` — AI-ASSISTED INTERNAL AUDIT",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Markdown index written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Review Package Generator"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args()

    package_commit = get_current_commit()
    print(f"Package commit: {package_commit}")

    # Load frozen queue (never modified)
    queue_records = load_queue()
    print(f"Loaded {len(queue_records)} queue records")

    # Load full corpus for Gate 14 scan
    corpus_records = load_all_corpus()
    print(f"Loaded {len(corpus_records)} corpus records")

    # Generate sidecar
    sidecar_path = args.docs_dir / "REVIEW_RISK_FINDINGS.json"
    findings_by_id = generate_sidecar(corpus_records, package_commit, sidecar_path)

    # Generate reviewer-facing representations
    generate_reviewer_package(
        queue_records, findings_by_id,
        args.docs_dir / "REVIEWER_PACKAGE.jsonl"
    )
    generate_reviewer_csv(
        queue_records, findings_by_id,
        args.docs_dir / "REVIEWER_TEMPLATE.csv"
    )
    generate_markdown(
        queue_records, findings_by_id,
        args.docs_dir / "ARABIC_HUMAN_REVIEW_QUEUE.md"
    )

    print("\nReview package generation complete.")
    print(f"Flagged records: {sorted(findings_by_id.keys())}")


if __name__ == "__main__":
    main()
