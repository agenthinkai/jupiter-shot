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
  - content_risk_status = NONE for unflagged records.
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
    derive_effective_review_set,
    REVIEWER_JUDGMENT_FIELDS,
    is_valid_git_commit,
    resolve_git_commit,
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
            "content_risk_status": "NONE",
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
    base_queue_records: List[Dict],
    package_commit: str,
    output_path: Path,
) -> tuple[Dict[str, List[Dict]], Dict]:
    """Build sidecar from production Gate 14 and the deterministic effective review-set policy."""
    gate14 = gate_content_risk(corpus_records)
    raw_findings = gate14.metadata.get("findings", [])
    findings_by_id: Dict[str, List[Dict]] = {}
    findings_list: List[Dict] = []
    for raw in raw_findings:
        eid = raw.get("example_id", "")
        rule_id = raw.get("rule_id", "")
        finding = {
            "record_id": eid,
            "severity": raw.get("severity", ""),
            "status": raw.get("severity", ""),
            "source_gate": "gate_14_content_risk",
            "matched_rule_id": rule_id,
            "matched_rule_description": raw.get("rule_description", build_rule_description(rule_id)),
            "matched_field": raw.get("field", ""),
            "safe_excerpt": raw.get("excerpt", ""),
            "reviewer_instruction": (
                "This record contains a GCC regulatory advisory phrase. Please verify that the advice is "
                "accurate, current, and appropriate for the target audience. Mark ACCEPT if correct, "
                "REVISE if wording needs adjustment, or REJECT if the advice is misleading."
            ),
            "rule_id": rule_id,
            "rule_description": raw.get("rule_description", build_rule_description(rule_id)),
            "field": raw.get("field", ""),
            "excerpt": raw.get("excerpt", ""),
        }
        findings_list.append(finding)
        findings_by_id.setdefault(eid, []).append(finding)
    base_ids = {r.get("example_id", "") for r in base_queue_records}
    required_ids = set(findings_by_id)
    supplemental_ids = sorted(required_ids - base_ids)
    corpus_by_id = {r.get("example_id", ""): r for r in corpus_records}
    missing = sorted(eid for eid in supplemental_ids if eid not in corpus_by_id)
    if missing:
        raise ValueError(f"Mandatory supplements missing from corpus: {missing}")
    effective_records = list(base_queue_records) + [corpus_by_id[eid] for eid in supplemental_ids]
    policy = {
        "frozen_base_queue_count": len(base_queue_records),
        "mandatory_risk_supplement_count": len(supplemental_ids),
        "effective_review_set_count": len(effective_records),
        "review_required_count": len(required_ids),
        "none_count": len(effective_records) - len(required_ids),
        "clear_count": 0,
        "supplemental_ids": supplemental_ids,
        "review_required_ids": sorted(required_ids),
        "selection_rule": "FROZEN_BASE_QUEUE UNION ALL_GATE_14_FINDINGS_NOT_ALREADY_IN_BASE_QUEUE",
    }
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "package_commit": package_commit,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_findings": len(findings_list),
        "blocked_count": sum(1 for f in findings_list if f["severity"] == "BLOCKED"),
        **policy,
        "findings": [
            {k: v for k, v in f.items() if k not in {"rule_id", "rule_description", "field", "excerpt"}}
            | {"schema_version": SIDECAR_SCHEMA_VERSION, "package_commit": package_commit,
               "corpus_sha256": EXPECTED_CORPUS_SHA256,
               "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            for f in findings_list
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Sidecar written: {output_path} ({len(findings_list)} findings)")
    return findings_by_id, policy, effective_records

def generate_reviewer_package(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    package_commit: str,
    output_path: Path,
) -> None:
    """Generate REVIEWER_PACKAGE.jsonl — joined representation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in queue_records:
            risk_fields = compute_content_risk_fields(r, findings_by_id)
            row = dict(r)
            row.update(risk_fields)
            row["package_commit"] = package_commit
            # Ensure all human judgment fields are empty
            for jf in REVIEWER_JUDGMENT_FIELDS:
                row[jf] = ""
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Reviewer package written: {output_path} ({len(queue_records)} records)")


def generate_reviewer_csv(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    package_commit: str,
    output_path: Path,
) -> None:
    """Generate REVIEWER_TEMPLATE.csv — UTF-8 BOM for Excel."""
    if not queue_records:
        return

    # Build column order: identity fields, content-risk fields, human judgment fields
    identity_cols = [
        "example_id", "package_commit", "split", "language", "domain", "task_type",
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
            row["package_commit"] = package_commit
            # Ensure all human judgment fields are empty
            for jf in judgment_cols:
                row[jf] = ""
            writer.writerow(row)
    print(f"Reviewer CSV written: {output_path} ({len(queue_records)} records)")


def generate_markdown(
    queue_records: List[Dict],
    findings_by_id: Dict[str, List[Dict]],
    package_commit: str,
    output_path: Path,
    policy: Optional[Dict] = None,
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
        f"> Package source commit: `{package_commit}`",
        "> Package identity scheme: `package_commit` identifies the source commit; the artifact release commit is its direct child.",
        "",
        "## Coverage",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Total records | {len(queue_records)} |",
        f"| Flagged (REVIEW_REQUIRED) | {len(flagged_ids)} |",
        f"| Not flagged (NONE) | {len(queue_records) - len(flagged_ids)} |",
        f"| Frozen base queue | {(policy or {}).get('frozen_base_queue_count', len(queue_records))} |",
        f"| Mandatory risk supplement | {(policy or {}).get('mandatory_risk_supplement_count', 0)} |",
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
    parser.add_argument(
        "--package-commit",
        default=None,
        help=(
            "Full 40-character artifact source commit to embed. Defaults to HEAD. "
            "The release procedure supplies HEAD^ after the source commit."
        ),
    )
    args = parser.parse_args()

    package_commit = args.package_commit or resolve_git_commit("HEAD")
    if not is_valid_git_commit(package_commit):
        raise ValueError("package_commit must be a full 40-character hexadecimal Git commit")
    package_commit = package_commit.lower()
    print(f"Package source commit: {package_commit}")

    # Load frozen queue (never modified)
    queue_records = load_queue()
    print(f"Loaded {len(queue_records)} queue records")

    # Load full corpus for Gate 14 scan
    corpus_records = load_all_corpus()
    print(f"Loaded {len(corpus_records)} corpus records")

    # Generate sidecar
    sidecar_path = args.docs_dir / "REVIEW_RISK_FINDINGS.json"
    findings_by_id, policy, effective_records = generate_sidecar(corpus_records, queue_records, package_commit, sidecar_path)

    # Generate reviewer-facing representations
    generate_reviewer_package(
        effective_records, findings_by_id, package_commit,
        args.docs_dir / "REVIEWER_PACKAGE.jsonl"
    )
    generate_reviewer_csv(
        effective_records, findings_by_id, package_commit,
        args.docs_dir / "REVIEWER_TEMPLATE.csv"
    )
    generate_markdown(
        effective_records, findings_by_id, package_commit,
        args.docs_dir / "ARABIC_HUMAN_REVIEW_QUEUE.md", policy
    )

    print("\nReview package generation complete.")
    print(f"Flagged records: {sorted(findings_by_id.keys())}")
    print(f"Effective review set: {policy['frozen_base_queue_count']} base + {policy['mandatory_risk_supplement_count']} supplement = {policy['effective_review_set_count']}")


if __name__ == "__main__":
    main()
