#!/usr/bin/env python3
"""
Jupiter Seed 4B — Dataset Provenance Loader
============================================
Loads, validates, and enforces provenance requirements for every training
example. Fails loudly on missing or incomplete provenance records.

Usage:
    python3 dataset_loader.py --dataset-path <path> --output-path <path> [--seed 42]
    python3 dataset_loader.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "example_id",
    "source",
    "licence_status",
    "author",
    "creation_method",
    "language",
    "domain",
    "review_status",
    "instruction",
    "response",
]

PROHIBITED_SOURCES = [
    "warba",
    "investgb",
    "client_data",
    "confidential",
    "teacher_model",
    "scraped_private",
]

ALLOWED_DOMAINS = {
    "islamic_finance",
    "gcc_banking",
    "telecommunications",
    "government_regulation",
    "energy_logistics",
    "executive_decision",
    "arabic_english_correspondence",
}

ALLOWED_LANGUAGES = {"ar", "en", "ar-en"}

ALLOWED_REVIEW_STATUSES = {"pending_human", "ai_reviewed", "human_approved", "rejected"}


@dataclass
class ProvenanceRecord:
    example_id: str
    source: str
    source_url: Optional[str]
    licence_status: str
    author: str
    creation_method: str
    language: str
    domain: str
    review_status: str
    modification_history: List[str]
    inclusion_reason: Optional[str]
    instruction: str
    response: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ProvenanceError(ValueError):
    """Raised when a record fails provenance validation."""


def validate_record(raw: dict) -> ProvenanceRecord:
    """Validate a single raw dict and return a typed ProvenanceRecord.

    Raises ProvenanceError immediately on any violation.
    """
    # 1. Required fields
    for f in REQUIRED_FIELDS:
        if f not in raw or not raw[f]:
            raise ProvenanceError(
                f"Missing required provenance field '{f}' in example: "
                f"{raw.get('example_id', '<unknown>')}"
            )

    # 2. Prohibited sources
    src_lower = raw["source"].lower()
    for prohibited in PROHIBITED_SOURCES:
        if prohibited in src_lower:
            raise ProvenanceError(
                f"Prohibited source '{raw['source']}' in example "
                f"'{raw['example_id']}'. Stop immediately."
            )

    # 3. Domain whitelist
    if raw["domain"] not in ALLOWED_DOMAINS:
        raise ProvenanceError(
            f"Unknown domain '{raw['domain']}' in example '{raw['example_id']}'. "
            f"Allowed: {ALLOWED_DOMAINS}"
        )

    # 4. Language whitelist
    if raw["language"] not in ALLOWED_LANGUAGES:
        raise ProvenanceError(
            f"Unknown language '{raw['language']}' in example '{raw['example_id']}'. "
            f"Allowed: {ALLOWED_LANGUAGES}"
        )

    # 5. Review status whitelist
    if raw["review_status"] not in ALLOWED_REVIEW_STATUSES:
        raise ProvenanceError(
            f"Unknown review_status '{raw['review_status']}' in example "
            f"'{raw['example_id']}'. Allowed: {ALLOWED_REVIEW_STATUSES}"
        )

    # 6. Instruction and response must be non-trivial strings
    if len(raw["instruction"].strip()) < 10:
        raise ProvenanceError(
            f"Instruction too short in example '{raw['example_id']}'"
        )
    if len(raw["response"].strip()) < 10:
        raise ProvenanceError(
            f"Response too short in example '{raw['example_id']}'"
        )

    return ProvenanceRecord(
        example_id=raw["example_id"],
        source=raw["source"],
        source_url=raw.get("source_url"),
        licence_status=raw["licence_status"],
        author=raw["author"],
        creation_method=raw["creation_method"],
        language=raw["language"],
        domain=raw["domain"],
        review_status=raw["review_status"],
        modification_history=raw.get("modification_history", []),
        inclusion_reason=raw.get("inclusion_reason"),
        instruction=raw["instruction"],
        response=raw["response"],
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_path: Path,
    strict: bool = True,
) -> List[ProvenanceRecord]:
    """Load and validate all records from a JSONL file.

    Args:
        dataset_path: Path to a .jsonl file where each line is a JSON object.
        strict: If True, abort on first provenance error. If False, log and skip.

    Returns:
        List of validated ProvenanceRecord objects.
    """
    if not dataset_path.exists():
        log.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    records: List[ProvenanceRecord] = []
    errors: List[str] = []

    with dataset_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Line {line_no}: JSON parse error — {exc}"
                if strict:
                    log.error(msg)
                    sys.exit(1)
                errors.append(msg)
                continue

            try:
                record = validate_record(raw)
                records.append(record)
            except ProvenanceError as exc:
                msg = f"Line {line_no}: {exc}"
                if strict:
                    log.error(msg)
                    sys.exit(1)
                errors.append(msg)

    if errors:
        log.warning("%d provenance errors encountered (non-strict mode):", len(errors))
        for e in errors:
            log.warning("  %s", e)

    log.info("Loaded %d valid records from %s", len(records), dataset_path)
    return records


def generate_provenance_report(records: List[ProvenanceRecord], output_path: Path) -> None:
    """Write a JSON provenance summary report."""
    domain_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}

    for r in records:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1
        review_counts[r.review_status] = review_counts.get(r.review_status, 0) + 1

    report = {
        "total_records": len(records),
        "provenance_coverage_pct": 100.0,  # All records passed validation
        "domain_distribution": domain_counts,
        "language_distribution": lang_counts,
        "review_status_distribution": review_counts,
        "human_approved_count": review_counts.get("human_approved", 0),
        "pending_human_review_count": review_counts.get("pending_human", 0) + review_counts.get("ai_reviewed", 0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log.info("Provenance report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Dataset Provenance Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/provenance_report.json"),
        help="Path for the output provenance report JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Abort on first provenance error (default: True).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_dataset(args.dataset_path, strict=args.strict)
    generate_provenance_report(records, args.output_path)
    log.info("Dataset loader complete. %d records validated.", len(records))


if __name__ == "__main__":
    main()
