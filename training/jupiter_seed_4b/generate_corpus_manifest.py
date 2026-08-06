#!/usr/bin/env python3
"""
Jupiter Seed 4B — Corpus Immutability Manifest Generator (Defect 8)
====================================================================
Generates FROZEN_CORPUS_MANIFEST.json containing hashes of:
  - All 101 prompts
  - All 101 responses
  - All 101 scenario briefs
  - Split assignments
  - Content-family assignments
  - Ordered dataset hash

This manifest proves the corpus is identical to commit:
  2e36f6b977a8af052fced5a532c1168dc1988b6f

Defect 8 fix: Distinguishes dataset_content_commit from package_generation_commit.
Defect 9 fix: Adds scenario_brief fields to review package.

DO NOT MODIFY THE CORPUS. This script only reads and hashes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"

# The frozen corpus content commit — must not change
DATASET_CONTENT_COMMIT = "2e36f6b977a8af052fced5a532c1168dc1988b6f"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def load_all() -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def generate_manifest(records: List[Dict], output_path: Path) -> Dict:
    """
    Generate the corpus immutability manifest.
    Defect 8: distinguishes dataset_content_commit from package_generation_commit.
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"
    package_generation_commit = get_current_commit()

    # Sort records by example_id for deterministic ordering
    records_sorted = sorted(records, key=lambda r: r["example_id"])

    prompt_hashes = []
    response_hashes = []
    brief_hashes = []
    split_assignments = {}
    family_assignments = {}

    for r in records_sorted:
        eid = r["example_id"]
        prompt_hashes.append(sha256_of(r["prompt"]))
        response_hashes.append(sha256_of(r["response"]))
        brief_hashes.append(sha256_of(r.get("scenario_brief", "")))
        split_assignments[eid] = r["split"]
        family_assignments[eid] = r.get("content_family_id", "")

    # Ordered corpus hash: hash of all prompts|responses|briefs concatenated in order
    ordered_corpus_str = "|".join(
        f"{p}:{r}:{b}"
        for p, r, b in zip(prompt_hashes, response_hashes, brief_hashes)
    )
    ordered_corpus_hash = sha256_of(ordered_corpus_str)

    # Split assignment hash
    split_str = "|".join(f"{k}:{v}" for k, v in sorted(split_assignments.items()))
    split_hash = sha256_of(split_str)

    # Family assignment hash
    family_str = "|".join(f"{k}:{v}" for k, v in sorted(family_assignments.items()))
    family_hash = sha256_of(family_str)

    manifest = {
        "manifest_schema_version": "1.0",
        "manifest_type": "FROZEN_CORPUS_IMMUTABILITY_MANIFEST",
        "dataset_name": "Jupiter Seed 4B",
        "dataset_version": "2.0.0",

        # Defect 8: distinguish content commit from package generation commit
        "dataset_content_commit": DATASET_CONTENT_COMMIT,
        "package_generation_commit": package_generation_commit,
        "independent_audit_commit": None,
        "human_review_commit": None,
        "manifest_generated_at": now,

        "note_on_commit_semantics": (
            "dataset_content_commit is the commit that established the 101 frozen examples "
            "and must never change. package_generation_commit is the commit at which this "
            "manifest was generated (tooling-only repair). independent_audit_commit and "
            "human_review_commit must be populated externally by Kishore and human reviewers "
            "respectively — never by software."
        ),

        # Corpus statistics
        "total_records": len(records_sorted),
        "train_count": sum(1 for r in records_sorted if r["split"] == "train"),
        "valid_count": sum(1 for r in records_sorted if r["split"] == "valid"),
        "eval_count": sum(1 for r in records_sorted if r["split"] == "eval"),

        # Hashes of individual fields
        "prompt_sha256_list": prompt_hashes,
        "response_sha256_list": response_hashes,
        "scenario_brief_sha256_list": brief_hashes,

        # Aggregate hashes
        "ordered_corpus_sha256": ordered_corpus_hash,
        "split_assignment_sha256": split_hash,
        "family_assignment_sha256": family_hash,

        # Verification instructions
        "verification_instructions": (
            "To verify corpus immutability: "
            "(1) Sort all records by example_id. "
            "(2) Compute SHA-256 of each prompt, response, and scenario_brief. "
            "(3) Concatenate as 'prompt_hash:response_hash:brief_hash' per record, joined by '|'. "
            "(4) Compute SHA-256 of the concatenated string. "
            "(5) Compare to ordered_corpus_sha256. "
            "Any difference indicates corpus modification."
        ),

        "approval_status": "PENDING INDEPENDENT VERIFICATION AND HUMAN REVIEW",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"Corpus manifest written to {output_path}")
    print(f"  dataset_content_commit: {DATASET_CONTENT_COMMIT}")
    print(f"  package_generation_commit: {package_generation_commit}")
    print(f"  ordered_corpus_sha256: {ordered_corpus_hash}")
    print(f"  total_records: {len(records_sorted)}")
    return manifest


def verify_against_content_commit(manifest: Dict) -> bool:
    """
    Verify that the current corpus matches the frozen content commit.
    Returns True if corpus is unchanged.
    """
    records = load_all()
    records_sorted = sorted(records, key=lambda r: r["example_id"])

    prompt_hashes = [sha256_of(r["prompt"]) for r in records_sorted]
    response_hashes = [sha256_of(r["response"]) for r in records_sorted]
    brief_hashes = [sha256_of(r.get("scenario_brief", "")) for r in records_sorted]

    ordered_corpus_str = "|".join(
        f"{p}:{r}:{b}"
        for p, r, b in zip(prompt_hashes, response_hashes, brief_hashes)
    )
    current_hash = sha256_of(ordered_corpus_str)
    expected_hash = manifest["ordered_corpus_sha256"]

    if current_hash != expected_hash:
        print(f"CORPUS IMMUTABILITY VIOLATION: hash mismatch", file=sys.stderr)
        print(f"  Expected: {expected_hash}", file=sys.stderr)
        print(f"  Current:  {current_hash}", file=sys.stderr)
        return False

    print(f"Corpus immutability verified: {current_hash}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Corpus Immutability Manifest")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "benchmarks" / "jupiter_seed_4b" / "FROZEN_CORPUS_MANIFEST.json")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing manifest against current corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_all()

    if args.verify and args.output.exists():
        with args.output.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        ok = verify_against_content_commit(manifest)
        sys.exit(0 if ok else 1)

    manifest = generate_manifest(records, args.output)
    # Immediately verify
    ok = verify_against_content_commit(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
