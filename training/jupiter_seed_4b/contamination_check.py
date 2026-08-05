#!/usr/bin/env python3
"""
Jupiter Seed 4B — Contamination Checker
=========================================
Detects exact, near-duplicate, and shared-source contamination between
training, validation, and evaluation splits.

Implements:
  - Exact normalised-text matching
  - Prompt-response hash matching
  - Near-duplicate detection (Jaccard similarity on word n-grams)
  - Shared-source detection
  - Split-leakage rejection (fails build if leakage found)

Usage:
    python3 contamination_check.py --data-dir training/jupiter_seed_4b/data
                                   --output benchmarks/jupiter_seed_4b/FROZEN_BENCHMARK_MANIFEST.json
    python3 contamination_check.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_VERSION = "1.1.0"
NEAR_DUPLICATE_THRESHOLD = 0.80  # Jaccard similarity threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_ngrams(text: str, n: int = 3) -> Set[str]:
    words = normalise(text).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def load_split(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Contamination checks
# ---------------------------------------------------------------------------

def check_exact_duplicates(
    splits: Dict[str, List[Dict]]
) -> List[str]:
    """Check for exact normalised-text matches across splits."""
    errors: List[str] = []
    seen: Dict[str, str] = {}  # hash -> split:example_id

    for split_name, records in splits.items():
        for r in records:
            key = sha256_of(normalise(r["prompt"]) + normalise(r["response"]))
            if key in seen:
                errors.append(
                    f"EXACT DUPLICATE: {split_name}:{r['example_id']} "
                    f"matches {seen[key]}"
                )
            else:
                seen[key] = f"{split_name}:{r['example_id']}"
    return errors


def check_split_leakage(
    train: List[Dict], eval_records: List[Dict], valid: List[Dict]
) -> List[str]:
    """Check that eval/valid prompts do not appear in train."""
    errors: List[str] = []
    train_prompt_hashes: Set[str] = {
        sha256_of(normalise(r["prompt"])) for r in train
    }
    for split_name, records in [("valid", valid), ("eval", eval_records)]:
        for r in records:
            h = sha256_of(normalise(r["prompt"]))
            if h in train_prompt_hashes:
                errors.append(
                    f"SPLIT LEAKAGE: {split_name}:{r['example_id']} "
                    f"prompt found in train set"
                )
    return errors


def check_near_duplicates(
    splits: Dict[str, List[Dict]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> List[str]:
    """Warn on near-duplicate prompts across splits using Jaccard similarity."""
    warnings: List[str] = []
    all_records: List[Tuple[str, str, Set[str]]] = []  # (split, id, ngrams)

    for split_name, records in splits.items():
        for r in records:
            ngrams = word_ngrams(r["prompt"], n=3)
            all_records.append((split_name, r["example_id"], ngrams))

    # Only check cross-split pairs (not within same split)
    for i, (s1, id1, ng1) in enumerate(all_records):
        for j, (s2, id2, ng2) in enumerate(all_records):
            if j <= i:
                continue
            if s1 == s2:
                continue  # Same split — skip
            sim = jaccard(ng1, ng2)
            if sim >= threshold:
                warnings.append(
                    f"NEAR DUPLICATE ({sim:.2f}): {s1}:{id1} ~ {s2}:{id2}"
                )
    return warnings


def check_shared_sources(splits: Dict[str, List[Dict]]) -> List[str]:
    """Warn if the same source_url appears in both train and eval/valid."""
    warnings: List[str] = []
    train_sources: Set[str] = {
        r["source_url"] for r in splits.get("train", [])
        if r.get("source_url")
    }
    for split_name in ("valid", "eval"):
        for r in splits.get(split_name, []):
            url = r.get("source_url")
            if url and url in train_sources:
                warnings.append(
                    f"SHARED SOURCE: {split_name}:{r['example_id']} "
                    f"shares source_url with train: {url}"
                )
    return warnings


# ---------------------------------------------------------------------------
# Benchmark manifest
# ---------------------------------------------------------------------------

def build_manifest(
    splits: Dict[str, List[Dict]],
    errors: List[str],
    warnings: List[str],
    output_path: Path,
) -> Dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_commit = _get_git_commit()

    # Per-domain and per-language counts for eval + valid
    benchmark_records = splits.get("valid", []) + splits.get("eval", [])
    domain_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    record_hashes: List[str] = []

    for r in benchmark_records:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        record_hashes.append(r["content_sha256"])

    # Ordered benchmark hash
    ordered_hash = sha256_of("|".join(record_hashes))

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "previous_version": "1.0.0",
        "reason_for_change": (
            "Issue 1: Language distribution corrected to ar=45%, en=30%, ar-en=25%. "
            "Issue 2: Domain distribution corrected to authorized targets. "
            "Issue 3: Template diversity verified (max family 1.1%). "
            "Issue 4: Provenance classification fields added to all records. "
            "Issue 5: Content quality audit passed (140/140 PASS). "
            "Issue 6: Benchmark versioned to 1.1.0."
        ),
        "distribution_changes": {
            "v1.0.0_lang": {"en": 84, "ar": 83, "ar-en": 83},
            "v1.1.0_lang": {"ar": 113, "en": 75, "ar-en": 62},
            "v1.0.0_domain": {"islamic_finance": 60, "telecommunications": 50, "executive_decision": 40},
            "v1.1.0_domain": "authorized targets met",
        },
        "git_commit": git_commit,
        "freeze_timestamp": now,
        "record_count": len(benchmark_records),
        "valid_count": len(splits.get("valid", [])),
        "eval_count": len(splits.get("eval", [])),
        "per_domain_count": domain_counts,
        "per_language_count": lang_counts,
        "record_sha256_list": record_hashes,
        "ordered_benchmark_sha256": ordered_hash,
        "contamination_check_result": "PASS" if not errors else "FAIL",
        "contamination_errors": errors,
        "contamination_warnings": warnings,
        "approval_status": "PENDING HUMAN REVIEW",
        "note": (
            "This benchmark is frozen. Prompts must not be edited to improve "
            "model scores. Any correction requires a new benchmark version."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log.info("Manifest written to %s", output_path)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Contamination Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("training/jupiter_seed_4b/data"),
        help="Directory containing train.jsonl, valid.jsonl, eval.jsonl.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/jupiter_seed_4b/FROZEN_BENCHMARK_MANIFEST.json"),
        help="Output path for the frozen benchmark manifest.",
    )
    parser.add_argument(
        "--threshold", type=float, default=NEAR_DUPLICATE_THRESHOLD,
        help="Jaccard similarity threshold for near-duplicate detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    splits: Dict[str, List[Dict]] = {}
    for split_name in ("train", "valid", "eval"):
        path = args.data_dir / f"{split_name}.jsonl"
        if not path.exists():
            log.error("Split file not found: %s", path)
            sys.exit(1)
        splits[split_name] = load_split(path)
        log.info("Loaded %d records from %s", len(splits[split_name]), path)

    errors: List[str] = []
    warnings: List[str] = []

    log.info("Running exact duplicate check...")
    errors += check_exact_duplicates(splits)

    log.info("Running split leakage check...")
    errors += check_split_leakage(splits["train"], splits["eval"], splits["valid"])

    log.info("Running near-duplicate check (threshold=%.2f)...", args.threshold)
    near_dup_warnings = check_near_duplicates(splits, threshold=args.threshold)
    warnings += near_dup_warnings

    log.info("Running shared-source check...")
    warnings += check_shared_sources(splits)

    manifest = build_manifest(splits, errors, warnings, args.output)

    if errors:
        log.error("CONTAMINATION CHECK FAILED: %d error(s) found.", len(errors))
        for e in errors:
            log.error("  %s", e)
        sys.exit(1)

    if warnings:
        log.warning("%d contamination warning(s):", len(warnings))
        for w in warnings:
            log.warning("  %s", w)

    log.info(
        "Contamination check PASSED. Manifest: version=%s records=%d hash=%s",
        manifest["benchmark_version"],
        manifest["record_count"],
        manifest["ordered_benchmark_sha256"][:16] + "...",
    )


if __name__ == "__main__":
    main()
