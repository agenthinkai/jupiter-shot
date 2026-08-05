#!/usr/bin/env python3
"""
Jupiter Seed 4B — AI-Assisted Arabic Review
============================================
Performs automated first-pass quality review of Arabic-English examples.

IMPORTANT: AI review does NOT equal human approval.
This script labels examples for human review; it does not authorize
commercial release or mark examples as production-ready.

Usage:
    python3 arabic_review.py --dataset-path <path> --output-path <path> [--seed 42]
    python3 arabic_review.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Review dimensions
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    example_id: str
    language: str
    domain: str
    # Scores: 1 (fail) to 5 (excellent)
    msa_quality: int                  # Modern Standard Arabic grammar/fluency
    gcc_terminology: int              # GCC-appropriate vocabulary and style
    meaning_preservation: int         # English-Arabic meaning alignment
    grammar_fluency: int              # Overall grammar and fluency
    islamic_finance_accuracy: int     # Accuracy of Islamic-finance terms
    banking_telecom_accuracy: int     # Accuracy of banking/telecom terms
    hallucination_risk: int           # 5=low risk, 1=high risk
    cultural_appropriateness: int     # GCC cultural sensitivity
    safety_score: int                 # 5=safe, 1=unsafe/discriminatory
    # Derived
    composite_score: float
    flag_for_human_review: bool
    rejection_reason: Optional[str]
    ai_review_note: str


# ---------------------------------------------------------------------------
# Heuristic reviewers (no external API calls)
# ---------------------------------------------------------------------------

ARABIC_UNICODE_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
UNSAFE_PATTERNS = re.compile(
    r"\b(kill|bomb|attack|weapon|terrorist|نفجر|قتل|إرهاب|سلاح)\b",
    re.IGNORECASE,
)
HALLUCINATION_TRIGGERS = re.compile(
    r"\b(always|never|guaranteed|100%|proven|دائماً|مضمون|مؤكد)\b",
    re.IGNORECASE,
)
ISLAMIC_FINANCE_TERMS = {
    "murabaha", "sukuk", "ijara", "musharaka", "mudaraba", "takaful",
    "zakat", "waqf", "riba", "halal", "مرابحة", "صكوك", "إجارة",
    "مشاركة", "مضاربة", "تكافل", "زكاة", "وقف", "ربا",
}
BANKING_TELECOM_TERMS = {
    "basel", "aml", "kyc", "swift", "iban", "netting", "collateral",
    "bandwidth", "latency", "qos", "roaming", "spectrum",
    "بازل", "غسيل أموال", "اعرف عميلك", "ضمان", "نطاق ترددي",
}


def score_arabic_quality(text: str) -> int:
    """Score Modern Standard Arabic quality heuristically (1-5)."""
    arabic_chars = len(ARABIC_UNICODE_RANGE.findall(text))
    total_chars = max(len(text.replace(" ", "")), 1)
    arabic_ratio = arabic_chars / total_chars
    if arabic_ratio > 0.6:
        return 4
    elif arabic_ratio > 0.3:
        return 3
    elif arabic_ratio > 0.0:
        return 2
    return 1  # No Arabic detected


def score_gcc_terminology(text: str, domain: str) -> int:
    """Score GCC-appropriate terminology (1-5)."""
    if domain in ("islamic_finance", "gcc_banking"):
        hits = sum(1 for t in ISLAMIC_FINANCE_TERMS if t in text.lower())
        return min(5, 2 + hits)
    if domain in ("telecommunications",):
        hits = sum(1 for t in BANKING_TELECOM_TERMS if t in text.lower())
        return min(5, 2 + hits)
    return 3  # Neutral for other domains


def score_meaning_preservation(instruction: str, response: str) -> int:
    """Heuristic: check response length relative to instruction (1-5)."""
    ratio = len(response) / max(len(instruction), 1)
    if ratio < 0.2:
        return 1  # Response too short relative to instruction
    elif ratio < 0.5:
        return 2
    elif ratio < 5.0:
        return 4
    return 3  # Very long response — may contain padding


def score_safety(text: str) -> int:
    """Score safety (5=safe, 1=unsafe)."""
    if UNSAFE_PATTERNS.search(text):
        return 1
    return 5


def score_hallucination_risk(text: str) -> int:
    """Score hallucination risk (5=low risk, 1=high risk)."""
    hits = len(HALLUCINATION_TRIGGERS.findall(text))
    return max(1, 5 - hits)


def review_example(raw: dict) -> ReviewResult:
    """Run all heuristic reviewers on a single example."""
    example_id = raw.get("example_id", "unknown")
    instruction = raw.get("instruction", "")
    response = raw.get("response", "")
    language = raw.get("language", "en")
    domain = raw.get("domain", "unknown")
    full_text = f"{instruction} {response}"

    msa = score_arabic_quality(full_text) if language in ("ar", "ar-en") else 3
    gcc = score_gcc_terminology(full_text, domain)
    meaning = score_meaning_preservation(instruction, response)
    grammar = msa  # Proxy: same heuristic for now
    islamic = (
        score_gcc_terminology(full_text, "islamic_finance")
        if domain == "islamic_finance"
        else 3
    )
    banking = (
        score_gcc_terminology(full_text, "gcc_banking")
        if domain in ("gcc_banking", "telecommunications")
        else 3
    )
    hallucination = score_hallucination_risk(full_text)
    cultural = 4  # Default: assume appropriate unless flagged
    safety = score_safety(full_text)

    composite = (
        msa * 0.20
        + gcc * 0.15
        + meaning * 0.15
        + grammar * 0.10
        + islamic * 0.10
        + banking * 0.10
        + hallucination * 0.10
        + cultural * 0.05
        + safety * 0.05
    )

    rejection_reason: Optional[str] = None
    if safety == 1:
        rejection_reason = "UNSAFE_CONTENT_DETECTED"
    elif hallucination <= 2:
        rejection_reason = "HIGH_HALLUCINATION_RISK"
    elif composite < 2.5:
        rejection_reason = "LOW_COMPOSITE_SCORE"

    # Flag for human review if score < 4.0 or any dimension scores 1
    flag = (
        composite < 4.0
        or safety == 1
        or hallucination <= 2
        or msa == 1
    )

    return ReviewResult(
        example_id=example_id,
        language=language,
        domain=domain,
        msa_quality=msa,
        gcc_terminology=gcc,
        meaning_preservation=meaning,
        grammar_fluency=grammar,
        islamic_finance_accuracy=islamic,
        banking_telecom_accuracy=banking,
        hallucination_risk=hallucination,
        cultural_appropriateness=cultural,
        safety_score=safety,
        composite_score=round(composite, 3),
        flag_for_human_review=flag,
        rejection_reason=rejection_reason,
        ai_review_note=(
            "AI-ASSISTED REVIEW ONLY. Not equivalent to human approval. "
            "Examples flagged for human review must be reviewed before any release."
        ),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_review(dataset_path: Path, output_path: Path, top_n_human: int = 50) -> None:
    """Review all examples and write results."""
    if not dataset_path.exists():
        log.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    results: List[ReviewResult] = []
    rejected: List[str] = []

    with dataset_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                log.error("Line %d: JSON parse error — %s", line_no, exc)
                sys.exit(1)

            result = review_example(raw)
            results.append(result)
            if result.rejection_reason:
                rejected.append(result.example_id)
                log.warning(
                    "Example '%s' flagged for rejection: %s",
                    result.example_id,
                    result.rejection_reason,
                )

    # Select top-N for human review (lowest composite scores among non-rejected)
    non_rejected = [r for r in results if not r.rejection_reason]
    top_for_human = sorted(non_rejected, key=lambda r: r.composite_score)[:top_n_human]
    for r in top_for_human:
        r.flag_for_human_review = True

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Summary report
    summary = {
        "total_reviewed": len(results),
        "rejected_count": len(rejected),
        "flagged_for_human_review": sum(1 for r in results if r.flag_for_human_review),
        "human_review_required_before_release": True,
        "ai_review_note": (
            "AI review is a first-pass filter only. "
            "Human approval is mandatory before any commercial or public release."
        ),
        "rejected_ids": rejected,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info(
        "Review complete: %d total, %d rejected, %d flagged for human review.",
        len(results),
        len(rejected),
        summary["flagged_for_human_review"],
    )
    log.info("Results written to %s", output_path)
    log.info("Summary written to %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — AI-Assisted Arabic Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path("artifacts/arabic_review_results.jsonl"),
        help="Path for the output review JSONL.",
    )
    parser.add_argument(
        "--top-n-human", type=int, default=50,
        help="Number of examples to select for mandatory human review.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_review(args.dataset_path, args.output_path, args.top_n_human)


if __name__ == "__main__":
    main()
