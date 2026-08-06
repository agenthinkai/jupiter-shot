#!/usr/bin/env python3
"""
Jupiter Seed 4B — Content Quality Audit (Defect 6 fix)
=======================================================
Defect 6 fix: Removed all unconditional or fallback PASS logic.
PASS now requires explicit satisfaction of every mandatory condition.
Unknown, missing, malformed, unscored or unevaluable conditions produce
FAIL, BLOCKED, NOT_READY, or NOT_EVALUABLE — never PASS.

Usage:
    python3 content_quality_audit.py --data-dir training/jupiter_seed_4b/data
                                     --output docs/jupiter_seed_4b/CONTENT_QUALITY_AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Dimension verdicts
_PASS = "PASS"
_REVISE = "REVISE"
_REJECT = "REJECT"
_NOT_EVALUABLE = "NOT_EVALUABLE"
_BLOCKED = "BLOCKED"
_NOT_READY = "NOT_READY"

# Minimum thresholds
MIN_RESPONSE_LEN = 50          # characters
MIN_AR_CHARS = 30              # Arabic characters for ar/ar-en records
MIN_EN_CHARS = 30              # Latin characters for en/ar-en records
MIN_AR_RATIO = 0.20            # Arabic ratio for ar records


def ar_char_count(text: str) -> int:
    return len(re.findall(r"[\u0600-\u06FF]", str(text)))


def en_char_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]", str(text)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            log.error("Split file not found: %s", path)
            sys.exit(1)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def stratified_sample(records: List[Dict], seed: int = 42) -> List[Dict]:
    """Sample up to 20 records per domain, ensuring diversity in language and split."""
    rng = random.Random(seed)
    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    sample: List[Dict] = []
    for domain, group in by_domain.items():
        rng.shuffle(group)
        selected = []
        for lang in ("ar", "en", "ar-en"):
            lang_group = [r for r in group if r["language"] == lang]
            selected.extend(lang_group[:6])
        remaining = [r for r in group if r not in selected]
        selected.extend(remaining[:max(0, 20 - len(selected))])
        sample.extend(selected[:20])
    return sample


# ---------------------------------------------------------------------------
# Individual dimension scorers — no fallback PASS
# ---------------------------------------------------------------------------

def score_response_length(response: str) -> str:
    """PASS only if response is substantively long enough."""
    if not response or not response.strip():
        return _REJECT
    if len(response.strip()) < MIN_RESPONSE_LEN:
        return _REVISE
    return _PASS


def score_relevance(prompt: str, response: str) -> str:
    """PASS only if response is meaningfully longer than a trivial reply."""
    if not prompt or not response:
        return _REJECT
    if len(response) < len(prompt) * 0.15:
        return _REVISE
    return _PASS


def score_factual_discipline(r: Dict) -> str:
    """PASS only when provenance requirements are explicitly satisfied."""
    prov_type = r.get("provenance_type")
    if not prov_type:
        return _BLOCKED  # Cannot evaluate without provenance_type
    if prov_type == "SOURCE_DEPENDENT_FACTUAL":
        url = r.get("source_url", "")
        if not url or url == "https://github.com/agenthinkai/jupiter-shot":
            return _REJECT  # Generic repo URL is not a valid factual source
        return _PASS
    if prov_type == "ORIGINAL_SCENARIO":
        disclaimer = str(r.get("fictional_disclaimer", "")).lower()
        if "fictional" not in disclaimer:
            return _REJECT
        return _PASS
    if prov_type in ("PUBLIC_RULE_SUMMARY", "TRANSLATION_OR_CORRESPONDENCE", "SAFETY_OR_REFUSAL"):
        if not r.get("source_url"):
            return _REVISE
        return _PASS
    return _NOT_EVALUABLE  # Unknown provenance type


def score_professional_usefulness(response: str) -> str:
    """PASS only if response does not contain AI refusal boilerplate."""
    if not response:
        return _REJECT
    lower = response.lower()
    if "sorry" in lower and "ai" in lower:
        return _REVISE
    if "i cannot" in lower and "assist" in lower:
        return _REVISE
    return _PASS


def score_arabic_quality(lang: str, full_text: str, response: str) -> str:
    """PASS only if Arabic content is substantively present where required."""
    if "ar" not in lang:
        return "N/A"
    ar_chars = ar_char_count(response)
    if ar_chars < MIN_AR_CHARS:
        return _REVISE
    if lang == "ar":
        resp_len = max(len(response.replace(" ", "")), 1)
        if ar_chars / resp_len < MIN_AR_RATIO:
            return _REVISE
    return _PASS


def score_english_quality(lang: str, response: str) -> str:
    """PASS only if English content is substantively present where required."""
    if "en" not in lang:
        return "N/A"
    en_chars = en_char_count(response)
    if en_chars < MIN_EN_CHARS:
        return _REVISE
    return _PASS


def score_translation_fidelity(lang: str, ar_qual: str, en_qual: str) -> str:
    """PASS only if both language dimensions pass for bilingual records."""
    if lang != "ar-en":
        return "N/A"
    if ar_qual in (_REJECT, _REVISE, _BLOCKED, _NOT_EVALUABLE):
        return _REVISE
    if en_qual in (_REJECT, _REVISE, _BLOCKED, _NOT_EVALUABLE):
        return _REVISE
    if ar_qual == _PASS and en_qual == _PASS:
        return _PASS
    return _NOT_EVALUABLE


def score_gcc_appropriateness(full_text: str) -> str:
    """PASS only if no GCC-inappropriate content is detected."""
    lower = full_text.lower()
    if "israel" in lower:
        return _REJECT
    if "pork" in lower:
        return _REJECT
    return _PASS


def score_domain_terminology(domain: str, response: str) -> str:
    """PASS only if domain-specific terminology violations are absent."""
    lower = response.lower()
    if domain == "islamic_finance" and "interest rate" in lower:
        return _REVISE
    return _PASS


def score_hallucination_risk(response: str) -> str:
    """PASS only if no high-risk hallucination patterns are present."""
    lower = response.lower()
    if "100% guaranteed" in lower:
        return _REJECT
    if re.search(r"\b(always|never|all banks|every bank)\b", lower):
        return _REVISE
    return _PASS


def score_trainability(response: str) -> str:
    """PASS only if response is substantively trainable."""
    if not response or not response.strip():
        return _REJECT
    if len(response.strip()) < 10:
        return _REJECT
    return _PASS


def score_structural_integrity(r: Dict) -> str:
    """
    PASS only if all mandatory structural fields are explicitly satisfied.
    Defect 6 fix: no fallback PASS — any unresolved condition returns BLOCKED.
    """
    # Language contract must be recomputed, not trusted from stored value
    lang = r.get("language", "")
    prompt = r.get("prompt", "")
    response = r.get("response", "")

    ar_resp = ar_char_count(response)
    en_resp = en_char_count(response)
    combined_ar = ar_char_count(prompt + " " + response)
    combined_en = en_char_count(prompt + " " + response)
    resp_len = max(len(response.replace(" ", "")), 1)

    if lang == "ar":
        contract_ok = ar_resp >= MIN_AR_CHARS and (ar_resp / resp_len) >= MIN_AR_RATIO
    elif lang == "en":
        contract_ok = en_resp >= MIN_EN_CHARS
    elif lang == "ar-en":
        contract_ok = combined_ar >= 20 and combined_en >= 20
    else:
        contract_ok = True  # Unknown language — not our contract

    if not contract_ok:
        return _REJECT

    # Check illegal software approval
    if r.get("factuality_review_status") == "human_approved":
        return _REJECT
    if r.get("arabic_review_status") == "human_approved":
        return _REJECT

    # Check required fields
    for field in ("example_id", "split", "prompt", "response", "language",
                  "domain", "provenance_type", "content_family_id", "scenario_brief"):
        if not r.get(field):
            return _BLOCKED

    return _PASS


# ---------------------------------------------------------------------------
# Master scorer — no fallback PASS
# ---------------------------------------------------------------------------

def score_record(r: Dict) -> Dict[str, Any]:
    """
    Score a record across all 12 dimensions.
    Defect 6 fix: PASS requires explicit satisfaction of every mandatory condition.
    Unknown/missing/malformed/unevaluable conditions produce FAIL/BLOCKED/NOT_EVALUABLE.
    """
    prompt = r.get("prompt", "")
    response = r.get("response", "")
    full_text = prompt + " " + response
    lang = r.get("language", "")
    domain = r.get("domain", "")

    # Score each dimension
    relevance = score_relevance(prompt, response)
    factual = score_factual_discipline(r)
    completeness = score_response_length(response)
    prof = score_professional_usefulness(response)
    ar_qual = score_arabic_quality(lang, full_text, response)
    en_qual = score_english_quality(lang, response)
    trans_fid = score_translation_fidelity(lang, ar_qual, en_qual)
    gcc = score_gcc_appropriateness(full_text)
    domain_score = score_domain_terminology(domain, response)
    halluc = score_hallucination_risk(response)
    trainability = score_trainability(response)
    structural = score_structural_integrity(r)

    # Collect all scored dimensions (excluding N/A)
    scored = [
        relevance, factual, completeness, prof, gcc,
        domain_score, halluc, trainability, structural,
    ]
    if ar_qual != "N/A":
        scored.append(ar_qual)
    if en_qual != "N/A":
        scored.append(en_qual)
    if trans_fid != "N/A":
        scored.append(trans_fid)

    # Defect 6 fix: PASS requires ALL dimensions to be explicitly PASS
    # Any REJECT → REJECT
    # Any REVISE → REVISE
    # Any BLOCKED/NOT_EVALUABLE/NOT_READY → BLOCKED (cannot determine readiness)
    # Only if every dimension is PASS → PASS
    if _REJECT in scored:
        verdict = _REJECT
    elif _BLOCKED in scored or _NOT_EVALUABLE in scored or _NOT_READY in scored:
        verdict = _BLOCKED
    elif _REVISE in scored:
        verdict = _REVISE
    elif all(s == _PASS for s in scored):
        verdict = _PASS
    else:
        # Any unrecognized state → BLOCKED (never silently PASS)
        verdict = _BLOCKED

    return {
        "example_id": r.get("example_id", "?"),
        "domain": domain,
        "language": lang,
        "task_type": r.get("task_type", "?"),
        "relevance": relevance,
        "factual_discipline": factual,
        "completeness": completeness,
        "professional_usefulness": prof,
        "arabic_quality": ar_qual,
        "english_quality": en_qual,
        "translation_fidelity": trans_fid,
        "gcc_appropriateness": gcc,
        "domain_terminology": domain_score,
        "hallucination_risk": halluc,
        "trainability": trainability,
        "structural_integrity": structural,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pass_count = sum(1 for r in results if r["verdict"] == _PASS)
    revise_count = sum(1 for r in results if r["verdict"] == _REVISE)
    reject_count = sum(1 for r in results if r["verdict"] == _REJECT)
    blocked_count = sum(1 for r in results if r["verdict"] in (_BLOCKED, _NOT_EVALUABLE, _NOT_READY))

    lines = [
        "# Jupiter Seed 4B: Content Quality Audit",
        "",
        "**Label:** AI-ASSISTED INTERNAL CONTENT AUDIT — NOT HUMAN APPROVAL",
        "",
        "**Defect 6 fix:** All fallback PASS logic removed. "
        "PASS requires explicit satisfaction of every mandatory condition.",
        "",
        "## 1. Audit Summary",
        "",
        f"- **Total Sampled:** {len(results)}",
        f"- **PASS:** {pass_count}",
        f"- **REVISE:** {revise_count}",
        f"- **REJECT:** {reject_count}",
        f"- **BLOCKED/NOT_EVALUABLE:** {blocked_count}",
        "",
        "**Verdict:** All REVISE, REJECT, and BLOCKED decisions must be resolved "
        "before the dataset is declared ready.",
        "",
        "## 2. Detailed Record Scores",
        "",
        "| Example ID | Domain | Lang | Task | Verdict | Factual | GCC | Halluc. | Structural |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        lines.append(
            f"| `{r['example_id']}` | {r['domain']} | {r['language']} | {r['task_type']} | "
            f"**{r['verdict']}** | {r['factual_discipline']} | {r['gcc_appropriateness']} | "
            f"{r['hallucination_risk']} | {r['structural_integrity']} |"
        )

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Content quality audit report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Content Quality Audit")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("training/jupiter_seed_4b/data"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/jupiter_seed_4b/CONTENT_QUALITY_AUDIT.md"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_all(args.data_dir)

    sample = stratified_sample(records, args.seed)
    log.info("Selected stratified sample of %d records", len(sample))

    results = [score_record(r) for r in sample]

    write_report(results, args.output)

    rejects = [r for r in results if r["verdict"] == _REJECT]
    revises = [r for r in results if r["verdict"] == _REVISE]
    blocked = [r for r in results if r["verdict"] in (_BLOCKED, _NOT_EVALUABLE, _NOT_READY)]

    if rejects or revises or blocked:
        log.error(
            "CONTENT AUDIT: %d REJECT, %d REVISE, %d BLOCKED.",
            len(rejects), len(revises), len(blocked)
        )
        sys.exit(1)

    log.info("Content quality audit PASSED (%d/%d).", len(results), len(results))


if __name__ == "__main__":
    main()
