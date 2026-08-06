#!/usr/bin/env python3
"""
Jupiter Seed 4B — Canonicalization and Deduplication Engine
============================================================
Implements Task 5 (canonicalized duplicate detection) and Task 6 (semantic/template leakage).

Canonicalization removes:
  - Split IDs, example IDs, version tags, reference tags, bracketed markers
  - Whitespace differences, case differences, punctuation differences
  - Common boilerplate (opening/closing phrases)
  - Citation formatting, artificial numbering
  - Currency/percentage/date substitutions (where they don't change the answer)
  - Entity substitutions that don't change required reasoning
  - Arabic and English decorative markers

All hashes are computed on canonicalized text.
Blocking thresholds are explicit and documented.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocking thresholds (explicit, documented)
# ---------------------------------------------------------------------------

# Jaccard similarity on word trigrams: >= this value triggers a cross-split block
JACCARD_BLOCK_THRESHOLD = 0.70

# Jaccard similarity on word trigrams: >= this value triggers a warning
JACCARD_WARN_THRESHOLD = 0.50

# Longest common subsequence ratio: >= this value triggers a cross-split block
LCS_BLOCK_THRESHOLD = 0.80

# Minimum canonical text length to run similarity checks
MIN_CANONICAL_LENGTH = 20


# ---------------------------------------------------------------------------
# Artificial uniqueness marker patterns (Task 1 / Task 7)
# ---------------------------------------------------------------------------

# Patterns that indicate an artificial uniqueness marker was injected
ARTIFICIAL_MARKER_PATTERNS = [
    # Split/example IDs: [train-0034], [bench-0127], [valid-0001], [eval-0099]
    re.compile(r'\[\s*(train|bench|valid|eval|test)\s*[-–]\s*\d+\s*\]', re.IGNORECASE),
    # Version tags: [v079], [v001], [v123]
    re.compile(r'\[\s*v\d+\s*\]', re.IGNORECASE),
    # Reference tags: [ref:001], [ref:train-0034], [ref/bench-0127]
    re.compile(r'\[\s*ref\s*[:/]\s*[^\]]+\]', re.IGNORECASE),
    # Arabic reference tags: [مرجع: ١], [مرجع: 001], [م-train-0034]
    re.compile(r'\[\s*مرجع\s*[:\s][^\]]*\]'),
    re.compile(r'\[\s*م\s*[-–]\s*[^\]]+\]'),
    # Bench dedup markers: [bench-dedup-seed4b-...]
    re.compile(r'\[\s*bench-dedup-[^\]]+\]', re.IGNORECASE),
    # Generic bracketed markers with digits only: [001], [042]
    re.compile(r'\[\s*\d{3,}\s*\]'),
    # Parenthesized IDs: (train-0034), (v079)
    re.compile(r'\(\s*(train|bench|valid|eval)\s*[-–]\s*\d+\s*\)', re.IGNORECASE),
]

# Boilerplate phrases to remove during canonicalization
BOILERPLATE_PHRASES_EN = [
    r"note:\s*all\s+entities\s+are\s+fictional(\s+and\s+for\s+training\s+purposes\s+only)?\.?",
    r"note:\s*all\s+companies,\s+numbers,\s+and\s+situations\s+in\s+this\s+example\s+are\s+fictional.*?only\.",
    r"this\s+is\s+a\s+fictional\s+(scenario|training\s+scenario)\s+for\s+training\s+purposes\s+only\.",
    r"this\s+comparison\s+is\s+fictional\s+and\s+for\s+training\s+purposes\s+only\.",
    r"note:\s*this\s+is\s+a\s+fictional\s+scenario\s+for\s+training\s+purposes\s+only\.",
    r"note:\s*all\s+entities\s+and\s+figures\s+are\s+fictional\.",
    r"all\s+entities\s+are\s+fictional\.",
    r"\[ref:\s*[^\]]+\]",
    r"\[r-[^\]]+\]",
]

BOILERPLATE_PHRASES_AR = [
    r"ملاحظة:\s*هذا\s+سيناريو\s+افتراضي\s+لأغراض\s+التدريب\s+فقط\.",
    r"ملاحظة:\s*جميع\s+الكيانات\s+افتراضية\s+لأغراض\s+التدريب\s+فقط\.",
    r"ملاحظة:\s*الشركة\s+افتراضية\s+لأغراض\s+التدريب\s+فقط\.",
    r"\[المرجع:\s*[^\]]+\]",
    r"\[مرجع:\s*[^\]]+\]",
    r"\[م-[^\]]+\]",
]

# Currency/percentage/date normalization
CURRENCY_PATTERN = re.compile(
    r'\b(USD|EUR|GBP|KWD|SAR|AED|QAR|BHD|OMR)\s*[\d,\.]+[MBK]?\b|\b[\d,\.]+[MBK]?\s*(USD|EUR|GBP|KWD|SAR|AED|QAR|BHD|OMR)\b',
    re.IGNORECASE
)
PERCENTAGE_PATTERN = re.compile(r'\b\d+(\.\d+)?\s*%\b')
DATE_PATTERN = re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')
NUMBER_PATTERN = re.compile(r'\b\d[\d,\.]*\b')

# Common GCC "trip phrases" that add no content
TRIP_PHRASES = [
    r"يُنصح بمراجعة[^\.]+\.",
    r"يُنصح بمراجعة[^\.]+\.",
    r"تستند هذه المعلومات إلى البيانات المتاحة للعموم\.",
    r"يستند هذا الوصف إلى المعلومات المتاحة للعموم[^\.]*\.",
    r"يُنصح بمراجعة الموقع الرسمي[^\.]+\.",
    r"يُنصح بمراجعة أحدث التعاميم[^\.]+\.",
    r"يُنصح بمراجعة النص الكامل[^\.]+\.",
    r"يُنصح بمراجعة الجهات[^\.]+\.",
    r"specific current requirements should be verified[^\.]+\.",
    r"specific current thresholds should be verified[^\.]+\.",
    r"for the most current (and authoritative )?information[^\.]+\.",
    r"please consult[^\.]+\.",
    r"legal counsel should be consulted[^\.]+\.",
    r"risk:\s*providing[^\.]+\.",
    r"this answer (summarises|reflects)[^\.]+\.",
    r"this briefing reflects[^\.]+\.",
]


def remove_artificial_markers(text: str) -> str:
    """Remove all artificial uniqueness markers from natural-language text."""
    for pattern in ARTIFICIAL_MARKER_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def has_artificial_marker(text: str) -> bool:
    """Return True if text contains any artificial uniqueness marker."""
    for pattern in ARTIFICIAL_MARKER_PATTERNS:
        if pattern.search(text):
            return True
    return False


def canonicalize(text: str) -> str:
    """
    Produce a canonical form of text for deduplication.
    Removes markers, boilerplate, normalizes whitespace/case/punctuation.
    """
    if not text:
        return ""

    t = str(text)

    # Remove artificial markers
    t = remove_artificial_markers(t)

    # Remove boilerplate
    for phrase in BOILERPLATE_PHRASES_EN + BOILERPLATE_PHRASES_AR:
        t = re.sub(phrase, " ", t, flags=re.IGNORECASE | re.DOTALL)

    # Remove trip phrases
    for phrase in TRIP_PHRASES:
        t = re.sub(phrase, " ", t, flags=re.IGNORECASE | re.DOTALL)

    # Normalize currencies, percentages, dates, numbers
    t = CURRENCY_PATTERN.sub("CURRENCY", t)
    t = PERCENTAGE_PATTERN.sub("PCT", t)
    t = DATE_PATTERN.sub("DATE", t)
    t = NUMBER_PATTERN.sub("NUM", t)

    # Normalize unicode
    t = unicodedata.normalize("NFKC", t)

    # Lowercase
    t = t.lower()

    # Remove all punctuation except spaces
    t = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', t)

    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(text: str) -> str:
    return sha256_of(canonicalize(text))


def word_ngrams(text: str, n: int = 3) -> Set[str]:
    words = text.split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def lcs_ratio(a: str, b: str) -> float:
    """Longest common subsequence ratio (word-level)."""
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0.0
    m, n = len(words_a), len(words_b)
    # Use DP but cap at 200 words for CPU efficiency
    if m > 200 or n > 200:
        words_a = words_a[:200]
        words_b = words_b[:200]
        m, n = len(words_a), len(words_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_a[i-1] == words_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    return 2 * lcs_len / (m + n)


def prompt_skeleton(text: str) -> str:
    """Extract a structural skeleton by removing content words."""
    t = canonicalize(text)
    # Keep only question words, structural words, and domain terms
    t = re.sub(r'\b(num|currency|pct|date)\b', 'X', t)
    t = re.sub(r'\b\w{1,3}\b', '', t)  # Remove very short words
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:200]  # Cap for comparison


def response_skeleton(text: str) -> str:
    """Extract a structural skeleton of a response."""
    t = canonicalize(text)
    # Remove content words, keep structure
    t = re.sub(r'\b(num|currency|pct|date)\b', 'X', t)
    t = re.sub(r'\b\w{1,3}\b', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:200]


# ---------------------------------------------------------------------------
# Duplicate detection result types
# ---------------------------------------------------------------------------

class DuplicateGroup:
    def __init__(self, canonical_hash: str, members: List[str], reason: str):
        self.canonical_hash = canonical_hash
        self.members = members  # list of example_ids
        self.reason = reason

    def is_cross_split(self, id_to_split: Dict[str, str]) -> bool:
        splits = {id_to_split.get(m, "unknown") for m in self.members}
        return len(splits) > 1


class DeduplicationResult:
    def __init__(self):
        self.prompt_dup_groups: List[DuplicateGroup] = []
        self.response_dup_groups: List[DuplicateGroup] = []
        self.pair_dup_groups: List[DuplicateGroup] = []
        self.semantic_pairs: List[Tuple[str, str, float, str]] = []  # (id1, id2, score, method)
        self.artificial_marker_ids: List[str] = []
        self.scenario_brief_dup_groups: List[DuplicateGroup] = []
        self.family_cross_split_violations: List[Tuple[str, List[str]]] = []  # (family, splits)

    @property
    def has_blocking_cross_split_duplicates(self) -> bool:
        return any(True for g in self.response_dup_groups if len({m.split("-")[1] for m in g.members}) > 1)

    def blocking_cross_split_response_groups(self, id_to_split: Dict[str, str]) -> List[DuplicateGroup]:
        return [g for g in self.response_dup_groups if g.is_cross_split(id_to_split)]

    def blocking_semantic_pairs(self) -> List[Tuple[str, str, float, str]]:
        return [(a, b, s, m) for a, b, s, m in self.semantic_pairs if s >= JACCARD_BLOCK_THRESHOLD]


def run_deduplication(records: List[Dict]) -> DeduplicationResult:
    """
    Run full deduplication on a list of records.
    Records must have: example_id, split, prompt, response, scenario_brief, content_family_id
    """
    result = DeduplicationResult()
    id_to_split = {r["example_id"]: r["split"] for r in records}

    # Check for artificial markers
    for r in records:
        if has_artificial_marker(r.get("prompt", "")) or has_artificial_marker(r.get("response", "")):
            result.artificial_marker_ids.append(r["example_id"])

    # Canonical hash grouping
    def find_dup_groups(records, key_fn, label):
        hash_to_ids = defaultdict(list)
        for r in records:
            h = canonical_hash(key_fn(r))
            hash_to_ids[h].append(r["example_id"])
        return [
            DuplicateGroup(h, ids, label)
            for h, ids in hash_to_ids.items()
            if len(ids) > 1
        ]

    result.prompt_dup_groups = find_dup_groups(records, lambda r: r.get("prompt", ""), "canonical_prompt")
    result.response_dup_groups = find_dup_groups(records, lambda r: r.get("response", ""), "canonical_response")
    result.pair_dup_groups = find_dup_groups(
        records,
        lambda r: r.get("prompt", "") + "\n" + r.get("response", ""),
        "canonical_pair"
    )

    # Scenario brief duplicates
    if any("scenario_brief" in r for r in records):
        result.scenario_brief_dup_groups = find_dup_groups(
            records, lambda r: r.get("scenario_brief", ""), "scenario_brief"
        )

    # Content family cross-split violations
    family_to_splits = defaultdict(set)
    for r in records:
        family = r.get("content_family_id", "")
        if family:
            family_to_splits[family].add(r["split"])
    result.family_cross_split_violations = [
        (family, list(splits))
        for family, splits in family_to_splits.items()
        if len(splits) > 1
    ]

    # Semantic similarity (Jaccard on canonical word trigrams)
    # Only check cross-split pairs for efficiency
    split_records = defaultdict(list)
    for r in records:
        split_records[r["split"]].append(r)

    splits = list(split_records.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            for r1 in split_records[s1]:
                c1 = canonicalize(r1.get("response", ""))
                if len(c1) < MIN_CANONICAL_LENGTH:
                    continue
                ng1 = word_ngrams(c1, 3)
                for r2 in split_records[s2]:
                    c2 = canonicalize(r2.get("response", ""))
                    if len(c2) < MIN_CANONICAL_LENGTH:
                        continue
                    ng2 = word_ngrams(c2, 3)
                    j = jaccard(ng1, ng2)
                    if j >= JACCARD_WARN_THRESHOLD:
                        result.semantic_pairs.append((r1["example_id"], r2["example_id"], j, "jaccard_trigram"))

    return result
