"""
Jupiter Seed 4B V2.3 — Regex Regression Tests
================================================
Verifies the corrected GCC_TRIP_PHRASES patterns:
  - detect phrases at beginning, middle, and end of string
  - detect phrases followed by additional text
  - do NOT match on unrelated control strings
  - do NOT match phrases followed only by non-period punctuation (! ?)
    unless the pattern explicitly allows it
  - the old end-anchor behaviour is documented and NOT restored

Also verifies BLOCKING_CONTENT_PATTERNS boundary behaviour.

CPU-only. No corpus modification.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "training" / "jupiter_seed_4b"))

from readiness_gate import GCC_TRIP_PHRASES, BLOCKING_CONTENT_PATTERNS


# ─── Helper ───────────────────────────────────────────────────────────────

def matches_any_gcc(text: str) -> bool:
    return any(p.search(text) for p in GCC_TRIP_PHRASES)


def matches_any_blocking(text: str) -> bool:
    return any(p.search(text) for p, _ in BLOCKING_CONTENT_PATTERNS)


# ===========================================================================
# GCC_TRIP_PHRASES — Pattern 0 (English)
# ===========================================================================

class TestGCCEnglishPattern:
    """Tests for the English GCC trip phrase pattern."""

    PAT = GCC_TRIP_PHRASES[1]  # English pattern

    def test_repr_has_no_dollar_end_anchor(self) -> None:
        """Pattern must not end with '$' (end-of-string anchor)."""
        assert not self.PAT.pattern.endswith("$"), (
            f"Pattern must not end with '$'. Got: {repr(self.PAT.pattern)}"
        )

    def test_repr_is_correct(self) -> None:
        """Pattern repr must match the corrected form (no $ anchor)."""
        expected = r"specific current requirements should be verified[^\.]{0,60}\."
        assert self.PAT.pattern == expected, (
            f"Pattern mismatch. Expected: {repr(expected)} Got: {repr(self.PAT.pattern)}"
        )

    def test_matches_phrase_at_beginning(self) -> None:
        text = "Specific current requirements should be verified with the relevant authority."
        assert self.PAT.search(text), f"Must match phrase at beginning. Text: {repr(text)}"

    def test_matches_phrase_in_middle(self) -> None:
        text = "Please note that specific current requirements should be verified with the relevant authority. More text follows."
        assert self.PAT.search(text), f"Must match phrase in middle. Text: {repr(text)}"

    def test_matches_phrase_at_end(self) -> None:
        text = "All figures are indicative. Specific current requirements should be verified with the relevant authority."
        assert self.PAT.search(text), f"Must match phrase at end. Text: {repr(text)}"

    def test_matches_phrase_followed_by_additional_sentence(self) -> None:
        text = "Specific current requirements should be verified with the relevant authority. Please consult a licensed advisor."
        assert self.PAT.search(text), (
            f"Must match phrase followed by additional text. Text: {repr(text)}"
        )

    def test_does_not_match_phrase_followed_by_exclamation(self) -> None:
        """Pattern ends with \\. so it requires a period — ! should not match."""
        text = "Specific current requirements should be verified with the relevant authority!"
        assert not self.PAT.search(text), (
            f"Must NOT match phrase followed by '!' (no period). Text: {repr(text)}"
        )

    def test_does_not_match_unrelated_control(self) -> None:
        text = "This is a completely unrelated sentence about Murabaha financing."
        assert not self.PAT.search(text), (
            f"Must NOT match unrelated control string. Text: {repr(text)}"
        )

    def test_matches_seed4b_train_0032_response(self) -> None:
        """Must match the actual response of seed4b-train-0032."""
        import json
        data_dir = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
        path = data_dir / "train.jsonl"
        if not path.exists():
            pytest.skip("train.jsonl not found")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] == "seed4b-train-0032":
                    assert self.PAT.search(r["response"]), (
                        f"Must match seed4b-train-0032 response"
                    )
                    return
        pytest.skip("seed4b-train-0032 not found")

    def test_old_end_anchor_pattern_fails_mid_text(self) -> None:
        """Regression: the OLD pattern (with $) must fail on mid-text occurrence."""
        old_pat = re.compile(
            r"specific current requirements should be verified[^\.]{0,60}\.$",
            re.IGNORECASE
        )
        text = "Please note that specific current requirements should be verified with the relevant authority. More text follows."
        assert not old_pat.search(text), (
            "OLD pattern with $ anchor must NOT match mid-text phrase — "
            "this confirms the old pattern was defective"
        )

    def test_new_pattern_matches_where_old_failed(self) -> None:
        """New pattern must match where the old end-anchor pattern failed."""
        text = "Please note that specific current requirements should be verified with the relevant authority. More text follows."
        assert self.PAT.search(text), (
            "New pattern must match mid-text phrase where old $ pattern failed"
        )


# ===========================================================================
# GCC_TRIP_PHRASES — Pattern 1 (Arabic)
# ===========================================================================

class TestGCCArabicPattern:
    """Tests for the Arabic GCC trip phrase pattern."""

    PAT = GCC_TRIP_PHRASES[0]  # Arabic pattern

    def test_repr_has_no_dollar_end_anchor(self) -> None:
        """Pattern must not end with '$' (end-of-string anchor)."""
        assert not self.PAT.pattern.endswith("$"), (
            f"Pattern must not end with '$'. Got: {repr(self.PAT.pattern)}"
        )

    def test_repr_is_correct(self) -> None:
        """Pattern repr must match the corrected form (no $ anchor)."""
        expected = "يُنصح بمراجعة[^\\.]{0,50}\\."
        assert self.PAT.pattern == expected, (
            f"Pattern mismatch. Expected: {repr(expected)} Got: {repr(self.PAT.pattern)}"
        )

    def test_matches_arabic_phrase_at_beginning(self) -> None:
        text = "يُنصح بمراجعة الجهات المختصة."
        assert self.PAT.search(text), f"Must match Arabic phrase at beginning. Text: {repr(text)}"

    def test_matches_arabic_phrase_in_middle(self) -> None:
        text = "هذا النص يحتوي على معلومات مهمة. يُنصح بمراجعة الجهات المختصة. ثم نص إضافي."
        assert self.PAT.search(text), f"Must match Arabic phrase in middle. Text: {repr(text)}"

    def test_matches_arabic_phrase_at_end(self) -> None:
        text = "للمزيد من المعلومات يُنصح بمراجعة الجهات المختصة."
        assert self.PAT.search(text), f"Must match Arabic phrase at end. Text: {repr(text)}"

    def test_matches_arabic_phrase_followed_by_additional_sentence(self) -> None:
        text = "يُنصح بمراجعة الجهات المختصة. يرجى التواصل مع مستشار مرخص."
        assert self.PAT.search(text), (
            f"Must match Arabic phrase followed by additional text. Text: {repr(text)}"
        )

    def test_does_not_match_arabic_phrase_followed_by_exclamation(self) -> None:
        """Pattern ends with \\. so it requires a period — ! should not match."""
        text = "يُنصح بمراجعة الجهات المختصة!"
        assert not self.PAT.search(text), (
            f"Must NOT match Arabic phrase followed by '!' (no period). Text: {repr(text)}"
        )

    def test_does_not_match_unrelated_arabic_control(self) -> None:
        text = "هذه جملة غير ذات صلة تتحدث عن تمويل المرابحة."
        assert not self.PAT.search(text), (
            f"Must NOT match unrelated Arabic control string. Text: {repr(text)}"
        )

    def test_matches_seed4b_train_0038_response(self) -> None:
        """Must match the actual response of seed4b-train-0038."""
        import json
        data_dir = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
        path = data_dir / "train.jsonl"
        if not path.exists():
            pytest.skip("train.jsonl not found")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] == "seed4b-train-0038":
                    assert self.PAT.search(r["response"]), (
                        f"Must match seed4b-train-0038 response"
                    )
                    return
        pytest.skip("seed4b-train-0038 not found")

    def test_matches_seed4b_valid_0005_response(self) -> None:
        """Must match the actual response of seed4b-valid-0005."""
        import json
        data_dir = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"
        path = data_dir / "valid.jsonl"
        if not path.exists():
            pytest.skip("valid.jsonl not found")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["example_id"] == "seed4b-valid-0005":
                    assert self.PAT.search(r["response"]), (
                        f"Must match seed4b-valid-0005 response"
                    )
                    return
        pytest.skip("seed4b-valid-0005 not found")

    def test_old_end_anchor_pattern_fails_mid_text(self) -> None:
        """Regression: the OLD Arabic pattern (with $) must fail on mid-text occurrence."""
        old_pat = re.compile(r"يُنصح بمراجعة[^\.]{0,50}\.$", re.UNICODE)
        text = "هذا النص يحتوي على معلومات مهمة. يُنصح بمراجعة الجهات المختصة. ثم نص إضافي."
        assert not old_pat.search(text), (
            "OLD Arabic pattern with $ anchor must NOT match mid-text phrase — "
            "this confirms the old pattern was defective"
        )

    def test_new_arabic_pattern_matches_where_old_failed(self) -> None:
        """New Arabic pattern must match where the old end-anchor pattern failed."""
        text = "هذا النص يحتوي على معلومات مهمة. يُنصح بمراجعة الجهات المختصة. ثم نص إضافي."
        assert self.PAT.search(text), (
            "New Arabic pattern must match mid-text phrase where old $ pattern failed"
        )


# ===========================================================================
# BLOCKING_CONTENT_PATTERNS boundary tests
# ===========================================================================

class TestBlockingContentPatterns:
    """Boundary tests for BLOCKING_CONTENT_PATTERNS."""

    def test_100_percent_guaranteed_matches(self) -> None:
        text = "This investment is 100% guaranteed to return your principal."
        assert matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must match '100% guaranteed'"
        )

    def test_100_percent_guaranteed_case_insensitive(self) -> None:
        text = "This investment is 100% GUARANTEED to return your principal."
        assert matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must match '100% GUARANTEED' (case-insensitive)"
        )

    def test_100_percent_guaranteed_mid_text(self) -> None:
        text = "The fund manager claims 100% guaranteed returns on all deposits."
        assert matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must match '100% guaranteed' in mid-text"
        )

    def test_placeholder_matches(self) -> None:
        text = "[placeholder]"
        assert matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must match '[placeholder]'"
        )

    def test_unrelated_text_does_not_match_blocking(self) -> None:
        text = "Murabaha is a cost-plus financing structure used in Islamic banking."
        assert not matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must NOT match unrelated text"
        )

    def test_empty_string_matches_blocking(self) -> None:
        """Empty response must be caught by BLOCKING_CONTENT_PATTERNS."""
        text = ""
        assert matches_any_blocking(text), (
            "BLOCKING_CONTENT_PATTERNS must match empty string"
        )
