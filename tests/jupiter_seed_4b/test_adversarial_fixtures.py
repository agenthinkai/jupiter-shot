"""
Jupiter Seed 4B — Adversarial Regression Fixtures (Task 7)
===========================================================
Deliberately deceptive test fixtures proving the canonicalization engine
correctly detects all forms of artificial uniqueness markers and
near-duplicate content.

All fixtures must be detected as duplicates or members of the same content family.
CPU-only. No GPU, model downloads, paid APIs, or cloud resources.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = REPO_ROOT / "training" / "jupiter_seed_4b"
sys.path.insert(0, str(TRAINING_DIR))

from canonicalize import (
    canonicalize,
    canonical_hash,
    has_artificial_marker,
    jaccard,
    lcs_ratio,
    remove_artificial_markers,
    word_ngrams,
    JACCARD_BLOCK_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Base text for adversarial fixtures
# ---------------------------------------------------------------------------

BASE_RESPONSE = (
    "Murabaha is a cost-plus-profit sale where the bank discloses the cost "
    "and agreed profit margin. The bank must own the asset before selling it. "
    "This structure is widely used in GCC Islamic banking."
)

BASE_RESPONSE_AR = (
    "المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح. "
    "يشترط أن يكون البائع مالكاً للسلعة قبل البيع. "
    "تُستخدم هذه الصيغة على نطاق واسع في البنوك الإسلامية الخليجية."
)


# ---------------------------------------------------------------------------
# Fixture 1: [train-0034] versus [bench-0127]
# ---------------------------------------------------------------------------

def test_split_id_markers_detected_as_duplicate() -> None:
    text_a = BASE_RESPONSE + " [train-0034]"
    text_b = BASE_RESPONSE + " [bench-0127]"
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Split ID markers [train-0034] and [bench-0127] must not create distinct canonical hashes"
    )
    assert has_artificial_marker(text_a), "[train-0034] must be detected as artificial marker"
    assert has_artificial_marker(text_b), "[bench-0127] must be detected as artificial marker"


# ---------------------------------------------------------------------------
# Fixture 2: [v079] versus [v067]
# ---------------------------------------------------------------------------

def test_version_tags_detected_as_duplicate() -> None:
    text_a = BASE_RESPONSE + " [v079]"
    text_b = BASE_RESPONSE + " [v067]"
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Version tags [v079] and [v067] must not create distinct canonical hashes"
    )
    assert has_artificial_marker(text_a), "[v079] must be detected as artificial marker"
    assert has_artificial_marker(text_b), "[v067] must be detected as artificial marker"


# ---------------------------------------------------------------------------
# Fixture 3: [ref:001] versus [ref:002]
# ---------------------------------------------------------------------------

def test_ref_tags_detected_as_duplicate() -> None:
    text_a = BASE_RESPONSE + " [ref:001]"
    text_b = BASE_RESPONSE + " [ref:002]"
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Reference tags [ref:001] and [ref:002] must not create distinct canonical hashes"
    )
    assert has_artificial_marker(text_a), "[ref:001] must be detected as artificial marker"
    assert has_artificial_marker(text_b), "[ref:002] must be detected as artificial marker"


# ---------------------------------------------------------------------------
# Fixture 4: [مرجع: ١] versus [مرجع: ٢]
# ---------------------------------------------------------------------------

def test_arabic_ref_tags_detected_as_duplicate() -> None:
    text_a = BASE_RESPONSE_AR + " [مرجع: ١]"
    text_b = BASE_RESPONSE_AR + " [مرجع: ٢]"
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Arabic reference tags must not create distinct canonical hashes"
    )
    assert has_artificial_marker(text_a), "[مرجع: ١] must be detected as artificial marker"
    assert has_artificial_marker(text_b), "[مرجع: ٢] must be detected as artificial marker"


# ---------------------------------------------------------------------------
# Fixture 5: Different company names with identical substance
# ---------------------------------------------------------------------------

def test_company_name_substitution_detected() -> None:
    text_a = (
        "Gulf Finance Corp (fictional) reported a CET1 ratio of 15.2%, "
        "exceeding the CBK minimum requirement of 11%."
    )
    text_b = (
        "Al Noor Capital (fictional) reported a CET1 ratio of 15.2%, "
        "exceeding the CBK minimum requirement of 11%."
    )
    # After canonicalization (currency/percentage normalization), the substantive content
    # should be highly similar. Company names differ but the required answer is the same.
    c_a = canonicalize(text_a)
    c_b = canonicalize(text_b)
    ng_a = word_ngrams(c_a, 3)
    ng_b = word_ngrams(c_b, 3)
    j = jaccard(ng_a, ng_b)
    # Use bigrams for short texts where trigram overlap is lower
    ng_a2 = word_ngrams(c_a, 2)
    ng_b2 = word_ngrams(c_b, 2)
    j2 = jaccard(ng_a2, ng_b2)
    # Either trigram or bigram Jaccard must exceed the warning threshold (0.50)
    WARN_THRESHOLD = 0.50
    assert j >= WARN_THRESHOLD or j2 >= WARN_THRESHOLD, (
        f"Company name substitution must be detected as near-duplicate "
        f"(trigram Jaccard={j:.3f}, bigram Jaccard={j2:.3f}, threshold={WARN_THRESHOLD})"
    )


# ---------------------------------------------------------------------------
# Fixture 6: Different currencies with identical substance
# ---------------------------------------------------------------------------

def test_currency_substitution_detected() -> None:
    text_a = "The facility size is USD 500M with a tenor of 7 years."
    text_b = "The facility size is KWD 150M with a tenor of 7 years."
    # After currency normalization, both should be CURRENCY NUM with tenor NUM years
    c_a = canonicalize(text_a)
    c_b = canonicalize(text_b)
    assert c_a == c_b, (
        f"Currency substitution must produce identical canonical text.\n"
        f"  a: '{c_a}'\n  b: '{c_b}'"
    )


# ---------------------------------------------------------------------------
# Fixture 7: Different percentages with identical substance
# ---------------------------------------------------------------------------

def test_percentage_substitution_detected() -> None:
    text_a = "The profit rate is 4.25% per annum."
    text_b = "The profit rate is 5.50% per annum."
    c_a = canonicalize(text_a)
    c_b = canonicalize(text_b)
    assert c_a == c_b, (
        f"Percentage substitution must produce identical canonical text.\n"
        f"  a: '{c_a}'\n  b: '{c_b}'"
    )


# ---------------------------------------------------------------------------
# Fixture 8: Same answer with reordered sentences
# ---------------------------------------------------------------------------

def test_reordered_sentences_detected_as_near_duplicate() -> None:
    text_a = (
        "Murabaha requires the bank to own the asset before selling it. "
        "The profit margin must be disclosed. "
        "This structure avoids riba."
    )
    text_b = (
        "The profit margin must be disclosed. "
        "This structure avoids riba. "
        "Murabaha requires the bank to own the asset before selling it."
    )
    ng_a = word_ngrams(canonicalize(text_a), 3)
    ng_b = word_ngrams(canonicalize(text_b), 3)
    j = jaccard(ng_a, ng_b)
    assert j >= JACCARD_BLOCK_THRESHOLD, (
        f"Reordered sentences must be detected as near-duplicate (Jaccard={j:.3f})"
    )


# ---------------------------------------------------------------------------
# Fixture 9: Same answer with changed punctuation
# ---------------------------------------------------------------------------

def test_punctuation_variation_detected_as_duplicate() -> None:
    text_a = "The bank must own the asset; the profit margin must be disclosed."
    text_b = "The bank must own the asset, the profit margin must be disclosed."
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Punctuation-only variation must produce identical canonical hash"
    )


# ---------------------------------------------------------------------------
# Fixture 10: English and Arabic decorative tags
# ---------------------------------------------------------------------------

def test_decorative_tags_removed() -> None:
    text_a = BASE_RESPONSE + " [ref/train-0001 / مرجع: 0001]"
    text_b = BASE_RESPONSE
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Decorative English/Arabic tags must be removed during canonicalization"
    )
    assert has_artificial_marker(text_a), "Decorative tag must be detected as artificial marker"


# ---------------------------------------------------------------------------
# Fixture 11: Whitespace-only variation
# ---------------------------------------------------------------------------

def test_whitespace_variation_detected_as_duplicate() -> None:
    text_a = "The bank  must  own  the  asset."
    text_b = "The bank must own the asset."
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Whitespace-only variation must produce identical canonical hash"
    )


# ---------------------------------------------------------------------------
# Fixture 12: Boilerplate-prefix variation
# ---------------------------------------------------------------------------

def test_boilerplate_prefix_variation_detected() -> None:
    text_a = "Note: All entities are fictional. " + BASE_RESPONSE
    text_b = BASE_RESPONSE
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Boilerplate prefix must be removed during canonicalization"
    )


# ---------------------------------------------------------------------------
# Fixture 13: Boilerplate-suffix variation
# ---------------------------------------------------------------------------

def test_boilerplate_suffix_variation_detected() -> None:
    text_a = BASE_RESPONSE + " Note: All entities are fictional and for training purposes only."
    text_b = BASE_RESPONSE
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Boilerplate suffix must be removed during canonicalization"
    )


# ---------------------------------------------------------------------------
# Fixture 14: Arabic boilerplate variation
# ---------------------------------------------------------------------------

def test_arabic_boilerplate_variation_detected() -> None:
    text_a = BASE_RESPONSE_AR + " ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."
    text_b = BASE_RESPONSE_AR
    assert canonical_hash(text_a) == canonical_hash(text_b), (
        "Arabic boilerplate suffix must be removed during canonicalization"
    )


# ---------------------------------------------------------------------------
# Fixture 15: Remove_artificial_markers cleans text correctly
# ---------------------------------------------------------------------------

def test_remove_artificial_markers_cleans_text() -> None:
    dirty = "The bank must own the asset [train-0034] before selling it [v079]."
    clean = remove_artificial_markers(dirty)
    assert "[train-0034]" not in clean, "Split ID not removed"
    assert "[v079]" not in clean, "Version tag not removed"
    assert "bank must own the asset" in clean, "Core content must be preserved"


# ---------------------------------------------------------------------------
# Fixture 16: Canonical hash is deterministic
# ---------------------------------------------------------------------------

def test_canonical_hash_is_deterministic() -> None:
    text = BASE_RESPONSE + " [train-0099]"
    h1 = canonical_hash(text)
    h2 = canonical_hash(text)
    assert h1 == h2, "Canonical hash must be deterministic"


# ---------------------------------------------------------------------------
# Fixture 17: Genuinely distinct texts have different canonical hashes
# ---------------------------------------------------------------------------

def test_genuinely_distinct_texts_have_different_hashes() -> None:
    text_a = (
        "Murabaha is a cost-plus-profit sale where the bank discloses the cost. "
        "The bank must own the asset before selling it."
    )
    text_b = (
        "Musharaka is a partnership where profits are shared according to agreed ratios. "
        "Losses are borne in proportion to capital contribution."
    )
    assert canonical_hash(text_a) != canonical_hash(text_b), (
        "Genuinely distinct texts must have different canonical hashes"
    )


# ---------------------------------------------------------------------------
# Fixture 18: Arabic text with different content has different canonical hash
# ---------------------------------------------------------------------------

def test_arabic_distinct_texts_have_different_hashes() -> None:
    text_a = "المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح."
    text_b = "المضاربة عقد شراكة بين رب المال والمضارب الذي يُقدّم الخبرة والعمل."
    assert canonical_hash(text_a) != canonical_hash(text_b), (
        "Arabic texts with different content must have different canonical hashes"
    )


# ---------------------------------------------------------------------------
# Fixture 19: LCS ratio detects near-duplicate responses
# ---------------------------------------------------------------------------

def test_lcs_ratio_detects_near_duplicates() -> None:
    text_a = canonicalize(
        "The bank must own the asset before selling it. "
        "The profit margin must be disclosed to the buyer."
    )
    text_b = canonicalize(
        "The bank must own the asset before selling it. "
        "The profit margin must be clearly disclosed to the buyer."
    )
    ratio = lcs_ratio(text_a, text_b)
    assert ratio >= 0.80, (
        f"Near-duplicate texts must have LCS ratio >= 0.80, got {ratio:.3f}"
    )


# ---------------------------------------------------------------------------
# Fixture 20: has_artificial_marker returns False for clean text
# ---------------------------------------------------------------------------

def test_has_artificial_marker_false_for_clean_text() -> None:
    clean_texts = [
        BASE_RESPONSE,
        BASE_RESPONSE_AR,
        "The bank must own the asset before selling it.",
        "المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة.",
        "Note: This is a training example.",  # Not an artificial marker
    ]
    for text in clean_texts:
        assert not has_artificial_marker(text), (
            f"Clean text incorrectly flagged as having artificial marker: '{text[:60]}'"
        )
