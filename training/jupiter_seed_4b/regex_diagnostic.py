"""
GCC_TRIP_PHRASES Regex Diagnostic
===================================
Diagnoses the exact semantics of the two GCC_TRIP_PHRASES patterns,
documents whether $ is an end-anchor or literal, and tests each pattern
against six boundary cases.

Run: python3 training/jupiter_seed_4b/regex_diagnostic.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# ─── Step 1: Print repr() of each pattern ────────────────────────────────

print("=" * 70)
print("STEP 1: repr() of each GCC_TRIP_PHRASES entry")
print("=" * 70)

# Import the actual patterns from readiness_gate
sys.path.insert(0, str(Path(__file__).parent))
from readiness_gate import GCC_TRIP_PHRASES

for i, pat in enumerate(GCC_TRIP_PHRASES):
    print(f"\nPattern {i}: type={type(pat).__name__}")
    print(f"  repr(pattern.pattern) = {repr(pat.pattern)}")
    print(f"  flags = {pat.flags}")
    # Does the pattern string end with a literal '$'?
    raw = pat.pattern
    ends_with_dollar = raw.endswith("$") or raw.endswith(".$")
    print(f"  Pattern string ends with '$': {ends_with_dollar}")
    print(f"  In Python regex, '$' means: END-OF-STRING or END-OF-LINE anchor")
    print(f"  A literal dollar sign requires: \\$ or [$]")

# ─── Step 2: Boundary tests for each pattern ─────────────────────────────

print("\n" + "=" * 70)
print("STEP 2: Boundary tests for each pattern")
print("=" * 70)

# Pattern 0: English GCC trip phrase
pat0 = GCC_TRIP_PHRASES[0]
print(f"\n--- Pattern 0: {repr(pat0.pattern)} ---")

test_cases_0 = [
    ("at_beginning",
     "specific current requirements should be verified with the relevant authority.",
     "phrase at beginning of string"),
    ("in_middle",
     "Please note that specific current requirements should be verified with the relevant authority. More text follows.",
     "phrase in middle, followed by more text"),
    ("at_end",
     "All figures are indicative. Specific current requirements should be verified with the relevant authority.",
     "phrase at end of string"),
    ("followed_by_punctuation",
     "Specific current requirements should be verified with the relevant authority!",
     "phrase followed by ! not ."),
    ("followed_by_additional_text",
     "Specific current requirements should be verified with the relevant authority. Please consult a licensed advisor.",
     "phrase followed by additional sentence"),
    ("unrelated_control",
     "This is a completely unrelated sentence about Murabaha financing.",
     "unrelated control string"),
]

for label, text, description in test_cases_0:
    m = pat0.search(text)
    if m:
        print(f"  [{label}] MATCH: span={m.span()} excerpt={repr(text[max(0,m.start()-10):m.end()+10])}")
    else:
        print(f"  [{label}] NO MATCH — {description}")

# Pattern 1: Arabic GCC trip phrase
pat1 = GCC_TRIP_PHRASES[1]
print(f"\n--- Pattern 1: {repr(pat1.pattern)} ---")

test_cases_1 = [
    ("at_beginning",
     "يُنصح بمراجعة الجهات المختصة.",
     "Arabic phrase at beginning"),
    ("in_middle",
     "هذا النص يحتوي على يُنصح بمراجعة الجهات المختصة. ثم نص إضافي.",
     "Arabic phrase in middle"),
    ("at_end",
     "للمزيد من المعلومات يُنصح بمراجعة الجهات المختصة.",
     "Arabic phrase at end"),
    ("followed_by_punctuation",
     "يُنصح بمراجعة الجهات المختصة!",
     "Arabic phrase followed by ! not ."),
    ("followed_by_additional_text",
     "يُنصح بمراجعة الجهات المختصة. يرجى التواصل مع مستشار مرخص.",
     "Arabic phrase followed by additional sentence"),
    ("unrelated_control",
     "هذه جملة غير ذات صلة تتحدث عن تمويل المرابحة.",
     "unrelated Arabic control string"),
]

for label, text, description in test_cases_1:
    m = pat1.search(text)
    if m:
        print(f"  [{label}] MATCH: span={m.span()} excerpt={repr(text[max(0,m.start()-5):m.end()+10])}")
    else:
        print(f"  [{label}] NO MATCH — {description}")

# ─── Step 3: Show exact matched substring for the three corpus records ────

print("\n" + "=" * 70)
print("STEP 3: Exact matched substring for the three flagged corpus records")
print("=" * 70)

flagged_ids = {"seed4b-train-0032", "seed4b-train-0038", "seed4b-valid-0005"}
data_dir = REPO_ROOT / "training" / "jupiter_seed_4b" / "data"

for split in ("train", "valid", "eval"):
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] not in flagged_ids:
                continue
            print(f"\nRecord: {r['example_id']}")
            for field in ("prompt", "response"):
                text = r.get(field, "")
                for i, pat in enumerate(GCC_TRIP_PHRASES):
                    m = pat.search(text)
                    if m:
                        print(f"  Field: {field}")
                        print(f"  Pattern {i}: {repr(pat.pattern)}")
                        print(f"  Match span: {m.span()}")
                        print(f"  Matched text: {repr(m.group())}")
                        start = max(0, m.start() - 20)
                        end = min(len(text), m.end() + 20)
                        print(f"  Context: {repr(text[start:end])}")
                        print(f"  Text length: {len(text)}")
                        print(f"  Match ends at position {m.end()} of {len(text)}")
                        print(f"  Is match at end of string: {m.end() == len(text)}")

# ─── Step 4: Diagnosis and conclusion ────────────────────────────────────

print("\n" + "=" * 70)
print("STEP 4: Diagnosis")
print("=" * 70)

print("""
Pattern 0 ends with: .$
  In Python regex: '.' matches any character (except newline by default)
  '$' is an END-OF-STRING anchor (or end-of-line with re.MULTILINE)
  Therefore '.$' means: any-char followed by end-of-string
  This means the pattern only matches if the phrase ends at the end of the string.

Pattern 1 ends with: .$
  Same analysis: '.$' = any-char + end-of-string anchor
  This means the Arabic phrase must appear at the end of the string to match.

CONCLUSION:
  Both patterns use '$' as an END-OF-STRING anchor, NOT a literal dollar sign.
  The patterns are OVERLY RESTRICTIVE: they only match when the phrase
  appears at the very end of the string (or line).
  
  If a sentence like "specific current requirements should be verified with
  the relevant authority. More text follows." is present, the pattern will
  NOT match because the phrase is not at the end.

  The three corpus records matched because their response fields happen to
  end with the matching phrase (or end with a period after the phrase).
  
  INTENDED BEHAVIOR (per task spec): detect these phrases anywhere in the text.
  CORRECTED PATTERN: remove the '$' anchor, or use a lookahead for sentence end.
""")

print("Diagnostic complete.")
