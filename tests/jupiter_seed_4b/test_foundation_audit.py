"""
Jupiter Seed 4B — Foundation Audit Test Suite

Verifies structural and content integrity of all Seed 4B planning documents.
No training, no API calls, no cloud resources are used in these tests.
"""

import os
import re
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED_DOCS = os.path.join(REPO_ROOT, "docs", "jupiter_seed_4b")
LICENSE_MATRIX = os.path.join(SEED_DOCS, "LICENSE_AND_DISTILLATION_MATRIX.md")
TRAINING_PLAN = os.path.join(SEED_DOCS, "TRAINING_PLAN.md")
SCAFFOLDING_DIRS = [
    os.path.join(REPO_ROOT, "training", "jupiter_seed_4b"),
    os.path.join(REPO_ROOT, "benchmarks", "jupiter_seed_4b"),
    os.path.join(REPO_ROOT, "tests", "jupiter_seed_4b"),
]
STALE_VALIDATION_COMMIT = "5106291"
CORRECT_VALIDATION_COMMIT_PREFIX = "6ef21d4"


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Every model row has an exact model ID (backtick-quoted HF path)
# ──────────────────────────────────────────────────────────────────────────────
def test_model_rows_have_exact_model_id() -> None:
    content = read_file(LICENSE_MATRIX)
    # Rows in the table start with | `org/model-name`
    model_id_pattern = re.compile(r"\|\s*`[A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+`")
    matches = model_id_pattern.findall(content)
    assert len(matches) >= 7, (
        f"Expected at least 7 model rows with exact model IDs, found {len(matches)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Every model row has a direct licence URL
# ──────────────────────────────────────────────────────────────────────────────
def test_licence_urls_present() -> None:
    content = read_file(LICENSE_MATRIX)
    # The License URLs section must contain at least 7 URLs
    url_pattern = re.compile(r"https://(?:github\.com|huggingface\.co)/\S+")
    urls = url_pattern.findall(content)
    assert len(urls) >= 7, (
        f"Expected at least 7 licence URLs, found {len(urls)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Code and weight licences are separated
# ──────────────────────────────────────────────────────────────────────────────
def test_code_and_weight_licences_separated() -> None:
    content = read_file(LICENSE_MATRIX)
    assert "Model-Weight Licence" in content, "Column 'Model-Weight Licence' not found"
    assert "Code Licence" in content, "Column 'Code Licence' not found"
    # Ensure DeepSeek has both MIT and DeepSeek Model License mentioned
    assert "MIT" in content, "MIT licence not mentioned"
    assert "DeepSeek Model License" in content, "DeepSeek Model License not mentioned"


# ──────────────────────────────────────────────────────────────────────────────
# 4. No model is marked APPROVED without human-review evidence
# ──────────────────────────────────────────────────────────────────────────────
def test_no_model_marked_approved() -> None:
    content = read_file(LICENSE_MATRIX)
    # APPROVED must not appear in any table row (only PENDING LEGAL REVIEW is allowed)
    # Strip the header rows and check data rows
    lines = content.split("\n")
    data_rows = [l for l in lines if l.startswith("|") and "APPROVED" in l and "PENDING" not in l]
    assert len(data_rows) == 0, (
        f"Found model rows marked APPROVED without human review: {data_rows}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. No stale validation commit (5106291) remains in Seed documents
# ──────────────────────────────────────────────────────────────────────────────
def test_no_stale_validation_commit_in_seed_docs() -> None:
    for filename in os.listdir(SEED_DOCS):
        filepath = os.path.join(SEED_DOCS, filename)
        if os.path.isfile(filepath):
            content = read_file(filepath)
            assert STALE_VALIDATION_COMMIT not in content, (
                f"Stale validation commit '{STALE_VALIDATION_COMMIT}' found in {filename}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Student and teacher candidates are not conflated
# ──────────────────────────────────────────────────────────────────────────────
def test_student_and_teacher_sections_are_separate() -> None:
    content = read_file(LICENSE_MATRIX)
    student_pos = content.find("## Student Candidates")
    teacher_pos = content.find("## Teacher Candidates")
    assert student_pos != -1, "Student Candidates section not found"
    assert teacher_pos != -1, "Teacher Candidates section not found"
    assert student_pos < teacher_pos, (
        "Student Candidates section must appear before Teacher Candidates section"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. MoE total and active parameters are distinguished
# ──────────────────────────────────────────────────────────────────────────────
def test_moe_total_and_active_params_distinguished() -> None:
    content = read_file(LICENSE_MATRIX)
    assert "Parameter Count" in content, "Column 'Parameter Count' not found"
    assert "Active Parameters" in content, "Column 'Active Parameters (MoE)' not found"
    # DeepSeek-V3 is MoE with 671B total and 37B active
    assert "671B" in content, "DeepSeek-V3 total parameter count (671B) not found"
    assert "37B" in content, "DeepSeek-V3 active parameter count (37B) not found"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Empty scaffolding directories contain tracked placeholder files
# ──────────────────────────────────────────────────────────────────────────────
def test_scaffolding_dirs_have_gitkeep() -> None:
    for dirpath in SCAFFOLDING_DIRS:
        gitkeep = os.path.join(dirpath, ".gitkeep")
        assert os.path.isdir(dirpath), f"Scaffolding directory not found: {dirpath}"
        assert os.path.isfile(gitkeep), (
            f".gitkeep placeholder missing in {dirpath}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 9. No paid API or cloud-execution code was introduced
# ──────────────────────────────────────────────────────────────────────────────
def test_no_paid_api_or_cloud_code() -> None:
    forbidden_patterns = [
        r"openai\.api_key",
        r"anthropic\.Anthropic",
        r"boto3\.client",
        r"azure\.mgmt",
        r"AzureOpenAI",
        r"requests\.post.*openai\.com",
    ]
    training_dir = os.path.join(REPO_ROOT, "training", "jupiter_seed_4b")
    tests_dir = os.path.join(REPO_ROOT, "tests", "jupiter_seed_4b")
    this_file = os.path.abspath(__file__)
    for search_dir in [training_dir, tests_dir]:
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.abspath(os.path.join(root, fname))
                    if fpath == this_file:
                        continue  # skip self to avoid false positive on pattern strings
                    code = read_file(fpath)
                    for pattern in forbidden_patterns:
                        assert not re.search(pattern, code), (
                            f"Forbidden API pattern '{pattern}' found in {fpath}"
                        )
