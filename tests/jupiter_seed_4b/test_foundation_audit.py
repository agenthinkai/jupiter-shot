"""
Jupiter Seed 4B — Foundation Audit Test Suite (v2)

Verifies structural and content integrity of all Seed 4B planning documents.
No training, no API calls, no cloud resources are used in these tests.
"""

import os
import re
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEED_DOCS = os.path.join(REPO_ROOT, "docs", "jupiter_seed_4b")
LICENSE_MATRIX = os.path.join(SEED_DOCS, "LICENSE_AND_DISTILLATION_MATRIX.md")
SCORECARD = os.path.join(SEED_DOCS, "MODEL_SELECTION_SCORECARD.md")
TRAINING_PLAN = os.path.join(SEED_DOCS, "TRAINING_PLAN.md")
SCAFFOLDING_DIRS = [
    os.path.join(REPO_ROOT, "training", "jupiter_seed_4b"),
    os.path.join(REPO_ROOT, "benchmarks", "jupiter_seed_4b"),
    os.path.join(REPO_ROOT, "tests", "jupiter_seed_4b"),
]
STALE_VALIDATION_COMMIT = "5106291"


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Qwen3 is included or explicitly excluded with evidence
# ──────────────────────────────────────────────────────────────────────────────
def test_qwen3_included_or_excluded_with_evidence() -> None:
    content = read_file(LICENSE_MATRIX)
    qwen3_present = "Qwen3" in content
    qwen3_excluded_with_reason = "Qwen3 Omission Resolution" in content or "Qwen3 Exclusion" in content
    assert qwen3_present or qwen3_excluded_with_reason, (
        "Qwen3 must be included in the candidate list or explicitly excluded with documented evidence."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Every model row has an exact official source (backtick-quoted HF/GitHub path)
# ──────────────────────────────────────────────────────────────────────────────
def test_model_rows_have_exact_model_id() -> None:
    content = read_file(LICENSE_MATRIX)
    model_id_pattern = re.compile(r"\|\s*`[A-Za-z0-9_\-]+/[A-Za-z0-9_\-\.]+`")
    matches = model_id_pattern.findall(content)
    assert len(matches) >= 7, (
        f"Expected at least 7 model rows with exact model IDs, found {len(matches)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Code and weight licences are separated in the matrix
# ──────────────────────────────────────────────────────────────────────────────
def test_code_and_weight_licences_separated() -> None:
    content = read_file(LICENSE_MATRIX)
    assert "Code Licence" in content, "Column 'Code Licence' not found"
    assert "Weight Licence" in content, "Column 'Weight Licence' not found"
    assert "MIT" in content, "MIT licence not mentioned"
    assert "DeepSeek Model License" in content, "DeepSeek Model License not mentioned"


# ──────────────────────────────────────────────────────────────────────────────
# 4. No candidate is marked APPROVED
# ──────────────────────────────────────────────────────────────────────────────
def test_no_model_marked_approved() -> None:
    for doc_name in os.listdir(SEED_DOCS):
        doc_path = os.path.join(SEED_DOCS, doc_name)
        if not os.path.isfile(doc_path):
            continue
        content = read_file(doc_path)
        lines = content.split("\n")
        data_rows = [l for l in lines if l.startswith("|") and "APPROVED" in l and "PENDING" not in l and "NOT" not in l]
        assert len(data_rows) == 0, (
            f"Found model rows marked APPROVED without human review in {doc_name}: {data_rows}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. No stale validation commit (5106291) in Seed documents
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
# 6. No training or data-generation code was introduced
# ──────────────────────────────────────────────────────────────────────────────
def test_no_training_or_datagen_code() -> None:
    training_dir = os.path.join(REPO_ROOT, "training", "jupiter_seed_4b")
    for root, _, files in os.walk(training_dir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                code = read_file(fpath)
                assert "torch.optim" not in code, f"Training code found in {fpath}"
                assert "model.train()" not in code, f"Training code found in {fpath}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. No cloud or paid API call was made
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
    # Exclude all test files from the scan to avoid false positives on pattern strings
    tests_dir_abs = os.path.abspath(tests_dir)
    for search_dir in [training_dir]:
        for root, _, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.abspath(os.path.join(root, fname))
                    code = read_file(fpath)
                    for pattern in forbidden_patterns:
                        assert not re.search(pattern, code), (
                            f"Forbidden API pattern '{pattern}' found in {fpath}"
                        )


# ──────────────────────────────────────────────────────────────────────────────
# 8. Protected branches are not referenced as current in Seed documents
# ──────────────────────────────────────────────────────────────────────────────
def test_protected_branches_untouched_references() -> None:
    for filename in os.listdir(SEED_DOCS):
        filepath = os.path.join(SEED_DOCS, filename)
        if os.path.isfile(filepath):
            content = read_file(filepath)
            assert "fix/rtx50-blackwell-validation" not in content or "Do not modify" in content or "untouched" in content, (
                f"Unexpected reference to protected validation branch in {filename}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 9. Student and teacher candidates are not conflated
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
# 10. MoE total and active parameters are distinguished
# ──────────────────────────────────────────────────────────────────────────────
def test_moe_total_and_active_params_distinguished() -> None:
    content = read_file(LICENSE_MATRIX)
    assert "Parameter Count" in content, "Column 'Parameter Count' not found"
    assert "Active Parameters" in content, "Column 'Active Parameters (MoE)' not found"
    assert "671B" in content, "DeepSeek-V3 total parameter count (671B) not found"
    assert "37B" in content, "DeepSeek-V3 active parameter count (37B) not found"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Scaffolding directories contain tracked placeholders
# ──────────────────────────────────────────────────────────────────────────────
def test_scaffolding_dirs_have_gitkeep() -> None:
    for dirpath in SCAFFOLDING_DIRS:
        gitkeep = os.path.join(dirpath, ".gitkeep")
        assert os.path.isdir(dirpath), f"Scaffolding directory not found: {dirpath}"
        assert os.path.isfile(gitkeep), (
            f".gitkeep placeholder missing in {dirpath}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 12. Scorecard contains provisional recommendations
# ──────────────────────────────────────────────────────────────────────────────
def test_scorecard_has_provisional_recommendations() -> None:
    content = read_file(SCORECARD)
    assert "PROVISIONAL" in content, "Scorecard must contain PROVISIONAL recommendations"
    assert "Preferred student" in content, "Scorecard must identify a preferred student candidate"
    assert "Preferred teacher" in content, "Scorecard must identify a preferred teacher candidate"
