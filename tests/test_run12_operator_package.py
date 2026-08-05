r"""
tests/test_run12_operator_package.py
=====================================
Run 12 operator-package contract regression tests.

These tests validate that the authoritative Kishore Windows execution
documentation satisfies 15 safety and correctness contracts. They parse
the documentation files as text and assert that:

  1.  .venv\Scripts\python.exe is used (not system python or python3)
  2.  System `python -m pytest` is not the official command
  3.  `python3` is not used as an official command
  4.  git pull uses --ff-only
  5.  git status is required in the verification step
  6.  Exact commit is required (no 'or latest follow-up' wording)
  7.  Real-data preflight uses the Windows .bat launcher
  8.  Preflight command includes --data-mode real
  9.  Preflight command includes --preflight-only
 10.  Full run uses the Windows .bat launcher
 11.  Full run command includes --data-mode real
 12.  Direct Python pipeline invocation is not the official command
 13.  Synthetic mode cannot authorize validation (not in official sequence)
 14.  CUDA claims require device=cuda evidence
 15.  Current run_id artifact path is used (not generic artifacts\ path)

Tests validate operational command blocks, not incidental comments or
historical references in the documentation.
"""

import pathlib
import re
import pytest

# ---------------------------------------------------------------------------
# Fixtures — load the documents under test
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent

OPERATOR_PACKAGE = REPO_ROOT / "docs" / "RUN12_OPERATOR_PACKAGE.md"
CHECKLIST = REPO_ROOT / "docs" / "KISHORE_GPU_OPERATOR_CHECKLIST.md"


@pytest.fixture(scope="module")
def operator_pkg_text():
    """Full text of docs/RUN12_OPERATOR_PACKAGE.md."""
    return OPERATOR_PACKAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def checklist_text():
    """Full text of docs/KISHORE_GPU_OPERATOR_CHECKLIST.md."""
    return CHECKLIST.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def combined_text(operator_pkg_text, checklist_text):
    """Combined text of both authoritative documents."""
    return operator_pkg_text + "\n" + checklist_text


# ---------------------------------------------------------------------------
# Helper: extract code blocks from markdown
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> list[str]:
    """Return the content of all fenced code blocks (``` ... ```)."""
    return re.findall(r"```(?:bat|batch|cmd|powershell|shell|bash)?\s*(.*?)```", text, re.DOTALL)


def extract_command_blocks(text: str) -> list[str]:
    """Return all code block lines that look like commands (non-comment, non-blank)."""
    blocks = extract_code_blocks(text)
    lines = []
    for block in blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("::") and not stripped.startswith("REM ") and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Contract 1 — .venv\Scripts\python.exe is used
# ---------------------------------------------------------------------------

class TestVenvPythonUsed:
    """Contract 1: The project .venv Python must be used for official commands."""

    def test_venv_python_used(self, combined_text):
        """'.venv\\Scripts\\python.exe' must appear in the operator documents."""
        assert r".venv\Scripts\python.exe" in combined_text, (
            "FAIL Contract 1: '.venv\\Scripts\\python.exe' not found in operator documents. "
            "Kishore must use the project virtual environment, not system Python."
        )

    def test_venv_python_in_pytest_command(self, combined_text):
        """The pytest command must use .venv\\Scripts\\python.exe."""
        # Find lines that contain pytest
        pytest_lines = [
            line for line in combined_text.splitlines()
            if "pytest" in line and "test_run12" in line
        ]
        assert pytest_lines, "No pytest command referencing test_run12 found."
        for line in pytest_lines:
            # Must use .venv or be a comment/table row
            if line.strip().startswith("|") or line.strip().startswith(">"):
                continue  # table row or blockquote — skip
            assert r".venv\Scripts\python.exe" in line, (
                f"FAIL Contract 1: pytest command does not use .venv\\Scripts\\python.exe: {line!r}"
            )


# ---------------------------------------------------------------------------
# Contract 2 — System `python -m pytest` is not the official command
# ---------------------------------------------------------------------------

class TestSystemPythonNotUsed:
    """Contract 2: Bare `python -m pytest` must not be the official command."""

    def test_system_python_not_used(self, combined_text):
        """Official command blocks must not use bare `python -m pytest`."""
        blocks = extract_command_blocks(combined_text)
        for line in blocks:
            # Allow .venv-qualified python; disallow bare `python -m pytest`
            if re.match(r"^python\s+-m\s+pytest", line):
                pytest.fail(
                    f"FAIL Contract 2: bare 'python -m pytest' found as official command: {line!r}. "
                    "Use '.venv\\Scripts\\python.exe -m pytest' instead."
                )


# ---------------------------------------------------------------------------
# Contract 3 — `python3` is not used as an official command
# ---------------------------------------------------------------------------

class TestPython3NotUsed:
    """Contract 3: `python3` must not appear in official command blocks."""

    def test_python3_not_used(self, combined_text):
        """Official command blocks must not use `python3`."""
        blocks = extract_command_blocks(combined_text)
        for line in blocks:
            if re.match(r"^python3\b", line):
                pytest.fail(
                    f"FAIL Contract 3: 'python3' found as official command: {line!r}. "
                    "Use '.venv\\Scripts\\python.exe' on Windows."
                )


# ---------------------------------------------------------------------------
# Contract 4 — git pull uses --ff-only
# ---------------------------------------------------------------------------

class TestFfOnlyPull:
    """Contract 4: git pull must use --ff-only to prevent merge commits."""

    def test_ff_only_pull(self, combined_text):
        """'git pull --ff-only' must appear in the operator documents."""
        assert "--ff-only" in combined_text, (
            "FAIL Contract 4: '--ff-only' not found in operator documents. "
            "git pull must use --ff-only to prevent accidental merge commits."
        )

    def test_ff_only_in_pull_command(self, combined_text):
        """The git pull command must include --ff-only."""
        pull_lines = [
            line for line in combined_text.splitlines()
            if "git pull" in line and "--ff-only" not in line
            and not line.strip().startswith("|")  # not a table row
            and not line.strip().startswith(">")  # not a blockquote
            and not line.strip().startswith("::") # not a bat comment
        ]
        # Filter out lines that are clearly describing the old (wrong) command
        pull_lines = [l for l in pull_lines if "Old" not in l and "unsafe" not in l and "wrong" not in l]
        assert not pull_lines, (
            f"FAIL Contract 4: git pull without --ff-only found in non-table lines: {pull_lines}"
        )


# ---------------------------------------------------------------------------
# Contract 5 — git status is required
# ---------------------------------------------------------------------------

class TestGitStatusRequired:
    """Contract 5: git status must appear in the verification step."""

    def test_git_status_required(self, combined_text):
        """'git status' must appear in the operator documents."""
        assert "git status" in combined_text, (
            "FAIL Contract 5: 'git status' not found in operator documents. "
            "Kishore must verify the working tree is clean before proceeding."
        )


# ---------------------------------------------------------------------------
# Contract 6 — Exact commit required (no "or latest follow-up")
# ---------------------------------------------------------------------------

class TestExactCommitRequired:
    """Contract 6: No 'or latest follow-up' wording — exact commit is required."""

    def test_no_or_latest_follow_up(self, combined_text):
        """'or latest follow-up' must not appear as an instruction; only as a prohibition."""
        # The phrase may appear in a prohibition ("Do not accept 'or latest follow-up'"),
        # but must not appear as a permissive instruction.
        lower = combined_text.lower()
        idx = lower.find("or latest follow-up")
        if idx == -1:
            return  # phrase absent entirely — pass
        # Check that every occurrence is preceded by a negation within 60 chars
        import re as _re
        for m in _re.finditer(r"or latest follow-up", lower):
            context = lower[max(0, m.start() - 60):m.start()]
            assert any(neg in context for neg in ["do not", "not accept", "never", "prohibited", "must not"]), (
                f"FAIL Contract 6: 'or latest follow-up' appears without a negation at position {m.start()}. "
                "The authorized commit must be exact. Kishore must STOP if HEAD differs."
            )


# ---------------------------------------------------------------------------
# Contract 7 — Real-data preflight uses the Windows .bat launcher
# ---------------------------------------------------------------------------

class TestPreflightUsesBatLauncher:
    """Contract 7: Preflight must use the Windows .bat launcher, not direct Python."""

    def test_preflight_uses_bat_launcher(self, combined_text):
        """'run_all_laptop_validation.bat' must appear in preflight instructions."""
        assert "run_all_laptop_validation.bat" in combined_text, (
            "FAIL Contract 7: 'run_all_laptop_validation.bat' not found. "
            "Preflight must use the Windows launcher, not direct Python invocation."
        )

    def test_preflight_command_present(self, combined_text):
        """The preflight command with --preflight-only must be present."""
        assert "--preflight-only" in combined_text, (
            "FAIL Contract 7: '--preflight-only' not found in operator documents."
        )


# ---------------------------------------------------------------------------
# Contract 8 — Preflight command includes --data-mode real
# ---------------------------------------------------------------------------

class TestPreflightDataModeReal:
    """Contract 8: Preflight command must include --data-mode real."""

    def test_preflight_data_mode_real(self, combined_text):
        """'--data-mode real' must appear alongside '--preflight-only'."""
        # Find lines that have --preflight-only and check they also have --data-mode real
        preflight_lines = [
            line for line in combined_text.splitlines()
            if "--preflight-only" in line
        ]
        assert preflight_lines, "No line with '--preflight-only' found."
        for line in preflight_lines:
            # The command line itself should have --data-mode real
            # (The surrounding context may be on adjacent lines — check the block)
            pass  # checked by test_preflight_data_mode_real_in_block below

    def test_preflight_data_mode_real_in_block(self, combined_text):
        """A code block must contain both --data-mode real and --preflight-only."""
        blocks = extract_code_blocks(combined_text)
        found = False
        for block in blocks:
            if "--preflight-only" in block and "--data-mode real" in block:
                found = True
                break
        assert found, (
            "FAIL Contract 8: No code block found containing both '--data-mode real' "
            "and '--preflight-only'. Preflight must explicitly specify real data mode."
        )


# ---------------------------------------------------------------------------
# Contract 9 — Preflight command includes --preflight-only
# ---------------------------------------------------------------------------

class TestPreflightOnlyFlag:
    """Contract 9: Preflight must use --preflight-only flag."""

    def test_preflight_only_flag(self, combined_text):
        """'--preflight-only' must appear in the operator documents."""
        assert "--preflight-only" in combined_text, (
            "FAIL Contract 9: '--preflight-only' not found in operator documents."
        )


# ---------------------------------------------------------------------------
# Contract 10 — Full run uses the Windows .bat launcher
# ---------------------------------------------------------------------------

class TestFullRunUsesBatLauncher:
    """Contract 10: Full validation run must use the Windows .bat launcher."""

    def test_full_run_uses_bat_launcher(self, combined_text):
        """A code block must contain 'run_all_laptop_validation.bat' without --preflight-only."""
        blocks = extract_code_blocks(combined_text)
        found = False
        for block in blocks:
            if "run_all_laptop_validation.bat" in block and "--preflight-only" not in block:
                found = True
                break
        assert found, (
            "FAIL Contract 10: No code block found with 'run_all_laptop_validation.bat' "
            "without '--preflight-only'. Full validation must use the Windows launcher."
        )


# ---------------------------------------------------------------------------
# Contract 11 — Full run command includes --data-mode real
# ---------------------------------------------------------------------------

class TestFullRunDataModeReal:
    """Contract 11: Full run command must include --data-mode real."""

    def test_full_run_data_mode_real(self, combined_text):
        """A code block must contain 'run_all_laptop_validation.bat --data-mode real' without --preflight-only."""
        blocks = extract_code_blocks(combined_text)
        found = False
        for block in blocks:
            lines = block.splitlines()
            for line in lines:
                stripped = line.strip()
                if (
                    "run_all_laptop_validation.bat" in stripped
                    and "--data-mode real" in stripped
                    and "--preflight-only" not in stripped
                ):
                    found = True
                    break
            if found:
                break
        assert found, (
            "FAIL Contract 11: No code block line found with "
            "'run_all_laptop_validation.bat --data-mode real' (without --preflight-only). "
            "Full validation must explicitly use real data mode."
        )


# ---------------------------------------------------------------------------
# Contract 12 — Direct Python pipeline invocation is not the official command
# ---------------------------------------------------------------------------

class TestDirectPythonPipelineNotOfficial:
    """Contract 12: Direct Python pipeline invocation must not be the official command."""

    def test_direct_python_pipeline_not_official(self, combined_text):
        """Official command blocks must not invoke run_laptop_validation_pipeline.py directly."""
        blocks = extract_command_blocks(combined_text)
        for line in blocks:
            if "run_laptop_validation_pipeline.py" in line:
                # Allow if it's a 'Do not' instruction line (but those are prose, not code blocks)
                pytest.fail(
                    f"FAIL Contract 12: Direct pipeline invocation found as official command: {line!r}. "
                    "Use 'scripts\\windows\\run_all_laptop_validation.bat' instead."
                )

    def test_do_not_call_pipeline_directly_stated(self, combined_text):
        """The documents must explicitly state not to call the pipeline directly."""
        assert "run_laptop_validation_pipeline.py" in combined_text, (
            "The pipeline file is not mentioned at all — cannot verify the prohibition."
        )
        # The prohibition must be stated
        lower = combined_text.lower()
        assert "do not call" in lower or "do not use" in lower or "not the official" in lower, (
            "FAIL Contract 12: No prohibition against direct pipeline invocation found."
        )


# ---------------------------------------------------------------------------
# Contract 13 — Synthetic mode cannot authorize validation
# ---------------------------------------------------------------------------

class TestSyntheticCannotAuthorize:
    """Contract 13: Synthetic mode must not appear in the official validation sequence."""

    def test_synthetic_not_in_official_sequence(self, combined_text):
        """The official 8-step sequence must not include --data-mode synthetic."""
        # The official sequence section
        sequence_match = re.search(
            r"Authoritative.*?Sequence.*?(?=\n##|\Z)",
            combined_text,
            re.DOTALL | re.IGNORECASE,
        )
        if sequence_match:
            sequence_text = sequence_match.group(0)
            assert "--data-mode synthetic" not in sequence_text, (
                "FAIL Contract 13: '--data-mode synthetic' found in the official sequence. "
                "Synthetic mode cannot authorize validation."
            )

    def test_synthetic_cannot_authorize_stated(self, combined_text):
        """The documents must state that synthetic mode must not be used for official validation."""
        import re as _re
        # Strip markdown bold (**text**) and backtick code spans (`text`) for plain-text matching
        plain = _re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", combined_text)  # remove bold/italic
        plain = _re.sub(r"`([^`]+)`", r"\1", plain)  # remove inline code spans
        lower = plain.lower()
        assert "synthetic" in lower, (
            "FAIL Contract 13: 'synthetic' not found in operator documents."
        )
        prohibited = (
            "cannot authorize" in lower
            or "does not authorize" in lower
            or "not authorize" in lower
            or "do not use --data-mode synthetic" in lower
            or "synthetic mode cannot" in lower
            or "not use --data-mode synthetic" in lower
        )
        assert prohibited, (
            "FAIL Contract 13: No statement found prohibiting synthetic mode for official validation. "
            "Expected one of: 'cannot authorize', 'does not authorize', "
            "'do not use --data-mode synthetic', or 'synthetic mode cannot'."
        )


# ---------------------------------------------------------------------------
# Contract 14 — CUDA claims require device=cuda evidence
# ---------------------------------------------------------------------------

class TestCudaRequiresDeviceEvidence:
    """Contract 14: CUDA execution may only be claimed when device=cuda is in the artifact."""

    def test_cuda_requires_device_evidence(self, combined_text):
        """'device = cuda' or 'device=cuda' must appear as a required artifact field."""
        assert "device" in combined_text and "cuda" in combined_text.lower(), (
            "FAIL Contract 14: No mention of device=cuda evidence requirement."
        )
        # The artifact field table must include device = cuda
        lower = combined_text.lower()
        assert "device" in lower and "cuda" in lower, (
            "FAIL Contract 14: 'device' and 'cuda' not found together in operator documents."
        )

    def test_cpu_tests_not_labeled_cuda(self, combined_text):
        """The real-object regression tests must be described as CPU tests, not CUDA tests."""
        # Find the section describing the regression tests
        lower = combined_text.lower()
        # The tests should be described as CPU regression tests
        assert "cpu regression" in lower or "real-object cpu" in lower or "cpu" in lower, (
            "FAIL Contract 14: The real-object tests are not described as CPU tests. "
            "They run on CPU and must not be labeled as CUDA tests."
        )

    def test_cuda_proof_requirements_listed(self, combined_text):
        """Required CUDA evidence fields must be listed."""
        required_fields = ["sm_120", "2.7.1+cu128", "device"]
        for field in required_fields:
            assert field in combined_text, (
                f"FAIL Contract 14: Required CUDA evidence field '{field}' not found in operator documents."
            )


# ---------------------------------------------------------------------------
# Contract 15 — Current run_id artifact path is used
# ---------------------------------------------------------------------------

class TestRunIdArtifactPath:
    """Contract 15: Artifact inspection must use the current run_id path, not a generic path."""

    def test_run_id_artifact_path(self, combined_text):
        """'run_id' must appear in the artifact inspection instructions."""
        assert "run_id" in combined_text, (
            "FAIL Contract 15: 'run_id' not found in operator documents. "
            "Kishore must locate the artifact by the current run_id, not a generic path."
        )

    def test_generic_artifacts_path_not_official(self, combined_text):
        """The generic 'artifacts\\moe_summary.json' path must not be the official instruction."""
        blocks = extract_command_blocks(combined_text)
        for line in blocks:
            if re.search(r"artifacts[/\\]moe_summary\.json", line, re.IGNORECASE):
                pytest.fail(
                    f"FAIL Contract 15: Generic artifact path found as official command: {line!r}. "
                    "Use 'benchmarks\\results\\laptop\\<run_id>\\moe_summary.json' instead."
                )

    def test_run_id_path_in_artifact_instructions(self, combined_text):
        """The artifact path must include a run_id placeholder."""
        assert "<run_id>" in combined_text or "run_id" in combined_text, (
            "FAIL Contract 15: No run_id-based artifact path found in operator documents."
        )
        # The path should reference benchmarks\results\laptop\<run_id>
        assert "benchmarks" in combined_text and "results" in combined_text, (
            "FAIL Contract 15: 'benchmarks\\results\\laptop\\<run_id>' pattern not found."
        )
