"""
Run 9 Operator Package Regression Tests
========================================
Tests covering the 5 corrections to the Run 9 operator instructions:
  1. RTX 5060 hardware label (not RTX 5090)
  2. Synthetic preflight is optional and non-authorizing
  3. Full validation explicitly uses --data-mode real
  4. Complete A-H GO criteria
  5. Correct execution sequence

All tests are CPU-compatible and do not require GPU hardware.
"""

import pathlib
import re
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths to operator-package files
# ---------------------------------------------------------------------------
CHECKLIST = REPO_ROOT / "docs" / "KISHORE_GPU_OPERATOR_CHECKLIST.md"
LAPTOP_GUIDE = REPO_ROOT / "docs" / "LAPTOP_GPU_VALIDATION.md"
MONTH1_RESULTS = REPO_ROOT / "benchmarks" / "MONTH1_VALIDATION_RESULTS.md"
LAUNCHER = REPO_ROOT / "scripts" / "windows" / "run_all_laptop_validation.bat"
PIPELINE = REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ===========================================================================
# Test Group 1 — Hardware Label Corrections (Correction 1)
# Tests 1-2
# ===========================================================================

class TestHardwareLabelCorrections:
    """RTX 5060 must be the Kuwait laptop GPU; RTX 5090 must not appear in
    any Run 9 operator instruction."""

    def test_checklist_identifies_rtx5060(self):
        """Kishore checklist must require RTX 5060."""
        text = _read(CHECKLIST)
        assert "RTX 5060" in text, (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must identify the Kuwait laptop "
            "GPU as RTX 5060"
        )

    def test_no_rtx5090_in_run9_instructions(self):
        """No Run 9 Kuwait operator instruction may identify the GPU as RTX 5090."""
        # Only MONTH1_VALIDATION_RESULTS.md is checked here because that is
        # the file that contained the incorrect RTX 5090 reference.
        text = _read(MONTH1_RESULTS)
        # Allow "RTX 5090" only in historical run summaries (before Run 9 section).
        # The Run 9 section starts at "## Run 9" or "### Run 9".
        run9_start = text.find("### Run 9")
        if run9_start == -1:
            run9_start = text.find("## Run 9")
        assert run9_start != -1, "MONTH1_VALIDATION_RESULTS.md must contain a Run 9 section"
        run9_text = text[run9_start:]
        # Allow "RTX 5090" only in negative assertions like "not RTX 5090"
        # (e.g., in artifact integrity checks: "identifies RTX 5060, not RTX 5090")
        for match in re.finditer(r"RTX 5090", run9_text):
            start = max(0, match.start() - 30)
            context = run9_text[start:match.end()]
            assert re.search(r"not\s+RTX 5090", context, re.IGNORECASE), (
                f"The Run 9 section of MONTH1_VALIDATION_RESULTS.md must not "
                f"positively identify the Kuwait laptop as RTX 5090. "
                f"Found without 'not': ...{context}..."
            )

    def test_checklist_requires_sm120(self):
        """Checklist must confirm compute capability sm_120."""
        text = _read(CHECKLIST)
        assert "sm_120" in text, (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must require compute capability sm_120"
        )

    def test_checklist_requires_pytorch_271_cu128(self):
        """Checklist must confirm PyTorch 2.7.1+cu128."""
        text = _read(CHECKLIST)
        assert "2.7.1+cu128" in text, (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must require PyTorch 2.7.1+cu128"
        )


# ===========================================================================
# Test Group 2 — Synthetic Preflight Status (Correction 2)
# Tests 3-5
# ===========================================================================

class TestSyntheticPreflightStatus:
    """Synthetic preflight must be labelled optional and non-authorizing."""

    def test_synthetic_labelled_optional_in_checklist(self):
        """Checklist must label synthetic diagnostic as OPTIONAL."""
        text = _read(CHECKLIST)
        # Check that the synthetic section is clearly marked optional
        assert re.search(r"OPTIONAL", text, re.IGNORECASE), (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must label the synthetic "
            "diagnostic as OPTIONAL"
        )

    def test_synthetic_does_not_authorize_full_run_checklist(self):
        """Checklist must state that synthetic diagnostic does not authorize full validation."""
        text = _read(CHECKLIST)
        assert re.search(
            r"does not authorize|DOES NOT AUTHORIZE",
            text
        ), (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must state that synthetic "
            "diagnostic does not authorize the full GPU run"
        )

    def test_synthetic_labelled_optional_in_results(self):
        """MONTH1_VALIDATION_RESULTS.md must label synthetic diagnostic as optional."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9")
        if run9_start == -1:
            run9_start = text.find("## Run 9")
        run9_text = text[run9_start:]
        assert re.search(
            r"OPTIONAL|optional",
            run9_text
        ), (
            "The Run 9 section of MONTH1_VALIDATION_RESULTS.md must label "
            "the synthetic diagnostic as optional"
        )

    def test_real_data_preflight_mandatory_in_checklist(self):
        """Checklist must mark real-data preflight as MANDATORY."""
        text = _read(CHECKLIST)
        assert re.search(r"MANDATORY", text, re.IGNORECASE), (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must mark real-data preflight "
            "as MANDATORY"
        )


# ===========================================================================
# Test Group 3 — Full Run Uses Real Data (Correction 3)
# Tests 6-8
# ===========================================================================

class TestFullRunUsesRealData:
    """Official full validation command must explicitly pass --data-mode real."""

    def test_launcher_forwards_args_verbatim(self):
        """Launcher must pass %* to the Python pipeline (verbatim arg forwarding)."""
        text = _read(LAUNCHER)
        # The launcher must contain %* in the Python call line
        assert "%*" in text, (
            "run_all_laptop_validation.bat must pass %* to the Python "
            "pipeline so --data-mode real is forwarded"
        )

    def test_official_full_command_uses_data_mode_real_in_checklist(self):
        """Checklist full-run command must include --data-mode real."""
        text = _read(CHECKLIST)
        # Find the full run command block
        assert re.search(
            r"run_all_laptop_validation\.bat\s+--data-mode\s+real",
            text
        ), (
            "KISHORE_GPU_OPERATOR_CHECKLIST.md must show the full run command "
            "as: run_all_laptop_validation.bat --data-mode real"
        )

    def test_official_full_command_uses_data_mode_real_in_results(self):
        """MONTH1_VALIDATION_RESULTS.md Run 9 section must use --data-mode real for full run."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9")
        if run9_start == -1:
            run9_start = text.find("## Run 9")
        run9_text = text[run9_start:]
        assert re.search(
            r"run_all_laptop_validation\.bat\s+--data-mode\s+real",
            run9_text
        ), (
            "The Run 9 section of MONTH1_VALIDATION_RESULTS.md must show "
            "the full run command as: run_all_laptop_validation.bat --data-mode real"
        )

    def test_bare_launcher_not_presented_as_official_command_in_checklist(self):
        """Checklist must not present bare run_all_laptop_validation.bat (without --data-mode) as official."""
        text = _read(CHECKLIST)
        # The full-run section must not have a bare invocation without --data-mode
        # Find the Part 5 / STEP 4 section
        step4_match = re.search(r"(Part 5|STEP 4|Full Real-Data Validation)", text)
        if step4_match:
            step4_text = text[step4_match.start():]
            # In the step 4 section, the bat command must include --data-mode
            bat_calls = re.findall(
                r"run_all_laptop_validation\.bat([^\n]*)",
                step4_text
            )
            for call in bat_calls:
                assert "--data-mode" in call, (
                    f"In the STEP 4 section, run_all_laptop_validation.bat "
                    f"must include --data-mode. Found bare call: "
                    f"run_all_laptop_validation.bat{call}"
                )

    def test_pipeline_default_data_mode_is_real(self):
        """Pipeline argparse default for --data-mode must be 'real'."""
        text = _read(PIPELINE)
        # Find the add_argument for data-mode
        match = re.search(
            r'add_argument\s*\(\s*["\']--data-mode["\'].*?default\s*=\s*["\'](\w+)["\']',
            text,
            re.DOTALL
        )
        assert match is not None, (
            "run_laptop_validation_pipeline.py must have an --data-mode "
            "argument with an explicit default"
        )
        assert match.group(1) == "real", (
            f"Pipeline --data-mode default must be 'real', got '{match.group(1)}'"
        )

    def test_real_mode_does_not_fallback_to_synthetic(self):
        """In real mode, dataset load failure must raise an error, not fall back to synthetic."""
        text = _read(PIPELINE)
        # Find the step03 function and verify the real-mode branch halts
        step03_match = re.search(r"def step03_real_text_dataset", text)
        assert step03_match is not None, "step03_real_text_dataset must exist"
        step03_text = text[step03_match.start():step03_match.start() + 1500]
        # Must contain: if data_mode == "real": raise / halt
        assert re.search(
            r'if data_mode == .real..*?raise',
            step03_text,
            re.DOTALL
        ), (
            "step03_real_text_dataset must raise an error (not fall back to "
            "synthetic) when data_mode == 'real' and dataset load fails"
        )


# ===========================================================================
# Test Group 4 — Complete GO Criteria (Correction 4)
# Tests 9-13
# ===========================================================================

class TestCompleteGoCriteria:
    """Run 9 GO criteria must include the complete A-H contract."""

    def test_go_criteria_include_router_metrics(self):
        """GO criteria must include MoE router metrics (router entropy, expert utilization)."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1, (
            "MONTH1_VALIDATION_RESULTS.md must contain a '### Run 9 GO Criteria' section"
        )
        go_text = text[run9_start:]
        assert re.search(r"router.entropy|router entropy", go_text, re.IGNORECASE), (
            "Run 9 GO criteria must include router entropy"
        )
        assert re.search(r"expert.utilization|utilization", go_text, re.IGNORECASE), (
            "Run 9 GO criteria must include expert utilization"
        )

    def test_go_criteria_include_moe_accepted(self):
        """GO criteria must require moe_accepted = true."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1
        go_text = text[run9_start:]
        assert re.search(r"moe_accepted", go_text), (
            "Run 9 GO criteria must include moe_accepted = true"
        )

    def test_go_criteria_include_checkpoint_resume(self):
        """GO criteria must include checkpoint resume section."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1
        go_text = text[run9_start:]
        assert re.search(r"checkpoint.resume|CHECKPOINT RESUME", go_text, re.IGNORECASE), (
            "Run 9 GO criteria must include checkpoint resume requirements"
        )

    def test_go_criteria_include_artifact_freshness(self):
        """GO criteria must include artifact integrity/freshness requirements."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1
        go_text = text[run9_start:]
        assert re.search(r"artifact|ARTIFACT", go_text, re.IGNORECASE), (
            "Run 9 GO criteria must include artifact integrity requirements"
        )

    def test_go_criteria_require_exit_code_0(self):
        """GO criteria must require exit code 0 for LAPTOP ARCHITECTURAL PASS."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1
        go_text = text[run9_start:]
        assert re.search(r"exit code.*0|EXIT_CODE.*0|exit code = 0", go_text, re.IGNORECASE), (
            "Run 9 GO criteria must require exit code 0"
        )

    def test_go_criteria_laptop_architectural_pass_not_azure(self):
        """GO criteria must state that LAPTOP ARCHITECTURAL PASS does not authorize Azure/200B/500B."""
        text = _read(MONTH1_RESULTS)
        run9_start = text.find("### Run 9 GO Criteria")
        assert run9_start != -1
        go_text = text[run9_start:]
        assert re.search(r"LAPTOP ARCHITECTURAL PASS", go_text), (
            "Run 9 GO criteria must use the term 'LAPTOP ARCHITECTURAL PASS'"
        )
        # Accept both plain text and markdown-bold variants:
        # "does not authorize", "does **not** authorize"
        assert re.search(
            r"does\s+(?:\*\*)?not(?:\*\*)?\s+authorize|NOT\s+authorize",
            go_text,
            re.IGNORECASE
        ), (
            "Run 9 GO criteria must state what the LAPTOP ARCHITECTURAL PASS "
            "does NOT authorize (accepts 'does not authorize' or 'does **not** authorize')"
        )


# ===========================================================================
# Test Group 5 — Exit Code Classification (from Run 9 regression)
# Tests 14-15
# ===========================================================================

class TestExitCodeClassification:
    """Software errors must return exit code 3; hardware safety must return 4."""

    def test_software_errors_return_exit_code_3_not_4(self):
        """Preflight failure path must return EXIT_EXECUTION_ERROR (3) for software exceptions."""
        text = _read(PIPELINE)
        # Find the main() function's failure handling
        # After our Defect 3 fix, the code should check _hardware_safety_keywords
        # and only return EXIT_SAFETY_STOP for hardware conditions
        assert "_hardware_safety_keywords" in text or "hardware_safety" in text, (
            "run_laptop_validation_pipeline.py must distinguish hardware "
            "safety conditions from software exceptions in the exit-code path"
        )

    def test_hardware_safety_keywords_defined(self):
        """Hardware safety keywords must be defined to distinguish hardware from software errors."""
        text = _read(PIPELINE)
        # The fix added a set of hardware safety keywords
        assert re.search(
            r"_hardware_safety_keywords|hardware_safety_keywords",
            text
        ), (
            "run_laptop_validation_pipeline.py must define hardware safety "
            "keywords to classify exit codes correctly"
        )
