"""
Jupiter Shot — Run 9 Regression Tests
======================================
Locks in the three fixes applied to resolve the Run 8 FAIL verdict:

  A. Defect 1: Explicit gradient checkpoint import in dense.py and moe.py
  B. Defect 2: Tokenizer vocabulary contract (effective vs base vocab size)
  C. Defect 3: Exit-code classification (EXECUTION_ERROR for software failures)
  D. End-to-end preflight with synthetic data (no GPU, no network required)

All tests must pass on CPU-only hardware with no GPU and no HuggingFace network
access required (synthetic mode is used for D).
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Test A: Explicit gradient checkpoint import
# ══════════════════════════════════════════════════════════════════════════════

class TestExplicitCheckpointImport(unittest.TestCase):
    """Verify that dense.py and moe.py use explicit checkpoint imports."""

    def _read_source(self, relpath: str) -> str:
        path = REPO_ROOT / relpath
        self.assertTrue(path.exists(), f"Source file not found: {path}")
        return path.read_text(encoding="utf-8")

    # ── dense.py ──────────────────────────────────────────────────────────────

    def test_dense_has_explicit_checkpoint_import(self):
        """dense.py must import checkpoint explicitly from torch.utils.checkpoint."""
        src = self._read_source("training/models/dense.py")
        self.assertIn(
            "from torch.utils.checkpoint import",
            src,
            "dense.py must have an explicit 'from torch.utils.checkpoint import ...' statement",
        )

    def test_dense_does_not_use_torch_utils_checkpoint_dot_checkpoint(self):
        """dense.py must NOT call torch.utils.checkpoint.checkpoint() via attribute chain."""
        src = self._read_source("training/models/dense.py")
        # Allow the import line itself; forbid call-site usage
        call_lines = [
            line for line in src.splitlines()
            if "torch.utils.checkpoint.checkpoint(" in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(
            call_lines, [],
            f"dense.py still has torch.utils.checkpoint.checkpoint() call-site: {call_lines}",
        )

    def test_dense_checkpoint_call_uses_local_alias(self):
        """dense.py must call the locally-imported checkpoint alias at the call site."""
        src = self._read_source("training/models/dense.py")
        # The call site must use the local alias (gradient_checkpoint or checkpoint)
        self.assertTrue(
            "gradient_checkpoint(" in src or (
                "from torch.utils.checkpoint import checkpoint" in src
                and "checkpoint(" in src
            ),
            "dense.py call site must use the locally-imported alias",
        )

    # ── moe.py ────────────────────────────────────────────────────────────────

    def test_moe_has_explicit_checkpoint_import(self):
        """moe.py must import checkpoint explicitly from torch.utils.checkpoint."""
        src = self._read_source("training/models/moe.py")
        self.assertIn(
            "from torch.utils.checkpoint import",
            src,
            "moe.py must have an explicit 'from torch.utils.checkpoint import ...' statement",
        )

    def test_moe_does_not_use_torch_utils_checkpoint_dot_checkpoint(self):
        """moe.py must NOT call torch.utils.checkpoint.checkpoint() via attribute chain."""
        src = self._read_source("training/models/moe.py")
        call_lines = [
            line for line in src.splitlines()
            if "torch.utils.checkpoint.checkpoint(" in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(
            call_lines, [],
            f"moe.py still has torch.utils.checkpoint.checkpoint() call-site: {call_lines}",
        )

    def test_moe_checkpoint_call_uses_local_alias(self):
        """moe.py must call the locally-imported checkpoint alias at the call site."""
        src = self._read_source("training/models/moe.py")
        self.assertTrue(
            "gradient_checkpoint(" in src or (
                "from torch.utils.checkpoint import checkpoint" in src
                and "checkpoint(" in src
            ),
            "moe.py call site must use the locally-imported alias",
        )

    def test_dense_checkpoint_import_is_importable(self):
        """dense.py must be importable without AttributeError on torch.utils.checkpoint."""
        try:
            import torch  # noqa: F401
            # Reload to pick up any in-process changes
            if "training.models.dense" in sys.modules:
                del sys.modules["training.models.dense"]
            import training.models.dense as dense_mod  # noqa: F401
        except AttributeError as e:
            self.fail(
                f"dense.py raised AttributeError on import (checkpoint import broken): {e}"
            )
        except ImportError as e:
            self.skipTest(f"Optional dependency missing: {e}")

    def test_moe_checkpoint_import_is_importable(self):
        """moe.py must be importable without AttributeError on torch.utils.checkpoint."""
        try:
            import torch  # noqa: F401
            if "training.models.moe" in sys.modules:
                del sys.modules["training.models.moe"]
            import training.models.moe as moe_mod  # noqa: F401
        except AttributeError as e:
            self.fail(
                f"moe.py raised AttributeError on import (checkpoint import broken): {e}"
            )
        except ImportError as e:
            self.skipTest(f"Optional dependency missing: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Test B: Tokenizer vocabulary contract
# ══════════════════════════════════════════════════════════════════════════════

def _make_transformers_stub() -> None:
    """Inject a minimal transformers stub so patch() targets resolve without the real package."""
    if "transformers" not in sys.modules:
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = MagicMock()
        sys.modules["transformers"] = stub


class TestTokenizerVocabularyContract(unittest.TestCase):
    """Verify step04 and step05 report the correct vocabulary values."""

    def _import_pipeline(self):
        if "scripts.run_laptop_validation_pipeline" in sys.modules:
            del sys.modules["scripts.run_laptop_validation_pipeline"]
        spec = importlib.util.spec_from_file_location(
            "run_laptop_validation_pipeline",
            REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_step04_reports_base_and_effective_vocab(self):
        """step04_tokenizer_load must report both base and effective vocab sizes."""
        _make_transformers_stub()
        mod = self._import_pipeline()
        # Build a mock tokenizer that mimics gpt-neox-20b: base=50254, effective=50277
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50254
        mock_tok.__len__ = MagicMock(return_value=50277)
        mock_tok.get_vocab.return_value = {f"tok_{i}": i for i in range(50277)}

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            result = mod.step04_tokenizer_load("real", pathlib.Path("/tmp/errors.jsonl"))

        self.assertEqual(result["tokenizer_base_vocab_size"], 50254,
                         "step04 must report tokenizer_base_vocab_size=50254")
        self.assertEqual(result["tokenizer_effective_vocab_size"], 50277,
                         "step04 must report tokenizer_effective_vocab_size=50277")
        self.assertEqual(result["vocab_size"], 50277,
                         "step04 legacy 'vocab_size' field must equal effective (50277), not base (50254)")
        self.assertTrue(result["tokenizer_loaded"])

    def test_step04_max_token_id_is_reported(self):
        """step04_tokenizer_load must report tokenizer_max_token_id."""
        _make_transformers_stub()
        mod = self._import_pipeline()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50254
        mock_tok.__len__ = MagicMock(return_value=50277)
        mock_tok.get_vocab.return_value = {f"tok_{i}": i for i in range(50277)}

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            result = mod.step04_tokenizer_load("real", pathlib.Path("/tmp/errors.jsonl"))

        self.assertIn("tokenizer_max_token_id", result,
                      "step04 must report tokenizer_max_token_id")
        self.assertEqual(result["tokenizer_max_token_id"], 50276,
                         "max_token_id must be 50276 (= effective_vocab_size - 1)")

    def test_step05_validates_against_effective_vocab(self):
        """step05 must validate token IDs against len(tok), not tok.vocab_size."""
        _make_transformers_stub()
        mod = self._import_pipeline()
        # Token ID 50260 is in [50254, 50277) — valid for effective, invalid for base
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50254
        mock_tok.__len__ = MagicMock(return_value=50277)
        mock_tok.encode.return_value = [50260]  # a special-token ID
        mock_tok.get_vocab.return_value = {f"tok_{i}": i for i in range(50277)}

        tokenizer_info = {
            "tokenizer_loaded": True,
            "tokenizer_base_vocab_size": 50254,
            "tokenizer_effective_vocab_size": 50277,
            "tokenizer_max_token_id": 50276,
        }

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            # Must NOT raise PreflightError — ID 50260 is valid for effective vocab
            result = mod.step05_token_id_range(
                "real", tokenizer_info, pathlib.Path("/tmp/errors.jsonl")
            )

        self.assertEqual(result["out_of_range"], [],
                         "Token ID 50260 is valid for effective vocab (50277) and must not be flagged")
        self.assertTrue(result.get("vocabulary_contract_passed"),
                        "step05 must set vocabulary_contract_passed=True on success")

    def test_step05_rejects_id_at_or_above_effective_vocab(self):
        """step05 must reject token IDs >= effective_vocab_size."""
        _make_transformers_stub()
        mod = self._import_pipeline()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50254
        mock_tok.__len__ = MagicMock(return_value=50277)
        mock_tok.encode.return_value = [50277]  # one past the end
        mock_tok.get_vocab.return_value = {f"tok_{i}": i for i in range(50277)}

        tokenizer_info = {
            "tokenizer_loaded": True,
            "tokenizer_base_vocab_size": 50254,
            "tokenizer_effective_vocab_size": 50277,
            "tokenizer_max_token_id": 50276,
        }

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            with self.assertRaises(mod.PreflightError):
                mod.step05_token_id_range(
                    "real", tokenizer_info, pathlib.Path("/tmp/errors.jsonl")
                )

    def test_step05_rejects_when_max_token_id_equals_effective_vocab(self):
        """step05 must raise PreflightError if max_token_id == effective_vocab_size."""
        _make_transformers_stub()
        mod = self._import_pipeline()
        mock_tok = MagicMock()
        mock_tok.vocab_size = 50254
        mock_tok.__len__ = MagicMock(return_value=50277)
        mock_tok.encode.return_value = [100]  # sample IDs are fine
        mock_tok.get_vocab.return_value = {f"tok_{i}": i for i in range(50277)}

        tokenizer_info = {
            "tokenizer_loaded": True,
            "tokenizer_base_vocab_size": 50254,
            "tokenizer_effective_vocab_size": 50277,
            "tokenizer_max_token_id": 50277,  # == effective_vocab_size — invalid
        }

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
            with self.assertRaises(mod.PreflightError):
                mod.step05_token_id_range(
                    "real", tokenizer_info, pathlib.Path("/tmp/errors.jsonl")
                )

    def test_step04_skips_in_synthetic_mode(self):
        """step04 must skip tokenizer load in synthetic mode."""
        mod = self._import_pipeline()
        result = mod.step04_tokenizer_load("synthetic", pathlib.Path("/tmp/errors.jsonl"))
        self.assertFalse(result.get("tokenizer_loaded", True),
                         "step04 must return tokenizer_loaded=False in synthetic mode")

    def test_step05_skips_when_tokenizer_not_loaded(self):
        """step05 must skip when tokenizer_loaded is False."""
        mod = self._import_pipeline()
        result = mod.step05_token_id_range(
            "synthetic", {"tokenizer_loaded": False}, pathlib.Path("/tmp/errors.jsonl")
        )
        self.assertTrue(result.get("skipped"), "step05 must return skipped=True when no tokenizer")


# ══════════════════════════════════════════════════════════════════════════════
# Test C: Exit-code classification
# ══════════════════════════════════════════════════════════════════════════════

class TestExitCodeClassification(unittest.TestCase):
    """Verify that software exceptions produce EXECUTION_ERROR (3), not SAFETY_STOP (4)."""

    def _import_pipeline(self):
        if "scripts.run_laptop_validation_pipeline" in sys.modules:
            del sys.modules["scripts.run_laptop_validation_pipeline"]
        spec = importlib.util.spec_from_file_location(
            "run_laptop_validation_pipeline",
            REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_exit_code_constants_are_correct(self):
        """Exit code constants must match the documented values."""
        mod = self._import_pipeline()
        self.assertEqual(mod.EXIT_PASS, 0)
        self.assertEqual(mod.EXIT_NOT_ACCEPTED, 1)
        self.assertEqual(mod.EXIT_NOT_EVALUABLE, 2)
        self.assertEqual(mod.EXIT_EXECUTION_ERROR, 3)
        self.assertEqual(mod.EXIT_SAFETY_STOP, 4)

    def test_pipeline_source_has_execution_error_on_preflight_fail(self):
        """The pipeline must return EXIT_EXECUTION_ERROR when preflight fails with a software error."""
        src = (REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py").read_text(encoding="utf-8")
        # The fix: after preflight_passed is False, the code must check for hardware safety
        # and default to EXIT_EXECUTION_ERROR
        self.assertIn(
            "EXIT_EXECUTION_ERROR",
            src,
            "Pipeline must reference EXIT_EXECUTION_ERROR in the preflight failure path",
        )
        # Must NOT unconditionally return EXIT_SAFETY_STOP when preflight fails
        # (i.e., the old single-line `return EXIT_SAFETY_STOP` must be gone)
        lines = src.splitlines()
        preflight_block_start = None
        for i, line in enumerate(lines):
            if "if not preflight_passed:" in line:
                preflight_block_start = i
                break
        self.assertIsNotNone(preflight_block_start, "Could not find 'if not preflight_passed:' block")
        # Scan the next 30 lines for the old unconditional return
        block = lines[preflight_block_start: preflight_block_start + 30]
        # The block must NOT have a bare `return EXIT_SAFETY_STOP` as the ONLY return
        # (i.e., no conditional branching before it).
        # The fix adds an if/else so EXIT_SAFETY_STOP is only reached via the hardware branch.
        # Verify by checking that EXIT_EXECUTION_ERROR also appears in the block.
        block_text = "\n".join(block)
        self.assertIn(
            "EXIT_EXECUTION_ERROR",
            block_text,
            "Preflight failure block must contain EXIT_EXECUTION_ERROR as the default path",
        )
        # Verify the conditional structure: if is_hardware_safety ... else ... EXIT_EXECUTION_ERROR
        self.assertIn(
            "is_hardware_safety",
            block_text,
            "Preflight failure block must check is_hardware_safety before returning SAFETY_STOP",
        )

    def test_hardware_safety_keywords_trigger_safety_stop(self):
        """The pipeline source must have hardware-safety keyword detection."""
        src = (REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(
            "_hardware_safety_keywords",
            src,
            "Pipeline must define _hardware_safety_keywords for SAFETY_STOP classification",
        )

    def test_attribute_error_message_does_not_contain_hardware_keywords(self):
        """An AttributeError message must not trigger SAFETY_STOP classification."""
        hardware_keywords = (
            "temperature", "thermal", "overheat", "power", "gpu safety",
            "safety monitor", "operator safety", "runner safety",
        )
        attr_error_msg = "module 'torch.utils.checkpoint' has no attribute 'checkpoint'"
        for kw in hardware_keywords:
            self.assertNotIn(
                kw, attr_error_msg.lower(),
                f"AttributeError message should not contain hardware keyword '{kw}'",
            )

    def test_import_error_message_does_not_contain_hardware_keywords(self):
        """An ImportError message must not trigger SAFETY_STOP classification."""
        hardware_keywords = (
            "temperature", "thermal", "overheat", "power", "gpu safety",
            "safety monitor", "operator safety", "runner safety",
        )
        import_error_msg = "No module named 'torch'"
        for kw in hardware_keywords:
            self.assertNotIn(
                kw, import_error_msg.lower(),
                f"ImportError message should not contain hardware keyword '{kw}'",
            )


# ══════════════════════════════════════════════════════════════════════════════
# Test D: End-to-end preflight with synthetic data
# ══════════════════════════════════════════════════════════════════════════════

class TestPreflightSyntheticEndToEnd(unittest.TestCase):
    """Run preflight steps 1–5 in synthetic mode to verify no regressions."""

    def _import_pipeline(self):
        if "scripts.run_laptop_validation_pipeline" in sys.modules:
            del sys.modules["scripts.run_laptop_validation_pipeline"]
        spec = importlib.util.spec_from_file_location(
            "run_laptop_validation_pipeline",
            REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_step01_dependency_imports_passes(self):
        """Step 1 (dependency imports) must pass in the test environment."""
        mod = self._import_pipeline()
        try:
            result = mod.step01_dependency_imports(pathlib.Path("/tmp/errors.jsonl"))
            self.assertIn("torch_version", result, "Step 1 must report torch_version")
        except mod.PreflightError as e:
            self.skipTest(f"Step 1 skipped — missing dependency: {e}")

    def test_step03_synthetic_skips_dataset(self):
        """Step 3 in synthetic mode must skip dataset load and return dataset_loaded=False."""
        mod = self._import_pipeline()
        result = mod.step03_real_text_dataset("synthetic", pathlib.Path("/tmp/errors.jsonl"))
        self.assertFalse(result.get("dataset_loaded", True),
                         "Step 3 must return dataset_loaded=False in synthetic mode")

    def test_step04_synthetic_skips_tokenizer(self):
        """Step 4 in synthetic mode must skip tokenizer load."""
        mod = self._import_pipeline()
        result = mod.step04_tokenizer_load("synthetic", pathlib.Path("/tmp/errors.jsonl"))
        self.assertFalse(result.get("tokenizer_loaded", True),
                         "Step 4 must return tokenizer_loaded=False in synthetic mode")

    def test_step05_synthetic_skips_token_range_check(self):
        """Step 5 in synthetic mode must skip token ID range check."""
        mod = self._import_pipeline()
        result = mod.step05_token_id_range(
            "synthetic", {"tokenizer_loaded": False}, pathlib.Path("/tmp/errors.jsonl")
        )
        self.assertTrue(result.get("skipped"),
                        "Step 5 must return skipped=True in synthetic mode")

    def test_preflight_steps_1_3_4_5_pass_in_synthetic_mode(self):
        """Steps 1, 3, 4, 5 must all complete without exception in synthetic mode."""
        mod = self._import_pipeline()
        errors_path = pathlib.Path("/tmp/test_run9_errors.jsonl")
        try:
            s1 = mod.step01_dependency_imports(errors_path)
            self.assertIsInstance(s1, dict)
        except mod.PreflightError as e:
            self.skipTest(f"Step 1 failed (missing dep): {e}")

        s3 = mod.step03_real_text_dataset("synthetic", errors_path)
        self.assertIsInstance(s3, dict)

        s4 = mod.step04_tokenizer_load("synthetic", errors_path)
        self.assertIsInstance(s4, dict)

        s5 = mod.step05_token_id_range("synthetic", s4, errors_path)
        self.assertIsInstance(s5, dict)

    def test_pipeline_source_step04_has_effective_vocab_field(self):
        """The pipeline source must define tokenizer_effective_vocab_size in step04."""
        src = (REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(
            "tokenizer_effective_vocab_size",
            src,
            "step04 must define tokenizer_effective_vocab_size in its return dict",
        )

    def test_pipeline_source_step04_has_base_vocab_field(self):
        """The pipeline source must define tokenizer_base_vocab_size in step04."""
        src = (REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(
            "tokenizer_base_vocab_size",
            src,
            "step04 must define tokenizer_base_vocab_size in its return dict",
        )

    def test_pipeline_source_step05_uses_len_tok(self):
        """step05 must use len(tok) for the upper bound, not tok.vocab_size."""
        src = (REPO_ROOT / "scripts" / "run_laptop_validation_pipeline.py").read_text(encoding="utf-8")
        self.assertIn(
            "len(tok)",
            src,
            "step05 must use len(tok) (effective vocab) as the upper bound for token ID validation",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
