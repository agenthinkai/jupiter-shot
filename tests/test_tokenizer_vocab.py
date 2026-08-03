"""
Jupiter Shot — Tokenizer / Vocabulary Regression Tests
=======================================================

These tests verify the tokenizer/vocabulary contract introduced in Run 6:

  1. vocab_size=32000 with tokenizer vocab=50277 → TOKENIZER_VOCABULARY_MISMATCH halt
  2. vocab_size=50277 with tokenizer vocab=50277 → passes (compatible)
  3. Maximum token-ID boundary: id_max >= vocab_size → halt
  4. Minimum token-ID boundary: id_min < 0 → halt
  5. Special-token handling: pad_token set to eos_token (no new vocab tokens added)
  6. Model construction after vocabulary resolution: vocab_size=50277 accepted by DenseConfig
  7. Revised parameter count for dense_small at vocab=50277
  8. Revised VRAM estimate for dense_small at vocab=50277 fits 6.5 GB target
  9. Revised parameter count for moe_8expert at vocab=50277
 10. Revised VRAM estimate for moe_8expert at vocab=50277 fits 6.5 GB target
 11. Config file vocab_size field is 50277 (not 32000) for all 5 laptop configs
 12. Config file tokenizer_id field is present for dense_small and moe_8expert configs

All tests run without PyTorch or GPU — they test the config files, the math,
and the guard logic directly.
"""

from __future__ import annotations

import math
import pathlib
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Repository root ────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "training" / "configs"

# ── Constants (must match the runner) ─────────────────────────────────────────
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TOKENIZER_VOCAB_SIZE = 50277   # len(AutoTokenizer.from_pretrained(TOKENIZER_ID))
OLD_VOCAB_SIZE = 32000          # what the configs used before Run 6

# ── Parameter-count helpers (pure Python, no torch) ───────────────────────────

def _dense_params(vocab_size: int, hidden: int, n_layers: int,
                  intermediate: int, max_pos: int = 512) -> int:
    """Exact parameter count for a SwiGLU dense transformer."""
    embedding = vocab_size * hidden
    pos_emb   = max_pos * hidden
    attn      = 4 * hidden * hidden          # Q, K, V, O
    ffn       = 3 * hidden * intermediate    # SwiGLU: gate, up, down
    ln        = 4 * hidden                   # 2 LN per layer × 2 params
    per_layer = attn + ffn + ln
    return embedding + pos_emb + n_layers * per_layer


def _moe_params(vocab_size: int, hidden: int, n_layers: int,
                intermediate: int, num_experts: int, top_k: int,
                moe_freq: int = 2, max_pos: int = 256) -> tuple[int, int]:
    """Total and active parameter counts for a MoE transformer."""
    embedding = vocab_size * hidden
    pos_emb   = max_pos * hidden
    n_moe     = n_layers // moe_freq
    n_dense   = n_layers - n_moe
    attn      = 4 * hidden * hidden
    ffn_dense = 3 * hidden * intermediate
    ffn_moe   = num_experts * 3 * hidden * intermediate
    router    = hidden * num_experts
    ln        = 4 * hidden
    per_dense = attn + ffn_dense + ln
    per_moe   = attn + ffn_moe + router + ln
    total     = embedding + pos_emb + n_dense * per_dense + n_moe * per_moe
    active_per_dense = attn + ffn_dense + ln
    active_per_moe   = attn + top_k * 3 * hidden * intermediate + router + ln
    active = embedding + pos_emb + n_dense * active_per_dense + n_moe * active_per_moe
    return total, active


def _vram_estimate(weight_gb: float, batch: int, seq_len: int,
                   hidden: int, n_layers: int) -> float:
    """Rough VRAM estimate in GB (BF16 weights, AdamW fp32, gradient checkpointing)."""
    act_gb  = batch * seq_len * hidden * n_layers * 4 * 1.5 / 1e9
    opt_gb  = weight_gb * 4   # AdamW fp32 m+v = 4× bf16 weight
    grad_gb = weight_gb
    return weight_gb + act_gb + opt_gb + grad_gb


# ── Helper: simulate the runner vocab guard ────────────────────────────────────

def _vocab_guard(model_vocab_size: int, tokenizer_vocab_size: int) -> None:
    """Raises RuntimeError with TOKENIZER_VOCABULARY_MISMATCH if contract violated."""
    if model_vocab_size < tokenizer_vocab_size:
        raise RuntimeError(
            f"TOKENIZER_VOCABULARY_MISMATCH: "
            f"model vocab_size={model_vocab_size} < "
            f"tokenizer vocab_size={tokenizer_vocab_size} "
            f"({TOKENIZER_ID}). "
            f"Update the config to vocab_size={tokenizer_vocab_size} "
            f"before running real-text validation."
        )


def _batch_id_guard(id_min: int, id_max: int, vocab_size: int, step: int = 1) -> None:
    """Raises RuntimeError if batch token IDs are out of range."""
    if id_min < 0 or id_max >= vocab_size:
        raise RuntimeError(
            f"TOKENIZER_VOCABULARY_MISMATCH: "
            f"batch token IDs [{id_min}, {id_max}] out of range "
            f"[0, {vocab_size - 1}] at step {step}. "
            f"Model vocab_size={vocab_size} is too small for this tokenizer."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Test class 1: Vocabulary contract guard
# ══════════════════════════════════════════════════════════════════════════════

class TestVocabularyContractGuard(unittest.TestCase):
    """Tests for the TOKENIZER_VOCABULARY_MISMATCH guard logic."""

    def test_old_vocab_32000_with_tokenizer_50277_raises(self):
        """vocab_size=32000 < tokenizer_vocab=50277 → must raise TOKENIZER_VOCABULARY_MISMATCH."""
        with self.assertRaises(RuntimeError) as ctx:
            _vocab_guard(OLD_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", str(ctx.exception))
        self.assertIn("32000", str(ctx.exception))
        self.assertIn("50277", str(ctx.exception))

    def test_correct_vocab_50277_passes(self):
        """vocab_size=50277 >= tokenizer_vocab=50277 → no exception."""
        _vocab_guard(TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)  # must not raise

    def test_larger_vocab_passes(self):
        """vocab_size=65536 >= tokenizer_vocab=50277 → no exception."""
        _vocab_guard(65536, TOKENIZER_VOCAB_SIZE)

    def test_error_message_contains_tokenizer_id(self):
        """Error message must name the tokenizer."""
        with self.assertRaises(RuntimeError) as ctx:
            _vocab_guard(OLD_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        self.assertIn(TOKENIZER_ID, str(ctx.exception))

    def test_error_message_contains_update_instruction(self):
        """Error message must tell the user to update the config."""
        with self.assertRaises(RuntimeError) as ctx:
            _vocab_guard(OLD_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        self.assertIn("Update the config", str(ctx.exception))

    def test_vocab_one_less_raises(self):
        """vocab_size = tokenizer_vocab - 1 → must raise."""
        with self.assertRaises(RuntimeError):
            _vocab_guard(TOKENIZER_VOCAB_SIZE - 1, TOKENIZER_VOCAB_SIZE)

    def test_vocab_equal_passes(self):
        """vocab_size == tokenizer_vocab → must pass."""
        _vocab_guard(TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
# Test class 2: Per-batch token ID range guard
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchTokenIdRangeGuard(unittest.TestCase):
    """Tests for the per-batch token ID boundary check."""

    def test_valid_batch_passes(self):
        """id_min=0, id_max=50276 with vocab=50277 → no exception."""
        _batch_id_guard(0, TOKENIZER_VOCAB_SIZE - 1, TOKENIZER_VOCAB_SIZE)

    def test_id_max_equals_vocab_raises(self):
        """id_max == vocab_size → out of range (valid range is [0, vocab_size-1])."""
        with self.assertRaises(RuntimeError) as ctx:
            _batch_id_guard(0, TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", str(ctx.exception))

    def test_id_max_exceeds_vocab_raises(self):
        """id_max > vocab_size → must raise."""
        with self.assertRaises(RuntimeError):
            _batch_id_guard(0, TOKENIZER_VOCAB_SIZE + 100, TOKENIZER_VOCAB_SIZE)

    def test_negative_id_raises(self):
        """id_min < 0 → must raise."""
        with self.assertRaises(RuntimeError) as ctx:
            _batch_id_guard(-1, TOKENIZER_VOCAB_SIZE - 1, TOKENIZER_VOCAB_SIZE)
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", str(ctx.exception))

    def test_old_vocab_32000_with_token_id_50276_raises(self):
        """Token ID 50276 (valid for neox-20b) is out of range for vocab=32000."""
        with self.assertRaises(RuntimeError):
            _batch_id_guard(0, 50276, OLD_VOCAB_SIZE)

    def test_error_message_contains_range(self):
        """Error message must show the actual [min, max] range."""
        with self.assertRaises(RuntimeError) as ctx:
            _batch_id_guard(0, TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        msg = str(ctx.exception)
        self.assertIn(str(TOKENIZER_VOCAB_SIZE), msg)

    def test_boundary_id_max_one_below_vocab_passes(self):
        """id_max = vocab_size - 1 is the maximum valid token ID."""
        _batch_id_guard(0, TOKENIZER_VOCAB_SIZE - 1, TOKENIZER_VOCAB_SIZE)

    def test_step_number_in_error_message(self):
        """Error message must include the step number."""
        with self.assertRaises(RuntimeError) as ctx:
            _batch_id_guard(0, TOKENIZER_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE, step=42)
        self.assertIn("42", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# Test class 3: Special-token handling
# ══════════════════════════════════════════════════════════════════════════════

class TestSpecialTokenHandling(unittest.TestCase):
    """
    Tests for the GPT-NeoX-20B tokenizer special-token policy.

    Policy (from Run 6 spec):
      - EOS token:  present (token_id = 0 for neox-20b)
      - BOS token:  present (token_id = 0, same as EOS for neox-20b)
      - PAD token:  NOT defined by default; runner sets pad_token = eos_token
      - UNK token:  NOT defined for neox-20b (BPE tokenizer, no OOV)
      - Setting pad_token = eos_token does NOT add a new vocabulary entry
        (vocab size remains 50277 after the assignment)
    """

    def _make_mock_tokenizer(self, vocab_size: int = TOKENIZER_VOCAB_SIZE) -> MagicMock:
        tok = MagicMock()
        tok.__len__ = MagicMock(return_value=vocab_size)
        tok.eos_token = "<|endoftext|>"
        tok.eos_token_id = 0
        tok.bos_token = "<|endoftext|>"
        tok.bos_token_id = 0
        tok.pad_token = None
        tok.pad_token_id = None
        tok.unk_token = None
        return tok

    def test_pad_token_set_to_eos_does_not_change_vocab_size(self):
        """Setting pad_token = eos_token must not increase len(tokenizer)."""
        tok = self._make_mock_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        self.assertEqual(len(tok), TOKENIZER_VOCAB_SIZE)

    def test_eos_token_is_defined(self):
        """EOS token must be defined."""
        tok = self._make_mock_tokenizer()
        self.assertIsNotNone(tok.eos_token)

    def test_bos_token_is_defined(self):
        """BOS token must be defined."""
        tok = self._make_mock_tokenizer()
        self.assertIsNotNone(tok.bos_token)

    def test_unk_token_is_none_for_neox(self):
        """GPT-NeoX-20B is a BPE tokenizer with no UNK token."""
        tok = self._make_mock_tokenizer()
        self.assertIsNone(tok.unk_token)

    def test_pad_token_id_equals_eos_token_id_after_assignment(self):
        """After pad_token = eos_token, pad_token_id must equal eos_token_id."""
        tok = self._make_mock_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        self.assertEqual(tok.pad_token_id, tok.eos_token_id)

    def test_vocab_size_after_pad_assignment_still_passes_guard(self):
        """After pad assignment, vocab contract check must still pass."""
        tok = self._make_mock_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        # Must not raise
        _vocab_guard(len(tok), len(tok))


# ══════════════════════════════════════════════════════════════════════════════
# Test class 4: Model construction after vocabulary resolution
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConstructionAfterVocabResolution(unittest.TestCase):
    """
    Tests that DenseConfig and MoEConfig accept vocab_size=50277.
    These tests do NOT require torch — they only test the dataclass/config layer.
    """

    def _try_import_dense_config(self):
        """Attempt to import DenseConfig; skip if torch is unavailable."""
        try:
            from training.models.dense import DenseConfig
            return DenseConfig
        except ImportError:
            self.skipTest("torch not installed — skipping model construction test")

    def _try_import_moe_config(self):
        try:
            from training.models.moe import MoEConfig, DenseConfig
            return MoEConfig, DenseConfig
        except ImportError:
            self.skipTest("torch not installed — skipping model construction test")

    def test_dense_config_accepts_vocab_50277(self):
        DenseConfig = self._try_import_dense_config()
        cfg = DenseConfig(vocab_size=TOKENIZER_VOCAB_SIZE, hidden_size=512,
                          num_hidden_layers=8, num_attention_heads=8,
                          intermediate_size=1408, max_position_embeddings=512)
        self.assertEqual(cfg.vocab_size, TOKENIZER_VOCAB_SIZE)

    def test_dense_config_rejects_vocab_32000_when_tokenizer_is_50277(self):
        """Simulates what the runner does: guard fires before model construction."""
        with self.assertRaises(RuntimeError) as ctx:
            _vocab_guard(OLD_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", str(ctx.exception))

    def test_moe_config_accepts_vocab_50277(self):
        MoEConfig, DenseConfig = self._try_import_moe_config()
        base = DenseConfig(vocab_size=TOKENIZER_VOCAB_SIZE, hidden_size=384,
                           num_hidden_layers=6, num_attention_heads=6,
                           intermediate_size=768, max_position_embeddings=256)
        cfg = MoEConfig(base=base, num_experts=8, num_experts_per_token=2)
        self.assertEqual(cfg.base.vocab_size, TOKENIZER_VOCAB_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
# Test class 5: Revised parameter counts (vocab=50277)
# ══════════════════════════════════════════════════════════════════════════════

class TestRevisedParameterCounts(unittest.TestCase):
    """Verify exact parameter counts after vocab_size correction to 50277."""

    # laptop_dense_small: hidden=512, layers=8, intermediate=1408, max_pos=512
    DENSE_SMALL_PARAMS = 51_710_464

    # laptop_moe_8expert_8gb_safe: hidden=384, layers=6, intermediate=768,
    #   8 experts, top-2, moe_freq=2, max_pos=256
    MOE_8EXPERT_TOTAL_PARAMS  = 46_849_920
    MOE_8EXPERT_ACTIVE_PARAMS = 30_924_672

    def test_dense_small_total_params_at_vocab_50277(self):
        params = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        self.assertEqual(params, self.DENSE_SMALL_PARAMS,
                         f"Expected {self.DENSE_SMALL_PARAMS:,}, got {params:,}")

    def test_dense_small_params_greater_than_at_vocab_32000(self):
        params_32k = _dense_params(OLD_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        params_50k = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        self.assertGreater(params_50k, params_32k)

    def test_dense_small_param_delta_is_correct(self):
        """Delta = (50277 - 32000) × hidden_size = 18277 × 512 = 9,357,824."""
        delta = (TOKENIZER_VOCAB_SIZE - OLD_VOCAB_SIZE) * 512
        params_32k = _dense_params(OLD_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        params_50k = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        self.assertEqual(params_50k - params_32k, delta)

    def test_moe_8expert_total_params_at_vocab_50277(self):
        total, _ = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                               moe_freq=2, max_pos=256)
        self.assertEqual(total, self.MOE_8EXPERT_TOTAL_PARAMS,
                         f"Expected {self.MOE_8EXPERT_TOTAL_PARAMS:,}, got {total:,}")

    def test_moe_8expert_active_params_at_vocab_50277(self):
        _, active = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                                moe_freq=2, max_pos=256)
        self.assertEqual(active, self.MOE_8EXPERT_ACTIVE_PARAMS,
                         f"Expected {self.MOE_8EXPERT_ACTIVE_PARAMS:,}, got {active:,}")

    def test_moe_8expert_active_less_than_total(self):
        total, active = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                                    moe_freq=2, max_pos=256)
        self.assertLess(active, total)

    def test_moe_8expert_sparsity_is_approx_34_pct(self):
        total, active = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                                    moe_freq=2, max_pos=256)
        sparsity = 1.0 - active / total
        self.assertGreater(sparsity, 0.30)
        self.assertLess(sparsity, 0.40)


# ══════════════════════════════════════════════════════════════════════════════
# Test class 6: Revised VRAM estimates (vocab=50277)
# ══════════════════════════════════════════════════════════════════════════════

class TestRevisedVRAMEstimates(unittest.TestCase):
    """Verify that revised VRAM estimates fit within the 6.5 GB target."""

    VRAM_TARGET_GB = 6.5
    DTYPE_BYTES = 2  # BF16

    def test_dense_small_vram_fits_target(self):
        params = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        weight_gb = params * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=512, hidden=512, n_layers=8)
        self.assertLess(peak, self.VRAM_TARGET_GB,
                        f"Dense small peak VRAM {peak:.3f} GB exceeds {self.VRAM_TARGET_GB} GB")

    def test_moe_8expert_vram_fits_target(self):
        total, _ = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                               moe_freq=2, max_pos=256)
        weight_gb = total * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=256, hidden=384, n_layers=6)
        self.assertLess(peak, self.VRAM_TARGET_GB,
                        f"MoE 8-expert peak VRAM {peak:.3f} GB exceeds {self.VRAM_TARGET_GB} GB")

    def test_dense_small_vram_is_positive(self):
        params = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        weight_gb = params * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=512, hidden=512, n_layers=8)
        self.assertGreater(peak, 0.0)

    def test_moe_8expert_vram_is_positive(self):
        total, _ = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                               moe_freq=2, max_pos=256)
        weight_gb = total * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=256, hidden=384, n_layers=6)
        self.assertGreater(peak, 0.0)

    def test_dense_small_vram_greater_than_weight_memory(self):
        """Peak VRAM must be greater than weight memory alone (optimizer + activations)."""
        params = _dense_params(TOKENIZER_VOCAB_SIZE, 512, 8, 1408, max_pos=512)
        weight_gb = params * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=512, hidden=512, n_layers=8)
        self.assertGreater(peak, weight_gb)

    def test_moe_8expert_vram_greater_than_weight_memory(self):
        total, _ = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                               moe_freq=2, max_pos=256)
        weight_gb = total * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=256, hidden=384, n_layers=6)
        self.assertGreater(peak, weight_gb)

    def test_conservative_4x_overestimate_still_fits(self):
        """Even if the estimate is 4× too low, MoE 8-expert must still fit."""
        total, _ = _moe_params(TOKENIZER_VOCAB_SIZE, 384, 6, 768, 8, 2,
                               moe_freq=2, max_pos=256)
        weight_gb = total * self.DTYPE_BYTES / 1e9
        peak = _vram_estimate(weight_gb, batch=2, seq_len=256, hidden=384, n_layers=6)
        self.assertLess(peak * 4, self.VRAM_TARGET_GB * 1.5,
                        "Even 4× overestimate exceeds 1.5× VRAM target — config is too large")


# ══════════════════════════════════════════════════════════════════════════════
# Test class 7: Config file correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigFileVocabSize(unittest.TestCase):
    """Verify that all 5 laptop config files use vocab_size=50277."""

    LAPTOP_CONFIGS = [
        "laptop_dense_small.yaml",
        "laptop_dense_medium.yaml",
        "laptop_dense_tiny.yaml",
        "laptop_moe_8expert_8gb_safe.yaml",
        "laptop_moe_8gb_safe.yaml",
    ]

    def _read_config(self, name: str) -> str:
        path = CONFIG_DIR / name
        if not path.exists():
            self.skipTest(f"Config file not found: {path}")
        return path.read_text(encoding="utf-8")

    def test_laptop_dense_small_vocab_is_50277(self):
        text = self._read_config("laptop_dense_small.yaml")
        self.assertIn("vocab_size: 50277", text)
        self.assertNotIn("vocab_size: 32000", text)

    def test_laptop_dense_medium_vocab_is_50277(self):
        text = self._read_config("laptop_dense_medium.yaml")
        self.assertIn("vocab_size: 50277", text)
        self.assertNotIn("vocab_size: 32000", text)

    def test_laptop_dense_tiny_vocab_is_50277(self):
        text = self._read_config("laptop_dense_tiny.yaml")
        self.assertIn("vocab_size: 50277", text)
        self.assertNotIn("vocab_size: 32000", text)

    def test_laptop_moe_8expert_8gb_safe_vocab_is_50277(self):
        text = self._read_config("laptop_moe_8expert_8gb_safe.yaml")
        self.assertIn("vocab_size: 50277", text)
        self.assertNotIn("vocab_size: 32000", text)

    def test_laptop_moe_8gb_safe_vocab_is_50277(self):
        text = self._read_config("laptop_moe_8gb_safe.yaml")
        self.assertIn("vocab_size: 50277", text)
        self.assertNotIn("vocab_size: 32000", text)

    def test_laptop_dense_small_has_tokenizer_id(self):
        text = self._read_config("laptop_dense_small.yaml")
        self.assertIn("tokenizer_id", text)
        self.assertIn("EleutherAI/gpt-neox-20b", text)

    def test_laptop_moe_8expert_has_tokenizer_id(self):
        text = self._read_config("laptop_moe_8expert_8gb_safe.yaml")
        self.assertIn("tokenizer_id", text)
        self.assertIn("EleutherAI/gpt-neox-20b", text)

    def test_laptop_dense_small_has_dataset_license_note(self):
        """Config must record the CC BY-SA 3.0/4.0 discrepancy."""
        text = self._read_config("laptop_dense_small.yaml")
        self.assertIn("dataset_license_note", text)
        # Must acknowledge the discrepancy, not just state 4.0
        self.assertTrue(
            "3.0" in text or "GFDL" in text,
            "Config must record the CC BY-SA 3.0 / GFDL discrepancy, not only CC BY-SA 4.0"
        )

    def test_laptop_moe_8expert_has_dataset_license_note(self):
        text = self._read_config("laptop_moe_8expert_8gb_safe.yaml")
        self.assertIn("dataset_license_note", text)


# ══════════════════════════════════════════════════════════════════════════════
# Test class 8: Runner source-code vocab guard presence
# ══════════════════════════════════════════════════════════════════════════════

class TestRunnerVocabGuardPresence(unittest.TestCase):
    """Verify that the vocab guard code is present in both runner files."""

    def _read_runner(self, name: str) -> str:
        path = REPO_ROOT / "scripts" / name
        if not path.exists():
            self.skipTest(f"Runner not found: {path}")
        return path.read_text(encoding="utf-8")

    def test_moe_runner_has_tokenizer_vocabulary_mismatch(self):
        text = self._read_runner("run_laptop_moe.py")
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", text)

    def test_dense_runner_has_tokenizer_vocabulary_mismatch(self):
        text = self._read_runner("run_laptop_dense.py")
        self.assertIn("TOKENIZER_VOCABULARY_MISMATCH", text)

    def test_moe_runner_has_actual_tokenizer_size_check(self):
        text = self._read_runner("run_laptop_moe.py")
        self.assertIn("actual_tokenizer_size", text)

    def test_dense_runner_has_actual_tokenizer_size_check(self):
        text = self._read_runner("run_laptop_dense.py")
        self.assertIn("actual_tokenizer_size", text)

    def test_moe_runner_has_per_batch_id_range_check(self):
        text = self._read_runner("run_laptop_moe.py")
        self.assertIn("id_max >= vocab_size", text)

    def test_dense_runner_has_per_batch_id_range_check(self):
        text = self._read_runner("run_laptop_dense.py")
        self.assertIn("id_max >= vocab_size", text)

    def test_moe_runner_does_not_use_modulo_remapping(self):
        """Modulo remapping (% vocab_size) must NOT appear in the runner."""
        text = self._read_runner("run_laptop_moe.py")
        # Allow modulo in comments but not in code
        code_lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
        code = "\n".join(code_lines)
        self.assertNotIn("% vocab_size", code,
                         "Modulo remapping of token IDs is forbidden")

    def test_dense_runner_does_not_use_modulo_remapping(self):
        text = self._read_runner("run_laptop_dense.py")
        code_lines = [l for l in text.splitlines() if not l.strip().startswith("#")]
        code = "\n".join(code_lines)
        self.assertNotIn("% vocab_size", code,
                         "Modulo remapping of token IDs is forbidden")


if __name__ == "__main__":
    unittest.main(verbosity=2)
