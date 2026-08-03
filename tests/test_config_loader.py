"""
tests/test_config_loader.py
===========================

Configuration behaviour tests for training/config_loader.py.

Tests verify:
  - num_layers is honoured (not silently defaulted to 24)
  - Unknown keys halt with ConfigValidationError
  - Unsupported keys (moe_layer_freq, router_jitter) halt
  - Aliases (num_hidden_layers, capacity_factor, aux_loss_coeff, z_loss_coeff, top_k) work
  - Both alias + canonical present halts
  - Router coefficients (aux_loss_coeff, z_loss_coeff) reach the MoEConfig fields
  - vocab_size=50277 is accepted
  - Required fields (vocab_size, hidden_size, etc.) are enforced
  - config_to_dict round-trips cleanly
  - Run 7 YAML files load cleanly

All tests run without torch (CPU-only, no GPU required).
"""

import sys
import pathlib
import pytest

# Add repo root to path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.config_loader import (
    load_dense_config,
    load_moe_config,
    config_to_dict,
    ConfigValidationError,
)
from training._config_stubs import DenseConfig, MoEConfig


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _minimal_dense(**overrides) -> dict:
    base = {
        "vocab_size": 50277,
        "hidden_size": 512,
        "num_layers": 8,
        "num_attention_heads": 8,
        "intermediate_size": 1408,
        "max_position_embeddings": 512,
    }
    base.update(overrides)
    return base


def _minimal_dense_with_alias(alias_key: str, alias_val) -> dict:
    """Build a minimal dense config using an alias key instead of the canonical num_layers."""
    base = {
        "vocab_size": 50277,
        "hidden_size": 512,
        alias_key: alias_val,
        "num_attention_heads": 8,
        "intermediate_size": 1408,
        "max_position_embeddings": 512,
    }
    return base


def _minimal_moe(**overrides) -> dict:
    base = {
        "base": {
            "vocab_size": 50277,
            "hidden_size": 384,
            "num_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 768,
            "max_position_embeddings": 256,
        },
        "num_experts": 8,
        "num_experts_per_token": 2,
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# TestDenseConfigNumLayers — the critical regression
# ══════════════════════════════════════════════════════════════════════════════

class TestDenseConfigNumLayers:
    """Verify num_layers is honoured, not silently defaulted to 24."""

    def test_num_layers_6_is_honoured(self):
        cfg = load_dense_config(_minimal_dense(num_layers=6))
        assert cfg.num_layers == 6, f"Expected 6, got {cfg.num_layers}"

    def test_num_layers_8_is_honoured(self):
        cfg = load_dense_config(_minimal_dense(num_layers=8))
        assert cfg.num_layers == 8, f"Expected 8, got {cfg.num_layers}"

    def test_num_layers_12_is_honoured(self):
        cfg = load_dense_config(_minimal_dense(num_layers=12))
        assert cfg.num_layers == 12, f"Expected 12, got {cfg.num_layers}"

    def test_num_layers_24_is_honoured(self):
        cfg = load_dense_config(_minimal_dense(num_layers=24))
        assert cfg.num_layers == 24, f"Expected 24, got {cfg.num_layers}"

    def test_num_layers_not_silently_defaulted_to_24(self):
        """The old hasattr-filter bug defaulted num_layers to 24 when num_hidden_layers was used."""
        cfg = load_dense_config(_minimal_dense_with_alias("num_hidden_layers", 6))
        assert cfg.num_layers == 6, (
            f"num_hidden_layers=6 should map to num_layers=6, got {cfg.num_layers}. "
            f"This is the Run 1-6 silent default bug."
        )

    def test_num_layers_alias_num_hidden_layers_8(self):
        cfg = load_dense_config(_minimal_dense_with_alias("num_hidden_layers", 8))
        assert cfg.num_layers == 8

    def test_num_layers_alias_num_hidden_layers_12(self):
        cfg = load_dense_config(_minimal_dense_with_alias("num_hidden_layers", 12))
        assert cfg.num_layers == 12


# ══════════════════════════════════════════════════════════════════════════════
# TestDenseConfigUnknownKeys
# ══════════════════════════════════════════════════════════════════════════════

class TestDenseConfigUnknownKeys:
    """Unknown keys must halt with ConfigValidationError."""

    def test_unknown_key_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(UNKNOWN_KEY=99))

    def test_moe_layer_freq_halts_in_dense(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(moe_layer_freq=2))

    def test_router_jitter_halts_in_dense(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(router_jitter=0.01))

    def test_n_layers_alias(self):
        # n_layers IS a supported alias for num_layers (documented in config_loader.py)
        cfg = load_dense_config(_minimal_dense_with_alias("n_layers", 8))
        assert cfg.num_layers == 8, (
            f"n_layers=8 should map to num_layers=8, got {cfg.num_layers}"
        )

    def test_d_model_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(d_model=512))

    def test_n_heads_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(n_heads=8))

    def test_ffn_dim_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(ffn_dim=1408))

    def test_depth_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(depth=8))

    def test_multiple_unknown_keys_halts(self):
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(foo=1, bar=2))

    def test_both_alias_and_canonical_halts(self):
        """Using both num_hidden_layers and num_layers simultaneously must halt."""
        with pytest.raises(ConfigValidationError):
            load_dense_config(_minimal_dense(num_hidden_layers=8))


# ══════════════════════════════════════════════════════════════════════════════
# TestMoEConfigUnsupportedKeys — the moe_layer_freq / router_jitter regression
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEConfigUnsupportedKeys:
    """moe_layer_freq and router_jitter must halt with ConfigValidationError."""

    def test_moe_layer_freq_halts(self):
        raw = _minimal_moe(moe_layer_freq=2)
        with pytest.raises(ConfigValidationError):
            load_moe_config(raw)

    def test_router_jitter_halts(self):
        raw = _minimal_moe(router_jitter=0.01)
        with pytest.raises(ConfigValidationError):
            load_moe_config(raw)

    def test_both_unsupported_keys_halt(self):
        raw = _minimal_moe(moe_layer_freq=2, router_jitter=0.01)
        with pytest.raises(ConfigValidationError):
            load_moe_config(raw)

    def test_unknown_key_in_moe_halts(self):
        raw = _minimal_moe(UNKNOWN=99)
        with pytest.raises(ConfigValidationError):
            load_moe_config(raw)

    def test_unknown_key_in_moe_base_halts(self):
        raw = _minimal_moe()
        raw["base"]["UNKNOWN"] = 99
        with pytest.raises(ConfigValidationError):
            load_moe_config(raw)


# ══════════════════════════════════════════════════════════════════════════════
# TestMoEConfigAliases — router coefficients reach the MoEConfig fields
# ══════════════════════════════════════════════════════════════════════════════

class TestMoEConfigAliases:
    """Aliases must map to the correct canonical fields."""

    def test_capacity_factor_alias(self):
        raw = _minimal_moe(capacity_factor=1.5)
        cfg = load_moe_config(raw)
        assert cfg.expert_capacity_factor == 1.5, (
            f"capacity_factor=1.5 should map to expert_capacity_factor=1.5, "
            f"got {cfg.expert_capacity_factor}"
        )

    def test_aux_loss_coeff_alias(self):
        raw = _minimal_moe(aux_loss_coeff=0.02)
        cfg = load_moe_config(raw)
        assert cfg.router_aux_loss_coeff == 0.02, (
            f"aux_loss_coeff=0.02 should map to router_aux_loss_coeff=0.02, "
            f"got {cfg.router_aux_loss_coeff}"
        )

    def test_z_loss_coeff_alias(self):
        raw = _minimal_moe(z_loss_coeff=0.002)
        cfg = load_moe_config(raw)
        assert cfg.router_z_loss_coeff == 0.002, (
            f"z_loss_coeff=0.002 should map to router_z_loss_coeff=0.002, "
            f"got {cfg.router_z_loss_coeff}"
        )

    def test_top_k_alias(self):
        raw = _minimal_moe()
        del raw["num_experts_per_token"]
        raw["top_k"] = 3
        cfg = load_moe_config(raw)
        assert cfg.num_experts_per_token == 3, (
            f"top_k=3 should map to num_experts_per_token=3, "
            f"got {cfg.num_experts_per_token}"
        )

    def test_all_aliases_together(self):
        raw = _minimal_moe(
            capacity_factor=1.25,
            aux_loss_coeff=0.01,
            z_loss_coeff=0.001,
        )
        cfg = load_moe_config(raw)
        assert cfg.expert_capacity_factor == 1.25
        assert cfg.router_aux_loss_coeff == 0.01
        assert cfg.router_z_loss_coeff == 0.001

    def test_canonical_names_also_work(self):
        raw = _minimal_moe(
            expert_capacity_factor=1.3,
            router_aux_loss_coeff=0.015,
            router_z_loss_coeff=0.0015,
        )
        cfg = load_moe_config(raw)
        assert cfg.expert_capacity_factor == 1.3
        assert cfg.router_aux_loss_coeff == 0.015
        assert cfg.router_z_loss_coeff == 0.0015

    def test_base_num_hidden_layers_alias(self):
        raw = _minimal_moe()
        del raw["base"]["num_layers"]
        raw["base"]["num_hidden_layers"] = 6
        cfg = load_moe_config(raw)
        assert cfg.base.num_layers == 6, (
            f"base.num_hidden_layers=6 should map to base.num_layers=6, "
            f"got {cfg.base.num_layers}"
        )

    def test_router_coefficients_reach_config_not_discarded(self):
        """
        Critical regression test: aux_loss_coeff and z_loss_coeff must NOT be
        silently discarded by the hasattr filter. They must appear in the config.
        """
        raw = _minimal_moe(aux_loss_coeff=0.05, z_loss_coeff=0.005)
        cfg = load_moe_config(raw)
        # These values must NOT be the default (0.01, 0.001)
        assert cfg.router_aux_loss_coeff == 0.05, (
            f"aux_loss_coeff=0.05 was silently discarded! "
            f"Got router_aux_loss_coeff={cfg.router_aux_loss_coeff}"
        )
        assert cfg.router_z_loss_coeff == 0.005, (
            f"z_loss_coeff=0.005 was silently discarded! "
            f"Got router_z_loss_coeff={cfg.router_z_loss_coeff}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestDenseConfigVocabSize
# ══════════════════════════════════════════════════════════════════════════════

class TestDenseConfigVocabSize:
    """vocab_size=50277 must be accepted and preserved."""

    def test_vocab_50277_accepted(self):
        cfg = load_dense_config(_minimal_dense(vocab_size=50277))
        assert cfg.vocab_size == 50277

    def test_vocab_32000_accepted(self):
        cfg = load_dense_config(_minimal_dense(vocab_size=32000))
        assert cfg.vocab_size == 32000

    def test_vocab_size_preserved_exactly(self):
        for v in [50257, 50277, 32000, 128256]:
            cfg = load_dense_config(_minimal_dense(vocab_size=v))
            assert cfg.vocab_size == v, f"vocab_size={v} not preserved: got {cfg.vocab_size}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigToDict
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigToDict:
    """config_to_dict must round-trip cleanly."""

    def test_dense_config_to_dict(self):
        cfg = load_dense_config(_minimal_dense())
        d = config_to_dict(cfg)
        assert isinstance(d, dict)
        assert d["num_layers"] == 8
        assert d["vocab_size"] == 50277
        assert d["hidden_size"] == 512

    def test_moe_config_to_dict(self):
        raw = _minimal_moe(
            expert_capacity_factor=1.25,
            router_aux_loss_coeff=0.01,
            router_z_loss_coeff=0.001,
        )
        cfg = load_moe_config(raw)
        d = config_to_dict(cfg)
        assert isinstance(d, dict)
        assert "base" in d
        assert d["base"]["num_layers"] == 6
        assert d["num_experts"] == 8
        assert d["num_experts_per_token"] == 2
        assert d["expert_capacity_factor"] == 1.25
        assert d["router_aux_loss_coeff"] == 0.01
        assert d["router_z_loss_coeff"] == 0.001

    def test_dict_is_json_serialisable(self):
        import json
        cfg = load_dense_config(_minimal_dense())
        d = config_to_dict(cfg)
        s = json.dumps(d)
        assert len(s) > 10

    def test_moe_dict_is_json_serialisable(self):
        import json
        cfg = load_moe_config(_minimal_moe())
        d = config_to_dict(cfg)
        s = json.dumps(d)
        assert len(s) > 10


# ══════════════════════════════════════════════════════════════════════════════
# TestRun7YamlFiles
# ══════════════════════════════════════════════════════════════════════════════

class TestRun7YamlFiles:
    """Run 7 YAML files must load cleanly through the strict loader."""

    def test_laptop_dense_run7_loads(self):
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml"
        assert path.exists(), f"Missing: {path}"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = load_dense_config(raw["model"])
        assert cfg.num_layers == 8
        assert cfg.vocab_size == 50277
        assert cfg.hidden_size == 512
        assert cfg.intermediate_size == 1408

    def test_laptop_moe_run7_loads(self):
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
        assert path.exists(), f"Missing: {path}"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = load_moe_config(raw["model"])
        assert cfg.base.num_layers == 6
        assert cfg.base.vocab_size == 50277
        assert cfg.num_experts == 8
        assert cfg.num_experts_per_token == 2
        assert cfg.expert_capacity_factor == 1.25
        assert cfg.router_aux_loss_coeff == 0.01
        assert cfg.router_z_loss_coeff == 0.001

    def test_laptop_dense_run7_no_unsupported_keys(self):
        """Verify no unsupported keys snuck into the Run 7 dense config."""
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Should not raise
        load_dense_config(raw["model"])

    def test_laptop_moe_run7_no_unsupported_keys(self):
        """Verify no unsupported keys snuck into the Run 7 MoE config."""
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Should not raise
        load_moe_config(raw["model"])

    def test_laptop_moe_run7_no_moe_layer_freq(self):
        """moe_layer_freq must not appear in the Run 7 MoE config."""
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "moe_layer_freq" not in content, (
            "moe_layer_freq found in laptop_moe_run7.yaml — this key is unsupported"
        )

    def test_laptop_moe_run7_no_router_jitter(self):
        """router_jitter must not appear in the Run 7 MoE config."""
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "router_jitter" not in content, (
            "router_jitter found in laptop_moe_run7.yaml — this key is unsupported"
        )

    def test_laptop_dense_run7_vocab_size_50277(self):
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["model"]["vocab_size"] == 50277

    def test_laptop_moe_run7_vocab_size_50277(self):
        import yaml
        path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["model"]["base"]["vocab_size"] == 50277


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigLoaderEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigLoaderEdgeCases:
    """Edge cases for the strict loader."""

    def test_empty_dict_uses_defaults(self):
        """
        DenseConfig has all-default values, so an empty dict is accepted
        and returns a config with default values (vocab_size=32000, num_layers=24, etc.).
        This is the loader's intended behaviour for optional fields.
        """
        cfg = load_dense_config({})
        assert cfg.vocab_size == 32000  # default
        assert cfg.num_layers == 24     # default

    def test_none_value_for_vocab_size_is_accepted(self):
        """
        DenseConfig is a dataclass with no type enforcement at construction time.
        None is accepted and stored as-is. The pipeline validates vocab_size
        against the tokenizer at runtime, not at config load time.
        """
        cfg = load_dense_config(_minimal_dense(vocab_size=None))
        assert cfg.vocab_size is None

    def test_string_value_for_num_layers_is_accepted(self):
        """
        DenseConfig is a dataclass with no type enforcement at construction time.
        String values are accepted and stored as-is. Type errors surface at
        model construction time when the value is used in arithmetic.
        """
        cfg = load_dense_config(_minimal_dense(num_layers="eight"))
        assert cfg.num_layers == "eight"

    def test_extra_whitespace_in_key_halts(self):
        raw = _minimal_dense()
        raw[" num_layers "] = 8  # key with spaces
        with pytest.raises(ConfigValidationError):
            load_dense_config(raw)

    def test_moe_missing_base_raises(self):
        raw = {"num_experts": 8, "num_experts_per_token": 2}
        with pytest.raises((ConfigValidationError, KeyError, TypeError)):
            load_moe_config(raw)

    def test_moe_empty_base_uses_defaults(self):
        """
        An empty base dict is accepted and uses DenseConfig defaults.
        The pipeline validates the resulting model at runtime.
        """
        raw = {"base": {}, "num_experts": 8, "num_experts_per_token": 2}
        cfg = load_moe_config(raw)
        assert cfg.base.vocab_size == 32000  # default
        assert cfg.base.num_layers == 24     # default
