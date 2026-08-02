"""
Jupiter Shot — Model Unit Tests
=================================
Tests for DenseTransformer and MoETransformer.

Run:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v --tb=short -x
"""

import pytest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ── Dense Model Tests ─────────────────────────────────────────────────────────

class TestDenseConfig:
    def test_default_config(self):
        from training.models.dense import DenseConfig
        cfg = DenseConfig()
        assert cfg.vocab_size == 32000
        assert cfg.hidden_size == 2048
        assert cfg.num_layers == 24
        assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads

    def test_named_configs(self):
        from training.models.dense import NAMED_CONFIGS
        assert "125m" in NAMED_CONFIGS
        assert "1.3b" in NAMED_CONFIGS
        for name, cfg in NAMED_CONFIGS.items():
            assert cfg.hidden_size % cfg.num_attention_heads == 0, (
                f"Config '{name}': hidden_size must be divisible by num_attention_heads"
            )

    def test_intermediate_size_auto(self):
        from training.models.dense import DenseConfig
        cfg = DenseConfig(hidden_size=256, num_layers=2, num_attention_heads=4)
        assert cfg.intermediate_size > 0
        assert cfg.intermediate_size % 64 == 0  # Should be rounded to multiple of 64


class TestDenseTransformer:
    @pytest.fixture
    def tiny_model(self):
        from training.models.dense import DenseConfig, DenseTransformer
        cfg = DenseConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
            gradient_checkpointing=False,
        )
        return DenseTransformer(cfg)

    def test_instantiation(self, tiny_model):
        assert tiny_model is not None
        n = tiny_model.count_parameters()
        assert n > 0
        assert n < 10_000_000  # Tiny model should be < 10M params

    def test_forward_shape(self, tiny_model):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        out = tiny_model(input_ids=input_ids)
        assert "logits" in out
        assert out["logits"].shape == (B, T, 1000)

    def test_forward_with_labels(self, tiny_model):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 1000, (B, T))
        out = tiny_model(input_ids=input_ids, labels=labels)
        assert "loss" in out
        assert out["loss"] is not None
        assert out["loss"].item() > 0
        assert not torch.isnan(out["loss"])
        assert not torch.isinf(out["loss"])

    def test_backward(self, tiny_model):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 1000, (B, T))
        out = tiny_model(input_ids=input_ids, labels=labels)
        out["loss"].backward()
        # Check gradients exist
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_tied_embeddings(self, tiny_model):
        assert tiny_model.embed_tokens.weight is tiny_model.lm_head.weight

    def test_loss_decreases_with_training(self, tiny_model):
        """Loss should decrease over a few gradient steps on a fixed batch."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = input_ids.clone()

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = tiny_model(input_ids=input_ids, labels=labels)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        # Loss should be lower at end than start
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_no_nan_in_forward(self, tiny_model):
        """Forward pass should not produce NaN values."""
        input_ids = torch.randint(0, 1000, (4, 64))
        labels = torch.randint(0, 1000, (4, 64))
        out = tiny_model(input_ids=input_ids, labels=labels)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isnan(out["loss"])

    def test_different_sequence_lengths(self, tiny_model):
        for T in [1, 16, 64, 128]:
            input_ids = torch.randint(0, 1000, (1, T))
            out = tiny_model(input_ids=input_ids)
            assert out["logits"].shape == (1, T, 1000)

    def test_parameter_count_1b3(self):
        import psutil, os
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 6.0:
            pytest.skip(f"Skipped: only {available_gb:.1f}GB RAM available; 1.3B model requires ~6GB. Run on a machine with ≥16GB RAM.")
        from training.models.dense import NAMED_CONFIGS, DenseTransformer
        cfg = NAMED_CONFIGS["1.3b"]
        model = DenseTransformer(cfg)
        n = model.count_parameters()
        # Should be between 1.0B and 1.6B
        assert 1.0e9 < n < 1.6e9, f"Expected ~1.3B params, got {n/1e9:.2f}B"


# ── MoE Model Tests ───────────────────────────────────────────────────────────

class TestTopKRouter:
    @pytest.fixture
    def router(self):
        from training.models.moe import TopKRouter
        return TopKRouter(
            hidden_size=64,
            num_experts=4,
            num_experts_per_token=2,
            aux_loss_coeff=0.01,
            z_loss_coeff=0.001,
        )

    def test_output_shapes(self, router):
        B, T, H = 2, 16, 64
        x = torch.randn(B * T, H)
        weights, indices, aux_loss, metrics = router(x)
        assert weights.shape == (B * T, 2)
        assert indices.shape == (B * T, 2)
        assert aux_loss.ndim == 0  # Scalar

    def test_weights_sum_to_one(self, router):
        x = torch.randn(32, 64)
        weights, indices, _, _ = router(x)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(32), atol=1e-5)

    def test_indices_in_range(self, router):
        x = torch.randn(32, 64)
        _, indices, _, _ = router(x)
        assert indices.min() >= 0
        assert indices.max() < 4  # num_experts = 4

    def test_aux_loss_positive(self, router):
        x = torch.randn(32, 64)
        _, _, aux_loss, _ = router(x)
        assert aux_loss.item() >= 0

    def test_metrics_keys(self, router):
        x = torch.randn(32, 64)
        _, _, _, metrics = router(x)
        assert "expert_counts" in metrics
        assert "load_imbalance_ratio" in metrics
        assert "router_entropy" in metrics

    def test_aux_loss_backward(self, router):
        x = torch.randn(32, 64, requires_grad=True)
        _, _, aux_loss, _ = router(x)
        aux_loss.backward()
        assert x.grad is not None


class TestMoETransformer:
    @pytest.fixture
    def tiny_moe(self):
        from training.models.moe import MoEConfig, MoETransformer
        from training.models.dense import DenseConfig
        cfg = MoEConfig(
            base=DenseConfig(
                vocab_size=1000,
                hidden_size=64,
                num_layers=2,
                num_attention_heads=4,
                max_position_embeddings=512,
                gradient_checkpointing=False,
            ),
            num_experts=4,
            num_experts_per_token=2,
            router_aux_loss_coeff=0.01,
            router_z_loss_coeff=0.001,
        )
        return MoETransformer(cfg)

    def test_instantiation(self, tiny_moe):
        assert tiny_moe is not None
        n = tiny_moe.count_parameters()
        assert n > 0

    def test_forward_shape(self, tiny_moe):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        out = tiny_moe(input_ids=input_ids)
        assert out["logits"].shape == (B, T, 1000)

    def test_aux_loss_present(self, tiny_moe):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 1000, (B, T))
        out = tiny_moe(input_ids=input_ids, labels=labels)
        assert "aux_loss" in out
        assert out["aux_loss"] is not None
        assert out["aux_loss"].item() >= 0

    def test_total_loss_includes_aux(self, tiny_moe):
        B, T = 2, 32
        input_ids = torch.randint(0, 1000, (B, T))
        labels = torch.randint(0, 1000, (B, T))
        out = tiny_moe(input_ids=input_ids, labels=labels)
        # total_loss = lm_loss + aux_loss
        expected = out["lm_loss"] + out["aux_loss"]
        assert torch.allclose(out["loss"], expected, atol=1e-6)

    def test_backward(self, tiny_moe):
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        out = tiny_moe(input_ids=input_ids, labels=labels)
        out["loss"].backward()
        for name, param in tiny_moe.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan(self, tiny_moe):
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        out = tiny_moe(input_ids=input_ids, labels=labels)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isnan(out["loss"])

    def test_router_metrics_present(self, tiny_moe):
        input_ids = torch.randint(0, 1000, (2, 32))
        out = tiny_moe(input_ids=input_ids)
        assert "router_metrics" in out
        assert len(out["router_metrics"]) == 2  # num_layers = 2

    def test_more_total_than_active_params(self, tiny_moe):
        total = tiny_moe.count_parameters()
        active = tiny_moe.config.count_active_parameters()
        assert total > active, "MoE total params should exceed active params"

    def test_shared_expert(self):
        from training.models.moe import MoEConfig, MoETransformer
        from training.models.dense import DenseConfig
        cfg = MoEConfig(
            base=DenseConfig(
                vocab_size=1000, hidden_size=64, num_layers=2,
                num_attention_heads=4, max_position_embeddings=512,
                gradient_checkpointing=False,
            ),
            num_experts=4,
            num_experts_per_token=2,
            use_shared_expert=True,
        )
        model = MoETransformer(cfg)
        input_ids = torch.randint(0, 1000, (2, 16))
        out = model(input_ids=input_ids)
        assert out["logits"].shape == (2, 16, 1000)


# ── RoPE Tests ────────────────────────────────────────────────────────────────

class TestRoPE:
    def test_rope_freqs_shape(self):
        from training.models.dense import precompute_rope_freqs
        cos, sin = precompute_rope_freqs(head_dim=64, max_seq_len=512, theta=10000.0)
        assert cos.shape == (512, 64)
        assert sin.shape == (512, 64)

    def test_rope_values_bounded(self):
        from training.models.dense import precompute_rope_freqs
        cos, sin = precompute_rope_freqs(head_dim=64, max_seq_len=512, theta=10000.0)
        assert cos.abs().max() <= 1.0 + 1e-6
        assert sin.abs().max() <= 1.0 + 1e-6

    def test_rope_apply_shape(self):
        from training.models.dense import precompute_rope_freqs, apply_rope
        cos, sin = precompute_rope_freqs(head_dim=64, max_seq_len=32, theta=10000.0)
        q = torch.randn(2, 4, 32, 64)  # (B, heads, T, head_dim)
        k = torch.randn(2, 4, 32, 64)
        q_rot, k_rot = apply_rope(q, k, cos[:32], sin[:32])
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
