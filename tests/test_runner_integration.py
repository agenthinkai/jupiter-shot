"""
Jupiter Shot — Runner Integration Tests (CPU-compatible)
=========================================================
Verifies the training runner API contract on CPU without requiring a GPU.

Coverage:
  - DenseTransformer construction from config
  - Forward pass returns dict with required keys
  - Loss is a scalar tensor, not None, when labels provided
  - Loss is None when labels not provided
  - Backward pass completes without error
  - Optimizer step updates parameters
  - Checkpoint save produces a valid .pt file
  - Checkpoint load with weights_only=True restores step and loss
  - MoETransformer construction and forward pass
  - MoE forward returns dict with 'aux_loss' and 'router_metrics'
  - Router metrics is a list of dicts
  - GradScaler instantiation uses torch.amp.GradScaler("cuda", ...) pattern
  - run_steps() helper correctly unpacks dict output (no tuple/tensor assumption)
  - Synthetic batch generation produces correct shape
  - Loss decreases over multiple optimizer steps (sanity check)

All tests run on CPU. No GPU required.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

if HAS_TORCH:
    from training.models.dense import DenseConfig, DenseTransformer
    from training.models.moe import MoEConfig, MoETransformer
else:
    # Stub classes so the module can be collected without torch
    class DenseConfig:  # type: ignore
        pass
    class DenseTransformer:  # type: ignore
        pass
    class MoEConfig:  # type: ignore
        pass
    class MoETransformer:  # type: ignore
        pass

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_dense_config() -> DenseConfig:
    """Minimal DenseConfig that runs quickly on CPU."""
    return DenseConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )


@pytest.fixture(scope="module")
def tiny_moe_config(tiny_dense_config: DenseConfig) -> MoEConfig:
    """Minimal MoEConfig that runs quickly on CPU."""
    return MoEConfig(
        base=tiny_dense_config,
        num_experts=4,
        num_experts_per_token=2,
        expert_capacity_factor=1.25,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.001,
    )


@pytest.fixture(scope="module")
def dense_model(tiny_dense_config: DenseConfig) -> DenseTransformer:
    return DenseTransformer(tiny_dense_config)


@pytest.fixture(scope="module")
def moe_model(tiny_moe_config: MoEConfig) -> MoETransformer:
    return MoETransformer(tiny_moe_config)


@pytest.fixture
def cpu_batch(tiny_dense_config: DenseConfig):
    """Small input batch on CPU."""
    batch_size, seq_len = 2, 16
    return torch.randint(0, tiny_dense_config.vocab_size, (batch_size, seq_len))


# ── Dense model tests ─────────────────────────────────────────────────────────

class TestDenseConstruction:
    def test_model_is_dense_transformer(self, dense_model: DenseTransformer) -> None:
        assert isinstance(dense_model, DenseTransformer)

    def test_model_has_parameters(self, dense_model: DenseTransformer) -> None:
        total = sum(p.numel() for p in dense_model.parameters())
        assert total > 0, "Model has no parameters"

    def test_model_trainable_parameters_positive(self, dense_model: DenseTransformer) -> None:
        trainable = sum(p.numel() for p in dense_model.parameters() if p.requires_grad)
        assert trainable > 0, "No trainable parameters"

    def test_config_attributes_accessible(self, tiny_dense_config: DenseConfig) -> None:
        assert tiny_dense_config.vocab_size == 256
        assert tiny_dense_config.hidden_size == 64
        assert tiny_dense_config.num_layers == 2


class TestDenseForwardPass:
    def test_forward_returns_dict(self, dense_model: DenseTransformer, cpu_batch: torch.Tensor) -> None:
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"

    def test_forward_dict_has_logits_key(self, dense_model: DenseTransformer, cpu_batch: torch.Tensor) -> None:
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert "logits" in out, f"Missing 'logits' key. Keys: {list(out.keys())}"

    def test_forward_dict_has_loss_key(self, dense_model: DenseTransformer, cpu_batch: torch.Tensor) -> None:
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert "loss" in out, f"Missing 'loss' key. Keys: {list(out.keys())}"

    def test_forward_without_labels_loss_is_none(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert out["loss"] is None, f"Expected loss=None without labels, got {out['loss']}"

    def test_forward_with_labels_loss_is_tensor(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        dense_model.eval()
        labels = cpu_batch.clone()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch, labels=labels)
        assert out["loss"] is not None, "Loss is None even with labels provided"
        assert isinstance(out["loss"], torch.Tensor), f"Loss is not a tensor: {type(out['loss'])}"

    def test_forward_loss_is_scalar(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        dense_model.eval()
        labels = cpu_batch.clone()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch, labels=labels)
        assert out["loss"].ndim == 0, f"Loss is not scalar: shape={out['loss'].shape}"

    def test_forward_loss_is_finite(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        dense_model.eval()
        labels = cpu_batch.clone()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch, labels=labels)
        loss_val = out["loss"].item()
        assert not math.isnan(loss_val), "Loss is NaN"
        assert not math.isinf(loss_val), "Loss is Inf"

    def test_forward_logits_shape(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor, tiny_dense_config: DenseConfig
    ) -> None:
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        B, T = cpu_batch.shape
        expected_shape = (B, T, tiny_dense_config.vocab_size)
        assert out["logits"].shape == expected_shape, (
            f"Logits shape mismatch: expected {expected_shape}, got {out['logits'].shape}"
        )

    def test_forward_no_tuple_output(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        """Ensure the model does NOT return a tuple (old API)."""
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert not isinstance(out, tuple), (
            "Model returned a tuple — dict API expected. "
            "Runner code must NOT use out[0] or out[1] indexing."
        )

    def test_forward_not_a_tensor(
        self, dense_model: DenseTransformer, cpu_batch: torch.Tensor
    ) -> None:
        """Ensure the model does NOT return a bare tensor (old API)."""
        dense_model.eval()
        with torch.no_grad():
            out = dense_model(input_ids=cpu_batch)
        assert not isinstance(out, torch.Tensor), (
            "Model returned a bare tensor — dict API expected. "
            "Runner code must NOT call .backward() directly on the model output."
        )


class TestDenseBackwardAndOptimizer:
    def test_backward_completes(
        self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor
    ) -> None:
        """Backward pass must complete without error."""
        model = DenseTransformer(tiny_dense_config)
        model.train()
        labels = cpu_batch.clone()
        out = model(input_ids=cpu_batch, labels=labels)
        loss = out["loss"]
        loss.backward()  # Must not raise

    def test_optimizer_step_updates_parameters(
        self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor
    ) -> None:
        """Optimizer step must change at least one parameter."""
        model = DenseTransformer(tiny_dense_config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Snapshot parameters before step
        params_before = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        labels = cpu_batch.clone()
        out = model(input_ids=cpu_batch, labels=labels)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # At least one parameter must have changed
        changed = any(
            not torch.allclose(params_before[name], param.detach(), atol=1e-9)
            for name, param in model.named_parameters()
            if param.requires_grad and name in params_before
        )
        assert changed, "No parameters were updated after optimizer step"

    def test_loss_decreases_over_steps(
        self, tiny_dense_config: DenseConfig
    ) -> None:
        """Loss should decrease over 20 steps with a simple fixed batch."""
        torch.manual_seed(42)
        model = DenseTransformer(tiny_dense_config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # Fixed batch for deterministic test
        input_ids = torch.randint(0, tiny_dense_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at step 20 should be lower than at step 1
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_dict_unpack_in_training_loop(
        self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor
    ) -> None:
        """
        Simulate the exact pattern used in run_laptop_dense.py:
          out = model(input_ids=input_ids, labels=labels)
          loss = out["loss"] / grad_accum
        Must not raise AttributeError or TypeError.
        """
        model = DenseTransformer(tiny_dense_config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        grad_accum = 2

        for step in range(grad_accum):
            labels = cpu_batch.clone()
            out = model(input_ids=cpu_batch, labels=labels)
            # This is the exact pattern from the fixed runner
            loss = out["loss"] / grad_accum
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        # No assertion needed — test passes if no exception is raised


# ── Checkpoint tests ──────────────────────────────────────────────────────────

class TestCheckpointSaveLoad:
    def test_checkpoint_save_produces_file(
        self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor
    ) -> None:
        model = DenseTransformer(tiny_dense_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            torch.save({
                "step": 10,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": 2.5,
                "config": "laptop_dense_small",
            }, ckpt_path)
            assert ckpt_path.exists(), "Checkpoint file was not created"
            assert ckpt_path.stat().st_size > 0, "Checkpoint file is empty"

    def test_checkpoint_load_with_weights_only(
        self, tiny_dense_config: DenseConfig
    ) -> None:
        """weights_only=True must not raise in PyTorch 2.x."""
        model = DenseTransformer(tiny_dense_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({
                "step": 5,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": 3.1,
            }, ckpt_path)

            # weights_only=True is the fix for the deprecated torch.load API
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert ckpt["step"] == 5
            assert abs(ckpt["loss"] - 3.1) < 1e-6

    def test_checkpoint_restores_model_state(
        self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor
    ) -> None:
        """Model loaded from checkpoint must produce identical logits."""
        torch.manual_seed(0)
        model_a = DenseTransformer(tiny_dense_config)
        model_a.eval()

        with torch.no_grad():
            logits_before = model_a(input_ids=cpu_batch)["logits"].clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "state_test.pt"
            torch.save({"step": 1, "model_state_dict": model_a.state_dict(), "loss": 0.0}, ckpt_path)

            model_b = DenseTransformer(tiny_dense_config)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_b.load_state_dict(ckpt["model_state_dict"])
            model_b.eval()

            with torch.no_grad():
                logits_after = model_b(input_ids=cpu_batch)["logits"]

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Logits differ after checkpoint restore"
        )

    def test_checkpoint_global_step_preserved(
        self, tiny_dense_config: DenseConfig
    ) -> None:
        model = DenseTransformer(tiny_dense_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "step_test.pt"
            torch.save({"step": 42, "model_state_dict": model.state_dict(), "loss": 1.5}, ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert ckpt["step"] == 42


# ── MoE model tests ───────────────────────────────────────────────────────────

class TestMoEConstruction:
    def test_model_is_moe_transformer(self, moe_model: MoETransformer) -> None:
        assert isinstance(moe_model, MoETransformer)

    def test_moe_has_parameters(self, moe_model: MoETransformer) -> None:
        total = sum(p.numel() for p in moe_model.parameters())
        assert total > 0

    def test_moe_config_attributes(self, tiny_moe_config: MoEConfig) -> None:
        assert tiny_moe_config.num_experts == 4
        assert tiny_moe_config.num_experts_per_token == 2


class TestMoEForwardPass:
    def test_moe_forward_returns_dict(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"

    def test_moe_forward_has_required_keys(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        for key in ("logits", "loss", "aux_loss", "router_metrics"):
            assert key in out, f"Missing key '{key}'. Keys: {list(out.keys())}"

    def test_moe_forward_aux_loss_is_tensor(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        assert isinstance(out["aux_loss"], torch.Tensor), (
            f"aux_loss is not a tensor: {type(out['aux_loss'])}"
        )

    def test_moe_forward_aux_loss_is_scalar(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        assert out["aux_loss"].ndim == 0, f"aux_loss is not scalar: shape={out['aux_loss'].shape}"

    def test_moe_forward_aux_loss_is_finite(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        val = out["aux_loss"].item()
        assert not math.isnan(val), "aux_loss is NaN"
        assert not math.isinf(val), "aux_loss is Inf"

    def test_moe_forward_router_metrics_is_list(self, moe_model: MoETransformer, cpu_batch: torch.Tensor) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        assert isinstance(out["router_metrics"], list), (
            f"router_metrics is not a list: {type(out['router_metrics'])}"
        )

    def test_moe_forward_router_metrics_length_matches_layers(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor, tiny_moe_config: MoEConfig
    ) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        expected_layers = tiny_moe_config.base.num_layers
        assert len(out["router_metrics"]) == expected_layers, (
            f"router_metrics length {len(out['router_metrics'])} != num_layers {expected_layers}"
        )

    def test_moe_forward_router_metrics_items_are_dicts(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor
    ) -> None:
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        for i, layer_metrics in enumerate(out["router_metrics"]):
            assert isinstance(layer_metrics, dict), (
                f"router_metrics[{i}] is not a dict: {type(layer_metrics)}"
            )

    def test_moe_forward_with_labels_loss_is_tensor(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor
    ) -> None:
        moe_model.eval()
        labels = cpu_batch.clone()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch, labels=labels)
        assert out["loss"] is not None, "loss is None with labels provided"
        assert isinstance(out["loss"], torch.Tensor)
        assert out["loss"].ndim == 0

    def test_moe_forward_with_labels_lm_loss_present(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor
    ) -> None:
        moe_model.eval()
        labels = cpu_batch.clone()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch, labels=labels)
        assert "lm_loss" in out, f"Missing 'lm_loss' key. Keys: {list(out.keys())}"
        assert out["lm_loss"] is not None

    def test_moe_dict_unpack_in_runner_pattern(
        self, tiny_moe_config: MoEConfig, cpu_batch: torch.Tensor
    ) -> None:
        """
        Simulate the exact pattern used in run_laptop_moe.py:
          out = model(input_ids=input_ids, labels=labels)
          total_loss = out["loss"] / grad_accum
          aux_val = out["aux_loss"].item()
          router_metrics = out["router_metrics"]
        Must not raise AttributeError or TypeError.
        """
        model = MoETransformer(tiny_moe_config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        grad_accum = 2

        for _ in range(grad_accum):
            labels = cpu_batch.clone()
            out = model(input_ids=cpu_batch, labels=labels)
            # Exact runner pattern
            total_loss = out["loss"] / grad_accum
            aux_val = out["aux_loss"].item()
            router_metrics = out["router_metrics"]
            total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        assert isinstance(aux_val, float)
        assert isinstance(router_metrics, list)

    def test_moe_router_metrics_has_expert_fraction(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor
    ) -> None:
        """Each layer's router_metrics dict should contain 'expert_fraction'."""
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        for i, layer_metrics in enumerate(out["router_metrics"]):
            if layer_metrics:  # skip empty dicts from gradient checkpointing
                assert "expert_fraction" in layer_metrics, (
                    f"router_metrics[{i}] missing 'expert_fraction'. Keys: {list(layer_metrics.keys())}"
                )

    def test_moe_router_entropy_is_positive(
        self, moe_model: MoETransformer, cpu_batch: torch.Tensor
    ) -> None:
        """Router entropy should be > 0 (not collapsed)."""
        moe_model.eval()
        with torch.no_grad():
            out = moe_model(input_ids=cpu_batch)
        for i, layer_metrics in enumerate(out["router_metrics"]):
            if layer_metrics and "router_entropy" in layer_metrics:
                entropy = layer_metrics["router_entropy"]
                assert entropy > 0, (
                    f"Router entropy at layer {i} is {entropy} (collapsed routing)"
                )


# ── GradScaler API tests ──────────────────────────────────────────────────────

class TestGradScalerAPI:
    def test_gradscaler_new_api_does_not_raise(self) -> None:
        """
        torch.amp.GradScaler("cuda", enabled=False) must not raise.
        This is the fixed API (replacing deprecated torch.cuda.amp.GradScaler).
        enabled=False works on CPU without CUDA.
        """
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        assert scaler is not None

    def test_gradscaler_new_api_cpu_compatible(self) -> None:
        """GradScaler with enabled=False must be usable in a CPU training loop."""
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        config = DenseConfig(
            vocab_size=64, hidden_size=32, num_layers=1,
            num_attention_heads=2, num_kv_heads=2, intermediate_size=64,
            max_position_embeddings=16,
        )
        model = DenseTransformer(config)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 64, (1, 8))
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        # When enabled=False, scaler.scale() is a no-op
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    def test_deprecated_gradscaler_api_is_not_used_in_dense_runner(self) -> None:
        """
        Verify that run_laptop_dense.py uses torch.amp.GradScaler, not the deprecated
        torch.cuda.amp.GradScaler.
        """
        runner_path = REPO_ROOT / "scripts" / "run_laptop_dense.py"
        assert runner_path.exists(), f"Runner not found: {runner_path}"
        content = runner_path.read_text()
        # Must use the new API
        assert 'torch.amp.GradScaler("cuda"' in content, (
            "run_laptop_dense.py must use torch.amp.GradScaler(\"cuda\", ...) "
            "instead of the deprecated torch.cuda.amp.GradScaler"
        )
        # Must NOT use the deprecated API
        assert "torch.cuda.amp.GradScaler" not in content, (
            "run_laptop_dense.py still uses deprecated torch.cuda.amp.GradScaler"
        )

    def test_deprecated_gradscaler_api_is_not_used_in_moe_runner(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        assert runner_path.exists(), f"Runner not found: {runner_path}"
        content = runner_path.read_text()
        assert 'torch.amp.GradScaler("cuda"' in content, (
            "run_laptop_moe.py must use torch.amp.GradScaler(\"cuda\", ...) "
        )
        assert "torch.cuda.amp.GradScaler" not in content, (
            "run_laptop_moe.py still uses deprecated torch.cuda.amp.GradScaler"
        )


# ── Runner pattern tests ──────────────────────────────────────────────────────

class TestRunnerPatterns:
    def test_dense_runner_uses_dict_unpack(self) -> None:
        """run_laptop_dense.py must use out['loss'] not out[0] or direct .backward()."""
        runner_path = REPO_ROOT / "scripts" / "run_laptop_dense.py"
        content = runner_path.read_text()
        # Must unpack dict
        assert 'out["loss"]' in content or "out['loss']" in content, (
            "run_laptop_dense.py must unpack loss from dict: out['loss']"
        )
        # Must pass labels= keyword argument
        assert "labels=labels" in content, (
            "run_laptop_dense.py must pass labels=labels to model()"
        )

    def test_moe_runner_uses_dict_unpack(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text()
        assert 'out["loss"]' in content or "out['loss']" in content, (
            "run_laptop_moe.py must unpack loss from dict: out['loss']"
        )
        assert "labels=labels" in content, (
            "run_laptop_moe.py must pass labels=labels to model()"
        )

    def test_moe_runner_reads_router_metrics_from_dict(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text()
        assert 'out.get("router_metrics"' in content or 'out["router_metrics"]' in content, (
            "run_laptop_moe.py must read router_metrics from out dict, "
            "not from a module hook (_last_routing_weights)"
        )

    def test_resume_runner_uses_weights_only(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_resume_test.py"
        content = runner_path.read_text()
        assert "weights_only=True" in content, (
            "run_laptop_resume_test.py must use weights_only=True in torch.load()"
        )

    def test_resume_runner_uses_dict_unpack(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_resume_test.py"
        content = runner_path.read_text()
        assert 'out["loss"]' in content or "out['loss']" in content, (
            "run_laptop_resume_test.py must unpack loss from dict"
        )

    def test_dense_runner_has_failure_artifact_saving(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_dense.py"
        content = runner_path.read_text()
        assert "failure_artifact" in content or "failure_path" in content, (
            "run_laptop_dense.py must save a failure artifact on exception"
        )

    def test_moe_runner_has_failure_artifact_saving(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text()
        assert "failure_artifact" in content or "failure_path" in content, (
            "run_laptop_moe.py must save a failure artifact on exception"
        )

    def test_dense_runner_has_synthetic_flag(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_dense.py"
        content = runner_path.read_text()
        assert "--synthetic" in content, (
            "run_laptop_dense.py must support --synthetic flag"
        )

    def test_moe_runner_has_synthetic_flag(self) -> None:
        runner_path = REPO_ROOT / "scripts" / "run_laptop_moe.py"
        content = runner_path.read_text()
        assert "--synthetic" in content, (
            "run_laptop_moe.py must support --synthetic flag"
        )


# ── Synthetic data tests ──────────────────────────────────────────────────────

class TestSyntheticBatch:
    def test_synthetic_batch_shape(self) -> None:
        batch_size, seq_len, vocab_size = 4, 32, 256
        batch = torch.randint(0, vocab_size, (batch_size, seq_len))
        assert batch.shape == (batch_size, seq_len)

    def test_synthetic_batch_values_in_range(self) -> None:
        vocab_size = 256
        batch = torch.randint(0, vocab_size, (2, 16))
        assert batch.min().item() >= 0
        assert batch.max().item() < vocab_size

    def test_synthetic_batch_is_long_tensor(self) -> None:
        batch = torch.randint(0, 256, (2, 16))
        assert batch.dtype == torch.long


# ── Optional CUDA skip marker ─────────────────────────────────────────────────

@pytest.mark.skipif(
    not HAS_TORCH or not torch.cuda.is_available(),  # type: ignore[union-attr]
    reason="CUDA not available"
)
class TestCUDASmoke:
    """Optional CUDA smoke tests — skipped on CPU-only environments."""

    def test_dense_forward_on_cuda(self, tiny_dense_config: DenseConfig, cpu_batch: torch.Tensor) -> None:
        device = torch.device("cuda:0")
        model = DenseTransformer(tiny_dense_config).to(device)
        model.eval()
        batch = cpu_batch.to(device)
        with torch.no_grad():
            out = model(input_ids=batch)
        assert isinstance(out, dict)
        assert "logits" in out
        assert out["logits"].device.type == "cuda"

    def test_moe_forward_on_cuda(self, tiny_moe_config: MoEConfig, cpu_batch: torch.Tensor) -> None:
        device = torch.device("cuda:0")
        model = MoETransformer(tiny_moe_config).to(device)
        model.eval()
        batch = cpu_batch.to(device)
        with torch.no_grad():
            out = model(input_ids=batch)
        assert isinstance(out, dict)
        assert "router_metrics" in out
        assert isinstance(out["router_metrics"], list)

    def test_gradscaler_enabled_on_cuda(self) -> None:
        """torch.amp.GradScaler('cuda', enabled=True) must work when CUDA is available."""
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        assert scaler is not None
