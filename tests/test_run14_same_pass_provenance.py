"""
test_run14_same_pass_provenance.py
==================================
Regression test — Run 14 same-pass router-metrics provenance.

Requirement (from Run 14 review):
    "The assignments and statistics used for acceptance must derive from the
    exact router logits, probabilities, expert indices and assignment counts
    used by the training forward pass."

    "Add a regression test with deliberately stochastic router jitter enabled.
    Instrument router calls and prove that the metrics correspond to the routing
    assignment used by the loss-producing forward pass—not a later probe."

Strategy
--------
1. Patch TopKRouter.forward to count how many times it is called per
   MoETransformer.forward() invocation.
2. Run a training-mode forward pass with gradient_checkpointing=True.
3. Assert that the router was called exactly once per layer — not twice.
   A second call (the old probe pattern) would show count == 2.
4. Inject deliberate stochastic jitter (additive Gaussian noise on the gate
   weight output) so that a second call would produce measurably different
   routing decisions.
5. Verify that the metrics dict returned by the model matches the routing
   decisions recorded during the single router call — not a hypothetical
   second call.

Test IDs
--------
T16  router_called_exactly_once_per_layer_under_gc
T17  metrics_match_training_pass_assignments_under_jitter
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from training.models.moe import MoEConfig, MoETransformer
from training.models.dense import DenseConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_moe_config(gradient_checkpointing: bool = True) -> MoEConfig:
    """Return a minimal MoE config for fast CPU tests."""
    base = DenseConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
        gradient_checkpointing=gradient_checkpointing,
    )
    return MoEConfig(
        base=base,
        num_experts=4,
        num_experts_per_token=2,
        expert_capacity_factor=1.5,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.001,
    )


@contextmanager
def _count_router_calls(model: MoETransformer):
    """
    Context manager that patches every TopKRouter.forward in the model to
    record how many times it is called.  Yields a dict mapping layer_index
    to call_count.

    The patch wraps the real forward so routing still executes normally —
    we are only counting calls, not replacing them.
    """
    call_counts: dict[int, int] = {}
    original_forwards: dict[int, Any] = {}
    lock = threading.Lock()

    for layer_idx, layer in enumerate(model.layers):
        call_counts[layer_idx] = 0
        original_forward = layer.moe_ffn.router.forward

        def make_wrapper(idx, orig):
            def wrapper(*args, **kwargs):
                with lock:
                    call_counts[idx] += 1
                return orig(*args, **kwargs)
            return wrapper

        original_forwards[layer_idx] = original_forward
        layer.moe_ffn.router.forward = make_wrapper(layer_idx, original_forward)

    try:
        yield call_counts
    finally:
        # Restore originals
        for layer_idx, layer in enumerate(model.layers):
            layer.moe_ffn.router.forward = original_forwards[layer_idx]


def _inject_gate_jitter(model: MoETransformer, noise_std: float = 0.5) -> None:
    """
    Register a forward hook on every router gate (nn.Linear) that adds
    Gaussian noise to the output logits.  This makes each router call
    produce different logits, so a second call would produce different
    routing decisions.

    The hook is stored on the gate module itself so it can be removed later.
    """
    hooks = []
    for layer in model.layers:
        gate = layer.moe_ffn.router.gate

        def make_hook(std):
            def hook(module, input, output):
                if module.training:
                    return output + torch.randn_like(output) * std
                return output
            return hook

        h = gate.register_forward_hook(make_hook(noise_std))
        hooks.append(h)
    model._jitter_hooks = hooks  # type: ignore[attr-defined]


def _remove_gate_jitter(model: MoETransformer) -> None:
    for h in getattr(model, "_jitter_hooks", []):
        h.remove()
    model._jitter_hooks = []


# ── Test class ────────────────────────────────────────────────────────────────

class TestSamePassProvenance:
    """
    Prove that router metrics originate from the training-pass router call,
    not a second probe call.
    """

    def _build_model_and_batch(self, gc: bool = True):
        cfg = _tiny_moe_config(gradient_checkpointing=gc)
        model = MoETransformer(cfg)
        model.train()
        B, T = 2, 8
        input_ids = torch.randint(0, cfg.base.vocab_size, (B, T))
        labels = torch.randint(0, cfg.base.vocab_size, (B, T))
        return model, cfg, input_ids, labels

    # T16 ─────────────────────────────────────────────────────────────────────

    def test_t16_router_called_exactly_once_per_layer_under_gc(self):
        """
        With gradient_checkpointing=True, TopKRouter.forward must be called
        exactly once per layer per forward pass.

        The old probe implementation called it twice: once inside the
        gradient_checkpoint boundary (for the training loss) and once outside
        (the no_grad probe).  The same-pass tensor transport must call it
        exactly once.

        Note: gradient_checkpoint with use_reentrant=False does NOT rerun the
        forward during the forward pass itself; recomputation happens during
        backward.  We only count calls during the forward pass here.
        """
        model, cfg, input_ids, labels = self._build_model_and_batch(gc=True)

        with _count_router_calls(model) as counts:
            out = model(input_ids, labels=labels)
            # Do NOT call backward here — we are only counting forward calls.

        num_layers = cfg.base.num_layers
        assert len(counts) == num_layers, (
            f"Expected {num_layers} layers in call_counts, got {len(counts)}"
        )
        for layer_idx, count in counts.items():
            assert count == 1, (
                f"Layer {layer_idx}: router called {count} times during forward "
                f"pass with gradient_checkpointing=True.  Expected exactly 1.  "
                f"A count of 2 indicates the old probe pattern is still active."
            )

    # T17 ─────────────────────────────────────────────────────────────────────

    def test_t17_metrics_match_training_pass_assignments_under_jitter(self):
        """
        With deliberate stochastic jitter on the router gate, a second router
        call would produce different expert_assignment_counts.  Prove that the
        metrics dict returned by the model matches the routing decisions made
        during the single (training-loss) router call.

        Method:
        1. Inject jitter hooks so each gate call adds large Gaussian noise.
        2. Intercept the router call to record the exact expert_indices it
           produced.
        3. Run the forward pass.
        4. Compare the expert_assignment_counts in the returned metrics dict
           against the counts derived from the intercepted expert_indices.
        5. They must match exactly.  If a second call were made, the jitter
           would produce different indices and the counts would differ.
        """
        model, cfg, input_ids, labels = self._build_model_and_batch(gc=True)
        _inject_gate_jitter(model, noise_std=2.0)  # large jitter → very different on second call

        # Intercept router calls to record expert_indices per layer
        recorded_indices: dict[int, torch.Tensor] = {}
        call_counts: dict[int, int] = {}
        original_forwards: dict[int, Any] = {}

        for layer_idx, layer in enumerate(model.layers):
            call_counts[layer_idx] = 0
            original_forward = layer.moe_ffn.router.forward

            def make_wrapper(idx, orig):
                def wrapper(x, return_metric_tensors=False):
                    result = orig(x, return_metric_tensors=return_metric_tensors)
                    call_counts[idx] += 1
                    if call_counts[idx] == 1:
                        # Record expert_indices from the FIRST (and only) call
                        # result[1] is expert_indices: (tokens, top_k)
                        recorded_indices[idx] = result[1].detach().cpu().clone()
                    return result
                return wrapper

            original_forwards[layer_idx] = original_forward
            layer.moe_ffn.router.forward = make_wrapper(layer_idx, original_forward)

        try:
            out = model(input_ids, labels=labels)
        finally:
            for layer_idx, layer in enumerate(model.layers):
                layer.moe_ffn.router.forward = original_forwards[layer_idx]
            _remove_gate_jitter(model)

        router_metrics = out["router_metrics"]
        assert len(router_metrics) == cfg.base.num_layers, (
            f"Expected {cfg.base.num_layers} metric dicts, got {len(router_metrics)}"
        )

        E = cfg.num_experts
        top_k = cfg.num_experts_per_token

        for layer_idx in range(cfg.base.num_layers):
            expert_indices = recorded_indices[layer_idx]  # (tokens, top_k)
            num_tokens = expert_indices.shape[0]

            # Compute expected assignment counts from the recorded indices
            one_hot = torch.zeros(num_tokens, top_k, E)
            for k in range(top_k):
                one_hot[:, k, :] = torch.nn.functional.one_hot(
                    expert_indices[:, k], num_classes=E
                ).float()
            expected_counts = one_hot.sum(dim=(0, 1)).long().tolist()

            # Get actual counts from the metrics dict
            from training.router_metrics import K_EXPERT_ASSIGNMENT_COUNTS
            actual_counts = router_metrics[layer_idx][K_EXPERT_ASSIGNMENT_COUNTS]

            assert actual_counts == [int(c) for c in expected_counts], (
                f"Layer {layer_idx}: metrics expert_assignment_counts {actual_counts} "
                f"do not match the routing decisions from the training-pass router call "
                f"{expected_counts}.  This indicates the metrics were derived from a "
                f"different (second) router call."
            )

        # Final sanity: router was called exactly once per layer
        for layer_idx, count in call_counts.items():
            assert count == 1, (
                f"Layer {layer_idx}: router called {count} times. Expected 1."
            )
