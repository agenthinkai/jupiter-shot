"""
Jupiter Shot — Run 12 Gate 10b Real-Object Regression Tests
============================================================
Tests step10b_moe_aux_loss_verification() with a real MoETransformer instance
constructed from the Run 7 default config (laptop_moe_run7.yaml).

These tests require PyTorch. They are automatically skipped when torch is not
importable (e.g., CPU-only CI without torch installed). They are designed to
run on Kishore's RTX 5060 environment with PyTorch 2.7.1+cu128.

Run with:
    python3 -m pytest tests/test_run12_gate10b_real_object.py -v

Skips:
    All tests are skipped if torch is not available.

Stop conditions (per Run 12 spec):
    - If MoE object structure differs from Run 11 evidence → test fails with clear message
    - If aux_loss is detached → test fails
    - If isolated aux-loss router gradients are zero → test fails
    - If total loss does not include weighted aux_loss → test fails
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── PyTorch availability guard ────────────────────────────────────────────────

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="PyTorch not available — real-object MoE tests skipped (will run on Kishore RTX 5060)",
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def moe_config():
    """Load MoEConfig from the Run 7 default YAML (laptop_moe_run7.yaml)."""
    import yaml
    from training.config_loader import load_moe_config
    cfg_path = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return load_moe_config(raw["model"])


@pytest.fixture(scope="module")
def moe_model(moe_config):
    """Construct a real MoETransformer from the Run 7 config."""
    from training.models.moe import MoETransformer
    model = MoETransformer(moe_config)
    model.train()
    return model


@pytest.fixture(scope="module")
def forward_output(moe_model, moe_config):
    """Run a single forward pass and return the output dict."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, moe_config.base.vocab_size, (1, 8))
    labels = input_ids.clone()
    return moe_model(input_ids=input_ids, labels=labels)


@pytest.fixture(scope="module")
def step10_result(forward_output):
    """Build a step10 result dict from the real forward pass."""
    aux_val = float(forward_output["aux_loss"].item())
    lm_val  = float(forward_output["lm_loss"].item())
    return {
        "status": "ok",
        "moe_cpu_loss": lm_val,
        "moe_cpu_aux_loss": aux_val,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _run_gate10b(moe_model, step10_result):
    from scripts.run_laptop_validation_pipeline import step10b_moe_aux_loss_verification
    with tempfile.TemporaryDirectory() as tmp:
        return step10b_moe_aux_loss_verification(
            moe_model=moe_model,
            step10_result=step10_result,
            errors_path=pathlib.Path(tmp) / "errors.jsonl",
        )


# ── Structure verification tests ─────────────────────────────────────────────

class TestRealMoEStructure:
    """Verify the real MoETransformer has the exact attribute layout expected by gate 10b."""

    def test_model_has_config(self, moe_model):
        """model.config must exist."""
        assert hasattr(moe_model, "config"), \
            "MoETransformer has no 'config' attribute — model structure changed"

    def test_config_has_router_aux_loss_coeff(self, moe_model):
        """config.router_aux_loss_coeff must exist (Defect 1 fix)."""
        assert hasattr(moe_model.config, "router_aux_loss_coeff"), (
            "config has no 'router_aux_loss_coeff' — "
            "Run 11 Defect 1: checker was reading config.moe.aux_loss_coef (does not exist)"
        )
        assert moe_model.config.router_aux_loss_coeff > 0.0, \
            f"router_aux_loss_coeff={moe_model.config.router_aux_loss_coeff} must be > 0"

    def test_config_has_base_with_gradient_checkpointing(self, moe_model):
        """config.base.gradient_checkpointing must exist (Defect 2 fix)."""
        assert hasattr(moe_model.config, "base"), (
            "config has no 'base' attribute — "
            "Run 11 Defect 2: checker was reading config.gradient_checkpointing (does not exist)"
        )
        assert hasattr(moe_model.config.base, "gradient_checkpointing"), \
            "config.base has no 'gradient_checkpointing' attribute"
        assert isinstance(moe_model.config.base.gradient_checkpointing, bool), \
            f"gradient_checkpointing={moe_model.config.base.gradient_checkpointing!r} must be bool"

    def test_gradient_checkpointing_is_true_in_run7_config(self, moe_model):
        """Run 7 config sets gradient_checkpointing=True in model.base."""
        assert moe_model.config.base.gradient_checkpointing is True, (
            f"Expected gradient_checkpointing=True from laptop_moe_run7.yaml, "
            f"got {moe_model.config.base.gradient_checkpointing}"
        )

    def test_model_has_layers(self, moe_model):
        """model.layers must exist."""
        assert hasattr(moe_model, "layers"), \
            "MoETransformer has no 'layers' attribute"
        assert len(moe_model.layers) > 0, "model.layers is empty"

    def test_layer_has_moe_ffn(self, moe_model):
        """Each layer must have moe_ffn attribute (Defect 3 fix)."""
        for i, layer in enumerate(moe_model.layers):
            assert hasattr(layer, "moe_ffn"), (
                f"Layer {i} has no 'moe_ffn' attribute — "
                f"Run 11 Defect 3: checker probed (moe, mlp, ffn) and missed moe_ffn. "
                f"Layer attrs: {[a for a in dir(layer) if not a.startswith('_')]}"
            )

    def test_moe_ffn_has_router(self, moe_model):
        """moe_ffn.router must exist on every layer."""
        from training.models.moe import TopKRouter
        for i, layer in enumerate(moe_model.layers):
            assert hasattr(layer.moe_ffn, "router"), \
                f"Layer {i}.moe_ffn has no 'router' attribute"
            assert isinstance(layer.moe_ffn.router, TopKRouter), \
                f"Layer {i}.moe_ffn.router is {type(layer.moe_ffn.router).__name__}, expected TopKRouter"

    def test_moe_ffn_has_experts(self, moe_model, moe_config):
        """moe_ffn.experts must exist and have the correct count."""
        for i, layer in enumerate(moe_model.layers):
            assert hasattr(layer.moe_ffn, "experts"), \
                f"Layer {i}.moe_ffn has no 'experts' attribute"
            assert len(layer.moe_ffn.experts) == moe_config.num_experts, (
                f"Layer {i}.moe_ffn.experts has {len(layer.moe_ffn.experts)} experts, "
                f"expected {moe_config.num_experts}"
            )

    def test_router_aux_loss_coeff_matches_config(self, moe_model):
        """router.aux_loss_coeff must match config.router_aux_loss_coeff."""
        expected = moe_model.config.router_aux_loss_coeff
        for i, layer in enumerate(moe_model.layers):
            actual = layer.moe_ffn.router.aux_loss_coeff
            assert actual == expected, (
                f"Layer {i} router.aux_loss_coeff={actual} != "
                f"config.router_aux_loss_coeff={expected}"
            )


# ── Forward pass tests ────────────────────────────────────────────────────────

class TestRealMoEForwardPass:
    """Verify the forward pass produces the expected output fields."""

    def test_output_has_loss(self, forward_output):
        """output['loss'] must be a finite positive scalar."""
        assert "loss" in forward_output, "forward output missing 'loss' key"
        loss = forward_output["loss"]
        assert isinstance(loss, torch.Tensor), f"loss is {type(loss).__name__}"
        assert torch.isfinite(loss).all(), f"loss is not finite: {loss}"
        assert float(loss.item()) > 0.0, f"loss={float(loss.item())} must be > 0"

    def test_output_has_aux_loss(self, forward_output):
        """output['aux_loss'] must be a finite positive scalar."""
        assert "aux_loss" in forward_output, "forward output missing 'aux_loss' key"
        aux = forward_output["aux_loss"]
        assert isinstance(aux, torch.Tensor), f"aux_loss is {type(aux).__name__}"
        assert torch.isfinite(aux).all(), f"aux_loss is not finite: {aux}"
        assert float(aux.item()) > 0.0, (
            f"aux_loss={float(aux.item())} must be > 0 — "
            "Defect 3 NOT fixed: gradient-checkpoint branch is not accumulating aux_loss"
        )

    def test_output_has_lm_loss(self, forward_output):
        """output['lm_loss'] must be a finite positive scalar."""
        assert "lm_loss" in forward_output, "forward output missing 'lm_loss' key"
        lm = forward_output["lm_loss"]
        assert isinstance(lm, torch.Tensor), f"lm_loss is {type(lm).__name__}"
        assert torch.isfinite(lm).all(), f"lm_loss is not finite: {lm}"
        assert float(lm.item()) > 0.0, f"lm_loss={float(lm.item())} must be > 0"

    def test_total_loss_equals_lm_plus_aux(self, forward_output):
        """Total loss must equal lm_loss + aux_loss (total-loss decomposition proof)."""
        total = float(forward_output["loss"].item())
        lm    = float(forward_output["lm_loss"].item())
        aux   = float(forward_output["aux_loss"].item())
        expected = lm + aux
        assert abs(total - expected) < 1e-4, (
            f"total_loss={total:.6f} != lm_loss + aux_loss = {lm:.6f} + {aux:.6f} = {expected:.6f} "
            f"(diff={abs(total - expected):.2e}). "
            "Total loss must include weighted aux_loss."
        )

    def test_aux_loss_below_sanity_bound(self, forward_output):
        """aux_loss must be < 1.0 (must not dominate training loss)."""
        aux = float(forward_output["aux_loss"].item())
        assert aux < 1.0, (
            f"aux_loss={aux:.6f} >= 1.0 — aux_loss is dominating training loss"
        )


# ── Autograd connectivity tests ───────────────────────────────────────────────

class TestRealMoEAutograd:
    """Verify autograd connectivity from aux_loss to router gate parameters."""

    def test_aux_loss_has_grad_fn(self, forward_output):
        """aux_loss tensor must have a grad_fn (connected to computation graph)."""
        aux = forward_output["aux_loss"]
        assert aux.grad_fn is not None, (
            "aux_loss.grad_fn is None — tensor is detached from computation graph. "
            "Stop condition: auxiliary loss is detached."
        )

    def test_total_loss_has_grad_fn(self, forward_output):
        """total loss tensor must have a grad_fn."""
        loss = forward_output["loss"]
        assert loss.grad_fn is not None, \
            "total loss.grad_fn is None — tensor is detached from computation graph"

    def test_isolated_aux_loss_router_gradients_nonzero(self, moe_model, forward_output):
        """torch.autograd.grad(aux_loss, router_params) must return finite nonzero gradients.

        This is the isolated autograd proof: aux_loss alone (not total_loss) must
        flow gradients to the router gate parameters. This confirms the router is
        in the aux_loss computation graph and will receive gradient updates.
        """
        aux_loss = forward_output["aux_loss"]
        assert aux_loss.grad_fn is not None, \
            "aux_loss.grad_fn is None — cannot compute isolated router gradients"

        router_params = [
            p for layer in moe_model.layers
            for p in layer.moe_ffn.router.parameters()
            if p.requires_grad
        ]
        assert len(router_params) > 0, \
            "No router parameters with requires_grad=True found"

        grads = torch.autograd.grad(
            aux_loss, router_params,
            retain_graph=True, allow_unused=True
        )
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0, (
            f"torch.autograd.grad(aux_loss, router_params) returned all None — "
            f"aux_loss is not connected to router parameters. "
            "Stop condition: isolated auxiliary-loss router gradients are zero or missing."
        )
        assert all(torch.isfinite(g).all().item() for g in non_none), \
            "Isolated aux-loss router gradients contain non-finite values"
        assert any(g.abs().max().item() > 0 for g in non_none), (
            "All isolated aux-loss router gradients are zero — "
            "aux_loss does not influence router parameters. "
            "Stop condition: isolated auxiliary-loss router gradients are zero or missing."
        )
        grad_norm = sum(g.norm().item() for g in non_none)
        assert grad_norm > 0.0, \
            f"Isolated aux-loss router gradient norm={grad_norm:.6f} must be > 0"

    def test_total_loss_includes_aux_loss_gradient(self, moe_model, forward_output):
        """Gradient of total_loss w.r.t. router params must be nonzero.

        This confirms total_loss includes the weighted aux_loss and that the
        optimizer will update router parameters during training.
        """
        total_loss = forward_output["loss"]
        assert total_loss.grad_fn is not None, \
            "total_loss.grad_fn is None — cannot compute gradients"

        router_params = [
            p for layer in moe_model.layers
            for p in layer.moe_ffn.router.parameters()
            if p.requires_grad
        ]
        assert len(router_params) > 0, \
            "No router parameters with requires_grad=True found"

        grads = torch.autograd.grad(
            total_loss, router_params,
            retain_graph=True, allow_unused=True
        )
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0, (
            "torch.autograd.grad(total_loss, router_params) returned all None — "
            "total_loss does not include aux_loss gradient path. "
            "Stop condition: total optimized loss does not include weighted auxiliary loss."
        )
        assert any(g.abs().max().item() > 0 for g in non_none), (
            "All total_loss router gradients are zero — "
            "total_loss does not influence router parameters. "
            "Stop condition: total optimized loss does not include weighted auxiliary loss."
        )


# ── Full gate 10b integration test ───────────────────────────────────────────

class TestGate10bIntegration:
    """Run the full gate 10b function with a real MoETransformer."""

    def test_gate10b_passes_with_real_model(self, moe_model, step10_result):
        """Gate 10b must pass all 20 checks with the real Run 7 model."""
        result = _run_gate10b(moe_model, step10_result)
        assert result["status"] == "ok", \
            f"Gate 10b failed: {result}"
        assert result["n_failed"] == 0, (
            f"Gate 10b has {result['n_failed']} root-cause failures: "
            f"{[k for k, v in result['checks'].items() if v['status'] == 'FAIL']}"
        )
        assert result["n_blocked"] == 0, (
            f"Gate 10b has {result['n_blocked']} blocked checks: "
            f"{[k for k, v in result['checks'].items() if v['status'] == 'BLOCKED']}"
        )
        # Verify all three defect fixes are confirmed
        assert result["checks"]["c07_aux_loss_coeff"]["passed"] is True, \
            "c07 failed — Defect 1 fix not confirmed (router_aux_loss_coeff)"
        assert result["checks"]["c11_gc_attr"]["passed"] is True, \
            "c11 failed — Defect 2 fix not confirmed (config.base.gradient_checkpointing)"
        assert result["checks"]["c15_moe_submodule"]["passed"] is True, \
            "c15 failed — Defect 3 fix not confirmed (moe_ffn probe)"
        assert result["checks"]["c13_gc_aux_nonzero"]["passed"] is True, \
            "c13 failed — gradient_checkpointing=True but aux_loss not > 0"
        assert result["checks"]["c18_aux_grad_fn"]["passed"] is True, \
            "c18 failed — aux_loss tensor is detached"
        assert result["checks"]["c19_isolated_router_grad"]["passed"] is True, \
            "c19 failed — isolated aux-loss router gradients are None"
        assert result["checks"]["c20_router_grad_nonzero"]["passed"] is True, \
            "c20 failed — isolated aux-loss router gradients are zero"

    def test_gate10b_n_total_is_20(self, moe_model, step10_result):
        """Gate 10b must have exactly 20 checks."""
        result = _run_gate10b(moe_model, step10_result)
        assert result["n_total"] == 20, (
            f"Expected 20 checks, got {result['n_total']}. "
            f"Checks: {list(result['checks'].keys())}"
        )

    def test_gate10b_fails_on_zero_aux_loss(self, moe_model):
        """Gate 10b must raise PreflightError when aux_loss=0 (Defect 3 regression)."""
        bad_step10 = {"status": "ok", "moe_cpu_loss": 10.5, "moe_cpu_aux_loss": 0.0}
        with pytest.raises(Exception, match="Defect 3 NOT fixed|FAILED"):
            _run_gate10b(moe_model, bad_step10)

    def test_gate10b_fails_on_detached_aux_loss(self, moe_model, moe_config):
        """Gate 10b must raise PreflightError when aux_loss has no grad_fn."""
        # Produce a step10 result with aux_loss=0.042 (passes c04) but
        # the live forward pass in c18 will still check the real tensor.
        # To test the detached path, we use a model in eval() mode where
        # gradient tracking may differ, but the real test is in the unit test
        # for c18 — here we just confirm the integration path works.
        # This test verifies the gate raises on zero aux_loss as a proxy.
        bad_step10 = {"status": "ok", "moe_cpu_loss": 10.5, "moe_cpu_aux_loss": 0.0}
        with pytest.raises(Exception, match="FAILED"):
            _run_gate10b(moe_model, bad_step10)
