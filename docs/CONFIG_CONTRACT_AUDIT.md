# Jupiter Shot — Config Contract Audit

**Status:** Run 7 — STRICT LOADER ENFORCED  
**Loader:** `training/config_loader.py`  
**Generated:** 2026-08-03 (Run 7 repair)

---

## Summary

All laptop validation configs are now loaded through `training/config_loader.py`,
which enforces a strict contract:

1. **Unknown keys halt execution** — no silent discard.
2. **Aliases are explicit** — every alias is declared in the loader; no implicit mapping.
3. **Unsupported keys are listed** — attempting to use them produces a clear error.
4. **Resolved config is saved as JSON** — the artifact directory contains the exact
   config that was used, not the YAML source.

The root cause of all Run 1–6 silent failures was the `hasattr`-filtering pattern:

```python
# OLD (silently discards unknown keys — REMOVED):
cfg = DenseConfig(**{k: v for k, v in raw.items() if hasattr(DenseConfig, k)})
```

This is now replaced by:

```python
# NEW (halts on unknown keys):
cfg = load_dense_config(raw["model"])   # raises ConfigValidationError on unknown keys
```

---

## DenseConfig Field Contract

**Dataclass:** `training/models/dense.py::DenseConfig`  
**Loader function:** `training/config_loader.load_dense_config()`

| YAML Key | Status | Canonical Field | Notes |
|----------|--------|----------------|-------|
| `vocab_size` | ✅ Canonical | `vocab_size` | Must equal tokenizer vocab_size (50,277 for gpt-neox-20b) |
| `hidden_size` | ✅ Canonical | `hidden_size` | |
| `num_layers` | ✅ Canonical | `num_layers` | **Use this, not `num_hidden_layers`** |
| `num_hidden_layers` | ⚠️ Alias | `num_layers` | Supported for legacy compat; prefer `num_layers` |
| `num_attention_heads` | ✅ Canonical | `num_attention_heads` | |
| `intermediate_size` | ✅ Canonical | `intermediate_size` | |
| `max_position_embeddings` | ✅ Canonical | `max_position_embeddings` | |
| `rms_norm_eps` | ✅ Canonical | `rms_norm_eps` | Default: 1e-5 |
| `rope_theta` | ✅ Canonical | `rope_theta` | Default: 10000.0 |
| `tie_word_embeddings` | ✅ Canonical | `tie_word_embeddings` | Default: true |
| `gradient_checkpointing` | ✅ Canonical | `gradient_checkpointing` | Default: false |
| `attention_dropout` | ✅ Canonical | `attention_dropout` | Default: 0.0 |
| `hidden_dropout` | ✅ Canonical | `hidden_dropout` | Default: 0.0 |
| `num_hidden_layers` | ⚠️ Alias | `num_layers` | Accepted; mapped to `num_layers` |
| `n_layers` | ❌ Unsupported | — | Halts with ConfigValidationError |
| `depth` | ❌ Unsupported | — | Halts with ConfigValidationError |
| `d_model` | ❌ Unsupported | — | Use `hidden_size` |
| `n_heads` | ❌ Unsupported | — | Use `num_attention_heads` |
| `ffn_dim` | ❌ Unsupported | — | Use `intermediate_size` |
| `ffn_hidden_size` | ❌ Unsupported | — | Use `intermediate_size` |

### DenseConfig Derived Fields (computed, not settable in YAML)

| Field | Derivation |
|-------|-----------|
| `num_kv_heads` | Defaults to `num_attention_heads` (no GQA by default) |
| `head_dim` | `hidden_size // num_attention_heads` |

---

## MoEConfig Field Contract

**Dataclass:** `training/models/moe.py::MoEConfig`  
**Loader function:** `training/config_loader.load_moe_config()`

The MoE YAML has a nested structure:
```yaml
model:
  base:           # → DenseConfig (see above)
    vocab_size: ...
    num_layers: ...
    ...
  num_experts: 8  # → MoEConfig top-level fields
  ...
```

### MoEConfig Top-Level Fields

| YAML Key | Status | Canonical Field | Notes |
|----------|--------|----------------|-------|
| `num_experts` | ✅ Canonical | `num_experts` | |
| `num_experts_per_token` | ✅ Canonical | `num_experts_per_token` | |
| `top_k` | ⚠️ Alias | `num_experts_per_token` | Accepted; prefer `num_experts_per_token` |
| `expert_capacity_factor` | ✅ Canonical | `expert_capacity_factor` | Default: 1.25 |
| `capacity_factor` | ⚠️ Alias | `expert_capacity_factor` | Accepted; prefer `expert_capacity_factor` |
| `router_aux_loss_coeff` | ✅ Canonical | `router_aux_loss_coeff` | Default: 0.01 |
| `aux_loss_coeff` | ⚠️ Alias | `router_aux_loss_coeff` | Accepted; prefer `router_aux_loss_coeff` |
| `router_z_loss_coeff` | ✅ Canonical | `router_z_loss_coeff` | Default: 0.001 |
| `z_loss_coeff` | ⚠️ Alias | `router_z_loss_coeff` | Accepted; prefer `router_z_loss_coeff` |
| `normalize_router_probs` | ✅ Canonical | `normalize_router_probs` | Default: true |
| `use_shared_expert` | ✅ Canonical | `use_shared_expert` | Default: false |
| `expert_dropout` | ✅ Canonical | `expert_dropout` | Default: 0.0 |
| `moe_layer_freq` | ❌ **Unsupported** | — | **Not implemented.** MoETransformer uses ALL layers as MoE. Halts with ConfigValidationError. |
| `router_jitter` | ❌ **Unsupported** | — | **Not implemented** in TopKRouter. Halts with ConfigValidationError. |
| `expert_parallel` | ❌ Unsupported | — | Not implemented for laptop validation |
| `load_balancing_loss` | ❌ Unsupported | — | Use `router_aux_loss_coeff` |

---

## Router Metrics Schema

**Module:** `training/router_metrics.py`  
**Used by:** `TopKRouter.forward()`, `run_laptop_moe.py`, `test_acceptance_e2e.py`

All router metrics keys are defined as constants (`K_*`) in `training/router_metrics.py`.
The model emits a list of per-layer dicts, one per MoE layer.

### Required Keys (must be present for acceptance evaluation)

| Constant | Key Name | Type | Description |
|----------|----------|------|-------------|
| `K_NUM_EXPERTS` | `num_experts` | int | Total experts in this layer |
| `K_TOP_K` | `top_k` | int | Experts selected per token |
| `K_EXPERT_ASSIGNMENT_FRACTIONS` | `expert_assignment_fractions` | list[float] | Fraction of all assignments going to each expert. Balanced value = `top_k / num_experts` |
| `K_ROUTING_ENTROPY_NATS` | `routing_entropy_nats` | float | Mean routing entropy in nats. Max = ln(num_experts) |
| `K_EXPERT_UTILIZATION_RATE` | `expert_utilization_rate` | float | Fraction of experts that received ≥1 token |
| `K_LOAD_BALANCE_SCORE` | `load_balance_score` | float | 1 - max_deviation from uniform. Higher = more balanced |
| `K_AUX_LOSS` | `aux_loss` | float | Auxiliary load-balancing loss for this layer |
| `K_Z_LOSS` | `z_loss` | float | Router z-loss for this layer |
| `K_OVERFLOW_FRACTION` | `overflow_fraction` | float | Fraction of tokens dropped due to capacity overflow |
| `K_MEAN_ROUTING_WEIGHT` | `mean_routing_weight` | float | Mean softmax weight of selected experts |

### Acceptance Thresholds

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| `load_balance_score` | ≥ 0.50 | Higher is better |
| `expert_utilization_rate` | ≥ 0.50 | Higher is better |
| `routing_entropy_nats` | ≥ 0.25 × ln(num_experts) | Higher is better |
| `overflow_fraction` | ≤ 0.20 | Lower is better |

---

## Config Loader Error Reference

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `Unknown keys in DenseConfig: ['moe_layer_freq']` | Key not in DenseConfig dataclass | Remove the key from YAML |
| `Unknown keys in MoEConfig: ['router_jitter']` | Key not in MoEConfig dataclass | Remove the key; `router_jitter` is not implemented |
| `Unknown keys in MoEConfig: ['moe_layer_freq']` | Key not in MoEConfig dataclass | Remove the key; all layers are MoE |
| `Both alias 'num_hidden_layers' and canonical 'num_layers' present` | Duplicate specification | Use only `num_layers` |
| `Missing required field: vocab_size` | Required field absent from YAML | Add `vocab_size: 50277` |
| `TOKENIZER_VOCABULARY_MISMATCH` | Config `vocab_size` ≠ tokenizer `vocab_size` | Set `vocab_size: 50277` in config |

---

## Alias Deprecation Schedule

All aliases are supported in Run 7 for backward compatibility. They will be removed in Run 8.

| Alias | Canonical | Remove in |
|-------|-----------|-----------|
| `num_hidden_layers` | `num_layers` | Run 8 |
| `top_k` | `num_experts_per_token` | Run 8 |
| `capacity_factor` | `expert_capacity_factor` | Run 8 |
| `aux_loss_coeff` | `router_aux_loss_coeff` | Run 8 |
| `z_loss_coeff` | `router_z_loss_coeff` | Run 8 |

---

## Config Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `training/configs/laptop_dense_run7.yaml` | Dense laptop validation (Run 7) | ✅ Canonical field names |
| `training/configs/laptop_moe_run7.yaml` | MoE laptop validation (Run 7) | ✅ Canonical field names |
| `training/configs/laptop_moe_8expert_8gb_safe.yaml` | 8-expert 8 GB VRAM config | ✅ Updated Run 6 |
| `training/configs/laptop_dense_small.yaml` | Legacy dense config | ⚠️ Uses `num_hidden_layers` alias |
| `training/configs/laptop_moe_small.yaml` | Legacy MoE config | ❌ Contains `moe_layer_freq` (unsupported) — do not use |

---

*This document is generated from the Run 7 config audit. Update when adding new config fields.*
