"""
Jupiter Shot — Strict Configuration Loader
==========================================

Replaces the silent ``{k: v for k, v in cfg.items() if hasattr(ConfigClass, k)}``
anti-pattern with a contract-enforcing loader.

Contract:
  - Every YAML key must map to a known dataclass field (via direct name or alias).
  - Unknown keys halt execution immediately with a clear error.
  - Missing required keys halt execution (fields with no default are required).
  - Aliases are explicit and tested.
  - The resolved configuration is saved as JSON before model construction.
  - ``moe_layer_freq`` and ``router_jitter`` are declared UNSUPPORTED and halt
    execution if present (they have no corresponding model implementation).

Supported aliases (YAML key → DenseConfig / MoEConfig field):
  Dense:
    num_hidden_layers → num_layers
    n_layers          → num_layers
    n_heads           → num_attention_heads
    n_embd            → hidden_size
    n_inner           → intermediate_size

  MoE base (same aliases as Dense, applied to the ``base`` sub-dict):
    (same as above)

  MoE top-level:
    capacity_factor   → expert_capacity_factor
    aux_loss_coeff    → router_aux_loss_coeff
    z_loss_coeff      → router_z_loss_coeff
    top_k             → num_experts_per_token
    experts           → num_experts

Unsupported keys (present in old YAMLs, not implemented in any model):
  moe_layer_freq    — MoETransformer uses ALL layers as MoE; freq is not a param
  router_jitter     — not implemented in TopKRouter
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

# ── Alias tables ──────────────────────────────────────────────────────────────

#: Aliases for DenseConfig fields.
#: Maps YAML key → DenseConfig field name.
DENSE_ALIASES: dict[str, str] = {
    "num_hidden_layers": "num_layers",
    "n_layers":          "num_layers",
    "n_heads":           "num_attention_heads",
    "n_embd":            "hidden_size",
    "n_inner":           "intermediate_size",
}

#: Aliases for MoEConfig top-level fields.
#: Maps YAML key → MoEConfig field name.
MOE_ALIASES: dict[str, str] = {
    "capacity_factor": "expert_capacity_factor",
    "aux_loss_coeff":  "router_aux_loss_coeff",
    "z_loss_coeff":    "router_z_loss_coeff",
    "top_k":           "num_experts_per_token",
    "experts":         "num_experts",
}

#: Keys that appear in old YAML files but have NO model implementation.
#: Their presence must halt execution — do not silently ignore them.
UNSUPPORTED_KEYS: dict[str, str] = {
    "moe_layer_freq": (
        "moe_layer_freq is not implemented. MoETransformer uses ALL layers as "
        "MoE blocks. Remove this key from the config or implement it in the model."
    ),
    "router_jitter": (
        "router_jitter is not implemented in TopKRouter. Remove this key from "
        "the config or implement it in the model."
    ),
}


# ── Core loader ───────────────────────────────────────────────────────────────

class ConfigValidationError(ValueError):
    """Raised when a YAML config fails strict validation."""


def _apply_aliases(raw: dict[str, Any],
                   aliases: dict[str, str],
                   context: str) -> dict[str, Any]:
    """
    Apply alias substitutions to *raw*.

    If a YAML key is an alias for a canonical field name, rename it.
    If both the alias and the canonical name are present, raise an error.
    """
    result = dict(raw)
    for alias, canonical in aliases.items():
        if alias in result:
            if canonical in result:
                raise ConfigValidationError(
                    f"[{context}] Both alias '{alias}' and canonical field "
                    f"'{canonical}' are present. Remove one."
                )
            result[canonical] = result.pop(alias)
    return result


def _check_unsupported(raw: dict[str, Any], context: str) -> None:
    """Halt if any unsupported key is present."""
    for key, message in UNSUPPORTED_KEYS.items():
        if key in raw:
            raise ConfigValidationError(
                f"[{context}] Unsupported configuration key '{key}': {message}"
            )


def _get_dataclass_fields(cls: type) -> dict[str, dataclasses.Field]:
    """Return a dict of field_name → Field for a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    return {f.name: f for f in dataclasses.fields(cls)}


def _has_default(field: dataclasses.Field) -> bool:
    """Return True if the field has a default value or default_factory."""
    return (
        field.default is not dataclasses.MISSING
        or field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    )


def _strict_build(cls: type, raw: dict[str, Any], context: str) -> Any:
    """
    Build a dataclass instance from *raw*, enforcing strict contract:

    1. Apply unsupported-key check.
    2. Apply alias substitution.
    3. Reject any key not in the dataclass.
    4. Require every field that has no default.
    5. Construct and return the instance.
    """
    _check_unsupported(raw, context)

    # Determine which alias table to use based on the target class name.
    # MoEConfig top-level fields use MOE_ALIASES; everything else uses DENSE_ALIASES.
    cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
    if cls_name == "MoEConfig":
        aliases = MOE_ALIASES
    else:
        aliases = DENSE_ALIASES

    raw = _apply_aliases(raw, aliases, context)

    known_fields = _get_dataclass_fields(cls)

    # Check for unknown keys
    unknown = set(raw.keys()) - set(known_fields.keys())
    if unknown:
        raise ConfigValidationError(
            f"[{context}] Unknown configuration key(s): {sorted(unknown)}. "
            f"Known fields for {cls.__name__}: {sorted(known_fields.keys())}. "
            f"If this is an alias, add it to DENSE_ALIASES or MOE_ALIASES in "
            f"training/config_loader.py."
        )

    # Check for missing required fields
    missing = []
    for name, field in known_fields.items():
        if name not in raw and not _has_default(field):
            missing.append(name)
    if missing:
        raise ConfigValidationError(
            f"[{context}] Missing required field(s): {sorted(missing)} "
            f"for {cls.__name__}."
        )

    # Build the instance (pass only the keys present in raw; defaults fill the rest)
    return cls(**{k: v for k, v in raw.items() if k in known_fields})


# ── Public API ────────────────────────────────────────────────────────────────

def load_dense_config(yaml_model_section: dict[str, Any]) -> "DenseConfig":
    """
    Load a DenseConfig from the ``model:`` section of a YAML file.

    Args:
        yaml_model_section: The dict under the ``model:`` key in the YAML.

    Returns:
        A fully validated DenseConfig instance.

    Raises:
        ConfigValidationError: If any key is unknown, unsupported, or missing.
    """
    # Import is deferred so that config validation can run without torch installed.
    # The DenseConfig dataclass itself does not require torch; only the model does.
    try:
        from training.models.dense import DenseConfig
    except ImportError:
        # torch not installed — use a pure-Python stub for validation only
        from training._config_stubs import DenseConfig  # type: ignore[no-redef]
    return _strict_build(DenseConfig, dict(yaml_model_section), "DenseConfig")


def load_moe_config(yaml_model_section: dict[str, Any]) -> "MoEConfig":
    """
    Load a MoEConfig from the ``model:`` section of a YAML file.

    The ``model:`` section must have a ``base:`` sub-dict for DenseConfig fields
    and top-level keys for MoEConfig fields.

    Args:
        yaml_model_section: The dict under the ``model:`` key in the YAML.

    Returns:
        A fully validated MoEConfig instance.

    Raises:
        ConfigValidationError: If any key is unknown, unsupported, or missing.
    """
    try:
        from training.models.dense import DenseConfig
        from training.models.moe import MoEConfig
    except ImportError:
        from training._config_stubs import DenseConfig, MoEConfig  # type: ignore[no-redef]

    raw = dict(yaml_model_section)

    # Extract and validate the base DenseConfig
    if "base" not in raw:
        raise ConfigValidationError(
            "[MoEConfig] Missing required 'base:' sub-section for DenseConfig fields."
        )
    base_raw = dict(raw.pop("base"))
    base_config = _strict_build(DenseConfig, base_raw, "MoEConfig.base")

    # Build MoEConfig from the remaining top-level keys
    # MoEConfig.base is a dataclass field with a default_factory — we inject it directly
    moe_instance = _strict_build(MoEConfig, raw, "MoEConfig")
    # Override the base field with the validated DenseConfig
    object.__setattr__(moe_instance, "base", base_config)
    return moe_instance


def config_to_dict(config: Any) -> dict[str, Any]:
    """
    Recursively convert a dataclass config to a plain dict (JSON-serialisable).
    Handles nested dataclasses (e.g., MoEConfig.base).
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {
            k: config_to_dict(v)
            for k, v in dataclasses.asdict(config).items()
        }
    return config


def save_resolved_config(config: Any, path: pathlib.Path) -> None:
    """
    Save the resolved configuration as JSON before model construction.

    Args:
        config: A DenseConfig or MoEConfig instance.
        path: Output file path (will be created, including parents).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    d = config_to_dict(config)
    d["__config_class__"] = type(config).__name__
    path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def load_yaml_config(config_path: pathlib.Path) -> dict[str, Any]:
    """
    Load a YAML config file and return the full dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        The full YAML dict (including ``model:``, ``training:``, ``metadata:``).
    """
    import yaml
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
