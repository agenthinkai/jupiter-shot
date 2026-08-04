# Runner Interface Manifest
## Jupiter Shot — Laptop Validation Runners

Version: Run 13  
Branch: `fix/rtx50-blackwell-validation`  
Last updated: 2026-08-04

---

## Purpose

This document is the authoritative contract for all three laptop validation
runners.  The pipeline (`run_laptop_validation_pipeline.py`) invokes each
runner as a subprocess and validates the structured artifact it writes.  Any
change to a runner's CLI arguments, exit codes, artifact schema, or config
resolution behaviour must be reflected here and covered by a test.

---

## Shared Config Resolution Contract

All three runners use `training/config_path.resolve_config_path()`.

### Supported `--config` input forms

| Form | Example | Rule |
|------|---------|------|
| Absolute path with `.yaml` or `.yml` | `/home/user/jupiter-shot/training/configs/laptop_dense_run7.yaml` | A |
| Repository-relative path with `.yaml` or `.yml` | `training/configs/laptop_dense_run7.yaml` | B |
| Bare config name (no suffix, no path separators) | `laptop_dense_run7` | C |
| Repository-relative path without suffix | `training/configs/laptop_dense_run7` | D |

**Guarantee:** `.yaml` is never appended twice.  
**Guarantee:** `.yml` is never changed to `.yml.yaml`.  
**Guarantee:** A missing file raises `FileNotFoundError` with an
`EXECUTION_ERROR` diagnostic block listing valid alternatives.

### Run 12 root cause (for reference)

The pipeline passed `str(args.dense_config)` = `training/configs/laptop_dense_run7.yaml`
(a relative path with `.yaml` suffix — Rule B).  The runners constructed
`REPO_ROOT / "training" / "configs" / f"{config_name}.yaml"` which produced
`training/configs/training/configs/laptop_dense_run7.yaml.yaml`.  The shared
resolver prevents this by detecting the suffix before appending.

---

## Runner 1: `scripts/run_laptop_dense.py`

### Purpose
Dense transformer CUDA validation.

### CLI arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--config` | `str` | `laptop_dense_small` | No | Config name or path (see shared resolver) |
| `--steps` | `int` | `100` | No | Training steps |
| `--confirmed` | flag | `False` | No | Required for >100 steps |
| `--data-mode` | `str` | `auto` | No | `real`, `synthetic`, or `auto` |
| `--synthetic` | flag | `False` | No | Alias for `--data-mode synthetic` (deprecated) |
| `--thermal-warn` | `int` | `80` | No | GPU temperature warning threshold (°C) |
| `--thermal-stop` | `int` | `90` | No | GPU temperature stop threshold (°C) |
| `--output-dir` | `str` | `benchmarks/results/laptop` | No | Artifact output directory |
| `--run-id` | `str` | auto-generated | No | Run identifier for artifact traceability |

### Exit codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `EXIT_PASS` | All acceptance criteria met, real-text data used |
| 1 | `EXIT_NOT_ACCEPTED` | Training completed but acceptance criteria failed |
| 2 | `EXIT_NOT_EVALUABLE` | Training completed but required metrics absent |
| 3 | `EXIT_EXECUTION_ERROR` | Unrecoverable runtime error |
| 4 | `EXIT_SAFETY_STOP` | Thermal or safety threshold triggered |

### Artifact: `dense_summary.json`

Required fields: `schema_version`, `outcome`, `exit_code`, `run_id`,
`timestamp`, `config`, `data_mode`, `total_params`, `trainable_params`,
`precision`, `steps_completed`.

---

## Runner 2: `scripts/run_laptop_moe.py`

### Purpose
MoE transformer CUDA validation (8 experts, top-2 routing).

### CLI arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--config` | `str` | `laptop_moe_8gb_safe` | No | Config name or path (see shared resolver) |
| `--steps` | `int` | `100` | No | Training steps |
| `--confirmed` | flag | `False` | No | Required for >100 steps |
| `--data-mode` | `str` | `auto` | No | `real`, `synthetic`, or `auto` |
| `--synthetic` | flag | `False` | No | Alias for `--data-mode synthetic` (deprecated) |
| `--thermal-warn` | `int` | `80` | No | GPU temperature warning threshold (°C) |
| `--thermal-stop` | `int` | `90` | No | GPU temperature stop threshold (°C) |
| `--output-dir` | `str` | `benchmarks/results/laptop` | No | Artifact output directory |
| `--run-id` | `str` | auto-generated | No | Run identifier for artifact traceability |

### Exit codes

Same as Runner 1 (same `EXIT_*` constants).

### Artifact: `moe_summary.json`

Required fields: `schema_version`, `outcome`, `exit_code`, `run_id`,
`timestamp`, `config`, `data_mode`, `total_params`, `active_params_per_token`,
`num_experts`, `top_k`, `precision`, `steps_completed`, `aux_loss_semantics`.

**`aux_loss_semantics`** must be `"WEIGHTED"`.  The `aux_loss` field in the
artifact is the weighted total auxiliary loss
(`aux_loss_coeff × load_balance + z_loss_coeff × z_loss`, summed across all
layers).  It is **not** the raw unscaled value.

---

## Runner 3: `scripts/run_laptop_resume_test.py`

### Purpose
Dense checkpoint-resume correctness test (CPU or CUDA).

### CLI arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--config` | `str` | `laptop_dense_small` | No | Config name or path (see shared resolver) |
| `--steps` | `int` | `20` | No | Initial steps before checkpoint |
| `--output-dir` | `str` | `benchmarks/results/laptop` | No | Artifact output directory |
| `--run-id` | `str` | auto-generated | No | Run identifier for artifact traceability |
| `--data-mode` | `str` | `synthetic` | No | Accepted for CLI contract compatibility; test always uses deterministic synthetic tensors |

### Exit codes

Same as Runner 1 (same `EXIT_*` constants).

### Artifact: `resume_test.json`

Required fields: `schema_version`, `outcome`, `exit_code`, `run_id`,
`timestamp`, `config`, `data_mode_arg`, `resume_test_data_source`.

**`data_mode_arg`**: the raw value of `--data-mode` as passed by the caller.  
**`resume_test_data_source`**: must be `"deterministic_synthetic_torch_randint"`.
This field makes explicit that the resume test always uses deterministic
synthetic data regardless of `--data-mode`.

---

## Pipeline invocation contract

The pipeline (`run_laptop_validation_pipeline.py`) invokes runners as:

```
sys.executable scripts/run_laptop_{dense,moe,resume_test}.py
    --config <full_path_with_yaml_suffix>
    --data-mode <real|synthetic|auto>
    --steps <n>
    --run-id <run_id>
    --output-dir <run_dir>
```

The `--config` value is always a full absolute path with `.yaml` suffix
(Rule A or B in the shared resolver).  Runners must not append `.yaml` to it.

---

## Invariants

1. No runner may construct a config path with `f"{config_name}.yaml"` directly.
   All path construction must go through `resolve_config_path()`.
2. A missing config file must produce exit code 3 (`EXECUTION_ERROR`) and
   print the full `format_missing_error()` diagnostic block.
3. Artifact `run_id` must match the `--run-id` argument passed by the pipeline.
4. `aux_loss_semantics: WEIGHTED` must appear in `moe_summary.json`.
5. `resume_test_data_source: deterministic_synthetic_torch_randint` must appear
   in `resume_test.json`.
6. CUDA device evidence: `moe_summary.json` must include a `device` field.
   Gate 10b checks `c18`–`c20` are recorded as `SKIPPED` (not `PASS`) when
   the model is not a real `nn.Module` or when torch is unavailable.
