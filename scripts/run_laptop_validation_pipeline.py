"""
Jupiter Shot — Laptop Validation Pipeline (Run 7+)
===================================================

This is the AUTHORITATIVE orchestrator for all laptop validation runs.
The Windows batch file (run_all_laptop_validation.bat) is a thin launcher
that activates the .venv and calls this script.

Usage
-----
    python scripts/run_laptop_validation_pipeline.py [OPTIONS]

Options
-------
    --data-mode {real,synthetic,auto}
        real:      Load Wikitext-2 via HuggingFace datasets. Halt with
                   REAL_TEXT_DATA_UNAVAILABLE if the dataset cannot be loaded.
        synthetic: Use randomly generated token IDs. Run is marked NOT_ACCEPTED
                   in the final verdict (synthetic data does not validate the
                   real-text pipeline).
        auto:      Try real first; fall back to synthetic if unavailable.
        Default: real

    --preflight-only
        Run only the 14-step preflight gate. Do not start the 100-step GPU
        workloads. Exit 0 if all preflight steps pass, non-zero otherwise.
        Kishore should use this for the initial Run 7 review.

    --dense-config PATH
        Path to the dense YAML config file.
        Default: training/configs/laptop_dense_run7.yaml

    --moe-config PATH
        Path to the MoE YAML config file.
        Default: training/configs/laptop_moe_run7.yaml

    --run-id ID
        Override the auto-generated UTC run ID (YYYYMMDD_HHMMSS_UTC).
        Useful for re-running a specific artifact directory.

    --steps N
        Number of training steps for the GPU workloads.
        Default: 100

Exit codes
----------
    0  PASS             — all preflight + GPU workloads passed acceptance
    1  NOT_ACCEPTED     — ran successfully but did not meet acceptance criteria
    2  NOT_EVALUABLE    — preflight passed but GPU run produced no usable metrics
    3  EXECUTION_ERROR  — unhandled exception or dependency failure
    4  SAFETY_STOP      — a safety check (VRAM, vocab mismatch, etc.) halted the run

Artifact directory
------------------
    benchmarks/results/laptop/runs/<RUN_ID>/
        run_manifest.json
        preflight.json
        resolved_dense_config.json
        resolved_moe_config.json
        architecture_manifest.json
        dense_metrics.jsonl
        moe_metrics.jsonl
        router_metrics.jsonl
        resume_result.json
        errors.jsonl
        validation_report.md
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import traceback
from typing import Any

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Exit codes ────────────────────────────────────────────────────────────────
EXIT_PASS            = 0
EXIT_NOT_ACCEPTED    = 1
EXIT_NOT_EVALUABLE   = 2
EXIT_EXECUTION_ERROR = 3
EXIT_SAFETY_STOP     = 4

# ── Default config paths ──────────────────────────────────────────────────────
DEFAULT_DENSE_CONFIG = REPO_ROOT / "training" / "configs" / "laptop_dense_run7.yaml"
DEFAULT_MOE_CONFIG   = REPO_ROOT / "training" / "configs" / "laptop_moe_run7.yaml"

# ── VRAM safety limit ─────────────────────────────────────────────────────────
VRAM_TARGET_GB = 6.5


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def _git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        branch, commit = "unknown", "unknown"
    return {"branch": branch, "commit": commit}


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def _write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _append_error(errors_path: pathlib.Path, step: str, exc: Exception) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "error": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    with open(errors_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _count_params(model: Any) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_moe_active_params(model: Any, moe_cfg: Any) -> int:
    """
    Estimate active parameters per token for a MoETransformer.
    Active = embedding + (attn + router + top_k × expert_ffn + norms) × layers + final_norm.
    """
    import torch
    total, _ = _count_params(model)
    base = moe_cfg.base
    top_k = moe_cfg.num_experts_per_token
    num_experts = moe_cfg.num_experts
    # Expert FFN params per expert
    expert_ffn_params = (
        base.hidden_size * base.intermediate_size * 2
        + base.intermediate_size * base.hidden_size
    )
    # Total expert params per layer
    all_experts_params = expert_ffn_params * num_experts
    # Active expert params per layer (top_k experts)
    active_experts_params = expert_ffn_params * top_k
    # Difference per layer
    inactive_per_layer = all_experts_params - active_experts_params
    # Total inactive across all layers
    total_inactive = inactive_per_layer * base.num_layers
    return total - total_inactive


# ══════════════════════════════════════════════════════════════════════════════
# Preflight steps (14 steps)
# ══════════════════════════════════════════════════════════════════════════════

class PreflightError(RuntimeError):
    """Raised when a preflight step fails and the run must be halted."""


def step01_dependency_imports(errors_path: pathlib.Path) -> dict:
    """Step 1: Verify all required imports succeed."""
    print("[Preflight 01/14] Dependency imports ...", flush=True)
    results = {}
    required = ["torch", "transformers", "datasets", "pandas", "pyarrow", "yaml"]
    for mod in required:
        try:
            __import__(mod)
            results[mod] = "ok"
        except ImportError as e:
            results[mod] = f"MISSING: {e}"
    failed = [k for k, v in results.items() if v != "ok"]
    if failed:
        raise PreflightError(
            f"Step 1 FAILED — missing imports: {failed}. "
            f"Run: .venv\\Scripts\\python.exe -m pip install -r requirements-laptop.txt"
        )
    import torch
    results["torch_version"] = torch.__version__
    print(f"  torch={torch.__version__}  [OK]", flush=True)
    return results


def step02_pip_check(errors_path: pathlib.Path) -> dict:
    """Step 2: Run pip check to verify dependency consistency."""
    print("[Preflight 02/14] pip check ...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise PreflightError(
            f"Step 2 FAILED — pip check reported dependency conflicts:\n{result.stdout}\n{result.stderr}"
        )
    print("  pip check: OK", flush=True)
    return {"pip_check": "ok", "output": result.stdout.strip()}


def step03_real_text_dataset(data_mode: str, errors_path: pathlib.Path) -> dict:
    """Step 3: Load real-text dataset (or skip for synthetic mode)."""
    print(f"[Preflight 03/14] Dataset load (data_mode={data_mode}) ...", flush=True)
    if data_mode == "synthetic":
        print("  data_mode=synthetic: skipping dataset load", flush=True)
        return {"data_mode": "synthetic", "dataset_loaded": False}
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        num_rows = len(ds)
        print(f"  Wikitext-2 loaded: {num_rows:,} rows  [OK]", flush=True)
        return {
            "data_mode": data_mode,
            "dataset_loaded": True,
            "dataset_id": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "dataset_split": "train",
            "num_rows": num_rows,
        }
    except Exception as e:
        if data_mode == "real":
            raise PreflightError(
                f"REAL_TEXT_DATA_UNAVAILABLE: {e}. "
                f"Use --data-mode synthetic to run without real text data."
            ) from e
        # auto mode: fall back to synthetic
        print(f"  WARNING: dataset load failed ({e}); falling back to synthetic", flush=True)
        return {"data_mode": "synthetic", "dataset_loaded": False, "fallback_reason": str(e)}


def step04_tokenizer_load(data_mode: str, errors_path: pathlib.Path) -> dict:
    """Step 4: Load tokenizer and verify vocab size."""
    print("[Preflight 04/14] Tokenizer load ...", flush=True)
    if data_mode == "synthetic":
        print("  data_mode=synthetic: skipping tokenizer load", flush=True)
        return {"data_mode": "synthetic", "tokenizer_loaded": False}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        vocab_size = tok.vocab_size
        print(f"  Tokenizer: EleutherAI/gpt-neox-20b, vocab_size={vocab_size}  [OK]", flush=True)
        return {
            "tokenizer_id": "EleutherAI/gpt-neox-20b",
            "vocab_size": vocab_size,
            "tokenizer_loaded": True,
        }
    except Exception as e:
        if data_mode == "real":
            raise PreflightError(f"Tokenizer load failed: {e}") from e
        return {"tokenizer_loaded": False, "error": str(e)}


def step05_token_id_range(data_mode: str, tokenizer_info: dict,
                          errors_path: pathlib.Path) -> dict:
    """Step 5: Verify token IDs are within [0, vocab_size)."""
    print("[Preflight 05/14] Token ID range check ...", flush=True)
    if data_mode == "synthetic" or not tokenizer_info.get("tokenizer_loaded"):
        print("  Skipping (no real tokenizer)", flush=True)
        return {"skipped": True}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    sample = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(sample)
    vocab_size = tok.vocab_size
    bad = [i for i in ids if i < 0 or i >= vocab_size]
    if bad:
        raise PreflightError(
            f"Step 5 FAILED — token IDs out of range [0, {vocab_size}): {bad}"
        )
    print(f"  Sample encoded: {len(ids)} tokens, all in [0, {vocab_size})  [OK]", flush=True)
    return {"sample_tokens": len(ids), "vocab_size": vocab_size, "out_of_range": []}


def step06_strict_config_validation(dense_config_path: pathlib.Path,
                                     moe_config_path: pathlib.Path,
                                     errors_path: pathlib.Path) -> dict:
    """Step 6: Load and strictly validate both configs."""
    print("[Preflight 06/14] Strict config validation ...", flush=True)
    import yaml
    from training.config_loader import (
        load_dense_config, load_moe_config, config_to_dict,
        ConfigValidationError,
    )
    results = {}

    # Dense
    with open(dense_config_path, encoding="utf-8") as f:
        dense_raw = yaml.safe_load(f)
    try:
        dense_cfg = load_dense_config(dense_raw["model"])
        results["dense"] = {
            "status": "ok",
            "num_layers": dense_cfg.num_layers,
            "vocab_size": dense_cfg.vocab_size,
            "hidden_size": dense_cfg.hidden_size,
            "intermediate_size": dense_cfg.intermediate_size,
            "config_hash": _file_sha256(dense_config_path),
        }
        print(f"  Dense: num_layers={dense_cfg.num_layers}, vocab={dense_cfg.vocab_size}  [OK]", flush=True)
    except ConfigValidationError as e:
        raise PreflightError(f"Dense config validation failed: {e}") from e

    # MoE
    with open(moe_config_path, encoding="utf-8") as f:
        moe_raw = yaml.safe_load(f)
    try:
        moe_cfg = load_moe_config(moe_raw["model"])
        results["moe"] = {
            "status": "ok",
            "num_layers": moe_cfg.base.num_layers,
            "vocab_size": moe_cfg.base.vocab_size,
            "hidden_size": moe_cfg.base.hidden_size,
            "num_experts": moe_cfg.num_experts,
            "num_experts_per_token": moe_cfg.num_experts_per_token,
            "expert_capacity_factor": moe_cfg.expert_capacity_factor,
            "router_aux_loss_coeff": moe_cfg.router_aux_loss_coeff,
            "router_z_loss_coeff": moe_cfg.router_z_loss_coeff,
            "config_hash": _file_sha256(moe_config_path),
        }
        print(f"  MoE: num_layers={moe_cfg.base.num_layers}, experts={moe_cfg.num_experts}, top_k={moe_cfg.num_experts_per_token}  [OK]", flush=True)
    except ConfigValidationError as e:
        raise PreflightError(f"MoE config validation failed: {e}") from e

    return results


def step07_model_construction(dense_config_path: pathlib.Path,
                               moe_config_path: pathlib.Path,
                               errors_path: pathlib.Path) -> tuple[dict, Any, Any]:
    """Step 7: Construct actual models and compute exact parameter counts."""
    print("[Preflight 07/14] Actual model construction ...", flush=True)
    import yaml
    import torch
    from training.config_loader import load_dense_config, load_moe_config
    from training.models.dense import DenseTransformer
    from training.models.moe import MoETransformer

    with open(dense_config_path, encoding="utf-8") as f:
        dense_raw = yaml.safe_load(f)
    with open(moe_config_path, encoding="utf-8") as f:
        moe_raw = yaml.safe_load(f)

    dense_cfg = load_dense_config(dense_raw["model"])
    moe_cfg   = load_moe_config(moe_raw["model"])

    dense_model = DenseTransformer(dense_cfg)
    moe_model   = MoETransformer(moe_cfg)

    dense_total, dense_trainable = _count_params(dense_model)
    moe_total,   moe_trainable   = _count_params(moe_model)
    moe_active = _count_moe_active_params(moe_model, moe_cfg)

    # Verify against expected values from YAML metadata (±1% tolerance)
    dense_expected = dense_raw.get("metadata", {}).get("total_params")
    moe_expected   = moe_raw.get("metadata", {}).get("total_params")
    tolerance      = 0.01

    def _check_tolerance(actual, expected, label):
        if expected is None:
            return f"no expected value in metadata"
        diff_pct = abs(actual - expected) / expected
        if diff_pct > tolerance:
            raise PreflightError(
                f"Step 7 FAILED — {label} param count mismatch: "
                f"actual={actual:,}, expected={expected:,}, diff={diff_pct:.1%} > {tolerance:.0%}"
            )
        return f"ok (diff={diff_pct:.2%})"

    dense_check = _check_tolerance(dense_total, dense_expected, "Dense")
    moe_check   = _check_tolerance(moe_total, moe_expected, "MoE total")

    print(f"  Dense: {dense_total:,} params ({dense_total/1e6:.2f}M)  {dense_check}", flush=True)
    print(f"  MoE total: {moe_total:,} params ({moe_total/1e6:.2f}M)  {moe_check}", flush=True)
    print(f"  MoE active: {moe_active:,} params ({moe_active/1e6:.2f}M)", flush=True)

    manifest = {
        "dense": {
            "total_params": dense_total,
            "trainable_params": dense_trainable,
            "total_params_m": round(dense_total / 1e6, 3),
            "num_layers": dense_cfg.num_layers,
            "hidden_size": dense_cfg.hidden_size,
            "intermediate_size": dense_cfg.intermediate_size,
            "vocab_size": dense_cfg.vocab_size,
        },
        "moe": {
            "total_params": moe_total,
            "trainable_params": moe_trainable,
            "active_params_per_token": moe_active,
            "total_params_m": round(moe_total / 1e6, 3),
            "active_params_m": round(moe_active / 1e6, 3),
            "num_layers": moe_cfg.base.num_layers,
            "hidden_size": moe_cfg.base.hidden_size,
            "intermediate_size": moe_cfg.base.intermediate_size,
            "vocab_size": moe_cfg.base.vocab_size,
            "num_experts": moe_cfg.num_experts,
            "num_experts_per_token": moe_cfg.num_experts_per_token,
            "expert_capacity_factor": moe_cfg.expert_capacity_factor,
            "router_aux_loss_coeff": moe_cfg.router_aux_loss_coeff,
            "router_z_loss_coeff": moe_cfg.router_z_loss_coeff,
        },
    }
    return manifest, dense_model, moe_model


def step08_param_count_verification(manifest: dict, errors_path: pathlib.Path) -> dict:
    """Step 8: Record and return the exact parameter counts (already computed in step 7)."""
    print("[Preflight 08/14] Parameter count verification ...", flush=True)
    print(f"  Dense total:  {manifest['dense']['total_params']:,}  ({manifest['dense']['total_params_m']}M)  [OK]", flush=True)
    print(f"  MoE total:    {manifest['moe']['total_params']:,}  ({manifest['moe']['total_params_m']}M)  [OK]", flush=True)
    print(f"  MoE active:   {manifest['moe']['active_params_per_token']:,}  ({manifest['moe']['active_params_m']}M)  [OK]", flush=True)
    return manifest


def step09_dense_cpu_step(dense_model: Any, data_mode: str,
                           errors_path: pathlib.Path) -> dict:
    """Step 9: One dense CPU forward/backward step."""
    print("[Preflight 09/14] Dense CPU forward/backward ...", flush=True)
    import torch
    device = torch.device("cpu")
    dense_model = dense_model.to(device)
    dense_model.train()

    vocab_size = dense_model.config.vocab_size
    batch, seq = 1, 16
    if data_mode == "synthetic":
        input_ids = torch.randint(0, vocab_size, (batch, seq))
    else:
        input_ids = torch.randint(0, vocab_size, (batch, seq))  # CPU step always uses synthetic

    out = dense_model(input_ids=input_ids, labels=input_ids)
    loss = out["loss"]
    loss.backward()
    loss_val = float(loss.item())
    print(f"  Dense CPU step: loss={loss_val:.4f}  [OK]", flush=True)
    return {"dense_cpu_loss": loss_val, "status": "ok"}


def step10_moe_cpu_step(moe_model: Any, data_mode: str,
                         errors_path: pathlib.Path) -> dict:
    """Step 10: One MoE CPU forward/backward step."""
    print("[Preflight 10/14] MoE CPU forward/backward ...", flush=True)
    import torch
    device = torch.device("cpu")
    moe_model = moe_model.to(device)
    moe_model.train()

    vocab_size = moe_model.config.base.vocab_size
    batch, seq = 1, 16
    input_ids = torch.randint(0, vocab_size, (batch, seq))

    out = moe_model(input_ids=input_ids, labels=input_ids)
    loss = out["loss"]
    loss.backward()
    loss_val = float(loss.item())
    aux_loss_val = float(out.get("aux_loss", torch.tensor(0.0)).item())
    print(f"  MoE CPU step: loss={loss_val:.4f}, aux_loss={aux_loss_val:.6f}  [OK]", flush=True)
    return {"moe_cpu_loss": loss_val, "moe_cpu_aux_loss": aux_loss_val, "status": "ok"}


def step11_dense_cuda_step(dense_model: Any, data_mode: str,
                            errors_path: pathlib.Path) -> dict:
    """Step 11: One dense CUDA forward/backward step."""
    print("[Preflight 11/14] Dense CUDA forward/backward ...", flush=True)
    import torch
    if not torch.cuda.is_available():
        print("  CUDA not available — skipping", flush=True)
        return {"skipped": True, "reason": "CUDA not available"}

    device = torch.device("cuda")
    dense_model = dense_model.to(device)
    dense_model.train()

    vocab_size = dense_model.config.vocab_size
    batch, seq = 1, 16
    input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = dense_model(input_ids=input_ids, labels=input_ids)
    loss = out["loss"]
    loss.backward()
    loss_val = float(loss.item())
    vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Dense CUDA step: loss={loss_val:.4f}, VRAM={vram_gb:.3f} GB  [OK]", flush=True)
    return {"dense_cuda_loss": loss_val, "dense_cuda_vram_gb": vram_gb, "status": "ok"}


def step12_moe_cuda_step(moe_model: Any, data_mode: str,
                          errors_path: pathlib.Path) -> dict:
    """Step 12: One MoE CUDA forward/backward step."""
    print("[Preflight 12/14] MoE CUDA forward/backward ...", flush=True)
    import torch
    if not torch.cuda.is_available():
        print("  CUDA not available — skipping", flush=True)
        return {"skipped": True, "reason": "CUDA not available"}

    device = torch.device("cuda")
    moe_model = moe_model.to(device)
    moe_model.train()

    vocab_size = moe_model.config.base.vocab_size
    batch, seq = 1, 16
    input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = moe_model(input_ids=input_ids, labels=input_ids)
    loss = out["loss"]
    loss.backward()
    loss_val = float(loss.item())
    aux_loss_val = float(out.get("aux_loss", torch.tensor(0.0)).item())
    vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  MoE CUDA step: loss={loss_val:.4f}, aux={aux_loss_val:.6f}, VRAM={vram_gb:.3f} GB  [OK]", flush=True)

    if vram_gb > VRAM_TARGET_GB:
        raise PreflightError(
            f"VRAM_EXCEEDED: MoE CUDA step used {vram_gb:.3f} GB > target {VRAM_TARGET_GB} GB. "
            f"Reduce batch_size or seq_length in the MoE config."
        )
    return {"moe_cuda_loss": loss_val, "moe_cuda_aux_loss": aux_loss_val,
            "moe_cuda_vram_gb": vram_gb, "status": "ok"}


def step13_router_metrics_schema(moe_model: Any, errors_path: pathlib.Path) -> dict:
    """Step 13: Validate router metrics schema from a CPU forward pass."""
    print("[Preflight 13/14] Router metrics schema validation ...", flush=True)
    import torch
    from training.router_metrics import REQUIRED_ACCEPTANCE_KEYS

    device = torch.device("cpu")
    moe_model = moe_model.to(device)
    moe_model.eval()

    vocab_size = moe_model.config.base.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 16))
    with torch.no_grad():
        out = moe_model(input_ids=input_ids, labels=input_ids)

    router_metrics_list = out.get("router_metrics", [])
    if not router_metrics_list:
        raise PreflightError(
            "Step 13 FAILED — model output contains no router_metrics. "
            "Check MoETransformer.forward() returns router_metrics in the dict."
        )

    # Check first layer's metrics for all required keys
    first_layer = router_metrics_list[0]
    missing = [k for k in REQUIRED_ACCEPTANCE_KEYS if k not in first_layer]
    if missing:
        raise PreflightError(
            f"Step 13 FAILED — router metrics missing required keys: {missing}. "
            f"Present keys: {sorted(first_layer.keys())}"
        )

    print(f"  Router metrics: {len(router_metrics_list)} layers, all {len(REQUIRED_ACCEPTANCE_KEYS)} required keys present  [OK]", flush=True)
    return {
        "num_layers": len(router_metrics_list),
        "required_keys_present": list(REQUIRED_ACCEPTANCE_KEYS),
        "status": "ok",
    }


def step14_artifact_freshness(run_dir: pathlib.Path, run_start: datetime.datetime,
                               errors_path: pathlib.Path) -> dict:
    """Step 14: Verify artifact directory is fresh (created after run_start)."""
    print("[Preflight 14/14] Artifact directory freshness ...", flush=True)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {run_dir}  [OK]", flush=True)
        return {"run_dir": str(run_dir), "status": "created"}

    # Check for stale artifacts from a previous run
    stale = []
    for f in run_dir.iterdir():
        mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime, tz=datetime.timezone.utc)
        if mtime < run_start:
            stale.append(f.name)
    if stale:
        raise PreflightError(
            f"Step 14 FAILED — artifact directory {run_dir} contains stale files "
            f"predating run start {run_start.isoformat()}: {stale}. "
            f"Use a new --run-id or delete the directory."
        )
    print(f"  Artifact directory: {run_dir}  [OK]", flush=True)
    return {"run_dir": str(run_dir), "status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_preflight(args: argparse.Namespace, run_dir: pathlib.Path,
                  run_start: datetime.datetime) -> tuple[bool, dict]:
    """Run all 14 preflight steps. Return (all_passed, results_dict)."""
    errors_path = run_dir / "errors.jsonl"
    results: dict[str, Any] = {}
    dense_model = None
    moe_model = None

    steps = [
        ("01_dependency_imports",    lambda: step01_dependency_imports(errors_path)),
        ("02_pip_check",             lambda: step02_pip_check(errors_path)),
        ("03_real_text_dataset",     lambda: step03_real_text_dataset(args.data_mode, errors_path)),
        ("04_tokenizer_load",        lambda: step04_tokenizer_load(args.data_mode, errors_path)),
        ("05_token_id_range",        lambda: step05_token_id_range(
            args.data_mode, results.get("04_tokenizer_load", {}), errors_path)),
        ("06_strict_config",         lambda: step06_strict_config_validation(
            args.dense_config, args.moe_config, errors_path)),
        ("07_model_construction",    None),   # handled separately (returns models)
        ("08_param_count",           None),   # handled after 07
        ("09_dense_cpu_step",        None),
        ("10_moe_cpu_step",          None),
        ("11_dense_cuda_step",       None),
        ("12_moe_cuda_step",         None),
        ("13_router_metrics_schema", None),
        ("14_artifact_freshness",    lambda: step14_artifact_freshness(run_dir, run_start, errors_path)),
    ]

    all_passed = True

    for step_name, step_fn in steps:
        try:
            if step_name == "07_model_construction":
                manifest, dense_model, moe_model = step07_model_construction(
                    args.dense_config, args.moe_config, errors_path)
                results[step_name] = manifest
            elif step_name == "08_param_count":
                results[step_name] = step08_param_count_verification(
                    results["07_model_construction"], errors_path)
            elif step_name == "09_dense_cpu_step":
                results[step_name] = step09_dense_cpu_step(dense_model, args.data_mode, errors_path)
            elif step_name == "10_moe_cpu_step":
                results[step_name] = step10_moe_cpu_step(moe_model, args.data_mode, errors_path)
            elif step_name == "11_dense_cuda_step":
                results[step_name] = step11_dense_cuda_step(dense_model, args.data_mode, errors_path)
            elif step_name == "12_moe_cuda_step":
                results[step_name] = step12_moe_cuda_step(moe_model, args.data_mode, errors_path)
            elif step_name == "13_router_metrics_schema":
                results[step_name] = step13_router_metrics_schema(moe_model, errors_path)
            else:
                results[step_name] = step_fn()
        except PreflightError as e:
            print(f"  !! PREFLIGHT HALTED at {step_name}: {e}", flush=True)
            _append_error(errors_path, step_name, e)
            results[step_name] = {"status": "FAILED", "error": str(e)}
            all_passed = False
            break
        except Exception as e:
            print(f"  !! UNEXPECTED ERROR at {step_name}: {e}", flush=True)
            _append_error(errors_path, step_name, e)
            results[step_name] = {"status": "ERROR", "error": str(e)}
            all_passed = False
            break

    return all_passed, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jupiter Shot Laptop Validation Pipeline"
    )
    parser.add_argument(
        "--data-mode", choices=["real", "synthetic", "auto"], default="real",
        help="Data source for training steps (default: real)"
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run only the 14-step preflight gate; do not start GPU workloads"
    )
    parser.add_argument(
        "--dense-config", type=pathlib.Path, default=DEFAULT_DENSE_CONFIG,
        help="Path to dense YAML config"
    )
    parser.add_argument(
        "--moe-config", type=pathlib.Path, default=DEFAULT_MOE_CONFIG,
        help="Path to MoE YAML config"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override auto-generated UTC run ID"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of training steps for GPU workloads (default: 100)"
    )
    args = parser.parse_args()

    run_start = datetime.datetime.now(datetime.timezone.utc)
    run_id    = args.run_id or _utc_now()
    run_dir   = REPO_ROOT / "benchmarks" / "results" / "laptop" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git = _git_info()
    print(f"\nJupiter Shot Laptop Validation Pipeline", flush=True)
    print(f"  Run ID:     {run_id}", flush=True)
    print(f"  Branch:     {git['branch']}", flush=True)
    print(f"  Commit:     {git['commit']}", flush=True)
    print(f"  Data mode:  {args.data_mode}", flush=True)
    print(f"  Artifacts:  {run_dir}", flush=True)
    print(f"  Preflight only: {args.preflight_only}", flush=True)
    print(flush=True)

    # Write initial run manifest
    manifest = {
        "run_id": run_id,
        "run_start_utc": run_start.isoformat(),
        "git": git,
        "data_mode": args.data_mode,
        "preflight_only": args.preflight_only,
        "dense_config": str(args.dense_config),
        "moe_config": str(args.moe_config),
        "steps": args.steps,
        "pipeline_version": "7.0",
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    # ── Run preflight ─────────────────────────────────────────────────────────
    preflight_passed, preflight_results = run_preflight(args, run_dir, run_start)
    _write_json(run_dir / "preflight.json", {
        "run_id": run_id,
        "all_passed": preflight_passed,
        "steps": preflight_results,
    })

    if not preflight_passed:
        print(f"\n[PIPELINE] Preflight FAILED. See {run_dir / 'preflight.json'}", flush=True)
        print(f"[PIPELINE] Exit code: {EXIT_SAFETY_STOP} (SAFETY_STOP)", flush=True)
        return EXIT_SAFETY_STOP

    print(f"\n[PIPELINE] All 14 preflight steps PASSED.", flush=True)

    if args.preflight_only:
        print(f"[PIPELINE] --preflight-only: stopping before GPU workloads.", flush=True)
        print(f"[PIPELINE] Artifacts: {run_dir}", flush=True)
        print(f"[PIPELINE] Exit code: {EXIT_PASS} (PASS)", flush=True)
        return EXIT_PASS

    # ── GPU workloads (100-step runs) ─────────────────────────────────────────
    # These call the existing runner scripts, which now use the strict config loader.
    print(f"\n[PIPELINE] Starting GPU workloads ({args.steps} steps) ...", flush=True)

    runners = [
        ("dense",  "scripts/run_laptop_dense.py",   str(args.dense_config)),
        ("moe",    "scripts/run_laptop_moe.py",      str(args.moe_config)),
        ("resume", "scripts/run_laptop_resume_test.py", str(args.dense_config)),
    ]

    all_gpu_passed = True
    gpu_results = {}

    for name, script, config_path in runners:
        script_path = REPO_ROOT / script
        if not script_path.exists():
            print(f"  [SKIP] {name}: {script} not found", flush=True)
            gpu_results[name] = {"status": "skipped", "reason": "script not found"}
            continue

        print(f"\n[PIPELINE] Running {name} ({script}) ...", flush=True)
        cmd = [
            sys.executable, str(script_path),
            "--config", config_path,
            "--data-mode", args.data_mode,
            "--steps", str(args.steps),
            "--run-id", run_id,
            "--output-dir", str(run_dir),
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        exit_code = result.returncode
        gpu_results[name] = {"exit_code": exit_code}

        if exit_code == EXIT_PASS:
            print(f"  {name}: PASS (exit 0)", flush=True)
        elif exit_code == EXIT_NOT_ACCEPTED:
            print(f"  {name}: NOT_ACCEPTED (exit 1)", flush=True)
            all_gpu_passed = False
        elif exit_code == EXIT_NOT_EVALUABLE:
            print(f"  {name}: NOT_EVALUABLE (exit 2)", flush=True)
            all_gpu_passed = False
        elif exit_code == EXIT_EXECUTION_ERROR:
            print(f"  {name}: EXECUTION_ERROR (exit 3)", flush=True)
            all_gpu_passed = False
        elif exit_code == EXIT_SAFETY_STOP:
            print(f"  {name}: SAFETY_STOP (exit 4)", flush=True)
            all_gpu_passed = False
        else:
            print(f"  {name}: UNKNOWN exit code {exit_code}", flush=True)
            all_gpu_passed = False

    _write_json(run_dir / "gpu_results.json", gpu_results)

    # ── Final verdict ─────────────────────────────────────────────────────────
    if all_gpu_passed:
        verdict = "PASS"
        exit_code = EXIT_PASS
    elif args.data_mode == "synthetic":
        verdict = "NOT_ACCEPTED (synthetic data)"
        exit_code = EXIT_NOT_ACCEPTED
    else:
        verdict = "NOT_ACCEPTED"
        exit_code = EXIT_NOT_ACCEPTED

    run_end = datetime.datetime.now(datetime.timezone.utc)
    duration = (run_end - run_start).total_seconds()

    summary = {
        "run_id": run_id,
        "verdict": verdict,
        "exit_code": exit_code,
        "data_mode": args.data_mode,
        "preflight_passed": preflight_passed,
        "gpu_results": gpu_results,
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "duration_seconds": duration,
        "git": git,
    }
    _write_json(run_dir / "summary.json", summary)

    print(f"\n[PIPELINE] Verdict: {verdict}", flush=True)
    print(f"[PIPELINE] Duration: {duration:.1f}s", flush=True)
    print(f"[PIPELINE] Artifacts: {run_dir}", flush=True)
    print(f"[PIPELINE] Exit code: {exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
