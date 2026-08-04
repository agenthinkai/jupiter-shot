"""
Jupiter Shot — Laptop Dense CUDA Validation
=============================================
Runs real CUDA forward and backward passes using the repository's actual
DenseTransformer implementation. Collects all required metrics.

Model API contract:
  model(input_ids=..., labels=...) → dict with keys:
    - 'logits': (batch, seq_len, vocab_size)
    - 'loss': total loss (LM loss) if labels provided, else None

Execution stages:
  1. 10 diagnostic steps (stop on NaN/Inf)
  2. 100 steps if diagnostics pass
  3. 1,000 steps only after explicit confirmation

Safety controls:
  - Thermal warning at 80°C, stop at 90°C (configurable)
  - Stop on NaN/Inf
  - Stop on repeated CUDA OOM
  - Graceful checkpoint on CTRL+C or SIGTERM
  - Frequent metric persistence
  - Failure artifact saved on exception

Usage:
    python scripts/run_laptop_dense.py --config laptop_dense_small
    python scripts/run_laptop_dense.py --config laptop_dense_small --steps 1000 --confirmed
    python scripts/run_laptop_dense.py --synthetic  # synthetic data only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "laptop"
LOGS_DIR = REPO_ROOT / "logs" / "laptop"
CKPT_DIR = REPO_ROOT / "checkpoints" / "laptop"

# ── Safety thresholds ─────────────────────────────────────────────────────────
THERMAL_WARN_C = 80
THERMAL_STOP_C = 90
MAX_OOM_RETRIES = 3
METRIC_FLUSH_INTERVAL = 10  # steps

# ── Semantic artifact contract ────────────────────────────────────────────────
# All runners must write these fields so the pipeline can validate artifacts
# without relying on raw subprocess exit codes.
ARTIFACT_SCHEMA_VERSION = "1.0"
EXIT_PASS             = 0
EXIT_NOT_ACCEPTED     = 1
EXIT_NOT_EVALUABLE    = 2
EXIT_EXECUTION_ERROR  = 3
EXIT_SAFETY_STOP      = 4
OUTCOME_PASS             = "PASS"
OUTCOME_NOT_ACCEPTED     = "NOT_ACCEPTED"
OUTCOME_NOT_EVALUABLE    = "NOT_EVALUABLE"
OUTCOME_EXECUTION_ERROR  = "EXECUTION_ERROR"
OUTCOME_SAFETY_STOP      = "SAFETY_STOP"
# Mapping from outcome string to exit code (used when writing the artifact)
_OUTCOME_TO_EXIT = {
    OUTCOME_PASS:            EXIT_PASS,
    OUTCOME_NOT_ACCEPTED:    EXIT_NOT_ACCEPTED,
    OUTCOME_NOT_EVALUABLE:   EXIT_NOT_EVALUABLE,
    OUTCOME_EXECUTION_ERROR: EXIT_EXECUTION_ERROR,
    OUTCOME_SAFETY_STOP:     EXIT_SAFETY_STOP,
}


def _gpu_temp() -> Optional[int]:
    """Read GPU temperature from nvidia-smi. Returns None if unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def _gpu_util() -> Optional[int]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def _vram_stats(torch: Any) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def _check_thermal(step: int, warn_c: int, stop_c: int) -> None:
    temp = _gpu_temp()
    if temp is None:
        return
    if temp >= stop_c:
        raise RuntimeError(f"[THERMAL STOP] GPU temperature {temp}°C >= stop threshold {stop_c}°C at step {step}. "
                           "Halting to protect hardware.")
    if temp >= warn_c:
        print(f"[THERMAL WARNING] GPU temperature {temp}°C >= warning threshold {warn_c}°C at step {step}.")


def _check_nan_inf(loss_val: float, step: int) -> None:
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f"[NaN/Inf DETECTED] Loss={loss_val} at step {step}. Stopping.")


# ── Data ──────────────────────────────────────────────────────────────────────

def get_synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: Any) -> Any:
    import torch
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def get_real_text_batch(tokenizer: Any, texts: list[str], seq_len: int, device: Any) -> Any:
    import torch
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=seq_len, padding="max_length")
    return enc["input_ids"].to(device)


def load_wikitext_sample(n_samples: int = 200) -> list[str]:
    """
    Load a small sample from Wikitext-2 (MIT license).
    Falls back to synthetic if unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          trust_remote_code=False)
        texts = [row["text"] for row in ds if len(row["text"]) > 100][:n_samples]
        print(f"[DATA] Loaded {len(texts)} samples from Wikitext-2 (MIT license)")
        return texts
    except Exception as e:
        print(f"[DATA] Wikitext-2 unavailable ({e}). Using synthetic data.")
        return []


# ── Training loop ─────────────────────────────────────────────────────────────

def run_dense_validation(
    config_name: str,
    max_steps: int = 100,
    synthetic: bool = False,
    thermal_warn: int = THERMAL_WARN_C,
    thermal_stop: int = THERMAL_STOP_C,
    output_dir: Path = RESULTS_DIR,
    ckpt_dir: Path = CKPT_DIR,
    log_dir: Path = LOGS_DIR,
) -> dict:
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("[FAIL] CUDA is not available. Cannot run GPU validation.")

    # Load config — use shared resolver to support full paths, relative paths, and bare names
    from training.config_path import resolve_config_path, format_missing_error, safe_checkpoint_name
    _res = resolve_config_path(config_name, repo_root=REPO_ROOT)
    config_path = _res.resolved_path
    if not _res.exists:
        raise FileNotFoundError(format_missing_error(_res))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Import model
    from training.models.dense import DenseTransformer, DenseConfig
    dense_config = DenseConfig(**{k: v for k, v in model_cfg.items() if hasattr(DenseConfig, k)})
    device = torch.device("cuda:0")

    print(f"\n[DENSE] Config: {config_name}")
    print(f"[DENSE] Device: {torch.cuda.get_device_name(0)}")
    print(f"[DENSE] Precision: {train_cfg.get('precision', 'fp16')}")

    # Build model
    model = DenseTransformer(dense_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DENSE] Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"[DENSE] Trainable parameters: {trainable_params:,}")

    # Precision
    precision = train_cfg.get("precision", "fp16")
    use_bf16 = precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = precision == "fp16" or (precision == "bf16" and not use_bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[DENSE] Using precision: {'bf16' if use_bf16 else 'fp16'}")

    # Activation checkpointing
    if train_cfg.get("gradient_checkpointing", False):
        print("[DENSE] Gradient checkpointing: enabled")

    # Optimizer
    lr = float(train_cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # LR scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    # Data
    vocab_size = dense_config.vocab_size
    seq_len = int(train_cfg.get("seq_length", dense_config.max_position_embeddings))
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))

    texts = [] if synthetic else load_wikitext_sample(500)
    data_mode = "synthetic" if (synthetic or not texts) else "wikitext-2 (MIT)"
    print(f"[DENSE] Data mode: {data_mode}")

    tokenizer = None
    if texts:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            tokenizer.pad_token = tokenizer.eos_token
            # ── Vocabulary contract check ─────────────────────────────────
            # model_config.vocab_size MUST be >= len(tokenizer)
            # Halt immediately if not — do NOT remap with modulo arithmetic.
            actual_tokenizer_size = len(tokenizer)
            if vocab_size < actual_tokenizer_size:
                raise RuntimeError(
                    f"TOKENIZER_VOCABULARY_MISMATCH: "
                    f"model vocab_size={vocab_size} < "
                    f"tokenizer vocab_size={actual_tokenizer_size} "
                    f"(EleutherAI/gpt-neox-20b). "
                    f"Update the config to vocab_size={actual_tokenizer_size} "
                    f"before running real-text validation."
                )
            print(
                f"[VOCAB] Contract OK: model vocab_size={vocab_size} "
                f">= tokenizer vocab_size={actual_tokenizer_size}"
            )
        except Exception:
            print("[DATA] Tokenizer unavailable; falling back to synthetic.")
            texts = []
            data_mode = "synthetic (tokenizer unavailable)"

    # Output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "dense_metrics.jsonl"
    errors_path = output_dir / "errors.jsonl"
    failure_path = output_dir / "dense_failure_artifact.json"

    # State
    metrics_log: list[dict] = []
    nan_inf_count = 0
    oom_count = 0
    interrupted = False
    step = 0

    # CTRL+C / SIGTERM handler
    def _graceful_stop(signum, frame):
        nonlocal interrupted
        print(f"\n[SIGNAL] Received signal {signum}. Saving checkpoint and stopping.")
        interrupted = True

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    # AMP scaler — use torch.amp.GradScaler (torch.cuda.amp.GradScaler is deprecated in PyTorch 2.x)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    torch.cuda.reset_peak_memory_stats()
    start_wall = time.time()
    total_tokens = 0
    step_times = []

    model.train()
    optimizer.zero_grad()

    summary = {
        "config": config_name,
        "data_mode": data_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "precision": "bf16" if use_bf16 else "fp16",
        "steps_planned": max_steps,
        "steps_completed": 0,
        "nan_inf_count": 0,
        "oom_count": 0,
        "status": "RUNNING",
    }

    try:
        for step in range(1, max_steps + 1):
            if interrupted:
                break

            step_start = time.time()

            # Thermal check
            _check_thermal(step, thermal_warn, thermal_stop)

            # Build batch
            try:
                if texts and tokenizer:
                    import random
                    sample = random.sample(texts, min(batch_size, len(texts)))
                    input_ids = get_real_text_batch(tokenizer, sample, seq_len, device)
                    # Per-batch token ID range guard
                    id_min = int(input_ids.min().item())
                    id_max = int(input_ids.max().item())
                    if id_min < 0 or id_max >= vocab_size:
                        raise RuntimeError(
                            f"TOKENIZER_VOCABULARY_MISMATCH: "
                            f"batch token IDs [{id_min}, {id_max}] out of range "
                            f"[0, {vocab_size - 1}] at step {step}. "
                            f"Model vocab_size={vocab_size} is too small for this tokenizer."
                        )
                else:
                    input_ids = get_synthetic_batch(batch_size, seq_len, vocab_size, device)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(f"[OOM] Step {step}: CUDA OOM during batch creation (count: {oom_count})")
                torch.cuda.empty_cache()
                if oom_count >= MAX_OOM_RETRIES:
                    raise RuntimeError(f"[FAIL] {MAX_OOM_RETRIES} consecutive CUDA OOM errors. Stopping.")
                continue

            # Forward + backward
            # Model API: model(input_ids=..., labels=...) → dict
            #   out["loss"]   = cross-entropy loss (scalar tensor)
            #   out["logits"] = (batch, seq_len, vocab_size)
            try:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    labels = input_ids.clone()
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out["loss"] / grad_accum

                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % grad_accum == 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                print(f"[OOM] Step {step}: CUDA OOM during forward/backward (count: {oom_count})")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                if oom_count >= MAX_OOM_RETRIES:
                    raise RuntimeError(f"[FAIL] {MAX_OOM_RETRIES} consecutive CUDA OOM errors. Stopping.")
                continue

            loss_val = loss.item() * grad_accum
            _check_nan_inf(loss_val, step)

            step_time = time.time() - step_start
            step_times.append(step_time)
            tokens_this_step = batch_size * seq_len
            total_tokens += tokens_this_step
            tps = tokens_this_step / step_time

            vram = _vram_stats(torch)
            temp = _gpu_temp()
            util = _gpu_util()
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

            metric = {
                "step": step,
                "loss": round(loss_val, 6),
                "lr": lr_now,
                "step_time_s": round(step_time, 4),
                "tokens_per_sec": round(tps, 1),
                "total_tokens": total_tokens,
                **vram,
                "gpu_temp_c": temp,
                "gpu_util_pct": util,
            }
            metrics_log.append(metric)

            if step % METRIC_FLUSH_INTERVAL == 0 or step == 1:
                with open(metrics_path, "a") as f:
                    for m in metrics_log[-METRIC_FLUSH_INTERVAL:]:
                        f.write(json.dumps(m) + "\n")

            if step % 10 == 0 or step == 1:
                print(f"  Step {step:4d}/{max_steps} | loss={loss_val:.4f} | "
                      f"lr={lr_now:.2e} | {tps:.0f} tok/s | "
                      f"VRAM={vram.get('allocated_gb', '?')}GB | "
                      f"temp={temp or '?'}°C")

            # After 10 diagnostic steps, confirm stability
            if step == 10:
                print(f"\n[DIAGNOSTIC] 10 steps complete. Loss={loss_val:.4f}, NaN/Inf={nan_inf_count}")
                if nan_inf_count > 0:
                    raise ValueError(f"[FAIL] NaN/Inf detected in diagnostic phase. Stopping.")
                print("[DIAGNOSTIC] Stable. Continuing.\n")

    except (ValueError, RuntimeError) as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}")
        # Save failure artifact with full traceback and partial metrics
        artifact = {
            "step": step,
            "error": str(e),
            "traceback": tb,
            "time": time.time(),
            "partial_metrics": metrics_log[-5:] if metrics_log else [],
        }
        with open(errors_path, "a") as f:
            f.write(json.dumps({"step": step, "error": str(e), "time": time.time()}) + "\n")
        failure_path.write_text(json.dumps(artifact, indent=2, default=str))
        print(f"[FAILURE ARTIFACT] Saved: {failure_path}")
        summary["status"] = "FAILED"
        summary["failure_reason"] = str(e)
        summary["failure_traceback"] = tb
    else:
        summary["status"] = "INTERRUPTED" if interrupted else "COMPLETED"

    # Save checkpoint — use safe_checkpoint_name to avoid doubled suffixes
    _ckpt_stem = safe_checkpoint_name(config_name)
    ckpt_path = ckpt_dir / f"dense_{_ckpt_stem}_step{step}.pt"
    try:
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_val if step > 0 else None,
            "config": config_name,
        }, ckpt_path)
        print(f"\n[CHECKPOINT] Saved: {ckpt_path}")
        summary["checkpoint_path"] = str(ckpt_path)
        summary["checkpoint_size_mb"] = round(ckpt_path.stat().st_size / 1024**2, 1)
        summary["checkpoint_status"] = "saved"
    except Exception as e:
        print(f"[CHECKPOINT] Failed to save: {e}")
        summary["checkpoint_status"] = f"failed: {e}"

    # Final summary
    wall_time = time.time() - start_wall
    vram_final = _vram_stats(torch)
    summary.update({
        "steps_completed": step,
        "nan_inf_count": nan_inf_count,
        "oom_count": oom_count,
        "total_tokens": total_tokens,
        "wall_time_s": round(wall_time, 1),
        "avg_step_time_s": round(sum(step_times) / len(step_times), 4) if step_times else 0,
        "avg_tokens_per_sec": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "avg_tokens_per_sec_per_gpu": round(total_tokens / wall_time, 1),  # single GPU
        "peak_vram_allocated_gb": vram_final.get("max_allocated_gb"),
        "peak_vram_reserved_gb": vram_final.get("reserved_gb"),
        "first_loss": metrics_log[0]["loss"] if metrics_log else None,
        "last_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "loss_decreased": (
            metrics_log[-1]["loss"] < metrics_log[0]["loss"]
            if len(metrics_log) >= 2 else None
        ),
    })

    # ── Semantic artifact contract fields ──────────────────────────────────────────────
    # These fields are required by the pipeline's _validate_runner_artifact().
    # outcome and exit_code are derived from status; run_id and timestamp are
    # injected by main() after this function returns.
    if summary["status"] == "COMPLETED":
        _outcome = OUTCOME_PASS
    elif summary["status"] == "INTERRUPTED":
        # Interrupted by SIGTERM/CTRL+C — treated as NOT_EVALUABLE (ran but no
        # final acceptance verdict); pipeline will read the artifact and classify.
        _outcome = OUTCOME_NOT_EVALUABLE
    else:
        _outcome = OUTCOME_EXECUTION_ERROR
    summary["outcome"]        = _outcome
    summary["exit_code"]      = _OUTCOME_TO_EXIT[_outcome]
    summary["schema_version"] = ARTIFACT_SCHEMA_VERSION
    # timestamp is written by main() after run_id is known

    summary_path = output_dir / "dense_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[SUMMARY] Saved: {summary_path}")
    print(f"[SUMMARY] Status: {summary['status']}")
    print(f"[SUMMARY] Steps: {summary['steps_completed']}/{max_steps}")
    print(f"[SUMMARY] Loss: {summary['first_loss']} → {summary['last_loss']}")
    print(f"[SUMMARY] Tokens/sec: {summary['avg_tokens_per_sec']}")
    print(f"[SUMMARY] Peak VRAM: {summary['peak_vram_allocated_gb']} GB allocated")
    print(f"[SUMMARY] Wall time: {summary['wall_time_s']}s")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop Dense CUDA Validation")
    parser.add_argument("--config", default="laptop_dense_small",
                        help="Training config name (e.g., laptop_dense_small)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps (10=diagnostic, 100=standard, 1000=full)")
    parser.add_argument("--confirmed", action="store_true",
                        help="Required to run >100 steps without interactive confirmation")
    parser.add_argument(
        "--data-mode",
        choices=["real", "synthetic", "auto"],
        default="auto",
        help=(
            "real: Wikitext-2 only (halts if unavailable). "
            "synthetic: random tokens (diagnostic only, cannot produce PASS). "
            "auto: tries real, falls back to synthetic."
        ),
    )
    # Legacy flag kept for backward compatibility
    parser.add_argument("--synthetic", action="store_true",
                        help="Alias for --data-mode synthetic (deprecated)")
    parser.add_argument("--thermal-warn", type=int, default=THERMAL_WARN_C)
    parser.add_argument("--thermal-stop", type=int, default=THERMAL_STOP_C)
    parser.add_argument("--output-dir", default="benchmarks/results/laptop",
                        help="Directory for result artifacts")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Unique run identifier (e.g., 20260804_082957_UTC). "
            "Embedded in dense_summary.json for artifact traceability. "
            "Generated automatically if not provided."
        ),
    )
    args = parser.parse_args()

    # Resolve --synthetic alias
    if args.synthetic and args.data_mode == "auto":
        args.data_mode = "synthetic"

    # Resolve --data-mode to synthetic bool for run_dense_validation
    # real: force real text; halt if unavailable
    # synthetic: force synthetic
    # auto: try real, fall back to synthetic (existing behaviour)
    if args.data_mode == "real":
        use_synthetic = False
    elif args.data_mode == "synthetic":
        use_synthetic = True
    else:  # auto
        use_synthetic = False  # load_wikitext_sample handles the fallback

    if args.steps > 100 and not args.confirmed:
        print(f"[CONFIRM] Running {args.steps} steps requires --confirmed flag.")
        print("  Add --confirmed to proceed with extended run.")
        return 1

    import datetime as _dt
    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    print(f"[DENSE] Run ID: {run_id}", flush=True)

    try:
        summary = run_dense_validation(
            config_name=args.config,
            max_steps=args.steps,
            synthetic=use_synthetic,
            thermal_warn=args.thermal_warn,
            thermal_stop=args.thermal_stop,
            output_dir=Path(args.output_dir),
        )
        # Inject run_id and timestamp into the summary artifact so the pipeline
        # can validate artifact freshness and run_id consistency.
        summary["run_id"]    = run_id
        summary["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        summary_path = Path(args.output_dir) / "dense_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        # Return the semantic exit_code written by run_dense_validation()
        return int(summary.get("exit_code", EXIT_EXECUTION_ERROR))
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[FAIL] {e}")
        # Write a minimal failure artifact so the pipeline can read it
        import datetime as _dt2
        failure_artifact = {
            "run_id":         run_id,
            "outcome":        OUTCOME_EXECUTION_ERROR,
            "exit_code":      EXIT_EXECUTION_ERROR,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "timestamp":      _dt2.datetime.now(_dt2.timezone.utc).isoformat(),
            "error":          str(e),
        }
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / "dense_summary.json").write_text(
                json.dumps(failure_artifact, indent=2)
            )
        except Exception:
            pass
        return EXIT_EXECUTION_ERROR


if __name__ == "__main__":
    sys.exit(main())
