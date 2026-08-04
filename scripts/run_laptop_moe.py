"""
Jupiter Shot — Laptop MoE CUDA Validation
==========================================
Runs MoE training validation with 8 experts, top-2 routing, and full
router metric collection.

Model API contract:
  model(input_ids=..., labels=...) -> dict with keys:
    - 'logits':         (batch, seq_len, vocab_size)
    - 'loss':           total loss (LM + aux) if labels provided, else None
    - 'lm_loss':        language modeling loss only
    - 'aux_loss':       total auxiliary routing loss (scalar tensor)
    - 'router_metrics': list of per-layer routing metric dicts

Router metric key names are defined in training/router_metrics.py.
Do NOT hardcode key names here; import the K_* constants.

Exit codes:
  0  PASS          — all acceptance criteria met, real-text data used
  1  NOT_ACCEPTED  — training completed but one or more acceptance criteria failed
  2  NOT_EVALUABLE — training completed but required metrics were absent
  3  EXECUTION_ERROR — unrecoverable runtime error (OOM, NaN, CUDA failure)
  4  SAFETY_STOP   — thermal or other safety threshold triggered

Data modes (--data-mode):
  real      — Wikitext-2 (MIT license). Halts with exit 3 if unavailable.
  synthetic — Random token IDs. Diagnostic only; cannot produce a PASS.
  auto      — (default) Tries real; falls back to synthetic with a warning.

Usage:
    python scripts/run_laptop_moe.py --config laptop_moe_8gb_safe
    python scripts/run_laptop_moe.py --config laptop_moe_8gb_safe --steps 1000 --confirmed
    python scripts/run_laptop_moe.py --data-mode synthetic  # diagnostic only
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "laptop"
CKPT_DIR = REPO_ROOT / "checkpoints" / "laptop"
LOGS_DIR = REPO_ROOT / "logs" / "laptop"

THERMAL_WARN_C = 80
THERMAL_STOP_C = 90
MAX_OOM_RETRIES = 3
METRIC_FLUSH_INTERVAL = 10

# Exit codes
EXIT_PASS             = 0
EXIT_NOT_ACCEPTED     = 1
EXIT_NOT_EVALUABLE    = 2
EXIT_EXECUTION_ERROR  = 3
EXIT_SAFETY_STOP      = 4

# Data mode constants
DATA_MODE_REAL      = "real"
DATA_MODE_SYNTHETIC = "synthetic"
DATA_MODE_AUTO      = "auto"


def _gpu_temp() -> Optional[int]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
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
        raise SystemExit(EXIT_SAFETY_STOP)  # caught by main() as safety stop
    if temp >= warn_c:
        print(f"[THERMAL WARNING] GPU {temp}C >= {warn_c}C at step {step}.")


def _check_nan_inf(loss_val: float, step: int) -> None:
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f"[NaN/Inf] Loss={loss_val} at step {step}.")


def load_wikitext_sample(n_samples: int = 500) -> list[str]:
    """
    Load a sample from Wikitext-2 (MIT license).

    Returns list of text strings, or [] on failure.
    Does NOT raise — callers decide whether to abort or fall back.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          trust_remote_code=False)
        texts = [row["text"] for row in ds if len(row["text"]) > 100][:n_samples]
        print(f"[DATA] Loaded {len(texts)} samples from Wikitext-2 (MIT license)")
        return texts
    except Exception as e:
        print(f"[DATA] Wikitext-2 unavailable: {e}")
        return []


def get_real_text_batch(tokenizer: Any, texts: list[str], seq_len: int, device: Any) -> Any:
    import torch
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=seq_len, padding="max_length")
    return enc["input_ids"].to(device)


def run_moe_validation(
    config_name: str,
    max_steps: int = 100,
    data_mode: str = DATA_MODE_AUTO,
    thermal_warn: int = THERMAL_WARN_C,
    thermal_stop: int = THERMAL_STOP_C,
    output_dir: Path = RESULTS_DIR,
    ckpt_dir: Path = CKPT_DIR,
) -> dict:
    """
    Run MoE laptop validation.

    Returns summary dict with keys:
      status:        COMPLETED | INTERRUPTED | FAILED
      outcome:       PASS | NOT_ACCEPTED | NOT_EVALUABLE | EXECUTION_ERROR | SAFETY_STOP
      exit_code:     int (EXIT_* constant)
      data_mode:     str describing the data source used
      moe_accepted:  bool (True only when outcome is PASS)
      ...
    """
    import torch
    import yaml
    from training.router_metrics import (
        aggregate_layer_metrics,
        evaluate_acceptance,
        OUTCOME_PASS,
        OUTCOME_NOT_ACCEPTED,
        OUTCOME_NOT_EVALUABLE,
        OUTCOME_EXECUTION_ERROR,
        OUTCOME_SAFETY_STOP,
        K_ROUTER_ENTROPY,
        K_UTILIZATION_CV,
        K_NUM_INACTIVE_EXPERTS,
        K_DROPPED_TOKEN_FRACTION,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("[FAIL] CUDA is not available.")

    # Use shared resolver to support full paths, relative paths, and bare names
    from training.config_path import resolve_config_path, format_missing_error
    _res = resolve_config_path(config_name, repo_root=REPO_ROOT)
    config_path = _res.resolved_path
    if not _res.exists:
        raise FileNotFoundError(format_missing_error(_res))
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    from training.models.moe import MoETransformer, MoEConfig, DenseConfig
    base_cfg_dict = model_cfg.get("base", model_cfg)
    base_config = DenseConfig(**{k: v for k, v in base_cfg_dict.items() if hasattr(DenseConfig, k)})
    moe_config = MoEConfig(
        base=base_config,
        **{k: v for k, v in model_cfg.items() if k != "base" and hasattr(MoEConfig, k)}
    )

    device = torch.device("cuda:0")
    model = MoETransformer(moe_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for n, p in model.named_parameters() if "expert" in n.lower())
    non_expert_params = total_params - expert_params
    active_params = non_expert_params + (moe_config.num_experts_per_token / moe_config.num_experts) * expert_params

    print(f"\n[MOE] Config: {config_name}")
    print(f"[MOE] Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"[MOE] Active per token: {active_params:,.0f} ({active_params/1e6:.1f}M)")
    print(f"[MOE] Experts: {moe_config.num_experts}, top-{moe_config.num_experts_per_token}")

    precision = train_cfg.get("precision", "fp16")
    use_bf16 = precision == "bf16" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[MOE] Precision: {'bf16' if use_bf16 else 'fp16'}")

    lr = float(train_cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    # AMP scaler — torch.amp.GradScaler("cuda", ...) replaces deprecated torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    vocab_size = base_config.vocab_size
    seq_len = int(train_cfg.get("seq_length", base_config.max_position_embeddings))
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))

    # ── Data loading ──────────────────────────────────────────────────────────
    texts: list[str] = []
    tokenizer = None
    actual_data_mode: str = data_mode

    if data_mode in (DATA_MODE_REAL, DATA_MODE_AUTO):
        texts = load_wikitext_sample(500)
        if texts:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                tokenizer.pad_token = tokenizer.eos_token
                actual_data_mode = "wikitext-2 (MIT license)"
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
            except Exception as e:
                print(f"[DATA] Tokenizer unavailable ({e}).")
                texts = []

        if not texts:
            if data_mode == DATA_MODE_REAL:
                # Hard stop: caller explicitly requested real data
                raise RuntimeError(
                    "[FAIL] --data-mode real was requested but Wikitext-2 is unavailable. "
                    "Install the 'datasets' package and ensure internet access, "
                    "or use --data-mode auto to fall back to synthetic."
                )
            # AUTO mode: fall back to synthetic with a prominent warning
            print(
                "\n[DATA WARNING] Real text data unavailable. Falling back to synthetic.\n"
                "  Synthetic-only results are DIAGNOSTIC ONLY and cannot produce a PASS.\n"
                "  Install 'datasets' and 'transformers' for a valid run.\n"
            )
            actual_data_mode = "synthetic (real text unavailable)"
    else:
        actual_data_mode = "synthetic (--data-mode synthetic)"

    print(f"[MOE] Data mode: {actual_data_mode}")

    # ── Output setup ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "moe_metrics.jsonl"
    errors_path = output_dir / "errors.jsonl"
    failure_path = output_dir / "moe_failure_artifact.json"

    metrics_log: list[dict] = []
    nan_inf_count = 0
    oom_count = 0
    interrupted = False
    safety_stopped = False
    step = 0

    def _graceful_stop(signum, frame):
        nonlocal interrupted
        print(f"\n[SIGNAL] Received {signum}. Saving checkpoint.")
        interrupted = True

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    torch.cuda.reset_peak_memory_stats()
    start_wall = time.time()
    total_tokens = 0
    step_times: list[float] = []

    summary: dict[str, Any] = {
        "config": config_name,
        "data_mode": actual_data_mode,
        "total_params": total_params,
        "active_params_per_token": round(active_params),
        "num_experts": moe_config.num_experts,
        "top_k": moe_config.num_experts_per_token,
        "precision": "bf16" if use_bf16 else "fp16",
        "steps_planned": max_steps,
        "steps_completed": 0,
        "nan_inf_count": 0,
        "oom_count": 0,
        "status": "RUNNING",
        "outcome": None,
        "exit_code": None,
    }

    model.train()
    optimizer.zero_grad()

    try:
        for step in range(1, max_steps + 1):
            if interrupted:
                break

            step_start = time.time()
            try:
                _check_thermal(step, thermal_warn, thermal_stop)
            except SystemExit:
                safety_stopped = True
                print(f"[THERMAL STOP] GPU temp >= {thermal_stop}C at step {step}. Stopping.")
                break

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
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                if oom_count >= MAX_OOM_RETRIES:
                    raise RuntimeError(f"[FAIL] {MAX_OOM_RETRIES} OOM errors.")
                continue

            # Forward + backward
            # Model API: model(input_ids=..., labels=...) -> dict
            #   out["loss"]           = total loss (LM + aux)
            #   out["lm_loss"]        = language modeling loss only
            #   out["aux_loss"]       = auxiliary routing loss (scalar tensor)
            #   out["router_metrics"] = list of per-layer routing metric dicts
            try:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    labels = input_ids.clone()
                    out = model(input_ids=input_ids, labels=labels)
                    total_loss = out["loss"] / grad_accum

                if not use_bf16:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                if step % grad_accum == 0:
                    if not use_bf16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not use_bf16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                if oom_count >= MAX_OOM_RETRIES:
                    raise RuntimeError(f"[FAIL] {MAX_OOM_RETRIES} OOM errors.")
                continue

            # Unpack losses from dict output
            lm_loss_tensor = out.get("lm_loss")
            aux_loss_tensor = out.get("aux_loss")
            loss_val = (lm_loss_tensor.item() if lm_loss_tensor is not None
                        else (out["loss"].item() * grad_accum))
            aux_val = aux_loss_tensor.item() if aux_loss_tensor is not None else None
            _check_nan_inf(loss_val, step)

            # Router metrics — aggregate per-layer dicts using shared schema
            raw_router_metrics: list[dict] = out.get("router_metrics", [])
            router_metrics: dict = aggregate_layer_metrics(raw_router_metrics)

            step_time = time.time() - step_start
            step_times.append(step_time)
            tokens_this_step = batch_size * seq_len
            total_tokens += tokens_this_step
            tps = tokens_this_step / step_time
            vram = _vram_stats(torch)
            temp = _gpu_temp()
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

            metric: dict[str, Any] = {
                "step": step,
                "loss": round(loss_val, 6),
                "aux_loss": round(aux_val, 6) if aux_val is not None else None,
                "lr": lr_now,
                "step_time_s": round(step_time, 4),
                "tokens_per_sec": round(tps, 1),
                "total_tokens": total_tokens,
                **vram,
                "gpu_temp_c": temp,
                **router_metrics,
            }
            metrics_log.append(metric)

            if step % METRIC_FLUSH_INTERVAL == 0 or step == 1:
                with open(metrics_path, "a", encoding="utf-8") as f:
                    for m in metrics_log[-METRIC_FLUSH_INTERVAL:]:
                        f.write(json.dumps(m) + "\n")

            if step % 10 == 0 or step == 1:
                inactive = router_metrics.get(K_NUM_INACTIVE_EXPERTS, "?")
                aux_str = f"aux={aux_val:.4f}" if aux_val is not None else "aux=?"
                inactive_str = f"inactive={inactive}" if router_metrics else ""
                print(f"  Step {step:4d}/{max_steps} | loss={loss_val:.4f} | "
                      f"{aux_str} | {tps:.0f} tok/s | {inactive_str}")

            if step == 10:
                print(f"\n[DIAGNOSTIC] 10 steps complete. Loss={loss_val:.4f}")
                if nan_inf_count > 0:
                    raise ValueError("[FAIL] NaN/Inf in diagnostic phase.")
                print("[DIAGNOSTIC] Stable. Continuing.\n")

    except (ValueError, RuntimeError) as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}")
        artifact = {
            "step": step,
            "error": str(e),
            "traceback": tb,
            "time": time.time(),
            "partial_metrics": metrics_log[-5:] if metrics_log else [],
        }
        with open(errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "error": str(e)}) + "\n")
        failure_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
        print(f"[FAILURE ARTIFACT] Saved: {failure_path}")
        summary["status"] = "FAILED"
        summary["failure_reason"] = str(e)
        summary["failure_traceback"] = tb
        summary["outcome"] = OUTCOME_EXECUTION_ERROR
        summary["exit_code"] = EXIT_EXECUTION_ERROR
    else:
        if safety_stopped:
            summary["status"] = "SAFETY_STOP"
            summary["outcome"] = OUTCOME_SAFETY_STOP
            summary["exit_code"] = EXIT_SAFETY_STOP
        else:
            summary["status"] = "INTERRUPTED" if interrupted else "COMPLETED"

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if step > 0:
        ckpt_path = ckpt_dir / f"moe_{config_name}_step{step}.pt"
        ckpt_start = time.time()
        try:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_val if step > 0 else None,
            }, ckpt_path)
            ckpt_duration = time.time() - ckpt_start
            print(f"\n[CHECKPOINT] Saved: {ckpt_path} ({ckpt_duration:.1f}s)")
            summary["checkpoint_path"] = str(ckpt_path)
            summary["checkpoint_size_mb"] = round(ckpt_path.stat().st_size / 1024**2, 1)
            summary["checkpoint_save_duration_s"] = round(ckpt_duration, 2)
        except Exception as e:
            print(f"[CHECKPOINT] Failed: {e}")

    # ── Acceptance evaluation ─────────────────────────────────────────────────
    # Only evaluate if training completed or was interrupted (not FAILED/SAFETY_STOP)
    if summary["status"] in ("COMPLETED", "INTERRUPTED"):
        last_metrics_list = metrics_log[-10:] if len(metrics_log) >= 10 else metrics_log
        # Aggregate the last-10-step router metrics
        last_router_metrics_per_step = [
            {k: v for k, v in m.items() if not k.startswith(("step", "loss", "aux", "lr",
                                                               "step_time", "tokens", "total",
                                                               "allocated", "reserved", "max_",
                                                               "gpu_temp"))}
            for m in last_metrics_list
        ]
        agg_last10 = aggregate_layer_metrics(last_router_metrics_per_step)
        acceptance_result = evaluate_acceptance(
            agg_last10,
            window_description="last-10-steps",
        )

        # Synthetic-only runs cannot PASS regardless of metrics
        is_synthetic = "synthetic" in actual_data_mode.lower()
        if is_synthetic and acceptance_result["outcome"] == OUTCOME_PASS:
            acceptance_result["outcome"] = OUTCOME_NOT_ACCEPTED
            acceptance_result["errors"].append(
                "Synthetic data mode: a PASS requires real text data (Wikitext-2). "
                "Re-run with --data-mode real or --data-mode auto."
            )

        outcome = acceptance_result["outcome"]
        exit_code = {
            OUTCOME_PASS:          EXIT_PASS,
            OUTCOME_NOT_ACCEPTED:  EXIT_NOT_ACCEPTED,
            OUTCOME_NOT_EVALUABLE: EXIT_NOT_EVALUABLE,
        }.get(outcome, EXIT_NOT_EVALUABLE)

        summary["outcome"] = outcome
        summary["exit_code"] = exit_code
        summary["moe_accepted"] = (outcome == OUTCOME_PASS)
        summary["moe_acceptance"] = acceptance_result

        # Aggregate scalar metrics for the summary
        cv_vals = [m.get(K_UTILIZATION_CV) for m in last_metrics_list
                   if m.get(K_UTILIZATION_CV) is not None]
        entropy_vals = [m.get(K_ROUTER_ENTROPY) for m in last_metrics_list
                        if m.get(K_ROUTER_ENTROPY) is not None]
        avg_cv = sum(cv_vals) / len(cv_vals) if cv_vals else None
        avg_entropy = sum(entropy_vals) / len(entropy_vals) if entropy_vals else None

        summary.update({
            "steps_completed": step,
            "nan_inf_count": nan_inf_count,
            "oom_count": oom_count,
            "total_tokens": total_tokens,
            "wall_time_s": round(time.time() - start_wall, 1),
            "avg_tokens_per_sec": round(total_tokens / max(time.time() - start_wall, 1e-9), 1),
            "peak_vram_allocated_gb": _vram_stats(torch).get("max_allocated_gb"),
            "first_loss": metrics_log[0]["loss"] if metrics_log else None,
            "last_loss": metrics_log[-1]["loss"] if metrics_log else None,
            "avg_aux_loss_last10": round(
                sum(m["aux_loss"] for m in last_metrics_list if m.get("aux_loss") is not None)
                / max(1, sum(1 for m in last_metrics_list if m.get("aux_loss") is not None)),
                6
            ) if any(m.get("aux_loss") is not None for m in last_metrics_list) else None,
            "avg_router_entropy_last10": round(avg_entropy, 6) if avg_entropy is not None else None,
            "avg_utilization_cv_last10": round(avg_cv, 6) if avg_cv is not None else None,
            "dropped_token_fraction": agg_last10.get(K_DROPPED_TOKEN_FRACTION, 0.0),
        })
    else:
        # FAILED or SAFETY_STOP — fill in what we have
        summary.update({
            "steps_completed": step,
            "nan_inf_count": nan_inf_count,
            "oom_count": oom_count,
            "total_tokens": total_tokens,
            "wall_time_s": round(time.time() - start_wall, 1),
            "moe_accepted": False,
        })

    summary_path = output_dir / "moe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[SUMMARY] Saved: {summary_path}")
    print(f"[SUMMARY] Outcome: {summary.get('outcome')} (exit {summary.get('exit_code')})")
    print(f"[SUMMARY] MoE Accepted: {summary.get('moe_accepted', False)}")

    if "moe_acceptance" in summary:
        acc = summary["moe_acceptance"]
        for k, v in acc.get("criteria", {}).items():
            icon = "OK" if v.get("pass") else "FAIL"
            print(f"  [{icon}] {k}")
        for err in acc.get("errors", []):
            print(f"  [!] {err}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop MoE CUDA Validation")
    parser.add_argument("--config", default="laptop_moe_8gb_safe",
                        help="Config name under training/configs/ (without .yaml)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--confirmed", action="store_true",
                        help="Required to run >100 steps")
    parser.add_argument(
        "--data-mode",
        choices=[DATA_MODE_REAL, DATA_MODE_SYNTHETIC, DATA_MODE_AUTO],
        default=DATA_MODE_AUTO,
        help=(
            "real: Wikitext-2 only (halts if unavailable). "
            "synthetic: random tokens (diagnostic only, cannot produce PASS). "
            "auto: tries real, falls back to synthetic."
        ),
    )
    # Legacy flag kept for backward compatibility with run_all_laptop_validation.bat
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
            "Embedded in moe_summary.json for artifact traceability. "
            "Generated automatically if not provided."
        ),
    )
    args = parser.parse_args()

    # Resolve --synthetic alias
    if args.synthetic and args.data_mode == DATA_MODE_AUTO:
        args.data_mode = DATA_MODE_SYNTHETIC

    if args.steps > 100 and not args.confirmed:
        print(f"[CONFIRM] Running {args.steps} steps requires --confirmed flag.")
        return EXIT_EXECUTION_ERROR

    import datetime as _dt
    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    print(f"[MOE] Run ID: {run_id}", flush=True)

    try:
        summary = run_moe_validation(
            config_name=args.config,
            max_steps=args.steps,
            data_mode=args.data_mode,
            thermal_warn=args.thermal_warn,
            thermal_stop=args.thermal_stop,
            output_dir=Path(args.output_dir),
        )
        # Embed run_id into the summary artifact for pipeline artifact validation
        summary["run_id"] = run_id
        import json as _json
        summary_path = Path(args.output_dir) / "moe_summary.json"
        if summary_path.exists():
            existing = _json.loads(summary_path.read_text())
            existing["run_id"] = run_id
            summary_path.write_text(_json.dumps(existing, indent=2, default=str))
        return int(summary.get("exit_code", EXIT_EXECUTION_ERROR))
    except SystemExit as e:
        # Thermal stop propagated as SystemExit
        print(f"\n[SAFETY STOP] Exiting with code {EXIT_SAFETY_STOP}.")
        return EXIT_SAFETY_STOP
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[FAIL] {e}")
        return EXIT_EXECUTION_ERROR


if __name__ == "__main__":
    sys.exit(main())
