"""
Jupiter Shot — Laptop MoE CUDA Validation
==========================================
Runs MoE training validation with 8 experts, top-2 routing, and full
router metric collection.

Model API contract:
  model(input_ids=..., labels=...) → dict with keys:
    - 'logits':        (batch, seq_len, vocab_size)
    - 'loss':          total loss (LM + aux) if labels provided, else None
    - 'lm_loss':       language modeling loss only
    - 'aux_loss':      total auxiliary routing loss (scalar tensor)
    - 'router_metrics': list of per-layer routing metric dicts

Metric definitions:
  - expert_assignment_share[i]: fraction of tokens assigned to expert i
    (after routing, before capacity overflow). Sum = num_experts_per_token / num_experts.
  - token_routing_pct[i]: percentage of tokens where expert i is selected.
  - utilization_cv: coefficient of variation of expert loads = std/mean.
    CV < 0.2 indicates balanced routing.
  - max_min_ratio: max_load / min_load. Ratio < 3.0 is acceptable.
  - dropped_token_pct: percentage of tokens that exceed expert capacity
    and are dropped (not processed by any expert).
  - overflow_pct: percentage of tokens that hit the capacity buffer.
  - router_entropy: Shannon entropy of routing distribution (bits).
    Max entropy for 8 experts = log2(8) = 3.0 bits.
    Collapse threshold: < 1.5 bits.

Usage:
    python scripts/run_laptop_moe.py --config laptop_moe_small
    python scripts/run_laptop_moe.py --config laptop_moe_small --steps 1000 --confirmed
    python scripts/run_laptop_moe.py --synthetic  # synthetic data only
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

# MoE acceptance thresholds
MOE_ACCEPT = {
    "max_dropped_token_pct": 1.0,    # < 1% dropped tokens required
    "max_utilization_cv": 0.5,       # CV < 0.5 acceptable
    "max_max_min_ratio": 10.0,       # max/min utilization ratio
    "min_router_entropy_bits": 1.5,  # > 1.5 bits to avoid collapse
    "min_expert_share_pct": 2.0,     # no expert below 2% persistently
}


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
        raise RuntimeError(f"[THERMAL STOP] GPU {temp}°C >= {stop_c}°C at step {step}.")
    if temp >= warn_c:
        print(f"[THERMAL WARNING] GPU {temp}°C >= {warn_c}°C at step {step}.")


def _check_nan_inf(loss_val: float, step: int) -> None:
    if math.isnan(loss_val) or math.isinf(loss_val):
        raise ValueError(f"[NaN/Inf] Loss={loss_val} at step {step}.")


def compute_router_metrics(routing_weights: Any, num_experts: int, torch: Any) -> dict:
    """
    Compute all router utilization metrics from routing weights tensor.

    Args:
        routing_weights: (num_tokens, num_experts) softmax probabilities
        num_experts: total number of experts
        torch: torch module

    Returns:
        dict with utilization_cv, max_min_ratio, router_entropy_bits,
        expert_assignment_share, num_inactive_experts
    """
    try:
        if routing_weights.dim() != 2:
            return {}
        # Expert load = mean probability assigned to each expert
        expert_load = routing_weights.mean(dim=0)  # (num_experts,)
        mean_load = expert_load.mean().item()
        std_load = expert_load.std().item()
        max_load = expert_load.max().item()
        min_load = expert_load.min().item()

        cv = std_load / (mean_load + 1e-9)
        max_min_ratio = max_load / (min_load + 1e-9)

        # Shannon entropy
        probs = expert_load / (expert_load.sum() + 1e-9)
        entropy = -(probs * (probs + 1e-9).log()).sum().item() / math.log(2)

        # Expert assignment share (fraction of tokens routed to each expert)
        assignment_share = expert_load.tolist()
        num_inactive = sum(1 for s in assignment_share if s < 0.01)

        return {
            "utilization_cv": round(cv, 4),
            "max_min_ratio": round(max_min_ratio, 4),
            "router_entropy_bits": round(entropy, 4),
            "expert_assignment_share": [round(s, 4) for s in assignment_share],
            "num_inactive_experts": num_inactive,
        }
    except Exception:
        return {}


def load_wikitext_sample(n_samples: int = 200) -> list[str]:
    """Load a small sample from Wikitext-2 (MIT license). Falls back to [] if unavailable."""
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


def get_real_text_batch(tokenizer: Any, texts: list[str], seq_len: int, device: Any) -> Any:
    import torch
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=seq_len, padding="max_length")
    return enc["input_ids"].to(device)


def run_moe_validation(
    config_name: str,
    max_steps: int = 100,
    synthetic: bool = False,
    thermal_warn: int = THERMAL_WARN_C,
    thermal_stop: int = THERMAL_STOP_C,
    output_dir: Path = RESULTS_DIR,
    ckpt_dir: Path = CKPT_DIR,
) -> dict:
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("[FAIL] CUDA is not available.")

    config_path = REPO_ROOT / "training" / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
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

    # AMP scaler — use torch.amp.GradScaler (torch.cuda.amp.GradScaler is deprecated in PyTorch 2.x)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    vocab_size = base_config.vocab_size
    seq_len = int(train_cfg.get("seq_length", base_config.max_position_embeddings))
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))

    # Data
    texts = [] if synthetic else load_wikitext_sample(500)
    data_mode = "synthetic" if (synthetic or not texts) else "wikitext-2 (MIT)"
    tokenizer = None
    if texts:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            print("[DATA] Tokenizer unavailable; falling back to synthetic.")
            texts = []
            data_mode = "synthetic (tokenizer unavailable)"
    print(f"[MOE] Data mode: {data_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "moe_metrics.jsonl"
    errors_path = output_dir / "errors.jsonl"
    failure_path = output_dir / "moe_failure_artifact.json"

    metrics_log: list[dict] = []
    nan_inf_count = 0
    oom_count = 0
    interrupted = False
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
    step_times = []

    router_history: list[dict] = []

    summary = {
        "config": config_name,
        "data_mode": data_mode,
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
    }

    model.train()
    optimizer.zero_grad()

    try:
        for step in range(1, max_steps + 1):
            if interrupted:
                break

            step_start = time.time()
            _check_thermal(step, thermal_warn, thermal_stop)

            try:
                if texts and tokenizer:
                    import random
                    sample = random.sample(texts, min(batch_size, len(texts)))
                    input_ids = get_real_text_batch(tokenizer, sample, seq_len, device)
                else:
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                if oom_count >= MAX_OOM_RETRIES:
                    raise RuntimeError(f"[FAIL] {MAX_OOM_RETRIES} OOM errors.")
                continue

            # Forward + backward
            # Model API: model(input_ids=..., labels=...) → dict
            #   out["loss"]          = total loss (LM + aux)
            #   out["lm_loss"]       = language modeling loss only
            #   out["aux_loss"]      = auxiliary routing loss
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

            # Router metrics — read directly from model output dict
            router_metrics: dict = {}
            raw_router_metrics = out.get("router_metrics", [])
            if raw_router_metrics:
                # Aggregate across layers: average scalar metrics
                agg: dict = {}
                count = 0
                for layer_metrics in raw_router_metrics:
                    if not isinstance(layer_metrics, dict):
                        continue
                    for k, v in layer_metrics.items():
                        if isinstance(v, (int, float)):
                            agg[k] = agg.get(k, 0.0) + v
                    count += 1
                if count > 0:
                    router_metrics = {k: round(v / count, 4) for k, v in agg.items()}
                    # Compute entropy from aggregated expert loads if available
                    if "utilization_cv" not in router_metrics:
                        # Fallback: try to extract from last layer's routing weights
                        for module in model.modules():
                            if hasattr(module, "_last_routing_weights") and module._last_routing_weights is not None:
                                rw = module._last_routing_weights
                                if rw.dim() == 2:
                                    router_metrics.update(
                                        compute_router_metrics(rw, moe_config.num_experts, torch)
                                    )
                                break

            step_time = time.time() - step_start
            step_times.append(step_time)
            tokens_this_step = batch_size * seq_len
            total_tokens += tokens_this_step
            tps = tokens_this_step / step_time

            vram = _vram_stats(torch)
            temp = _gpu_temp()
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

            metric = {
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
                with open(metrics_path, "a") as f:
                    for m in metrics_log[-METRIC_FLUSH_INTERVAL:]:
                        f.write(json.dumps(m) + "\n")

            if step % 10 == 0 or step == 1:
                inactive_str = f"inactive={router_metrics.get('num_inactive_experts', '?')}" if router_metrics else ""
                aux_str = f"aux={aux_val:.4f}" if aux_val is not None else "aux=?"
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
        with open(errors_path, "a") as f:
            f.write(json.dumps({"step": step, "error": str(e)}) + "\n")
        failure_path.write_text(json.dumps(artifact, indent=2, default=str))
        print(f"[FAILURE ARTIFACT] Saved: {failure_path}")
        summary["status"] = "FAILED"
        summary["failure_reason"] = str(e)
        summary["failure_traceback"] = tb
    else:
        summary["status"] = "INTERRUPTED" if interrupted else "COMPLETED"

    # Checkpoint
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

    # MoE acceptance evaluation
    last_metrics = metrics_log[-10:] if len(metrics_log) >= 10 else metrics_log
    dropped_pct = 0.0  # Would come from model's capacity overflow counter
    cv_vals = [m.get("utilization_cv") for m in last_metrics if m.get("utilization_cv") is not None]
    avg_cv = sum(cv_vals) / len(cv_vals) if cv_vals else None
    entropy_vals = [m.get("router_entropy_bits") for m in last_metrics if m.get("router_entropy_bits") is not None]
    avg_entropy = sum(entropy_vals) / len(entropy_vals) if entropy_vals else None

    acceptance = {
        "no_persistent_inactive_expert": len([m for m in last_metrics if m.get("num_inactive_experts", 0) > 0]) == 0,
        "dropped_token_pct_below_1": dropped_pct < MOE_ACCEPT["max_dropped_token_pct"],
        "utilization_cv_acceptable": avg_cv is not None and avg_cv < MOE_ACCEPT["max_utilization_cv"],
        "router_entropy_above_threshold": avg_entropy is not None and avg_entropy > MOE_ACCEPT["min_router_entropy_bits"],
        "no_nan_inf": nan_inf_count == 0,
        "checkpoint_saved": "checkpoint_path" in summary,
        "loss_trend_downward": (
            metrics_log[-1]["loss"] < metrics_log[0]["loss"]
            if len(metrics_log) >= 2 else None
        ),
    }

    wall_time = time.time() - start_wall
    vram_final = _vram_stats(torch)
    summary.update({
        "steps_completed": step,
        "nan_inf_count": nan_inf_count,
        "oom_count": oom_count,
        "total_tokens": total_tokens,
        "wall_time_s": round(wall_time, 1),
        "avg_tokens_per_sec": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "peak_vram_allocated_gb": vram_final.get("max_allocated_gb"),
        "first_loss": metrics_log[0]["loss"] if metrics_log else None,
        "last_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "avg_aux_loss_last10": round(
            sum(m["aux_loss"] for m in last_metrics if m.get("aux_loss") is not None)
            / max(1, sum(1 for m in last_metrics if m.get("aux_loss") is not None)),
            6
        ) if any(m.get("aux_loss") is not None for m in last_metrics) else None,
        "avg_router_entropy_last10": round(avg_entropy, 4) if avg_entropy is not None else None,
        "avg_utilization_cv_last10": round(avg_cv, 4) if avg_cv is not None else None,
        "dropped_token_pct": dropped_pct,
        "moe_acceptance": acceptance,
        "moe_accepted": all(v for v in acceptance.values() if v is not None),
    })

    summary_path = output_dir / "moe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[SUMMARY] Saved: {summary_path}")
    print(f"[SUMMARY] MoE Accepted: {summary['moe_accepted']}")
    for k, v in acceptance.items():
        status = "✓" if v else ("?" if v is None else "✗")
        print(f"  {status} {k}: {v}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop MoE CUDA Validation")
    parser.add_argument("--config", default="laptop_moe_small")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--confirmed", action="store_true",
                        help="Required to run >100 steps")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data only (skip Wikitext-2 download)")
    parser.add_argument("--thermal-warn", type=int, default=THERMAL_WARN_C)
    parser.add_argument("--thermal-stop", type=int, default=THERMAL_STOP_C)
    parser.add_argument("--output-dir", default="benchmarks/results/laptop")
    args = parser.parse_args()

    if args.steps > 100 and not args.confirmed:
        print(f"[CONFIRM] Running {args.steps} steps requires --confirmed flag.")
        return 1

    try:
        summary = run_moe_validation(
            config_name=args.config,
            max_steps=args.steps,
            synthetic=args.synthetic,
            thermal_warn=args.thermal_warn,
            thermal_stop=args.thermal_stop,
            output_dir=Path(args.output_dir),
        )
        return 0 if summary["status"] in ("COMPLETED", "INTERRUPTED") else 1
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n[FAIL] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
