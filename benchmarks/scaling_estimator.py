"""
Jupiter Shot — Scaling and Cost Estimator
==========================================
Estimates training FLOPs, duration, GPU hours, cost, and memory requirements
for dense and sparse MoE language models.

Methodology:
  - FLOPs: Chinchilla approximation C ≈ 6 × N_active × D
  - Memory: Detailed breakdown for weights, gradients, optimizer states, activations
  - Throughput: Based on measured or estimated tokens/sec
  - Cost: Based on GPU hourly pricing

All projections are estimates. Replace with measured values from Gate C.

Usage:
    python benchmarks/scaling_estimator.py --help
    python benchmarks/scaling_estimator.py --preset all
    python benchmarks/scaling_estimator.py \
        --total-params 1.3e9 --active-params 1.3e9 \
        --training-tokens 26e9 --num-gpus 8 \
        --gpu-type a100_80gb --measured-tokens-per-sec 35000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional


# ── GPU Specifications ────────────────────────────────────────────────────────

GPU_SPECS = {
    "a100_40gb": {
        "name": "NVIDIA A100 40GB",
        "memory_gb": 40,
        "bf16_tflops": 312,
        "fp16_tflops": 312,
        "fp32_tflops": 19.5,
        "nvlink_bandwidth_gbps": 600,
        "spot_cost_usd_per_hr": 1.10,
        "ondemand_cost_usd_per_hr": 2.21,
    },
    "a100_80gb": {
        "name": "NVIDIA A100 80GB",
        "memory_gb": 80,
        "bf16_tflops": 312,
        "fp16_tflops": 312,
        "fp32_tflops": 19.5,
        "nvlink_bandwidth_gbps": 600,
        "spot_cost_usd_per_hr": 1.89,
        "ondemand_cost_usd_per_hr": 3.78,
    },
    "h100_sxm5": {
        "name": "NVIDIA H100 SXM5 80GB",
        "memory_gb": 80,
        "bf16_tflops": 989,
        "fp16_tflops": 989,
        "fp32_tflops": 67,
        "nvlink_bandwidth_gbps": 900,
        "spot_cost_usd_per_hr": 2.80,
        "ondemand_cost_usd_per_hr": 3.50,
    },
    "h200_sxm": {
        "name": "NVIDIA H200 SXM 141GB",
        "memory_gb": 141,
        "bf16_tflops": 1979,
        "fp16_tflops": 1979,
        "fp32_tflops": 134,
        "nvlink_bandwidth_gbps": 900,
        "spot_cost_usd_per_hr": 4.50,
        "ondemand_cost_usd_per_hr": 6.00,
    },
}


# ── Scaling Presets ───────────────────────────────────────────────────────────

PRESETS = {
    "1.3b_dense": {
        "name": "1.3B Dense (Stage 1)",
        "total_params": 1.3e9,
        "active_params": 1.3e9,
        "num_experts": 1,
        "experts_per_token": 1,
        "training_tokens": 26e9,
        "num_gpus": 8,
        "gpu_type": "a100_80gb",
        "measured_tokens_per_sec": None,  # Replace with Gate C measurement
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "zero2",
        "hidden_size": 2048,
        "num_layers": 24,
        "seq_length": 2048,
        "batch_size_per_gpu": 4,
        "grad_accum": 8,
    },
    "1b_moe_prototype": {
        "name": "1B–3B MoE Prototype (Stage 2)",
        "total_params": 2.0e9,
        "active_params": 500e6,
        "num_experts": 8,
        "experts_per_token": 2,
        "training_tokens": 10e9,
        "num_gpus": 8,
        "gpu_type": "a100_80gb",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "zero2",
        "hidden_size": 1024,
        "num_layers": 12,
        "seq_length": 2048,
        "batch_size_per_gpu": 4,
        "grad_accum": 8,
    },
    "30b_moe": {
        "name": "30B Total MoE (Stage 3)",
        "total_params": 30e9,
        "active_params": 4e9,
        "num_experts": 32,
        "experts_per_token": 2,
        "training_tokens": 80e9,
        "num_gpus": 64,
        "gpu_type": "h100_sxm5",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert",
        "hidden_size": 4096,
        "num_layers": 32,
        "seq_length": 4096,
        "batch_size_per_gpu": 2,
        "grad_accum": 16,
    },
    "200b_moe": {
        "name": "200B Total / ~20B Active MoE (Stage 4)",
        "total_params": 200e9,
        "active_params": 20e9,
        "num_experts": 64,
        "experts_per_token": 2,
        "training_tokens": 400e9,
        "num_gpus": 512,
        "gpu_type": "h100_sxm5",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert",
        "hidden_size": 8192,
        "num_layers": 64,
        "seq_length": 4096,
        "batch_size_per_gpu": 1,
        "grad_accum": 32,
    },
    "500b_moe": {
        "name": "500B Total / ~30B–50B Active MoE (Stage 4+)",
        "total_params": 500e9,
        "active_params": 40e9,
        "num_experts": 128,
        "experts_per_token": 2,
        "training_tokens": 800e9,
        "num_gpus": 1024,
        "gpu_type": "h100_sxm5",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert",
        "hidden_size": 8192,
        "num_layers": 80,
        "seq_length": 4096,
        "batch_size_per_gpu": 1,
        "grad_accum": 64,
    },
    "1t_moe": {
        "name": "1T Total Sparse MoE (Stage 5)",
        "total_params": 1e12,
        "active_params": 75e9,
        "num_experts": 256,
        "experts_per_token": 2,
        "training_tokens": 1.5e12,
        "num_gpus": 4096,
        "gpu_type": "h100_sxm5",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert+data",
        "hidden_size": 12288,
        "num_layers": 96,
        "seq_length": 8192,
        "batch_size_per_gpu": 1,
        "grad_accum": 128,
    },
    "5t_moe": {
        "name": "5T Total Sparse MoE (Stage 5+)",
        "total_params": 5e12,
        "active_params": 150e9,
        "num_experts": 512,
        "experts_per_token": 2,
        "training_tokens": 3e12,
        "num_gpus": 16384,
        "gpu_type": "h200_sxm",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert+data",
        "hidden_size": 16384,
        "num_layers": 128,
        "seq_length": 8192,
        "batch_size_per_gpu": 1,
        "grad_accum": 256,
    },
    "20t_moe_scenario_a": {
        "name": "20T Total Sparse MoE — Scenario A: 200B Active (Stage 6)",
        "total_params": 20e12,
        "active_params": 200e9,
        "num_experts": 1000,
        "experts_per_token": 2,
        "training_tokens": 4e12,
        "num_gpus": 65536,
        "gpu_type": "h200_sxm",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert+data+hierarchical",
        "hidden_size": 16384,
        "num_layers": 160,
        "seq_length": 8192,
        "batch_size_per_gpu": 1,
        "grad_accum": 512,
    },
    "20t_moe_scenario_b": {
        "name": "20T Total Sparse MoE — Scenario B: 500B Active (Stage 6)",
        "total_params": 20e12,
        "active_params": 500e9,
        "num_experts": 1000,
        "experts_per_token": 5,
        "training_tokens": 10e12,
        "num_gpus": 65536,
        "gpu_type": "h200_sxm",
        "measured_tokens_per_sec": None,
        "checkpoint_precision": "bf16",
        "optimizer": "adamw",
        "parallelism": "tensor+pipeline+expert+data+hierarchical",
        "hidden_size": 16384,
        "num_layers": 160,
        "seq_length": 8192,
        "batch_size_per_gpu": 1,
        "grad_accum": 512,
    },
}


# ── Estimation Functions ──────────────────────────────────────────────────────

def estimate_training_flops(active_params: float, training_tokens: float) -> float:
    """
    Estimate total training FLOPs using Chinchilla approximation.
    C ≈ 6 × N_active × D

    For MoE models, N_active is the active parameters per token (not total).
    """
    return 6.0 * active_params * training_tokens


def estimate_training_duration(
    total_flops: float,
    num_gpus: int,
    gpu_type: str,
    measured_tokens_per_sec: Optional[float] = None,
    training_tokens: Optional[float] = None,
    mfu: float = 0.40,
) -> dict:
    """
    Estimate training duration.

    Args:
        total_flops: Total training FLOPs.
        num_gpus: Number of GPUs.
        gpu_type: GPU type key in GPU_SPECS.
        measured_tokens_per_sec: Measured throughput (tokens/sec across all GPUs).
                                  If provided, used directly.
        training_tokens: Total training tokens (for throughput-based estimate).
        mfu: Model FLOPs utilization (fraction of peak GPU FLOP/s actually used).
             Typical: 0.35–0.45 for well-optimized training.

    Returns:
        Dict with duration estimates.
    """
    gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["a100_80gb"])
    peak_tflops_per_gpu = gpu["bf16_tflops"]
    total_peak_tflops = peak_tflops_per_gpu * num_gpus * 1e12  # Convert to FLOP/s

    # FLOPs-based estimate
    effective_flops_per_sec = total_peak_tflops * mfu
    duration_seconds_flops = total_flops / effective_flops_per_sec

    # Throughput-based estimate (more accurate if measured)
    duration_seconds_throughput = None
    if measured_tokens_per_sec and training_tokens:
        duration_seconds_throughput = training_tokens / measured_tokens_per_sec

    return {
        "duration_seconds_flops_estimate": duration_seconds_flops,
        "duration_hours_flops_estimate": duration_seconds_flops / 3600,
        "duration_days_flops_estimate": duration_seconds_flops / 86400,
        "duration_seconds_throughput": duration_seconds_throughput,
        "duration_hours_throughput": duration_seconds_throughput / 3600 if duration_seconds_throughput else None,
        "duration_days_throughput": duration_seconds_throughput / 86400 if duration_seconds_throughput else None,
        "assumed_mfu": mfu,
        "peak_bf16_tflops_total": total_peak_tflops / 1e12,
        "effective_tflops_total": effective_flops_per_sec / 1e12,
    }


def estimate_gpu_hours_and_cost(
    duration_hours: float,
    num_gpus: int,
    gpu_type: str,
    use_spot: bool = True,
) -> dict:
    """Estimate GPU hours and compute cost."""
    gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["a100_80gb"])
    cost_per_gpu_hr = gpu["spot_cost_usd_per_hr"] if use_spot else gpu["ondemand_cost_usd_per_hr"]

    gpu_hours = duration_hours * num_gpus
    total_cost = gpu_hours * cost_per_gpu_hr

    return {
        "gpu_hours": gpu_hours,
        "cost_usd": total_cost,
        "cost_per_gpu_hr_usd": cost_per_gpu_hr,
        "pricing_type": "spot" if use_spot else "on-demand",
        "gpu_type": gpu["name"],
    }


def estimate_memory(
    total_params: float,
    active_params: float,
    num_gpus: int,
    optimizer: str = "adamw",
    parallelism: str = "zero2",
    checkpoint_precision: str = "bf16",
    seq_length: int = 2048,
    batch_size_per_gpu: int = 4,
    num_layers: int = 24,
    hidden_size: int = 2048,
) -> dict:
    """
    Estimate GPU memory requirements.

    Returns detailed breakdown of memory components.
    """
    bytes_per_param = 2 if checkpoint_precision == "bf16" else 4

    # Model weights
    weight_bytes = total_params * bytes_per_param

    # Gradients (same precision as weights)
    gradient_bytes = total_params * bytes_per_param

    # Optimizer states
    if optimizer == "adamw":
        # FP32 copy + m (FP32) + v (FP32) = 12 bytes per param
        optimizer_bytes = total_params * 12
    elif optimizer == "adam8bit":
        # 8-bit Adam: ~4 bytes per param
        optimizer_bytes = total_params * 4
    else:
        optimizer_bytes = total_params * 8

    # ZeRO partitioning
    if "zero3" in parallelism:
        # ZeRO-3: weights + gradients + optimizer all sharded
        per_gpu_weight = weight_bytes / num_gpus
        per_gpu_gradient = gradient_bytes / num_gpus
        per_gpu_optimizer = optimizer_bytes / num_gpus
    elif "zero2" in parallelism:
        # ZeRO-2: optimizer + gradients sharded, weights replicated
        per_gpu_weight = weight_bytes
        per_gpu_gradient = gradient_bytes / num_gpus
        per_gpu_optimizer = optimizer_bytes / num_gpus
    else:
        # DDP: everything replicated
        per_gpu_weight = weight_bytes
        per_gpu_gradient = gradient_bytes
        per_gpu_optimizer = optimizer_bytes

    # Activation memory (with gradient checkpointing)
    # Without checkpointing: ~12 × seq_length × hidden_size × num_layers × batch_size × 2 bytes
    # With checkpointing: ~sqrt(num_layers) × seq_length × hidden_size × batch_size × 2 bytes
    activation_bytes_no_ckpt = (
        12 * seq_length * hidden_size * num_layers * batch_size_per_gpu * 2
    )
    activation_bytes_with_ckpt = (
        math.sqrt(num_layers) * seq_length * hidden_size * batch_size_per_gpu * 2 * 4
    )

    total_per_gpu_no_ckpt = (
        per_gpu_weight + per_gpu_gradient + per_gpu_optimizer + activation_bytes_no_ckpt
    )
    total_per_gpu_with_ckpt = (
        per_gpu_weight + per_gpu_gradient + per_gpu_optimizer + activation_bytes_with_ckpt
    )

    def gb(b):
        return b / 1e9

    return {
        "weight_gb_total": gb(weight_bytes),
        "weight_gb_per_gpu": gb(per_gpu_weight),
        "gradient_gb_per_gpu": gb(per_gpu_gradient),
        "optimizer_gb_per_gpu": gb(per_gpu_optimizer),
        "activation_gb_per_gpu_no_checkpointing": gb(activation_bytes_no_ckpt),
        "activation_gb_per_gpu_with_checkpointing": gb(activation_bytes_with_ckpt),
        "total_gb_per_gpu_no_checkpointing": gb(total_per_gpu_no_ckpt),
        "total_gb_per_gpu_with_checkpointing": gb(total_per_gpu_with_ckpt),
        "checkpoint_size_gb": gb(weight_bytes),
        "inference_memory_gb": gb(total_params * bytes_per_param),
        "parallelism_strategy": parallelism,
    }


def generate_feasibility_warnings(preset: dict, memory: dict, gpu_type: str) -> list[str]:
    """Generate feasibility warnings based on memory and hardware requirements."""
    warnings = []
    gpu = GPU_SPECS.get(gpu_type, GPU_SPECS["a100_80gb"])
    gpu_memory_gb = gpu["memory_gb"]

    mem_with_ckpt = memory["total_gb_per_gpu_with_checkpointing"]
    mem_no_ckpt = memory["total_gb_per_gpu_no_checkpointing"]

    if mem_with_ckpt > gpu_memory_gb:
        warnings.append(
            f"CRITICAL: Estimated memory per GPU ({mem_with_ckpt:.0f} GB) exceeds "
            f"{gpu['name']} capacity ({gpu_memory_gb} GB) even with gradient checkpointing. "
            "Requires more GPUs, ZeRO-3, or pipeline parallelism."
        )
    elif mem_no_ckpt > gpu_memory_gb:
        warnings.append(
            f"WARNING: Memory per GPU without checkpointing ({mem_no_ckpt:.0f} GB) exceeds "
            f"{gpu['name']} capacity ({gpu_memory_gb} GB). "
            "Gradient checkpointing is REQUIRED."
        )
    elif mem_with_ckpt > gpu_memory_gb * 0.8:
        warnings.append(
            f"CAUTION: Memory per GPU ({mem_with_ckpt:.0f} GB) is close to "
            f"{gpu['name']} capacity ({gpu_memory_gb} GB). "
            "Reduce batch size if OOM errors occur."
        )

    num_gpus = preset.get("num_gpus", 8)
    if num_gpus > 8 and "infiniband" not in preset.get("parallelism", "").lower():
        warnings.append(
            f"WARNING: {num_gpus} GPUs requires multi-node training. "
            "InfiniBand (200+ Gbps) is required for efficient expert all-to-all communication. "
            "NVLink alone is insufficient for multi-node MoE."
        )

    if preset.get("total_params", 0) > 10e9 and preset.get("num_gpus", 8) <= 8:
        warnings.append(
            f"CRITICAL: {preset['total_params']/1e9:.0f}B parameter model cannot be trained "
            f"on only {num_gpus} GPUs. Minimum GPU count for this scale is significantly higher."
        )

    return warnings


def estimate_preset(preset_name: str, use_spot: bool = True) -> dict:
    """Run full estimation for a named preset."""
    preset = PRESETS[preset_name]

    total_flops = estimate_training_flops(
        preset["active_params"], preset["training_tokens"]
    )

    duration = estimate_training_duration(
        total_flops=total_flops,
        num_gpus=preset["num_gpus"],
        gpu_type=preset["gpu_type"],
        measured_tokens_per_sec=preset.get("measured_tokens_per_sec"),
        training_tokens=preset["training_tokens"],
    )

    # Use throughput-based duration if available, else FLOPs-based
    duration_hours = (
        duration["duration_hours_throughput"]
        if duration["duration_hours_throughput"]
        else duration["duration_hours_flops_estimate"]
    )

    cost = estimate_gpu_hours_and_cost(
        duration_hours=duration_hours,
        num_gpus=preset["num_gpus"],
        gpu_type=preset["gpu_type"],
        use_spot=use_spot,
    )

    memory = estimate_memory(
        total_params=preset["total_params"],
        active_params=preset["active_params"],
        num_gpus=preset["num_gpus"],
        optimizer=preset.get("optimizer", "adamw"),
        parallelism=preset.get("parallelism", "zero2"),
        checkpoint_precision=preset.get("checkpoint_precision", "bf16"),
        seq_length=preset.get("seq_length", 2048),
        batch_size_per_gpu=preset.get("batch_size_per_gpu", 4),
        num_layers=preset.get("num_layers", 24),
        hidden_size=preset.get("hidden_size", 2048),
    )

    warnings = generate_feasibility_warnings(preset, memory, preset["gpu_type"])

    return {
        "preset_name": preset_name,
        "description": preset["name"],
        "total_params_b": preset["total_params"] / 1e9,
        "active_params_b": preset["active_params"] / 1e9,
        "num_experts": preset.get("num_experts", 1),
        "experts_per_token": preset.get("experts_per_token", 1),
        "activation_ratio_pct": preset["active_params"] / preset["total_params"] * 100,
        "training_tokens_b": preset["training_tokens"] / 1e9,
        "training_flops": total_flops,
        "training_flops_scientific": f"{total_flops:.2e}",
        "num_gpus": preset["num_gpus"],
        "gpu_type": GPU_SPECS[preset["gpu_type"]]["name"],
        "duration": duration,
        "cost": cost,
        "memory": memory,
        "feasibility_warnings": warnings,
        "note": (
            "All estimates are projections based on scaling laws and GPU pricing. "
            "Replace measured_tokens_per_sec with Gate C results for accurate estimates."
            if not preset.get("measured_tokens_per_sec")
            else "Duration based on measured throughput."
        ),
    }


def print_summary(result: dict) -> None:
    """Print a human-readable summary of scaling estimates."""
    print(f"\n{'='*70}")
    print(f"  {result['description']}")
    print(f"{'='*70}")
    print(f"  Total parameters:    {result['total_params_b']:.1f}B")
    print(f"  Active per token:    {result['active_params_b']:.1f}B ({result['activation_ratio_pct']:.1f}%)")
    print(f"  Experts:             {result['num_experts']} total, top-{result['experts_per_token']}")
    print(f"  Training tokens:     {result['training_tokens_b']:.0f}B")
    print(f"  Training FLOPs:      {result['training_flops_scientific']}")
    print(f"  Hardware:            {result['num_gpus']}× {result['gpu_type']}")
    print()

    d = result["duration"]
    if d["duration_hours_throughput"]:
        print(f"  Duration (measured): {d['duration_hours_throughput']:.1f} hours ({d['duration_days_throughput']:.1f} days)")
    else:
        print(f"  Duration (est.):     {d['duration_hours_flops_estimate']:.1f} hours ({d['duration_days_flops_estimate']:.1f} days)")
        print(f"  (Assumes MFU={d['assumed_mfu']:.0%}; replace with measured throughput)")

    c = result["cost"]
    print(f"  GPU hours:           {c['gpu_hours']:.0f}")
    print(f"  Estimated cost:      ${c['cost_usd']:,.0f} ({c['pricing_type']} @ ${c['cost_per_gpu_hr_usd']:.2f}/GPU-hr)")
    print()

    m = result["memory"]
    print(f"  Weight memory:       {m['weight_gb_total']:.1f} GB total, {m['weight_gb_per_gpu']:.1f} GB/GPU")
    print(f"  Optimizer memory:    {m['optimizer_gb_per_gpu']:.1f} GB/GPU ({result['memory']['parallelism_strategy']})")
    print(f"  Total/GPU (w/ ckpt): {m['total_gb_per_gpu_with_checkpointing']:.1f} GB")
    print(f"  Checkpoint size:     {m['checkpoint_size_gb']:.1f} GB")
    print(f"  Inference memory:    {m['inference_memory_gb']:.1f} GB")

    if result["feasibility_warnings"]:
        print()
        for w in result["feasibility_warnings"]:
            print(f"  ⚠  {w}")

    print(f"\n  Note: {result['note']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jupiter Shot — Scaling and Cost Estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()) + ["all"],
        default=None,
        help="Named preset to estimate",
    )
    parser.add_argument("--total-params", type=float, help="Total parameters")
    parser.add_argument("--active-params", type=float, help="Active parameters per token")
    parser.add_argument("--training-tokens", type=float, help="Training token count")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-type", default="a100_80gb", choices=list(GPU_SPECS.keys()))
    parser.add_argument("--measured-tokens-per-sec", type=float, default=None)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam8bit"])
    parser.add_argument("--parallelism", default="zero2")
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--use-spot", action="store_true", default=True)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    results = []

    if args.preset == "all":
        for preset_name in PRESETS:
            result = estimate_preset(preset_name, use_spot=args.use_spot)
            print_summary(result)
            results.append(result)
    elif args.preset:
        result = estimate_preset(args.preset, use_spot=args.use_spot)
        print_summary(result)
        results.append(result)
    elif args.total_params and args.active_params and args.training_tokens:
        # Custom estimation
        total_flops = estimate_training_flops(args.active_params, args.training_tokens)
        duration = estimate_training_duration(
            total_flops=total_flops,
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            measured_tokens_per_sec=args.measured_tokens_per_sec,
            training_tokens=args.training_tokens,
        )
        duration_hours = (
            duration["duration_hours_throughput"]
            if duration["duration_hours_throughput"]
            else duration["duration_hours_flops_estimate"]
        )
        cost = estimate_gpu_hours_and_cost(duration_hours, args.num_gpus, args.gpu_type, args.use_spot)
        memory = estimate_memory(
            total_params=args.total_params,
            active_params=args.active_params,
            num_gpus=args.num_gpus,
            optimizer=args.optimizer,
            parallelism=args.parallelism,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            batch_size_per_gpu=args.batch_size_per_gpu,
        )
        result = {
            "preset_name": "custom",
            "description": "Custom Configuration",
            "total_params_b": args.total_params / 1e9,
            "active_params_b": args.active_params / 1e9,
            "num_experts": 1,
            "experts_per_token": 1,
            "activation_ratio_pct": args.active_params / args.total_params * 100,
            "training_tokens_b": args.training_tokens / 1e9,
            "training_flops": total_flops,
            "training_flops_scientific": f"{total_flops:.2e}",
            "num_gpus": args.num_gpus,
            "gpu_type": GPU_SPECS[args.gpu_type]["name"],
            "duration": duration,
            "cost": cost,
            "memory": memory,
            "feasibility_warnings": generate_feasibility_warnings(
                {"total_params": args.total_params, "num_gpus": args.num_gpus, "parallelism": args.parallelism},
                memory, args.gpu_type
            ),
            "note": "Custom configuration.",
        }
        print_summary(result)
        results.append(result)
    else:
        parser.print_help()
        print("\nRun with --preset all to see all presets.")

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")
