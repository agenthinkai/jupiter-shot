"""
Jupiter Shot — Multi-Stage Scaling Simulator

Simulates training cost, time, and hardware requirements for each stage
of the Jupiter Shot scaling roadmap. All estimates are based on:
- Chinchilla scaling laws (Hoffmann et al., 2022)
- Empirical MFU measurements from public literature
- 2024 H100/A100 pricing

Usage:
    python benchmarks/simulators/scaling_simulator.py
    python benchmarks/simulators/scaling_simulator.py --stage 2
    python benchmarks/simulators/scaling_simulator.py --gpu-price 20000
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware specifications
# ---------------------------------------------------------------------------

@dataclass
class GPUSpec:
    name: str
    bf16_tflops: float          # BF16 Tensor Core TFLOPS
    hbm_gb: float               # HBM capacity in GB
    hbm_bw_tbs: float           # HBM bandwidth in TB/s
    nvlink_bw_gbs: float        # NVLink bandwidth per GPU (GB/s, bidirectional)
    tdp_w: float                # Thermal design power in watts
    list_price_usd: float       # List price USD (2024)
    spot_price_per_hour: float  # Estimated spot/reserved price per GPU-hour


GPU_SPECS: dict[str, GPUSpec] = {
    "A100_80GB": GPUSpec(
        name="NVIDIA A100 80GB",
        bf16_tflops=312.0,
        hbm_gb=80.0,
        hbm_bw_tbs=2.0,
        nvlink_bw_gbs=600.0,
        tdp_w=400.0,
        list_price_usd=15000.0,
        spot_price_per_hour=2.0,
    ),
    "H100_80GB": GPUSpec(
        name="NVIDIA H100 80GB SXM",
        bf16_tflops=989.0,
        hbm_gb=80.0,
        hbm_bw_tbs=3.35,
        nvlink_bw_gbs=900.0,
        tdp_w=700.0,
        list_price_usd=30000.0,
        spot_price_per_hour=4.0,
    ),
    "H100_NVL": GPUSpec(
        name="NVIDIA H100 NVL 94GB",
        bf16_tflops=989.0,
        hbm_gb=94.0,
        hbm_bw_tbs=3.9,
        nvlink_bw_gbs=900.0,
        tdp_w=700.0,
        list_price_usd=35000.0,
        spot_price_per_hour=5.0,
    ),
    "H200": GPUSpec(
        name="NVIDIA H200 141GB",
        bf16_tflops=1979.0,
        hbm_gb=141.0,
        hbm_bw_tbs=4.8,
        nvlink_bw_gbs=900.0,
        tdp_w=700.0,
        list_price_usd=40000.0,
        spot_price_per_hour=6.0,
    ),
}


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    stage: int
    name: str
    total_params: int           # Total parameters
    active_params: int          # Active parameters per token (for MoE)
    is_moe: bool
    num_experts: int
    top_k: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    context_length: int
    training_tokens: int        # Target training tokens
    gpu_type: str               # Key into GPU_SPECS
    num_gpus: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    data_parallel: int
    target_mfu: float           # Target model FLOPs utilization (0-1)
    batch_size_tokens: int      # Global batch size in tokens
    gradient_checkpointing: bool
    prerequisite_stage: Optional[int] = None


STAGES: list[StageSpec] = [
    StageSpec(
        stage=1, name="1.3B Dense Baseline",
        total_params=1_298_761_728, active_params=1_298_761_728,
        is_moe=False, num_experts=1, top_k=1,
        hidden_size=2048, num_layers=24, num_attention_heads=16, num_kv_heads=16,
        context_length=2048, training_tokens=26_000_000_000,
        gpu_type="A100_80GB", num_gpus=8,
        tensor_parallel=1, pipeline_parallel=1, expert_parallel=1, data_parallel=8,
        target_mfu=0.45, batch_size_tokens=2_097_152,
        gradient_checkpointing=False, prerequisite_stage=None,
    ),
    StageSpec(
        stage=2, name="47B MoE Prototype",
        total_params=47_000_000_000, active_params=7_000_000_000,
        is_moe=True, num_experts=8, top_k=2,
        hidden_size=4096, num_layers=32, num_attention_heads=32, num_kv_heads=32,
        context_length=4096, training_tokens=100_000_000_000,
        gpu_type="A100_80GB", num_gpus=32,
        tensor_parallel=4, pipeline_parallel=2, expert_parallel=8, data_parallel=2,
        target_mfu=0.35, batch_size_tokens=4_194_304,
        gradient_checkpointing=True, prerequisite_stage=1,
    ),
    StageSpec(
        stage=3, name="200B MoE",
        total_params=200_000_000_000, active_params=25_000_000_000,
        is_moe=True, num_experts=16, top_k=2,
        hidden_size=6144, num_layers=48, num_attention_heads=48, num_kv_heads=8,
        context_length=8192, training_tokens=500_000_000_000,
        gpu_type="H100_80GB", num_gpus=128,
        tensor_parallel=8, pipeline_parallel=4, expert_parallel=16, data_parallel=2,
        target_mfu=0.40, batch_size_tokens=8_388_608,
        gradient_checkpointing=True, prerequisite_stage=2,
    ),
    StageSpec(
        stage=4, name="1T MoE",
        total_params=1_000_000_000_000, active_params=62_000_000_000,
        is_moe=True, num_experts=64, top_k=4,
        hidden_size=8192, num_layers=64, num_attention_heads=64, num_kv_heads=8,
        context_length=32768, training_tokens=2_000_000_000_000,
        gpu_type="H100_NVL", num_gpus=512,
        tensor_parallel=8, pipeline_parallel=8, expert_parallel=64, data_parallel=1,
        target_mfu=0.35, batch_size_tokens=16_777_216,
        gradient_checkpointing=True, prerequisite_stage=3,
    ),
    StageSpec(
        stage=5, name="5T MoE",
        total_params=5_000_000_000_000, active_params=156_000_000_000,
        is_moe=True, num_experts=128, top_k=4,
        hidden_size=12288, num_layers=80, num_attention_heads=96, num_kv_heads=8,
        context_length=65536, training_tokens=5_000_000_000_000,
        gpu_type="H100_NVL", num_gpus=2048,
        tensor_parallel=8, pipeline_parallel=16, expert_parallel=128, data_parallel=1,
        target_mfu=0.30, batch_size_tokens=33_554_432,
        gradient_checkpointing=True, prerequisite_stage=4,
    ),
    StageSpec(
        stage=6, name="20T MoE (Target)",
        total_params=20_000_000_000_000, active_params=625_000_000_000,
        is_moe=True, num_experts=512, top_k=4,
        hidden_size=16384, num_layers=96, num_attention_heads=128, num_kv_heads=8,
        context_length=131072, training_tokens=15_000_000_000_000,
        gpu_type="H100_NVL", num_gpus=8192,
        tensor_parallel=8, pipeline_parallel=32, expert_parallel=512, data_parallel=1,
        target_mfu=0.25, batch_size_tokens=67_108_864,
        gradient_checkpointing=True, prerequisite_stage=5,
    ),
]


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------

def compute_flops_per_token(spec: StageSpec) -> float:
    """
    Estimate FLOPs per token using the 6*N*D approximation for dense models,
    adjusted for MoE active parameters.

    For MoE: FLOPs ≈ 6 * active_params (only active experts are computed)
    plus router overhead (~2 * hidden_size * num_experts per token).
    """
    dense_flops = 6.0 * spec.active_params
    if spec.is_moe:
        router_flops = 2.0 * spec.hidden_size * spec.num_experts * spec.num_layers
        return dense_flops + router_flops
    return dense_flops


def compute_training_flops(spec: StageSpec) -> float:
    """Total FLOPs for the full training run."""
    return compute_flops_per_token(spec) * spec.training_tokens


def compute_peak_throughput(spec: StageSpec, gpu_spec: GPUSpec) -> float:
    """
    Peak tokens/second across all GPUs, accounting for MFU.
    throughput = (num_gpus * gpu_tflops * mfu) / flops_per_token
    """
    total_tflops = spec.num_gpus * gpu_spec.bf16_tflops * 1e12  # convert to FLOPS
    flops_per_token = compute_flops_per_token(spec)
    return total_tflops * spec.target_mfu / flops_per_token


def compute_training_duration_days(spec: StageSpec, gpu_spec: GPUSpec) -> float:
    """Estimated training duration in days."""
    throughput = compute_peak_throughput(spec, gpu_spec)
    seconds = spec.training_tokens / throughput
    return seconds / 86400.0


def compute_training_cost(spec: StageSpec, gpu_spec: GPUSpec) -> float:
    """Estimated training cost in USD (spot pricing)."""
    duration_hours = compute_training_duration_days(spec, gpu_spec) * 24.0
    return spec.num_gpus * gpu_spec.spot_price_per_hour * duration_hours


def compute_weight_memory_gb(spec: StageSpec) -> float:
    """BF16 weight memory in GB."""
    return spec.total_params * 2 / 1e9


def compute_active_weight_memory_gb(spec: StageSpec) -> float:
    """BF16 active weight memory per token in GB (for MoE)."""
    return spec.active_params * 2 / 1e9


def compute_training_state_memory_gb(spec: StageSpec) -> float:
    """
    Estimated training state memory per GPU for ZeRO-2/3.
    Includes: weights (BF16) + optimizer states (FP32 master weights + Adam m/v)
    ZeRO-2 shards optimizer states across data-parallel ranks.
    """
    # Per-parameter memory: 2 (BF16 param) + 4 (FP32 master) + 4 (Adam m) + 4 (Adam v) = 14 bytes
    bytes_per_param = 14.0
    # ZeRO-2: optimizer states sharded across DP ranks
    dp_sharding_factor = spec.data_parallel
    # Active params only for MoE (each GPU holds its expert shard)
    params_per_gpu = spec.active_params / spec.expert_parallel if spec.is_moe else spec.total_params
    # Tensor parallel sharding
    params_per_gpu /= spec.tensor_parallel
    # ZeRO-2 optimizer sharding
    optimizer_bytes = (params_per_gpu * 12) / dp_sharding_factor  # 12 = FP32 master + Adam m/v
    weight_bytes = params_per_gpu * 2  # BF16 weights not sharded in ZeRO-2
    total_bytes = weight_bytes + optimizer_bytes
    return total_bytes / 1e9


def compute_activation_memory_gb(spec: StageSpec) -> float:
    """
    Rough estimate of activation memory per GPU.
    Without gradient checkpointing: O(batch_size * seq_len * hidden_size * num_layers)
    With gradient checkpointing: O(batch_size * seq_len * hidden_size * sqrt(num_layers))
    """
    micro_batch = max(1, spec.batch_size_tokens // spec.context_length // spec.data_parallel)
    if spec.gradient_checkpointing:
        layers_factor = math.sqrt(spec.num_layers)
    else:
        layers_factor = spec.num_layers
    # Rough: 4 tensors of shape (batch, seq, hidden) per layer, BF16
    bytes_per_layer = micro_batch * spec.context_length * spec.hidden_size * 2 * 4
    total_bytes = bytes_per_layer * layers_factor
    # Tensor parallel sharding
    total_bytes /= spec.tensor_parallel
    return total_bytes / 1e9


def compute_kv_cache_gb_per_sequence(spec: StageSpec) -> float:
    """KV cache memory for a single sequence in BF16."""
    # 2 (K+V) * num_kv_heads * head_dim * num_layers * 2 bytes
    head_dim = spec.hidden_size // spec.num_attention_heads
    return (2 * spec.num_kv_heads * head_dim * spec.num_layers * 2 * spec.context_length) / 1e9


def simulate_stage(spec: StageSpec, gpu_price_override: Optional[float] = None) -> dict:
    """Run the full simulation for a single stage."""
    gpu_spec = GPU_SPECS[spec.gpu_type]
    if gpu_price_override is not None:
        gpu_spec = GPUSpec(**{**gpu_spec.__dict__, "list_price_usd": gpu_price_override})

    flops_per_token = compute_flops_per_token(spec)
    total_flops = compute_training_flops(spec)
    throughput = compute_peak_throughput(spec, gpu_spec)
    duration_days = compute_training_duration_days(spec, gpu_spec)
    training_cost = compute_training_cost(spec, gpu_spec)
    weight_memory = compute_weight_memory_gb(spec)
    active_weight_memory = compute_active_weight_memory_gb(spec)
    training_state_memory = compute_training_state_memory_gb(spec)
    activation_memory = compute_activation_memory_gb(spec)
    kv_cache_per_seq = compute_kv_cache_gb_per_sequence(spec)
    hardware_cost = spec.num_gpus * gpu_spec.list_price_usd

    # Expected GPU failure rate (MTBF ~100,000 hours per GPU)
    gpu_mtbf_hours = 100_000.0
    cluster_mtbf_hours = gpu_mtbf_hours / spec.num_gpus
    cluster_mtbf_days = cluster_mtbf_hours / 24.0

    return {
        "stage": spec.stage,
        "name": spec.name,
        "total_params": spec.total_params,
        "active_params": spec.active_params,
        "sparsity_pct": 100.0 * (1.0 - spec.active_params / spec.total_params) if spec.is_moe else 0.0,
        "is_moe": spec.is_moe,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "training_tokens": spec.training_tokens,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "gpu_type": gpu_spec.name,
        "num_gpus": spec.num_gpus,
        "parallelism": f"TP={spec.tensor_parallel} PP={spec.pipeline_parallel} EP={spec.expert_parallel} DP={spec.data_parallel}",
        "target_mfu": spec.target_mfu,
        "throughput_tokens_per_sec": throughput,
        "throughput_tokens_per_sec_per_gpu": throughput / spec.num_gpus,
        "duration_days": duration_days,
        "training_cost_usd": training_cost,
        "hardware_cost_usd": hardware_cost,
        "weight_memory_gb": weight_memory,
        "active_weight_memory_gb": active_weight_memory,
        "training_state_memory_per_gpu_gb": training_state_memory,
        "activation_memory_per_gpu_gb": activation_memory,
        "total_memory_per_gpu_gb": training_state_memory + activation_memory,
        "gpu_hbm_gb": gpu_spec.hbm_gb,
        "memory_fits": (training_state_memory + activation_memory) <= gpu_spec.hbm_gb,
        "kv_cache_per_sequence_gb": kv_cache_per_seq,
        "cluster_mtbf_days": cluster_mtbf_days,
        "prerequisite_stage": spec.prerequisite_stage,
    }


def format_number(n: float, unit: str = "") -> str:
    """Format large numbers with appropriate suffixes."""
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T{unit}"
    elif abs(n) >= 1e9:
        return f"{n/1e9:.2f}B{unit}"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M{unit}"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.2f}K{unit}"
    else:
        return f"{n:.2f}{unit}"


def print_stage_report(result: dict) -> None:
    """Print a formatted report for a single stage."""
    print(f"\n{'='*70}")
    print(f"  Stage {result['stage']}: {result['name']}")
    print(f"{'='*70}")

    print(f"\n  Model Architecture")
    print(f"  {'Total Parameters':<40} {format_number(result['total_params'])}")
    print(f"  {'Active Parameters per Token':<40} {format_number(result['active_params'])}")
    if result['is_moe']:
        print(f"  {'Sparsity':<40} {result['sparsity_pct']:.1f}%")
        print(f"  {'Experts':<40} {result['num_experts']} total, top-{result['top_k']} routing")

    print(f"\n  Training Configuration")
    print(f"  {'Training Tokens':<40} {format_number(result['training_tokens'])}")
    print(f"  {'FLOPs per Token':<40} {format_number(result['flops_per_token'])}")
    print(f"  {'Total Training FLOPs':<40} {format_number(result['total_flops'])}")

    print(f"\n  Hardware")
    print(f"  {'GPU Type':<40} {result['gpu_type']}")
    print(f"  {'GPU Count':<40} {result['num_gpus']:,}")
    print(f"  {'Parallelism':<40} {result['parallelism']}")
    print(f"  {'Target MFU':<40} {result['target_mfu']*100:.0f}%")

    print(f"\n  Performance Estimates")
    print(f"  {'Throughput (tokens/sec)':<40} {format_number(result['throughput_tokens_per_sec'])}")
    print(f"  {'Throughput (tokens/sec/GPU)':<40} {format_number(result['throughput_tokens_per_sec_per_gpu'])}")
    print(f"  {'Training Duration':<40} {result['duration_days']:.1f} days")
    print(f"  {'Expected GPU Failure Interval':<40} {result['cluster_mtbf_days']:.1f} days")

    print(f"\n  Memory (per GPU)")
    print(f"  {'GPU HBM Capacity':<40} {result['gpu_hbm_gb']:.0f} GB")
    print(f"  {'Training State Memory':<40} {result['training_state_memory_per_gpu_gb']:.1f} GB")
    print(f"  {'Activation Memory':<40} {result['activation_memory_per_gpu_gb']:.1f} GB")
    print(f"  {'Total Estimated Memory':<40} {result['total_memory_per_gpu_gb']:.1f} GB")
    fits = "✓ FITS" if result['memory_fits'] else "✗ DOES NOT FIT"
    print(f"  {'Memory Feasibility':<40} {fits}")
    print(f"  {'KV Cache per Sequence':<40} {result['kv_cache_per_sequence_gb']:.1f} GB")

    print(f"\n  Cost Estimates (2024 pricing)")
    print(f"  {'Training Cost (spot)':<40} ${result['training_cost_usd']:,.0f}")
    print(f"  {'Hardware Cost (list)':<40} ${result['hardware_cost_usd']:,.0f}")

    if result['prerequisite_stage']:
        print(f"\n  ⚠  Prerequisite: Stage {result['prerequisite_stage']} Go/No-Go FULL GO required")


def print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table of all stages."""
    print(f"\n{'='*110}")
    print(f"  Jupiter Shot — Scaling Simulation Summary")
    print(f"{'='*110}")
    header = f"  {'Stage':<8} {'Name':<25} {'Total Params':<15} {'Active Params':<15} {'GPUs':<8} {'Duration':<12} {'Train Cost':<15}"
    print(header)
    print(f"  {'-'*100}")
    for r in results:
        fits = "✓" if r['memory_fits'] else "✗"
        print(f"  {r['stage']:<8} {r['name']:<25} {format_number(r['total_params']):<15} "
              f"{format_number(r['active_params']):<15} {r['num_gpus']:<8} "
              f"{r['duration_days']:.0f}d{'':<8} ${r['training_cost_usd']:>12,.0f}  {fits}")
    print(f"{'='*110}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Jupiter Shot Scaling Simulator")
    parser.add_argument("--stage", type=int, default=None,
                        help="Simulate a specific stage (1-6). Default: all stages.")
    parser.add_argument("--gpu-price", type=float, default=None,
                        help="Override GPU list price in USD.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary table only, no per-stage details.")
    args = parser.parse_args()

    stages_to_run = STAGES if args.stage is None else [s for s in STAGES if s.stage == args.stage]
    if not stages_to_run:
        print(f"Error: Stage {args.stage} not found. Valid stages: 1-6.")
        return

    results = [simulate_stage(s, args.gpu_price) for s in stages_to_run]

    if not args.summary_only:
        for r in results:
            print_stage_report(r)

    if len(results) > 1 or args.summary_only:
        print_summary_table(results)

    print(f"\n  ⚠  All estimates are order-of-magnitude only.")
    print(f"     Memory estimates exclude communication buffers and framework overhead.")
    print(f"     Cost estimates use spot/reserved pricing; on-demand is 2-3× higher.")
    print(f"     Stage 6 feasibility depends on open problems documented in RISK_REGISTER.md.\n")


if __name__ == "__main__":
    main()
