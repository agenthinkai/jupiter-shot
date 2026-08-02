"""
Jupiter Shot — Laptop GPU Preflight Check
==========================================
Detects hardware, CUDA availability, precision support, and recommends
the appropriate training configuration for the available VRAM.

Usage:
    python scripts/laptop_gpu_preflight.py
    python scripts/laptop_gpu_preflight.py --output-dir benchmarks/results/laptop

Outputs:
    benchmarks/results/laptop/preflight.json
    benchmarks/results/laptop/preflight.txt

Exit codes:
    0 — CUDA available, preflight passed
    1 — CUDA unavailable (CPU-only environment)
    2 — CUDA available but insufficient VRAM for any config
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _safe_import_torch() -> Optional[Any]:
    try:
        import torch
        return torch
    except ImportError:
        return None


# ── System info ───────────────────────────────────────────────────────────────

def collect_system_info() -> dict:
    info: dict[str, Any] = {}
    info["date"] = datetime.datetime.now().isoformat()
    info["os"] = platform.platform()
    info["python_version"] = sys.version
    info["cpu"] = platform.processor() or _run(["cat", "/proc/cpuinfo"]).split("\n")[4] if os.path.exists("/proc/cpuinfo") else "unknown"
    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["system_ram_gb"] = round(vm.total / 1024**3, 2)
        info["system_ram_free_gb"] = round(vm.available / 1024**3, 2)
    except ImportError:
        info["system_ram_gb"] = "psutil not installed"
        info["system_ram_free_gb"] = "psutil not installed"
    # Disk
    disk = shutil.disk_usage(".")
    info["disk_total_gb"] = round(disk.total / 1024**3, 1)
    info["disk_free_gb"] = round(disk.free / 1024**3, 1)
    return info


def collect_gpu_info(torch: Any) -> dict:
    info: dict[str, Any] = {}
    info["cuda_available"] = torch.cuda.is_available()
    if not info["cuda_available"]:
        info["cuda_unavailable_reason"] = "torch.cuda.is_available() returned False"
        return info

    info["cuda_version"] = torch.version.cuda or "unknown"
    info["pytorch_version"] = torch.__version__
    info["gpu_count"] = torch.cuda.device_count()
    info["gpu_model"] = torch.cuda.get_device_name(0)
    info["compute_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))

    props = torch.cuda.get_device_properties(0)
    info["vram_total_gb"] = round(props.total_memory / 1024**3, 2)
    info["vram_free_gb"] = round((props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3, 2)

    # Precision support
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    info["fp16_supported"] = cc_major >= 5
    info["bf16_supported"] = cc_major >= 8  # Ampere+

    # Driver version
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["driver_version"] = driver or "nvidia-smi not available"

    # NCCL version
    try:
        import torch.distributed as dist
        info["nccl_version"] = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "unknown"
    except Exception:
        info["nccl_version"] = "unknown"

    # DeepSpeed version
    try:
        import deepspeed
        info["deepspeed_version"] = deepspeed.__version__
    except ImportError:
        info["deepspeed_version"] = "not installed"

    # Flash Attention
    try:
        import flash_attn
        info["flash_attention_version"] = flash_attn.__version__
        info["flash_attention_available"] = True
    except ImportError:
        info["flash_attention_version"] = "not installed"
        info["flash_attention_available"] = False

    # Temperature (best-effort)
    temp = _run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"])
    info["gpu_temperature_c"] = int(temp) if temp.isdigit() else "unavailable"

    # GPU utilization
    util = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    info["gpu_utilization_pct"] = int(util) if util.isdigit() else "unavailable"

    return info


# ── VRAM-aware config selection ───────────────────────────────────────────────

VRAM_TIERS = [
    {
        "min_vram_gb": 16.0,
        "tier": "16GB+",
        "dense_config": "laptop_dense_medium",
        "moe_config": "laptop_moe_medium",
        "dense_params_m": "300–700M",
        "moe_total_params_m": "300–600M total",
        "rationale": "Sufficient VRAM for medium-scale validation with comfortable headroom.",
    },
    {
        "min_vram_gb": 10.0,
        "tier": "10–12GB",
        "dense_config": "laptop_dense_small",
        "moe_config": "laptop_moe_small",
        "dense_params_m": "150–300M",
        "moe_total_params_m": "300–600M total",
        "rationale": "Adequate for small-scale validation; activations and optimizer states fit.",
    },
    {
        "min_vram_gb": 6.0,
        "tier": "6–8GB",
        "dense_config": "laptop_dense_small",
        "moe_config": "laptop_moe_small",
        "dense_params_m": "50–150M",
        "moe_total_params_m": "100–300M total",
        "rationale": "Tight but workable with gradient checkpointing and small batch sizes.",
    },
    {
        "min_vram_gb": 3.5,
        "tier": "4GB",
        "dense_config": "laptop_dense_tiny",
        "moe_config": "laptop_moe_tiny",
        "dense_params_m": "20–40M",
        "moe_total_params_m": "Very small 8-expert MoE",
        "rationale": "Minimal config; gradient checkpointing mandatory.",
    },
]


def select_config(vram_gb: float) -> dict:
    """Select the appropriate training config tier based on available VRAM."""
    # Apply conservative headroom: reserve 1.5 GB for Windows GPU usage,
    # CUDA buffers, activations, temporary tensors, and framework overhead.
    effective_vram = vram_gb - 1.5
    for tier in VRAM_TIERS:
        if effective_vram >= tier["min_vram_gb"]:
            return tier
    return {
        "tier": "INSUFFICIENT",
        "dense_config": None,
        "moe_config": None,
        "rationale": f"Available VRAM ({vram_gb:.1f} GB) is below the minimum 4 GB required.",
    }


def estimate_memory(params_m: float, precision: str = "fp16") -> dict:
    """
    Estimate GPU memory requirements for a model.

    Excludes activations, temporary tensors, communication buffers, and
    framework overhead — those add 20–50% on top of these estimates.
    """
    bytes_per_param = 2 if precision in ("fp16", "bf16") else 4
    weights_gb = params_m * 1e6 * bytes_per_param / 1024**3
    # AdamW: fp32 master copy + m + v = 3× fp32 = 12 bytes/param
    optimizer_gb = params_m * 1e6 * 12 / 1024**3
    # Gradients: same dtype as weights
    grad_gb = weights_gb
    total_training_gb = weights_gb + optimizer_gb + grad_gb
    return {
        "weights_gb": round(weights_gb, 3),
        "optimizer_gb": round(optimizer_gb, 3),
        "gradients_gb": round(grad_gb, 3),
        "total_training_gb": round(total_training_gb, 3),
        "note": (
            "Excludes activations, temporary tensors, communication buffers, "
            "and framework overhead. Add 20–50% for a realistic estimate."
        ),
    }


# ── Precision selection ───────────────────────────────────────────────────────

def select_precision(gpu_info: dict) -> str:
    """Select the safest supported precision for training."""
    if gpu_info.get("bf16_supported"):
        return "bf16"
    if gpu_info.get("fp16_supported"):
        return "fp16"
    return "fp32"


# ── Report generation ─────────────────────────────────────────────────────────

def build_report(system_info: dict, gpu_info: dict, config_rec: dict, precision: str) -> dict:
    report = {
        "preflight_status": "PASS" if gpu_info.get("cuda_available") else "FAIL",
        "system": system_info,
        "gpu": gpu_info,
        "recommended_precision": precision,
        "recommended_dense_config": config_rec.get("dense_config"),
        "recommended_moe_config": config_rec.get("moe_config"),
        "config_tier": config_rec.get("tier"),
        "config_rationale": config_rec.get("rationale"),
        "memory_estimate_note": (
            "The 7.14 GB/GPU ZeRO-2 estimate in Month 1 docs excludes activations, "
            "temporary tensors, communication buffers, and framework overhead. "
            "Actual peak VRAM will be 20–50% higher."
        ),
    }
    if not gpu_info.get("cuda_available"):
        report["failure_reason"] = gpu_info.get("cuda_unavailable_reason", "CUDA not available")
    return report


def format_text_report(report: dict) -> str:
    lines = [
        "=" * 60,
        "  JUPITER SHOT — LAPTOP GPU PREFLIGHT REPORT",
        "=" * 60,
        f"  Status:  {report['preflight_status']}",
        f"  Date:    {report['system']['date']}",
        "",
        "SYSTEM",
        f"  OS:         {report['system']['os']}",
        f"  CPU:        {report['system']['cpu']}",
        f"  RAM:        {report['system'].get('system_ram_gb', '?')} GB total, "
        f"{report['system'].get('system_ram_free_gb', '?')} GB free",
        f"  Disk free:  {report['system']['disk_free_gb']} GB",
        "",
        "GPU",
    ]
    g = report["gpu"]
    if g.get("cuda_available"):
        lines += [
            f"  Model:              {g.get('gpu_model', '?')}",
            f"  Compute capability: {g.get('compute_capability', '?')}",
            f"  VRAM total:         {g.get('vram_total_gb', '?')} GB",
            f"  VRAM free:          {g.get('vram_free_gb', '?')} GB",
            f"  Driver:             {g.get('driver_version', '?')}",
            f"  CUDA runtime:       {g.get('cuda_version', '?')}",
            f"  PyTorch:            {g.get('pytorch_version', '?')}",
            f"  DeepSpeed:          {g.get('deepspeed_version', '?')}",
            f"  Flash Attention:    {g.get('flash_attention_version', '?')}",
            f"  NCCL:               {g.get('nccl_version', '?')}",
            f"  FP16 support:       {g.get('fp16_supported', '?')}",
            f"  BF16 support:       {g.get('bf16_supported', '?')}",
            f"  Temperature:        {g.get('gpu_temperature_c', 'unavailable')} °C",
            f"  GPU utilization:    {g.get('gpu_utilization_pct', 'unavailable')} %",
        ]
    else:
        lines.append(f"  CUDA UNAVAILABLE: {g.get('cuda_unavailable_reason', 'unknown')}")

    lines += [
        "",
        "RECOMMENDATION",
        f"  Precision:      {report['recommended_precision']}",
        f"  Dense config:   {report['recommended_dense_config'] or 'NONE — insufficient VRAM'}",
        f"  MoE config:     {report['recommended_moe_config'] or 'NONE — insufficient VRAM'}",
        f"  Tier:           {report['config_tier']}",
        f"  Rationale:      {report['config_rationale']}",
        "",
        "NOTE",
        f"  {report['memory_estimate_note']}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_dir: str = "benchmarks/results/laptop") -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Jupiter Shot — Laptop GPU Preflight")
    print("=" * 60)

    system_info = collect_system_info()
    torch = _safe_import_torch()

    if torch is None:
        gpu_info = {
            "cuda_available": False,
            "cuda_unavailable_reason": "PyTorch is not installed (import failed)",
        }
        precision = "fp32"
        config_rec = {"tier": "NO_TORCH", "dense_config": None, "moe_config": None,
                      "rationale": "Install PyTorch before running validation."}
    else:
        gpu_info = collect_gpu_info(torch)
        precision = select_precision(gpu_info)
        vram = gpu_info.get("vram_total_gb", 0) if gpu_info.get("cuda_available") else 0
        config_rec = select_config(float(vram))

    report = build_report(system_info, gpu_info, config_rec, precision)
    text = format_text_report(report)

    # Save outputs
    json_path = out / "preflight.json"
    txt_path = out / "preflight.txt"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    txt_path.write_text(text)

    print(text)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")

    if not gpu_info.get("cuda_available"):
        print("\n[FAIL] CUDA is not available. Cannot proceed with GPU validation.")
        return 1

    if config_rec.get("tier") == "INSUFFICIENT":
        print("\n[FAIL] Insufficient VRAM for any training configuration.")
        return 2

    print("\n[PASS] Preflight complete. Proceed with validation.")
    return 0


def run_preflight(output_dir: "Path | str" = "benchmarks/results/laptop") -> dict:
    """
    Programmatic entry point for tests and other scripts.
    Returns the full preflight report dict.
    """
    import torch as _torch_check
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()
    torch = _safe_import_torch()
    if torch is None or not _torch_check.cuda.is_available():
        gpu_info = {
            "cuda_available": False,
            "cuda_unavailable_reason": "CUDA not available",
        }
        precision = "fp32"
        config_rec = {"tier": "NO_TORCH", "dense_config": None, "moe_config": None,
                      "rationale": "CUDA not available."}
    else:
        gpu_info = collect_gpu_info(torch)
        precision = select_precision(gpu_info)
        vram = gpu_info.get("vram_total_gb", 0) if gpu_info.get("cuda_available") else 0
        config_rec = select_config(float(vram))
    report = build_report(system_info, gpu_info, config_rec, precision)
    report["pass"] = bool(gpu_info.get("cuda_available"))
    text = format_text_report(report)
    json_path = out / "preflight.json"
    txt_path = out / "preflight.txt"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    txt_path.write_text(text)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop GPU Preflight")
    parser.add_argument("--output-dir", default="benchmarks/results/laptop",
                        help="Directory to save preflight.json and preflight.txt")
    args = parser.parse_args()
    sys.exit(main(args.output_dir))
