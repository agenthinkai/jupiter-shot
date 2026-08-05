"""
Jupiter Shot — Laptop GPU Preflight Check
==========================================
Detects hardware, CUDA availability, precision support, and recommends
the appropriate training configuration for the available VRAM.

IMPORTANT: This preflight executes a real CUDA kernel to verify that the
installed PyTorch wheel actually supports the GPU architecture. Checking
torch.cuda.is_available() alone is insufficient — it only confirms driver
registration, not kernel execution. RTX 50-series (Blackwell, sm_120) GPUs
require PyTorch 2.7.1+cu128 or newer.

Usage:
    python scripts/laptop_gpu_preflight.py
    python scripts/laptop_gpu_preflight.py --output-dir benchmarks/results/laptop

Outputs:
    benchmarks/results/laptop/preflight.json
    benchmarks/results/laptop/preflight.txt

Exit codes:
    0 — CUDA available, real kernel verified, preflight passed
    1 — CUDA unavailable or kernel launch failed (environment blocked)
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


# ── Real CUDA kernel verification ─────────────────────────────────────────────

def run_kernel_test(torch: Any) -> dict:
    """
    Execute a real CUDA kernel to verify the PyTorch wheel supports this GPU.

    torch.cuda.is_available() only checks driver registration. On RTX 50-series
    (Blackwell, sm_120), PyTorch < 2.7.1+cu128 will report CUDA available but
    fail on the first actual kernel launch.

    Returns a dict with:
        kernel_pass: bool
        kernel_error: str or None
        bf16_pass: bool
        fp16_pass: bool
        arch_list: list[str]
        torch_version: str
        cuda_version: str
        gpu_name: str
        compute_capability: str
    """
    result: dict[str, Any] = {
        "kernel_pass": False,
        "kernel_error": None,
        "bf16_pass": False,
        "fp16_pass": False,
        "arch_list": [],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "gpu_name": "unknown",
        "compute_capability": "unknown",
    }

    if not torch.cuda.is_available():
        result["kernel_error"] = "torch.cuda.is_available() returned False"
        return result

    try:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        result["compute_capability"] = f"{cc[0]}.{cc[1]}"
        result["arch_list"] = torch.cuda.get_arch_list()
    except Exception as e:
        result["kernel_error"] = f"Failed to query GPU properties: {e}"
        return result

    # Check if sm_120 (Blackwell) is in arch list when needed
    cc_major = cc[0]
    if cc_major >= 12:
        sm_key = f"sm_{cc_major}{cc[1]}"
        if sm_key not in result["arch_list"] and f"compute_{cc_major}{cc[1]}" not in result["arch_list"]:
            result["kernel_error"] = (
                f"GPU requires {sm_key} but it is not in torch.cuda.get_arch_list(). "
                f"Install PyTorch 2.7.1+cu128: "
                f"pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128"
            )
            return result

    # Execute a real CUDA kernel: allocate, matmul, synchronize, verify
    try:
        a = torch.ones(64, 64, device="cuda", dtype=torch.float32)
        b = torch.ones(64, 64, device="cuda", dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        expected = 64.0
        actual = c[0, 0].item()
        if abs(actual - expected) > 1e-3:
            result["kernel_error"] = (
                f"Kernel result mismatch: expected {expected}, got {actual}"
            )
            return result
        result["kernel_pass"] = True
    except Exception as e:
        result["kernel_error"] = (
            f"CUDA kernel launch failed: {e}. "
            f"This usually means the PyTorch wheel does not support this GPU architecture. "
            f"Compute capability: {result['compute_capability']}. "
            f"Arch list: {result['arch_list']}."
        )
        return result

    # BF16 test
    try:
        a_bf16 = torch.ones(32, 32, device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.ones(32, 32, device="cuda", dtype=torch.bfloat16)
        c_bf16 = torch.matmul(a_bf16, b_bf16)
        torch.cuda.synchronize()
        result["bf16_pass"] = True
    except Exception as e:
        result["bf16_pass"] = False
        result["bf16_error"] = str(e)

    # FP16 test
    try:
        a_fp16 = torch.ones(32, 32, device="cuda", dtype=torch.float16)
        b_fp16 = torch.ones(32, 32, device="cuda", dtype=torch.float16)
        c_fp16 = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        result["fp16_pass"] = True
    except Exception as e:
        result["fp16_pass"] = False
        result["fp16_error"] = str(e)

    return result


# ── System info ───────────────────────────────────────────────────────────────

def collect_system_info() -> dict:
    info: dict[str, Any] = {}
    info["date"] = datetime.datetime.now().isoformat()
    info["os"] = platform.platform()
    info["python_version"] = sys.version
    info["python_version_tuple"] = list(sys.version_info[:3])
    info["cpu"] = platform.processor() or "unknown"
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
    info["arch_list"] = torch.cuda.get_arch_list()

    props = torch.cuda.get_device_properties(0)
    info["vram_total_gb"] = round(props.total_memory / 1024**3, 2)
    info["vram_free_gb"] = round((props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3, 2)

    # Precision support
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    info["fp16_supported"] = cc_major >= 5
    info["bf16_supported"] = cc_major >= 8  # Ampere+; Blackwell also supports BF16

    # Driver version
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    info["driver_version"] = driver or "nvidia-smi not available"

    # NCCL version
    try:
        info["nccl_version"] = torch.cuda.nccl.version() if hasattr(torch.cuda, "nccl") else "unknown"
    except Exception:
        info["nccl_version"] = "unknown"

    # DeepSpeed version (optional on laptop)
    try:
        import deepspeed
        info["deepspeed_version"] = deepspeed.__version__
    except ImportError:
        info["deepspeed_version"] = "not installed (not required for laptop validation)"

    # Flash Attention (optional on laptop)
    try:
        import flash_attn
        info["flash_attention_version"] = flash_attn.__version__
        info["flash_attention_available"] = True
    except ImportError:
        info["flash_attention_version"] = "not installed (not required for laptop validation)"
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
        "rationale": "Tight but workable with gradient checkpointing and small batch sizes. "
                     "SMALL config enforced. 20% VRAM headroom preserved.",
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
    # For 8 GB VRAM (RTX 5060), effective VRAM = 6.5 GB → SMALL tier.
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
    optimizer_gb = params_m * 1e6 * 12 / 1024**3  # AdamW fp32: 12 bytes/param
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

def build_report(
    system_info: dict,
    gpu_info: dict,
    config_rec: dict,
    precision: str,
    kernel_test: Optional[dict] = None,
) -> dict:
    kernel_pass = kernel_test.get("kernel_pass", False) if kernel_test else None
    report = {
        "preflight_status": (
            "PASS" if (gpu_info.get("cuda_available") and kernel_pass) else "FAIL"
        ),
        "system": system_info,
        "gpu": gpu_info,
        "kernel_test": kernel_test or {},
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
    elif kernel_test and not kernel_pass:
        report["failure_reason"] = kernel_test.get("kernel_error", "Kernel test failed")
    return report


def format_text_report(report: dict) -> str:
    kt = report.get("kernel_test", {})
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
            f"  Arch list:          {g.get('arch_list', [])}",
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

    if kt:
        lines += [
            "",
            "KERNEL TEST",
            f"  Kernel launch:      {'PASS' if kt.get('kernel_pass') else 'FAIL'}",
            f"  BF16 test:          {'PASS' if kt.get('bf16_pass') else 'FAIL'}",
            f"  FP16 test:          {'PASS' if kt.get('fp16_pass') else 'FAIL'}",
        ]
        if kt.get("kernel_error"):
            lines.append(f"  Error:              {kt['kernel_error']}")

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
    if report.get("failure_reason"):
        lines += [
            "",
            "FAILURE REASON",
            f"  {report['failure_reason']}",
            "",
            "REPAIR",
            "  If this is an RTX 50-series (Blackwell) GPU, install:",
            "    pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
            "  Then re-run this preflight.",
            "",
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
        kernel_test = None
    else:
        gpu_info = collect_gpu_info(torch)
        precision = select_precision(gpu_info)
        vram = gpu_info.get("vram_total_gb", 0) if gpu_info.get("cuda_available") else 0
        config_rec = select_config(float(vram))
        # Run real kernel test
        if gpu_info.get("cuda_available"):
            print("\n[INFO] Running real CUDA kernel test...")
            kernel_test = run_kernel_test(torch)
        else:
            kernel_test = None

    report = build_report(system_info, gpu_info, config_rec, precision, kernel_test)
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

    if kernel_test and not kernel_test.get("kernel_pass"):
        print(f"\n[FAIL] CUDA kernel test failed: {kernel_test.get('kernel_error')}")
        print("\n       The environment is BLOCKED. Do not proceed with training.")
        print("       See docs/BLACKWELL_ENVIRONMENT.md for repair instructions.")
        return 1

    if config_rec.get("tier") == "INSUFFICIENT":
        print("\n[FAIL] Insufficient VRAM for any training configuration.")
        return 2

    print("\n[PASS] Preflight complete. Real CUDA kernel verified. Proceed with validation.")
    return 0


def run_preflight(output_dir: "Path | str" = "benchmarks/results/laptop") -> dict:
    """
    Programmatic entry point for tests and other scripts.
    Returns the full preflight report dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()
    torch = _safe_import_torch()
    if torch is None or not torch.cuda.is_available():
        gpu_info = {
            "cuda_available": False,
            "cuda_unavailable_reason": "CUDA not available",
        }
        precision = "fp32"
        config_rec = {"tier": "NO_TORCH", "dense_config": None, "moe_config": None,
                      "rationale": "CUDA not available."}
        kernel_test = None
    else:
        gpu_info = collect_gpu_info(torch)
        precision = select_precision(gpu_info)
        vram = gpu_info.get("vram_total_gb", 0) if gpu_info.get("cuda_available") else 0
        config_rec = select_config(float(vram))
        kernel_test = run_kernel_test(torch)

    report = build_report(system_info, gpu_info, config_rec, precision, kernel_test)
    report["pass"] = (
        bool(gpu_info.get("cuda_available"))
        and (kernel_test.get("kernel_pass", False) if kernel_test else False)
    )
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
