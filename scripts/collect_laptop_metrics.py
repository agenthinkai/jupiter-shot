"""
Jupiter Shot — Laptop Metrics Collector
========================================
Aggregates all per-step metrics from JSONL files into summary statistics.

Usage:
    python scripts/collect_laptop_metrics.py
    python scripts/collect_laptop_metrics.py --results-dir benchmarks/results/laptop
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines


def summarize_series(values: list[float], name: str) -> dict:
    if not values:
        return {f"{name}_count": 0}
    clean = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not clean:
        return {f"{name}_count": 0}
    return {
        f"{name}_count": len(clean),
        f"{name}_first": round(clean[0], 6),
        f"{name}_last": round(clean[-1], 6),
        f"{name}_min": round(min(clean), 6),
        f"{name}_max": round(max(clean), 6),
        f"{name}_mean": round(sum(clean) / len(clean), 6),
        f"{name}_trend": "decreasing" if clean[-1] < clean[0] else "increasing" if clean[-1] > clean[0] else "flat",
    }


def compute_expert_utilization_stats(metrics: list[dict]) -> dict:
    """Aggregate expert utilization across all steps."""
    all_cvs = [m["utilization_cv"] for m in metrics if m.get("utilization_cv") is not None]
    all_entropies = [m["router_entropy_bits"] for m in metrics if m.get("router_entropy_bits") is not None]
    all_inactive = [m["num_inactive_experts"] for m in metrics if m.get("num_inactive_experts") is not None]
    all_max_min = [m["max_min_ratio"] for m in metrics if m.get("max_min_ratio") is not None]

    # Expert assignment shares across steps
    expert_shares_by_step = [m["expert_assignment_share"] for m in metrics if m.get("expert_assignment_share")]
    if expert_shares_by_step:
        n_experts = len(expert_shares_by_step[0])
        mean_shares = [
            sum(step[i] for step in expert_shares_by_step) / len(expert_shares_by_step)
            for i in range(n_experts)
        ]
    else:
        mean_shares = []

    return {
        "utilization_cv": summarize_series(all_cvs, "cv"),
        "router_entropy_bits": summarize_series(all_entropies, "entropy"),
        "inactive_experts": summarize_series(all_inactive, "inactive"),
        "max_min_ratio": summarize_series(all_max_min, "max_min"),
        "mean_expert_assignment_share": [round(s, 6) for s in mean_shares],
    }


def collect_metrics(results_dir: Path) -> dict:
    dense_metrics = load_jsonl(results_dir / "dense_metrics.jsonl")
    moe_metrics = load_jsonl(results_dir / "moe_metrics.jsonl")
    errors = load_jsonl(results_dir / "errors.jsonl")

    dense_summary_path = results_dir / "dense_summary.json"
    moe_summary_path = results_dir / "moe_summary.json"
    resume_path = results_dir / "resume_test.json"
    preflight_path = results_dir / "preflight.json"

    dense_summary = json.loads(dense_summary_path.read_text()) if dense_summary_path.exists() else {}
    moe_summary = json.loads(moe_summary_path.read_text()) if moe_summary_path.exists() else {}
    resume_result = json.loads(resume_path.read_text()) if resume_path.exists() else {}
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else {}

    report: dict[str, Any] = {
        "preflight": {
            "gpu_model": preflight.get("gpu", {}).get("gpu_model"),
            "vram_total_gb": preflight.get("gpu", {}).get("vram_total_gb"),
            "cuda_version": preflight.get("gpu", {}).get("cuda_version"),
            "pytorch_version": preflight.get("gpu", {}).get("pytorch_version"),
            "precision": preflight.get("recommended_precision"),
        },
        "dense": {},
        "moe": {},
        "resume": {},
        "errors": {"count": len(errors), "entries": errors[:5]},
    }

    if dense_metrics:
        losses = [m["loss"] for m in dense_metrics if m.get("loss") is not None]
        tps_vals = [m["tokens_per_sec"] for m in dense_metrics if m.get("tokens_per_sec")]
        vram_vals = [m.get("allocated_gb") for m in dense_metrics if m.get("allocated_gb")]
        temps = [m["gpu_temp_c"] for m in dense_metrics if m.get("gpu_temp_c") is not None]

        report["dense"] = {
            **summarize_series(losses, "loss"),
            **summarize_series(tps_vals, "tokens_per_sec"),
            "peak_vram_allocated_gb": max(vram_vals) if vram_vals else None,
            "max_gpu_temp_c": max(temps) if temps else "unavailable",
            "steps_recorded": len(dense_metrics),
            "nan_inf_count": dense_summary.get("nan_inf_count", 0),
            "status": dense_summary.get("status"),
            "total_tokens": dense_summary.get("total_tokens"),
            "wall_time_s": dense_summary.get("wall_time_s"),
        }

    if moe_metrics:
        losses = [m["loss"] for m in moe_metrics if m.get("loss") is not None]
        aux_losses = [m["aux_loss"] for m in moe_metrics if m.get("aux_loss") is not None]
        z_losses = [m["z_loss"] for m in moe_metrics if m.get("z_loss") is not None]
        tps_vals = [m["tokens_per_sec"] for m in moe_metrics if m.get("tokens_per_sec")]

        report["moe"] = {
            **summarize_series(losses, "loss"),
            **summarize_series(aux_losses, "aux_loss"),
            **summarize_series(z_losses, "z_loss"),
            **summarize_series(tps_vals, "tokens_per_sec"),
            "steps_recorded": len(moe_metrics),
            "nan_inf_count": moe_summary.get("nan_inf_count", 0),
            "status": moe_summary.get("status"),
            "moe_accepted": moe_summary.get("moe_accepted"),
            "moe_acceptance_detail": moe_summary.get("moe_acceptance"),
            **compute_expert_utilization_stats(moe_metrics),
        }

    if resume_result:
        report["resume"] = {
            "passed": resume_result.get("passed"),
            "tests": resume_result.get("tests"),
            "checkpoint_size_mb": resume_result.get("checkpoint_size_mb"),
            "save_duration_s": resume_result.get("checkpoint_save_duration_s"),
            "load_duration_s": resume_result.get("checkpoint_load_duration_s"),
        }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop Metrics Collector")
    parser.add_argument("--results-dir", default="benchmarks/results/laptop")
    parser.add_argument("--output", default="benchmarks/results/laptop/aggregated_metrics.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return 1

    report = collect_metrics(results_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Metrics aggregated: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
