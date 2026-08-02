"""
Jupiter Shot — Laptop Validation Draft Report Generator
=========================================================
Reads all result files and generates docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md.

Does NOT overwrite docs/MONTH1_VALIDATION_RESULTS.md or docs/MONTH1_GO_NO_GO.md.

Usage:
    python scripts/generate_laptop_validation_draft.py
    python scripts/generate_laptop_validation_draft.py --results-dir benchmarks/results/laptop
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


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
                except Exception:
                    pass
    return lines


def fmt(v, decimals: int = 4, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}{suffix}"
    return str(v) + suffix


def generate_draft(results_dir: Path, output_path: Path) -> None:
    preflight = load_json(results_dir / "preflight.json")
    dense_summary = load_json(results_dir / "dense_summary.json")
    moe_summary = load_json(results_dir / "moe_summary.json")
    resume = load_json(results_dir / "resume_test.json")
    dense_metrics = load_jsonl(results_dir / "dense_metrics.jsonl")
    moe_metrics = load_jsonl(results_dir / "moe_metrics.jsonl")
    errors = load_jsonl(results_dir / "errors.jsonl")

    gpu = preflight.get("gpu", {})
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Jupiter Shot — Kuwait Laptop GPU Validation Draft",
        f"",
        f"> **Generated:** {now}  ",
        f"> **Branch:** validation/kuwait-laptop-gpu  ",
        f"> **Status:** DRAFT — awaiting Kishore's hardware execution  ",
        f"> **Note:** This document will be populated with measured results after Kishore runs the validation scripts.",
        f"",
        f"---",
        f"",
        f"## 1. Hardware Environment",
        f"",
    ]

    if gpu.get("cuda_available"):
        lines += [
            f"| Field | Value |",
            f"|-------|-------|",
            f"| GPU Model | {gpu.get('gpu_model', '—')} |",
            f"| Compute Capability | {gpu.get('compute_capability', '—')} |",
            f"| VRAM Total | {gpu.get('vram_total_gb', '—')} GB |",
            f"| VRAM Free (at preflight) | {gpu.get('vram_free_gb', '—')} GB |",
            f"| Driver Version | {gpu.get('driver_version', '—')} |",
            f"| CUDA Runtime | {gpu.get('cuda_version', '—')} |",
            f"| PyTorch Version | {gpu.get('pytorch_version', '—')} |",
            f"| DeepSpeed Version | {gpu.get('deepspeed_version', '—')} |",
            f"| Flash Attention | {gpu.get('flash_attention_version', '—')} |",
            f"| NCCL Version | {gpu.get('nccl_version', '—')} |",
            f"| FP16 Support | {gpu.get('fp16_supported', '—')} |",
            f"| BF16 Support | {gpu.get('bf16_supported', '—')} |",
            f"| GPU Temperature at Preflight | {gpu.get('gpu_temperature_c', 'unavailable')} °C |",
            f"| Recommended Precision | {preflight.get('recommended_precision', '—')} |",
            f"| Recommended Dense Config | {preflight.get('recommended_dense_config', '—')} |",
            f"| Recommended MoE Config | {preflight.get('recommended_moe_config', '—')} |",
            f"",
        ]
    else:
        lines += [
            f"**Preflight not yet run.** Execute `scripts\\windows\\run_preflight.bat` first.",
            f"",
        ]

    # Dense results
    lines += [
        f"---",
        f"",
        f"## 2. Dense CUDA Validation",
        f"",
    ]

    if dense_summary:
        status = dense_summary.get("status", "—")
        lines += [
            f"**Status:** {status}",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Config | {dense_summary.get('config', '—')} |",
            f"| Total Parameters | {dense_summary.get('total_params', '—'):,} |" if isinstance(dense_summary.get('total_params'), int) else f"| Total Parameters | — |",
            f"| Precision | {dense_summary.get('precision', '—')} |",
            f"| Data Mode | {dense_summary.get('data_mode', '—')} |",
            f"| Steps Completed | {dense_summary.get('steps_completed', '—')} / {dense_summary.get('steps_planned', '—')} |",
            f"| First Loss | {fmt(dense_summary.get('first_loss'))} |",
            f"| Last Loss | {fmt(dense_summary.get('last_loss'))} |",
            f"| Loss Decreased | {dense_summary.get('loss_decreased', '—')} |",
            f"| NaN/Inf Count | {dense_summary.get('nan_inf_count', '—')} |",
            f"| OOM Count | {dense_summary.get('oom_count', '—')} |",
            f"| Total Tokens | {dense_summary.get('total_tokens', '—'):,} |" if isinstance(dense_summary.get('total_tokens'), int) else f"| Total Tokens | — |",
            f"| Avg Tokens/sec | {fmt(dense_summary.get('avg_tokens_per_sec'), 1)} |",
            f"| Peak VRAM Allocated | {fmt(dense_summary.get('peak_vram_allocated_gb'), 3)} GB |",
            f"| Wall Time | {fmt(dense_summary.get('wall_time_s'), 1)} s |",
            f"| Checkpoint Size | {fmt(dense_summary.get('checkpoint_size_mb'), 1)} MB |",
            f"",
        ]

        if dense_metrics:
            lines += [
                f"### Loss Curve (every 10 steps)",
                f"",
                f"| Step | Loss | Tokens/sec | VRAM (GB) | Temp (°C) |",
                f"|------|------|-----------|-----------|-----------|",
            ]
            for m in dense_metrics:
                if m["step"] % 10 == 0 or m["step"] == 1:
                    lines.append(
                        f"| {m['step']} | {fmt(m.get('loss'))} | "
                        f"{fmt(m.get('tokens_per_sec'), 1)} | "
                        f"{fmt(m.get('allocated_gb'), 3)} | "
                        f"{m.get('gpu_temp_c', '—')} |"
                    )
            lines.append("")
    else:
        lines += [
            f"**Not yet run.** Execute `scripts\\windows\\run_dense_validation.bat`.",
            f"",
        ]

    # MoE results
    lines += [
        f"---",
        f"",
        f"## 3. MoE CUDA Validation",
        f"",
    ]

    if moe_summary:
        status = moe_summary.get("status", "—")
        lines += [
            f"**Status:** {status}",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Config | {moe_summary.get('config', '—')} |",
            f"| Total Parameters | {moe_summary.get('total_params', '—'):,} |" if isinstance(moe_summary.get('total_params'), int) else f"| Total Parameters | — |",
            f"| Active Parameters per Token | {moe_summary.get('active_params_per_token', '—'):,} |" if isinstance(moe_summary.get('active_params_per_token'), int) else f"| Active Parameters per Token | — |",
            f"| Experts | {moe_summary.get('num_experts', '—')} total, top-{moe_summary.get('top_k', '—')} routing |",
            f"| Steps Completed | {moe_summary.get('steps_completed', '—')} / {moe_summary.get('steps_planned', '—')} |",
            f"| First Loss | {fmt(moe_summary.get('first_loss'))} |",
            f"| Last Loss | {fmt(moe_summary.get('last_loss'))} |",
            f"| Avg Aux Loss (last 10) | {fmt(moe_summary.get('avg_aux_loss_last10'), 6)} |",
            f"| Avg Router Entropy (last 10) | {fmt(moe_summary.get('avg_router_entropy_last10'), 4)} bits |",
            f"| Avg Utilization CV (last 10) | {fmt(moe_summary.get('avg_utilization_cv_last10'), 4)} |",
            f"| Dropped Token % | {fmt(moe_summary.get('dropped_token_pct'), 2)} % |",
            f"| NaN/Inf Count | {moe_summary.get('nan_inf_count', '—')} |",
            f"| Peak VRAM Allocated | {fmt(moe_summary.get('peak_vram_allocated_gb'), 3)} GB |",
            f"| MoE Accepted | {moe_summary.get('moe_accepted', '—')} |",
            f"",
        ]

        acc = moe_summary.get("moe_acceptance", {})
        if acc:
            lines += [
                f"### MoE Acceptance Criteria",
                f"",
                f"| Criterion | Result |",
                f"|-----------|--------|",
            ]
            for k, v in acc.items():
                icon = "✓" if v is True else ("?" if v is None else "✗")
                lines.append(f"| {k.replace('_', ' ').title()} | {icon} {v} |")
            lines.append("")
    else:
        lines += [
            f"**Not yet run.** Execute `scripts\\windows\\run_moe_validation.bat`.",
            f"",
        ]

    # Resume test
    lines += [
        f"---",
        f"",
        f"## 4. Checkpoint Resume Test",
        f"",
    ]

    if resume:
        lines += [
            f"**Overall:** {'PASS' if resume.get('passed') else 'FAIL'}",
            f"",
            f"| Test | Result |",
            f"|------|--------|",
        ]
        for test_name, test_data in resume.get("tests", {}).items():
            icon = "✓" if test_data.get("passed") else "✗"
            lines.append(f"| {test_name.replace('_', ' ').title()} | {icon} |")
        lines += [
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Checkpoint Size | {fmt(resume.get('checkpoint_size_mb'), 1)} MB |",
            f"| Save Duration | {fmt(resume.get('checkpoint_save_duration_s'), 3)} s |",
            f"| Load Duration | {fmt(resume.get('checkpoint_load_duration_s'), 3)} s |",
            f"",
        ]
    else:
        lines += [
            f"**Not yet run.** Execute `scripts\\windows\\run_resume_validation.bat`.",
            f"",
        ]

    # Errors
    if errors:
        lines += [
            f"---",
            f"",
            f"## 5. Errors",
            f"",
            f"```",
        ]
        for e in errors[:10]:
            lines.append(json.dumps(e))
        lines += ["```", ""]

    lines += [
        f"---",
        f"",
        f"## 6. Known Limitations",
        f"",
        f"- This validation uses a single GPU laptop. Results do not demonstrate multi-node or distributed training.",
        f"- Laptop thermal throttling may reduce tokens/sec compared to server hardware.",
        f"- Windows GPU driver overhead adds approximately 1–1.5 GB VRAM baseline usage.",
        f"- The 7.14 GB/GPU ZeRO-2 estimate from Month 1 docs excludes activations, temporary tensors,",
        f"  communication buffers, and framework overhead. Actual peak VRAM will be 20–50% higher.",
        f"- A final PASS requires a real-text run (Wikitext-2, MIT license). Synthetic-only results",
        f"  are diagnostic only.",
        f"",
        f"---",
        f"",
        f"*This document was auto-generated by `scripts/generate_laptop_validation_draft.py`.*  ",
        f"*Do not edit manually — re-run the script after adding results.*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Draft report generated: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Laptop Validation Draft Report")
    parser.add_argument("--results-dir", default="benchmarks/results/laptop")
    parser.add_argument("--output", default="docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md")
    args = parser.parse_args()

    generate_draft(
        results_dir=Path(args.results_dir),
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
