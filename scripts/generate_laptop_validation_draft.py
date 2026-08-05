"""
Jupiter Shot — Laptop Validation Draft Report Generator
=========================================================
Reads all result files and generates docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md.
Does NOT overwrite docs/LAPTOP_GPU_VALIDATION.md.

Usage:
    python scripts/generate_laptop_validation_draft.py
    python scripts/generate_laptop_validation_draft.py --results-dir benchmarks/results/laptop
    python scripts/generate_laptop_validation_draft.py --output docs/generated/MY_DRAFT.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines


def fmt(v: object, decimals: int = 4, suffix: str = "") -> str:
    if v is None:
        return "\u2014"
    if isinstance(v, float):
        return f"{v:.{decimals}f}{suffix}"
    return str(v) + suffix


def _git_info() -> tuple[str, str]:
    """Return (branch_name, short_commit_hash) from the current git repo."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        branch = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    return branch, commit


def generate_draft(results_dir: Path, output_path: Path) -> None:
    preflight     = load_json(results_dir / "preflight.json")
    dense_summary = load_json(results_dir / "dense_summary.json")
    moe_summary   = load_json(results_dir / "moe_summary.json")
    resume        = load_json(results_dir / "resume_test.json")
    dense_metrics = load_jsonl(results_dir / "dense_metrics.jsonl")
    moe_metrics   = load_jsonl(results_dir / "moe_metrics.jsonl")
    errors        = load_jsonl(results_dir / "errors.jsonl")

    gpu = preflight.get("gpu", {})
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    branch, commit = _git_info()
    dash = "\u2014"
    deg  = "\u00b0"
    tick = "\u2713"
    cross= "\u2717"

    lines: list[str] = [
        "# Jupiter Shot \u2014 Kuwait Laptop GPU Validation Draft",
        "",
        f"> **Generated:** {now}  ",
        f"> **Branch:** {branch}  ",
        f"> **Commit:** {commit}  ",
        "> **Status:** DRAFT \u2014 awaiting Kishore\u2019s hardware execution  ",
        "> **Note:** This document is auto-generated. Re-run the script after adding results.",
        "",
        "---",
        "",
        "## 1. Hardware Environment",
        "",
    ]

    if gpu.get("cuda_available"):
        lines += [
            "| Field | Value |",
            "|-------|-------|",
            f"| GPU Model | {gpu.get('gpu_model', dash)} |",
            f"| Compute Capability | {gpu.get('compute_capability', dash)} |",
            f"| VRAM Total | {gpu.get('vram_total_gb', dash)} GB |",
            f"| VRAM Free (at preflight) | {gpu.get('vram_free_gb', dash)} GB |",
            f"| Driver Version | {gpu.get('driver_version', dash)} |",
            f"| CUDA Runtime | {gpu.get('cuda_version', dash)} |",
            f"| PyTorch Version | {gpu.get('pytorch_version', dash)} |",
            f"| DeepSpeed Version | {gpu.get('deepspeed_version', dash)} |",
            f"| Flash Attention | {gpu.get('flash_attention_version', dash)} |",
            f"| NCCL Version | {gpu.get('nccl_version', dash)} |",
            f"| FP16 Support | {gpu.get('fp16_supported', dash)} |",
            f"| BF16 Support | {gpu.get('bf16_supported', dash)} |",
            f"| GPU Temperature at Preflight | {gpu.get('gpu_temperature_c', 'unavailable')} {deg}C |",
            f"| Recommended Precision | {preflight.get('recommended_precision', dash)} |",
            f"| Recommended Dense Config | {preflight.get('recommended_dense_config', dash)} |",
            f"| Recommended MoE Config | {preflight.get('recommended_moe_config', dash)} |",
            "",
        ]
    else:
        lines += [
            "**Preflight not yet run.** Execute `scripts\\\\windows\\\\run_preflight.bat` first.",
            "",
        ]

    # Dense results
    lines += ["---", "", "## 2. Dense CUDA Validation", ""]
    if dense_summary:
        status  = dense_summary.get("status", dash)
        outcome = dense_summary.get("outcome", dash)
        exit_c  = dense_summary.get("exit_code", dash)
        lines += [
            f"**Status:** {status} | **Outcome:** {outcome} | **Exit code:** {exit_c}",
            "",
            "### Training Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Config | {dense_summary.get('config', dash)} |",
            (f"| Total Parameters | {dense_summary.get('total_params', dash):,} |"
             if isinstance(dense_summary.get("total_params"), int)
             else f"| Total Parameters | {dash} |"),
            f"| Precision | {dense_summary.get('precision', dash)} |",
            f"| Data Mode | {dense_summary.get('data_mode', dash)} |",
            f"| Steps Completed | {dense_summary.get('steps_completed', dash)} / {dense_summary.get('steps_planned', dash)} |",
            f"| First Loss | {fmt(dense_summary.get('first_loss'))} |",
            f"| Last Loss | {fmt(dense_summary.get('last_loss'))} |",
            f"| Loss Decreased | {dense_summary.get('loss_decreased', dash)} |",
            f"| NaN/Inf Count | {dense_summary.get('nan_inf_count', dash)} |",
            f"| OOM Count | {dense_summary.get('oom_count', dash)} |",
            (f"| Total Tokens | {dense_summary.get('total_tokens', dash):,} |"
             if isinstance(dense_summary.get("total_tokens"), int)
             else f"| Total Tokens | {dash} |"),
            f"| Avg Tokens/sec | {fmt(dense_summary.get('avg_tokens_per_sec'), 1)} |",
            f"| Peak VRAM Allocated | {fmt(dense_summary.get('peak_vram_allocated_gb'), 3)} GB |",
            f"| Wall Time | {fmt(dense_summary.get('wall_time_s'), 1)} s |",
            f"| Checkpoint Size | {fmt(dense_summary.get('checkpoint_size_mb'), 1)} MB |",
            "",
        ]
        if dense_metrics:
            lines += [
                "### Loss Curve (every 10 steps)",
                "",
                f"| Step | Loss | Tokens/sec | VRAM (GB) | Temp ({deg}C) |",
                "|------|------|-----------|-----------|-----------|",
            ]
            for m in dense_metrics:
                if m["step"] % 10 == 0 or m["step"] == 1:
                    lines.append(
                        f"| {m['step']} | {fmt(m.get('loss'))} | "
                        f"{fmt(m.get('tokens_per_sec'), 1)} | "
                        f"{fmt(m.get('allocated_gb'), 3)} | "
                        f"{m.get('gpu_temp_c', dash)} |"
                    )
            lines.append("")
    else:
        lines += ["**Not yet run.** Execute `scripts\\\\windows\\\\run_dense_validation.bat`.", ""]

    # MoE results
    lines += ["---", "", "## 3. MoE CUDA Validation", ""]
    if moe_summary:
        status  = moe_summary.get("status", dash)
        outcome = moe_summary.get("outcome", dash)
        exit_c  = moe_summary.get("exit_code", dash)
        lines += [
            f"**Status:** {status} | **Outcome:** {outcome} | **Exit code:** {exit_c}",
            "",
            "### Architecture",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Config | {moe_summary.get('config', dash)} |",
            (f"| Total Parameters | {moe_summary.get('total_params', dash):,} |"
             if isinstance(moe_summary.get("total_params"), int)
             else f"| Total Parameters | {dash} |"),
            (f"| Active Parameters per Token | {moe_summary.get('active_params_per_token', dash):,} |"
             if isinstance(moe_summary.get("active_params_per_token"), int)
             else f"| Active Parameters per Token | {dash} |"),
            f"| Experts | {moe_summary.get('num_experts', dash)} total, top-{moe_summary.get('top_k', dash)} routing |",
            f"| Precision | {moe_summary.get('precision', dash)} |",
            f"| Data Mode | {moe_summary.get('data_mode', dash)} |",
            "",
            "### Training Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Steps Completed | {moe_summary.get('steps_completed', dash)} / {moe_summary.get('steps_planned', dash)} |",
            f"| First Loss | {fmt(moe_summary.get('first_loss'))} |",
            f"| Last Loss | {fmt(moe_summary.get('last_loss'))} |",
            f"| Avg Aux Loss (last 10) | {fmt(moe_summary.get('avg_aux_loss_last10'), 6)} |",
            f"| NaN/Inf Count | {moe_summary.get('nan_inf_count', dash)} |",
            f"| OOM Count | {moe_summary.get('oom_count', dash)} |",
            (f"| Total Tokens | {moe_summary.get('total_tokens', dash):,} |"
             if isinstance(moe_summary.get("total_tokens"), int)
             else f"| Total Tokens | {dash} |"),
            f"| Avg Tokens/sec | {fmt(moe_summary.get('avg_tokens_per_sec'), 1)} |",
            f"| Peak VRAM Allocated | {fmt(moe_summary.get('peak_vram_allocated_gb'), 3)} GB |",
            f"| Wall Time | {fmt(moe_summary.get('wall_time_s'), 1)} s |",
            "",
            "### Router Metrics (last 10 steps)",
            "",
            "| Metric | Value | Threshold |",
            "|--------|-------|-----------|",
            f"| Avg Router Entropy | {fmt(moe_summary.get('avg_router_entropy_last10'), 6)} nats | > 1.0 nats |",
            f"| Avg Utilization CV | {fmt(moe_summary.get('avg_utilization_cv_last10'), 6)} | < 0.5 |",
            f"| Dropped Token Fraction | {fmt(moe_summary.get('dropped_token_fraction'), 6)} | < 0.01 |",
            f"| MoE Accepted | {moe_summary.get('moe_accepted', dash)} | True required for PASS |",
            "",
        ]
        acc = moe_summary.get("moe_acceptance", {})
        if acc and isinstance(acc, dict):
            criteria   = acc.get("criteria", {})
            err_list   = acc.get("errors", [])
            if criteria:
                lines += [
                    "### MoE Acceptance Criteria",
                    "",
                    "| Criterion | Pass | Details |",
                    "|-----------|------|---------|",
                ]
                for k, v in criteria.items():
                    if isinstance(v, dict):
                        icon   = tick if v.get("pass") else cross
                        detail = v.get("threshold", "")
                        lines.append(f"| {k.replace('_', ' ').title()} | {icon} | {detail} |")
                    else:
                        icon = tick if v is True else ("?" if v is None else cross)
                        lines.append(f"| {k.replace('_', ' ').title()} | {icon} | {dash} |")
                lines.append("")
            if err_list:
                lines += ["**Acceptance errors:**", ""]
                for err in err_list:
                    lines.append(f"- {err}")
                lines.append("")
        if moe_metrics:
            lines += [
                "### Router Metrics Curve (every 10 steps)",
                "",
                f"| Step | Loss | Aux Loss | Entropy (nats) | CV | Inactive | Temp ({deg}C) |",
                "|------|------|----------|----------------|-----|----------|-----------|",
            ]
            for m in moe_metrics:
                if m["step"] % 10 == 0 or m["step"] == 1:
                    lines.append(
                        f"| {m['step']} | {fmt(m.get('loss'))} | "
                        f"{fmt(m.get('aux_loss'), 6)} | "
                        f"{fmt(m.get('router_entropy'), 4)} | "
                        f"{fmt(m.get('utilization_cv'), 4)} | "
                        f"{m.get('number_of_inactive_experts', dash)} | "
                        f"{m.get('gpu_temp_c', dash)} |"
                    )
            lines.append("")
    else:
        lines += ["**Not yet run.** Execute `scripts\\\\windows\\\\run_moe_validation.bat`.", ""]

    # Resume test
    lines += ["---", "", "## 4. Checkpoint Resume Test", ""]
    if resume:
        passed = resume.get("passed", False)
        lines += [
            f"**Overall:** {'PASS' if passed else 'FAIL'}",
            "",
            "| Test | Result |",
            "|------|--------|",
        ]
        for test_name, test_data in resume.get("tests", {}).items():
            ok = test_data.get("passed") if isinstance(test_data, dict) else test_data
            icon = tick if ok else cross
            lines.append(f"| {test_name.replace('_', ' ').title()} | {icon} |")
        lines += [
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Checkpoint Size | {fmt(resume.get('checkpoint_size_mb'), 1)} MB |",
            f"| Save Duration | {fmt(resume.get('checkpoint_save_duration_s'), 3)} s |",
            f"| Load Duration | {fmt(resume.get('checkpoint_load_duration_s'), 3)} s |",
            "",
        ]
    else:
        lines += ["**Not yet run.** Execute `scripts\\\\windows\\\\run_resume_validation.bat`.", ""]

    if errors:
        lines += ["---", "", "## 5. Errors", "", "```"]
        for e in errors[:10]:
            lines.append(json.dumps(e))
        lines += ["```", ""]

    lines += [
        "---",
        "",
        "## 6. Known Limitations",
        "",
        "- This validation uses a single GPU laptop. Results do not demonstrate multi-node or distributed training.",
        "- Laptop thermal throttling may reduce tokens/sec compared to server hardware.",
        "- Windows GPU driver overhead adds approximately 1\u20131.5 GB VRAM baseline usage.",
        "- The 7.14 GB/GPU ZeRO-2 estimate from Month 1 docs excludes activations, temporary tensors,",
        "  communication buffers, and framework overhead. Actual peak VRAM will be 20\u201350% higher.",
        "- A final PASS requires a real-text run (Wikitext-2, MIT license). Synthetic-only results",
        "  are diagnostic only and produce outcome=NOT_ACCEPTED regardless of router metrics.",
        "- Router entropy is reported in **nats** (natural logarithm base). The acceptance threshold",
        "  is 1.0 nats. For reference: max entropy for 8 experts = ln(8) \u2248 2.08 nats.",
        "",
        "---",
        "",
        f"*Generated by `scripts/generate_laptop_validation_draft.py` at {now}.*  ",
        f"*Branch: `{branch}` \u00b7 Commit: `{commit}`.*  ",
        "*Do not edit manually \u2014 re-run the script after adding results.*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Draft report generated: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Laptop Validation Draft Report")
    parser.add_argument("--results-dir", default="benchmarks/results/laptop")
    parser.add_argument("--output", default="docs/generated/LAPTOP_GPU_VALIDATION_DRAFT.md")
    args = parser.parse_args()
    generate_draft(
        results_dir=REPO_ROOT / args.results_dir,
        output_path=REPO_ROOT / args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
