"""
Jupiter Shot — Laptop Config Selector
======================================
Reads preflight.json and selects the appropriate training configuration.
Supports manual override with an explicit warning.

Usage:
    python scripts/select_laptop_config.py
    python scripts/select_laptop_config.py --override laptop_dense_medium --reason "Testing larger config"
    python scripts/select_laptop_config.py --preflight benchmarks/results/laptop/preflight.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VALID_DENSE_CONFIGS = ["laptop_dense_tiny", "laptop_dense_small", "laptop_dense_medium"]
VALID_MOE_CONFIGS = ["laptop_moe_tiny", "laptop_moe_small", "laptop_moe_medium"]
VALID_CONFIGS = VALID_DENSE_CONFIGS + VALID_MOE_CONFIGS


def load_preflight(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Preflight file not found: {path}")
        print("        Run: python scripts/laptop_gpu_preflight.py")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def select_config(preflight: dict, override: str | None = None, reason: str | None = None) -> dict:
    """
    Select training config from preflight data.

    If override is provided, print a warning and use it only if reason is given.
    """
    recommended_dense = preflight.get("recommended_dense_config")
    recommended_moe = preflight.get("recommended_moe_config")
    tier = preflight.get("config_tier")
    vram = preflight.get("gpu", {}).get("vram_total_gb", 0)
    precision = preflight.get("recommended_precision", "fp16")

    if override:
        if not reason:
            print("[ERROR] --override requires --reason to document why the recommended config is being bypassed.")
            print(f"        Recommended: dense={recommended_dense}, moe={recommended_moe}")
            sys.exit(1)
        if override not in VALID_CONFIGS:
            print(f"[ERROR] Unknown config: {override}. Valid: {', '.join(VALID_CONFIGS)}")
            sys.exit(1)
        print(f"\n[WARNING] Manual override: {override}")
        print(f"          Reason: {reason}")
        print(f"          Recommended was: dense={recommended_dense}, moe={recommended_moe}")
        print(f"          VRAM: {vram} GB. Ensure the override fits within available VRAM.")
        print("          Proceeding with override.\n")
        if override in VALID_DENSE_CONFIGS:
            return {"dense_config": override, "moe_config": recommended_moe, "precision": precision, "override": True}
        else:
            return {"dense_config": recommended_dense, "moe_config": override, "precision": precision, "override": True}

    if tier == "INSUFFICIENT" or tier == "NO_TORCH":
        print(f"[ERROR] Cannot select config: {preflight.get('config_rationale')}")
        sys.exit(2)

    result = {
        "dense_config": recommended_dense,
        "moe_config": recommended_moe,
        "precision": precision,
        "tier": tier,
        "vram_gb": vram,
        "override": False,
    }

    print(f"Selected configuration:")
    print(f"  VRAM tier:    {tier}")
    print(f"  Dense config: {recommended_dense}")
    print(f"  MoE config:   {recommended_moe}")
    print(f"  Precision:    {precision}")
    print(f"  VRAM:         {vram} GB")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Jupiter Shot Laptop Config Selector")
    parser.add_argument("--preflight", default="benchmarks/results/laptop/preflight.json")
    parser.add_argument("--override", help="Override config name (requires --reason)")
    parser.add_argument("--reason", help="Reason for overriding the recommended config")
    parser.add_argument("--output", default="benchmarks/results/laptop/selected_config.json")
    args = parser.parse_args()

    preflight = load_preflight(args.preflight)
    config = select_config(preflight, args.override, args.reason)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
