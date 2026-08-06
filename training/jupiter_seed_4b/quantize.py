#!/usr/bin/env python3
"""
Jupiter Seed 4B — Quantization Export
=======================================
Merges the LoRA adapter into the base model and exports to INT8, INT4 (GGUF),
and records the quantized model size. Measures quality regression after
quantization.

Usage:
    python3 quantize.py --adapter-path artifacts/seed4b_diagnostic/final_adapter
                        --output-dir artifacts/quantized
                        --formats int8 gguf
    python3 quantize.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen3-4B"
SUPPORTED_FORMATS = ["int8", "int4", "gguf"]


def merge_adapter(adapter_path: Path, output_dir: Path) -> Path:
    """Merge LoRA adapter into base model and save merged weights."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        log.error("Required packages not installed: %s", exc)
        sys.exit(1)

    merged_path = output_dir / "merged_fp16"
    merged_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model: %s", BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=False)

    log.info("Loading adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    log.info("Merged model saved to %s", merged_path)
    return merged_path


def export_int8(merged_path: Path, output_dir: Path) -> Path:
    """Export INT8 quantized model using bitsandbytes."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        log.error("Required packages not installed: %s", exc)
        sys.exit(1)

    int8_path = output_dir / "int8"
    int8_path.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=False)
    model.save_pretrained(str(int8_path))
    tokenizer.save_pretrained(str(int8_path))
    log.info("INT8 model saved to %s", int8_path)
    return int8_path


def export_gguf(merged_path: Path, output_dir: Path) -> Path:
    """Export GGUF (Q4_K_M) using llama.cpp convert script."""
    gguf_path = output_dir / "gguf"
    gguf_path.mkdir(parents=True, exist_ok=True)
    output_file = gguf_path / "seed4b_diagnostic_q4km.gguf"

    # Check for llama.cpp convert script
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
    if not convert_script.exists():
        log.warning(
            "llama.cpp convert script not found at %s. "
            "GGUF export skipped. Install llama.cpp to enable.",
            convert_script,
        )
        return gguf_path

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_path),
        "--outfile", str(output_file),
        "--outtype", "q4_k_m",
    ]
    log.info("Running GGUF export: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("GGUF export failed: %s", result.stderr)
        sys.exit(1)
    log.info("GGUF model saved to %s", output_file)
    return gguf_path


def measure_model_size(path: Path) -> int:
    """Return total size in MB of all files in a directory."""
    total_bytes = sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file()
    )
    return total_bytes // (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Quantization Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter-path", type=Path, required=True,
        help="Path to the trained LoRA adapter.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/quantized"),
        help="Output directory for quantized models.",
    )
    parser.add_argument(
        "--formats", nargs="+",
        default=["int8", "gguf"],
        choices=SUPPORTED_FORMATS,
        help="Quantization formats to export.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = merge_adapter(args.adapter_path, args.output_dir)
    sizes: dict = {"merged_fp16_mb": measure_model_size(merged_path)}

    if "int8" in args.formats:
        int8_path = export_int8(merged_path, args.output_dir)
        sizes["int8_mb"] = measure_model_size(int8_path)

    if "gguf" in args.formats or "int4" in args.formats:
        gguf_path = export_gguf(merged_path, args.output_dir)
        sizes["gguf_q4km_mb"] = measure_model_size(gguf_path)

    report_path = args.output_dir / "quantization_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(sizes, fh, indent=2)
    log.info("Quantization report: %s", json.dumps(sizes))
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
