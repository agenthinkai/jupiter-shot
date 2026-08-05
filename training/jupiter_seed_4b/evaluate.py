#!/usr/bin/env python3
"""
Jupiter Seed 4B — Evaluation Script
=====================================
Evaluates the untouched Qwen3-4B base model and the adapted model
on the same frozen test set. Measures domain improvement and
general-capability regression.

Usage:
    python3 evaluate.py --test-set benchmarks/jupiter_seed_4b/test_sets/diagnostic_test.jsonl
                        --base-model Qwen/Qwen3-4B
                        --adapter-path artifacts/seed4b_diagnostic/final_adapter
                        --output artifacts/evaluation_results.json
    python3 evaluate.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_MODEL_ID = "Qwen/Qwen3-4B"

EVALUATION_DIMENSIONS = [
    "arabic_instruction_following",
    "english_instruction_following",
    "arabic_english_translation",
    "gcc_banking_terminology",
    "islamic_finance_terminology",
    "telecom_scenarios",
    "regulatory_document_summarization",
    "enterprise_correspondence",
    "hallucination_checks",
    "safety_refusal_behavior",
]

PERFORMANCE_METRICS = [
    "latency_ms",
    "tokens_per_second",
    "peak_vram_mb",
    "cpu_ram_mb",
    "quantized_model_size_mb",
]


def load_test_set(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        log.error("Test set not found: %s", path)
        sys.exit(1)
    examples = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    log.info("Loaded %d test examples from %s", len(examples), path)
    return examples


def evaluate_model(
    model_id_or_path: str,
    test_examples: List[Dict[str, Any]],
    adapter_path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run evaluation on a model. On CPU (unit tests), returns mock metrics.
    On GPU, loads the model and runs actual inference.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available or device == "cpu":
        log.warning(
            "GPU not available. Returning mock evaluation metrics for CPU smoke test. "
            "DO NOT represent these as real evaluation results."
        )
        return _mock_evaluation(model_id_or_path, test_examples, adapter_path)

    return _gpu_evaluation(model_id_or_path, test_examples, adapter_path)


def _mock_evaluation(
    model_id_or_path: str,
    test_examples: List[Dict[str, Any]],
    adapter_path: Optional[Path],
) -> Dict[str, Any]:
    """Return clearly labelled mock metrics for CPU smoke testing.

    SAFETY CONTRACT:
    - evaluation_mode must be 'MOCK'
    - authoritative must be False
    - model_loaded must be False
    - publishable must be False
    - acceptance_eligible must be False
    - Mock results MUST NOT produce PASS or authorize training, release, or comparison claims.
    """
    return {
        "model": model_id_or_path,
        "adapter": str(adapter_path) if adapter_path else None,
        "num_examples": len(test_examples),
        # --- SAFETY FIELDS (must never be changed to True by software) ---
        "evaluation_mode": "MOCK",
        "authoritative": False,
        "model_loaded": False,
        "publishable": False,
        "acceptance_eligible": False,
        # --- Scores are None; must never be used for comparison claims ---
        "domain_scores": {dim: None for dim in EVALUATION_DIMENSIONS},
        "performance_metrics": {m: None for m in PERFORMANCE_METRICS},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "MOCK RESULTS ONLY. CPU smoke test. No model was loaded. "
            "These results are not authoritative and must not be used for "
            "training decisions, release authorization, or comparison claims. "
            "Real evaluation requires GPU execution with a loaded model."
        ),
    }


def _gpu_evaluation(
    model_id_or_path: str,
    test_examples: List[Dict[str, Any]],
    adapter_path: Optional[Path],
) -> Dict[str, Any]:
    """Run actual GPU evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading model for evaluation: %s", model_id_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    if adapter_path and adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        log.info("Loaded adapter from %s", adapter_path)

    model.eval()
    domain_correct: Dict[str, int] = {dim: 0 for dim in EVALUATION_DIMENSIONS}
    domain_total: Dict[str, int] = {dim: 0 for dim in EVALUATION_DIMENSIONS}
    latencies: List[float] = []
    peak_vram = 0

    with torch.no_grad():
        for ex in test_examples:
            dimension = ex.get("dimension", "arabic_instruction_following")
            if dimension not in domain_total:
                continue
            domain_total[dimension] += 1

            inputs = tokenizer(
                ex["instruction"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            if torch.cuda.is_available():
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() // (1024 * 1024))

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Simple correctness check: expected answer substring match
            if ex.get("expected_answer", "") in response:
                domain_correct[dimension] += 1

    domain_scores = {
        dim: round(domain_correct[dim] / max(domain_total[dim], 1), 3)
        for dim in EVALUATION_DIMENSIONS
    }
    avg_latency = sum(latencies) / max(len(latencies), 1)
    tokens_per_sec = 128 / (avg_latency / 1000) if avg_latency > 0 else 0

    import subprocess
    def _git_commit() -> str:
        try:
            r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return r.stdout.strip()
        except Exception:
            return "unknown"

    return {
        "model": model_id_or_path,
        "model_revision": "main",
        "adapter": str(adapter_path) if adapter_path else None,
        "num_examples": len(test_examples),
        # --- SAFETY FIELDS ---
        "evaluation_mode": "GPU",
        "authoritative": True,
        "model_loaded": True,
        "publishable": False,  # Requires human sign-off before publication
        "acceptance_eligible": True,
        # --- Provenance ---
        "device": str(model.device),
        "precision": "bfloat16",
        "quantization": "none" if adapter_path is None else "qlora_adapter",
        "git_commit": _git_commit(),
        "run_id": f"eval-{int(time.time())}",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # --- Results ---
        "domain_scores": domain_scores,
        "performance_metrics": {
            "latency_ms": round(avg_latency, 2),
            "tokens_per_second": round(tokens_per_sec, 2),
            "peak_vram_mb": peak_vram,
            "cpu_ram_mb": None,
            "quantized_model_size_mb": None,
        },
    }


def compare_results(base: Dict[str, Any], adapted: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-dimension delta between base and adapted model."""
    deltas: Dict[str, Optional[float]] = {}
    for dim in EVALUATION_DIMENSIONS:
        b = base["domain_scores"].get(dim)
        a = adapted["domain_scores"].get(dim)
        deltas[dim] = round(a - b, 3) if (a is not None and b is not None) else None

    return {
        "base_model": base["model"],
        "adapted_model": adapted["model"],
        "adapter": adapted.get("adapter"),
        "per_dimension_delta": deltas,
        "overall_delta": (
            round(
                sum(v for v in deltas.values() if v is not None)
                / max(sum(1 for v in deltas.values() if v is not None), 1),
                3,
            )
        ),
        "evaluation_mode": base.get("evaluation_mode"),
        "note": (
            "CPU mock results are not valid for release decisions. "
            "GPU evaluation required."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-set", type=Path, required=True,
        help="Path to frozen test set JSONL.",
    )
    parser.add_argument(
        "--base-model", type=str, default=BASE_MODEL_ID,
        help="Base model ID or path.",
    )
    parser.add_argument(
        "--adapter-path", type=Path, default=None,
        help="Path to the trained LoRA adapter.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/evaluation_results.json"),
        help="Output path for evaluation results JSON.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_examples = load_test_set(args.test_set)
    device = "cpu" if args.device == "cpu" else "auto"

    log.info("Evaluating base model: %s", args.base_model)
    base_results = evaluate_model(args.base_model, test_examples, adapter_path=None, device=device)

    adapted_results = None
    comparison = None
    if args.adapter_path:
        log.info("Evaluating adapted model with adapter: %s", args.adapter_path)
        adapted_results = evaluate_model(
            args.base_model, test_examples, adapter_path=args.adapter_path, device=device
        )
        comparison = compare_results(base_results, adapted_results)

    output = {
        "base": base_results,
        "adapted": adapted_results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    log.info("Evaluation results written to %s", args.output)


if __name__ == "__main__":
    main()
