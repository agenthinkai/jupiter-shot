"""
Jupiter Shot — Quantization Foundation
=======================================
Provides INT8 and INT4 post-training quantization (PTQ) for dense and MoE models.

Supported quantization methods:
  - INT8 dynamic quantization (PyTorch native): fastest, no calibration needed
  - INT8 static quantization (PyTorch native): requires calibration dataset
  - GPTQ-style INT4 weight-only quantization: best quality/size tradeoff
  - BitsAndBytes INT8/INT4: drop-in for HuggingFace-compatible models

Month 1 scope: INT8 dynamic quantization only (no calibration required).
INT4 GPTQ is scaffolded for Month 2.

Usage:
    # Quantize a checkpoint to INT8:
    python inference/quantization.py \
        --model-path checkpoints/dense_1b3/final_step_100000 \
        --model-type dense \
        --model-config training/configs/dense_1b3.yaml \
        --method int8_dynamic \
        --output-path checkpoints/dense_1b3_int8

    # Benchmark quantized vs. original:
    python inference/quantization.py \
        --model-path checkpoints/dense_1b3/final_step_100000 \
        --model-config training/configs/dense_1b3.yaml \
        --benchmark

References:
  - GPTQ (Frantar et al., 2022): https://arxiv.org/abs/2210.17323
  - LLM.int8() (Dettmers et al., 2022): https://arxiv.org/abs/2208.07339
  - SmoothQuant (Xiao et al., 2022): https://arxiv.org/abs/2211.10438
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── INT8 Dynamic Quantization ─────────────────────────────────────────────────

def quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to a model.

    Dynamic quantization computes scale/zero-point at runtime from actual
    activations. No calibration dataset required. Works well for LLMs where
    activations vary significantly across inputs.

    Quantizes: nn.Linear layers (weights to INT8, activations dynamically)
    Leaves unchanged: nn.Embedding, nn.LayerNorm, RMSNorm, attention ops

    Args:
        model: Trained model to quantize.

    Returns:
        Quantized model (in-place modification + return).
    """
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    logger.info("Applied INT8 dynamic quantization to all nn.Linear layers")
    return quantized


# ── INT8 Static Quantization ──────────────────────────────────────────────────

class QuantizationCalibrator:
    """
    Calibration helper for static INT8 quantization.

    Runs the model on a representative dataset to compute
    activation statistics (min/max or percentile) for each layer.
    """

    def __init__(self, model: nn.Module, num_calibration_batches: int = 100) -> None:
        self.model = model
        self.num_calibration_batches = num_calibration_batches

    def prepare(self) -> nn.Module:
        """Prepare model for calibration (insert observers)."""
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(self.model, inplace=True)
        return self.model

    def calibrate(self, dataloader, device: str = "cpu") -> None:
        """Run calibration forward passes."""
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.num_calibration_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                self.model(input_ids=input_ids)
        logger.info(f"Calibration complete ({self.num_calibration_batches} batches)")

    def convert(self) -> nn.Module:
        """Convert calibrated model to quantized INT8."""
        torch.quantization.convert(self.model, inplace=True)
        logger.info("Converted to static INT8 quantization")
        return self.model


# ── GPTQ-style INT4 Quantization (Scaffold for Month 2) ──────────────────────

class GPTQQuantizer:
    """
    GPTQ-style INT4 weight-only quantization.

    Month 1: Scaffold only. Full implementation in Month 2.
    Uses the Hessian-based optimal quantization from Frantar et al. (2022).

    Key parameters:
      - group_size: Number of weights per quantization group (128 or 64)
      - bits: Quantization bits (4 for INT4)
      - actorder: Whether to use activation-order permutation
      - percdamp: Percentage of average Hessian diagonal for dampening

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = False,
        percdamp: float = 0.01,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder
        self.percdamp = percdamp
        self._hessians: dict[str, torch.Tensor] = {}

    def add_batch(self, name: str, inp: torch.Tensor) -> None:
        """
        Accumulate Hessian statistics for a layer.
        Called during calibration forward passes.
        """
        # Month 2 implementation: accumulate H = 2 * X^T X
        raise NotImplementedError(
            "GPTQ quantization is scaffolded for Month 2. "
            "Use INT8 dynamic quantization for Month 1."
        )

    def quantize_layer(self, layer: nn.Linear) -> nn.Linear:
        """
        Quantize a single linear layer using GPTQ.
        Month 2 implementation.
        """
        raise NotImplementedError("GPTQ quantization: Month 2 deliverable.")

    def quantize_model(self, model: nn.Module, dataloader) -> nn.Module:
        """
        Quantize entire model using GPTQ.
        Month 2 implementation.
        """
        raise NotImplementedError("GPTQ quantization: Month 2 deliverable.")


# ── Quantization Benchmarking ─────────────────────────────────────────────────

def benchmark_model(
    model: nn.Module,
    tokenizer,
    num_warmup: int = 5,
    num_benchmark: int = 20,
    seq_length: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    """
    Benchmark model inference latency, throughput, and memory.

    Args:
        model: Model to benchmark.
        tokenizer: Tokenizer for vocabulary size.
        num_warmup: Number of warmup iterations.
        num_benchmark: Number of benchmark iterations.
        seq_length: Input sequence length.
        batch_size: Batch size.
        device: Device to run on.

    Returns:
        Dict with latency, throughput, and memory statistics.
    """
    model = model.to(device)
    model.eval()

    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=input_ids)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    memory_before = torch.cuda.memory_allocated(device) if device == "cuda" else 0

    with torch.no_grad():
        for _ in range(num_benchmark):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    memory_after = torch.cuda.memory_allocated(device) if device == "cuda" else 0

    latency_ms = sorted(latencies)
    tokens_per_sec = (batch_size * seq_length) / (sum(latencies) / len(latencies) / 1000)

    return {
        "latency_mean_ms": sum(latency_ms) / len(latency_ms),
        "latency_p50_ms": latency_ms[len(latency_ms) // 2],
        "latency_p95_ms": latency_ms[int(len(latency_ms) * 0.95)],
        "latency_p99_ms": latency_ms[int(len(latency_ms) * 0.99)],
        "tokens_per_sec": tokens_per_sec,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "memory_delta_mb": (memory_after - memory_before) / 1e6,
        "device": device,
        "num_iterations": num_benchmark,
    }


def compare_quantization_methods(
    model_path: str,
    model_type: str = "dense",
    model_config: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Compare original BF16 vs INT8 dynamic quantization.

    Returns comparison dict with latency, memory, and model size.
    """
    from training.evaluate import load_model_for_eval

    results = {}

    # Load original model
    logger.info("Loading original BF16 model...")
    model_bf16, tokenizer = load_model_for_eval(model_path, model_type, model_config, device)

    # Model size
    def model_size_mb(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6

    results["bf16"] = {
        "model_size_mb": model_size_mb(model_bf16),
        "dtype": "bfloat16",
    }

    # INT8 dynamic
    logger.info("Applying INT8 dynamic quantization...")
    model_int8 = quantize_int8_dynamic(model_bf16)
    results["int8_dynamic"] = {
        "model_size_mb": model_size_mb(model_int8),
        "dtype": "int8",
        "compression_ratio": model_size_mb(model_bf16) / model_size_mb(model_int8),
    }

    # Benchmark both
    if torch.cuda.is_available() or device == "cpu":
        logger.info("Benchmarking BF16 model...")
        results["bf16"]["benchmark"] = benchmark_model(
            model_bf16, tokenizer, device=device
        )
        logger.info("Benchmarking INT8 model...")
        results["int8_dynamic"]["benchmark"] = benchmark_model(
            model_int8, tokenizer, device=device
        )

    return results


def save_quantized_model(
    model: nn.Module,
    output_path: str,
    method: str = "int8_dynamic",
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save a quantized model to disk.

    Args:
        model: Quantized model.
        output_path: Directory to save to.
        method: Quantization method name (for metadata).
        metadata: Additional metadata to save.

    Returns:
        Path to saved model.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model_quantized.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "quantization_method": method,
        "pytorch_version": torch.__version__,
    }
    if metadata:
        meta.update(metadata)

    with open(output_path / "quantization_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Quantized model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jupiter Shot — Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="dense", choices=["dense", "moe"])
    parser.add_argument("--model-config", default=None)
    parser.add_argument(
        "--method",
        default="int8_dynamic",
        choices=["int8_dynamic", "int8_static"],
        help="Quantization method (int4_gptq requires Month 2 implementation)",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.benchmark:
        results = compare_quantization_methods(
            args.model_path, args.model_type, args.model_config, args.device
        )
        print(json.dumps(results, indent=2, default=str))
    else:
        from training.evaluate import load_model_for_eval
        model, tokenizer = load_model_for_eval(
            args.model_path, args.model_type, args.model_config, args.device
        )

        if args.method == "int8_dynamic":
            model = quantize_int8_dynamic(model)
        elif args.method == "int8_static":
            raise NotImplementedError("Static INT8 requires calibration dataset. Use int8_dynamic.")

        if args.output_path:
            save_quantized_model(model, args.output_path, method=args.method)
        else:
            logger.info("No --output-path specified. Use --output-path to save the quantized model.")
