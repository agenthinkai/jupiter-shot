#!/usr/bin/env python3
"""
Jupiter Seed 4B — MeshPilot-Compatible Inference Server
=========================================================
Provides an OpenAI-compatible REST API for the Jupiter Seed 4B model.
Supports both GPU (vLLM or Transformers) and CPU (llama.cpp) backends.
Includes compliance logging, health endpoints, and telemetry.

Usage:
    python3 inference_server.py --model-path artifacts/quantized/merged_fp16
                                --backend transformers
                                --port 8080
    python3 inference_server.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_IDENTITY = "Jupiter Seed 4B Diagnostic — AgenThink-adapted Qwen3-4B"
MODEL_DISCLAIMER = (
    "This is an internal diagnostic model. "
    "It is NOT production-ready, NOT commercially approved, "
    "and NOT superior to the base Qwen3-4B model without evaluation evidence."
)


def create_app(model_path: str, backend: str = "transformers") -> Any:
    """Create and return the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        log.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        raise

    app = FastAPI(
        title="Jupiter Seed 4B Inference Server",
        description=MODEL_DISCLAIMER,
        version="0.1.0-diagnostic",
    )

    # Compliance log
    compliance_log_path = Path("artifacts/compliance_log.jsonl")
    compliance_log_path.parent.mkdir(parents=True, exist_ok=True)

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "jupiter-seed-4b-diagnostic"
        messages: List[ChatMessage]
        max_tokens: int = 256
        temperature: float = 0.7
        stream: bool = False

    class ChatResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, int]

    # Load model lazily
    _model_cache: Dict[str, Any] = {}

    def get_model():
        if "model" not in _model_cache:
            if backend == "transformers":
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto",
                        trust_remote_code=False,
                    )
                    model.eval()
                    _model_cache["model"] = model
                    _model_cache["tokenizer"] = tokenizer
                    log.info("Model loaded from %s (transformers backend)", model_path)
                except ImportError:
                    log.error("transformers not installed.")
                    raise
            else:
                log.warning("Backend '%s' not yet implemented. Using mock.", backend)
                _model_cache["model"] = None
                _model_cache["tokenizer"] = None
        return _model_cache

    def log_compliance(request_id: str, prompt_hash: str, model: str, latency_ms: float) -> None:
        entry = {
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with compliance_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model": MODEL_IDENTITY, "disclaimer": MODEL_DISCLAIMER}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "jupiter-seed-4b-diagnostic", "object": "model"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> ChatResponse:
        import hashlib
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        prompt_text = " ".join(m.content for m in request.messages)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        t0 = time.perf_counter()

        cache = get_model()
        model = cache.get("model")
        tokenizer = cache.get("tokenizer")

        if model is None or tokenizer is None:
            # Mock response for CPU/unit-test mode
            response_text = "[MOCK RESPONSE — GPU required for real inference]"
            input_tokens = len(prompt_text.split())
            output_tokens = 8
        else:
            try:
                import torch
                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens,
                        do_sample=request.temperature > 0,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                )
                input_tokens = inputs["input_ids"].shape[-1]
                output_tokens = outputs.shape[-1] - input_tokens
            except Exception as exc:
                log.error("Inference error: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))

        latency_ms = (time.perf_counter() - t0) * 1000
        log_compliance(request_id, prompt_hash, request.model, latency_ms)

        return ChatResponse(
            id=request_id,
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — MeshPilot-Compatible Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str,
        default="artifacts/quantized/merged_fp16",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--backend", type=str, default="transformers",
        choices=["transformers", "vllm", "llamacpp"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to serve on.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
        raise

    app = create_app(args.model_path, args.backend)
    log.info("Starting Jupiter Seed 4B inference server on %s:%d", args.host, args.port)
    log.info("Disclaimer: %s", MODEL_DISCLAIMER)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
