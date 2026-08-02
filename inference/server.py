"""
Jupiter Shot — Inference Server
================================
FastAPI-based inference server with OpenAI-compatible API endpoints.
Supports both dense and MoE models, with optional INT8 quantization.

Endpoints:
  POST /v1/completions       — Text completion (OpenAI-compatible)
  POST /v1/chat/completions  — Chat completion (OpenAI-compatible)
  GET  /v1/models            — List available models
  GET  /health               — Health check
  GET  /metrics              — Prometheus-compatible metrics

Usage:
    # Start server:
    python inference/server.py \
        --model-path checkpoints/dense_1b3/final_step_100000 \
        --model-type dense \
        --model-config training/configs/dense_1b3.yaml \
        --port 8000

    # With INT8 quantization:
    python inference/server.py \
        --model-path checkpoints/dense_1b3/final_step_100000 \
        --model-config training/configs/dense_1b3.yaml \
        --quantize int8

    # Test:
    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "jupiter-dense-1b3", "prompt": "The capital of France is", "max_tokens": 50}'
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "jupiter-dense-1b3"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    echo: bool = False
    n: int = Field(default=1, ge=1, le=4)


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "jupiter-dense-1b3"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=4)


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str
    logprobs: Optional[dict] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: dict


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


# ── Model Manager ─────────────────────────────────────────────────────────────

class ModelManager:
    """Manages model loading, tokenization, and generation."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self._request_count = 0
        self._total_tokens_generated = 0
        self._load_time = None

    def load(
        self,
        model_path: str,
        model_type: str = "dense",
        model_config: Optional[str] = None,
        quantize: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """Load model and tokenizer."""
        from training.evaluate import load_model_for_eval

        t0 = time.time()
        logger.info(f"Loading {model_type} model from {model_path}")

        self.model, self.tokenizer = load_model_for_eval(
            model_path, model_type, model_config, device
        )
        self.device = device

        if quantize == "int8":
            from inference.quantization import quantize_int8_dynamic
            logger.info("Applying INT8 dynamic quantization...")
            self.model = quantize_int8_dynamic(self.model)

        self.model.eval()
        self.model_name = f"jupiter-{model_type}-{model_path.split('/')[-2]}"
        self._load_time = time.time() - t0
        logger.info(f"Model loaded in {self._load_time:.1f}s on {device}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> tuple[str, int, str]:
        """
        Generate text from a prompt.

        Returns:
            Tuple of (generated_text, num_tokens_generated, finish_reason)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long
        ).to(self.device)

        generated_ids = []
        finish_reason = "length"

        for _ in range(max_new_tokens):
            out = self.model(input_ids=input_ids)
            logits = out["logits"][0, -1, :]  # (vocab_size,)

            # Repetition penalty
            if repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < top_k_vals[-1]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter(
                    0, sorted_indices, sorted_logits
                )

            # Sample or greedy
            if temperature == 0.0:
                next_token = logits.argmax().unsqueeze(0)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            generated_ids.append(token_id)

            # Check EOS
            if token_id == self.tokenizer.eos_token_id:
                finish_reason = "stop"
                break

            # Update input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Check stop sequences
            if stop_sequences:
                decoded = self.tokenizer.decode(generated_ids)
                for stop in stop_sequences:
                    if stop in decoded:
                        finish_reason = "stop"
                        break
                if finish_reason == "stop":
                    break

        generated_text = self.tokenizer.decode(generated_ids)
        self._request_count += 1
        self._total_tokens_generated += len(generated_ids)

        return generated_text, len(generated_ids), finish_reason

    def get_metrics(self) -> dict:
        """Return server metrics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "request_count": self._request_count,
            "total_tokens_generated": self._total_tokens_generated,
            "model_load_time_seconds": self._load_time,
        }


# ── Global model manager ──────────────────────────────────────────────────────
_manager = ModelManager()


# ── FastAPI Application ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    logger.info("Jupiter Shot inference server starting up...")
    yield
    logger.info("Jupiter Shot inference server shutting down.")


app = FastAPI(
    title="Jupiter Shot Inference Server",
    description="OpenAI-compatible inference API for Jupiter Shot language models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _manager.model is not None,
        "model_name": _manager.model_name,
        "device": _manager.device,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics."""
    m = _manager.get_metrics()
    lines = [
        f"# HELP jupiter_requests_total Total inference requests",
        f"# TYPE jupiter_requests_total counter",
        f'jupiter_requests_total {m["request_count"]}',
        f"# HELP jupiter_tokens_generated_total Total tokens generated",
        f"# TYPE jupiter_tokens_generated_total counter",
        f'jupiter_tokens_generated_total {m["total_tokens_generated"]}',
    ]
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": _manager.model_name or "jupiter-dense-1b3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "jupiter-shot",
            }
        ],
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint."""
    if _manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    stop = [request.stop] if isinstance(request.stop, str) else request.stop

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, prompt in enumerate(prompts[:request.n]):
        prompt_tokens = len(_manager.tokenizer.encode(prompt))
        total_prompt_tokens += prompt_tokens

        generated_text, num_tokens, finish_reason = _manager.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=stop,
        )
        total_completion_tokens += num_tokens

        choices.append(CompletionChoice(
            text=(prompt + generated_text) if request.echo else generated_text,
            index=i,
            finish_reason=finish_reason,
        ))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=_manager.model_name or request.model,
        choices=choices,
        usage={
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    if _manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Format messages into a single prompt
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    stop = [request.stop] if isinstance(request.stop, str) else request.stop

    generated_text, num_tokens, finish_reason = _manager.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=stop,
    )

    prompt_tokens = len(_manager.tokenizer.encode(prompt))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=_manager.model_name or request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=generated_text),
                index=0,
                finish_reason=finish_reason,
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": num_tokens,
            "total_tokens": prompt_tokens + num_tokens,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Jupiter Shot Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to checkpoint directory")
    parser.add_argument("--model-type", default="dense", choices=["dense", "moe"])
    parser.add_argument("--model-config", default=None, help="Path to YAML config")
    parser.add_argument("--quantize", default=None, choices=["int8"], help="Quantization method")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model before starting server
    _manager.load(
        model_path=args.model_path,
        model_type=args.model_type,
        model_config=args.model_config,
        quantize=args.quantize,
        device=args.device,
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
