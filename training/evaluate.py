"""
Jupiter Shot — Evaluation Suite
================================
Evaluates trained models using the lm-evaluation-harness (EleutherAI).
Supports both dense and MoE models.

Supported tasks (Month 1):
  - hellaswag: Common sense reasoning (acc, acc_norm)
  - lambada_openai: Language modeling (acc, ppl)
  - mmlu: Massive Multitask Language Understanding (acc)
  - humaneval: Code generation (pass@1) — requires separate execution sandbox
  - winogrande: Commonsense reasoning (acc)
  - arc_easy, arc_challenge: ARC reasoning (acc, acc_norm)

Usage:
    # Evaluate a checkpoint:
    python training/evaluate.py \
        --model-path checkpoints/dense_1b3/step_10000 \
        --model-type dense \
        --tasks hellaswag lambada_openai \
        --output-dir results/eval_step10000

    # Evaluate with custom model config:
    python training/evaluate.py \
        --model-path checkpoints/moe_prototype/step_5000 \
        --model-type moe \
        --model-config training/configs/moe_prototype.yaml \
        --tasks hellaswag \
        --num-fewshot 0

Note: lm-eval-harness wraps models in its own interface.
This script provides a HuggingFace-compatible wrapper for Jupiter models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)


class JupiterLMWrapper:
    """
    Wrapper that makes Jupiter models compatible with lm-evaluation-harness.

    lm-eval expects a model with:
      - loglikelihood(requests): compute log-likelihood of continuations
      - loglikelihood_rolling(requests): compute rolling log-likelihood
      - generate_until(requests): generate text until stop condition

    This wrapper adapts JupiterDenseTransformer and JupiterMoETransformer
    to the lm-eval interface.
    """

    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
        device: str = "cuda",
        max_length: int = 2048,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._device = device
        self.max_length = max_length
        self.model.eval()

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tok_encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        """Run model forward pass and return logits."""
        with torch.no_grad():
            out = self.model(input_ids=inps.to(self._device))
            return out["logits"].cpu()

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of (context, continuation) pairs.
        Returns list of (log_likelihood, is_greedy) tuples.
        """
        results = []
        for context, continuation in requests:
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)

            # Concatenate context + continuation
            all_tokens = ctx_tokens + cont_tokens
            if len(all_tokens) > self.max_length:
                all_tokens = all_tokens[-self.max_length:]

            input_ids = torch.tensor([all_tokens], dtype=torch.long)
            logits = self._model_call(input_ids)  # (1, seq, vocab)

            # Compute log-likelihood of continuation tokens
            cont_start = len(ctx_tokens) - (len(all_tokens) - len(ctx_tokens + cont_tokens))
            cont_end = cont_start + len(cont_tokens)

            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            cont_log_probs = []
            for i, token_id in enumerate(cont_tokens):
                pos = cont_start + i - 1  # -1 because logits predict next token
                if 0 <= pos < log_probs.shape[0]:
                    cont_log_probs.append(log_probs[pos, token_id].item())

            ll = sum(cont_log_probs)

            # Check if greedy (argmax matches continuation)
            greedy = True
            for i, token_id in enumerate(cont_tokens):
                pos = cont_start + i - 1
                if 0 <= pos < logits.shape[1]:
                    if logits[0, pos].argmax().item() != token_id:
                        greedy = False
                        break

            results.append((ll, greedy))

        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute rolling log-likelihood (for perplexity computation)."""
        results = []
        for (text,) in requests:
            tokens = self.tok_encode(text)
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length:]

            input_ids = torch.tensor([tokens], dtype=torch.long)
            logits = self._model_call(input_ids)

            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            ll = sum(
                log_probs[i, tokens[i + 1]].item()
                for i in range(len(tokens) - 1)
            )
            results.append(ll)

        return results

    def generate_until(self, requests) -> list[str]:
        """Generate text until stop condition (for generative tasks)."""
        results = []
        for context, gen_kwargs in requests:
            tokens = self.tok_encode(context)
            max_new_tokens = gen_kwargs.get("max_gen_toks", 256)
            stop_sequences = gen_kwargs.get("until", [])

            input_ids = torch.tensor([tokens], dtype=torch.long).to(self._device)

            with torch.no_grad():
                generated = []
                for _ in range(max_new_tokens):
                    out = self.model(input_ids=input_ids)
                    next_token = out["logits"][0, -1].argmax().item()
                    generated.append(next_token)
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_token]], device=self._device)
                    ], dim=1)

                    if next_token == self.eot_token_id:
                        break

                    # Check stop sequences
                    decoded = self.tok_decode(generated)
                    if any(stop in decoded for stop in stop_sequences):
                        break

            results.append(self.tok_decode(generated))

        return results


def load_model_for_eval(
    model_path: str,
    model_type: str = "dense",
    model_config: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load a Jupiter model from a checkpoint for evaluation.

    Args:
        model_path: Path to checkpoint directory.
        model_type: 'dense' or 'moe'.
        model_config: Path to YAML config (optional; uses checkpoint config if available).
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    from training.checkpoint import load_checkpoint

    # Load config
    cfg = None
    if model_config:
        with open(model_config) as f:
            cfg = yaml.safe_load(f)
    else:
        # Try to load config from checkpoint
        state_path = Path(model_path) / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            cfg = state.get("config_snapshot")

    if cfg is None:
        raise ValueError(
            f"No config found. Provide --model-config or ensure checkpoint contains config_snapshot."
        )

    # Build model
    if model_type == "dense":
        from training.train_dense import build_model_from_config
        model = build_model_from_config(cfg)
    elif model_type == "moe":
        from training.train_moe import build_moe_model_from_config
        model = build_moe_model_from_config(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'dense' or 'moe'.")

    # Load weights
    load_checkpoint(model_path, model, strict=True)
    model = model.to(device)
    model.eval()

    # Build tokenizer
    from training.tokenizer import get_tokenizer
    tok_cfg = cfg.get("tokenizer", {})
    tokenizer = get_tokenizer(
        tok_cfg.get("name_or_path", "gpt-neox"),
        max_length=tok_cfg.get("max_length", 2048),
    )

    return model, tokenizer


def run_evaluation(
    model_path: str,
    model_type: str = "dense",
    model_config: Optional[str] = None,
    tasks: list[str] = None,
    num_fewshot: int = 0,
    batch_size: int = 8,
    output_dir: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Run evaluation using lm-evaluation-harness.

    Args:
        model_path: Path to checkpoint directory.
        model_type: 'dense' or 'moe'.
        model_config: Path to YAML config.
        tasks: List of task names (e.g., ['hellaswag', 'lambada_openai']).
        num_fewshot: Number of few-shot examples.
        batch_size: Evaluation batch size.
        output_dir: Directory to save results.
        device: Device for evaluation.

    Returns:
        Dict with evaluation results.
    """
    if tasks is None:
        tasks = ["hellaswag", "lambada_openai"]

    try:
        import lm_eval
        from lm_eval import evaluator
    except ImportError:
        raise ImportError(
            "lm-eval not installed. Run: pip install lm-eval==0.4.2"
        )

    logger.info(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_eval(model_path, model_type, model_config, device)

    wrapper = JupiterLMWrapper(model, tokenizer, batch_size=batch_size, device=device)

    logger.info(f"Running evaluation on tasks: {tasks}")
    results = evaluator.evaluate(
        lm=wrapper,
        task_manager=lm_eval.tasks.TaskManager(),
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    # Format results
    output = {
        "model_path": model_path,
        "model_type": model_type,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "results": results.get("results", {}),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    for task, task_results in output["results"].items():
        logger.info(f"\n{task}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jupiter Shot — Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path to checkpoint directory")
    parser.add_argument("--model-type", default="dense", choices=["dense", "moe"])
    parser.add_argument("--model-config", default=None, help="Path to YAML config")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["hellaswag", "lambada_openai"],
        help="Evaluation tasks",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_evaluation(
        model_path=args.model_path,
        model_type=args.model_type,
        model_config=args.model_config,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )
