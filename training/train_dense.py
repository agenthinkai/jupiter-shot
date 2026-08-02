"""
Jupiter Shot — Dense Baseline Training Script
=============================================
Trains the 1.3B dense transformer using DeepSpeed ZeRO-2.

Usage:
    # Single GPU (Gate A):
    python training/train_dense.py --config training/configs/dense_smoke.yaml --synthetic

    # Multi-GPU with DeepSpeed (Gate B/C):
    deepspeed --num_gpus 8 training/train_dense.py \
        --config training/configs/dense_1b3.yaml \
        --deepspeed

    # Resume from checkpoint:
    deepspeed --num_gpus 8 training/train_dense.py \
        --config training/configs/dense_1b3.yaml \
        --resume checkpoints/dense_1b3/step_1000

All results are logged to stdout and optionally to TensorBoard.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jupiter.train_dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Shot — Dense Baseline Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no download required; for Gate A testing)",
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max_steps from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize model and data loader, then exit (no training)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_distributed(local_rank: int) -> tuple[int, int, torch.device]:
    """Initialize distributed training. Returns (rank, world_size, device)."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, device


def build_model_from_config(cfg: dict):
    """Build model from config dict."""
    from training.models.dense import DenseConfig, DenseTransformer, NAMED_CONFIGS
    import dataclasses

    model_cfg = cfg["model"]
    config_name = model_cfg.get("config_name", "1.3b")

    if config_name in NAMED_CONFIGS:
        base = NAMED_CONFIGS[config_name]
    else:
        base = DenseConfig()

    # Override with explicit config values
    override_fields = {
        k: v for k, v in model_cfg.items()
        if k != "config_name" and v is not None
        and k in {f.name for f in dataclasses.fields(DenseConfig)}
    }
    config = dataclasses.replace(base, **override_fields)
    return DenseTransformer(config)


def build_optimizer(model, cfg: dict):
    """Build AdamW optimizer with weight decay applied only to non-bias/norm params."""
    train_cfg = cfg["training"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    beta1 = train_cfg.get("beta1", 0.9)
    beta2 = train_cfg.get("beta2", 0.95)
    eps = train_cfg.get("epsilon", 1e-8)

    # Separate parameters: no weight decay for bias, norm weights
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name or name.endswith(".weight") and param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=(beta1, beta2), eps=eps)


def build_scheduler(optimizer, cfg: dict, last_step: int = -1):
    """Build cosine LR scheduler with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    train_cfg = cfg["training"]
    warmup_steps = train_cfg.get("warmup_steps", 2000)
    max_steps = train_cfg.get("max_steps", 100000)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_step)


def build_dataloader(cfg: dict, tokenizer, use_synthetic: bool = False):
    """Build training dataloader from config."""
    from training.data_loader import (
        SyntheticDataset, PackedSequenceDataset, DatasetSource, create_dataloader
    )

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    batch_size = train_cfg["per_device_train_batch_size"]

    if use_synthetic or data_cfg.get("use_synthetic", False):
        logger.info("Using synthetic dataset (Gate A mode)")
        dataset = SyntheticDataset(
            vocab_size=data_cfg.get("synthetic_vocab_size", 32000),
            seq_length=data_cfg["seq_length"],
            num_samples=data_cfg.get("synthetic_num_samples", 10000),
        )
    else:
        sources = [
            DatasetSource(
                name=s["name"],
                weight=s.get("weight", 1.0),
                text_field=s.get("text_field", "text"),
                hf_subset=s.get("hf_subset"),
                max_documents=s.get("max_documents"),
            )
            for s in data_cfg["sources"]
        ]
        dataset = PackedSequenceDataset(
            sources=sources,
            tokenizer=tokenizer,
            seq_length=data_cfg["seq_length"],
            shuffle_seed=data_cfg.get("shuffle_seed", 42),
            buffer_size=data_cfg.get("buffer_size", 10000),
        )

    return create_dataloader(dataset, batch_size=batch_size, num_workers=2)


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # Override config with CLI args
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    rank, world_size, device = setup_distributed(args.local_rank)
    is_main = rank == 0

    if is_main:
        logger.info(f"Jupiter Shot — Dense Training")
        logger.info(f"Config: {args.config}")
        logger.info(f"World size: {world_size}, Device: {device}")

    # Build tokenizer
    from training.tokenizer import get_tokenizer
    tok_cfg = cfg["tokenizer"]
    tokenizer = get_tokenizer(
        tok_cfg["name_or_path"],
        max_length=tok_cfg.get("max_length", 2048),
    )
    if is_main:
        logger.info(f"Tokenizer: {tokenizer}")

    # Build model
    model = build_model_from_config(cfg)
    if is_main:
        n_params = model.count_parameters()
        logger.info(f"Model: {model}")
        logger.info(f"Parameters: {n_params/1e9:.3f}B")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # DeepSpeed initialization
    if args.deepspeed:
        try:
            import deepspeed
        except ImportError:
            logger.error("DeepSpeed not installed. Run: pip install deepspeed")
            sys.exit(1)

        ds_cfg = cfg.get("deepspeed", {})
        ds_config = {
            "train_micro_batch_size_per_gpu": cfg["training"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": cfg["training"]["learning_rate"],
                    "betas": [cfg["training"].get("beta1", 0.9), cfg["training"].get("beta2", 0.95)],
                    "eps": cfg["training"].get("epsilon", 1e-8),
                    "weight_decay": cfg["training"]["weight_decay"],
                },
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_num_steps": cfg["training"].get("warmup_steps", 2000),
                    "total_num_steps": cfg["training"]["max_steps"],
                },
            },
            "zero_optimization": {
                "stage": ds_cfg.get("zero_stage", 2),
                "allgather_partitions": ds_cfg.get("allgather_partitions", True),
                "reduce_scatter": ds_cfg.get("reduce_scatter", True),
                "overlap_comm": ds_cfg.get("overlap_comm", True),
                "contiguous_gradients": ds_cfg.get("contiguous_gradients", True),
            },
            "bf16": {"enabled": cfg["training"].get("bf16", True)},
            "gradient_clipping": cfg["training"].get("max_grad_norm", 1.0),
            "steps_per_print": cfg["training"].get("log_interval", 10),
        }

        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )
    else:
        model = model.to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Build dataloader
    dataloader = build_dataloader(cfg, tokenizer, use_synthetic=args.synthetic)

    # Resume from checkpoint
    global_step = 0
    global_tokens = 0
    if args.resume:
        from training.checkpoint import load_checkpoint
        state = load_checkpoint(args.resume, model, optimizer, scheduler)
        global_step = state.get("global_step", 0)
        global_tokens = state.get("global_tokens", 0)
        if is_main:
            logger.info(f"Resumed from step {global_step} ({global_tokens/1e9:.2f}B tokens)")

    if args.dry_run:
        if is_main:
            logger.info("Dry run complete. Exiting.")
        return

    # SIGTERM handler for spot instance preemption
    _sigterm_received = [False]
    def _sigterm_handler(signum, frame):
        _sigterm_received[0] = True
        if is_main:
            logger.warning("SIGTERM received — will save emergency checkpoint after current step")
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Training loop
    train_cfg = cfg["training"]
    max_steps = train_cfg["max_steps"]
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 1000)
    output_dir = Path(train_cfg["output_dir"])
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    seq_length = cfg["data"]["seq_length"]

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step_start_time = time.time()
    accum_loss = 0.0
    accum_steps = 0

    for batch in dataloader:
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        if args.deepspeed:
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"] / grad_accum
            model.backward(loss)
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"] / grad_accum
            loss.backward()

        accum_loss += loss.item() * grad_accum
        accum_steps += 1
        global_tokens += input_ids.numel()

        if accum_steps % grad_accum == 0:
            if not args.deepspeed:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                model.step()

            global_step += 1
            avg_loss = accum_loss / grad_accum
            accum_loss = 0.0

            if is_main and global_step % log_interval == 0:
                elapsed = time.time() - step_start_time
                tokens_per_sec = (log_interval * grad_accum * input_ids.numel()) / elapsed
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
                logger.info(
                    f"step={global_step:6d} | loss={avg_loss:.4f} | "
                    f"lr={lr:.2e} | tokens/sec={tokens_per_sec:.0f} | "
                    f"tokens={global_tokens/1e9:.3f}B"
                )
                step_start_time = time.time()

            # Save checkpoint
            if is_main and global_step % save_interval == 0:
                from training.checkpoint import save_checkpoint
                save_checkpoint(
                    output_dir / f"step_{global_step}",
                    model, optimizer, scheduler,
                    global_step=global_step,
                    global_tokens=global_tokens,
                    config=cfg,
                )

            # Handle SIGTERM
            if _sigterm_received[0]:
                if is_main:
                    logger.warning(f"Saving emergency checkpoint at step {global_step}")
                    from training.checkpoint import save_checkpoint
                    save_checkpoint(
                        output_dir / f"emergency_step_{global_step}",
                        model, optimizer, scheduler,
                        global_step=global_step,
                        global_tokens=global_tokens,
                        config=cfg,
                    )
                break

    if is_main:
        logger.info(f"Training complete at step {global_step} ({global_tokens/1e9:.2f}B tokens)")
        from training.checkpoint import save_checkpoint
        save_checkpoint(
            output_dir / f"final_step_{global_step}",
            model, optimizer, scheduler,
            global_step=global_step,
            global_tokens=global_tokens,
            config=cfg,
        )


if __name__ == "__main__":
    args = parse_args()
    train(args)
