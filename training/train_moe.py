"""
Jupiter Shot — MoE Prototype Training Script
============================================
Trains the sparse MoE transformer using DeepSpeed ZeRO-2.
Extends train_dense.py with MoE-specific logging (router metrics, aux loss).

Usage:
    # Single GPU (Gate A):
    python training/train_moe.py --config training/configs/moe_prototype.yaml --synthetic

    # Multi-GPU with DeepSpeed (Gate B/C):
    deepspeed --num_gpus 8 training/train_moe.py \
        --config training/configs/moe_prototype.yaml \
        --deepspeed

    # Resume from checkpoint:
    deepspeed --num_gpus 8 training/train_moe.py \
        --config training/configs/moe_prototype.yaml \
        --resume checkpoints/moe_prototype/step_1000 \
        --deepspeed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jupiter.train_moe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Shot — MoE Prototype Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_moe_model_from_config(cfg: dict):
    """Build MoE model from config dict."""
    import dataclasses
    from training.models.moe import MoEConfig, MoETransformer, MOE_NAMED_CONFIGS
    from training.models.dense import DenseConfig

    model_cfg = cfg["model"]
    config_name = model_cfg.get("config_name", "moe_1b")

    if config_name in MOE_NAMED_CONFIGS:
        config = MOE_NAMED_CONFIGS[config_name]
    else:
        config = MoEConfig()

    # Override base config fields
    base_fields = {f.name for f in dataclasses.fields(DenseConfig)}
    base_overrides = {k: v for k, v in model_cfg.items() if k in base_fields and v is not None}
    if base_overrides:
        new_base = dataclasses.replace(config.base, **base_overrides)
        config = dataclasses.replace(config, base=new_base)

    # Override MoE config fields
    moe_fields = {f.name for f in dataclasses.fields(MoEConfig)} - {"base"}
    moe_overrides = {k: v for k, v in model_cfg.items() if k in moe_fields and v is not None}
    if moe_overrides:
        config = dataclasses.replace(config, **moe_overrides)

    return MoETransformer(config)


def log_router_metrics(metrics_list: list[dict], step: int, is_main: bool) -> None:
    """Log expert utilization metrics across all MoE layers."""
    if not is_main or not metrics_list:
        return

    # Aggregate across layers
    all_imbalance = []
    all_entropy = []
    for layer_idx, metrics in enumerate(metrics_list):
        if not metrics:
            continue
        imbalance = metrics.get("load_imbalance_ratio", 1.0)
        entropy = metrics.get("router_entropy", 0.0)
        all_imbalance.append(imbalance)
        all_entropy.append(entropy)

    if all_imbalance:
        avg_imbalance = sum(all_imbalance) / len(all_imbalance)
        avg_entropy = sum(all_entropy) / len(all_entropy)
        max_imbalance = max(all_imbalance)

        logger.info(
            f"step={step:6d} | router: avg_imbalance={avg_imbalance:.2f}x "
            f"max_imbalance={max_imbalance:.2f}x avg_entropy={avg_entropy:.3f}"
        )

        # Warn if load imbalance is severe
        if max_imbalance > 4.0:
            logger.warning(
                f"HIGH LOAD IMBALANCE at step {step}: max={max_imbalance:.2f}x "
                "(target: < 2.0x). Consider increasing router_aux_loss_coeff."
            )


def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    # Setup distributed
    from training.train_dense import setup_distributed, build_optimizer, build_scheduler, build_dataloader
    rank, world_size, device = setup_distributed(args.local_rank)
    is_main = rank == 0

    if is_main:
        logger.info("Jupiter Shot — MoE Prototype Training")
        logger.info(f"Config: {args.config}")
        logger.info(f"World size: {world_size}, Device: {device}")

    # Build tokenizer
    from training.tokenizer import get_tokenizer
    tok_cfg = cfg["tokenizer"]
    tokenizer = get_tokenizer(tok_cfg["name_or_path"], max_length=tok_cfg.get("max_length", 2048))

    # Build MoE model
    model = build_moe_model_from_config(cfg)
    if is_main:
        n_total = model.count_parameters()
        n_active = model.config.count_active_parameters()
        logger.info(f"Model: {model}")
        logger.info(f"Total parameters: {n_total/1e9:.3f}B")
        logger.info(f"Active parameters per token: {n_active/1e9:.3f}B")
        logger.info(f"Activation ratio: {n_active/n_total:.1%}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    if args.deepspeed:
        try:
            import deepspeed
        except ImportError:
            logger.error("DeepSpeed not installed.")
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
            "zero_optimization": {"stage": ds_cfg.get("zero_stage", 2)},
            "bf16": {"enabled": cfg["training"].get("bf16", True)},
            "gradient_clipping": cfg["training"].get("max_grad_norm", 1.0),
        }
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config
        )
    else:
        model = model.to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    dataloader = build_dataloader(cfg, tokenizer, use_synthetic=args.synthetic)

    global_step = 0
    global_tokens = 0
    if args.resume:
        from training.checkpoint import load_checkpoint
        state = load_checkpoint(args.resume, model, optimizer, scheduler)
        global_step = state.get("global_step", 0)
        global_tokens = state.get("global_tokens", 0)
        if is_main:
            logger.info(f"Resumed from step {global_step}")

    if args.dry_run:
        if is_main:
            logger.info("Dry run complete.")
        return

    _sigterm_received = [False]
    def _sigterm_handler(signum, frame):
        _sigterm_received[0] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)

    train_cfg = cfg["training"]
    max_steps = train_cfg["max_steps"]
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 1000)
    router_metrics_interval = train_cfg.get("router_metrics_interval", 100)
    output_dir = Path(train_cfg["output_dir"])
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    log_router = train_cfg.get("log_router_metrics", True)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step_start_time = time.time()
    accum_loss = 0.0
    accum_lm_loss = 0.0
    accum_aux_loss = 0.0
    accum_steps = 0
    last_router_metrics = []

    for batch in dataloader:
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if args.deepspeed:
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"] / grad_accum
            model.backward(loss)
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"] / grad_accum
            loss.backward()

        accum_loss += loss.item() * grad_accum
        accum_lm_loss += (out.get("lm_loss") or loss).item()
        accum_aux_loss += (out.get("aux_loss") or torch.tensor(0.0)).item()
        accum_steps += 1
        global_tokens += input_ids.numel()

        if out.get("router_metrics"):
            last_router_metrics = out["router_metrics"]

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
            avg_lm_loss = accum_lm_loss / grad_accum
            avg_aux_loss = accum_aux_loss / grad_accum
            accum_loss = accum_lm_loss = accum_aux_loss = 0.0

            if is_main and global_step % log_interval == 0:
                elapsed = time.time() - step_start_time
                tokens_per_sec = (log_interval * grad_accum * input_ids.numel()) / elapsed
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
                logger.info(
                    f"step={global_step:6d} | loss={avg_loss:.4f} "
                    f"lm={avg_lm_loss:.4f} aux={avg_aux_loss:.6f} | "
                    f"lr={lr:.2e} | tok/s={tokens_per_sec:.0f} | "
                    f"tokens={global_tokens/1e9:.3f}B"
                )
                step_start_time = time.time()

            if is_main and log_router and global_step % router_metrics_interval == 0:
                log_router_metrics(last_router_metrics, global_step, is_main)

            if is_main and global_step % save_interval == 0:
                from training.checkpoint import save_checkpoint, rotate_checkpoints
                save_checkpoint(
                    output_dir / f"step_{global_step}",
                    model, optimizer, scheduler,
                    global_step=global_step,
                    global_tokens=global_tokens,
                    config=cfg,
                    extra_state={"router_metrics": last_router_metrics},
                )
                rotate_checkpoints(output_dir, keep_last_n=train_cfg.get("keep_last_n_checkpoints", 3))

            if _sigterm_received[0]:
                if is_main:
                    from training.checkpoint import save_checkpoint
                    save_checkpoint(
                        output_dir / f"emergency_step_{global_step}",
                        model, optimizer, scheduler,
                        global_step=global_step, global_tokens=global_tokens, config=cfg,
                    )
                break

    if is_main:
        logger.info(f"MoE training complete at step {global_step} ({global_tokens/1e9:.2f}B tokens)")
        from training.checkpoint import save_checkpoint
        save_checkpoint(
            output_dir / f"final_step_{global_step}",
            model, optimizer, scheduler,
            global_step=global_step, global_tokens=global_tokens, config=cfg,
        )


if __name__ == "__main__":
    args = parse_args()
    train(args)
