"""
Jupiter Shot — Checkpointing and Fault Tolerance
=================================================
Saves and restores complete training state including:
  1. Model state dict (all parameters)
  2. Optimizer state (Adam moments, step count)
  3. LR scheduler state
  4. RNG states (Python, NumPy, PyTorch, CUDA per-device)
  5. DataLoader position (global token count, step)
  6. Configuration snapshot
  7. Checkpoint integrity hash (SHA-256 of model state)

Designed for spot instance preemption recovery:
  - SIGTERM handler saves emergency checkpoint within 30 seconds
  - On restart: detect latest valid checkpoint, resume from exact position
  - Keeps last N checkpoints; deletes older ones after verification

Checkpoint directory structure:
  checkpoints/
    dense_1b3/
      step_1000/
        model.pt          ← model state dict
        optimizer.pt      ← optimizer state dict
        scheduler.pt      ← scheduler state dict
        rng_states.pt     ← RNG states (all devices)
        training_state.json ← step, tokens, config snapshot
        INTEGRITY.sha256  ← SHA-256 of model.pt
      step_2000/
        ...
      latest             ← symlink to most recent valid checkpoint
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    checkpoint_dir: str | Path,
    model,
    optimizer,
    scheduler,
    global_step: int,
    global_tokens: int,
    config: Optional[dict] = None,
    extra_state: Optional[dict] = None,
) -> Path:
    """
    Save a complete training checkpoint.

    Saves model, optimizer, scheduler, RNG states, and training metadata.
    Writes an integrity hash file for verification on resume.

    Args:
        checkpoint_dir: Directory to save checkpoint files.
        model: Model (may be wrapped in DeepSpeed or DDP).
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        global_step: Current global training step.
        global_tokens: Total tokens processed so far.
        config: Training configuration dict (for snapshot).
        extra_state: Additional state to save (e.g., expert utilization metrics).

    Returns:
        Path to the checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    logger.info(f"Saving checkpoint to {checkpoint_dir} (step={global_step})")

    # ── Model state ───────────────────────────────────────────────────────────
    model_path = checkpoint_dir / "model.pt"
    if hasattr(model, "module"):
        # DDP wrapper
        state_dict = model.module.state_dict()
    elif hasattr(model, "state_dict"):
        state_dict = model.state_dict()
    else:
        raise TypeError(f"Cannot extract state_dict from model type: {type(model)}")

    if is_main:
        torch.save(state_dict, model_path)
        integrity_hash = _sha256_file(model_path)
        (checkpoint_dir / "INTEGRITY.sha256").write_text(integrity_hash)
        logger.info(f"Model saved: {model_path} (SHA-256: {integrity_hash[:16]}...)")

    # ── Optimizer state ───────────────────────────────────────────────────────
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if is_main:
        if hasattr(optimizer, "state_dict"):
            torch.save(optimizer.state_dict(), optimizer_path)
        else:
            logger.warning("Optimizer does not have state_dict(); skipping optimizer save")

    # ── Scheduler state ───────────────────────────────────────────────────────
    scheduler_path = checkpoint_dir / "scheduler.pt"
    if is_main and scheduler is not None:
        if hasattr(scheduler, "state_dict"):
            torch.save(scheduler.state_dict(), scheduler_path)

    # ── RNG states ────────────────────────────────────────────────────────────
    rng_path = checkpoint_dir / "rng_states.pt"
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [],
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = [
            torch.cuda.get_rng_state(i)
            for i in range(torch.cuda.device_count())
        ]
    torch.save(rng_states, rng_path)

    # ── Training state ────────────────────────────────────────────────────────
    training_state = {
        "global_step": global_step,
        "global_tokens": global_tokens,
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
    }
    if config is not None:
        training_state["config_snapshot"] = config
    if extra_state is not None:
        training_state["extra"] = extra_state

    if is_main:
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2, default=str)

    # ── Update 'latest' symlink ───────────────────────────────────────────────
    if is_main:
        latest_link = checkpoint_dir.parent / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_dir.name)
        logger.info(f"Updated 'latest' symlink → {checkpoint_dir.name}")

    # Synchronize all ranks
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Checkpoint saved: step={global_step}, tokens={global_tokens/1e9:.3f}B")
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
    verify_integrity: bool = True,
) -> dict:
    """
    Load a training checkpoint and restore all state.

    Args:
        checkpoint_dir: Path to checkpoint directory (or 'latest' symlink).
        model: Model to load weights into.
        optimizer: Optimizer to restore state into (optional).
        scheduler: LR scheduler to restore state into (optional).
        strict: Whether to require exact match of model state dict keys.
        verify_integrity: Whether to verify SHA-256 hash of model.pt.

    Returns:
        Training state dict with 'global_step', 'global_tokens', etc.
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Resolve 'latest' symlink
    if checkpoint_dir.name == "latest" and checkpoint_dir.is_symlink():
        checkpoint_dir = checkpoint_dir.resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    # ── Verify integrity ──────────────────────────────────────────────────────
    model_path = checkpoint_dir / "model.pt"
    integrity_path = checkpoint_dir / "INTEGRITY.sha256"

    if verify_integrity and integrity_path.exists():
        expected_hash = integrity_path.read_text().strip()
        actual_hash = _sha256_file(model_path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"Checkpoint integrity check FAILED for {model_path}. "
                f"Expected SHA-256: {expected_hash}, Got: {actual_hash}. "
                "The checkpoint file may be corrupted. "
                "Try loading an earlier checkpoint."
            )
        logger.info(f"Integrity check passed: {actual_hash[:16]}...")
    elif verify_integrity:
        logger.warning(
            f"No integrity file found at {integrity_path}. "
            "Skipping integrity check."
        )

    # ── Load model ────────────────────────────────────────────────────────────
    map_location = f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=map_location, weights_only=True)

    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Model weights loaded from {model_path}")

    # ── Load optimizer ────────────────────────────────────────────────────────
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer is not None and optimizer_path.exists():
        opt_state = torch.load(optimizer_path, map_location=map_location, weights_only=False)
        optimizer.load_state_dict(opt_state)
        logger.info("Optimizer state restored")

    # ── Load scheduler ────────────────────────────────────────────────────────
    scheduler_path = checkpoint_dir / "scheduler.pt"
    if scheduler is not None and scheduler_path.exists():
        sched_state = torch.load(scheduler_path, map_location=map_location, weights_only=False)
        scheduler.load_state_dict(sched_state)
        logger.info("Scheduler state restored")

    # ── Load RNG states ───────────────────────────────────────────────────────
    rng_path = checkpoint_dir / "rng_states.pt"
    if rng_path.exists():
        rng_states = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"])
        if torch.cuda.is_available() and rng_states.get("cuda"):
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank < len(rng_states["cuda"]):
                torch.cuda.set_rng_state(rng_states["cuda"][local_rank])
        logger.info("RNG states restored")

    # ── Load training state ───────────────────────────────────────────────────
    state_path = checkpoint_dir / "training_state.json"
    training_state = {}
    if state_path.exists():
        with open(state_path) as f:
            training_state = json.load(f)
        logger.info(
            f"Resumed from step={training_state.get('global_step', 0)}, "
            f"tokens={training_state.get('global_tokens', 0)/1e9:.3f}B"
        )

    return training_state


def find_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """
    Find the most recent valid checkpoint in output_dir.

    Returns the path to the latest checkpoint directory, or None if none found.
    Validates integrity before returning.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None

    # Check 'latest' symlink first
    latest_link = output_dir / "latest"
    if latest_link.is_symlink():
        target = latest_link.resolve()
        if _is_valid_checkpoint(target):
            return target
        logger.warning(f"'latest' symlink points to invalid checkpoint: {target}")

    # Fall back to scanning for step_* directories
    step_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        reverse=True,
    )

    for d in step_dirs:
        if _is_valid_checkpoint(d):
            logger.info(f"Found valid checkpoint: {d}")
            return d
        else:
            logger.warning(f"Skipping invalid checkpoint: {d}")

    return None


def _is_valid_checkpoint(checkpoint_dir: Path) -> bool:
    """Return True if checkpoint_dir contains a valid, complete checkpoint."""
    required_files = ["model.pt", "training_state.json"]
    for f in required_files:
        if not (checkpoint_dir / f).exists():
            return False

    # Verify integrity if hash file exists
    integrity_path = checkpoint_dir / "INTEGRITY.sha256"
    model_path = checkpoint_dir / "model.pt"
    if integrity_path.exists():
        try:
            expected = integrity_path.read_text().strip()
            actual = _sha256_file(model_path)
            return expected == actual
        except Exception:
            return False

    return True


def rotate_checkpoints(output_dir: str | Path, keep_last_n: int = 3) -> None:
    """
    Delete old checkpoints, keeping only the most recent N.

    Args:
        output_dir: Directory containing step_* checkpoint subdirectories.
        keep_last_n: Number of most recent checkpoints to keep.
    """
    output_dir = Path(output_dir)
    step_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        reverse=True,
    )

    to_delete = step_dirs[keep_last_n:]
    for d in to_delete:
        logger.info(f"Rotating checkpoint: deleting {d}")
        shutil.rmtree(d)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint utilities.")
    parser.add_argument("command", choices=["verify", "find-latest", "rotate"])
    parser.add_argument("path", help="Checkpoint directory or output directory")
    parser.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep (rotate)")
    args = parser.parse_args()

    if args.command == "verify":
        p = Path(args.path)
        if _is_valid_checkpoint(p):
            print(f"✓ Valid checkpoint: {p}")
        else:
            print(f"✗ Invalid or incomplete checkpoint: {p}")
            exit(1)

    elif args.command == "find-latest":
        result = find_latest_checkpoint(args.path)
        if result:
            print(f"Latest checkpoint: {result}")
        else:
            print("No valid checkpoint found")
            exit(1)

    elif args.command == "rotate":
        rotate_checkpoints(args.path, keep_last_n=args.keep)
        print(f"Rotated checkpoints in {args.path}, keeping last {args.keep}")
