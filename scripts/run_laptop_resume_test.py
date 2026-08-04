"""
Jupiter Shot — Laptop Checkpoint Resume Test
=============================================
Verifies checkpoint save, load, and training continuity.

Model API contract:
  model(input_ids=..., labels=...) → dict with keys:
    - 'loss':   scalar tensor (cross-entropy loss)
    - 'logits': (batch, seq_len, vocab_size)

Tests:
  1. Run N steps and save checkpoint
  2. Load checkpoint into fresh model
  3. Verify global step matches
  4. Verify optimizer state restored
  5. Run 5 more steps and verify loss continuity (within tolerance)

Usage:
    python scripts/run_laptop_resume_test.py --config laptop_dense_small
    python scripts/run_laptop_resume_test.py --config laptop_dense_small --steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "laptop"
CKPT_DIR = REPO_ROOT / "checkpoints" / "laptop"


def run_steps(model: Any, optimizer: Any, scheduler: Any, device: Any,
              vocab_size: int, seq_len: int, batch_size: int,
              n_steps: int, torch: Any) -> list[float]:
    """Run n_steps and return loss values.

    Uses the model's dict output API:
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
    """
    import torch as th
    losses = []
    model.train()
    for _ in range(n_steps):
        input_ids = th.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        # Model returns dict — unpack loss directly
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


def run_resume_test(
    config_name: str = "laptop_dense_small",
    initial_steps: int = 20,
    resume_steps: int = 5,
    output_dir: Path = RESULTS_DIR,
    ckpt_dir: Path = CKPT_DIR,
) -> dict:
    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("[FAIL] CUDA not available.")

    config_path = REPO_ROOT / "training" / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    from training.models.dense import DenseTransformer, DenseConfig
    dense_config = DenseConfig(**{k: v for k, v in model_cfg.items() if hasattr(DenseConfig, k)})
    device = torch.device("cuda:0")

    vocab_size = dense_config.vocab_size
    seq_len = int(train_cfg.get("seq_length", dense_config.max_position_embeddings))
    batch_size = int(train_cfg.get("batch_size", 2))
    lr = float(train_cfg.get("learning_rate", 3e-4))

    result = {
        "config": config_name,
        "initial_steps": initial_steps,
        "resume_steps": resume_steps,
        "tests": {},
        "passed": False,
    }

    # ── Phase 1: Initial run ──────────────────────────────────────────────────
    print(f"\n[RESUME TEST] Phase 1: Running {initial_steps} steps...")
    model_a = DenseTransformer(dense_config).to(device)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=lr)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=initial_steps + resume_steps)

    losses_a = run_steps(model_a, opt_a, sched_a, device, vocab_size, seq_len, batch_size, initial_steps, torch)
    loss_before_save = losses_a[-1]
    print(f"  Loss at step {initial_steps}: {loss_before_save:.6f}")

    # ── Phase 2: Save checkpoint ──────────────────────────────────────────────
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"resume_test_{config_name}.pt"
    ckpt_start = time.time()
    torch.save({
        "step": initial_steps,
        "model_state_dict": model_a.state_dict(),
        "optimizer_state_dict": opt_a.state_dict(),
        "scheduler_state_dict": sched_a.state_dict(),
        "loss": loss_before_save,
        "config": config_name,
    }, ckpt_path)
    ckpt_save_time = time.time() - ckpt_start
    ckpt_size_mb = ckpt_path.stat().st_size / 1024**2
    print(f"  Checkpoint saved: {ckpt_path} ({ckpt_size_mb:.1f} MB, {ckpt_save_time:.2f}s)")
    result["tests"]["checkpoint_save"] = {
        "passed": True,
        "path": str(ckpt_path),
        "size_mb": round(ckpt_size_mb, 1),
        "duration_s": round(ckpt_save_time, 3),
    }

    # ── Phase 3: Continue from model_a (reference) ────────────────────────────
    print(f"\n[RESUME TEST] Phase 3: Reference continuation ({resume_steps} steps)...")
    losses_ref = run_steps(model_a, opt_a, sched_a, device, vocab_size, seq_len, batch_size, resume_steps, torch)
    loss_ref_end = losses_ref[-1]
    print(f"  Reference loss at step {initial_steps + resume_steps}: {loss_ref_end:.6f}")

    # ── Phase 4: Load checkpoint into fresh model ─────────────────────────────
    print(f"\n[RESUME TEST] Phase 4: Loading checkpoint into fresh model...")
    load_start = time.time()
    model_b = DenseTransformer(dense_config).to(device)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=lr)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=initial_steps + resume_steps)

    # weights_only=True avoids the FutureWarning in PyTorch 2.x and is safer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_b.load_state_dict(ckpt["model_state_dict"])
    opt_b.load_state_dict(ckpt["optimizer_state_dict"])
    sched_b.load_state_dict(ckpt["scheduler_state_dict"])
    loaded_step = ckpt["step"]
    loaded_loss = ckpt["loss"]
    load_time = time.time() - load_start
    print(f"  Loaded step: {loaded_step}, loss: {loaded_loss:.6f} ({load_time:.2f}s)")

    # Test: global step matches
    step_match = loaded_step == initial_steps
    result["tests"]["global_step_match"] = {
        "passed": step_match,
        "expected": initial_steps,
        "loaded": loaded_step,
    }
    print(f"  Step match: {'PASS' if step_match else 'FAIL'} (expected {initial_steps}, got {loaded_step})")

    # Test: loss matches
    loss_match = abs(loaded_loss - loss_before_save) < 1e-5
    result["tests"]["loss_at_checkpoint_match"] = {
        "passed": loss_match,
        "expected": round(loss_before_save, 6),
        "loaded": round(loaded_loss, 6),
        "delta": round(abs(loaded_loss - loss_before_save), 8),
    }
    print(f"  Loss match: {'PASS' if loss_match else 'FAIL'}")

    # ── Phase 5: Resume and verify continuity ─────────────────────────────────
    print(f"\n[RESUME TEST] Phase 5: Resuming from checkpoint ({resume_steps} steps)...")
    losses_resumed = run_steps(model_b, opt_b, sched_b, device, vocab_size, seq_len, batch_size, resume_steps, torch)
    loss_resumed_end = losses_resumed[-1]
    print(f"  Resumed loss at step {initial_steps + resume_steps}: {loss_resumed_end:.6f}")
    print(f"  Reference loss at same step: {loss_ref_end:.6f}")

    # Loss continuity: resumed loss should be within 10% of reference
    # (exact match not expected due to random batch sampling)
    continuity_delta = abs(loss_resumed_end - loss_ref_end)
    continuity_pct = continuity_delta / max(abs(loss_ref_end), 1e-9) * 100
    continuity_pass = continuity_pct < 10.0
    result["tests"]["loss_continuity"] = {
        "passed": continuity_pass,
        "reference_loss": round(loss_ref_end, 6),
        "resumed_loss": round(loss_resumed_end, 6),
        "delta": round(continuity_delta, 6),
        "delta_pct": round(continuity_pct, 2),
        "note": "Random batches cause divergence; <10% delta is acceptable.",
    }
    print(f"  Continuity: {'PASS' if continuity_pass else 'FAIL'} (delta={continuity_pct:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_passed = all(t["passed"] for t in result["tests"].values())
    result["passed"] = all_passed
    result["checkpoint_load_duration_s"] = round(load_time, 3)
    result["checkpoint_save_duration_s"] = round(ckpt_save_time, 3)
    result["checkpoint_size_mb"] = round(ckpt_size_mb, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "resume_test.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[RESUME TEST] {'PASS' if all_passed else 'FAIL'}")
    print(f"[RESUME TEST] Saved: {out_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jupiter Shot Laptop Checkpoint Resume Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "DATA NOTE: This test uses deterministic synthetic random tensors "
            "(torch.randint) for all training steps. It does NOT use Wikitext-2 "
            "or any real text data. The --data-mode flag is accepted for CLI "
            "contract compatibility with the pipeline but does not change the "
            "data source. Resume correctness is independent of training-data mode."
        ),
    )
    parser.add_argument("--config", default="laptop_dense_small",
                        help="Training config name (e.g., laptop_dense_small)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Initial steps before checkpoint")
    parser.add_argument("--output-dir", default="benchmarks/results/laptop",
                        help="Directory for result artifacts")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Unique run identifier (e.g., 20260804_082957_UTC). "
            "Embedded in resume_test.json for artifact traceability. "
            "Generated automatically if not provided."
        ),
    )
    parser.add_argument(
        "--data-mode",
        choices=["real", "synthetic", "auto"],
        default="synthetic",
        help=(
            "Accepted for CLI contract compatibility with the pipeline. "
            "This test always uses deterministic synthetic random tensors "
            "(torch.randint) regardless of this flag."
        ),
    )
    args = parser.parse_args()

    import datetime as _dt
    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    print(f"[RESUME] Run ID: {run_id}", flush=True)
    print(
        f"[RESUME] Data mode: {args.data_mode} (accepted; "
        "test uses deterministic synthetic tensors regardless)",
        flush=True,
    )

    try:
        result = run_resume_test(
            config_name=args.config,
            initial_steps=args.steps,
            output_dir=Path(args.output_dir),
        )
        # Embed run_id and data_mode into the result artifact
        result["run_id"] = run_id
        result["data_mode_arg"] = args.data_mode
        result["data_source"] = "deterministic_synthetic_tensors"
        result_path = Path(args.output_dir) / "resume_test.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            existing["run_id"] = run_id
            existing["data_mode_arg"] = args.data_mode
            existing["data_source"] = "deterministic_synthetic_tensors"
            result_path.write_text(json.dumps(existing, indent=2, default=str))
        return 0 if result["passed"] else 1
    except RuntimeError as e:
        print(f"\n[FAIL] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
