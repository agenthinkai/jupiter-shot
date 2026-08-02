"""
Jupiter Shot — CPU Smoke Test (Gate 3)
Runs 20 training steps on CPU with a tiny model and synthetic data.
Records: loss per step, runtime, peak RAM, checkpoint save/resume.
"""
import sys
import os
import time
import resource
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from training.models.dense import DenseConfig, DenseTransformer
from training.data_loader import SyntheticDataset
from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

print("=" * 60)
print("JUPITER SHOT — CPU SMOKE TEST")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: CPU (no CUDA available)")
print(f"Platform: {sys.platform}")
print()

# ── Config: tiny model for CPU ────────────────────────────────────────────────
cfg = DenseConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=6,
    num_attention_heads=8,
    max_position_embeddings=512,
    gradient_checkpointing=False,
)
print(f"Model config: hidden={cfg.hidden_size}, layers={cfg.num_layers}, heads={cfg.num_attention_heads}")
print(f"Intermediate size: {cfg.intermediate_size}")

# Count params
model = DenseTransformer(cfg)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} = {total_params/1e6:.1f}M")
print(f"Trainable parameters: {trainable_params:,} = {trainable_params/1e6:.1f}M")
print()

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = SyntheticDataset(
    vocab_size=32000,
    seq_length=512,
    num_samples=200,
    seed=42,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=3e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
print("Starting training loop (20 steps on CPU)...")
print(f"{'Step':>5}  {'Loss':>8}  {'LR':>10}  {'Elapsed':>8}")
print("-" * 40)

model.train()
losses = []
start_time = time.time()
nan_count = 0
inf_count = 0

for step, batch in enumerate(dataloader):
    if step >= 20:
        break
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    loss = out["loss"]

    # Check for NaN/Inf
    if torch.isnan(loss):
        nan_count += 1
    if torch.isinf(loss):
        inf_count += 1

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_val = loss.item()
    lr_val = scheduler.get_last_lr()[0]
    elapsed = time.time() - start_time
    losses.append(loss_val)

    print(f"{step+1:>5}  {loss_val:>8.4f}  {lr_val:>10.2e}  {elapsed:>7.1f}s")

total_time = time.time() - start_time
print("-" * 40)
print(f"Training complete: {len(losses)} steps in {total_time:.1f}s")
print(f"NaN count: {nan_count}")
print(f"Inf count: {inf_count}")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
print(f"Loss trend:   {'DECREASING' if losses[-1] < losses[0] else 'NOT DECREASING'}")

# ── Peak RAM ──────────────────────────────────────────────────────────────────
peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_ram_mb = peak_ram_kb / 1024
print(f"\nPeak RAM: {peak_ram_mb:.0f} MB ({peak_ram_kb:,} KB)")

# ── Checkpoint save/resume test ───────────────────────────────────────────────
print("\n" + "-" * 40)
print("CHECKPOINT SAVE/RESUME TEST")
print("-" * 40)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save checkpoint
    step_dir = Path(tmpdir) / "step_0000020"
    t_save_start = time.time()
    ckpt_path = save_checkpoint(
        checkpoint_dir=step_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=20,
        global_tokens=20 * 2 * 512,
        config={"test": True},
    )
    save_duration = time.time() - t_save_start
    print(f"Checkpoint saved to: {ckpt_path}")
    print(f"Save duration: {save_duration:.2f}s")
    print(f"Files saved: {sorted(f.name for f in ckpt_path.iterdir())}")

    # Verify files
    assert (ckpt_path / "model.pt").exists(), "model.pt missing"
    assert (ckpt_path / "optimizer.pt").exists(), "optimizer.pt missing"
    assert (ckpt_path / "training_state.json").exists(), "training_state.json missing"
    assert (ckpt_path / "INTEGRITY.sha256").exists(), "INTEGRITY.sha256 missing"
    print("All required files present: PASS")

    # Corrupt weights, then resume
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)

    # Find latest and resume
    latest = find_latest_checkpoint(tmpdir)
    assert latest is not None, "find_latest_checkpoint returned None"

    t_load_start = time.time()
    meta = load_checkpoint(
        checkpoint_dir=latest,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        verify_integrity=True,
    )
    load_duration = time.time() - t_load_start
    print(f"Checkpoint loaded from: {latest}")
    print(f"Load duration: {load_duration:.2f}s")
    print(f"Resumed step: {meta['global_step']}")
    print(f"Resumed tokens: {meta['global_tokens']:,}")

    # Verify weights restored
    first_param = next(model.parameters())
    assert not torch.all(first_param == 999.0), "Weights were NOT restored"
    print("Weights correctly restored: PASS")
    print("Integrity verification passed: PASS")

    # Run 1 more step to confirm model is functional after resume
    model.train()
    batch = next(iter(dataloader))
    optimizer.zero_grad()
    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
    loss_after_resume = out["loss"]
    loss_after_resume.backward()
    optimizer.step()
    print(f"Post-resume forward pass: loss={loss_after_resume.item():.4f} PASS")

print("\n" + "=" * 60)
print("CPU SMOKE TEST COMPLETE")
print("=" * 60)
print(f"Status: PASSED")
print(f"Steps completed: 20/20")
print(f"NaN/Inf: {nan_count}/{inf_count}")
print(f"Total runtime: {total_time:.1f}s")
print(f"Peak RAM: {peak_ram_mb:.0f} MB")
print(f"Checkpoint save: {save_duration:.2f}s")
print(f"Checkpoint load: {load_duration:.2f}s")
print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
