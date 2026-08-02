"""Jupiter Shot — Checkpoint Unit Tests"""
import pytest
import tempfile
from pathlib import Path


class TestCheckpointManager:
    """Torch-dependent tests — skipped if torch not available."""

    def _make_model(self):
        import torch
        from training.models.dense import DenseConfig, DenseTransformer
        cfg = DenseConfig(vocab_size=100, hidden_size=32, num_layers=2,
                          num_attention_heads=4, max_position_embeddings=64,
                          gradient_checkpointing=False)
        return DenseTransformer(cfg)

    def test_save_creates_files(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        from training.checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            step_dir = Path(tmpdir) / "step_0000100"
            ckpt_path = save_checkpoint(
                checkpoint_dir=step_dir, model=model,
                optimizer=optimizer, scheduler=scheduler,
                global_step=100, global_tokens=100_000)
            assert ckpt_path.exists()
            assert (ckpt_path / "model.pt").exists()
            assert (ckpt_path / "optimizer.pt").exists()
            assert (ckpt_path / "training_state.json").exists()

    def test_load_restores_weights(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        from training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            with torch.no_grad():
                for p in model.parameters(): p.fill_(1.0)
            step_dir = Path(tmpdir) / "step_0000050"
            save_checkpoint(checkpoint_dir=step_dir, model=model,
                            optimizer=optimizer, scheduler=scheduler,
                            global_step=50, global_tokens=50_000)
            with torch.no_grad():
                for p in model.parameters(): p.fill_(0.0)
            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None, f"find_latest_checkpoint returned None for {tmpdir}"
            meta = load_checkpoint(checkpoint_dir=latest, model=model,
                                   optimizer=optimizer, scheduler=scheduler,
                                   verify_integrity=False)
            for p in model.parameters():
                assert torch.all(p == 1.0), "Weights not restored"
            assert meta["global_step"] == 50

    def test_rotate_checkpoints(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        from training.checkpoint import save_checkpoint, rotate_checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 200, 300, 400]:
                step_dir = Path(tmpdir) / f"step_{step:07d}"
                save_checkpoint(checkpoint_dir=step_dir, model=model,
                                optimizer=optimizer, scheduler=scheduler,
                                global_step=step, global_tokens=step * 1000)
            rotate_checkpoints(tmpdir, keep_last_n=2)
            ckpt_dirs = [d for d in Path(tmpdir).iterdir()
                         if d.is_dir() and d.name.startswith("step_")]
            assert len(ckpt_dirs) <= 2

    def test_find_latest_returns_most_recent(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        from training.checkpoint import save_checkpoint, find_latest_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 200, 300]:
                step_dir = Path(tmpdir) / f"step_{step:07d}"
                save_checkpoint(checkpoint_dir=step_dir, model=model,
                                optimizer=optimizer, scheduler=scheduler,
                                global_step=step, global_tokens=step * 1000)
            latest = find_latest_checkpoint(tmpdir)
            assert latest is not None, f"find_latest_checkpoint returned None for {tmpdir}"
            meta = load_checkpoint(checkpoint_dir=latest, model=model,
                                   optimizer=optimizer, scheduler=scheduler,
                                   verify_integrity=False)
            assert meta["global_step"] == 300

    def test_no_checkpoint_returns_none(self):
        from training.checkpoint import find_latest_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_latest_checkpoint(tmpdir) is None

    def test_metadata_contains_step(self):
        try:
            import torch
            import json
        except ImportError:
            pytest.skip("torch not available")
        model = self._make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        from training.checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            step_dir = Path(tmpdir) / "step_0000042"
            ckpt_path = save_checkpoint(checkpoint_dir=step_dir, model=model,
                                        optimizer=optimizer, scheduler=scheduler,
                                        global_step=42, global_tokens=42_000)
            with open(ckpt_path / "training_state.json") as f:
                meta = json.load(f)
            assert meta["global_step"] == 42
            assert meta["global_tokens"] == 42_000


class TestGradientAccumulation:
    def test_accumulated_equals_full_batch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from training.models.dense import DenseConfig, DenseTransformer
        cfg = DenseConfig(vocab_size=100, hidden_size=32, num_layers=2,
                          num_attention_heads=4, max_position_embeddings=64,
                          gradient_checkpointing=False)
        model_full = DenseTransformer(cfg)
        model_accum = DenseTransformer(cfg)
        model_accum.load_state_dict(model_full.state_dict())
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (4, 16))
        labels = torch.randint(0, 100, (4, 16))
        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.0)
        opt_full.zero_grad()
        model_full(input_ids=input_ids, labels=labels)["loss"].backward()
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.0)
        opt_accum.zero_grad()
        for i in range(2):
            out = model_accum(input_ids=input_ids[i*2:(i+1)*2],
                              labels=labels[i*2:(i+1)*2])
            (out["loss"] / 2).backward()
        for (n1, p1), (_, p2) in zip(model_full.named_parameters(),
                                      model_accum.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-5), f"Grad mismatch: {n1}"
