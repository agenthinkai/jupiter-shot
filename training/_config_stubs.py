"""
Pure-Python stubs for DenseConfig and MoEConfig.

These stubs are used ONLY when torch is not installed (e.g., in the sandbox
test environment). They have identical field names and defaults to the real
dataclasses in training/models/dense.py and training/models/moe.py.

When torch IS installed, the real dataclasses are used instead.
Do not use these stubs for model construction — they have no forward() method.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DenseConfig:
    """Stub matching training.models.dense.DenseConfig (no torch dependency)."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    ffn_type: str = "swiglu"
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.intermediate_size is None:
            raw = int(2 / 3 * 4 * self.hidden_size)
            self.intermediate_size = (raw + 255) // 256 * 256


@dataclass
class MoEConfig:
    """Stub matching training.models.moe.MoEConfig (no torch dependency)."""
    base: DenseConfig = field(
        default_factory=lambda: DenseConfig(
            vocab_size=32000, hidden_size=1024, num_layers=12,
            num_attention_heads=8, max_position_embeddings=2048
        )
    )
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_aux_loss_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001
    use_shared_expert: bool = False
    expert_dropout: float = 0.0
    normalize_router_probs: bool = True
