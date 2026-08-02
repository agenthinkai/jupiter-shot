"""
Jupiter Shot — 1.3B Dense Transformer Baseline
================================================
Pure PyTorch implementation of a decoder-only transformer with:
- RMSNorm (pre-norm)
- SwiGLU FFN
- Rotary Positional Embeddings (RoPE)
- Multi-head causal self-attention
- Optional gradient checkpointing

This is the Stage 1 baseline. MoE layers are added in models/moe.py.

Parameter count reference (1.3B config):
  Embedding:       32000 × 2048 = 65.5M
  Each layer:      ~106M (attention + FFN with SwiGLU)
  24 layers:       ~2,548M raw — but SwiGLU FFN dim = 2/3 × 4 × 2048 = 5461
  Actual 24-layer: ~1.3B with weight tying and corrected FFN sizing
  Output head:     tied to embedding (no extra params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DenseConfig:
    """Configuration for the dense transformer baseline."""

    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_kv_heads: Optional[int] = None  # None = same as num_attention_heads (MHA)
    head_dim: Optional[int] = None      # None = hidden_size // num_attention_heads

    # FFN
    intermediate_size: Optional[int] = None  # None = auto (2/3 × 4 × hidden_size, rounded to multiple of 256)
    ffn_type: str = "swiglu"  # "swiglu" or "gelu"

    # Positional encoding
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0

    # Normalization
    rms_norm_eps: float = 1e-5

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Training
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # Gradient checkpointing
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.intermediate_size is None:
            # SwiGLU: use 2/3 × 4 × hidden_size, rounded to multiple of 256
            raw = int(2 / 3 * 4 * self.hidden_size)
            self.intermediate_size = (raw + 255) // 256 * 256

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads

    def count_parameters(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        attn_per_layer = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # Q
            + self.hidden_size * self.num_kv_heads * self.head_dim       # K
            + self.hidden_size * self.num_kv_heads * self.head_dim       # V
            + self.num_attention_heads * self.head_dim * self.hidden_size  # O
        )
        if self.ffn_type == "swiglu":
            ffn_per_layer = (
                self.hidden_size * self.intermediate_size  # gate
                + self.hidden_size * self.intermediate_size  # up
                + self.intermediate_size * self.hidden_size  # down
            )
        else:
            ffn_per_layer = (
                self.hidden_size * self.intermediate_size
                + self.intermediate_size * self.hidden_size
            )
        norm_per_layer = 2 * self.hidden_size  # attn norm + ffn norm
        final_norm = self.hidden_size
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        total = (
            embed
            + self.num_layers * (attn_per_layer + ffn_per_layer + norm_per_layer)
            + final_norm
            + lm_head
        )
        return total


# ── Named Configs ─────────────────────────────────────────────────────────────

NAMED_CONFIGS: dict[str, DenseConfig] = {
    "1.3b": DenseConfig(
        vocab_size=32000,
        hidden_size=2048,
        num_layers=24,
        num_attention_heads=16,
        max_position_embeddings=4096,
    ),
    "tiny": DenseConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512,
    ),
    "small": DenseConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8,
        max_position_embeddings=2048,
    ),
    "125m": DenseConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    ),
}


# ── Building Blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequency cosine and sine tensors.

    Returns:
        cos, sin tensors of shape (max_seq_len, head_dim)
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of x by half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Embeddings to query and key tensors."""
    # q, k: (batch, heads, seq, head_dim)
    # cos, sin: (seq, head_dim)
    cos = cos[: q.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: q.shape[2]].unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeluFFN(nn.Module):
    """Standard GELU Feed-Forward Network."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional Grouped Query Attention (GQA).

    Supports:
    - Standard MHA (num_kv_heads == num_attention_heads)
    - Grouped Query Attention (num_kv_heads < num_attention_heads)
    - Flash Attention 2 via F.scaled_dot_product_attention (PyTorch 2.0+)
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = config.attention_dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention if available)
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=(attention_mask is None),
        )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(
                config.hidden_size, config.intermediate_size, config.hidden_dropout
            )
        else:
            self.ffn = GeluFFN(
                config.hidden_size, config.intermediate_size, config.hidden_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Main Model ────────────────────────────────────────────────────────────────

class DenseTransformer(nn.Module):
    """
    Decoder-only dense transformer for causal language modeling.

    Architecture:
    - Token embedding
    - N × TransformerBlock (RMSNorm + CausalSelfAttention + RMSNorm + SwiGLU)
    - Final RMSNorm
    - LM head (optionally tied to embedding)

    Args:
        config: DenseConfig instance.
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_cos",
            torch.zeros(config.max_position_embeddings, config.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.zeros(config.max_position_embeddings, config.head_dim),
            persistent=False,
        )

        self._init_weights()
        self._precompute_rope()

    def _init_weights(self) -> None:
        """Initialize weights with scaled normal distribution."""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _precompute_rope(self) -> None:
        """Precompute RoPE frequencies and store in buffers."""
        cos, sin = precompute_rope_freqs(
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            theta=self.config.rope_theta,
        )
        self.rope_cos.copy_(cos)
        self.rope_sin.copy_(sin)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: Optional (batch, seq_len) attention mask.
            labels: Optional (batch, seq_len) target token IDs for loss computation.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        B, T = input_ids.shape

        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, attention_mask, use_reentrant=False
                )
            else:
                x = layer(x, cos, sin, attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Standard causal LM loss: shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n = self.count_parameters()
        return (
            f"DenseTransformer("
            f"layers={self.config.num_layers}, "
            f"hidden={self.config.hidden_size}, "
            f"heads={self.config.num_attention_heads}, "
            f"params={n/1e9:.2f}B)"
        )


def build_model(config_name: str = "1.3b", **kwargs) -> DenseTransformer:
    """
    Build a DenseTransformer from a named configuration.

    Args:
        config_name: One of 'tiny', 'small', '125m', '1.3b'.
        **kwargs: Override specific config fields.

    Returns:
        Initialized DenseTransformer.
    """
    if config_name not in NAMED_CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Available: {list(NAMED_CONFIGS.keys())}"
        )
    config = NAMED_CONFIGS[config_name]
    if kwargs:
        import dataclasses
        config = dataclasses.replace(config, **kwargs)
    return DenseTransformer(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dense transformer model.")
    parser.add_argument("--config", default="tiny", choices=list(NAMED_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    model = build_model(args.config)
    print(model)
    print(f"Parameters: {model.count_parameters()/1e6:.1f}M")

    # Test forward pass
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len))
    labels = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len))

    out = model(input_ids, labels=labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss'].item():.4f}")
    print("Forward pass: OK")

    # Test backward pass
    out["loss"].backward()
    print("Backward pass: OK")
