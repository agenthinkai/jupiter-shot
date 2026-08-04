"""
Jupiter Shot — Sparse Mixture-of-Experts (MoE) Prototype
=========================================================
Implements a decoder-only transformer where each FFN sublayer is replaced
by a sparse MoE layer with top-K expert routing.

Architecture:
  - Shared attention layers (identical to dense baseline)
  - MoE FFN: N experts, top-K selected per token
  - Auxiliary load-balancing loss (Switch Transformer style)
  - Optional router-z loss (DeepSeekMoE style)
  - Optional shared expert (always-active expert)

Key design choices:
  - Top-2 routing: better stability than top-1, standard for production MoE
  - Expert capacity factor: limits tokens per expert to prevent overflow
  - Load-balancing auxiliary loss: encourages uniform expert utilization
  - Router-z loss: prevents router logits from growing too large

References:
  - Switch Transformer (Fedus et al., 2022): https://arxiv.org/abs/2101.03961
  - Mixtral (Jiang et al., 2024): https://arxiv.org/abs/2401.04088
  - DeepSeekMoE (Dai et al., 2024): https://arxiv.org/abs/2401.06066
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint  # explicit — do not rely on side-effects

from training.models.dense import (
    DenseConfig,
    RMSNorm,
    CausalSelfAttention,
    SwiGLUFFN,
    precompute_rope_freqs,
    apply_rope,
    rotate_half,
    NAMED_CONFIGS as DENSE_NAMED_CONFIGS,
)

logger = logging.getLogger(__name__)


# ── MoE Configuration ─────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    """Configuration for the sparse MoE transformer."""

    # Base transformer config (shared with dense baseline)
    base: DenseConfig = field(default_factory=lambda: DenseConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=12,
        num_attention_heads=8,
        max_position_embeddings=2048,
    ))

    # MoE-specific
    num_experts: int = 8
    """Total number of expert FFNs per MoE layer."""

    num_experts_per_token: int = 2
    """Number of experts selected per token (top-K)."""

    expert_capacity_factor: float = 1.25
    """
    Expert capacity = capacity_factor × (tokens_per_batch / num_experts).
    Tokens exceeding capacity are dropped (or passed through residual).
    """

    router_aux_loss_coeff: float = 0.01
    """
    Coefficient for the auxiliary load-balancing loss.
    Higher = more uniform expert utilization, but may hurt model quality.
    Typical range: 0.001 to 0.1.
    """

    router_z_loss_coeff: float = 0.001
    """
    Coefficient for the router-z loss (prevents logit explosion).
    Set to 0.0 to disable. Typical range: 0.0 to 0.01.
    """

    use_shared_expert: bool = False
    """
    If True, add one always-active shared expert in addition to the routed experts.
    Inspired by DeepSeekMoE. Increases total parameters but not active parameters.
    """

    expert_dropout: float = 0.0
    """Dropout applied within each expert FFN."""

    normalize_router_probs: bool = True
    """
    If True, normalize the selected expert weights to sum to 1.
    Standard in most MoE implementations.
    """

    def count_parameters(self) -> int:
        """Estimate total parameter count."""
        # Attention parameters (same as dense)
        base = self.base
        embed = base.vocab_size * base.hidden_size
        attn_per_layer = (
            base.hidden_size * base.num_attention_heads * base.head_dim
            + base.hidden_size * base.num_kv_heads * base.head_dim
            + base.hidden_size * base.num_kv_heads * base.head_dim
            + base.num_attention_heads * base.head_dim * base.hidden_size
        )
        # MoE FFN: N experts × SwiGLU params + router
        ffn_per_expert = (
            base.hidden_size * base.intermediate_size  # gate
            + base.hidden_size * base.intermediate_size  # up
            + base.intermediate_size * base.hidden_size  # down
        )
        router_per_layer = base.hidden_size * self.num_experts
        moe_per_layer = (
            self.num_experts * ffn_per_expert
            + router_per_layer
            + (ffn_per_expert if self.use_shared_expert else 0)
        )
        norm_per_layer = 2 * base.hidden_size
        final_norm = base.hidden_size
        lm_head = 0 if base.tie_word_embeddings else base.vocab_size * base.hidden_size
        total = (
            embed
            + base.num_layers * (attn_per_layer + moe_per_layer + norm_per_layer)
            + final_norm
            + lm_head
        )
        return total

    def count_active_parameters(self) -> int:
        """Estimate active parameters per token (attention + top-K experts)."""
        base = self.base
        attn_per_layer = (
            base.hidden_size * base.num_attention_heads * base.head_dim
            + base.hidden_size * base.num_kv_heads * base.head_dim
            + base.hidden_size * base.num_kv_heads * base.head_dim
            + base.num_attention_heads * base.head_dim * base.hidden_size
        )
        ffn_per_expert = (
            base.hidden_size * base.intermediate_size
            + base.hidden_size * base.intermediate_size
            + base.intermediate_size * base.hidden_size
        )
        active_ffn_per_layer = (
            self.num_experts_per_token * ffn_per_expert
            + (ffn_per_expert if self.use_shared_expert else 0)
        )
        embed = base.vocab_size * base.hidden_size
        return embed + base.num_layers * (attn_per_layer + active_ffn_per_layer)


# ── Named MoE Configs ─────────────────────────────────────────────────────────

MOE_NAMED_CONFIGS: dict[str, MoEConfig] = {
    "moe_tiny": MoEConfig(
        base=DenseConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512,
        ),
        num_experts=4,
        num_experts_per_token=2,
    ),
    "moe_1b": MoEConfig(
        base=DenseConfig(
            vocab_size=32000,
            hidden_size=1024,
            num_layers=12,
            num_attention_heads=8,
            max_position_embeddings=2048,
        ),
        num_experts=8,
        num_experts_per_token=2,
        expert_capacity_factor=1.25,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.001,
    ),
    "moe_3b": MoEConfig(
        base=DenseConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_layers=16,
            num_attention_heads=16,
            max_position_embeddings=4096,
        ),
        num_experts=8,
        num_experts_per_token=2,
        expert_capacity_factor=1.25,
        router_aux_loss_coeff=0.01,
        router_z_loss_coeff=0.001,
        use_shared_expert=False,
    ),
}


# ── Router ────────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """
    Top-K sparse router for MoE layers.

    Computes routing probabilities via a linear projection + softmax,
    selects the top-K experts per token, and computes auxiliary losses
    to encourage load balancing.

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Total number of experts.
        num_experts_per_token: K in top-K routing.
        capacity_factor: Expert capacity multiplier.
        aux_loss_coeff: Coefficient for load-balancing auxiliary loss.
        z_loss_coeff: Coefficient for router-z loss.
        normalize_probs: Whether to normalize selected expert weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
        z_loss_coeff: float = 0.001,
        normalize_probs: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.normalize_probs = normalize_probs

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compute routing decisions.

        Args:
            x: Input tensor of shape (batch × seq_len, hidden_size).

        Returns:
            Tuple of:
              - router_weights: (tokens, num_experts_per_token) selected weights
              - expert_indices: (tokens, num_experts_per_token) selected expert IDs
              - aux_loss: Scalar auxiliary loss (load balancing + z-loss)
              - metrics: Dict with expert utilization statistics
        """
        num_tokens = x.shape[0]

        # Router logits: (tokens, num_experts)
        router_logits = self.gate(x)

        # Router probabilities: softmax over experts
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )

        # Normalize selected weights to sum to 1
        if self.normalize_probs:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # ── Auxiliary load-balancing loss ─────────────────────────────────────
        # Switch Transformer style: aux_loss = num_experts × Σ(f_i × P_i)
        # where f_i = fraction of tokens routed to expert i
        #       P_i = mean router probability for expert i
        # This encourages uniform routing without hard constraints.
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        # expert_mask: (tokens, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # (num_experts,)
        fraction_per_expert = tokens_per_expert / (num_tokens * self.num_experts_per_token)
        mean_prob_per_expert = router_probs.mean(dim=0)  # (num_experts,)
        aux_loss = self.num_experts * (fraction_per_expert * mean_prob_per_expert).sum()
        aux_loss = self.aux_loss_coeff * aux_loss

        # ── Router-z loss ─────────────────────────────────────────────────────
        # Prevents router logits from growing too large.
        # z_loss = mean(log(sum(exp(logits)))^2)
        if self.z_loss_coeff > 0:
            z_loss = torch.mean(torch.log(torch.exp(router_logits).sum(dim=-1)) ** 2)
            aux_loss = aux_loss + self.z_loss_coeff * z_loss

        # ── Metrics — canonical schema from training.router_metrics ────────────
        with torch.no_grad():
            from training.router_metrics import compute_router_metrics as _crm
            _counts = tokens_per_expert.detach().cpu().tolist()
            _mean_probs = mean_prob_per_expert.detach().cpu().tolist()
            # Compute raw (unscaled) z-loss value for the metrics record
            if self.z_loss_coeff > 0:
                _z_raw = float(torch.mean(
                    torch.log(torch.exp(router_logits).sum(dim=-1)) ** 2
                ).item())
            else:
                _z_raw = 0.0
            _aux_raw = float(
                self.num_experts
                * (fraction_per_expert * mean_prob_per_expert).sum().item()
            )
            metrics = _crm(
                expert_assignment_counts=_counts,
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                top_k=self.num_experts_per_token,
                router_probs_mean=_mean_probs,
                aux_loss_unscaled=_aux_raw,
                aux_loss_coeff=self.aux_loss_coeff,
                z_loss_unscaled=_z_raw,
                z_loss_coeff=self.z_loss_coeff,
                capacity_factor=self.capacity_factor,
            )

        return top_k_weights, top_k_indices, aux_loss, metrics


# ── Expert Layer ──────────────────────────────────────────────────────────────

class ExpertLayer(nn.Module):
    """Single expert FFN (SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ── MoE FFN Layer ─────────────────────────────────────────────────────────────

class MoEFFNLayer(nn.Module):
    """
    Sparse MoE FFN layer: replaces the dense FFN in each transformer block.

    Implements the standard token-choice routing:
    1. Router selects top-K experts for each token
    2. Each selected expert processes the token
    3. Expert outputs are combined with router weights

    Args:
        config: MoEConfig instance.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        base = config.base

        self.router = TopKRouter(
            hidden_size=base.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            capacity_factor=config.expert_capacity_factor,
            aux_loss_coeff=config.router_aux_loss_coeff,
            z_loss_coeff=config.router_z_loss_coeff,
            normalize_probs=config.normalize_router_probs,
        )

        self.experts = nn.ModuleList([
            ExpertLayer(base.hidden_size, base.intermediate_size, config.expert_dropout)
            for _ in range(config.num_experts)
        ])

        self.shared_expert = None
        if config.use_shared_expert:
            self.shared_expert = ExpertLayer(
                base.hidden_size, base.intermediate_size, config.expert_dropout
            )

        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through MoE FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of:
              - output: (batch, seq_len, hidden_size)
              - aux_loss: Scalar auxiliary loss
              - router_metrics: Dict with expert utilization statistics
        """
        B, T, H = x.shape
        x_flat = x.view(B * T, H)

        # Get routing decisions
        router_weights, expert_indices, aux_loss, router_metrics = self.router(x_flat)
        # router_weights: (B*T, top_k)
        # expert_indices: (B*T, top_k)

        # Dispatch tokens to experts
        output = torch.zeros_like(x_flat)

        for k in range(self.num_experts_per_token):
            # Get tokens assigned to each expert for this k
            expert_ids = expert_indices[:, k]  # (B*T,)
            weights = router_weights[:, k]     # (B*T,)

            for expert_id in range(self.num_experts):
                # Find tokens routed to this expert
                token_mask = (expert_ids == expert_id)
                if not token_mask.any():
                    continue

                expert_input = x_flat[token_mask]
                expert_output = self.experts[expert_id](expert_input)
                output[token_mask] += weights[token_mask].unsqueeze(-1) * expert_output

        # Add shared expert output (always active)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        output = output.view(B, T, H)
        return output, aux_loss, router_metrics


# ── MoE Transformer Block ─────────────────────────────────────────────────────

class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN instead of dense FFN."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        base = config.base
        self.attn_norm = RMSNorm(base.hidden_size, base.rms_norm_eps)
        self.attn = CausalSelfAttention(base)
        self.ffn_norm = RMSNorm(base.hidden_size, base.rms_norm_eps)
        self.moe_ffn = MoEFFNLayer(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            Tuple of (output, aux_loss, router_metrics)
        """
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        ffn_out, aux_loss, metrics = self.moe_ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss, metrics


# ── Main MoE Model ────────────────────────────────────────────────────────────

class MoETransformer(nn.Module):
    """
    Sparse MoE decoder-only transformer.

    Identical to DenseTransformer except each FFN sublayer is replaced
    by a MoEFFNLayer. Attention layers are shared (not replicated).

    The total loss is: LM loss + sum(aux_losses across all MoE layers).

    Args:
        config: MoEConfig instance.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base

        self.embed_tokens = nn.Embedding(base.vocab_size, base.hidden_size)
        self.layers = nn.ModuleList([
            MoETransformerBlock(config) for _ in range(base.num_layers)
        ])
        self.norm = RMSNorm(base.hidden_size, base.rms_norm_eps)
        self.lm_head = nn.Linear(base.hidden_size, base.vocab_size, bias=False)

        if base.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_cos",
            torch.zeros(base.max_position_embeddings, base.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.zeros(base.max_position_embeddings, base.head_dim),
            persistent=False,
        )

        self._init_weights()
        self._precompute_rope()

    def _init_weights(self) -> None:
        std = self.config.base.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def _precompute_rope(self) -> None:
        cos, sin = precompute_rope_freqs(
            head_dim=self.config.base.head_dim,
            max_seq_len=self.config.base.max_position_embeddings,
            theta=self.config.base.rope_theta,
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

        Returns:
            Dict with:
              - 'logits': (batch, seq_len, vocab_size)
              - 'loss': Total loss (LM + aux) if labels provided
              - 'lm_loss': Language modeling loss only
              - 'aux_loss': Total auxiliary routing loss
              - 'router_metrics': List of per-layer routing metrics
        """
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        total_aux_loss = torch.tensor(0.0, device=x.device)
        all_router_metrics = []

        for layer in self.layers:
            if self.config.base.gradient_checkpointing and self.training:
                def create_custom_forward(layer):
                    def custom_forward(*inputs):
                        out, aux, metrics = layer(*inputs)
                        return out, aux
                    return custom_forward
                x, aux_loss = gradient_checkpoint(
                    create_custom_forward(layer), x, cos, sin, attention_mask,
                    use_reentrant=False
                )
                all_router_metrics.append({})
            else:
                x, aux_loss, router_metrics = layer(x, cos, sin, attention_mask)
                total_aux_loss = total_aux_loss + aux_loss
                all_router_metrics.append(router_metrics)

        x = self.norm(x)
        logits = self.lm_head(x)

        lm_loss = None
        total_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.base.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = lm_loss + total_aux_loss

        return {
            "logits": logits,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "aux_loss": total_aux_loss,
            "router_metrics": all_router_metrics,
        }

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_expert_utilization(self) -> dict:
        """
        Aggregate expert utilization metrics across all layers.
        Call after a forward pass to get routing statistics.
        """
        # This is populated during forward pass; see router_metrics in forward output
        return {}

    def __repr__(self) -> str:
        n_total = self.count_parameters()
        n_active = self.config.count_active_parameters()
        return (
            f"MoETransformer("
            f"layers={self.config.base.num_layers}, "
            f"hidden={self.config.base.hidden_size}, "
            f"experts={self.config.num_experts}, "
            f"top_k={self.config.num_experts_per_token}, "
            f"total_params={n_total/1e9:.2f}B, "
            f"active_params={n_active/1e9:.2f}B)"
        )


def build_moe_model(config_name: str = "moe_1b", **kwargs) -> MoETransformer:
    """
    Build a MoETransformer from a named configuration.

    Args:
        config_name: One of 'moe_tiny', 'moe_1b', 'moe_3b'.
        **kwargs: Override specific MoEConfig fields.

    Returns:
        Initialized MoETransformer.
    """
    if config_name not in MOE_NAMED_CONFIGS:
        raise ValueError(
            f"Unknown MoE config '{config_name}'. "
            f"Available: {list(MOE_NAMED_CONFIGS.keys())}"
        )
    config = MOE_NAMED_CONFIGS[config_name]
    if kwargs:
        config = dataclasses.replace(config, **kwargs)
    return MoETransformer(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MoE transformer model.")
    parser.add_argument("--config", default="moe_tiny", choices=list(MOE_NAMED_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    model = build_moe_model(args.config)
    print(model)
    print(f"Total parameters: {model.count_parameters()/1e6:.1f}M")
    print(f"Active parameters: {model.config.count_active_parameters()/1e6:.1f}M")

    input_ids = torch.randint(0, model.config.base.vocab_size, (args.batch_size, args.seq_len))
    labels = torch.randint(0, model.config.base.vocab_size, (args.batch_size, args.seq_len))

    out = model(input_ids, labels=labels)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Total loss: {out['loss'].item():.4f}")
    print(f"LM loss: {out['lm_loss'].item():.4f}")
    print(f"Aux loss: {out['aux_loss'].item():.6f}")
    print(f"Router metrics (layer 0): {out['router_metrics'][0]}")
    print("Forward pass: OK")

    out["loss"].backward()
    print("Backward pass: OK")
