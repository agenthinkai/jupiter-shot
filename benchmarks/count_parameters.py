#!/usr/bin/env python3
"""
Jupiter Shot — Parameter Count and Memory Estimation
=====================================================
Computes exact parameter counts and memory estimates analytically
from model configs, without instantiating full-size models.

Usage:
    python benchmarks/count_parameters.py

Outputs:
    - Total parameters
    - Trainable parameters
    - Active parameters per token (MoE)
    - BF16 weight memory
    - Estimated training-state memory (Adam optimizer + gradients + activations)
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_dense_params(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    tie_embeddings: bool = True,
) -> dict:
    """
    Analytically compute parameter counts for a dense GPT-NeoX-style transformer.

    Architecture:
      - Token embedding: vocab_size × hidden_size
      - Per layer:
          - QKV projection: 3 × hidden_size × hidden_size (no bias)
          - O projection: hidden_size × hidden_size (no bias)
          - RMSNorm (pre-attn): hidden_size
          - FFN gate_proj: hidden_size × intermediate_size (no bias)
          - FFN up_proj: hidden_size × intermediate_size (no bias)
          - FFN down_proj: intermediate_size × hidden_size (no bias)
          - RMSNorm (pre-ffn): hidden_size
      - Final RMSNorm: hidden_size
      - LM head: vocab_size × hidden_size (tied to embedding if tie_embeddings=True)
    """
    # Embedding
    embed_params = vocab_size * hidden_size

    # Per-layer attention
    qkv_params = 3 * hidden_size * hidden_size  # no bias
    o_params = hidden_size * hidden_size         # no bias
    attn_norm_params = hidden_size               # RMSNorm weight only (no bias)

    # Per-layer FFN (SwiGLU: gate + up + down)
    ffn_gate_params = hidden_size * intermediate_size
    ffn_up_params = hidden_size * intermediate_size
    ffn_down_params = intermediate_size * hidden_size
    ffn_norm_params = hidden_size  # RMSNorm weight only

    per_layer_params = (
        qkv_params + o_params + attn_norm_params +
        ffn_gate_params + ffn_up_params + ffn_down_params + ffn_norm_params
    )

    # Final norm
    final_norm_params = hidden_size

    # LM head (tied or separate)
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_size

    total = embed_params + (per_layer_params * num_layers) + final_norm_params + lm_head_params

    return {
        "embedding": embed_params,
        "per_layer": per_layer_params,
        "num_layers": num_layers,
        "all_layers": per_layer_params * num_layers,
        "final_norm": final_norm_params,
        "lm_head": lm_head_params,
        "total": total,
        "trainable": total,  # All params trainable in dense
    }


def count_moe_params(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_token: int,
    use_shared_expert: bool = False,
    tie_embeddings: bool = True,
) -> dict:
    """
    Analytically compute parameter counts for a sparse MoE transformer.

    Architecture:
      - Same as dense but FFN is replaced by MoE FFN:
          - Router: hidden_size × num_experts (no bias)
          - num_experts × (gate_proj + up_proj + down_proj)
          - Optional shared expert: 1 × (gate_proj + up_proj + down_proj)
    """
    # Embedding
    embed_params = vocab_size * hidden_size

    # Per-layer attention (same as dense)
    qkv_params = 3 * hidden_size * hidden_size
    o_params = hidden_size * hidden_size
    attn_norm_params = hidden_size

    # Per-layer MoE FFN
    router_params = hidden_size * num_experts  # no bias
    expert_ffn_params = (
        hidden_size * intermediate_size +   # gate_proj
        hidden_size * intermediate_size +   # up_proj
        intermediate_size * hidden_size     # down_proj
    )
    all_expert_params = num_experts * expert_ffn_params
    shared_expert_params = expert_ffn_params if use_shared_expert else 0
    ffn_norm_params = hidden_size

    per_layer_params = (
        qkv_params + o_params + attn_norm_params +
        router_params + all_expert_params + shared_expert_params + ffn_norm_params
    )

    # Final norm
    final_norm_params = hidden_size

    # LM head
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_size

    total = embed_params + (per_layer_params * num_layers) + final_norm_params + lm_head_params

    # Active parameters per token (embedding + attention + top-k experts)
    per_layer_active = (
        qkv_params + o_params + attn_norm_params +
        router_params +                                          # router always active
        num_experts_per_token * expert_ffn_params +              # top-k experts
        shared_expert_params +                                   # shared expert always active
        ffn_norm_params
    )
    active_total = embed_params + (per_layer_active * num_layers) + final_norm_params + lm_head_params

    return {
        "embedding": embed_params,
        "per_layer_total": per_layer_params,
        "per_layer_active": per_layer_active,
        "num_layers": num_layers,
        "all_layers_total": per_layer_params * num_layers,
        "all_layers_active": per_layer_active * num_layers,
        "final_norm": final_norm_params,
        "lm_head": lm_head_params,
        "total": total,
        "trainable": total,
        "active_per_token": active_total,
        "sparsity_ratio": 1.0 - (active_total / total),
    }


def memory_estimates(total_params: int, active_params: int, batch_tokens: int = 2_097_152) -> dict:
    """
    Estimate memory requirements for training and inference.

    Training state (Adam optimizer):
      - FP32 master weights: 4 bytes/param
      - FP32 Adam m (first moment): 4 bytes/param
      - FP32 Adam v (second moment): 4 bytes/param
      - BF16 model weights: 2 bytes/param
      - BF16 gradients: 2 bytes/param
      Total per param: 16 bytes (ZeRO-0), sharded with ZeRO-2/3

    Activation memory (rough estimate for transformer):
      - ~12 × hidden_size × seq_len × batch_size bytes per layer
    """
    bytes_per_param_bf16 = 2
    bytes_per_param_fp32 = 4

    bf16_weights_gb = (total_params * bytes_per_param_bf16) / (1024**3)
    fp32_master_gb = (total_params * bytes_per_param_fp32) / (1024**3)
    fp32_adam_m_gb = (total_params * bytes_per_param_fp32) / (1024**3)
    fp32_adam_v_gb = (total_params * bytes_per_param_fp32) / (1024**3)
    bf16_gradients_gb = (total_params * bytes_per_param_bf16) / (1024**3)

    full_training_state_gb = (
        bf16_weights_gb + fp32_master_gb +
        fp32_adam_m_gb + fp32_adam_v_gb + bf16_gradients_gb
    )

    return {
        "bf16_weights_gb": round(bf16_weights_gb, 2),
        "fp32_master_weights_gb": round(fp32_master_gb, 2),
        "fp32_adam_m_gb": round(fp32_adam_m_gb, 2),
        "fp32_adam_v_gb": round(fp32_adam_v_gb, 2),
        "bf16_gradients_gb": round(bf16_gradients_gb, 2),
        "full_training_state_gb": round(full_training_state_gb, 2),
        "note_zero2_per_gpu_8gpu_gb": round(full_training_state_gb / 8 + bf16_weights_gb, 2),
        "note_zero3_per_gpu_8gpu_gb": round(full_training_state_gb / 8, 2),
        "inference_bf16_gb": round(bf16_weights_gb, 2),
        "inference_int8_gb": round(bf16_weights_gb / 2, 2),
        "inference_int4_gb": round(bf16_weights_gb / 4, 2),
    }


def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.4f}B ({n:,})"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M ({n:,})"
    else:
        return f"{n:,}"


def main():
    print("=" * 70)
    print("JUPITER SHOT — PARAMETER COUNT AND MEMORY ESTIMATION")
    print("=" * 70)

    # ── Dense 1.3B Config (from training/configs/dense_1b3.yaml) ─────────────
    DENSE_CONFIG = {
        "vocab_size": 50_257,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_attention_heads": 16,
        "max_position_embeddings": 2048,
        "tie_embeddings": True,
    }
    # intermediate_size = round_to_multiple(hidden_size * 4 * 2/3, 64) for SwiGLU
    # SwiGLU intermediate = 4 * hidden * 2/3 rounded to multiple of 64
    swiglu_intermediate = int(math.ceil(DENSE_CONFIG["hidden_size"] * 4 * 2 / 3 / 64) * 64)
    DENSE_CONFIG["intermediate_size"] = swiglu_intermediate

    print("\n" + "─" * 70)
    print("1. DENSE BASELINE — 1.3B")
    print("─" * 70)
    print(f"   Config: {DENSE_CONFIG}")

    dense = count_dense_params(**DENSE_CONFIG)
    print(f"\n   Embedding:          {fmt(dense['embedding'])}")
    print(f"   Per-layer params:   {fmt(dense['per_layer'])}")
    print(f"   All layers (×{DENSE_CONFIG['num_layers']}):   {fmt(dense['all_layers'])}")
    print(f"   Final norm:         {fmt(dense['final_norm'])}")
    print(f"   LM head:            {fmt(dense['lm_head'])} (tied to embedding)")
    print(f"\n   ┌─────────────────────────────────────────────────────┐")
    print(f"   │  TOTAL PARAMETERS:     {fmt(dense['total']):>30s}  │")
    print(f"   │  TRAINABLE PARAMETERS: {fmt(dense['trainable']):>30s}  │")
    print(f"   └─────────────────────────────────────────────────────┘")

    mem_dense = memory_estimates(dense["total"], dense["total"])
    print(f"\n   Memory Estimates:")
    print(f"   BF16 weights:              {mem_dense['bf16_weights_gb']:>6.2f} GB")
    print(f"   FP32 master weights:       {mem_dense['fp32_master_weights_gb']:>6.2f} GB")
    print(f"   FP32 Adam m (1st moment):  {mem_dense['fp32_adam_m_gb']:>6.2f} GB")
    print(f"   FP32 Adam v (2nd moment):  {mem_dense['fp32_adam_v_gb']:>6.2f} GB")
    print(f"   BF16 gradients:            {mem_dense['bf16_gradients_gb']:>6.2f} GB")
    print(f"   ─────────────────────────────────────────────────────")
    print(f"   Full training state:       {mem_dense['full_training_state_gb']:>6.2f} GB (ZeRO-0, 1 GPU)")
    print(f"   ZeRO-2 per GPU (8 GPUs):   {mem_dense['note_zero2_per_gpu_8gpu_gb']:>6.2f} GB (optimizer sharded)")
    print(f"   ZeRO-3 per GPU (8 GPUs):   {mem_dense['note_zero3_per_gpu_8gpu_gb']:>6.2f} GB (all sharded)")
    print(f"   Inference BF16:            {mem_dense['inference_bf16_gb']:>6.2f} GB")
    print(f"   Inference INT8:            {mem_dense['inference_int8_gb']:>6.2f} GB")
    print(f"   Inference INT4:            {mem_dense['inference_int4_gb']:>6.2f} GB")

    # ── MoE Prototype Config (from training/configs/moe_prototype.yaml) ──────
    MOE_CONFIG = {
        "vocab_size": 50_257,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": swiglu_intermediate,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "use_shared_expert": False,
        "tie_embeddings": True,
    }

    print("\n" + "─" * 70)
    print("2. SPARSE MOE PROTOTYPE — 8 EXPERTS, TOP-2")
    print("─" * 70)
    print(f"   Config: {MOE_CONFIG}")

    moe = count_moe_params(**MOE_CONFIG)
    print(f"\n   Embedding:               {fmt(moe['embedding'])}")
    print(f"   Per-layer total params:  {fmt(moe['per_layer_total'])}")
    print(f"   Per-layer active params: {fmt(moe['per_layer_active'])} (top-{MOE_CONFIG['num_experts_per_token']} of {MOE_CONFIG['num_experts']} experts)")
    print(f"   All layers total:        {fmt(moe['all_layers_total'])}")
    print(f"   All layers active:       {fmt(moe['all_layers_active'])}")
    print(f"   Final norm:              {fmt(moe['final_norm'])}")
    print(f"   LM head:                 {fmt(moe['lm_head'])} (tied to embedding)")
    print(f"\n   ┌─────────────────────────────────────────────────────┐")
    print(f"   │  TOTAL PARAMETERS:     {fmt(moe['total']):>30s}  │")
    print(f"   │  TRAINABLE PARAMETERS: {fmt(moe['trainable']):>30s}  │")
    print(f"   │  ACTIVE PER TOKEN:     {fmt(moe['active_per_token']):>30s}  │")
    print(f"   │  SPARSITY RATIO:       {moe['sparsity_ratio']:>30.1%}  │")
    print(f"   └─────────────────────────────────────────────────────┘")

    mem_moe = memory_estimates(moe["total"], moe["active_per_token"])
    print(f"\n   Memory Estimates:")
    print(f"   BF16 weights (all experts):  {mem_moe['bf16_weights_gb']:>6.2f} GB")
    print(f"   Full training state:         {mem_moe['full_training_state_gb']:>6.2f} GB (ZeRO-0, 1 GPU)")
    print(f"   ZeRO-2 per GPU (8 GPUs):     {mem_moe['note_zero2_per_gpu_8gpu_gb']:>6.2f} GB")
    print(f"   ZeRO-3 per GPU (8 GPUs):     {mem_moe['note_zero3_per_gpu_8gpu_gb']:>6.2f} GB")
    print(f"   Inference BF16:              {mem_moe['inference_bf16_gb']:>6.2f} GB")
    print(f"   Inference INT8:              {mem_moe['inference_int8_gb']:>6.2f} GB")
    print(f"   Inference INT4:              {mem_moe['inference_int4_gb']:>6.2f} GB")

    # ── Comparison Table ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. COMPARISON SUMMARY")
    print("─" * 70)
    print(f"\n   {'Metric':<35} {'Dense 1.3B':>15} {'MoE 8×1.3B':>15}")
    print(f"   {'─'*35} {'─'*15} {'─'*15}")
    print(f"   {'Total parameters':<35} {dense['total']/1e9:>14.4f}B {moe['total']/1e9:>14.4f}B")
    print(f"   {'Active per token':<35} {dense['total']/1e9:>14.4f}B {moe['active_per_token']/1e9:>14.4f}B")
    print(f"   {'BF16 weight memory':<35} {mem_dense['bf16_weights_gb']:>13.2f}GB {mem_moe['bf16_weights_gb']:>13.2f}GB")
    print(f"   {'Full training state (ZeRO-0)':<35} {mem_dense['full_training_state_gb']:>13.2f}GB {mem_moe['full_training_state_gb']:>13.2f}GB")
    print(f"   {'ZeRO-2 per GPU (8 GPUs)':<35} {mem_dense['note_zero2_per_gpu_8gpu_gb']:>13.2f}GB {mem_moe['note_zero2_per_gpu_8gpu_gb']:>13.2f}GB")
    print(f"   {'Inference BF16':<35} {mem_dense['inference_bf16_gb']:>13.2f}GB {mem_moe['inference_bf16_gb']:>13.2f}GB")

    print("\n" + "─" * 70)
    print("4. NOTES")
    print("─" * 70)
    print("""
   - Intermediate size uses SwiGLU formula: round_up(hidden * 4 * 2/3, 64)
     For hidden=2048: intermediate = round_up(5461.3, 64) = 5504
   - Embedding weights are tied to LM head (counted once in total)
   - RoPE positional encoding has NO learned parameters
   - Training state assumes Adam optimizer (m + v in FP32, weights in BF16)
   - ZeRO-2 shards optimizer state across GPUs; model weights replicated
   - ZeRO-3 shards optimizer state + weights + gradients across GPUs
   - Activation memory (~2-4GB per GPU for seq_len=2048, batch=4) not included
   - MoE expert parallelism: each GPU holds 1 expert (8 GPUs = 8 experts)
   - Communication overhead for MoE: 2× all-to-all per MoE layer
""")

    print("=" * 70)
    print("VERIFICATION: Instantiate tiny model to confirm analytical formula")
    print("=" * 70)
    try:
        import torch
        from training.models.dense import DenseConfig, DenseTransformer
        tiny_cfg = DenseConfig(
            vocab_size=1000, hidden_size=64, num_layers=2,
            num_attention_heads=4, max_position_embeddings=128,
            gradient_checkpointing=False,
        )
        tiny_intermediate = int(math.ceil(64 * 4 * 2 / 3 / 64) * 64)
        tiny_model = DenseTransformer(tiny_cfg)
        actual_params = sum(p.numel() for p in tiny_model.parameters())
        analytical = count_dense_params(
            vocab_size=1000, hidden_size=64, num_layers=2,
            num_attention_heads=4, max_position_embeddings=128,
            intermediate_size=tiny_intermediate, tie_embeddings=True,
        )
        print(f"\n   Tiny model actual params:     {actual_params:,}")
        print(f"   Analytical formula result:    {analytical['total']:,}")
        match = actual_params == analytical['total']
        print(f"   Match: {'✅ EXACT' if match else '❌ MISMATCH (delta=' + str(actual_params - analytical['total']) + ')'}")
        del tiny_model
    except Exception as e:
        print(f"\n   Verification skipped: {e}")

    return {
        "dense": dense,
        "moe": moe,
        "dense_memory": mem_dense,
        "moe_memory": mem_moe,
    }


if __name__ == "__main__":
    results = main()
