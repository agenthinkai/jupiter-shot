"""
Jupiter Shot — Tokenizer
=========================
Configurable tokenizer wrapper supporting:
- HuggingFace tokenizers (by name or local path)
- GPT-NeoX-compatible fallback (no authentication required)
- Optional Llama-compatible tokenizer (gated; requires authentication)

Authentication and licensing requirements are documented clearly.
Gated assets are never assumed to be accessible.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)

# ── Tokenizer Specifications ──────────────────────────────────────────────────

TOKENIZER_SPECS = {
    "gpt-neox": {
        "hf_name": "EleutherAI/gpt-neox-20b",
        "vocab_size": 50257,
        "auth_required": False,
        "license": "Apache-2.0",
        "commercial_ok": True,
        "notes": "Default fallback. No authentication required.",
    },
    "mistral": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "vocab_size": 32000,
        "auth_required": True,
        "license": "Apache-2.0",
        "commercial_ok": True,
        "notes": (
            "Requires HuggingFace authentication token. "
            "Run: huggingface-cli login"
        ),
    },
    "llama2": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "vocab_size": 32000,
        "auth_required": True,
        "license": "Meta Community License",
        "commercial_ok": False,
        "notes": (
            "GATED. Requires: (1) Meta approval at meta.ai, "
            "(2) HuggingFace authentication token. "
            "NOT approved for commercial training."
        ),
    },
    "falcon": {
        "hf_name": "tiiuae/falcon-7b",
        "vocab_size": 65024,
        "auth_required": False,
        "license": "Apache-2.0",
        "commercial_ok": True,
        "notes": "Large vocabulary. No authentication required.",
    },
}

DEFAULT_TOKENIZER = "gpt-neox"


class JupiterTokenizer:
    """
    Tokenizer wrapper for Jupiter Shot training.

    Supports HuggingFace tokenizers by name or local path.
    Defaults to GPT-NeoX (no authentication required).

    Args:
        tokenizer_name_or_path: HuggingFace model name, local path, or
                                 one of the short names in TOKENIZER_SPECS.
        max_length: Maximum sequence length for truncation/padding.
        add_bos_token: Whether to prepend BOS token.
        add_eos_token: Whether to append EOS token.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str = DEFAULT_TOKENIZER,
        max_length: int = 2048,
        add_bos_token: bool = False,
        add_eos_token: bool = True,
    ) -> None:
        from transformers import AutoTokenizer

        # Resolve short name to HF identifier
        if tokenizer_name_or_path in TOKENIZER_SPECS:
            spec = TOKENIZER_SPECS[tokenizer_name_or_path]
            hf_name = spec["hf_name"]

            if spec["auth_required"]:
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if not token:
                    raise EnvironmentError(
                        f"Tokenizer '{tokenizer_name_or_path}' requires HuggingFace authentication. "
                        f"Set HF_TOKEN environment variable or run: huggingface-cli login\n"
                        f"Notes: {spec['notes']}"
                    )

            if not spec["commercial_ok"]:
                allow = os.environ.get("JUPITER_ALLOW_NONCOMMERCIAL", "").lower()
                if allow not in ("1", "true", "yes"):
                    raise ValueError(
                        f"Tokenizer '{tokenizer_name_or_path}' is NOT approved for commercial use "
                        f"(license: {spec['license']}). "
                        "Set JUPITER_ALLOW_NONCOMMERCIAL=1 to override for research use."
                    )

            logger.info(
                f"Loading tokenizer '{tokenizer_name_or_path}' from {hf_name} "
                f"(license: {spec['license']})"
            )
            resolved_name = hf_name
        elif Path(tokenizer_name_or_path).exists():
            resolved_name = tokenizer_name_or_path
            logger.info(f"Loading local tokenizer from {resolved_name}")
        else:
            # Assume it's a direct HF identifier
            resolved_name = tokenizer_name_or_path
            logger.info(f"Loading tokenizer from HuggingFace: {resolved_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_name,
            use_fast=True,
            trust_remote_code=False,
        )
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        # Ensure pad token is set (required for batched encoding)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info(
                f"pad_token not set; using eos_token ({self._tokenizer.eos_token!r}) as pad_token"
            )

    @property
    def vocab_size(self) -> int:
        return len(self._tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[list[int], torch.Tensor]:
        """
        Encode a single text string to token IDs.

        Args:
            text: Input text.
            return_tensors: If 'pt', return a PyTorch tensor. Otherwise, return a list.

        Returns:
            Token IDs as list or tensor.
        """
        tokens = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        if return_tensors == "pt":
            return torch.tensor(tokens, dtype=torch.long)
        return tokens

    def decode(self, token_ids: Union[list[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Encode a batch of texts with padding.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        return self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def tokenize_for_training(self, text: str) -> list[int]:
        """
        Tokenize text for training (adds EOS token, no padding).
        Used by the data loader for packing sequences.
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if self.add_bos_token and self.bos_token_id is not None:
            tokens = [self.bos_token_id] + tokens
        if self.add_eos_token:
            tokens = tokens + [self.eos_token_id]
        return tokens

    def __repr__(self) -> str:
        return (
            f"JupiterTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"max_length={self.max_length}, "
            f"eos_token_id={self.eos_token_id})"
        )


def get_tokenizer(
    name_or_path: str = DEFAULT_TOKENIZER,
    max_length: int = 2048,
) -> JupiterTokenizer:
    """
    Convenience factory for creating a JupiterTokenizer.

    Args:
        name_or_path: Tokenizer name (see TOKENIZER_SPECS) or HF identifier or local path.
        max_length: Maximum sequence length.

    Returns:
        Configured JupiterTokenizer instance.
    """
    return JupiterTokenizer(tokenizer_name_or_path=name_or_path, max_length=max_length)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Jupiter tokenizer.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        choices=list(TOKENIZER_SPECS.keys()) + ["custom"],
        help="Tokenizer to use",
    )
    parser.add_argument("--text", default="Hello, world! This is a test.", help="Text to tokenize")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tok = get_tokenizer(args.tokenizer, args.max_length)
    print(f"Tokenizer: {tok}")
    tokens = tok.encode(args.text)
    print(f"Input: {args.text!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"Decoded: {tok.decode(tokens)!r}")
