"""
Jupiter Shot — Dataset Registry
================================
Central registry of approved training datasets with license metadata,
streaming support flags, and authentication requirements.

Every dataset used in training must be registered here.
Do NOT silently substitute one dataset for another.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Specification for a registered training dataset."""

    name: str
    """Human-readable name."""

    hf_identifier: str
    """HuggingFace dataset identifier (e.g., 'togethercomputer/RedPajama-Data-1T')."""

    license: str
    """SPDX license identifier or description."""

    commercial_training_ok: bool
    """Whether commercial model training is permitted under this license."""

    streaming_supported: bool
    """Whether the dataset supports HuggingFace streaming (do not download fully)."""

    auth_required: bool
    """Whether a HuggingFace token or other authentication is required."""

    approx_tokens: str
    """Approximate token count (informational)."""

    approx_size_gb: float
    """Approximate raw size in GB."""

    notes: str
    """License caveats, redistribution restrictions, or usage notes."""

    subset: Optional[str] = None
    """Optional HuggingFace subset/config name."""


# ── Approved Datasets ─────────────────────────────────────────────────────────

REGISTRY: dict[str, DatasetSpec] = {
    "redpajama_v1": DatasetSpec(
        name="RedPajama-Data-1T",
        hf_identifier="togethercomputer/RedPajama-Data-1T",
        license="Apache-2.0",
        commercial_training_ok=True,
        streaming_supported=True,
        auth_required=False,
        approx_tokens="~1T",
        approx_size_gb=2500.0,
        notes=(
            "Recommended for Month 1. Apache 2.0 — commercial training permitted. "
            "Use streaming; do not download the full corpus. "
            "Subsets: common_crawl, c4, github, books, arxiv, wikipedia, stackexchange."
        ),
        subset="wikipedia",  # Default to Wikipedia for smoke tests
    ),
    "redpajama_v1_books": DatasetSpec(
        name="RedPajama-Data-1T (Books)",
        hf_identifier="togethercomputer/RedPajama-Data-1T",
        license="Apache-2.0",
        commercial_training_ok=True,
        streaming_supported=True,
        auth_required=False,
        approx_tokens="~26B",
        approx_size_gb=26.0,
        notes="Books subset of RedPajama-1T. Higher quality than web text.",
        subset="book",
    ),
    "redpajama_v2": DatasetSpec(
        name="RedPajama-Data-v2",
        hf_identifier="togethercomputer/RedPajama-Data-V2",
        license="Apache-2.0",
        commercial_training_ok=True,
        streaming_supported=True,
        auth_required=False,
        approx_tokens="~30T",
        approx_size_gb=100000.0,
        notes=(
            "Very large corpus. Use streaming with subset selection. "
            "Quality signals available for filtering."
        ),
    ),
    "openwebtext2": DatasetSpec(
        name="OpenWebText2 (via The Pile)",
        hf_identifier="EleutherAI/the_pile_openwebtext2",
        license="MIT",
        commercial_training_ok=True,
        streaming_supported=True,
        auth_required=False,
        approx_tokens="~65B",
        approx_size_gb=65.0,
        notes="High-quality web text. MIT license — commercial training permitted.",
    ),
    "wikipedia_en": DatasetSpec(
        name="Wikipedia (English)",
        hf_identifier="wikimedia/wikipedia",
        license="CC-BY-SA-4.0",
        commercial_training_ok=True,
        streaming_supported=True,
        auth_required=False,
        approx_tokens="~4B",
        approx_size_gb=20.0,
        notes=(
            "CC BY-SA 4.0 — attribution required in model card. "
            "Commercial training permitted with attribution."
        ),
        subset="20231101.en",
    ),
    "dolma": DatasetSpec(
        name="Dolma v1.7",
        hf_identifier="allenai/dolma",
        license="AI2-ImpACT",
        commercial_training_ok=False,
        streaming_supported=True,
        auth_required=True,
        approx_tokens="~3T",
        approx_size_gb=5000.0,
        notes=(
            "AI2 ImpACT license — research use only, NOT commercial training. "
            "Requires HuggingFace authentication token."
        ),
    ),
}

# ── Not-Approved Datasets (documented for transparency) ──────────────────────

NOT_APPROVED: dict[str, str] = {
    "common_crawl_raw": "No license; requires extensive filtering and deduplication.",
    "books3": "Copyright status disputed; removed from The Pile v2.",
    "c4": "Derived from Common Crawl; license unclear for commercial training.",
    "pile_books3": "Same as books3 — copyright disputed.",
}


def get_dataset(name: str) -> DatasetSpec:
    """
    Retrieve a registered dataset specification by name.

    Raises:
        KeyError: If the dataset is not registered.
        ValueError: If the dataset is not approved for commercial training
                    and JUPITER_ALLOW_NONCOMMERCIAL is not set.
    """
    import os

    if name in NOT_APPROVED:
        raise ValueError(
            f"Dataset '{name}' is not approved: {NOT_APPROVED[name]}"
        )

    if name not in REGISTRY:
        raise KeyError(
            f"Dataset '{name}' is not registered. "
            f"Available datasets: {list(REGISTRY.keys())}. "
            "Do NOT silently substitute a different dataset."
        )

    spec = REGISTRY[name]

    if not spec.commercial_training_ok:
        allow = os.environ.get("JUPITER_ALLOW_NONCOMMERCIAL", "").lower()
        if allow not in ("1", "true", "yes"):
            raise ValueError(
                f"Dataset '{name}' is not approved for commercial training "
                f"(license: {spec.license}). "
                "Set JUPITER_ALLOW_NONCOMMERCIAL=1 to override for research use."
            )

    return spec


def list_approved_commercial() -> list[DatasetSpec]:
    """Return all datasets approved for commercial training."""
    return [s for s in REGISTRY.values() if s.commercial_training_ok]


def print_registry() -> None:
    """Print a summary of all registered datasets."""
    print(f"{'Name':<35} {'License':<20} {'Commercial':<12} {'Auth':<6} {'Size'}")
    print("-" * 90)
    for key, spec in REGISTRY.items():
        print(
            f"{spec.name:<35} {spec.license:<20} "
            f"{'Yes' if spec.commercial_training_ok else 'No':<12} "
            f"{'Yes' if spec.auth_required else 'No':<6} "
            f"{spec.approx_tokens}"
        )


if __name__ == "__main__":
    print_registry()
