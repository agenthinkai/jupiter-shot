"""
Jupiter Shot — Data Loader
===========================
Supports JSONL, Parquet, local datasets, and HuggingFace streaming datasets.
Implements resumable streaming, deterministic shuffling, dataset weighting/mixing,
and packed sequence batching for causal language model training.

RAM/disk requirements:
- Month 1 smoke test (500M tokens): ~1 GB RAM, ~1 GB disk
- Full RedPajama-1T (streaming): ~4 GB RAM (buffer), no disk required
- Full corpus download: ~2.5 TB disk — DO NOT download; use streaming
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Union

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetSource:
    """Configuration for a single dataset source in a mixed training corpus."""

    name: str
    """Dataset name (must be registered in dataset_registry or a local path)."""

    weight: float = 1.0
    """Sampling weight for dataset mixing. Higher = more samples from this source."""

    text_field: str = "text"
    """JSON field containing the document text."""

    local_path: Optional[str] = None
    """Path to local JSONL or Parquet file. If set, overrides HF streaming."""

    hf_subset: Optional[str] = None
    """HuggingFace dataset subset/config name."""

    hf_split: str = "train"
    """HuggingFace dataset split."""

    max_documents: Optional[int] = None
    """Maximum number of documents to use from this source (for smoke tests)."""

    metadata: dict = field(default_factory=dict)
    """Source metadata (license, version, etc.) for logging."""


class PackedSequenceDataset(IterableDataset):
    """
    Iterable dataset that packs tokenized documents into fixed-length sequences.

    Documents are concatenated with EOS tokens between them, then split into
    chunks of exactly `seq_length` tokens. This maximizes GPU utilization by
    avoiding padding.

    Supports:
    - Resumable streaming (via global_token_offset)
    - Deterministic shuffling (via shuffle_seed)
    - Dataset weighting and mixing
    - Multiple source formats (JSONL, Parquet, HF streaming)

    Args:
        sources: List of DatasetSource configurations.
        tokenizer: JupiterTokenizer instance.
        seq_length: Sequence length for packed batches.
        shuffle_seed: Seed for deterministic shuffling. None = no shuffle.
        global_token_offset: Number of tokens already consumed (for resumption).
        buffer_size: Number of documents to buffer for shuffling.
    """

    def __init__(
        self,
        sources: list[DatasetSource],
        tokenizer,
        seq_length: int = 2048,
        shuffle_seed: Optional[int] = 42,
        global_token_offset: int = 0,
        buffer_size: int = 10_000,
    ) -> None:
        self.sources = sources
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle_seed = shuffle_seed
        self.global_token_offset = global_token_offset
        self.buffer_size = buffer_size

        # Normalize weights
        total_weight = sum(s.weight for s in sources)
        self._weights = [s.weight / total_weight for s in sources]

    def _iter_source(self, source: DatasetSource) -> Generator[list[int], None, None]:
        """Yield tokenized documents from a single source."""
        if source.local_path is not None:
            yield from self._iter_local(source)
        else:
            yield from self._iter_hf_streaming(source)

    def _iter_local(self, source: DatasetSource) -> Generator[list[int], None, None]:
        """Yield tokenized documents from a local JSONL or Parquet file."""
        path = Path(source.local_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        count = 0
        if path.suffix == ".jsonl" or path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = doc.get(source.text_field, "")
                    if text:
                        tokens = self.tokenizer.tokenize_for_training(text)
                        if tokens:
                            yield tokens
                            count += 1
                            if source.max_documents and count >= source.max_documents:
                                return
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError("pyarrow required for Parquet support: pip install pyarrow")
            table = pq.read_table(path, columns=[source.text_field])
            for batch in table.to_batches():
                for text in batch.column(source.text_field).to_pylist():
                    if text:
                        tokens = self.tokenizer.tokenize_for_training(str(text))
                        if tokens:
                            yield tokens
                            count += 1
                            if source.max_documents and count >= source.max_documents:
                                return
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .jsonl or .parquet")

    def _iter_hf_streaming(self, source: DatasetSource) -> Generator[list[int], None, None]:
        """Yield tokenized documents from a HuggingFace streaming dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets required: pip install datasets")

        from training.dataset_registry import get_dataset
        spec = get_dataset(source.name)

        logger.info(
            f"Streaming dataset '{source.name}' from HuggingFace "
            f"(license: {spec.license}, commercial: {spec.commercial_training_ok})"
        )

        load_kwargs = dict(
            path=spec.hf_identifier,
            split=source.hf_split,
            streaming=True,
            trust_remote_code=False,
        )
        if spec.subset or source.hf_subset:
            load_kwargs["name"] = source.hf_subset or spec.subset

        if spec.auth_required:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                raise EnvironmentError(
                    f"Dataset '{source.name}' requires HuggingFace authentication. "
                    "Set HF_TOKEN environment variable."
                )
            load_kwargs["token"] = token

        dataset = load_dataset(**load_kwargs)

        count = 0
        for doc in dataset:
            text = doc.get(source.text_field, "")
            if text:
                tokens = self.tokenizer.tokenize_for_training(str(text))
                if tokens:
                    yield tokens
                    count += 1
                    if source.max_documents and count >= source.max_documents:
                        return

    def _iter_mixed(self) -> Generator[list[int], None, None]:
        """
        Yield tokenized documents from mixed sources according to weights.

        Uses a buffer for shuffling. Deterministic when shuffle_seed is set.
        """
        rng = random.Random(self.shuffle_seed)

        # Create iterators for each source
        iters = [self._iter_source(s) for s in self.sources]
        active = list(range(len(self.sources)))

        buffer: list[list[int]] = []

        while active:
            # Fill buffer
            while len(buffer) < self.buffer_size and active:
                # Sample a source according to weights
                active_weights = [self._weights[i] for i in active]
                total = sum(active_weights)
                normalized = [w / total for w in active_weights]

                chosen_idx = rng.choices(range(len(active)), weights=normalized, k=1)[0]
                source_idx = active[chosen_idx]

                try:
                    tokens = next(iters[source_idx])
                    buffer.append(tokens)
                except StopIteration:
                    active.remove(source_idx)

            if not buffer:
                break

            # Shuffle buffer
            if self.shuffle_seed is not None:
                rng.shuffle(buffer)

            # Yield from buffer
            yield buffer.pop(0)

        # Yield remaining buffer
        if self.shuffle_seed is not None:
            rng.shuffle(buffer)
        yield from buffer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield packed sequences as dicts with 'input_ids' and 'labels'.

        Sequences are packed to exactly seq_length tokens.
        Labels are shifted input_ids (standard causal LM objective).
        """
        token_buffer: list[int] = []
        tokens_consumed = 0

        for doc_tokens in self._iter_mixed():
            token_buffer.extend(doc_tokens)

            while len(token_buffer) >= self.seq_length + 1:
                # Extract one packed sequence (seq_length + 1 for labels)
                chunk = token_buffer[: self.seq_length + 1]
                token_buffer = token_buffer[self.seq_length + 1:]

                # Skip tokens already consumed (for resumption)
                tokens_consumed += self.seq_length
                if tokens_consumed <= self.global_token_offset:
                    continue

                input_ids = torch.tensor(chunk[:self.seq_length], dtype=torch.long)
                labels = torch.tensor(chunk[1:self.seq_length + 1], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for unit testing and Gate A validation.
    Generates random token sequences without requiring real data.

    Args:
        vocab_size: Vocabulary size for random token generation.
        seq_length: Sequence length.
        num_samples: Number of samples in the dataset.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        seq_length: int = 512,
        num_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.seed = seed

        rng = torch.Generator()
        rng.manual_seed(seed)
        self._data = torch.randint(
            0, vocab_size, (num_samples, seq_length + 1), generator=rng
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self._data[idx]
        return {
            "input_ids": chunk[:self.seq_length],
            "labels": chunk[1:self.seq_length + 1],
        }


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for training.

    Args:
        dataset: Dataset or IterableDataset instance.
        batch_size: Batch size per GPU.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # No shuffle for IterableDataset (shuffling is handled internally)
        shuffle=isinstance(dataset, Dataset),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Jupiter data loader.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--dataset", default="redpajama_v1", help="Dataset name")
    parser.add_argument("--max-docs", type=int, default=100, help="Max documents to load")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    if args.synthetic:
        dataset = SyntheticDataset(seq_length=args.seq_length, num_samples=1000)
        loader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=0)
        for i, batch in enumerate(loader):
            print(f"Batch {i}: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
            if i >= 2:
                break
    else:
        from training.tokenizer import get_tokenizer
        tokenizer = get_tokenizer("gpt-neox", max_length=args.seq_length)
        source = DatasetSource(
            name=args.dataset,
            max_documents=args.max_docs,
        )
        dataset = PackedSequenceDataset(
            sources=[source],
            tokenizer=tokenizer,
            seq_length=args.seq_length,
        )
        loader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=0)
        for i, batch in enumerate(loader):
            print(f"Batch {i}: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
            if i >= 2:
                break

    print("Data loader test complete.")
