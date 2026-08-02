"""
Jupiter Shot — Deduplication
==============================
Exact-match deduplication using xxHash (xxh64).
Sharded hashing for large corpora that exceed single-machine RAM.

Limitations documented:
- Full Common Crawl deduplication (~100B docs) cannot run in RAM of one machine.
- Near-duplicate detection (MinHash LSH) requires distributed infrastructure.
  The NearDuplicateDetector class provides a stub interface for future implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from pathlib import Path
from typing import Generator, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False
    logger.warning(
        "xxhash not available; falling back to SHA-256 for deduplication. "
        "Install xxhash for 10× faster hashing: pip install xxhash"
    )


def _hash_document(text: str) -> int:
    """
    Compute a 64-bit hash of a document for exact-match deduplication.

    Uses xxHash (xxh64) if available, otherwise SHA-256 truncated to 64 bits.
    xxHash is ~10× faster than SHA-256 for this use case.
    """
    if _HAS_XXHASH:
        return xxhash.xxh64(text.encode("utf-8")).intdigest()
    else:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return struct.unpack(">Q", digest[:8])[0]


class ExactMatchDeduplicator:
    """
    Exact-match document deduplicator using in-memory hash set.

    Suitable for corpora up to ~100M documents (requires ~800 MB RAM for hash set).
    For larger corpora, use ShardedDeduplicator.

    Args:
        seen_hashes: Optional pre-populated set of already-seen document hashes.
                     Useful for resuming deduplication across multiple files.
    """

    def __init__(self, seen_hashes: Optional[set[int]] = None) -> None:
        self._seen: set[int] = seen_hashes if seen_hashes is not None else set()
        self.total_seen = 0
        self.total_duplicates = 0

    @property
    def unique_count(self) -> int:
        return len(self._seen)

    def is_duplicate(self, text: str) -> bool:
        """Return True if this document has been seen before."""
        h = _hash_document(text)
        self.total_seen += 1
        if h in self._seen:
            self.total_duplicates += 1
            return True
        self._seen.add(h)
        return False

    def deduplicate(
        self, documents: Iterable[dict], text_field: str = "text"
    ) -> Generator[dict, None, None]:
        """
        Yield unique documents from an iterable, skipping duplicates.

        Args:
            documents: Iterable of dicts, each containing a text field.
            text_field: Key in each dict containing the document text.

        Yields:
            Unique documents (first occurrence of each hash).
        """
        for doc in documents:
            text = doc.get(text_field, "")
            if not text:
                continue
            if not self.is_duplicate(text):
                yield doc

    def save_hashes(self, path: str | Path) -> None:
        """Save seen hashes to disk for resumption."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for h in self._seen:
                f.write(struct.pack(">Q", h))
        logger.info(f"Saved {len(self._seen):,} hashes to {path}")

    @classmethod
    def load_hashes(cls, path: str | Path) -> "ExactMatchDeduplicator":
        """Load previously saved hashes from disk."""
        path = Path(path)
        seen: set[int] = set()
        with open(path, "rb") as f:
            while chunk := f.read(8):
                (h,) = struct.unpack(">Q", chunk)
                seen.add(h)
        logger.info(f"Loaded {len(seen):,} hashes from {path}")
        return cls(seen_hashes=seen)

    def stats(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_duplicates": self.total_duplicates,
            "unique_count": self.unique_count,
            "duplicate_rate": (
                self.total_duplicates / self.total_seen
                if self.total_seen > 0
                else 0.0
            ),
        }


class ShardedDeduplicator:
    """
    Sharded exact-match deduplicator for large corpora.

    Routes documents to shards by hash, then deduplicates within each shard.
    This allows deduplication of corpora larger than single-machine RAM.

    Memory requirement: (corpus_size / n_shards) × 8 bytes per shard.
    Example: 1B documents, 1000 shards → 1M docs/shard × 8 bytes = 8 MB/shard.

    Args:
        n_shards: Number of shards. More shards = less memory per shard.
        shard_dir: Directory to write shard files.
    """

    def __init__(self, n_shards: int = 1000, shard_dir: str | Path = "/tmp/dedup_shards") -> None:
        self.n_shards = n_shards
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._shard_files: list = [None] * n_shards

    def _get_shard_id(self, h: int) -> int:
        return h % self.n_shards

    def _get_shard_path(self, shard_id: int) -> Path:
        return self.shard_dir / f"shard_{shard_id:05d}.bin"

    def write_hashes(self, documents: Iterable[dict], text_field: str = "text") -> dict:
        """
        First pass: write document hashes to shard files.

        Returns statistics about the pass.
        """
        shard_handles = {}
        total = 0

        try:
            for doc in documents:
                text = doc.get(text_field, "")
                if not text:
                    continue
                h = _hash_document(text)
                shard_id = self._get_shard_id(h)

                if shard_id not in shard_handles:
                    path = self._get_shard_path(shard_id)
                    shard_handles[shard_id] = open(path, "ab")

                shard_handles[shard_id].write(struct.pack(">Q", h))
                total += 1

                if total % 100_000 == 0:
                    logger.info(f"Hashed {total:,} documents")
        finally:
            for f in shard_handles.values():
                f.close()

        return {"total_hashed": total, "shards_written": len(shard_handles)}

    def build_unique_set(self, shard_id: int) -> set[int]:
        """Load a shard and return the set of unique hashes."""
        path = self._get_shard_path(shard_id)
        if not path.exists():
            return set()
        seen: set[int] = set()
        with open(path, "rb") as f:
            while chunk := f.read(8):
                (h,) = struct.unpack(">Q", chunk)
                seen.add(h)
        return seen


class NearDuplicateDetector:
    """
    Stub interface for near-duplicate detection using MinHash LSH.

    NOT IMPLEMENTED in Month 1. Requires distributed infrastructure (Spark/Ray).
    This class documents the intended interface for future implementation.

    Reference: Broder (1997) "On the resemblance and containment of documents"
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.8) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        raise NotImplementedError(
            "NearDuplicateDetector is not implemented in Month 1. "
            "Near-duplicate detection requires distributed infrastructure (Apache Spark or Ray). "
            "See DATA_GOVERNANCE.md for details."
        )

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the MinHash index."""
        ...

    def find_duplicates(self, text: str) -> list[str]:
        """Return document IDs that are near-duplicates of the given text."""
        ...


def deduplicate_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    text_field: str = "text",
    deduplicator: Optional[ExactMatchDeduplicator] = None,
) -> dict:
    """
    Deduplicate a JSONL file and write unique documents to output.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file (unique documents only).
        text_field: Key in each JSON object containing the document text.
        deduplicator: Optional existing deduplicator (for cross-file dedup).

    Returns:
        Statistics dict with total_seen, total_duplicates, unique_count.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if deduplicator is None:
        deduplicator = ExactMatchDeduplicator()

    written = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line in {input_path}")
                continue

            if not deduplicator.is_duplicate(doc.get(text_field, "")):
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

    stats = deduplicator.stats()
    stats["written"] = written
    logger.info(
        f"Deduplication complete: {stats['total_seen']:,} seen, "
        f"{stats['total_duplicates']:,} duplicates ({stats['duplicate_rate']:.1%}), "
        f"{written:,} written"
    )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate a JSONL file using exact-match hashing."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file (unique documents)")
    parser.add_argument("--text-field", default="text", help="JSON key for document text")
    parser.add_argument("--load-hashes", help="Path to previously saved hash file (for resumption)")
    parser.add_argument("--save-hashes", help="Path to save hash file after deduplication")
    args = parser.parse_args()

    dedup = None
    if args.load_hashes:
        dedup = ExactMatchDeduplicator.load_hashes(args.load_hashes)

    stats = deduplicate_jsonl_file(args.input, args.output, args.text_field, dedup)
    print(json.dumps(stats, indent=2))

    if args.save_hashes and dedup:
        dedup.save_hashes(args.save_hashes)
