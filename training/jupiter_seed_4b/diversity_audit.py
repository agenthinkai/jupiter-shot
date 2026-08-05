#!/usr/bin/env python3
"""
Jupiter Seed 4B — Dataset Diversity Audit
==========================================
Deterministic, CPU-compatible diversity analysis.
No embedding models. No paid APIs. No GPU.

Measures:
  - Unique normalized prompts and responses
  - Unique prompt/response skeletons (stripped of entities/numbers)
  - Repeated opening/closing phrases
  - N-gram overlap
  - TF-IDF similarity clusters
  - Near-duplicate clusters (Jaccard on word 3-grams)
  - Template family sizes
  - Per-domain, per-language, per-task-type diversity

Usage:
    python3 diversity_audit.py --data-dir training/jupiter_seed_4b/data
                               --output docs/jupiter_seed_4b/DATASET_DIVERSITY_AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MAX_TEMPLATE_FAMILY_PCT = 2.0  # No family may exceed 2% of total dataset


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def skeleton(text: str) -> str:
    """Strip numbers, entity-like capitalized words, and variant markers."""
    t = normalise(text)
    t = re.sub(r"\b\d[\d,\.]*\b", "NUM", t)
    t = re.sub(r"\[.*?\]", "", t)  # Remove variant markers
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_ngrams(text: str, n: int = 3) -> Set[str]:
    words = normalise(text).split()
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def tfidf_vectors(docs: List[str]) -> List[Dict[str, float]]:
    """Compute TF-IDF vectors for a list of documents."""
    tokenized = [normalise(d).split() for d in docs]
    N = len(tokenized)
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    vectors: List[Dict[str, float]] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((N + 1) / (df[term] + 1)) + 1
            vec[term] = (count / total) * idf
        vectors.append(vec)
    return vectors


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v**2 for v in a.values()))
    norm_b = math.sqrt(sum(v**2 for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> List[Dict]:
    records = []
    for split in ("train", "valid", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            log.error("Split file not found: %s", path)
            sys.exit(1)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    log.info("Loaded %d total records", len(records))
    return records


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_uniqueness(records: List[Dict]) -> Dict:
    prompts = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]
    norm_prompts = [normalise(p) for p in prompts]
    norm_responses = [normalise(r) for r in responses]
    skel_prompts = [skeleton(p) for p in prompts]
    skel_responses = [skeleton(r) for r in responses]

    return {
        "total_records": len(records),
        "unique_normalized_prompts": len(set(norm_prompts)),
        "unique_normalized_responses": len(set(norm_responses)),
        "unique_prompt_skeletons": len(set(skel_prompts)),
        "unique_response_skeletons": len(set(skel_responses)),
        "duplicate_prompt_count": len(norm_prompts) - len(set(norm_prompts)),
        "duplicate_response_count": len(norm_responses) - len(set(norm_responses)),
    }


def audit_opening_closing(records: List[Dict], top_n: int = 10) -> Dict:
    openings: Counter = Counter()
    closings: Counter = Counter()
    for r in records:
        resp = normalise(r["response"])
        words = resp.split()
        if len(words) >= 5:
            openings[" ".join(words[:5])] += 1
        if len(words) >= 5:
            closings[" ".join(words[-5:])] += 1
    return {
        "top_opening_phrases": openings.most_common(top_n),
        "top_closing_phrases": closings.most_common(top_n),
        "max_opening_repetition": openings.most_common(1)[0][1] if openings else 0,
        "max_closing_repetition": closings.most_common(1)[0][1] if closings else 0,
    }


def audit_ngram_overlap(records: List[Dict], n: int = 3) -> Dict:
    """Compute average pairwise Jaccard similarity on a sample."""
    sample = records[:min(200, len(records))]
    ngram_sets = [word_ngrams(r["prompt"], n) for r in sample]
    similarities = []
    for i in range(len(ngram_sets)):
        for j in range(i + 1, min(i + 20, len(ngram_sets))):
            sim = jaccard(ngram_sets[i], ngram_sets[j])
            similarities.append(sim)
    avg_sim = sum(similarities) / max(len(similarities), 1)
    high_sim_pairs = sum(1 for s in similarities if s >= 0.5)
    return {
        "sample_size": len(sample),
        "pairs_checked": len(similarities),
        "average_jaccard_similarity": round(avg_sim, 4),
        "high_similarity_pairs_gte_50pct": high_sim_pairs,
        "ngram_size": n,
    }


def audit_template_families(records: List[Dict]) -> Dict:
    """Group records by prompt skeleton and measure family sizes."""
    families: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        skel = skeleton(r["prompt"])
        families[skel].append(r["example_id"])

    total = len(records)
    max_family_size = max(len(v) for v in families.values()) if families else 0
    max_family_pct = (max_family_size / total * 100) if total > 0 else 0
    oversized = {k: len(v) for k, v in families.items()
                 if len(v) / total * 100 > MAX_TEMPLATE_FAMILY_PCT}

    return {
        "total_families": len(families),
        "max_family_size": max_family_size,
        "max_family_pct": round(max_family_pct, 2),
        "families_exceeding_2pct": len(oversized),
        "oversized_family_details": {k[:60]: v for k, v in list(oversized.items())[:5]},
        "threshold_pct": MAX_TEMPLATE_FAMILY_PCT,
        "pass": len(oversized) == 0,
    }


def audit_per_dimension(records: List[Dict], key: str) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r.get(key, "unknown")].append(r)

    result = {}
    for group_name, group_records in sorted(groups.items()):
        skel_prompts = {skeleton(r["prompt"]) for r in group_records}
        result[group_name] = {
            "count": len(group_records),
            "unique_skeletons": len(skel_prompts),
            "diversity_ratio": round(len(skel_prompts) / max(len(group_records), 1), 3),
        }
    return result


def audit_near_duplicates(records: List[Dict], threshold: float = 0.7) -> Dict:
    """Find near-duplicate pairs using Jaccard on 3-grams (sample for performance)."""
    sample = records[:min(300, len(records))]
    ngram_sets = [word_ngrams(r["prompt"], 3) for r in sample]
    near_dup_pairs = []
    for i in range(len(ngram_sets)):
        for j in range(i + 1, min(i + 30, len(ngram_sets))):
            sim = jaccard(ngram_sets[i], ngram_sets[j])
            if sim >= threshold:
                near_dup_pairs.append({
                    "id_a": sample[i]["example_id"],
                    "id_b": sample[j]["example_id"],
                    "similarity": round(sim, 3),
                })
    return {
        "sample_size": len(sample),
        "near_duplicate_pairs": len(near_dup_pairs),
        "threshold": threshold,
        "examples": near_dup_pairs[:10],
    }


def audit_tfidf_clusters(records: List[Dict], sim_threshold: float = 0.8) -> Dict:
    """Find high-similarity TF-IDF pairs (sample for performance)."""
    sample = records[:min(150, len(records))]
    prompts = [r["prompt"] for r in sample]
    vectors = tfidf_vectors(prompts)
    high_sim_pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, min(i + 20, len(vectors))):
            sim = cosine_sim(vectors[i], vectors[j])
            if sim >= sim_threshold:
                high_sim_pairs.append({
                    "id_a": sample[i]["example_id"],
                    "id_b": sample[j]["example_id"],
                    "tfidf_cosine_similarity": round(sim, 3),
                })
    return {
        "sample_size": len(sample),
        "high_similarity_pairs": len(high_sim_pairs),
        "threshold": sim_threshold,
        "examples": high_sim_pairs[:5],
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Jupiter Seed 4B: Dataset Diversity Audit",
        "",
        "**Label:** AI-ASSISTED INTERNAL DIVERSITY AUDIT — NOT HUMAN APPROVAL",
        "",
        "This audit uses deterministic, CPU-compatible methods only. "
        "No embedding models, paid APIs, or GPU were used.",
        "",
        "---",
        "",
        "## 1. Uniqueness Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    u = results["uniqueness"]
    for k, v in u.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

    lines += [
        "",
        "## 2. Template Family Analysis",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    tf = results["template_families"]
    lines.append(f"| Total Families | {tf['total_families']} |")
    lines.append(f"| Max Family Size | {tf['max_family_size']} |")
    lines.append(f"| Max Family % | {tf['max_family_pct']}% |")
    lines.append(f"| Families Exceeding 2% | {tf['families_exceeding_2pct']} |")
    lines.append(f"| **Template Family Check** | {'**PASS**' if tf['pass'] else '**FAIL**'} |")
    if tf["oversized_family_details"]:
        lines += ["", "**Oversized families:**", ""]
        for skel, count in tf["oversized_family_details"].items():
            lines.append(f"- `{skel[:80]}...` — {count} examples")

    lines += [
        "",
        "## 3. N-gram Overlap (3-gram Jaccard, sample)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    ng = results["ngram_overlap"]
    lines.append(f"| Sample Size | {ng['sample_size']} |")
    lines.append(f"| Average Jaccard Similarity | {ng['average_jaccard_similarity']} |")
    lines.append(f"| High-Similarity Pairs (≥50%) | {ng['high_similarity_pairs_gte_50pct']} |")

    lines += [
        "",
        "## 4. Near-Duplicate Detection",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    nd = results["near_duplicates"]
    lines.append(f"| Sample Size | {nd['sample_size']} |")
    lines.append(f"| Near-Duplicate Pairs (≥{nd['threshold']}) | {nd['near_duplicate_pairs']} |")

    lines += [
        "",
        "## 5. TF-IDF Similarity Clusters (sample)",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    tc = results["tfidf_clusters"]
    lines.append(f"| Sample Size | {tc['sample_size']} |")
    lines.append(f"| High-Similarity Pairs (≥{tc['threshold']}) | {tc['high_similarity_pairs']} |")

    lines += [
        "",
        "## 6. Opening/Closing Phrase Repetition",
        "",
        f"| Max Opening Repetition | {results['opening_closing']['max_opening_repetition']} |",
        "| :--- | :--- |",
        f"| Max Closing Repetition | {results['opening_closing']['max_closing_repetition']} |",
        "",
        "**Top 5 Opening Phrases:**",
        "",
    ]
    for phrase, count in results["opening_closing"]["top_opening_phrases"][:5]:
        lines.append(f"- `{phrase}` — {count} occurrences")

    lines += [
        "",
        "## 7. Per-Domain Diversity",
        "",
        "| Domain | Count | Unique Skeletons | Diversity Ratio |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for domain, stats in results["per_domain"].items():
        lines.append(
            f"| {domain} | {stats['count']} | {stats['unique_skeletons']} | {stats['diversity_ratio']} |"
        )

    lines += [
        "",
        "## 8. Per-Language Diversity",
        "",
        "| Language | Count | Unique Skeletons | Diversity Ratio |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for lang, stats in results["per_language"].items():
        lines.append(
            f"| {lang} | {stats['count']} | {stats['unique_skeletons']} | {stats['diversity_ratio']} |"
        )

    lines += [
        "",
        "## 9. Per-Task-Type Diversity",
        "",
        "| Task Type | Count | Unique Skeletons | Diversity Ratio |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for task, stats in results["per_task_type"].items():
        lines.append(
            f"| {task} | {stats['count']} | {stats['unique_skeletons']} | {stats['diversity_ratio']} |"
        )

    lines += [
        "",
        "## 10. Verdict",
        "",
        "| Check | Result |",
        "| :--- | :--- |",
        f"| No template family exceeds 2% | {'PASS' if tf['pass'] else 'FAIL'} |",
        f"| Unique prompt skeletons > 50% of total | {'PASS' if u['unique_prompt_skeletons'] > u['total_records'] * 0.5 else 'REVISE'} |",
        f"| Average Jaccard similarity < 0.3 | {'PASS' if ng['average_jaccard_similarity'] < 0.3 else 'REVISE'} |",
        "",
        "**AI-ASSISTED INTERNAL DIVERSITY AUDIT — NOT HUMAN APPROVAL**",
    ]

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Diversity audit report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Diversity Audit")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("training/jupiter_seed_4b/data"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/jupiter_seed_4b/DATASET_DIVERSITY_AUDIT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_all(args.data_dir)

    results = {
        "uniqueness": audit_uniqueness(records),
        "template_families": audit_template_families(records),
        "ngram_overlap": audit_ngram_overlap(records),
        "near_duplicates": audit_near_duplicates(records),
        "tfidf_clusters": audit_tfidf_clusters(records),
        "opening_closing": audit_opening_closing(records),
        "per_domain": audit_per_dimension(records, "domain"),
        "per_language": audit_per_dimension(records, "language"),
        "per_task_type": audit_per_dimension(records, "task_type"),
    }

    write_report(results, args.output)

    tf = results["template_families"]
    if not tf["pass"]:
        log.error(
            "TEMPLATE FAMILY CHECK FAILED: %d families exceed 2%%. "
            "Largest family: %d records (%.1f%%)",
            tf["families_exceeding_2pct"],
            tf["max_family_size"],
            tf["max_family_pct"],
        )
        sys.exit(1)
    log.info("Diversity audit PASSED. Max template family: %.1f%%", tf["max_family_pct"])


if __name__ == "__main__":
    main()
