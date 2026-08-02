# Jupiter Shot — Data Governance

**Program:** Jupiter Shot  
**Phase:** Month 1  
**Last Updated:** 2025-08-02

---

## 1. Principles

1. **No silent substitution.** If a dataset is unavailable, the pipeline raises an error and documents the issue. It does not substitute a different dataset without explicit configuration.
2. **License tracking.** Every dataset used in training records its license, commercial training permissions, and redistribution restrictions in the dataset registry.
3. **Deduplication honesty.** Full Common Crawl deduplication cannot run on a single machine. The Month 1 pipeline uses sharded exact-match hashing. Near-duplicate detection (MinHash LSH) requires distributed infrastructure.
4. **Authentication transparency.** Gated datasets (requiring HuggingFace authentication or special licenses) are documented. The pipeline fails clearly if required credentials are absent.

---

## 2. Dataset Registry

### 2.1 Approved Month 1 Datasets

| Dataset | HF Identifier | License | Commercial Training | Streaming | Size | Auth Required | Notes |
|---------|--------------|---------|---------------------|-----------|------|---------------|-------|
| RedPajama-Data-1T | `togethercomputer/RedPajama-Data-1T` | Apache 2.0 | ✅ Yes | ✅ Yes | ~1T tokens | No | Recommended for Month 1 |
| RedPajama-Data-v2 | `togethercomputer/RedPajama-Data-V2` | Apache 2.0 | ✅ Yes | ✅ Yes | ~30T tokens | No | Large; use subset |
| Dolma v1.7 | `allenai/dolma` | AI2 ImpACT | Research only | ✅ Yes | ~3T tokens | HF token | Non-commercial research |
| OpenWebText2 (via Pile) | `EleutherAI/the_pile_openwebtext2` | MIT | ✅ Yes | ✅ Yes | ~65GB | No | Good quality web text |
| Wikipedia (HF) | `wikimedia/wikipedia` | CC BY-SA 4.0 | ⚠️ Attribution required | ✅ Yes | ~20GB | No | Attribution required in model card |

### 2.2 Conditionally Approved (Requires License Review)

| Dataset | HF Identifier | License | Issue | Resolution Required |
|---------|--------------|---------|-------|---------------------|
| The Pile | `EleutherAI/pile` | Mixed | Some subsets non-commercial (Books3, PubMed Central) | Use only CC/Apache subsets |
| StarCoder | `bigcode/starcoderdata` | BigCode OpenRAIL | No military use; attribution required | Acceptable for non-military use |
| LLaMA-2 tokenizer | `meta-llama/Llama-2-7b-hf` | Meta Community License | Gated; requires Meta approval | Request access at meta.ai |

### 2.3 Not Approved

| Dataset | Reason |
|---------|--------|
| Common Crawl (raw) | No license; requires extensive filtering and deduplication |
| Books3 | Copyright status disputed; removed from The Pile v2 |
| C4 | Derived from Common Crawl; license unclear for commercial training |
| Proprietary web crawls | No license |

---

## 3. Deduplication

### 3.1 Month 1: Exact-Match Deduplication

The Month 1 pipeline implements exact-match deduplication using xxHash (xxh64) of document content. This catches identical documents but not near-duplicates.

**Implementation:** `training/deduplication.py`

**Limitations:**
- Cannot detect near-duplicates (slightly modified copies)
- Memory scales with corpus size: 1T tokens at 1KB/doc = ~1B documents = ~8 GB for hash set
- For corpora > 100M documents, use sharded hashing (see below)

### 3.2 Sharded Hashing for Large Corpora

Full Common Crawl deduplication (~100B documents) cannot run in the RAM of one machine. The sharded approach:

1. Hash each document: `shard_id = hash(doc) % N_shards`
2. Route each document to its shard file
3. Within each shard, deduplicate independently
4. Merge results

This requires ~N_shards × (RAM for one shard). For 100B documents with 1,000 shards: ~100M docs/shard × 8 bytes/hash = ~800 MB/shard. Feasible on a single machine with 1,000 shards.

### 3.3 Near-Duplicate Detection (Future)

MinHash LSH (Locality-Sensitive Hashing) detects near-duplicates with configurable similarity threshold. Requires:
- Distributed processing (Apache Spark or Ray)
- ~10× more compute than exact-match deduplication
- Not implemented in Month 1

The `training/deduplication.py` module provides a `NearDuplicateDetector` interface stub for future implementation.

---

## 4. Tokenizer Governance

### 4.1 Default Tokenizer

**GPT-NeoX-20B tokenizer** (`EleutherAI/gpt-neox-20b`)
- License: Apache 2.0
- Vocabulary: 50,257 tokens
- No authentication required
- BPE tokenizer
- Compatible with all Month 1 training scripts

### 4.2 Alternative Tokenizers

| Tokenizer | Source | Vocab | Auth | License | Notes |
|-----------|--------|-------|------|---------|-------|
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | 32,000 | HF token | Apache 2.0 | Preferred for quality |
| LLaMA-2 | `meta-llama/Llama-2-7b-hf` | 32,000 | HF token + Meta approval | Meta Community | Gated; requires approval |
| Falcon | `tiiuae/falcon-7b` | 65,024 | No | Apache 2.0 | Large vocabulary |

**To use a gated tokenizer:**
```bash
huggingface-cli login  # Enter your HF token
# Then set tokenizer_name_or_path in config
```

### 4.3 Custom Tokenizer Training (Future)

For optimal performance on domain-specific corpora, a custom SentencePiece tokenizer should be trained on a representative sample of the training data. This is not implemented in Month 1 but the `training/tokenizer.py` module supports loading a local tokenizer path.

---

## 5. Data Pipeline Compliance

### 5.1 What Is Logged

For each training run, the pipeline logs:
- Dataset name and version
- Dataset license
- Commercial training permission
- Number of documents and tokens consumed
- Deduplication method applied
- Tokenizer name and version
- Sequence length and packing strategy

### 5.2 What Is NOT Stored

- Raw document content (not stored in training logs)
- Personal information (filtered at source; no additional PII scanning in Month 1)
- Prompt content from inference (see Mesh compliance logger)

### 5.3 Future Requirements

For production training at Stage 3+:
- PII detection and redaction pipeline
- Toxicity filtering
- Legal review of all dataset licenses
- Redistribution restrictions audit
- Data provenance tracking per training token

---

## 6. Estimated Storage Requirements

| Corpus | Raw Size | After Dedup | Tokenized (GPT-NeoX, 2048 seq) | Notes |
|--------|----------|-------------|-------------------------------|-------|
| RedPajama-1T (full) | ~2.5 TB | ~1.5 TB | ~2 TB | Stream; do not download fully |
| Month 1 sample (10B tokens) | ~25 GB | ~20 GB | ~20 GB | Feasible on local NVMe |
| Month 1 smoke test (500M tokens) | ~1.25 GB | ~1 GB | ~1 GB | Easily fits in RAM |

**Month 1 recommendation:** Stream from HuggingFace. Do not download the full corpus. Use `datasets.load_dataset(..., streaming=True)` for all training runs.

---

*This document is a living governance record. Update as datasets are added, licenses change, or deduplication infrastructure evolves.*
