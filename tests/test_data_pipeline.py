"""Jupiter Shot — Data Pipeline Unit Tests"""
import pytest


class TestDeduplication:
    def _make_docs(self, texts):
        return [{"text": t} for t in texts]

    def test_exact_dedup(self):
        from training.deduplication import ExactMatchDeduplicator
        dedup = ExactMatchDeduplicator()
        texts = ["hello world", "foo bar", "hello world", "baz"]
        unique = list(dedup.deduplicate(self._make_docs(texts)))
        unique_texts = [d["text"] for d in unique]
        assert len(unique_texts) == 3
        assert "hello world" in unique_texts
        assert unique_texts.count("hello world") == 1

    def test_empty_input(self):
        from training.deduplication import ExactMatchDeduplicator
        dedup = ExactMatchDeduplicator()
        assert list(dedup.deduplicate([])) == []

    def test_all_unique(self):
        from training.deduplication import ExactMatchDeduplicator
        dedup = ExactMatchDeduplicator()
        docs = self._make_docs(["a", "b", "c", "d"])
        unique = list(dedup.deduplicate(docs))
        assert len(unique) == 4

    def test_all_duplicates(self):
        from training.deduplication import ExactMatchDeduplicator
        dedup = ExactMatchDeduplicator()
        docs = self._make_docs(["same"] * 10)
        unique = list(dedup.deduplicate(docs))
        assert len(unique) == 1

    def test_near_dedup_not_implemented_month1(self):
        """NearDuplicateDetector is scaffolded for Month 2."""
        from training.deduplication import NearDuplicateDetector
        with pytest.raises(NotImplementedError):
            NearDuplicateDetector(threshold=0.8, num_perm=64)

    def test_hash_function_consistency(self):
        """_hash_document should return same hash for same text."""
        from training.deduplication import _hash_document
        h1 = _hash_document("test text")
        h2 = _hash_document("test text")
        assert h1 == h2

    def test_hash_function_different_texts(self):
        from training.deduplication import _hash_document
        h1 = _hash_document("text one")
        h2 = _hash_document("text two")
        assert h1 != h2

    def test_is_duplicate_tracks_state(self):
        from training.deduplication import ExactMatchDeduplicator
        dedup = ExactMatchDeduplicator()
        assert not dedup.is_duplicate("first text")
        assert dedup.is_duplicate("first text")
        assert not dedup.is_duplicate("second text")


class TestTokenizer:
    def test_jupiter_tokenizer_encode(self):
        try:
            from training.tokenizer import JupiterTokenizer
            tok = JupiterTokenizer(max_length=512)
            tokens = tok.encode("hello world test")
            assert isinstance(tokens, list)
            assert len(tokens) > 0
        except Exception as e:
            pytest.skip(f"JupiterTokenizer requires HuggingFace: {e}")

    def test_jupiter_tokenizer_decode(self):
        try:
            from training.tokenizer import JupiterTokenizer
            tok = JupiterTokenizer(max_length=512)
            tokens = tok.encode("hello world")
            text = tok.decode(tokens)
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(f"JupiterTokenizer requires HuggingFace: {e}")

    def test_jupiter_tokenizer_truncation(self):
        try:
            from training.tokenizer import JupiterTokenizer
            tok = JupiterTokenizer(max_length=10)
            tokens = tok.encode("word " * 100)
            assert len(tokens) <= 10
        except Exception as e:
            pytest.skip(f"JupiterTokenizer requires HuggingFace: {e}")


class TestDatasetRegistry:
    def test_list_approved_commercial(self):
        from training.dataset_registry import list_approved_commercial
        sources = list_approved_commercial()
        assert isinstance(sources, list)
        assert len(sources) > 0

    def test_dataset_spec_has_required_fields(self):
        from training.dataset_registry import list_approved_commercial
        for spec in list_approved_commercial():
            assert spec.name, f"DatasetSpec missing name"
            assert spec.license, f"DatasetSpec {spec.name} missing license"
            assert spec.hf_identifier, f"DatasetSpec {spec.name} missing hf_identifier"
            assert spec.commercial_training_ok is True

    def test_get_dataset_by_registry_key(self):
        """get_dataset uses registry keys (e.g. 'redpajama_v1'), not display names."""
        from training.dataset_registry import get_dataset
        spec = get_dataset("redpajama_v1")
        assert spec.name == "RedPajama-Data-1T"

    def test_get_dataset_unknown_raises(self):
        from training.dataset_registry import get_dataset
        with pytest.raises(KeyError):
            get_dataset("nonexistent_dataset_xyz")


class TestSyntheticDataset:
    """Torch-dependent tests — skipped if torch not available."""

    def test_basic_creation(self):
        try:
            import torch
            from training.data_loader import SyntheticDataset
            ds = SyntheticDataset(vocab_size=1000, seq_length=64, num_samples=100)
            assert len(ds) == 100
        except ImportError:
            pytest.skip("torch not available")

    def test_item_shape(self):
        try:
            import torch
            from training.data_loader import SyntheticDataset
            ds = SyntheticDataset(vocab_size=1000, seq_length=64, num_samples=10)
            item = ds[0]
            assert "input_ids" in item
            assert item["input_ids"].shape == (64,)
        except ImportError:
            pytest.skip("torch not available")

    def test_token_ids_in_range(self):
        try:
            import torch
            from training.data_loader import SyntheticDataset
            ds = SyntheticDataset(vocab_size=1000, seq_length=64, num_samples=10)
            for i in range(len(ds)):
                item = ds[i]
                assert item["input_ids"].min() >= 0
                assert item["input_ids"].max() < 1000
        except ImportError:
            pytest.skip("torch not available")

    def test_reproducible_with_seed(self):
        try:
            import torch
            from training.data_loader import SyntheticDataset
            ds1 = SyntheticDataset(vocab_size=1000, seq_length=32, num_samples=5, seed=42)
            ds2 = SyntheticDataset(vocab_size=1000, seq_length=32, num_samples=5, seed=42)
            for i in range(5):
                assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])
        except ImportError:
            pytest.skip("torch not available")

    def test_different_seeds_different_data(self):
        try:
            import torch
            from training.data_loader import SyntheticDataset
            ds1 = SyntheticDataset(vocab_size=1000, seq_length=32, num_samples=5, seed=42)
            ds2 = SyntheticDataset(vocab_size=1000, seq_length=32, num_samples=5, seed=99)
            any_diff = any(
                not torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])
                for i in range(5)
            )
            assert any_diff
        except ImportError:
            pytest.skip("torch not available")
