"""Tests for service layer."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.services.chromadb_client import ChromaDBClient, CollectionNotFoundError


class TestChromaDBClient:
    """Test ChromaDB client."""

    @pytest.fixture
    def client(self):
        """Create ChromaDB client."""
        return ChromaDBClient()

    def test_list_available_collections(self, client):
        """Test listing available collections."""
        collections = client.list_available_collections()

        # Should have at least some collections
        assert isinstance(collections, list)

        # Check expected collections exist
        expected = ["GLageKon", "NORM", "StrlSch", "StrlSchExt"]
        for name in expected:
            assert name in collections, f"Missing collection: {name}"

    def test_get_collection_stats(self, client):
        """Test getting collection statistics."""
        collections = client.list_available_collections()

        if collections:
            stats = client.get_collection_stats(collections[0])
            assert "name" in stats
            assert "count" in stats
            assert stats["count"] > 0

    def test_search_single_collection(self, client):
        """Test searching a single collection."""
        collections = client.list_available_collections()

        if collections:
            results = client.search(
                query="Strahlenschutz",
                collection_key=collections[0],
                top_k=3,
            )

            assert isinstance(results, list)
            # Should get results for German regulatory query
            if results:
                assert results[0].collection == collections[0]
                assert results[0].relevance_score > 0

    def test_search_invalid_collection(self, client):
        """Test searching non-existent collection."""
        with pytest.raises(CollectionNotFoundError):
            client.search("test", "NonExistent")

    def test_search_all_collections(self, client):
        """Test searching all collections."""
        results = client.search_all_collections(
            query="Grenzwerte",
            top_k=2,
        )

        assert isinstance(results, dict)
        # Should have entries for all available collections
        for key in client.list_available_collections():
            assert key in results

    def test_search_multi_collection(self, client):
        """Test multi-collection merged search."""
        results = client.search_multi_collection(
            query="Strahlenexposition",
            top_k=2,
        )

        assert isinstance(results, list)
        # Results should be sorted by relevance
        if len(results) > 1:
            assert results[0].relevance_score >= results[1].relevance_score

    def test_get_pdf_folder(self, client):
        """Test getting PDF folder path."""
        folder = client.get_pdf_folder("GLageKon")
        assert "GLageKon__db_inserted" in str(folder)

    def test_get_pdf_folder_invalid(self, client):
        """Test getting PDF folder for invalid collection."""
        with pytest.raises(CollectionNotFoundError):
            client.get_pdf_folder("NonExistent")


class TestEmbeddingDerivation:
    """Test embedding model derivation from database names."""

    def test_extract_embedding_model_standard(self):
        """Test extracting model from standard DB name pattern."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        assert (
            client.extract_embedding_model("GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000")
            == "Qwen/Qwen3-Embedding-0.6B"
        )

    def test_extract_embedding_model_different_model(self):
        """Test extracting a different model name."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        assert (
            client.extract_embedding_model("NORM__BAAI--bge-m3--3000--600")
            == "BAAI/bge-m3"
        )

    def test_extract_embedding_model_no_double_underscore(self):
        """Test returns None when DB name has no __ separator."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        assert client.extract_embedding_model("plain_name") is None

    def test_extract_embedding_model_single_model_part(self):
        """Test returns single part when no -- separator in model."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        assert client.extract_embedding_model("Coll__SomeModel") == "SomeModel"

    def test_resolve_embedding_model_success(self):
        """Test _resolve_embedding_model returns extracted model."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        result = client._resolve_embedding_model(
            "GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000"
        )
        assert result == "Qwen/Qwen3-Embedding-0.6B"

    def test_resolve_embedding_model_fallback(self):
        """Test _resolve_embedding_model falls back to config default."""
        client = ChromaDBClient.__new__(ChromaDBClient)
        result = client._resolve_embedding_model("plain_name")
        assert result == settings.default_embedding_model

    @patch("src.services.chromadb_client.HuggingFaceEmbeddings")
    def test_get_embeddings_caching(self, mock_hf):
        """Test that same model name returns same cached instance."""
        mock_hf.return_value = MagicMock()
        client = ChromaDBClient.__new__(ChromaDBClient)
        client._embedding_cache = {}

        emb1 = client._get_embeddings("Qwen/Qwen3-Embedding-0.6B")
        emb2 = client._get_embeddings("Qwen/Qwen3-Embedding-0.6B")

        assert emb1 is emb2
        assert mock_hf.call_count == 1

    @patch("src.services.chromadb_client.HuggingFaceEmbeddings")
    def test_different_models_get_different_embeddings(self, mock_hf):
        """Test that different model names create separate instances."""
        mock_hf.side_effect = [MagicMock(), MagicMock()]
        client = ChromaDBClient.__new__(ChromaDBClient)
        client._embedding_cache = {}

        emb1 = client._get_embeddings("Qwen/Qwen3-Embedding-0.6B")
        emb2 = client._get_embeddings("BAAI/bge-m3")

        assert emb1 is not emb2
        assert mock_hf.call_count == 2


class TestOllamaClient:
    """Test Ollama client."""

    @pytest.fixture
    def client(self):
        """Create Ollama client."""
        from src.services.ollama_client import OllamaClient
        return OllamaClient()

    def test_estimate_tokens(self, client):
        """Test token estimation."""
        text = "This is a test string with some words."
        tokens = client.estimate_tokens(text)

        # Rough estimate: ~4 chars per token
        assert tokens > 0
        assert tokens < len(text)

    def test_check_context_limit_pass(self, client):
        """Test context limit check passes for small text."""
        # Small text should pass
        client.check_context_limit("Small prompt")
        # No exception raised

    def test_check_context_limit_fail(self, client):
        """Test context limit check fails for huge text."""
        from src.services.ollama_client import ContextOverflowError

        # Create text that exceeds safe limit
        huge_text = "x" * (settings.safe_context_limit * 5)

        with pytest.raises(ContextOverflowError):
            client.check_context_limit(huge_text)


class TestConfig:
    """Test configuration."""

    def test_settings_loaded(self):
        """Test settings are loaded."""
        assert settings.ollama_model is not None
        assert settings.ollama_num_ctx > 0

    def test_safe_context_limit(self):
        """Test safe context limit calculation."""
        assert settings.safe_context_limit < settings.ollama_num_ctx
        assert settings.safe_context_limit == int(
            settings.ollama_num_ctx * settings.ollama_safe_limit
        )

    def test_chromadb_path(self):
        """Test ChromaDB path resolution."""
        path = settings.chromadb_path_resolved
        assert path.is_absolute()
