"""Tests for batch reranking in nodes.py."""

from unittest.mock import MagicMock, patch

from src.agents.nodes import (
    SCORE_TO_100,
    _build_reranker_batches,
    _format_batch_chunks_for_prompt,
    _normalize_batch_scores,
    _rerank_task_chunks,
)


def _make_chunk(doc="doc.pdf", page=1, text="some text", score=0.8, tier=1, **kwargs):
    """Helper to build a chunk dict."""
    return {
        "chunk": text,
        "document": doc,
        "page": page,
        "extracted_info": f"extracted: {text}",
        "relevance_score": score,
        "context_tier": tier,
        **kwargs,
    }


class TestBuildRerankerBatches:
    """Tests for _build_reranker_batches."""

    def test_empty_input(self):
        assert _build_reranker_batches([]) == []

    def test_round_robin_distribution(self):
        """7 chunks with batch_size=3 → 3 batches, round-robin."""
        chunks = [_make_chunk(text=f"c{i}") for i in range(7)]
        batches = _build_reranker_batches(chunks, batch_size=3)
        assert len(batches) == 3
        # Round-robin: batch 0 gets indices 0,3,6; batch 1 gets 1,4; batch 2 gets 2,5
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2

    def test_fewer_than_batch_size(self):
        """2 chunks with batch_size=6 → 1 batch."""
        chunks = [_make_chunk(text=f"c{i}") for i in range(2)]
        batches = _build_reranker_batches(chunks, batch_size=6)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_exact_batch_size(self):
        """6 chunks with batch_size=6 → 1 batch."""
        chunks = [_make_chunk(text=f"c{i}") for i in range(6)]
        batches = _build_reranker_batches(chunks, batch_size=6)
        assert len(batches) == 1
        assert len(batches[0]) == 6


class TestFormatBatchChunks:
    """Tests for _format_batch_chunks_for_prompt."""

    def test_formatting(self):
        batch = [_make_chunk(doc="A.pdf", page=5, text="hello")]
        result = _format_batch_chunks_for_prompt(batch)
        assert "[Chunk 0]" in result
        assert "A.pdf" in result
        assert "p.5" in result


class TestNormalizeBatchScores:
    """Tests for _normalize_batch_scores."""

    def test_zero_mean(self):
        scored = [
            {"_raw_score": 5, "other": "a"},
            {"_raw_score": 3, "other": "b"},
            {"_raw_score": 1, "other": "c"},
        ]
        result = _normalize_batch_scores(scored)
        # Mean = 3, so normalized: 2, 0, -2
        assert result[0]["_normalized_score"] == 2.0
        assert result[1]["_normalized_score"] == 0.0
        assert result[2]["_normalized_score"] == -2.0

    def test_descending_sort(self):
        scored = [
            {"_raw_score": 1},
            {"_raw_score": 5},
            {"_raw_score": 3},
        ]
        result = _normalize_batch_scores(scored)
        assert result[0]["_raw_score"] == 5
        assert result[-1]["_raw_score"] == 1

    def test_empty(self):
        assert _normalize_batch_scores([]) == []


class TestScoreTo100:
    """Tests for SCORE_TO_100 mapping."""

    def test_all_scores(self):
        assert SCORE_TO_100[5] == 95
        assert SCORE_TO_100[4] == 75
        assert SCORE_TO_100[3] == 50
        assert SCORE_TO_100[2] == 25
        assert SCORE_TO_100[1] == 10


class TestRerankTaskChunks:
    """Integration tests for _rerank_task_chunks with mocked LLM."""

    def _mock_batch_output(self, batch_size, scores):
        """Create mock RerankerBatchOutput."""
        from src.models.results import RerankerBatchOutput, RerankerChunkResult

        results = [
            RerankerChunkResult(
                id=i,
                score=scores[i] if i < len(scores) else 3,
                reason=f"reason {i}",
            )
            for i in range(batch_size)
        ]
        return RerankerBatchOutput(results=results)

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.settings")
    def test_hard_filter_drops_low_scores(self, mock_settings, mock_get_client):
        """Chunks with raw score < min_score should be dropped."""
        mock_settings.reranker_batch_size = 6
        mock_settings.reranker_strategy = "precision"
        mock_settings.reranker_min_score = 4

        # Mock LLM to return mixed scores
        client = MagicMock()
        mock_get_client.return_value = client
        client.generate_structured_messages.return_value = self._mock_batch_output(
            3, [5, 2, 4]
        )

        primary = [_make_chunk(text=f"p{i}") for i in range(3)]
        result = _rerank_task_chunks(
            primary, [], [], "test query", "hitl summary", "de",
        )
        # Only scores 5 and 4 should survive (score 2 dropped)
        raw_scores = [c["_raw_score"] for c in result]
        assert all(s >= 4 for s in raw_scores)
        assert len(result) == 2

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.settings")
    def test_preserves_provenance_fields(self, mock_settings, mock_get_client):
        """Provenance fields should pass through reranking."""
        mock_settings.reranker_batch_size = 6
        mock_settings.reranker_strategy = "precision"
        mock_settings.reranker_min_score = 1

        client = MagicMock()
        mock_get_client.return_value = client
        client.generate_structured_messages.return_value = self._mock_batch_output(1, [5])

        primary = [_make_chunk(
            parent_document="parent.pdf",
            parent_page=10,
            reference_original_text="§21 KrWG",
            reference_type="legal_section",
            reference_surrounding_context="context here",
        )]
        result = _rerank_task_chunks(primary, [], [], "query", "hitl", "de")
        assert len(result) == 1
        assert result[0]["parent_document"] == "parent.pdf"
        assert result[0]["reference_original_text"] == "§21 KrWG"

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.settings")
    def test_fallback_on_llm_error(self, mock_settings, mock_get_client):
        """LLM failure should produce fallback scores, not crash."""
        mock_settings.reranker_batch_size = 6
        mock_settings.reranker_strategy = "precision"
        mock_settings.reranker_min_score = 1

        client = MagicMock()
        mock_get_client.return_value = client
        client.generate_structured_messages.side_effect = RuntimeError("LLM down")

        primary = [_make_chunk(score=0.9)]
        result = _rerank_task_chunks(primary, [], [], "query", "hitl", "de")
        assert len(result) == 1
        assert "_raw_score" in result[0]
        assert result[0]["_llm_reasoning"] == "fallback"

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.settings")
    def test_llm_score_mapping(self, mock_settings, mock_get_client):
        """Raw scores should be mapped to 0-100 via SCORE_TO_100."""
        mock_settings.reranker_batch_size = 6
        mock_settings.reranker_strategy = "precision"
        mock_settings.reranker_min_score = 1

        client = MagicMock()
        mock_get_client.return_value = client
        client.generate_structured_messages.return_value = self._mock_batch_output(2, [5, 3])

        primary = [_make_chunk(text=f"p{i}") for i in range(2)]
        result = _rerank_task_chunks(primary, [], [], "query", "hitl", "de")
        scores_100 = {c["_raw_score"]: c["_llm_score"] for c in result}
        assert scores_100[5] == 95
        assert scores_100[3] == 50

    def test_empty_input(self):
        """Empty input should return empty list."""
        result = _rerank_task_chunks([], [], [], "query", "hitl", "de")
        assert result == []
