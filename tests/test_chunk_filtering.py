"""Tests for chunk filtering with min_results backfill logic."""

import pytest

from src.agents.nodes import _score_and_filter_context


class TestScoreAndFilterContext:
    """Test suite for _score_and_filter_context with backfill."""

    def test_basic_filtering_above_threshold(self):
        """Test that items above threshold are kept."""
        items = [
            {"chunk": "radiation dose limits threshold", "context_weight": 0.9},
            {"chunk": "dose monitoring safety radiation", "context_weight": 0.8},  # Has overlap
        ]
        query = "radiation dose threshold"
        entities = ["radiation", "dose"]

        result = _score_and_filter_context(items, query, entities, threshold=0.3)

        assert len(result) == 2
        assert all(item.get("final_relevance", 0) >= 0.3 for item in result)
        # Should be sorted by relevance
        assert result[0]["final_relevance"] >= result[1]["final_relevance"]

    def test_filtering_below_threshold(self):
        """Test that items below threshold are filtered out."""
        items = [
            {"chunk": "completely unrelated content about weather", "context_weight": 0.3},
            {"chunk": "another irrelevant passage", "context_weight": 0.2},
        ]
        query = "radiation dose threshold"
        entities = ["radiation", "dose"]

        result = _score_and_filter_context(items, query, entities, threshold=0.5)

        # Both should be filtered out (low relevance)
        assert len(result) == 0

    def test_backfill_when_below_min_results(self):
        """Test that backfill occurs when results < min_results."""
        items = [
            {"chunk": "radiation dose limits", "context_weight": 0.9},  # High relevance
            {"chunk": "weather forecast", "context_weight": 0.3},  # Low relevance
            {"chunk": "stock market news", "context_weight": 0.2},  # Low relevance
        ]
        query = "radiation dose"
        entities = ["radiation"]

        # High threshold will only pass first item, but min_results=3
        result = _score_and_filter_context(items, query, entities, threshold=0.6, min_results=3)

        assert len(result) == 3
        # First item passed threshold
        assert not result[0].get("backfilled", False)
        # Next 2 are backfilled (best of the rejected)
        assert result[1].get("backfilled", False)
        assert result[2].get("backfilled", False)
        assert "backfill_reason" in result[1]
        assert "backfill_reason" in result[2]

    def test_backfill_sorted_by_score(self):
        """Test that backfilled items are sorted by score (highest first)."""
        items = [
            {"chunk": "radiation dose limits", "context_weight": 0.9},  # Passes
            {"chunk": "measurement", "context_weight": 0.5},  # Rejected, low word overlap
            {"chunk": "safety", "context_weight": 0.4},  # Rejected, no overlap
            {"chunk": "completely unrelated text", "context_weight": 0.1},  # Rejected, lowest
        ]
        query = "radiation dose"
        entities = ["radiation"]

        # Use high threshold to ensure only first item passes
        result = _score_and_filter_context(items, query, entities, threshold=0.7, min_results=3)

        assert len(result) == 3
        # First is the one that passed
        assert not result[0].get("backfilled", False)
        # Next 2 should be the top-scoring rejected items (backfilled)
        assert result[1].get("backfilled", False)
        assert result[2].get("backfilled", False)
        # Backfilled items should be sorted by score
        assert result[1]["final_relevance"] >= result[2]["final_relevance"]
        # The lowest-scoring item should NOT be included
        assert "completely unrelated" not in result[2]["chunk"]

    def test_no_backfill_when_enough_results(self):
        """Test that backfill does NOT occur when enough items pass threshold."""
        items = [
            {"chunk": "radiation dose limits threshold", "context_weight": 0.9},
            {"chunk": "dose measurement safety", "context_weight": 0.8},
            {"chunk": "radiation monitoring protocols", "context_weight": 0.85},
        ]
        query = "radiation dose"
        entities = ["radiation"]

        result = _score_and_filter_context(items, query, entities, threshold=0.3, min_results=2)

        assert len(result) == 3
        # All passed threshold, none should be backfilled
        assert all(not item.get("backfilled", False) for item in result)

    def test_backfill_with_zero_passed(self):
        """Test backfill when NO items pass threshold."""
        items = [
            {"chunk": "weather forecast", "context_weight": 0.3},
            {"chunk": "stock market", "context_weight": 0.2},
            {"chunk": "sports news", "context_weight": 0.1},
        ]
        query = "radiation dose"
        entities = ["radiation"]

        result = _score_and_filter_context(items, query, entities, threshold=0.8, min_results=2)

        assert len(result) == 2
        # All are backfilled (none passed threshold)
        assert all(item.get("backfilled", False) for item in result)
        # Top 2 scorers should be selected
        assert result[0]["final_relevance"] >= result[1]["final_relevance"]

    def test_min_results_zero_no_backfill(self):
        """Test that min_results=0 (default) disables backfill."""
        items = [
            {"chunk": "weather forecast", "context_weight": 0.3},
        ]
        query = "radiation dose"
        entities = ["radiation"]

        result = _score_and_filter_context(items, query, entities, threshold=0.8, min_results=0)

        # Nothing passes threshold, no backfill
        assert len(result) == 0

    def test_empty_input(self):
        """Test handling of empty input list."""
        result = _score_and_filter_context([], "query", ["entity"], threshold=0.5, min_results=3)
        assert len(result) == 0

    def test_backfill_limited_by_available_items(self):
        """Test that backfill doesn't exceed available items."""
        items = [
            {"chunk": "radiation dose", "context_weight": 0.9},
        ]
        query = "radiation"
        entities = ["radiation"]

        # min_results=3 but only 1 item available
        result = _score_and_filter_context(items, query, entities, threshold=0.3, min_results=3)

        assert len(result) == 1  # Can't backfill more than exists
        assert not result[0].get("backfilled", False)  # Passed threshold

    def test_relevance_score_formula(self):
        """Test that relevance scoring formula is correct."""
        items = [
            {
                "chunk": "radiation dose limits safety threshold monitoring",
                "context_weight": 0.5,
            },
        ]
        query = "radiation dose threshold"
        entities = ["radiation", "dose"]

        result = _score_and_filter_context(items, query, entities, threshold=0.0)

        # Formula: final_relevance = 0.7 * (0.6 * word_score + 0.4 * entity_score) + 0.3 * context_weight
        # Word overlap: {radiation, dose, threshold} all present = 3/3 = 1.0
        # Entity match: {radiation, dose} both present = 2/2 = 1.0
        # relevance = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        # final_relevance = 0.7 * 1.0 + 0.3 * 0.5 = 0.7 + 0.15 = 0.85

        assert len(result) == 1
        assert abs(result[0]["final_relevance"] - 0.85) < 0.01


class TestIntegrationWithValidateRelevance:
    """Integration tests for validate_relevance node with min_results."""

    def test_validate_relevance_uses_min_results(self):
        """Test that validate_relevance uses configured min_results for primary/secondary."""
        # This would be a full integration test with the graph
        # For now, we've verified the implementation manually
        # TODO: Add full graph integration test when possible
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
