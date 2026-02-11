"""Tests for TaskSearchQueries model and multi-query execute_task logic."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.query import TaskSearchQueries
from src.models.results import VectorResult


class TestTaskSearchQueriesModel:
    """Test TaskSearchQueries Pydantic model."""

    def test_basic_creation(self):
        q = TaskSearchQueries(
            query_1="Grenzwerte Strahlenexposition StrlSchV",
            query_2="Dosisgrenzwerte beruflich exponierte Personen",
        )
        assert q.query_1 == "Grenzwerte Strahlenexposition StrlSchV"
        assert q.query_2 == "Dosisgrenzwerte beruflich exponierte Personen"

    def test_model_dump(self):
        q = TaskSearchQueries(query_1="q1", query_2="q2")
        d = q.model_dump()
        assert d == {"query_1": "q1", "query_2": "q2"}

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            TaskSearchQueries(query_1="q1")  # missing query_2


class TestMultiQueryDedup:
    """Test the dedup logic used in execute_task (extracted pattern)."""

    def _make_result(self, doc_name: str, page: int, text: str) -> VectorResult:
        return VectorResult(
            doc_id=f"{doc_name}_{page}",
            doc_name=doc_name,
            chunk_text=text,
            page_number=page,
            relevance_score=0.9,
            collection="test_col",
            query_used="test",
        )

    def test_dedup_removes_duplicates(self):
        """Identical chunks from different queries are deduplicated."""
        r1 = self._make_result("doc.pdf", 1, "same text here")
        r2 = self._make_result("doc.pdf", 1, "same text here")
        r3 = self._make_result("other.pdf", 2, "different text")

        all_results = [r1, r2, r3]
        seen: set[str] = set()
        deduped = []
        for r in all_results:
            key = f"{r.doc_name}:{r.page_number}:{r.chunk_text[:100]}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        assert len(deduped) == 2
        assert deduped[0].doc_name == "doc.pdf"
        assert deduped[1].doc_name == "other.pdf"

    def test_dedup_keeps_unique(self):
        """All unique chunks are preserved."""
        results = [
            self._make_result("a.pdf", 1, "text a"),
            self._make_result("b.pdf", 2, "text b"),
            self._make_result("c.pdf", 3, "text c"),
        ]
        seen: set[str] = set()
        deduped = []
        for r in results:
            key = f"{r.doc_name}:{r.page_number}:{r.chunk_text[:100]}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        assert len(deduped) == 3

    def test_dedup_same_doc_different_pages(self):
        """Same document, different pages are kept as separate."""
        r1 = self._make_result("doc.pdf", 1, "page 1 text")
        r2 = self._make_result("doc.pdf", 2, "page 2 text")

        seen: set[str] = set()
        deduped = []
        for r in [r1, r2]:
            key = f"{r.doc_name}:{r.page_number}:{r.chunk_text[:100]}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        assert len(deduped) == 2


class TestExecuteTaskMultiQuery:
    """Test execute_task generates 3 queries (1 base + 2 LLM-generated)."""

    @patch("src.agents.nodes.vector_search")
    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.filter_by_relevance", side_effect=lambda c, q, **kw: c)
    @patch("src.agents.nodes.create_chunk_with_info")
    @patch("src.agents.nodes.classify_context_tier", return_value=(1, 1.0))
    @patch("src.agents.nodes.create_tiered_context_entry", return_value={"tier": 1})
    @patch("src.agents.nodes._generate_task_summary", return_value={
        "task_id": 0, "summary": "test", "key_findings": [], "gaps": [],
        "preserved_quotes": [], "sources": [], "relevance_to_query": 0.8,
    })
    def test_three_queries_executed(
        self,
        mock_summary,
        mock_tier_entry,
        mock_classify,
        mock_chunk_info,
        mock_filter,
        mock_client,
        mock_search,
    ):
        """execute_task calls vector_search 3 times (base + 2 generated)."""
        from src.agents.nodes import execute_task

        # Mock LLM to return search queries
        client_inst = MagicMock()
        client_inst.generate_structured.return_value = TaskSearchQueries(
            query_1="focused query",
            query_2="broad query",
        )
        mock_client.return_value = client_inst

        # Mock vector_search to return empty
        mock_search.return_value = []

        # Mock chunk creation (won't be called if no results)
        mock_chunk_info.return_value = (MagicMock(extracted_info="", references=[]), [])

        state = {
            "current_task_id": 0,
            "todo_list": [{"id": 0, "task": "Find dose limits", "completed": False, "context": "", "subtasks": []}],
            "query_analysis": {
                "original_query": "dose limits",
                "key_concepts": ["dose", "limits"],
                "entities": [],
                "scope": "regulatory",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
                "hitl_refinements": [],
            },
            "research_context": {"search_queries": [], "metadata": {"total_iterations": 0, "documents_referenced": [], "visited_refs": []}},
        }

        result = execute_task(state)

        # vector_search should be called 3 times (one per query)
        assert mock_search.call_count == 3
        queries_used = [call.args[0] for call in mock_search.call_args_list]
        assert "focused query" in queries_used
        assert "broad query" in queries_used

    @patch("src.agents.nodes.vector_search")
    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.agents.nodes.filter_by_relevance", side_effect=lambda c, q, **kw: c)
    @patch("src.agents.nodes.create_chunk_with_info")
    @patch("src.agents.nodes.classify_context_tier", return_value=(1, 1.0))
    @patch("src.agents.nodes.create_tiered_context_entry", return_value={"tier": 1})
    @patch("src.agents.nodes._generate_task_summary", return_value={
        "task_id": 0, "summary": "test", "key_findings": [], "gaps": [],
        "preserved_quotes": [], "sources": [], "relevance_to_query": 0.8,
    })
    def test_fallback_on_llm_failure(
        self,
        mock_summary,
        mock_tier_entry,
        mock_classify,
        mock_chunk_info,
        mock_filter,
        mock_client,
        mock_search,
    ):
        """execute_task falls back to single base query when LLM fails."""
        from src.agents.nodes import execute_task

        client_inst = MagicMock()
        client_inst.generate_structured.side_effect = Exception("LLM error")
        mock_client.return_value = client_inst

        mock_search.return_value = []
        mock_chunk_info.return_value = (MagicMock(extracted_info="", references=[]), [])

        state = {
            "current_task_id": 0,
            "todo_list": [{"id": 0, "task": "Find dose limits", "completed": False, "context": "", "subtasks": []}],
            "query_analysis": {
                "original_query": "dose limits",
                "key_concepts": ["dose", "limits"],
                "entities": [],
                "scope": "regulatory",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
                "hitl_refinements": [],
            },
            "research_context": {"search_queries": [], "metadata": {"total_iterations": 0, "documents_referenced": [], "visited_refs": []}},
        }

        result = execute_task(state)

        # Only 1 call (base query, no generated queries)
        assert mock_search.call_count == 1
