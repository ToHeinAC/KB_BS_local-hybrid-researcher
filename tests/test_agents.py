"""Tests for agent components."""

import pytest

from src.agents.state import (
    AgentState,
    create_initial_state,
    get_next_task_id,
    get_pending_tasks,
    get_phase,
    is_hitl_pending,
    set_phase,
)
from src.agents.tools import detect_references, score_relevance


class TestAgentState:
    """Test agent state utilities."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("Test query")

        assert state["query"] == "Test query"
        assert state["phase"] == "hitl_init"  # Default with iterative HITL
        assert state["hitl_pending"] is False
        assert state["todo_list"] == []

    def test_get_phase(self):
        """Test get_phase utility."""
        state = create_initial_state("Test")
        assert get_phase(state) == "hitl_init"  # Default with iterative HITL

    def test_set_phase(self):
        """Test set_phase utility."""
        state = create_initial_state("Test")
        updated = set_phase(state, "execute")
        assert updated["phase"] == "execute"

    def test_is_hitl_pending(self):
        """Test HITL pending check."""
        state = create_initial_state("Test")
        assert is_hitl_pending(state) is False

        state["hitl_pending"] = True
        assert is_hitl_pending(state) is True

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        state = create_initial_state("Test")
        state["todo_list"] = [
            {"id": 1, "task": "Task 1", "completed": False},
            {"id": 2, "task": "Task 2", "completed": True},
            {"id": 3, "task": "Task 3", "completed": False},
        ]

        pending = get_pending_tasks(state)
        assert len(pending) == 2
        assert pending[0]["id"] == 1
        assert pending[1]["id"] == 3

    def test_get_next_task_id(self):
        """Test getting next task ID."""
        state = create_initial_state("Test")
        state["todo_list"] = [
            {"id": 1, "task": "Task 1", "completed": True},
            {"id": 2, "task": "Task 2", "completed": False},
        ]

        next_id = get_next_task_id(state)
        assert next_id == 2

    def test_get_next_task_id_empty(self):
        """Test getting next task ID when all completed."""
        state = create_initial_state("Test")
        state["todo_list"] = [
            {"id": 1, "task": "Task 1", "completed": True},
        ]

        next_id = get_next_task_id(state)
        assert next_id is None


class TestTools:
    """Test agent tools."""

    def test_detect_references_german_section(self):
        """Test detecting German section references."""
        text = "Siehe § 5 Abs. 2 für weitere Details."
        refs = detect_references(text)

        assert len(refs) >= 1
        section_refs = [r for r in refs if r.type == "section"]
        assert len(section_refs) >= 1

    def test_detect_references_german_document(self):
        """Test detecting German document references."""
        text = "Gemäß Dokument EU 208 sind die Anforderungen..."
        refs = detect_references(text)

        assert len(refs) >= 1
        doc_refs = [r for r in refs if r.type == "document"]
        assert len(doc_refs) >= 1

    def test_detect_references_english_section(self):
        """Test detecting English section references."""
        text = "See section 5.2 for more details."
        refs = detect_references(text)

        assert len(refs) >= 1
        section_refs = [r for r in refs if r.type == "section"]
        assert len(section_refs) >= 1

    def test_detect_references_external(self):
        """Test detecting external URL references."""
        text = "For more info visit https://example.com/docs"
        refs = detect_references(text)

        assert len(refs) >= 1
        external_refs = [r for r in refs if r.type == "external"]
        assert len(external_refs) >= 1
        assert "example.com" in external_refs[0].target

    def test_detect_references_no_duplicates(self):
        """Test that duplicate references are not returned."""
        text = "Siehe § 5 und dann nochmal § 5 für Details."
        refs = detect_references(text)

        # Should only have one reference to § 5
        section_refs = [r for r in refs if r.type == "section" and "5" in r.target]
        assert len(section_refs) == 1

    def test_score_relevance_high(self):
        """Test high relevance scoring."""
        chunk = "Die Grenzwerte für Strahlenexposition sind in der Verordnung festgelegt."
        query = "Grenzwerte Strahlenexposition"

        score = score_relevance(chunk, query)
        assert score > 0.5

    def test_score_relevance_low(self):
        """Test low relevance scoring."""
        chunk = "Das Wetter ist heute schön."
        query = "Grenzwerte Strahlenexposition"

        score = score_relevance(chunk, query)
        assert score < 0.3

    def test_score_relevance_empty_query(self):
        """Test scoring with empty query."""
        chunk = "Some text"
        query = ""

        score = score_relevance(chunk, query)
        assert score == 0.0


class TestGraph:
    """Test graph construction (without execution)."""

    def test_graph_creation(self):
        """Test that graph can be created."""
        from src.agents.graph import create_research_graph

        graph = create_research_graph()
        assert graph is not None

    def test_graph_has_entry_point(self):
        """Test that graph has correct entry point."""
        from src.agents.graph import create_research_graph

        graph = create_research_graph()
        # The graph should have nodes
        assert hasattr(graph, "nodes")


class TestRouteEntryPoint:
    """Test route_entry_point routing logic for HITL resume."""

    def test_route_to_hitl_init_on_new_session(self):
        """Test routing to hitl_init when starting new iterative HITL."""
        from src.agents.graph import route_entry_point

        state = {"hitl_active": True}
        result = route_entry_point(state)
        assert result == "hitl_init"

    def test_route_to_hitl_process_response_on_resume(self):
        """Test routing to hitl_process_response when resuming with decision."""
        from src.agents.graph import route_entry_point

        state = {
            "hitl_active": True,
            "hitl_decision": {"approved": True, "modifications": {"user_response": "test"}},
        }
        result = route_entry_point(state)
        assert result == "hitl_process_response"

    def test_route_to_assess_query_with_research_queries(self):
        """Test routing to assess_query when research_queries present."""
        from src.agents.graph import route_entry_point

        state = {"research_queries": ["query1", "query2"]}
        result = route_entry_point(state)
        assert result == "assess_query"

    def test_route_to_assess_query_with_phase(self):
        """Test routing to assess_query when phase explicitly set to generate_todo."""
        from src.agents.graph import route_entry_point

        state = {"phase": "generate_todo"}
        result = route_entry_point(state)
        assert result == "assess_query"

    def test_default_route_to_hitl_init(self):
        """Test default routing to hitl_init when no special conditions."""
        from src.agents.graph import route_entry_point

        state = {}
        result = route_entry_point(state)
        assert result == "hitl_init"

    def test_decision_without_hitl_active_routes_to_process_hitl_todo(self):
        """Test that decision without hitl_active routes to process_hitl_todo."""
        from src.agents.graph import route_entry_point

        state = {
            "hitl_active": False,
            "hitl_decision": {"approved": True},
        }
        result = route_entry_point(state)
        # With hitl_active=False, decision triggers todo processing (post-approval)
        assert result == "process_hitl_todo"


class TestGenerateHitlSummary:
    """Tests for _generate_hitl_summary helper."""

    def test_empty_conversation_and_retrieval_returns_empty(self):
        """Return empty string when no conversation or retrieval."""
        from src.agents.nodes import _generate_hitl_summary

        result = _generate_hitl_summary(
            query="test", conversation=[], retrieval="",
            knowledge_gaps=[], language="de",
        )
        assert result == ""

    def test_prompt_contains_citation_instructions(self):
        """Prompt sent to LLM includes citation and structure rules."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_hitl_summary

        mock_client = MagicMock()
        mock_client.generate_messages.return_value = "PRIMARY:\nFact [doc.pdf]\nSECONDARY:\nNone"

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _generate_hitl_summary(
                query="Grenzwerte",
                conversation=[{"role": "user", "content": "Frage"}],
                retrieval="[strlsch.pdf, p.5]: Grenzwert 6 mSv/a",
                knowledge_gaps=["gap1"],
                language="de",
            )

        # Verify LLM was called
        mock_client.generate_messages.assert_called_once()
        system_prompt = mock_client.generate_messages.call_args[0][0]
        human_prompt = mock_client.generate_messages.call_args[0][1]

        # Verify citation rules are in system prompt
        assert "[source_filename.pdf]" in system_prompt
        assert "PRIMARY" in system_prompt
        assert "FURTHER INFORMATION" in system_prompt
        # Language appears in both
        assert "German" in system_prompt or "German" in human_prompt
        assert result == "PRIMARY:\nFact [doc.pdf]\nSECONDARY:\nNone"

    def test_fallback_on_llm_error(self):
        """Return fallback string when LLM raises."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_hitl_summary

        mock_client = MagicMock()
        mock_client.generate_messages.side_effect = RuntimeError("LLM down")

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _generate_hitl_summary(
                query="test query",
                conversation=[{"role": "user", "content": "hello"}],
                retrieval="some text",
                knowledge_gaps=[],
                language="de",
            )

        assert "HITL Summary" in result
        assert "test query" in result

    def test_retrieval_truncation_at_12000(self):
        """Retrieval text is truncated at 12000 chars."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_hitl_summary

        mock_client = MagicMock()
        mock_client.generate_messages.return_value = "summary"

        long_retrieval = "x" * 15000

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_hitl_summary(
                query="q", conversation=[{"role": "user", "content": "c"}],
                retrieval=long_retrieval, knowledge_gaps=[], language="de",
            )

        human_prompt = mock_client.generate_messages.call_args[0][1]
        assert "x" * 12000 in human_prompt
        assert "x" * 12001 not in human_prompt


class TestGenerateTodoHitlSmry:
    """Tests for hitl_smry integration in generate_todo."""

    def test_llm_fallback_passes_hitl_smry_to_prompt(self):
        """LLM fallback path includes hitl_smry in the prompt."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            items=[{"id": 1, "task": "Research task", "context": "ctx"}]
        )

        state = {
            "query_analysis": {
                "original_query": "Grenzwerte",
                "key_concepts": ["Strahlung"],
                "entities": ["StrlSchV"],
                "scope": "radiation",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
            },
            "hitl_smry": "PRIMARY:\nGrenzwert 6 mSv/a [strlsch.pdf]",
            "research_queries": [],  # Force LLM fallback
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = generate_todo(state)

        mock_client.generate_structured_messages.assert_called_once()
        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "Grenzwert 6 mSv/a [strlsch.pdf]" in human_prompt
        assert "hitl_findings" in human_prompt

    def test_llm_fallback_uses_fallback_when_no_hitl_smry(self):
        """LLM fallback path uses 'No prior findings' when hitl_smry empty."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            items=[{"id": 1, "task": "Task", "context": "ctx"}]
        )

        state = {
            "query_analysis": {
                "original_query": "Test",
                "key_concepts": [],
                "entities": [],
                "scope": "",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "en",
            },
            "research_queries": [],
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            generate_todo(state)

        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "No prior findings" in human_prompt

    def test_research_queries_fallback_uses_hitl_smry_as_context(self):
        """When LLM fails, research_queries fallback prefers hitl_smry over additional_context."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("LLM error")

        state = {
            "query_analysis": {
                "original_query": "Test",
                "key_concepts": [],
                "entities": [],
                "scope": "",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
            },
            "research_queries": ["query1", "query2"],
            "hitl_smry": "Citation-aware summary [doc.pdf]",
            "additional_context": "Plain summary",
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = generate_todo(state)
        items = result["todo_list"]
        # Task 0 is prepended original query; task at index 1 is first research_query
        first_rq_item = items[1]
        assert first_rq_item["context"] == "Citation-aware summary [doc.pdf]"

    def test_research_queries_fallback_uses_additional_context(self):
        """When LLM fails, research_queries fallback uses additional_context when no hitl_smry."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("LLM error")

        state = {
            "query_analysis": {
                "original_query": "Test",
                "key_concepts": [],
                "entities": [],
                "scope": "",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
            },
            "research_queries": ["query1"],
            "additional_context": "Plain fallback",
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = generate_todo(state)
        items = result["todo_list"]
        first_rq_item = items[1]
        assert first_rq_item["context"] == "Plain fallback"


class TestTaskSummaryHitlSmry:
    """Tests for hitl_smry integration in _generate_task_summary."""

    def test_task_summary_passes_hitl_smry_to_prompt(self):
        """hitl_smry value is forwarded into the TASK_SUMMARY_PROMPT_HUMAN."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            summary="s", key_findings=[], gaps=[],
            relevance_assessment="ok", irrelevant_findings=[],
            relevance_score=75,
        )

        task = ToDoItem(id=1, task="Test task", context="ctx")
        anchor = {"original_query": "Q", "key_entities": [], "detected_language": "en"}

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="HITL established facts",
            )

        # human_prompt is the second positional arg
        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "HITL established facts" in human_prompt

    def test_task_summary_uses_fallback_when_no_hitl_smry(self):
        """Empty hitl_smry is replaced with 'No prior findings'."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            summary="s", key_findings=[], gaps=[],
            relevance_assessment="ok", irrelevant_findings=[],
            relevance_score=75,
        )

        task = ToDoItem(id=1, task="Test task", context="ctx")
        anchor = {"original_query": "Q", "key_entities": [], "detected_language": "en"}

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="",
            )

        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "No prior findings" in human_prompt

    def test_task_summary_uses_llm_relevance_score(self):
        """relevance_to_query uses LLM's relevance_score, not keyword overlap."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            summary="s", key_findings=[], gaps=[],
            relevance_assessment="ok", irrelevant_findings=[],
            relevance_score=85,
        )

        task = ToDoItem(id=1, task="Recherchiere Dosisgrenzwerte", context="ctx")
        anchor = {
            "original_query": "Grenzwerte fur Strahlenexposition",
            "key_entities": [],
            "detected_language": "de",
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="",
            )

        assert result["relevance_to_query"] == 0.85

    def test_task_summary_falls_back_to_keyword_on_llm_error(self):
        """On LLM failure, falls back to _calculate_task_relevance."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("LLM error")

        task = ToDoItem(id=1, task="Test task", context="ctx")
        anchor = {
            "original_query": "Test task query",
            "key_entities": [],
            "detected_language": "en",
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="",
            )

        # Should use keyword fallback, not crash
        assert 0.0 <= result["relevance_to_query"] <= 1.0
        assert result["summary"].startswith("Completed task:")


    def test_task_summary_system_prompt_has_no_markdown_fences(self):
        """TASK_SUMMARY_PROMPT_SYSTEM must instruct output raw JSON only."""
        from src.prompts.research import TASK_SUMMARY_PROMPT_SYSTEM

        # Prompt must instruct: no extra text outside JSON
        assert "no other text before or after" in TASK_SUMMARY_PROMPT_SYSTEM or "no code fences" in TASK_SUMMARY_PROMPT_SYSTEM
        # ranked_findings replaces the three tier variables
        assert "ranked_findings" in TASK_SUMMARY_PROMPT_SYSTEM
        assert "primary_findings" not in TASK_SUMMARY_PROMPT_SYSTEM


# =============================================================================
# Chunk Reranker Tests
# =============================================================================


class TestChunkReranker:
    """Tests for _rerank_task_chunks() helper (batch reranking)."""

    def _make_batch_output(self, n, scores):
        """Create a RerankerBatchOutput with given scores."""
        from src.models.results import RerankerBatchOutput, RerankerChunkResult
        return RerankerBatchOutput(results=[
            RerankerChunkResult(id=i, score=scores[i] if i < len(scores) else 3, reason="ok")
            for i in range(n)
        ])

    def test_reranker_returns_empty_on_empty_input(self):
        """Empty tier lists return [] without any LLM call."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _rerank_task_chunks

        mock_client = MagicMock()
        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _rerank_task_chunks(
                task_primary=[], task_secondary=[], task_tertiary=[],
                original_query="Q", hitl_smry="", language="de",
            )

        assert result == []
        mock_client.generate_structured_messages.assert_not_called()

    def test_reranker_sorted_descending_by_score(self):
        """Chunks are returned sorted best-first by _llm_score."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import SCORE_TO_100, _rerank_task_chunks

        chunk_a = {"extracted_info": "low relevance text", "document": "A.pdf", "page": 1, "relevance_score": 0.5}
        chunk_b = {"extracted_info": "high relevance text", "document": "B.pdf", "page": 2, "relevance_score": 0.9}

        # Batch has 2 chunks: chunk_a=id0 score 2, chunk_b=id1 score 5
        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = self._make_batch_output(2, [2, 5])

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client), \
             patch("src.agents.nodes.settings") as mock_settings:
            mock_settings.reranker_batch_size = 6
            mock_settings.reranker_strategy = "precision"
            mock_settings.reranker_min_score = 1
            result = _rerank_task_chunks(
                task_primary=[chunk_a, chunk_b], task_secondary=[], task_tertiary=[],
                original_query="Q", hitl_smry="", language="en",
            )

        assert len(result) == 2
        assert result[0]["document"] == "B.pdf"
        assert result[0]["_llm_score"] == SCORE_TO_100[5]
        assert result[1]["document"] == "A.pdf"
        assert result[1]["_llm_score"] == SCORE_TO_100[2]

    def test_reranker_fallback_on_llm_error(self):
        """On LLM error, fallback scores are used (not a crash)."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _rerank_task_chunks

        chunk = {"extracted_info": "text", "document": "X.pdf", "page": 1, "relevance_score": 0.75}

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("LLM down")

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client), \
             patch("src.agents.nodes.settings") as mock_settings:
            mock_settings.reranker_batch_size = 6
            mock_settings.reranker_strategy = "precision"
            mock_settings.reranker_min_score = 1
            result = _rerank_task_chunks(
                task_primary=[chunk], task_secondary=[], task_tertiary=[],
                original_query="Q", hitl_smry="", language="de",
            )

        assert len(result) == 1
        assert "_llm_score" in result[0]
        assert result[0]["_llm_reasoning"] == "fallback"

    def test_reranker_respects_max_chunks_cap(self):
        """Only max_chunks candidates are collected for scoring."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _rerank_task_chunks

        primary = [{"extracted_info": f"p{i}", "document": f"P{i}.pdf", "page": i, "relevance_score": 0.8} for i in range(3)]
        secondary = [{"extracted_info": f"s{i}", "document": f"S{i}.pdf", "page": i, "relevance_score": 0.6} for i in range(10)]

        mock_client = MagicMock()
        # Return matching batch output for 5 chunks
        mock_client.generate_structured_messages.return_value = self._make_batch_output(5, [4, 4, 4, 4, 4])

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client), \
             patch("src.agents.nodes.settings") as mock_settings:
            mock_settings.reranker_batch_size = 6
            mock_settings.reranker_strategy = "precision"
            mock_settings.reranker_min_score = 1
            result = _rerank_task_chunks(
                task_primary=primary, task_secondary=secondary, task_tertiary=[],
                original_query="Q", hitl_smry="", language="en", max_chunks=5,
            )

        assert len(result) == 5
        # With batch_size=6, 5 chunks fit in 1 batch → 1 LLM call
        assert mock_client.generate_structured_messages.call_count == 1


# =============================================================================
# Agentic Decision Tests
# =============================================================================


class TestReferenceDecisionModel:
    """Tests for the ReferenceDecision Pydantic model."""

    def test_follow_true(self):
        from src.models.research import ReferenceDecision

        d = ReferenceDecision(follow=True, reason="Directly relevant")
        assert d.follow is True
        assert d.reason == "Directly relevant"

    def test_follow_false(self):
        from src.models.research import ReferenceDecision

        d = ReferenceDecision(follow=False, reason="Tangential")
        assert d.follow is False

    def test_from_dict(self):
        from src.models.research import ReferenceDecision

        d = ReferenceDecision.model_validate({"follow": True, "reason": "test"})
        assert d.follow is True


class TestQualityRemediationDecisionModel:
    """Tests for the QualityRemediationDecision Pydantic model."""

    def test_retry_action(self):
        from src.models.research import QualityRemediationDecision

        d = QualityRemediationDecision(action="retry", focus_instructions="Fix citations")
        assert d.action == "retry"
        assert d.focus_instructions == "Fix citations"

    def test_accept_action(self):
        from src.models.research import QualityRemediationDecision

        d = QualityRemediationDecision(action="accept", focus_instructions="")
        assert d.action == "accept"

    def test_invalid_action_rejected(self):
        from src.models.research import QualityRemediationDecision

        with pytest.raises(Exception):
            QualityRemediationDecision(action="invalid", focus_instructions="")


class TestRouteAfterQuality:
    """Tests for route_after_quality with remediation loop."""

    def test_route_to_attribute_sources_default(self):
        """Default routing goes to attribute_sources."""
        from src.agents.graph import route_after_quality

        state = {"phase": "attribute_sources"}
        assert route_after_quality(state) == "attribute_sources"

    def test_route_to_synthesize_on_retry(self):
        """Routes to synthesize when phase is retry_synthesis."""
        from src.agents.graph import route_after_quality

        state = {"phase": "retry_synthesis"}
        assert route_after_quality(state) == "synthesize"

    def test_route_default_empty_state(self):
        """Empty state routes to attribute_sources."""
        from src.agents.graph import route_after_quality

        assert route_after_quality({}) == "attribute_sources"


class TestQualityRemediationIntegration:
    """Tests for quality remediation logic in quality_check node."""

    def test_remediation_triggers_retry_on_low_score(self):
        """Quality check triggers retry when LLM decides to retry."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import quality_check
        from src.models.research import QualityRemediationDecision

        mock_client = MagicMock()
        # First call: quality check returns low scores
        mock_quality = MagicMock(
            factual_accuracy=30, semantic_validity=40,
            structural_integrity=50, citation_correctness=20,
            query_relevance=60, issues_found=["Poor citations"],
        )
        # Second call: remediation decides to retry
        mock_remediation = QualityRemediationDecision(
            action="retry", focus_instructions="Improve citation format"
        )
        mock_client.generate_structured_messages.side_effect = [mock_quality, mock_remediation]

        state = {
            "research_context": {
                "search_queries": [{"query": "q", "chunks": [], "summary": "Some summary"}],
                "metadata": {"total_iterations": 1, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
            "query_analysis": {
                "original_query": "Test", "key_concepts": [], "entities": [],
                "scope": "", "assumed_context": [], "clarification_needed": False,
                "detected_language": "de",
            },
            "query_anchor": {"original_query": "Test", "detected_language": "de"},
            "synthesis_retry_count": 0,
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client), \
             patch("src.agents.nodes.settings") as mock_settings:
            mock_settings.enable_quality_checker = True
            mock_settings.quality_threshold = 375
            result = quality_check(state)

        assert result["phase"] == "retry_synthesis"
        assert result["synthesis_retry_count"] == 1
        assert result["quality_remediation_focus"] == "Improve citation format"

    def test_remediation_skipped_after_max_retries(self):
        """Quality check does not retry when retry count already at max."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import quality_check

        mock_client = MagicMock()
        mock_quality = MagicMock(
            factual_accuracy=30, semantic_validity=40,
            structural_integrity=50, citation_correctness=20,
            query_relevance=60, issues_found=["issue"],
        )
        mock_client.generate_structured_messages.return_value = mock_quality

        state = {
            "research_context": {
                "search_queries": [{"query": "q", "chunks": [], "summary": "Some summary"}],
                "metadata": {"total_iterations": 1, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
            "query_analysis": {
                "original_query": "Test", "key_concepts": [], "entities": [],
                "scope": "", "assumed_context": [], "clarification_needed": False,
                "detected_language": "de",
            },
            "query_anchor": {"original_query": "Test", "detected_language": "de"},
            "synthesis_retry_count": 1,  # Already retried once
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client), \
             patch("src.agents.nodes.settings") as mock_settings:
            mock_settings.enable_quality_checker = True
            mock_settings.quality_threshold = 375
            result = quality_check(state)

        # Should proceed to attribute_sources, not retry
        assert result["phase"] == "attribute_sources"
        # generate_structured_messages called only once (quality check, no remediation)
        assert mock_client.generate_structured_messages.call_count == 1


class TestSynthesizeRetryFocus:
    """Tests for remediation focus being appended to synthesis prompt."""

    def test_focus_appended_on_retry(self):
        """Remediation focus instructions are appended to synthesis prompt."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import synthesize

        mock_client = MagicMock()
        mock_result = MagicMock(
            summary="Improved report", key_findings=["f1"],
            query_coverage=80, remaining_gaps=[],
        )
        mock_client.generate_structured_messages_with_language.return_value = mock_result

        state = {
            "research_context": {
                "search_queries": [{"query": "q", "chunks": [], "summary": None}],
                "metadata": {"total_iterations": 1, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
            "query_analysis": {
                "original_query": "Test", "key_concepts": [], "entities": [],
                "scope": "", "assumed_context": [], "clarification_needed": False,
                "detected_language": "de",
            },
            "query_anchor": {"original_query": "Test", "detected_language": "de", "key_entities": []},
            "primary_context": [{"extracted_info": "data", "document": "d.pdf", "page": 1}],
            "secondary_context": [],
            "task_summaries": [{"task_id": 0, "summary": "s", "key_findings": [], "gaps": []}],
            "hitl_smry": "",
            "quality_remediation_focus": "Improve citation correctness",
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = synthesize(state)

        # Verify focus instructions were in the human prompt
        call_args = mock_client.generate_structured_messages_with_language.call_args
        human_prompt = call_args[0][1]
        assert "Improve citation correctness" in human_prompt
        assert "Additional focus for this re-synthesis attempt" in human_prompt
        # Verify focus was cleared
        assert result["quality_remediation_focus"] == ""

    def test_no_focus_when_not_retrying(self):
        """No extra focus appended when not retrying."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import synthesize

        mock_client = MagicMock()
        mock_result = MagicMock(
            summary="Report", key_findings=["f1"],
            query_coverage=90, remaining_gaps=[],
        )
        mock_client.generate_structured_messages_with_language.return_value = mock_result

        state = {
            "research_context": {
                "search_queries": [{"query": "q", "chunks": [], "summary": None}],
                "metadata": {"total_iterations": 1, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
            "query_analysis": {
                "original_query": "Test", "key_concepts": [], "entities": [],
                "scope": "", "assumed_context": [], "clarification_needed": False,
                "detected_language": "de",
            },
            "query_anchor": {"original_query": "Test", "detected_language": "de", "key_entities": []},
            "primary_context": [{"extracted_info": "data", "document": "d.pdf", "page": 1}],
            "secondary_context": [],
            "task_summaries": [{"task_id": 0, "summary": "s", "key_findings": [], "gaps": []}],
            "hitl_smry": "",
            "quality_remediation_focus": "",  # No focus
        }

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = synthesize(state)

        human_prompt = mock_client.generate_structured_messages_with_language.call_args[0][1]
        assert "Additional focus" not in human_prompt
        # No quality_remediation_focus key in return since it was already empty
        assert "quality_remediation_focus" not in result


class TestAgentStateNewFields:
    """Tests for new agentic decision fields in AgentState."""

    def test_initial_state_has_retry_count(self):
        """Initial state includes synthesis_retry_count = 0."""
        state = create_initial_state("Test query")
        assert state["synthesis_retry_count"] == 0

    def test_initial_state_has_remediation_focus(self):
        """Initial state includes empty quality_remediation_focus."""
        state = create_initial_state("Test query")
        assert state["quality_remediation_focus"] == ""


class TestHitlTerminationPaths:
    """Test that all termination paths sync hitl_conversation_history."""

    def _make_state(self, user_response="/end", iteration=0, coverage=0.0):
        """Helper to create minimal HITL state."""
        return {
            "hitl_state": {
                "conversation_history": [
                    {"role": "user", "content": "initial question"},
                    {"role": "assistant", "content": "follow-up"},
                    {"role": "user", "content": user_response},
                ],
                "user_query": "test query",
                "language": "de",
                "analysis": {},
            },
            "hitl_decision": {
                "approved": user_response != "/end",
                "modifications": {"user_response": user_response},
            },
            "hitl_iteration": iteration,
            "hitl_max_iterations": 5,
            "coverage_score": coverage,
        }

    def test_user_end_syncs_conversation(self):
        """The /end path includes hitl_conversation_history."""
        from src.agents.nodes import hitl_process_response

        state = self._make_state(user_response="/end")
        # Override decision to signal /end
        state["hitl_decision"] = {"approved": False}
        result = hitl_process_response(state)
        assert "hitl_conversation_history" in result
        assert result["hitl_termination_reason"] == "user_end"
        assert len(result["hitl_conversation_history"]) == 3

    def test_max_iterations_syncs_conversation(self):
        """The max_iterations path includes hitl_conversation_history."""
        from unittest.mock import patch
        from src.agents.nodes import hitl_process_response

        state = self._make_state(user_response="some answer", iteration=4)

        with patch("src.services.hitl_service.process_human_feedback", return_value=state["hitl_state"]):
            result = hitl_process_response(state)

        assert "hitl_conversation_history" in result
        assert result["hitl_termination_reason"] == "max_iterations"

    def test_convergence_syncs_conversation(self):
        """The convergence path includes hitl_conversation_history."""
        from unittest.mock import patch
        from src.agents.nodes import hitl_process_response

        state = self._make_state(user_response="some answer", coverage=0.95)

        with patch("src.services.hitl_service.process_human_feedback", return_value=state["hitl_state"]):
            result = hitl_process_response(state)

        assert "hitl_conversation_history" in result
        assert result["hitl_termination_reason"] == "convergence"

    def test_continue_syncs_conversation(self):
        """The continue (non-termination) path also includes hitl_conversation_history."""
        from unittest.mock import patch
        from src.agents.nodes import hitl_process_response

        state = self._make_state(user_response="some answer", iteration=0, coverage=0.3)

        with patch("src.services.hitl_service.process_human_feedback", return_value=state["hitl_state"]):
            result = hitl_process_response(state)

        assert "hitl_conversation_history" in result
        assert "hitl_termination_reason" not in result or result.get("hitl_termination_reason") is None


class TestRerankerTaskSummaries:
    """Tests for rerank_task_summaries() node."""

    def test_sorts_descending_by_relevance(self):
        from src.agents.nodes import rerank_task_summaries

        summaries = [
            {"task_id": 1, "task_text": "T1", "summary": "s", "relevance_to_query": 0.4},
            {"task_id": 2, "task_text": "T2", "summary": "s", "relevance_to_query": 0.9},
            {"task_id": 3, "task_text": "T3", "summary": "s", "relevance_to_query": 0.6},
        ]
        result = rerank_task_summaries({"task_summaries": summaries})
        ranked = result["task_summaries"]
        assert ranked[0]["task_id"] == 2   # 0.9 first
        assert ranked[1]["task_id"] == 3   # 0.6 second
        assert ranked[2]["task_id"] == 1   # 0.4 last

    def test_rank_field_stamped_correctly(self):
        from src.agents.nodes import rerank_task_summaries

        summaries = [
            {"task_id": 10, "task_text": "Low", "summary": "s", "relevance_to_query": 0.2},
            {"task_id": 11, "task_text": "High", "summary": "s", "relevance_to_query": 0.8},
        ]
        result = rerank_task_summaries({"task_summaries": summaries})
        ranked = result["task_summaries"]
        assert ranked[0]["rank"] == 1 and ranked[0]["task_id"] == 11
        assert ranked[1]["rank"] == 2 and ranked[1]["task_id"] == 10

    def test_tie_broken_by_task_id_ascending(self):
        from src.agents.nodes import rerank_task_summaries

        summaries = [
            {"task_id": 5, "task_text": "B", "summary": "s", "relevance_to_query": 0.7},
            {"task_id": 2, "task_text": "A", "summary": "s", "relevance_to_query": 0.7},
        ]
        result = rerank_task_summaries({"task_summaries": summaries})
        assert result["task_summaries"][0]["task_id"] == 2  # lower task_id first on tie

    def test_empty_summaries_returns_synthesize_phase(self):
        from src.agents.nodes import rerank_task_summaries

        result = rerank_task_summaries({"task_summaries": []})
        assert result["phase"] == "synthesize"

    def test_format_includes_rank_and_relevance(self):
        from src.agents.nodes import _format_task_summaries

        summaries = [
            {"task_id": 1, "task_text": "DoseLimit", "summary": "20 mSv",
             "relevance_to_query": 0.85, "rank": 1,
             "key_findings": [], "gaps": [], "preserved_quotes": []},
            {"task_id": 2, "task_text": "Monitoring", "summary": "quarterly",
             "relevance_to_query": 0.4, "rank": 2,
             "key_findings": [], "gaps": [], "preserved_quotes": []},
        ]
        text = _format_task_summaries(summaries)
        assert "[Rank: 1/2]" in text
        assert "[Relevance: 85/100]" in text
        assert "[Rank: 2/2]" in text
        assert "[Relevance: 40/100]" in text


class TestRouteAfterValidateRelevance:
    def test_routes_to_rerank(self):
        from src.agents.graph import route_after_validate_relevance

        assert route_after_validate_relevance({}) == "rerank_task_summaries"
