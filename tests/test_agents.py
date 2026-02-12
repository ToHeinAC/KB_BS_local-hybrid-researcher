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

    def test_route_to_generate_todo_with_research_queries(self):
        """Test routing to generate_todo when research_queries present."""
        from src.agents.graph import route_entry_point

        state = {"research_queries": ["query1", "query2"]}
        result = route_entry_point(state)
        assert result == "generate_todo"

    def test_route_to_generate_todo_with_phase(self):
        """Test routing to generate_todo when phase explicitly set."""
        from src.agents.graph import route_entry_point

        state = {"phase": "generate_todo"}
        result = route_entry_point(state)
        assert result == "generate_todo"

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
        mock_client.generate.return_value = "PRIMARY:\nFact [doc.pdf]\nSECONDARY:\nNone"

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            result = _generate_hitl_summary(
                query="Grenzwerte",
                conversation=[{"role": "user", "content": "Frage"}],
                retrieval="[strlsch.pdf, p.5]: Grenzwert 6 mSv/a",
                knowledge_gaps=["gap1"],
                language="German",
            )

        # Verify LLM was called
        mock_client.generate.assert_called_once()
        prompt = mock_client.generate.call_args[0][0]

        # Verify citation rules are in prompt
        assert "[Source_filename]" in prompt
        assert "PRIMARY" in prompt
        assert "SECONDARY" in prompt
        assert "German" in prompt
        assert result == "PRIMARY:\nFact [doc.pdf]\nSECONDARY:\nNone"

    def test_fallback_on_llm_error(self):
        """Return fallback string when LLM raises."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_hitl_summary

        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("LLM down")

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

    def test_retrieval_truncation_at_8000(self):
        """Retrieval text is truncated at 8000 chars, not 4000."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_hitl_summary

        mock_client = MagicMock()
        mock_client.generate.return_value = "summary"

        long_retrieval = "x" * 10000

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_hitl_summary(
                query="q", conversation=[{"role": "user", "content": "c"}],
                retrieval=long_retrieval, knowledge_gaps=[], language="de",
            )

        prompt = mock_client.generate.call_args[0][0]
        # 8000 x's should be in prompt, not 4000
        assert "x" * 8000 in prompt
        assert "x" * 8001 not in prompt


class TestGenerateTodoHitlSmry:
    """Tests for hitl_smry integration in generate_todo."""

    def test_llm_fallback_passes_hitl_smry_to_prompt(self):
        """LLM fallback path includes hitl_smry in the prompt."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured.return_value = MagicMock(
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

        mock_client.generate_structured.assert_called_once()
        prompt = mock_client.generate_structured.call_args[0][0]
        assert "Grenzwert 6 mSv/a [strlsch.pdf]" in prompt
        assert "hitl_findings" in prompt

    def test_llm_fallback_uses_fallback_when_no_hitl_smry(self):
        """LLM fallback path uses 'No prior findings' when hitl_smry empty."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured.return_value = MagicMock(
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

        prompt = mock_client.generate_structured.call_args[0][0]
        assert "No prior findings" in prompt

    def test_research_queries_path_uses_hitl_smry_as_context(self):
        """research_queries path prefers hitl_smry over additional_context."""
        from src.agents.nodes import generate_todo

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

        result = generate_todo(state)
        items = result["todo_list"]
        # Task 0 is prepended original query; task at index 1 is first research_query
        first_rq_item = items[1]
        assert first_rq_item["context"] == "Citation-aware summary [doc.pdf]"

    def test_research_queries_path_falls_back_to_additional_context(self):
        """research_queries path falls back to additional_context when no hitl_smry."""
        from src.agents.nodes import generate_todo

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

        result = generate_todo(state)
        items = result["todo_list"]
        first_rq_item = items[1]
        assert first_rq_item["context"] == "Plain fallback"


class TestTaskSummaryHitlSmry:
    """Tests for hitl_smry integration in _generate_task_summary."""

    def test_task_summary_passes_hitl_smry_to_prompt(self):
        """hitl_smry value is forwarded into the TASK_SUMMARY_PROMPT."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured.return_value = MagicMock(
            summary="s", key_findings=[], gaps=[],
            relevance_assessment="ok", irrelevant_findings=[],
        )

        task = ToDoItem(id=1, task="Test task", context="ctx")
        anchor = {"original_query": "Q", "key_entities": [], "detected_language": "en"}

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="HITL established facts",
            )

        prompt = mock_client.generate_structured.call_args[0][0]
        assert "HITL established facts" in prompt

    def test_task_summary_uses_fallback_when_no_hitl_smry(self):
        """Empty hitl_smry is replaced with 'No prior findings'."""
        from unittest.mock import MagicMock, patch

        from src.agents.nodes import _generate_task_summary
        from src.models.query import ToDoItem

        mock_client = MagicMock()
        mock_client.generate_structured.return_value = MagicMock(
            summary="s", key_findings=[], gaps=[],
            relevance_assessment="ok", irrelevant_findings=[],
        )

        task = ToDoItem(id=1, task="Test task", context="ctx")
        anchor = {"original_query": "Q", "key_entities": [], "detected_language": "en"}

        with patch("src.agents.nodes.get_ollama_client", return_value=mock_client):
            _generate_task_summary(
                task=task, task_primary=[], task_secondary=[],
                task_tertiary=[], preserved_quotes=[],
                query_anchor=anchor, hitl_smry="",
            )

        prompt = mock_client.generate_structured.call_args[0][0]
        assert "No prior findings" in prompt
