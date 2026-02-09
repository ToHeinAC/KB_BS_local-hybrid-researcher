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
