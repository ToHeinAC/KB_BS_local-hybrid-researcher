"""Tests for assess_query node and adaptive todo sizing."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.research import QueryAssessmentDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(original_query: str = "Strahlenexposition Grenzwerte") -> dict:
    """Return a minimal agent state for assess_query tests."""
    return {
        "query": original_query,
        "query_analysis": {
            "original_query": original_query,
            "key_concepts": ["Grenzwerte", "Strahlung"],
            "entities": ["StrlSchV", "StrlSchG"],
            "scope": "Strahlenschutzrecht",
            "assumed_context": [],
            "clarification_needed": False,
            "detected_language": "de",
            "hitl_refinements": [],
        },
        "hitl_smry": "Gefundene Dokumente: StrlSchG.pdf, StrlSchV.pdf",
        "knowledge_gaps": [],
        "detected_language": "de",
    }


def _todo_state(num_tasks: int, original_query: str = "dose limits") -> dict:
    """Return minimal state for generate_todo tests with a query_assessment."""
    return {
        "query_analysis": {
            "original_query": original_query,
            "key_concepts": ["dose"],
            "entities": ["StrlSchV"],
            "scope": "regulatory",
            "assumed_context": [],
            "clarification_needed": False,
            "detected_language": "de",
            "hitl_refinements": [],
        },
        "query_assessment": {"proceed": True, "num_tasks": num_tasks},
        "hitl_smry": "",
        "research_queries": [],  # LLM is primary path; no fallback queries needed
        "detected_language": "de",
    }


# ---------------------------------------------------------------------------
# Tests: adaptive todo item count
# ---------------------------------------------------------------------------


class TestAdaptiveTodoCount:
    """generate_todo uses num_tasks from query_assessment."""

    @patch("src.agents.nodes.get_ollama_client")
    def test_simple_query_generates_min_tasks(self, mock_get_client):
        """assessment num_tasks=3 → generate_todo produces 4 items (Task 0 + 3)."""
        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            items=[
                {"id": 1, "task": "Task A", "context": "ctx"},
                {"id": 2, "task": "Task B", "context": "ctx"},
                {"id": 3, "task": "Task C", "context": "ctx"},
            ]
        )
        mock_get_client.return_value = mock_client

        result = generate_todo(_todo_state(num_tasks=3))

        # Task 0 (original query) + 3 LLM-generated = 4
        assert len(result["todo_list"]) == 4
        # Verify LLM was asked for 3 items
        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "3" in human_prompt

    @patch("src.agents.nodes.get_ollama_client")
    def test_moderate_query_generates_mid_tasks(self, mock_get_client):
        """assessment num_tasks=4 → generate_todo produces 5 items (Task 0 + 4)."""
        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            items=[
                {"id": i + 1, "task": f"Task {i}", "context": "ctx"}
                for i in range(4)
            ]
        )
        mock_get_client.return_value = mock_client

        result = generate_todo(_todo_state(num_tasks=4))

        assert len(result["todo_list"]) == 5

    @patch("src.agents.nodes.get_ollama_client")
    def test_complex_query_generates_max_tasks(self, mock_get_client):
        """assessment num_tasks=6 → generate_todo produces 7 items (Task 0 + 6)."""
        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = MagicMock(
            items=[
                {"id": i + 1, "task": f"Task {i}", "context": "ctx"}
                for i in range(6)
            ]
        )
        mock_get_client.return_value = mock_client

        result = generate_todo(_todo_state(num_tasks=6))

        assert len(result["todo_list"]) == 7
        human_prompt = mock_client.generate_structured_messages.call_args[0][1]
        assert "6" in human_prompt

    @patch("src.agents.nodes.get_ollama_client")
    def test_research_queries_fallback_respects_num_tasks(self, mock_get_client):
        """When LLM fails, research_queries fallback is capped at num_tasks."""
        from src.agents.nodes import generate_todo

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("LLM error")
        mock_get_client.return_value = mock_client

        state = {
            "query_analysis": {
                "original_query": "dose limits",
                "key_concepts": ["dose"],
                "entities": ["StrlSchV"],
                "scope": "regulatory",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
                "hitl_refinements": [],
            },
            "query_assessment": {"proceed": True, "num_tasks": 3},
            "research_queries": ["q1", "q2", "q3", "q4", "q5"],  # 5 queries
            "hitl_smry": "",
            "additional_context": "",
            "detected_language": "de",
        }

        result = generate_todo(state)

        # Task 0 + 3 capped from fallback = 4 items
        assert len(result["todo_list"]) == 4
        task_texts = [item["task"] for item in result["todo_list"]]
        assert "q4" not in task_texts  # truncated by num_tasks cap


# ---------------------------------------------------------------------------
# Tests: assess_query decider node
# ---------------------------------------------------------------------------


class TestAssessQueryDecider:
    """assess_query routes correctly based on LLM decision."""

    @patch("src.agents.nodes.get_ollama_client")
    def test_appropriate_query_proceeds(self, mock_get_client):
        """Radiation protection query with coherent HITL → proceed=True, no final_report."""
        from src.agents.nodes import assess_query

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = QueryAssessmentDecision(
            proceed=True,
            num_tasks=4,
            reason=None,
            explanation="Query is within KB scope.",
        )
        mock_get_client.return_value = mock_client

        result = assess_query(_base_state())

        assert result["query_assessment"]["proceed"] is True
        assert result["query_assessment"]["num_tasks"] == 4
        assert "final_report" not in result or not result.get("final_report")

    @patch("src.agents.nodes.get_ollama_client")
    def test_no_clear_conversation_steering_rejected(self, mock_get_client):
        """Contradictory HITL answers → proceed=False, sorry message in final_report."""
        from src.agents.nodes import assess_query

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = QueryAssessmentDecision(
            proceed=False,
            num_tasks=5,
            reason="no_clear_conversation_steering",
            explanation="Die HITL-Konversation zeigt widersprüchliche Antworten.",
        )
        mock_get_client.return_value = mock_client

        state = _base_state()
        state["hitl_smry"] = "Konversation war widersprüchlich und unklar."
        result = assess_query(state)

        assert result["query_assessment"]["proceed"] is False
        assert result["query_assessment"]["reason"] == "no_clear_conversation_steering"
        assert "final_report" in result
        assert "Sorry" in result["final_report"]["answer"]
        assert "no_clear_conversation_steering" in result["final_report"]["answer"]

    @patch("src.agents.nodes.get_ollama_client")
    def test_no_live_data_rejected(self, mock_get_client):
        """Real-time data query → proceed=False, reason=no_live_data."""
        from src.agents.nodes import assess_query

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = QueryAssessmentDecision(
            proceed=False,
            num_tasks=5,
            reason="no_live_data",
            explanation="Wettervorhersage erfordert Live-Daten.",
        )
        mock_get_client.return_value = mock_client

        result = assess_query(_base_state("Wie ist das Wetter in Berlin morgen?"))

        assert result["query_assessment"]["proceed"] is False
        assert result["query_assessment"]["reason"] == "no_live_data"
        assert "final_report" in result
        assert "no_live_data" in result["final_report"]["answer"]
        assert result["phase"] == "complete"

    @patch("src.agents.nodes.get_ollama_client")
    def test_out_of_context_rejected(self, mock_get_client):
        """Cooking query against radiation KB → proceed=False, reason=out_of_context."""
        from src.agents.nodes import assess_query

        mock_client = MagicMock()
        mock_client.generate_structured_messages.return_value = QueryAssessmentDecision(
            proceed=False,
            num_tasks=5,
            reason="out_of_context",
            explanation="Die KB enthält Strahlenschutz-Dokumente, keine Rezepte.",
        )
        mock_get_client.return_value = mock_client

        result = assess_query(_base_state("Wie backe ich einen Kuchen?"))

        assert result["query_assessment"]["proceed"] is False
        assert result["query_assessment"]["reason"] == "out_of_context"
        assert "final_report" in result
        assert "out_of_context" in result["final_report"]["answer"]

    @patch("src.agents.nodes.get_ollama_client")
    def test_llm_error_defaults_to_proceed(self, mock_get_client):
        """LLM failure → safe default proceed=True, num_tasks=5."""
        from src.agents.nodes import assess_query

        mock_client = MagicMock()
        mock_client.generate_structured_messages.side_effect = Exception("connection error")
        mock_get_client.return_value = mock_client

        result = assess_query(_base_state())

        assert result["query_assessment"]["proceed"] is True
        assert result["query_assessment"]["num_tasks"] == 5
        assert "final_report" not in result or not result.get("final_report")


# ---------------------------------------------------------------------------
# Tests: route_after_assess_query
# ---------------------------------------------------------------------------


class TestRouteAfterAssessQuery:
    """route_after_assess_query returns correct next node."""

    def test_approved_routes_to_generate_todo(self):
        from src.agents.graph import route_after_assess_query

        state = {"query_assessment": {"proceed": True, "num_tasks": 4}}
        assert route_after_assess_query(state) == "generate_todo"

    def test_rejected_routes_to_end(self):
        from src.agents.graph import route_after_assess_query

        state = {"query_assessment": {"proceed": False, "reason": "no_live_data"}}
        assert route_after_assess_query(state) == "__end__"

    def test_missing_assessment_defaults_to_generate_todo(self):
        from src.agents.graph import route_after_assess_query

        state = {}
        assert route_after_assess_query(state) == "generate_todo"
