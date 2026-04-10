"""Tests for web search integration (Tavily API)."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.results import FinalReport, WebResult, WebSearchSummaryOutput


# --- Model Tests ---


class TestWebSearchSummaryOutput:
    """Test WebSearchSummaryOutput model."""

    def test_basic_creation(self):
        result = WebSearchSummaryOutput(
            web_summary="Some findings [Title](https://example.com)",
            contradictions=["KB says X, web says Y"],
        )
        assert "Some findings" in result.web_summary
        assert len(result.contradictions) == 1

    def test_empty_contradictions(self):
        result = WebSearchSummaryOutput(web_summary="No issues found.")
        assert result.contradictions == []

    def test_final_report_web_fields(self):
        report = FinalReport(
            query="test",
            answer="test answer",
            quality_score=0,
            todo_items_completed=0,
            research_iterations=0,
        )
        assert report.web_search_section == ""
        assert report.web_sources == []

    def test_final_report_with_web_sources(self):
        ws = WebResult(
            title="Example", url="https://example.com",
            snippet="content", query_used="test",
        )
        report = FinalReport(
            query="test",
            answer="test answer",
            quality_score=0,
            todo_items_completed=0,
            research_iterations=0,
            web_search_section="### Web\nsome results",
            web_sources=[ws],
        )
        assert len(report.web_sources) == 1
        assert report.web_search_section.startswith("### Web")


# --- Tavily Client Tests ---


class TestTavilyClient:
    """Test Tavily client functions."""

    @patch("src.services.tavily_client.requests.post")
    @patch("src.services.tavily_client.settings")
    def test_tavily_search_success(self, mock_settings, mock_post):
        mock_settings.tavily_api_key = "test-key"
        mock_settings.web_results_per_query = 3
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://a.com", "content": "Info about A"},
                {"title": "Result 2", "url": "https://b.com", "content": "Info about B"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from src.services.tavily_client import tavily_search
        results = tavily_search("test query")
        assert len(results) == 2
        assert results[0]["title"] == "Result 1"

    @patch("src.services.tavily_client.settings")
    def test_tavily_search_no_api_key(self, mock_settings):
        mock_settings.tavily_api_key = ""

        from src.services.tavily_client import tavily_search
        results = tavily_search("test query")
        assert results == []

    @patch("src.services.tavily_client.requests.post")
    @patch("src.services.tavily_client.settings")
    def test_tavily_search_http_error(self, mock_settings, mock_post):
        mock_settings.tavily_api_key = "test-key"
        mock_settings.web_results_per_query = 3
        mock_post.side_effect = Exception("Connection error")

        from src.services.tavily_client import tavily_search
        results = tavily_search("test query")
        assert results == []

    def test_format_tavily_results(self):
        from src.services.tavily_client import format_tavily_results
        raw = [
            {"title": "T1", "url": "https://t1.com", "content": "C1"},
            {"title": "T2", "url": "https://t2.com", "content": "C2"},
        ]
        results = format_tavily_results(raw, "test query")
        assert len(results) == 2
        assert isinstance(results[0], WebResult)
        assert results[0].title == "T1"
        assert results[0].query_used == "test query"

    def test_format_results_for_prompt(self):
        from src.services.tavily_client import format_results_for_prompt
        web_results = [
            WebResult(title="T1", url="https://t1.com", snippet="C1", query_used="q"),
        ]
        text = format_results_for_prompt(web_results)
        assert "T1" in text
        assert "https://t1.com" in text
        assert "C1" in text

    def test_format_results_for_prompt_empty(self):
        from src.services.tavily_client import format_results_for_prompt
        text = format_results_for_prompt([])
        assert "No web search results" in text


# --- Graph Routing Tests ---


class TestWebSearchRouting:
    """Test graph routing for web search."""

    def test_route_after_rerank_enabled(self):
        from src.agents.graph import route_after_rerank
        state = {"enable_web_search": True}
        assert route_after_rerank(state) == "web_search"

    def test_route_after_rerank_disabled(self):
        from src.agents.graph import route_after_rerank
        state = {"enable_web_search": False}
        assert route_after_rerank(state) == "synthesize"

    def test_route_after_rerank_default(self):
        from src.agents.graph import route_after_rerank
        state = {}
        assert route_after_rerank(state) == "synthesize"

    def test_route_after_web_search(self):
        from src.agents.graph import route_after_web_search
        state = {}
        assert route_after_web_search(state) == "synthesize"


# --- Web Search Node Tests ---


class TestWebSearchNode:
    """Test web_search graph node."""

    def test_web_search_disabled(self):
        from src.agents.nodes import web_search
        state = {"enable_web_search": False}
        result = web_search(state)
        assert result == {}

    def test_web_search_disabled_by_default(self):
        from src.agents.nodes import web_search
        state = {}
        result = web_search(state)
        assert result == {}

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.services.tavily_client.requests.post")
    @patch("src.services.tavily_client.settings")
    def test_web_search_full_flow(self, mock_settings, mock_post, mock_get_client):
        mock_settings.tavily_api_key = "test-key"
        mock_settings.web_results_per_query = 2

        # Mock Tavily API response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"title": "Dose Limits", "url": "https://dose.com", "content": "20 mSv"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # Mock OllamaClient
        mock_client = MagicMock()
        mock_client.generate_messages.return_value = "Strahlenschutz Grenzwerte aktuell"
        mock_client.generate_structured_messages.return_value = WebSearchSummaryOutput(
            web_summary="- 20 mSv limit [Dose Limits](https://dose.com)",
            contradictions=[],
        )
        mock_get_client.return_value = mock_client

        from src.agents.nodes import web_search
        state = {
            "enable_web_search": True,
            "query_anchor": {
                "original_query": "Strahlenschutz Grenzwerte",
                "detected_language": "de",
            },
            "task_summaries": [
                {"key_findings": ["20 mSv/year"], "gaps": ["recent changes"]},
            ],
            "research_context": {
                "search_queries": [],
                "metadata": {"total_iterations": 0, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
        }
        result = web_search(state)
        assert "web_search_results" in result
        assert len(result["web_search_results"]) == 1
        assert "web_search_summary" in result
        assert "20 mSv" in result["web_search_summary"]

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.services.tavily_client.requests.post")
    @patch("src.services.tavily_client.settings")
    def test_web_search_with_contradictions(self, mock_settings, mock_post, mock_get_client):
        mock_settings.tavily_api_key = "test-key"
        mock_settings.web_results_per_query = 2

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"title": "New Info", "url": "https://n.com", "content": "Changed"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.generate_messages.return_value = "search term"
        mock_client.generate_structured_messages.return_value = WebSearchSummaryOutput(
            web_summary="New info found",
            contradictions=["KB says 20 mSv, web says 10 mSv"],
        )
        mock_get_client.return_value = mock_client

        from src.agents.nodes import web_search
        state = {
            "enable_web_search": True,
            "query_anchor": {"original_query": "test", "detected_language": "de"},
            "task_summaries": [],
            "research_context": {
                "search_queries": [],
                "metadata": {"total_iterations": 0, "documents_referenced": [],
                             "external_sources_used": False, "visited_refs": []},
            },
        }
        result = web_search(state)
        assert "Widersprüche" in result["web_search_summary"]
        assert "KB says 20 mSv" in result["web_search_summary"]

    @patch("src.agents.nodes.get_ollama_client")
    @patch("src.services.tavily_client.requests.post")
    @patch("src.services.tavily_client.settings")
    def test_web_search_no_results(self, mock_settings, mock_post, mock_get_client):
        mock_settings.tavily_api_key = "test-key"
        mock_settings.web_results_per_query = 2

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.generate_messages.return_value = "search term"
        mock_get_client.return_value = mock_client

        from src.agents.nodes import web_search
        state = {
            "enable_web_search": True,
            "query_anchor": {"original_query": "test", "detected_language": "en"},
            "task_summaries": [],
        }
        result = web_search(state)
        assert result["web_search_results"] == []
        assert result["web_search_summary"] == ""


# --- Attribute Sources Integration ---


class TestAttributeSourcesWebSearch:
    """Test that attribute_sources correctly appends web search section."""

    def _make_base_state(self):
        return {
            "query_analysis": {
                "original_query": "test query",
                "key_concepts": [],
                "entities": [],
                "scope": "",
                "assumed_context": [],
                "clarification_needed": False,
                "detected_language": "de",
                "hitl_refinements": [],
            },
            "research_context": {
                "search_queries": [
                    {
                        "query": "test",
                        "chunks": [],
                        "summary": "KB report content here.",
                    }
                ],
                "metadata": {
                    "total_iterations": 1,
                    "documents_referenced": [],
                    "external_sources_used": False,
                    "visited_refs": [],
                },
            },
            "todo_list": [],
            "quality_assessment": None,
            "query_anchor": {"detected_language": "de"},
        }

    def test_attribute_sources_without_web(self):
        from src.agents.nodes import attribute_sources
        state = self._make_base_state()
        state["web_search_summary"] = ""
        state["web_search_results"] = []
        result = attribute_sources(state)
        report = result["final_report"]
        assert report["web_search_section"] == ""
        assert report["web_sources"] == []
        assert "Ergänzende Webrecherche" not in report["answer"]

    def test_attribute_sources_with_web(self):
        from src.agents.nodes import attribute_sources
        state = self._make_base_state()
        state["web_search_summary"] = "- Found info [Title](https://example.com)"
        state["web_search_results"] = [
            {"title": "Title", "url": "https://example.com",
             "snippet": "info", "content": None, "query_used": "q", "source": "web"},
        ]
        result = attribute_sources(state)
        report = result["final_report"]
        assert "Ergänzende Webrecherche" in report["answer"]
        assert "Ergänzende Webrecherche" in report["web_search_section"]
        assert len(report["web_sources"]) == 1
        assert report["metadata"]["web_search_used"] is True

    def test_attribute_sources_english_web(self):
        from src.agents.nodes import attribute_sources
        state = self._make_base_state()
        state["query_anchor"] = {"detected_language": "en"}
        state["web_search_summary"] = "- info [T](https://t.com)"
        state["web_search_results"] = []
        result = attribute_sources(state)
        report = result["final_report"]
        assert "Supplementary Web Research" in report["answer"]


# --- State Tests ---


class TestWebSearchState:
    """Test AgentState web search fields."""

    def test_create_initial_state_defaults(self):
        from src.agents.state import create_initial_state
        state = create_initial_state("test query")
        assert state["enable_web_search"] is False
        assert state["web_search_results"] == []
        assert state["web_search_summary"] == ""
