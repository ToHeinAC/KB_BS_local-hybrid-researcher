"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_query():
    """Sample research query."""
    return "Was sind die Grenzwerte für berufliche Strahlenexposition?"


@pytest.fixture
def sample_todo_items():
    """Sample todo items."""
    return [
        {"id": 1, "task": "Find dose limits in StrlSchV", "context": "Main query", "completed": False},
        {"id": 2, "task": "Check occupational exposure categories", "context": "Related", "completed": False},
        {"id": 3, "task": "Review monitoring requirements", "context": "Follow-up", "completed": False},
    ]


@pytest.fixture
def sample_chunk():
    """Sample vector search result chunk."""
    return {
        "doc_id": "chunk_123",
        "doc_name": "StrlSchV.pdf",
        "chunk_text": "Die effektive Dosis darf im Kalenderjahr 20 mSv nicht überschreiten.",
        "page_number": 15,
        "relevance_score": 0.92,
        "collection": "StrlSch",
        "query_used": "Grenzwerte Strahlenexposition",
    }


@pytest.fixture
def sample_query_analysis():
    """Sample query analysis result."""
    return {
        "original_query": "Was sind die Grenzwerte für berufliche Strahlenexposition?",
        "key_concepts": ["Grenzwerte", "Strahlenexposition", "beruflich"],
        "entities": ["StrlSchV"],
        "scope": "regulatory",
        "assumed_context": ["German radiation protection law"],
        "clarification_needed": False,
        "hitl_refinements": [],
        "detected_language": "de",
    }
