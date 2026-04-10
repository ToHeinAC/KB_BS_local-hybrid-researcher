"""Tavily web search client for supplementary internet research."""

import logging

import requests

from src.config import settings
from src.models.results import WebResult

logger = logging.getLogger(__name__)


def tavily_search(query: str, max_results: int | None = None) -> list[dict]:
    """Search the web using the Tavily API.

    Args:
        query: Search query string
        max_results: Max results to return (defaults to settings.web_results_per_query)

    Returns:
        List of raw result dicts from Tavily API, or empty list on failure.
    """
    api_key = settings.tavily_api_key
    if not api_key:
        logger.warning("Tavily API key not configured — skipping web search")
        return []

    if max_results is None:
        max_results = settings.web_results_per_query

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": max_results,
    }

    try:
        resp = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        logger.info("Tavily search returned %d results for: %s", len(results), query[:80])
        return results
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return []


def format_tavily_results(results: list[dict], query: str) -> list[WebResult]:
    """Convert raw Tavily API results to WebResult model instances.

    Args:
        results: Raw result dicts from tavily_search()
        query: The search query used

    Returns:
        List of WebResult instances
    """
    web_results = []
    for r in results:
        web_results.append(
            WebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", r.get("snippet", "")),
                content=r.get("content"),
                query_used=query,
                source="web",
            )
        )
    return web_results


def format_results_for_prompt(web_results: list[WebResult]) -> str:
    """Format WebResult list into a text block for LLM prompts.

    Args:
        web_results: List of WebResult instances

    Returns:
        Formatted text with numbered results
    """
    parts = []
    for i, r in enumerate(web_results, 1):
        parts.append(
            f"Result {i}:\n"
            f"Title: {r.title}\n"
            f"URL: {r.url}\n"
            f"Content: {r.snippet or ''}\n"
            f"---"
        )
    return "\n".join(parts) if parts else "No web search results found."
