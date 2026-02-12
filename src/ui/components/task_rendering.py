"""Shared rendering helpers for task summaries and chunk expanders.

Used by both live research view (app.py) and persistent results view (results_view.py).
"""

import streamlit as st


def _get(obj, key, default=""):
    """Get attribute from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def render_task_summary_markdown(task_summary) -> None:
    """Render a task summary as formatted markdown.

    Args:
        task_summary: Dict or object with summary, key_findings, gaps,
                      relevance_to_query, relevance_assessment fields.
    """
    summary_text = _get(task_summary, "summary")
    if summary_text:
        st.markdown(f"**Zusammenfassung:** {summary_text}")

    findings = _get(task_summary, "key_findings", [])
    if findings:
        st.markdown("**Ergebnisse:**")
        for f in findings:
            st.markdown(f"- {f}")

    gaps = _get(task_summary, "gaps", [])
    if gaps:
        st.markdown("**Lücken:**")
        for g in gaps:
            st.markdown(f"- {g}")

    relevance_num = _get(task_summary, "relevance_to_query", None)
    if relevance_num is not None and isinstance(relevance_num, (int, float)):
        st.markdown(f"**Relevanz:** {relevance_num:.0%}")

    relevance_text = _get(task_summary, "relevance_assessment", "")
    if relevance_text:
        st.markdown(f"**Relevanzeinschätzung:** {relevance_text}")


def render_chunk_expander(chunk, index: int) -> None:
    """Render a single chunk as its own expander with full text.

    Args:
        chunk: Dict or object with document, page, relevance_score,
               extracted_info, and chunk fields.
        index: Zero-based chunk index.
    """
    doc = _get(chunk, "document", "Unbekannt")
    page = _get(chunk, "page", "?")
    score = _get(chunk, "relevance_score", 0) or 0
    extracted = _get(chunk, "extracted_info", "")
    original = _get(chunk, "chunk", "")

    header = f"Chunk {index + 1}: {doc}"
    if page and page != "?":
        header += f" (S. {page})"
    header += f" | Relevanz: {score:.2f}"

    with st.expander(header, expanded=False):
        if extracted:
            st.markdown(f"**Extraktion:**\n\n{extracted}")
        if extracted and original:
            st.divider()
        if original:
            st.markdown(f"**Originaltext:**\n\n{original}")
        if not extracted and not original:
            st.caption("Keine Daten verfügbar")
