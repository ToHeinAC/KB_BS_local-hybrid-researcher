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


def filter_tiered_context_by_task(
    primary_ctx: list[dict],
    secondary_ctx: list[dict],
    tertiary_ctx: list[dict],
    task_id: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Filter tiered context lists to entries matching a specific task_id.

    Returns:
        Tuple of (primary, secondary, tertiary) filtered lists.
    """
    return (
        [e for e in primary_ctx if e.get("task_id") == task_id],
        [e for e in secondary_ctx if e.get("task_id") == task_id],
        [e for e in tertiary_ctx if e.get("task_id") == task_id],
    )


def has_task_id_entries(context_lists: list[list[dict]]) -> bool:
    """Check if any entry in any of the context lists has a task_id key.

    Used for backward compatibility: old states without task_id fall back
    to flat chunk rendering.
    """
    return any(
        "task_id" in entry
        for ctx in context_lists
        for entry in ctx
    )


_TIER_LABELS = {
    1: "Primäre Ergebnisse (Tier 1)",
    2: "Sekundäre Ergebnisse (Tier 2)",
    3: "Tertiäre Ergebnisse (Tier 3)",
}


def render_tiered_chunks(
    primary: list[dict],
    secondary: list[dict],
    tertiary: list[dict],
) -> None:
    """Render chunks grouped by tier as nested expanders.

    Tier 1 is expanded by default, others collapsed. Empty tiers are skipped.
    """
    total = len(primary) + len(secondary) + len(tertiary)
    if total == 0:
        st.caption("Keine relevanten Chunks gefunden")
        return

    st.markdown(f"**{total} Chunks gefunden:**")

    for tier_num, chunks in [(1, primary), (2, secondary), (3, tertiary)]:
        if not chunks:
            continue
        label = f"{_TIER_LABELS[tier_num]} ({len(chunks)} Chunks)"
        with st.expander(label, expanded=(tier_num == 1)):
            for i, chunk in enumerate(chunks):
                render_chunk_expander(chunk, i)
