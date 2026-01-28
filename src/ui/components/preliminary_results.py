"""Preliminary results component with nested expanders.

Displays all intermediate research results in browsable nested expanders
organized by task.
"""

import streamlit as st

from src.ui.state import get_session_state, get_todo_list


def render_preliminary_results() -> None:
    """Render results with nested expanders by task."""
    session = get_session_state()
    state = session.agent_state or {}

    if not isinstance(state, dict):
        return

    research_context = state.get("research_context", {})
    todo_list = get_todo_list()

    if not research_context:
        return

    search_queries = research_context.get("search_queries", [])
    if not search_queries:
        return

    # Only show if we have completed tasks
    completed_tasks = [t for t in todo_list if t.get("completed")]
    if not completed_tasks:
        return

    with st.expander("Vorlaeufige Ergebnisse", expanded=True):
        # Render by task
        for task in completed_tasks:
            task_id = task.get("id")
            task_desc = task.get("task", "")[:50]

            with st.expander(
                f":white_check_mark: Task {task_id}: {task_desc}...", expanded=False
            ):
                _render_task_results(task_id, search_queries)


def _render_task_results(task_id: int, search_queries: list) -> None:
    """Render search results for a task.

    Args:
        task_id: The task ID to filter results for
        search_queries: List of search query results
    """
    found_results = False

    for i, sq in enumerate(search_queries):
        if not isinstance(sq, dict):
            continue

        # Check if this query is associated with this task
        # (fall back to showing all if no task association)
        sq_task_id = sq.get("task_id")
        if sq_task_id is not None and sq_task_id != task_id:
            continue

        query = sq.get("query", f"Query {i + 1}")[:40]
        chunks = sq.get("chunks", [])
        summary = sq.get("summary")

        with st.expander(f"Suche: {query}...", expanded=False):
            found_results = True

            if summary:
                st.markdown("**Zusammenfassung:**")
                st.write(summary)

            if chunks:
                st.markdown(f"**{len(chunks)} Chunks:**")
                for j, chunk in enumerate(chunks):
                    _render_chunk(j, chunk)
            else:
                st.caption("Keine Chunks gefunden")

    if not found_results:
        st.caption("Keine Ergebnisse fuer diese Aufgabe")


def _render_chunk(index: int, chunk: dict) -> None:
    """Render single chunk with info.

    Args:
        index: Chunk index for display
        chunk: Chunk data dict
    """
    if not isinstance(chunk, dict):
        return

    document = chunk.get("document", "Unknown")
    page = chunk.get("page")
    relevance = chunk.get("relevance_score", 0)
    extracted = chunk.get("extracted_info")
    content = chunk.get("content", "")

    header = f"Chunk {index + 1}: {document}"
    if page:
        header += f" (S. {page})"

    with st.expander(header, expanded=False):
        if extracted:
            st.markdown("**Extrahierte Info:**")
            st.write(extracted)
        elif content:
            st.markdown("**Inhalt:**")
            st.write(content[:500] + "..." if len(content) > 500 else content)

        if relevance:
            st.caption(f"Relevanz: {relevance:.2f}")
