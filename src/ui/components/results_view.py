"""Results view component."""

import json

import streamlit as st

from src.models.results import FinalReport
from src.ui.components.task_rendering import (
    filter_tiered_context_by_task,
    has_task_id_entries,
    render_chunk_expander,
    render_task_summary_markdown,
    render_tiered_chunks,
)
from src.ui.state import get_session_state


def render_results_view() -> None:
    """Render the final results view."""
    session = get_session_state()

    if not session.final_report:
        return

    report = FinalReport.model_validate(session.final_report)

    st.subheader("Research Results")

    # Quality score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quality Score", f"{report.quality_score}/500")
    with col2:
        st.metric("Tasks Completed", report.todo_items_completed)
    with col3:
        st.metric("Iterations", report.research_iterations)

    st.divider()

    _render_hitl_expander(session)
    _render_task_expanders(session)
    st.divider()

    # Main answer
    st.markdown("### Answer")
    st.markdown(report.answer)

    st.divider()

    # Key findings
    if report.findings:
        st.markdown("### Key Findings")
        for i, finding in enumerate(report.findings, 1):
            with st.expander(f"Finding {i}: {finding.claim[:100]}..."):
                st.markdown(f"**Claim:** {finding.claim}")
                st.markdown(f"**Evidence:** {finding.evidence}")
                st.markdown(f"**Confidence:** {finding.confidence}")

                if finding.sources:
                    st.markdown("**Sources:**")
                    for source in finding.sources:
                        st.markdown(f"- {source.doc_name} (p. {source.page_number or 'N/A'})")

    st.divider()

    # Sources
    if report.sources:
        st.markdown("### Sources")
        _render_sources(report)

    st.divider()

    # Quality breakdown
    if report.quality_breakdown:
        st.markdown("### Quality Breakdown")
        cols = st.columns(5)
        for i, (dimension, score) in enumerate(report.quality_breakdown.items()):
            with cols[i % 5]:
                st.metric(dimension.replace("_", " ").title(), f"{score}/100")

    # Export options
    st.divider()
    st.markdown("### Export")

    col1, col2 = st.columns(2)

    with col1:
        # JSON export
        json_str = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="research_report.json",
            mime="application/json",
        )

    with col2:
        # Markdown export
        md_content = _generate_markdown(report)
        st.download_button(
            label="Download Markdown",
            data=md_content,
            file_name="research_report.md",
            mime="text/markdown",
        )


def _render_hitl_expander(session) -> None:
    """Render HITL conversation, summary, and research queries."""
    conversation = session.hitl_conversation_history
    hitl_smry = session.agent_state.get("hitl_smry", "")
    hitl_result = session.hitl_result
    research_queries = (hitl_result or {}).get("research_queries", [])

    if not conversation and not hitl_smry and not research_queries:
        return

    with st.expander("HITL - Klärungsgespräch", expanded=True):
        if conversation:
            st.markdown("#### Gesprächsverlauf")
            for msg in conversation:
                with st.chat_message(msg.get("role", "assistant")):
                    st.markdown(msg.get("content", ""))

        if hitl_smry:
            st.markdown("#### HITL Zusammenfassung")
            st.markdown(hitl_smry)

        if research_queries:
            st.markdown("#### Recherche-Abfragen")
            for i, q in enumerate(research_queries, 1):
                st.markdown(f"{i}. {q}")


def _render_task_expanders(session) -> None:
    """Render per-task expanders with summaries and retrieved chunks."""
    todo_list = session.agent_state.get("todo_list", [])
    if not todo_list:
        return

    task_summaries = session.agent_state.get("task_summaries", [])
    summary_by_id = {ts.get("task_id"): ts for ts in task_summaries}

    # Tiered context lists for per-task grouping
    primary_ctx = session.agent_state.get("primary_context", [])
    secondary_ctx = session.agent_state.get("secondary_context", [])
    tertiary_ctx = session.agent_state.get("tertiary_context", [])
    use_tiered = has_task_id_entries([primary_ctx, secondary_ctx, tertiary_ctx])

    research_context = session.agent_state.get("research_context", {})
    search_queries = []
    if isinstance(research_context, dict):
        search_queries = research_context.get("search_queries", [])

    for idx, task in enumerate(todo_list):
        task_id = task.get("id", idx)
        task_text = task.get("text", task.get("task", f"Aufgabe {task_id}"))
        short_text = task_text[:60] + ("..." if len(task_text) > 60 else "")

        with st.expander(f"Aufgabe {idx + 1}: {short_text}", expanded=True):
            ts = summary_by_id.get(task_id)
            if ts:
                render_task_summary_markdown(ts)
            else:
                st.caption("Keine Zusammenfassung verfügbar")

            # Tiered chunk rendering (new) or flat fallback
            if use_tiered:
                t_primary, t_secondary, t_tertiary = filter_tiered_context_by_task(
                    primary_ctx, secondary_ctx, tertiary_ctx, task_id,
                )
                render_tiered_chunks(t_primary, t_secondary, t_tertiary)
            elif idx < len(search_queries):
                sq = search_queries[idx]
                chunks = []
                if isinstance(sq, dict):
                    chunks = sq.get("chunks", [])
                elif hasattr(sq, "chunks"):
                    chunks = sq.chunks if isinstance(sq.chunks, list) else []

                if chunks:
                    st.markdown(f"**{len(chunks)} Chunks gefunden:**")
                    for ci, chunk in enumerate(chunks):
                        render_chunk_expander(chunk, ci)


def _render_sources(report: FinalReport) -> None:
    """Render source list with links."""
    # Group by document
    docs = {}
    for linked_source in report.sources:
        doc_name = linked_source.source.doc_name
        if doc_name not in docs:
            docs[doc_name] = []
        docs[doc_name].append(linked_source)

    for doc_name, sources in docs.items():
        with st.expander(f"{doc_name} ({len(sources)} chunks)"):
            for source in sources:
                page = source.source.page_number
                score = source.source.relevance_score

                st.markdown(
                    f"- Page {page or 'N/A'} | "
                    f"Relevance: {score:.2f} | "
                    f"[Open PDF]({source.resolved_path})"
                )

                # Show snippet
                snippet = source.source.chunk_text[:200]
                st.caption(f"Preview: {snippet}...")


def _generate_markdown(report: FinalReport) -> str:
    """Generate markdown export of report."""
    lines = [
        "# Research Report",
        "",
        f"**Query:** {report.query}",
        "",
        f"**Quality Score:** {report.quality_score}/500",
        f"**Tasks Completed:** {report.todo_items_completed}",
        f"**Research Iterations:** {report.research_iterations}",
        "",
        "## Answer",
        "",
        report.answer,
        "",
        "## Key Findings",
        "",
    ]

    for i, finding in enumerate(report.findings, 1):
        lines.extend([
            f"### Finding {i}",
            "",
            f"**Claim:** {finding.claim}",
            "",
            f"**Evidence:** {finding.evidence}",
            "",
            f"**Confidence:** {finding.confidence}",
            "",
            f"**Sources:** {', '.join(s.doc_name for s in finding.sources)}",
            "",
        ])

    lines.extend([
        "## Sources",
        "",
    ])

    for linked_source in report.sources:
        source = linked_source.source
        lines.append(
            f"- **{source.doc_name}** (p. {source.page_number or 'N/A'}): "
            f"Relevance {source.relevance_score:.2f}"
        )

    lines.extend([
        "",
        "---",
        "*Generated by Rabbithole-Agent*",
    ])

    return "\n".join(lines)
