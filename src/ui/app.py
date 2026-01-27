"""Main Streamlit application for Rabbithole-Agent."""

import logging
import uuid
from pathlib import Path

import streamlit as st

from src.agents.graph import create_research_graph, resume_research
from src.agents.state import create_initial_state
from src.config import settings
from src.services.chromadb_client import ChromaDBClient
from src.ui.components import (
    render_chat_hitl,
    render_hitl_panel,
    render_hitl_summary,
    render_query_input,
    render_research_status,
    render_results_view,
    render_safe_exit,
    render_todo_approval,
    render_todo_display,
)
from src.ui.components.safe_exit import render_connection_status
from src.ui.components.todo_display import render_messages
from src.ui.state import (
    add_message,
    get_current_phase,
    get_session_state,
    get_workflow_phase,
    reset_hitl_conversation,
    set_database_selection,
    set_error,
    set_workflow_phase,
    update_agent_state,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Asset paths
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
HEADER_IMAGE = ASSETS_DIR / "Header_fuer_Chatbot.png"

# Version info
VERSION = "V2.2"

# License text
LICENSE_TEXT = """
MIT License

Copyright (c) 2025 BrAIn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""


def get_license_content() -> str:
    """Return license text for tooltip."""
    return LICENSE_TEXT.strip()


def render_header():
    """Render the BrAIn header with image, version, and license."""
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        # Title with version tooltip
        st.markdown(
            f'<h1 style="margin-bottom: 0;" title="Version {VERSION}">Br<span style="color:darkorange;"><b>AI</b></span>n <sup style="font-size: 0.4em; color: gray;">{VERSION}</sup></h1>',
            unsafe_allow_html=True,
        )
        st.markdown("## Wissensdatenbank-Konnektor")
        st.caption("Local Hybrid Researcher mit Deep Reference-Following")

    with col2:
        # License indicator (right-aligned)
        license_col1, license_col2 = st.columns([3, 1])
        with license_col2:
            st.markdown(
                f'<p style="text-align: right; color: darkorange; font-size: 0.8em;" title="{get_license_content()[:200]}...">MIT License</p>',
                unsafe_allow_html=True,
            )
        if HEADER_IMAGE.exists():
            st.image(str(HEADER_IMAGE), use_container_width=True)
        else:
            st.warning("Header image not found")


def render_sidebar():
    """Render the sidebar with database selection and settings."""
    session = get_session_state()

    # Get ChromaDB client for database listing
    try:
        chromadb_client = ChromaDBClient()
        available_dbs = chromadb_client.list_database_directories()
    except Exception as e:
        logger.warning(f"Failed to list databases: {e}")
        available_dbs = []

    with st.sidebar:
        st.header("Einstellungen")

        # Connection status
        render_connection_status()

        # Database selection expander
        with st.expander("Wissensdatenbank", expanded=True):
            use_ext_db = st.checkbox(
                "Benutze externe Wissensdatenbank",
                value=session.use_ext_database,
                key="use_ext_db",
            )

            if use_ext_db:
                if available_dbs:
                    # Pre-select first database if none selected
                    current_idx = 0
                    if session.selected_database in available_dbs:
                        current_idx = available_dbs.index(session.selected_database)

                    selected = st.selectbox(
                        "Wissensdatenbank auswählen",
                        options=available_dbs,
                        index=current_idx,
                        key="selected_db",
                    )

                    # Extract and show embedding model
                    embedding_model = chromadb_client.extract_embedding_model(selected)
                    if embedding_model:
                        st.caption(f"Embedding: {embedding_model}")

                    k_results = st.slider(
                        "Ergebnisse pro Abfrage",
                        min_value=1,
                        max_value=10,
                        value=session.k_results,
                        key="k_results_slider",
                    )

                    # Update session state
                    set_database_selection(use_ext_db, selected, k_results)
                else:
                    st.warning("Keine Datenbanken gefunden")
                    set_database_selection(False, "", session.k_results)
            else:
                st.info("Alle Sammlungen werden durchsucht")
                set_database_selection(False, "", session.k_results)

        # Settings expander
        with st.expander("Erweiterte Einstellungen"):
            session.max_search_queries = st.slider(
                "Max. Forschungsabfragen",
                min_value=1,
                max_value=10,
                value=session.max_search_queries,
                key="max_queries_slider",
            )

            session.enable_web_search = st.checkbox(
                "Web Search aktivieren",
                value=session.enable_web_search,
                key="enable_web",
                disabled=True,  # Not yet implemented
            )

            session.enable_quality_checker = st.checkbox(
                "Qualitätsprüfung aktivieren",
                value=session.enable_quality_checker,
                key="enable_quality",
            )

        st.divider()

        # Safe exit button
        render_safe_exit()

        # Debug toggle
        if st.checkbox("Debug Info anzeigen", key="debug_toggle"):
            with st.expander("Session State"):
                st.json({
                    "phase": get_current_phase(),
                    "workflow_phase": session.workflow_phase,
                    "hitl_pending": session.hitl_pending,
                    "query": session.current_query,
                    "thread_id": session.thread_id,
                    "selected_database": session.selected_database,
                    "k_results": session.k_results,
                    "conversation_history_len": len(session.hitl_conversation_history),
                })


def main():
    """Main Streamlit application entry point."""
    # Page config
    st.set_page_config(
        page_title=f"BrAIn {VERSION} - Wissensdatenbank",
        page_icon=":brain:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    render_header()

    # Sidebar
    render_sidebar()

    # Main content area
    session = get_session_state()

    # Display any errors
    if session.error:
        st.error(session.error)

    # Phase-based rendering
    workflow_phase = get_workflow_phase()

    if workflow_phase == "hitl":
        # Chat-based HITL phase
        result = render_chat_hitl()

        if result:
            # HITL conversation ended, start research
            _start_research_from_hitl(result)

    elif workflow_phase == "research":
        # Research in progress
        phase = get_current_phase()

        # Show HITL summary if available (after HITL phase completes)
        if session.hitl_result:
            render_hitl_summary()

        # HITL checkpoints during research
        if session.hitl_pending:
            checkpoint_type = (
                session.hitl_checkpoint.get("checkpoint_type", "")
                if session.hitl_checkpoint
                else ""
            )

            if checkpoint_type == "query_clarify":
                decision = render_hitl_panel()
                if decision:
                    _resume_with_decision(decision.model_dump())
                    st.rerun()

            elif checkpoint_type == "todo_approve":
                decision = render_todo_approval()
                if decision:
                    _resume_with_decision(decision.model_dump())
                    st.rerun()

        # Progress status (spinner-based)
        render_research_status()

        # Progress display
        if phase not in ["idle", "analyze"]:
            render_todo_display()
            render_messages()

        # Results display
        if session.final_report:
            set_workflow_phase("completed")
            st.rerun()

    elif workflow_phase == "completed":
        # Show final results
        render_results_view()

        # Button to start new research
        if st.button("Neue Recherche starten", type="primary"):
            reset_hitl_conversation()
            st.rerun()


def _start_research_from_hitl(hitl_result: dict) -> None:
    """Start research from HITL conversation result.

    Args:
        hitl_result: Dict with research_queries and analysis from HITL
    """
    session = get_session_state()

    # Store HITL result for display
    session.hitl_result = hitl_result

    try:
        # Generate thread ID for HITL resume capability
        thread_id = str(uuid.uuid4())
        session.thread_id = thread_id

        # Get user query and research queries from HITL
        user_query = hitl_result.get("user_query", "")
        research_queries = hitl_result.get("research_queries", [user_query])

        add_message(f"Starting research: {user_query[:50]}...")

        # Create graph and initial state
        graph = create_research_graph()
        initial_state = create_initial_state(user_query)

        # Add HITL results to state
        initial_state["query_analysis"] = {
            "original_query": user_query,
            "keywords": [],
            "entities": hitl_result.get("analysis", {}).get("entities", []),
            "scope": hitl_result.get("analysis", {}).get("scope", ""),
            "assumed_context": [hitl_result.get("analysis", {}).get("context", "")],
            "clarification_needed": False,
            "hitl_refinements": [],
        }

        # Add database selection to state
        if session.selected_database:
            initial_state["selected_database"] = session.selected_database
        initial_state["k_results"] = session.k_results

        config = {"configurable": {"thread_id": thread_id}}

        # Run graph until HITL checkpoint or completion
        result = graph.invoke(initial_state, config)

        # Update session state
        update_agent_state(result)

        add_message(f"Phase: {result.get('phase', 'unknown')}")

    except Exception as e:
        logger.exception("Research failed")
        set_error(f"Research failed: {e}")


def _start_research(query: str) -> None:
    """Start a new research session.

    Args:
        query: The research query
    """
    session = get_session_state()

    try:
        # Generate thread ID for HITL resume capability
        thread_id = str(uuid.uuid4())
        session.thread_id = thread_id

        add_message(f"Starting research: {query[:50]}...")

        # Create graph and initial state
        graph = create_research_graph()
        initial_state = create_initial_state(query)

        # Add database selection to state
        if session.selected_database:
            initial_state["selected_database"] = session.selected_database
        initial_state["k_results"] = session.k_results

        config = {"configurable": {"thread_id": thread_id}}

        # Run graph until HITL checkpoint or completion
        result = graph.invoke(initial_state, config)

        # Update session state
        update_agent_state(result)

        add_message(f"Phase: {result.get('phase', 'unknown')}")

    except Exception as e:
        logger.exception("Research failed")
        set_error(f"Research failed: {e}")


def _resume_with_decision(decision: dict) -> None:
    """Resume research with HITL decision.

    Args:
        decision: The HITL decision dict
    """
    session = get_session_state()

    try:
        add_message("Resuming research with user input...")

        # Create graph
        graph = create_research_graph()

        config = {"configurable": {"thread_id": session.thread_id}}

        # Get current state and update with decision
        current_state = session.agent_state.copy()
        current_state["hitl_decision"] = decision
        current_state["hitl_pending"] = False

        # Resume from current state
        result = graph.invoke(current_state, config)

        # Update session state
        update_agent_state(result)

        add_message(f"Phase: {result.get('phase', 'unknown')}")

    except Exception as e:
        logger.exception("Resume failed")
        set_error(f"Resume failed: {e}")


if __name__ == "__main__":
    main()
