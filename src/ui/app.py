"""Main Streamlit application for Rabbithole-Agent."""

import logging
import uuid
from pathlib import Path

import streamlit as st

from src.agents.graph import create_research_graph
from src.agents.state import create_initial_state
from src.config import settings
from src.services.chromadb_client import ChromaDBClient
from src.ui.components import (
    render_chat_hitl,
    render_hitl_panel,
    render_hitl_summary,
    render_preliminary_results,
    render_research_status,
    render_results_view,
    render_safe_exit,
    render_todo_approval,
    render_todo_display,
    render_todo_side_panel,
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

LICENSE_FILE = Path(__file__).parent.parent.parent / "assets" / "LICENCE"


def get_license_content() -> str:
    """Return license text for tooltip."""
    try:
        return LICENSE_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def render_header():
    """Render the BrAIn header with image, version, and license."""
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        license_text = get_license_content()
        # Title row with version and license popover
        title_col, license_col = st.columns([4, 1])
        with title_col:
            st.markdown(
                f'<h1 style="margin-bottom: 0;">Br<span style="color:darkorange;"><b>AI</b></span>n <sup style="font-size: 0.4em; color: gray;">{VERSION}</sup></h1>',
                unsafe_allow_html=True,
            )
        with license_col:
            st.markdown(
                '<p style="text-align: right; font-size:12px; font-weight:bold; color:darkorange;">LIZENZ</p>',
                unsafe_allow_html=True,
                help=license_text if license_text else "Apache License 2.0",
            )

        st.markdown("## Wissensdatenbank-Konnektor")
        st.caption("Local Hybrid Researcher mit Deep Reference-Following")

    with col2:
        if HEADER_IMAGE.exists():
            st.image(str(HEADER_IMAGE), use_container_width=True)
        else:
            st.warning("Header image not found")


def _render_preliminary_results() -> None:
    session = get_session_state()
    state = session.agent_state or {}

    if not isinstance(state, dict):
        return

    research_context = state.get("research_context", {})
    todo_list = state.get("todo_list", [])
    phase = state.get("phase", "")

    if not research_context and not todo_list and not session.messages:
        return

    with st.expander("Preliminary Results", expanded=False):
        if todo_list:
            completed = sum(1 for t in todo_list if t.get("completed"))
            total = len(todo_list)
            st.write(f"Tasks: {completed}/{total}")

        metadata = research_context.get("metadata", {}) if isinstance(research_context, dict) else {}
        referenced = metadata.get("documents_referenced", []) if isinstance(metadata, dict) else []
        if referenced:
            st.write(f"Referenced documents: {len(referenced)}")

        if phase == "execute_tasks":
            if session.messages:
                st.write("Latest updates:")
                for msg in reversed(session.messages[-5:]):
                    st.text(msg)


def _run_graph_stream(graph, input_state: dict, config: dict) -> None:
    # Create column placeholders for streaming updates
    main_col_placeholder = st.empty()
    todo_col_placeholder = st.empty()

    last_state: dict | None = None
    for state in graph.stream(input_state, config, stream_mode="values"):
        last_state = state
        update_agent_state(state)

        phase = get_current_phase()

        # Use column layout during streaming
        col1, col2 = st.columns([2, 1])

        with col1:
            with main_col_placeholder.container():
                render_research_status()
                if phase not in ["idle", "analyze"]:
                    render_preliminary_results()
                render_messages()

        with col2:
            with todo_col_placeholder.container():
                render_todo_side_panel()

    if last_state:
        update_agent_state(last_state)


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

        # Show HITL summary if available (after HITL phase completes) - full width
        if session.hitl_result:
            render_hitl_summary()

        # HITL checkpoints during research - full width
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

        # Column layout: 2/3 main content, 1/3 todo side panel
        main_col, todo_col = st.columns([2, 1])

        with main_col:
            # Progress status (spinner-based)
            render_research_status()

            # Preliminary results with nested expanders
            if phase not in ["idle", "analyze"]:
                render_preliminary_results()

            # Activity log
            render_messages()

        with todo_col:
            # Side panel with progress, spinner, and task list
            render_todo_side_panel()

        # Results display - check for completion
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

    set_workflow_phase("research")

    try:
        # Generate thread ID for HITL resume capability
        thread_id = str(uuid.uuid4())
        session.thread_id = thread_id

        # Get user query and research queries from HITL
        user_query = hitl_result.get("user_query", "")
        research_queries = hitl_result.get("research_queries", [user_query])

        # Extract analysis fields (support both nested and flat structure)
        analysis = hitl_result.get("analysis", {})
        entities = hitl_result.get("entities", analysis.get("entities", []))
        scope = hitl_result.get("scope", analysis.get("scope", ""))
        context = hitl_result.get("context", analysis.get("context", ""))

        # Build additional context from HITL summary
        additional_context = hitl_result.get("summary", "")
        if context and context not in additional_context:
            additional_context = f"{additional_context} {context}".strip()

        add_message(f"Starting research: {user_query[:50]}...")

        # Create graph and initial state
        graph = create_research_graph()
        initial_state = create_initial_state(user_query)

        # Add HITL results to state
        initial_state["query_analysis"] = {
            "original_query": user_query,
            "keywords": [],
            "entities": entities,
            "scope": scope,
            "assumed_context": [context] if context else [],
            "clarification_needed": False,
            "hitl_refinements": [],
        }

        # Map research queries from HITL to state for Phase 2
        initial_state["research_queries"] = research_queries
        initial_state["additional_context"] = additional_context

        # Add database selection to state
        if session.selected_database:
            initial_state["selected_database"] = session.selected_database
        initial_state["k_results"] = session.k_results

        config = {"configurable": {"thread_id": thread_id}}

        _run_graph_stream(graph, initial_state, config)

        add_message(f"Phase: {get_current_phase()}")

        st.rerun()

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
        set_workflow_phase("research")

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

        _run_graph_stream(graph, initial_state, config)

        add_message(f"Phase: {get_current_phase()}")

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

        _run_graph_stream(graph, current_state, config)

        add_message(f"Phase: {get_current_phase()}")

    except Exception as e:
        logger.exception("Resume failed")
        set_error(f"Resume failed: {e}")


if __name__ == "__main__":
    main()
