"""Main Streamlit application for Rabbithole-Agent."""

import logging
import uuid

import streamlit as st

from src.agents.graph import create_research_graph, resume_research
from src.agents.state import create_initial_state
from src.config import settings
from src.ui.components import (
    render_hitl_panel,
    render_query_input,
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
    set_error,
    update_agent_state,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main Streamlit application entry point."""
    # Page config
    st.set_page_config(
        page_title="Rabbithole-Agent",
        page_icon=":rabbit:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title
    st.title("Rabbithole-Agent")
    st.caption("Local Hybrid Researcher with Deep Reference-Following")

    # Sidebar
    with st.sidebar:
        st.header("Rabbithole-Agent")
        render_connection_status()
        render_safe_exit()

        # Debug toggle
        if st.checkbox("Show Debug Info", key="debug_toggle"):
            session = get_session_state()
            with st.expander("Session State"):
                st.json({
                    "phase": get_current_phase(),
                    "hitl_pending": session.hitl_pending,
                    "query": session.current_query,
                    "thread_id": session.thread_id,
                })

    # Main content area
    session = get_session_state()

    # Display any errors
    if session.error:
        st.error(session.error)

    # Phase-based rendering
    phase = get_current_phase()

    # Query input (always visible at top)
    query = render_query_input()

    if query:
        # Start new research session
        _start_research(query)
        st.rerun()

    # HITL checkpoints
    if session.hitl_pending:
        checkpoint_type = session.hitl_checkpoint.get("checkpoint_type", "") if session.hitl_checkpoint else ""

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

    # Progress display
    if phase not in ["idle", "analyze"]:
        render_todo_display()
        render_messages()

    # Results display
    if session.final_report:
        render_results_view()


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
