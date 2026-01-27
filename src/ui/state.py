"""Session state management for Streamlit UI."""

from dataclasses import dataclass, field
from typing import Any

import streamlit as st


@dataclass
class SessionState:
    """Manages session state for the research UI."""

    # Query state
    current_query: str = ""
    query_submitted: bool = False

    # Agent state
    agent_state: dict = field(default_factory=dict)
    thread_id: str = ""

    # HITL state (legacy form-based)
    hitl_pending: bool = False
    hitl_checkpoint: dict | None = None
    hitl_answers: dict = field(default_factory=dict)

    # HITL chat-based state (new)
    hitl_conversation_history: list[dict] = field(default_factory=list)
    hitl_state: dict | None = None
    waiting_for_human_input: bool = False
    conversation_ended: bool = False
    input_counter: int = 0
    workflow_phase: str = "hitl"  # "hitl", "research", "completed"

    # Database selection
    use_ext_database: bool = True
    selected_database: str = ""
    k_results: int = 5

    # Settings
    max_search_queries: int = 5
    enable_web_search: bool = False
    enable_quality_checker: bool = True

    # Results
    final_report: dict | None = None
    messages: list[str] = field(default_factory=list)
    hitl_result: dict | None = None  # Store HITL phase results for display

    # UI state
    show_debug: bool = False
    error: str | None = None


def get_session_state() -> SessionState:
    """Get or initialize session state."""
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    return st.session_state.session


def reset_session_state() -> None:
    """Reset session state for new query."""
    st.session_state.session = SessionState()


def update_agent_state(state: dict) -> None:
    """Update agent state in session."""
    session = get_session_state()
    previous_agent_state = session.agent_state
    previous_messages = (
        previous_agent_state.get("messages", [])
        if isinstance(previous_agent_state, dict)
        else []
    )

    session.agent_state = state

    # Check for HITL pending
    if state.get("hitl_pending"):
        session.hitl_pending = True
        session.hitl_checkpoint = state.get("hitl_checkpoint")

    # Check for final report
    if state.get("final_report"):
        session.final_report = state.get("final_report")

    # Accumulate messages
    if state.get("messages"):
        new_messages = state.get("messages", [])
        if isinstance(new_messages, list) and isinstance(previous_messages, list):
            start_idx = min(len(previous_messages), len(new_messages))
            session.messages.extend(new_messages[start_idx:])
        else:
            session.messages.extend(list(new_messages))


def get_current_phase() -> str:
    """Get current agent phase."""
    session = get_session_state()
    return session.agent_state.get("phase", "idle")


def get_todo_list() -> list[dict]:
    """Get current todo list."""
    session = get_session_state()
    return session.agent_state.get("todo_list", [])


def get_research_context() -> dict:
    """Get current research context."""
    session = get_session_state()
    return session.agent_state.get("research_context", {})


def set_hitl_answer(question_id: str, answer: Any) -> None:
    """Set HITL answer."""
    session = get_session_state()
    session.hitl_answers[question_id] = answer


def get_hitl_answers() -> dict:
    """Get all HITL answers."""
    session = get_session_state()
    return session.hitl_answers


def clear_hitl_state() -> None:
    """Clear HITL state after processing."""
    session = get_session_state()
    session.hitl_pending = False
    session.hitl_checkpoint = None
    session.hitl_answers = {}


def add_message(message: str) -> None:
    """Add a message to the session."""
    session = get_session_state()
    session.messages.append(message)


def set_error(error: str) -> None:
    """Set error message."""
    session = get_session_state()
    session.error = error


def clear_error() -> None:
    """Clear error message."""
    session = get_session_state()
    session.error = None


# Chat-based HITL helper functions


def add_hitl_message(role: str, content: str) -> None:
    """Add a message to the HITL conversation history.

    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
    """
    session = get_session_state()
    session.hitl_conversation_history.append({"role": role, "content": content})


def reset_hitl_conversation() -> None:
    """Reset the HITL conversation for a new session."""
    session = get_session_state()
    session.hitl_conversation_history = []
    session.hitl_state = None
    session.waiting_for_human_input = False
    session.conversation_ended = False
    session.input_counter = 0
    session.workflow_phase = "hitl"


def get_selected_database() -> str:
    """Get the currently selected database name.

    Returns:
        Selected database name or empty string if none selected
    """
    session = get_session_state()
    return session.selected_database if session.use_ext_database else ""


def set_database_selection(use_ext: bool, db_name: str, k: int) -> None:
    """Set database selection settings.

    Args:
        use_ext: Whether to use external database
        db_name: Selected database name
        k: Number of results per query
    """
    session = get_session_state()
    session.use_ext_database = use_ext
    session.selected_database = db_name
    session.k_results = k


def set_workflow_phase(phase: str) -> None:
    """Set the current workflow phase.

    Args:
        phase: One of 'hitl', 'research', 'completed'
    """
    session = get_session_state()
    session.workflow_phase = phase


def get_workflow_phase() -> str:
    """Get the current workflow phase.

    Returns:
        Current phase string
    """
    session = get_session_state()
    return session.workflow_phase
