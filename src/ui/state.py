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

    # HITL state
    hitl_pending: bool = False
    hitl_checkpoint: dict | None = None
    hitl_answers: dict = field(default_factory=dict)

    # Results
    final_report: dict | None = None
    messages: list[str] = field(default_factory=list)

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
        session.messages.extend(state["messages"])


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
