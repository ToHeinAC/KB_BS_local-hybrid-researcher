"""ToDo list display component (read-only progress view)."""

import streamlit as st

from src.ui.state import get_current_phase, get_session_state, get_todo_list


def render_todo_display() -> None:
    """Render the ToDo list progress display."""
    session = get_session_state()
    todo_list = get_todo_list()
    phase = get_current_phase()

    if not todo_list:
        return

    st.subheader("Research Progress")

    # Calculate progress
    completed = sum(1 for item in todo_list if item.get("completed"))
    total = len(todo_list)
    progress = completed / total if total > 0 else 0

    # Progress bar
    st.progress(progress, text=f"Completed: {completed}/{total} tasks")

    # Current phase indicator
    phase_labels = {
        "analyze": "Analyzing Query",
        "hitl_clarify": "Awaiting Clarification",
        "generate_todo": "Generating Tasks",
        "hitl_approve_todo": "Awaiting Task Approval",
        "execute_tasks": "Executing Research",
        "synthesize": "Synthesizing Findings",
        "quality_check": "Checking Quality",
        "attribute_sources": "Adding Citations",
        "complete": "Complete",
    }
    phase_label = phase_labels.get(phase, phase)
    st.caption(f"Current phase: **{phase_label}**")

    # Task list
    with st.expander("Task Details", expanded=True):
        for item in todo_list:
            task_id = item.get("id")
            task = item.get("task", "")
            context = item.get("context", "")
            completed = item.get("completed", False)

            # Icon based on status
            current_task_id = session.agent_state.get("current_task_id")
            if completed:
                icon = ":white_check_mark:"
                status = "Completed"
            elif task_id == current_task_id:
                icon = ":hourglass_flowing_sand:"
                status = "In Progress"
            else:
                icon = ":clipboard:"
                status = "Pending"

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{icon} **Task {task_id}**: {task}")
                if context:
                    st.caption(context)
            with col2:
                st.caption(status)


def render_messages() -> None:
    """Render session messages/log."""
    session = get_session_state()

    if not session.messages:
        return

    with st.expander("Activity Log", expanded=False):
        for msg in reversed(session.messages[-20:]):  # Show last 20
            st.text(msg)
