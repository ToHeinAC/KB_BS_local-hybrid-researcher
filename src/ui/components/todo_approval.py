"""ToDo list approval component."""

import streamlit as st

from src.models.hitl import HITLDecision
from src.ui.state import clear_hitl_state, get_session_state


def render_todo_approval() -> HITLDecision | None:
    """Render the ToDo list approval panel.

    Returns:
        HITLDecision if user submits, None otherwise
    """
    session = get_session_state()

    if not session.hitl_pending or not session.hitl_checkpoint:
        return None

    checkpoint = session.hitl_checkpoint
    checkpoint_type = checkpoint.get("checkpoint_type", "")

    if checkpoint_type != "todo_approve":
        return None

    st.subheader("Research Tasks")
    st.info("Review and approve the research tasks. You can edit, remove, or add tasks.")

    content = checkpoint.get("content", {})
    items_data = content.get("items", [])

    if not items_data:
        st.warning("No tasks generated")
        return None

    # Convert to editable format
    tasks = []
    for item in items_data:
        tasks.append({
            "id": item.get("id"),
            "task": item.get("task", ""),
            "context": item.get("context", ""),
            "include": True,
        })

    # Render editable task list
    edited_tasks = []
    removed_ids = []

    for i, task in enumerate(tasks):
        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                new_task = st.text_input(
                    f"Task {task['id']}",
                    value=task["task"],
                    key=f"task_{task['id']}",
                )
                new_context = st.text_input(
                    "Context",
                    value=task["context"],
                    key=f"context_{task['id']}",
                    label_visibility="collapsed",
                )

            with col2:
                include = st.checkbox(
                    "Include",
                    value=True,
                    key=f"include_{task['id']}",
                )

            if include:
                edited_tasks.append({
                    "id": task["id"],
                    "task": new_task,
                    "context": new_context,
                })
            else:
                removed_ids.append(task["id"])

            st.divider()

    # Add new task
    with st.expander("Add new task"):
        new_task_text = st.text_input(
            "New task description",
            key="new_task_text",
        )
        new_task_context = st.text_input(
            "Context (optional)",
            key="new_task_context",
        )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Approve Tasks", type="primary"):
            modifications = {
                "edited_items": edited_tasks,
                "removed_ids": removed_ids,
            }

            if new_task_text:
                modifications["new_items"] = [{
                    "task": new_task_text,
                    "context": new_task_context or "",
                }]

            clear_hitl_state()
            return HITLDecision(
                approved=True,
                modifications=modifications,
            )

    with col2:
        if st.button("Use as-is", type="secondary"):
            clear_hitl_state()
            return HITLDecision(
                approved=True,
            )

    return None
