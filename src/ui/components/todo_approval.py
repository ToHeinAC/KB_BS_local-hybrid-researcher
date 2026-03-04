"""ToDo list approval component."""

import streamlit as st

from src.models.hitl import HITLDecision
from src.ui.state import clear_hitl_state, get_session_state


def _init_pending_tasks() -> None:
    """Ensure pending_new_tasks exists in session state."""
    if "pending_new_tasks" not in st.session_state:
        st.session_state.pending_new_tasks = []


def render_todo_approval() -> HITLDecision | None:
    """Render the ToDo list approval panel.

    Returns:
        HITLDecision if user submits, None otherwise
    """
    session = get_session_state()
    _init_pending_tasks()

    if not session.hitl_pending or not session.hitl_checkpoint:
        return None

    checkpoint = session.hitl_checkpoint
    checkpoint_type = checkpoint.get("checkpoint_type", "")

    if checkpoint_type != "todo_approve":
        return None

    st.subheader("Forschungsaufgaben")
    st.info("Prüfen und genehmigen Sie die Forschungsaufgaben. Sie können Aufgaben bearbeiten, entfernen oder hinzufügen.")

    content = checkpoint.get("content", {})
    items_data = content.get("items", [])

    if not items_data:
        st.warning("Keine Aufgaben generiert")
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
        with st.expander(f"Aufgabe {i + 1}", expanded=True):
            col1, col2 = st.columns([4, 1])

            with col1:
                new_task = st.text_area(
                    " ",
                    value=task["task"],
                    key=f"task_{task['id']}",
                    label_visibility="collapsed",
                    height=68,
                )
                with st.expander("Kontext", expanded=False):
                    new_context = st.text_input(
                        " ",
                        value=task["context"],
                        key=f"context_{task['id']}",
                        label_visibility="collapsed",
                    )

            with col2:
                include = st.checkbox(
                    "Einbeziehen",
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

    # Dynamic new tasks list
    st.markdown("**Neue Aufgaben**")
    pending = st.session_state.pending_new_tasks
    indices_to_remove: list[int] = []

    for idx, entry in enumerate(pending):
        col_task, col_ctx, col_btn = st.columns([3, 2, 0.5])
        with col_task:
            pending[idx]["task"] = st.text_input(
                "Aufgabe",
                value=entry.get("task", ""),
                key=f"new_task_{idx}",
                label_visibility="collapsed",
                placeholder="Aufgabenbeschreibung",
            )
        with col_ctx:
            pending[idx]["context"] = st.text_input(
                "Kontext",
                value=entry.get("context", ""),
                key=f"new_ctx_{idx}",
                label_visibility="collapsed",
                placeholder="Kontext (optional)",
            )
        with col_btn:
            if st.button("\u2212", key=f"rm_task_{idx}"):
                indices_to_remove.append(idx)

    # Process removals (reverse order to keep indices valid)
    if indices_to_remove:
        for idx in sorted(indices_to_remove, reverse=True):
            pending.pop(idx)
        st.rerun()

    if st.button("+ Aufgabe hinzufuegen"):
        pending.append({"task": "", "context": ""})
        st.rerun()

    # Action buttons
    if st.button("Aufgaben genehmigen", type="primary"):
        modifications = {
            "edited_items": edited_tasks,
            "removed_ids": removed_ids,
        }

        # Collect all non-empty pending new tasks
        new_items = [
            {"task": e["task"], "context": e.get("context", "")}
            for e in pending
            if e.get("task", "").strip()
        ]
        if new_items:
            modifications["new_items"] = new_items

        st.session_state.pending_new_tasks = []
        clear_hitl_state()
        return HITLDecision(
            approved=True,
            modifications=modifications,
        )

    return None
