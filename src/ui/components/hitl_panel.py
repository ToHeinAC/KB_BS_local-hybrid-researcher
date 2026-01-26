"""HITL clarification panel component."""

import streamlit as st

from src.models.hitl import ClarificationQuestion, HITLDecision
from src.ui.state import (
    clear_hitl_state,
    get_hitl_answers,
    get_session_state,
    set_hitl_answer,
)


def render_hitl_panel() -> HITLDecision | None:
    """Render the HITL clarification panel.

    Returns:
        HITLDecision if user submits, None otherwise
    """
    session = get_session_state()

    if not session.hitl_pending or not session.hitl_checkpoint:
        return None

    checkpoint = session.hitl_checkpoint
    checkpoint_type = checkpoint.get("checkpoint_type", "")

    st.subheader("Clarification Needed")

    if checkpoint_type == "query_clarify":
        return _render_query_clarification(checkpoint)
    elif checkpoint_type == "todo_approve":
        # This is handled by todo_approval component
        return None

    return None


def _render_query_clarification(checkpoint: dict) -> HITLDecision | None:
    """Render query clarification questions."""
    content = checkpoint.get("content", {})
    questions_data = content.get("questions", [])

    if not questions_data:
        return None

    questions = [ClarificationQuestion.model_validate(q) for q in questions_data]

    st.info("Please answer these questions to help refine the search:")

    with st.form("clarification_form"):
        for question in questions:
            st.markdown(f"**{question.question}**")

            if question.context:
                st.caption(question.context)

            if question.options:
                # Multiple choice
                answer = st.selectbox(
                    "Select an option",
                    options=[""] + question.options,
                    key=f"q_{question.id}",
                    label_visibility="collapsed",
                )
                if answer:
                    set_hitl_answer(question.id, answer)
            else:
                # Free text
                answer = st.text_input(
                    "Your answer",
                    key=f"q_{question.id}",
                    label_visibility="collapsed",
                )
                if answer:
                    set_hitl_answer(question.id, answer)

            st.divider()

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            submitted = st.form_submit_button("Continue", type="primary")
        with col2:
            skipped = st.form_submit_button("Skip", type="secondary")

        if submitted:
            answers = get_hitl_answers()
            clear_hitl_state()
            return HITLDecision(
                approved=True,
                modifications={"answers": answers},
            )

        if skipped:
            clear_hitl_state()
            return HITLDecision(
                approved=True,
                skip_reason="User skipped clarification",
            )

    return None
