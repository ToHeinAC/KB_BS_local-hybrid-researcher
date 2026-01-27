"""Chat-based HITL clarification panel component."""

import streamlit as st

from src.models.hitl import ClarificationQuestion, HITLDecision
from src.services.hitl_service import HITLService
from src.ui.state import (
    add_hitl_message,
    clear_hitl_state,
    get_hitl_answers,
    get_session_state,
    reset_hitl_conversation,
    set_hitl_answer,
    set_workflow_phase,
)


def render_hitl_panel() -> HITLDecision | None:
    """Render the legacy form-based HITL clarification panel.

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
    """Render query clarification questions (legacy form-based)."""
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


def render_chat_hitl() -> dict | None:
    """Render chat-based HITL interaction.

    Uses st.chat_input and st.chat_message for conversational flow.
    User types /end to finalize and proceed to research.

    Returns:
        Dict with research_queries and analysis if conversation ended, None otherwise
    """
    session = get_session_state()
    hitl_service = HITLService()

    # Display conversation history
    for msg in session.hitl_conversation_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial query (no history yet)
    if len(session.hitl_conversation_history) == 0:
        # Show initial prompt
        st.info("Geben Sie Ihre Forschungsanfrage ein. Ich werde Ihnen Fragen stellen, um die Suche zu verfeinern. Tippen Sie `/end` wenn Sie bereit sind, die Recherche zu starten.")

        user_query = st.chat_input(
            "Geben Sie Ihre Forschungsanfrage ein",
            key=f"hitl_input_{session.input_counter}",
        )

        if user_query:
            # Store initial query
            add_hitl_message("user", user_query)

            # Initialize HITL state
            language = hitl_service.detect_language(user_query)
            session.hitl_state = {
                "user_query": user_query,
                "language": language,
                "conversation_history": [{"role": "user", "content": user_query}],
                "analysis": {},
            }

            # Generate follow-up questions
            questions = hitl_service.generate_follow_up_questions(
                session.hitl_state,
                language=language,
            )

            # Add assistant response
            add_hitl_message("assistant", questions)
            session.hitl_state["conversation_history"].append({
                "role": "assistant",
                "content": questions,
            })

            session.waiting_for_human_input = True
            session.input_counter += 1
            st.rerun()

    # Feedback loop
    elif session.waiting_for_human_input and not session.conversation_ended:
        feedback = st.chat_input(
            "Ihre Antwort (/end zum Fortfahren)",
            key=f"hitl_input_{session.input_counter}",
        )

        if feedback:
            feedback_lower = feedback.strip().lower()

            if feedback_lower == "/end":
                # Finalize conversation
                session.conversation_ended = True

                # Analyze accumulated feedback
                if session.hitl_state:
                    analysis = hitl_service.analyse_user_feedback(session.hitl_state)
                    session.hitl_state["analysis"] = analysis

                    # Generate final research queries
                    result = hitl_service.generate_knowledge_base_questions(
                        session.hitl_state,
                        max_queries=session.max_search_queries,
                    )

                    # Add final message
                    summary = f"Starte Recherche mit {len(result.get('research_queries', []))} Suchanfragen:\n\n"
                    for i, q in enumerate(result.get("research_queries", []), 1):
                        summary += f"{i}. {q}\n"

                    add_hitl_message("assistant", summary)

                    # Transition to research phase
                    set_workflow_phase("research")
                    st.rerun()

                    return result

            else:
                # Add user feedback to history
                add_hitl_message("user", feedback)
                if session.hitl_state:
                    session.hitl_state["conversation_history"].append({
                        "role": "user",
                        "content": feedback,
                    })

                    # Analyze feedback incrementally
                    analysis = hitl_service.analyse_user_feedback(session.hitl_state)
                    session.hitl_state["analysis"] = analysis

                    # Generate follow-up questions if we haven't had many exchanges
                    if session.input_counter < 4:
                        language = session.hitl_state.get("language", "de")
                        questions = hitl_service.generate_follow_up_questions(
                            session.hitl_state,
                            language=language,
                        )

                        add_hitl_message("assistant", questions)
                        session.hitl_state["conversation_history"].append({
                            "role": "assistant",
                            "content": questions,
                        })
                    else:
                        # Suggest ending after several exchanges
                        if language == "de":
                            msg = "Wir haben genug Kontext gesammelt. Tippen Sie `/end` um die Recherche zu starten, oder fügen Sie weitere Details hinzu."
                        else:
                            msg = "We have collected enough context. Type `/end` to start the research, or add more details."
                        add_hitl_message("assistant", msg)
                        session.hitl_state["conversation_history"].append({
                            "role": "assistant",
                            "content": msg,
                        })

                session.input_counter += 1
                st.rerun()

    # Conversation ended, waiting for research to start
    elif session.conversation_ended:
        if session.hitl_state:
            result = hitl_service.generate_knowledge_base_questions(
                session.hitl_state,
                max_queries=session.max_search_queries,
            )
            return result

    return None


def create_hitl_result(hitl_state: dict) -> dict:
    """Create HITL result dict from conversation state.

    Args:
        hitl_state: The accumulated HITL state

    Returns:
        Dict with research_queries, analysis, and user_query
    """
    hitl_service = HITLService()

    result = hitl_service.generate_knowledge_base_questions(
        hitl_state,
        max_queries=5,
    )

    return {
        "research_queries": result.get("research_queries", []),
        "summary": result.get("summary", ""),
        "user_query": hitl_state.get("user_query", ""),
        "analysis": hitl_state.get("analysis", {}),
    }


def render_hitl_summary() -> None:
    """Render HITL results summary in an expander.

    Shows after HITL phase completes: original query, research queries, entities.
    """
    session = get_session_state()

    hitl_result = session.hitl_result
    if not hitl_result:
        return

    with st.expander("HITL Zusammenfassung", expanded=False):
        # Original query
        user_query = hitl_result.get("user_query", "")
        if user_query:
            st.markdown("**Urspruengliche Anfrage:**")
            st.write(user_query)

        # Research queries
        research_queries = hitl_result.get("research_queries", [])
        if research_queries:
            st.markdown("**Forschungsabfragen:**")
            for i, q in enumerate(research_queries, 1):
                st.write(f"{i}. {q}")

        # Analysis entities
        analysis = hitl_result.get("analysis", {})
        entities = analysis.get("entities", [])
        if entities:
            st.markdown("**Erkannte Entitaeten:**")
            st.write(", ".join(entities))

        # Scope
        scope = analysis.get("scope", "")
        if scope:
            st.markdown("**Umfang:**")
            st.write(scope)
