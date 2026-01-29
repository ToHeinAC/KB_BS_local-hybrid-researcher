"""Chat-based HITL clarification panel component."""

import streamlit as st

from src.models.hitl import ClarificationQuestion, HITLDecision
from src.services.hitl_service import HITLService, format_analysis_dict
from src.ui.state import (
    add_hitl_message,
    clear_hitl_state,
    get_hitl_answers,
    get_session_state,
    set_hitl_answer,
)


@st.cache_resource
def _get_hitl_service():
    """Return cached HITLService instance."""
    return HITLService()


def _perform_hitl_retrieval(query: str, session) -> dict:
    """Perform vector search and store results for display.

    Args:
        query: Search query text
        session: Session state object

    Returns:
        Dict of collection_name -> list of VectorResult
    """
    from src.ui.components.safe_exit import _get_chromadb_client

    client = _get_chromadb_client()

    # Respect user's database selection
    if session.selected_database:
        results_list = client.search_by_database_name(
            query, session.selected_database, top_k=3
        )
        # Convert list to dict format for consistency
        results = {session.selected_database: results_list}
    else:
        results = client.search_all_collections(query, top_k=3)

    # Initialize retrieval_history in hitl_state if needed
    if not session.hitl_state:
        session.hitl_state = {}
    if "retrieval_history" not in session.hitl_state:
        session.hitl_state["retrieval_history"] = {}

    # Count results and serialize chunks for storage
    total_chunks = 0
    chunks_data = []
    for collection_name, chunk_list in results.items():
        for chunk in chunk_list:
            total_chunks += 1
            chunks_data.append({
                "collection": collection_name,
                "doc_name": chunk.doc_name,
                "page": chunk.page_number,
                "score": round(chunk.relevance_score, 3),
                "text": chunk.chunk_text[:500],  # Truncate for display
            })

    # Track iteration
    iteration = len(session.hitl_state["retrieval_history"])
    session.hitl_state["retrieval_history"][f"iteration_{iteration}"] = {
        "queries": [query],
        "new_chunks": total_chunks,
        "duplicates": 0,
        "chunks": chunks_data,
    }

    return results


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


def render_hitl_understanding() -> None:
    """Display growing understanding after each user answer.

    Shows accumulated context (entities, scope, refined query) in a bordered container.
    Uses format_analysis_dict for consistent formatting.
    """
    session = get_session_state()

    if not session.hitl_state:
        return

    analysis = session.hitl_state.get("analysis", {})
    if not analysis:
        return

    # Use format_analysis_dict for consistent formatting
    formatted = format_analysis_dict(analysis)
    if not formatted:
        return

    with st.container(border=True):
        st.markdown("**Aktuelles Verstaendnis:**")
        st.markdown(formatted)


def _render_retrieval_history() -> None:
    """Display retrieval history from hitl_state or agent_state in an expander.

    Data structure from hitl_retrieve_chunks node:
    - queries: list of query strings
    - new_chunks: int count of new chunks retrieved
    - duplicates: int count of duplicate chunks skipped
    - chunks: list of chunk data dicts (optional)
    """
    session = get_session_state()

    # Read from hitl_state during HITL phase, agent_state during research
    hitl_state = session.hitl_state or {}
    agent_state = session.agent_state or {}
    if not isinstance(agent_state, dict):
        agent_state = {}

    retrieval_history = hitl_state.get("retrieval_history") or agent_state.get("retrieval_history", {})
    if not retrieval_history:
        return

    with st.expander("Retrieval History", expanded=False):
        for iteration_key, iteration_data in sorted(retrieval_history.items()):
            st.markdown(f"**{iteration_key}**")

            # Show queries
            queries = iteration_data.get("queries", [])
            if queries:
                st.markdown("*Queries:*")
                for q in queries:
                    st.markdown(f"- {q}")

            # Show chunk counts (matches hitl_retrieve_chunks node output)
            new_chunks = iteration_data.get("new_chunks", 0)
            duplicates = iteration_data.get("duplicates", 0)
            total = new_chunks + duplicates

            if total > 0:
                st.markdown(f"*Retrieved:* {new_chunks} new, {duplicates} duplicates")
                dedup_ratio = duplicates / total if total > 0 else 0
                st.markdown(f"*Dedup ratio:* {dedup_ratio:.0%}")

            # Show chunks in nested expanders
            chunks = iteration_data.get("chunks", [])
            if chunks:
                for i, chunk in enumerate(chunks):
                    doc_name = chunk.get("doc_name", "Unknown")
                    page = chunk.get("page")
                    score = chunk.get("score", 0)
                    collection = chunk.get("collection", "")

                    # Build header with source info
                    page_str = f" p.{page}" if page else ""
                    header = f"[{score:.3f}] {doc_name}{page_str} ({collection})"

                    with st.expander(header, expanded=False):
                        text = chunk.get("text", "")
                        st.text(text)

            st.divider()


def render_chat_hitl() -> dict | None:
    """Render chat-based HITL interaction.

    Uses st.chat_input and st.chat_message for conversational flow.
    User types /end to finalize and proceed to research.

    Returns:
        Dict with research_queries and analysis if conversation ended, None otherwise
    """
    session = get_session_state()
    hitl_service = _get_hitl_service()

    # Display conversation history
    for msg in session.hitl_conversation_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show accumulated understanding after conversation history
    render_hitl_understanding()

    # Show retrieval history if available
    _render_retrieval_history()

    # Initial query (no history yet)
    if len(session.hitl_conversation_history) == 0:
        # Show initial prompt
        st.info("Geben Sie Ihre Forschungsanfrage ein. Ich werde Ihnen Fragen stellen, um die Suche zu verfeinern. Tippen Sie `/end` wenn Sie bereit sind, die Recherche zu starten.")

        user_query = st.chat_input(
            "Geben Sie Ihre Forschungsanfrage ein",
            key=f"hitl_input_{session.input_counter}",
        )

        if user_query:
            # Store initial query immediately so user sees their message
            add_hitl_message("user", user_query)

            with st.spinner("Analysiere Anfrage..."):
                try:
                    # Initialize HITL state with LLM calls
                    language = hitl_service.detect_language(user_query)
                    session.hitl_state = {
                        "user_query": user_query,
                        "language": language,
                        "conversation_history": [{"role": "user", "content": user_query}],
                        "analysis": {},
                    }

                    # Perform initial vector retrieval
                    _perform_hitl_retrieval(user_query, session)

                    # Generate follow-up questions
                    questions = hitl_service.generate_follow_up_questions(
                        session.hitl_state,
                        language=language,
                    )
                except Exception as e:
                    st.error(f"LLM-Verbindung fehlgeschlagen: {e}")
                    # Use fallback questions and state
                    language = "de"
                    session.hitl_state = {
                        "user_query": user_query,
                        "language": language,
                        "conversation_history": [{"role": "user", "content": user_query}],
                        "analysis": {},
                    }
                    questions = "1. Welchen Bereich betrifft Ihre Anfrage?\n2. Gibt es spezifische Vorschriften, die relevant sind?\n3. Welche Art von Informationen suchen Sie?"

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
                session.waiting_for_human_input = False

                # Analyze accumulated feedback
                if session.hitl_state:
                    with st.spinner("Generiere Suchanfragen..."):
                        try:
                            analysis = hitl_service.analyse_user_feedback(session.hitl_state)
                            session.hitl_state["analysis"] = analysis

                            # Generate final research queries
                            result = hitl_service.generate_knowledge_base_questions(
                                session.hitl_state,
                                max_queries=session.max_search_queries,
                            )
                        except Exception as e:
                            st.error(f"LLM-Verbindung fehlgeschlagen: {e}")
                            # Use fallback result
                            result = {
                                "research_queries": [session.hitl_state.get("user_query", "")],
                                "summary": "Fallback: Verwende urspruengliche Anfrage",
                            }

                    # Add final message
                    summary = f"Starte Recherche mit {len(result.get('research_queries', []))} Suchanfragen:\n\n"
                    for i, q in enumerate(result.get("research_queries", []), 1):
                        summary += f"{i}. {q}\n"

                    add_hitl_message("assistant", summary)

                    return result

            else:
                # Add user feedback to history
                add_hitl_message("user", feedback)
                if session.hitl_state:
                    session.hitl_state["conversation_history"].append({
                        "role": "user",
                        "content": feedback,
                    })

                    # Get language before LLM calls
                    language = session.hitl_state.get("language", "de")

                    with st.spinner("Verarbeite Antwort..."):
                        try:
                            # Analyze feedback incrementally
                            analysis = hitl_service.analyse_user_feedback(session.hitl_state)
                            session.hitl_state["analysis"] = analysis

                            # Perform retrieval with refined query or feedback
                            refined = analysis.get("refined_query", feedback)
                            _perform_hitl_retrieval(refined or feedback, session)

                            # Generate follow-up questions if we haven't had many exchanges
                            if session.input_counter < 4:
                                questions = hitl_service.generate_follow_up_questions(
                                    session.hitl_state,
                                    language=language,
                                )

                                # Format combined response with analysis and questions
                                formatted_analysis = format_analysis_dict(analysis)
                                if formatted_analysis:
                                    combined_msg = f"**ANALYSE:**\n{formatted_analysis}\n\n**NACHFRAGEN:**\n{questions}"
                                else:
                                    combined_msg = questions

                                add_hitl_message("assistant", combined_msg)
                                session.hitl_state["conversation_history"].append({
                                    "role": "assistant",
                                    "content": combined_msg,
                                })
                            else:
                                # Suggest ending after several exchanges
                                if language == "de":
                                    msg = "Wir haben genug Kontext gesammelt. Tippen Sie `/end` um die Recherche zu starten, oder fuegen Sie weitere Details hinzu."
                                else:
                                    msg = "We have collected enough context. Type `/end` to start the research, or add more details."
                                add_hitl_message("assistant", msg)
                                session.hitl_state["conversation_history"].append({
                                    "role": "assistant",
                                    "content": msg,
                                })
                        except Exception as e:
                            st.error(f"LLM-Verbindung fehlgeschlagen: {e}")
                            # Use fallback message
                            if language == "de":
                                msg = "Ich konnte Ihre Antwort nicht verarbeiten. Bitte versuchen Sie es erneut oder tippen Sie `/end` um fortzufahren."
                            else:
                                msg = "Could not process your response. Please try again or type `/end` to continue."
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
        Dict with research_queries, analysis, user_query, and additional fields
    """
    hitl_service = _get_hitl_service()

    result = hitl_service.generate_knowledge_base_questions(
        hitl_state,
        max_queries=5,
    )

    analysis = hitl_state.get("analysis", {})

    return {
        "research_queries": result.get("research_queries", []),
        "summary": result.get("summary", ""),
        "user_query": hitl_state.get("user_query", ""),
        "analysis": analysis,
        # Additional fields for Phase 2 handoff
        "language": hitl_state.get("language", "de"),
        "conversation_history": hitl_state.get("conversation_history", []),
        "entities": analysis.get("entities", []),
        "scope": analysis.get("scope", ""),
        "context": analysis.get("context", ""),
    }


def render_hitl_summary() -> None:
    """Render HITL results summary in an expander.

    Shows after HITL phase completes: original query, research queries, entities.
    Enhanced with visual prominence for final context.
    """
    session = get_session_state()

    hitl_result = session.hitl_result
    if not hitl_result:
        return

    with st.expander("HITL Zusammenfassung - Finaler Kontext", expanded=False):
        with st.container(border=True):
            # Original query
            user_query = hitl_result.get("user_query", "")
            if user_query:
                st.markdown("**Urspruengliche Anfrage:**")
                st.write(user_query)

            # Analysis entities - display as inline code tags
            analysis = hitl_result.get("analysis", {})
            entities = analysis.get("entities", [])
            if entities:
                st.markdown("**Erkannte Entitaeten:**")
                entity_tags = " ".join([f"`{e}`" for e in entities])
                st.markdown(entity_tags)

            # Scope
            scope = analysis.get("scope", "")
            if scope:
                st.markdown("**Umfang:**")
                st.write(scope)

            # Context
            context = analysis.get("context", "")
            if context:
                st.markdown("**Kontext:**")
                st.caption(context)

        # Research queries - numbered list outside the bordered container
        research_queries = hitl_result.get("research_queries", [])
        if research_queries:
            st.markdown("**Forschungsabfragen:**")
            for i, q in enumerate(research_queries, 1):
                st.markdown(f"{i}. {q}")
