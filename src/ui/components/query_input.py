"""Query input component."""

import streamlit as st

from src.ui.state import get_session_state, reset_session_state


def render_query_input() -> str | None:
    """Render the query input form.

    Returns:
        Query string if submitted, None otherwise
    """
    st.subheader("Research Query")

    session = get_session_state()

    # Show current query if exists
    if session.current_query and session.query_submitted:
        st.info(f"Current query: {session.current_query}")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("New Query", type="secondary"):
                reset_session_state()
                st.rerun()

        return None

    # Query input form
    with st.form("query_form", clear_on_submit=False):
        query = st.text_area(
            "Enter your research question",
            placeholder="z.B. Was sind die Grenzwerte für berufliche Strahlenexposition?",
            height=100,
            key="query_input",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Start Research", type="primary")

        if submitted and query.strip():
            session.current_query = query.strip()
            session.query_submitted = True
            return query.strip()

    # Example queries
    with st.expander("Example queries"):
        examples = [
            "Was sind die Grenzwerte für berufliche Strahlenexposition?",
            "Welche Anforderungen stellt die StrlSchV an Strahlenschutzbeauftragte?",
            "Wie funktioniert die Einlagerung im Endlager Konrad?",
            "Was sind die Brandschutzanforderungen für die Schachtanlage?",
        ]

        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                session.current_query = example
                session.query_submitted = True
                st.rerun()

    return None
