"""Progress status component for research phases."""

import streamlit as st

from src.ui.state import get_current_phase, get_session_state

# Phase messages: (short_label, description)
PHASE_MESSAGES = {
    "hitl_init": ("HITL Initialisierung...", "Starte interaktive Klaerung"),
    "hitl_generate_queries": ("Generiere Suchbegriffe...", "Erstelle Suchbegriffe fuer Retrieval"),
    "hitl_generate_questions": ("Warte auf Antwort...", "Benutzer-Feedback erforderlich"),
    "generate_todo": ("Erstelle Aufgabenliste...", "Plane Forschungsschritte"),
    "hitl_approve_todo": ("Warte auf Genehmigung...", "Aufgabenliste pruefen"),
    "execute_tasks": ("Fuehre Recherche durch...", "Durchsuche Wissensdatenbank"),
    "synthesize": ("Synthesisiere Ergebnisse...", "Erstelle Zusammenfassung"),
    "quality_check": ("Pruefe Qualitaet...", "Validiere Ergebnisse"),
    "attribute_sources": ("Fuege Quellen hinzu...", "Generiere Zitationen"),
    "complete": ("Abgeschlossen", "Recherche erfolgreich beendet"),
}


def get_phase_message(phase: str) -> tuple[str, str]:
    """Get human-readable phase messages.

    Args:
        phase: The current phase identifier

    Returns:
        Tuple of (short_label, description)
    """
    return PHASE_MESSAGES.get(phase, (phase, ""))


def render_phase_indicator(phase: str) -> None:
    """Render icon and text for current phase.

    Args:
        phase: The current phase identifier
    """
    phase_icons = {
        "hitl_init": ":speech_balloon:",
        "hitl_generate_queries": ":mag:",
        "hitl_generate_questions": ":speech_balloon:",
        "generate_todo": ":clipboard:",
        "hitl_approve_todo": ":raised_hand:",
        "execute_tasks": ":books:",
        "synthesize": ":memo:",
        "quality_check": ":white_check_mark:",
        "attribute_sources": ":link:",
        "complete": ":sparkles:",
    }

    icon = phase_icons.get(phase, ":hourglass_flowing_sand:")
    label, description = get_phase_message(phase)

    st.markdown(f"{icon} **{label}**")
    if description:
        st.caption(description)


def render_research_status() -> None:
    """Render research progress with st.status().

    Shows current phase with spinner and phase-specific messages.
    """
    session = get_session_state()
    phase = get_current_phase()

    if phase in ["idle", "complete"]:
        return

    label, description = get_phase_message(phase)

    # Use st.status for expanded progress view
    with st.status(label, expanded=True) as status:
        st.write(description)

        # Show additional context based on phase
        if phase == "execute_tasks":
            todo_list = session.agent_state.get("todo_list", [])
            if todo_list:
                completed = sum(1 for t in todo_list if t.get("completed"))
                total = len(todo_list)
                st.write(f"Fortschritt: {completed}/{total} Aufgaben")

        elif phase == "synthesize":
            summaries = session.agent_state.get("task_summaries", [])
            st.write(f"Verarbeite {len(summaries)} Ergebnisse")

        elif phase == "quality_check":
            st.write("Pruefe Vollstaendigkeit und Relevanz")

        # Update status state based on phase
        if phase == "complete":
            status.update(label="Abgeschlossen", state="complete", expanded=False)
