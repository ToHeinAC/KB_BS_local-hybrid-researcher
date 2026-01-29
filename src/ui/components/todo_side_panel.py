"""ToDo side panel component for research phase.

Displays progress bar, spinner with current task, and compact task list
in a 1/3 column during research.
"""

import streamlit as st

from src.ui.state import get_current_phase, get_session_state, get_todo_list

# Phase labels and descriptions for side panel
PHASE_INFO = {
    "hitl_init": ("HITL Initialisierung", "Starte interaktive Klaerung"),
    "hitl_generate_queries": ("Generiere Suchbegriffe", "Erstelle Suchbegriffe"),
    "hitl_generate_questions": ("Warte auf Antwort", "Benutzer-Feedback"),
    "generate_todo": ("Erstelle Aufgaben", "Plane Forschungsschritte"),
    "hitl_approve_todo": ("Warte auf Genehmigung", "Pruefen Sie die Aufgaben"),
    "execute_tasks": ("Fuehre Recherche durch", "Durchsuche Wissensdatenbank"),
    "synthesize": ("Synthesisiere Ergebnisse", "Erstelle Zusammenfassung"),
    "quality_check": ("Pruefe Qualitaet", "Validiere Ergebnisse"),
    "attribute_sources": ("Fuege Quellen hinzu", "Generiere Zitationen"),
    "complete": ("Abgeschlossen", "Recherche beendet"),
}


def render_todo_side_panel() -> None:
    """Render side panel with progress, spinner, and task list."""
    session = get_session_state()
    todo_list = get_todo_list()
    phase = get_current_phase()

    with st.container(border=True):
        st.markdown("### Forschungsfortschritt")

        # Overall progress bar
        if todo_list:
            completed_count = sum(1 for item in todo_list if item.get("completed"))
            total = len(todo_list)
            progress = completed_count / total if total > 0 else 0
            st.progress(progress, text=f"{completed_count}/{total} Aufgaben")

        # Phase indicator
        phase_label, phase_desc = PHASE_INFO.get(phase, (phase, ""))
        if phase == "complete":
            st.success(f"**{phase_label}**")
        else:
            st.info(f"**{phase_label}**")
        st.caption(phase_desc)

        # Current task with spinner effect
        current_task_id = (
            session.agent_state.get("current_task_id")
            if session.agent_state
            else None
        )
        if current_task_id and todo_list:
            current_task = next(
                (t for t in todo_list if t.get("id") == current_task_id), None
            )
            if current_task:
                task_text = current_task.get("task", "")[:50]
                with st.spinner(f"Bearbeite: {task_text}..."):
                    # Spinner shows while this block is active
                    # Since we're just displaying, it will show briefly
                    pass

        st.divider()

        # Compact task list
        if todo_list:
            st.markdown("**Aufgaben:**")
            for item in todo_list:
                task_id = item.get("id")
                task = item.get("task", "")[:40]
                completed = item.get("completed", False)

                if completed:
                    icon = ":white_check_mark:"
                elif task_id == current_task_id:
                    icon = ":hourglass_flowing_sand:"
                else:
                    icon = ":clipboard:"

                st.markdown(f"{icon} **{task_id}**: {task}...")
        else:
            st.caption("Keine Aufgaben vorhanden")
