"""ToDo side panel component for research phase.

Displays progress bar, spinner with current task, and compact task list
in a 1/3 column during research.
"""

import streamlit as st

from src.ui.state import get_current_phase, get_session_state, get_todo_list

# Phase labels and descriptions for side panel
PHASE_INFO = {
    "hitl_init": ("HITL Initialisierung", "Starte interaktive Klärung"),
    "hitl_generate_queries": ("Generiere Suchbegriffe", "Erstelle Suchbegriffe"),
    "hitl_generate_questions": ("Warte auf Antwort", "Benutzer-Feedback"),
    "generate_todo": ("Erstelle Aufgaben", "Plane Forschungsschritte"),
    "hitl_approve_todo": ("Warte auf Genehmigung", "Prüfen Sie die Aufgaben"),
    "execute_tasks": ("Führe Recherche durch", "Durchsuche Wissensdatenbank"),
    "synthesize": ("Synthesisiere Ergebnisse", "Erstelle Zusammenfassung"),
    "quality_check": ("Prüfe Qualität", "Validiere Ergebnisse"),
    "attribute_sources": ("Füge Quellen hinzu", "Generiere Zitationen"),
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

        # Current task with verbose spinner for execute_tasks phase
        current_task_id = (
            session.agent_state.get("current_task_id")
            if session.agent_state
            else None
        )

        if phase == "execute_tasks" and current_task_id is not None and todo_list:
            # Find current task and its 1-based position
            current_task = None
            position = 0
            for idx, t in enumerate(todo_list, 1):
                if t.get("id") == current_task_id:
                    current_task = t
                    position = idx
                    break

            if current_task:
                task_text = current_task.get("task", "")[:80]
                total = len(todo_list)
                with st.spinner(f"Aufgabe {position}/{total}: {task_text}"):
                    st.markdown(
                        f"*Durchsuche Wissensdatenbank für Aufgabe {position}...*"
                    )
        else:
            st.caption(phase_desc)

        st.divider()

        # Compact task list with sequential numbering
        if todo_list:
            st.markdown("**Aufgaben:**")
            for idx, item in enumerate(todo_list, 1):
                task = item.get("task", "")
                completed = item.get("completed", False)
                task_id = item.get("id")

                if completed:
                    icon = "\u2705"
                elif task_id == current_task_id:
                    icon = "\u23f3"
                else:
                    icon = "\U0001f4cb"

                short = task[:40] + "..." if len(task) > 40 else task
                is_current = task_id == current_task_id
                with st.expander(f"{icon} {idx}. {short}", expanded=is_current):
                    st.markdown(task)
        else:
            st.caption("Keine Aufgaben vorhanden")
