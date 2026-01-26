"""Safe exit button component."""

import subprocess
import sys

import streamlit as st

from src.config import settings


def render_safe_exit() -> None:
    """Render the safe exit button in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("Session Control")

    if st.sidebar.button("Safe Exit", type="secondary", help="Cleanly terminate the application"):
        _perform_safe_exit()


def _perform_safe_exit() -> None:
    """Perform safe exit by killing the Streamlit process on the current port."""
    port = settings.streamlit_port

    st.sidebar.warning(f"Terminating application on port {port}...")

    try:
        # Find and kill process on the port
        # Uses lsof to find the process and xargs to kill it
        # -r flag ensures xargs doesn't fail if no PIDs are found
        cmd = f"lsof -ti:{port} | xargs -r kill -9"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            st.sidebar.success("Application terminated")
        else:
            # Alternative: just exit Python
            st.sidebar.info("Exiting application...")
            st.stop()
            sys.exit(0)

    except Exception as e:
        st.sidebar.error(f"Error during exit: {e}")
        # Fallback: exit Python directly
        sys.exit(0)


def render_connection_status() -> None:
    """Render connection status indicators in sidebar."""
    st.sidebar.subheader("System Status")

    # Check Ollama
    try:
        from src.services.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            st.sidebar.success("Ollama: Connected")
        else:
            st.sidebar.error("Ollama: Not Available")
    except Exception as e:
        st.sidebar.error(f"Ollama: Error - {e}")

    # Check ChromaDB
    try:
        from src.services.chromadb_client import ChromaDBClient
        client = ChromaDBClient()
        collections = client.list_available_collections()
        st.sidebar.success(f"ChromaDB: {len(collections)} collections")
    except Exception as e:
        st.sidebar.error(f"ChromaDB: Error - {e}")
