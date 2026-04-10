"""Safe exit button component."""

import gc
import logging

import requests
import streamlit as st

from src.config import settings

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_ollama_client():
    """Return cached OllamaClient instance."""
    from src.services.ollama_client import OllamaClient
    return OllamaClient()


@st.cache_resource
def _get_chromadb_client():
    """Return cached ChromaDBClient instance."""
    from src.services.chromadb_client import ChromaDBClient
    return ChromaDBClient()


def render_safe_exit() -> None:
    """Render the safe exit button in the sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("Session Control")

    if st.sidebar.button(
        "Free GPU & Reset",
        type="secondary",
        help="Unload models from GPU and reset session. Server stays running.",
    ):
        _perform_safe_exit()


def _perform_safe_exit() -> None:
    """Free GPU VRAM and reset session without stopping the server.

    Steps:
    1. Unload Ollama model from GPU via keep_alive=0 API call
    2. Clear all @st.cache_resource caches (releases Python refs to embedding model)
    3. Run GC + torch.cuda.empty_cache() to release CUDA memory
    4. Reset OllamaClient module singletons
    5. Reset Streamlit session state
    6. Rerun to fresh state (server stays alive)
    """
    st.sidebar.info("Freeing GPU memory...")

    # Step 1: Unload Ollama model from VRAM
    _unload_ollama_model()

    # Step 2: Clear all cached resource functions
    _clear_all_caches()

    # Step 3: Release CUDA allocator cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    except ImportError:
        pass

    gc.collect()

    # Step 4: Reset module-level OllamaClient singletons
    try:
        from src.agents import nodes as _nodes_mod
        from src.agents import tools as _tools_mod
        from src.services import hitl_service as _hitl_mod
        _nodes_mod.reset_ollama_client()
        _tools_mod.reset_ollama_client()
        _hitl_mod.reset_ollama_client()
    except Exception as e:
        logger.warning(f"Could not reset OllamaClient singletons: {e}")

    # Step 5: Reset session state
    from src.ui.state import reset_session_state
    from src.ui.components.gpu_widget import reset_research_timer
    reset_session_state()
    reset_research_timer()

    st.sidebar.success("GPU freed. Session reset. Server still running.")
    st.rerun()


def _unload_ollama_model() -> None:
    """Tell Ollama to evict the current model from VRAM (keep_alive=0)."""
    try:
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {"model": settings.ollama_model, "keep_alive": 0}
        requests.post(url, json=payload, timeout=5)
        logger.info(f"Requested Ollama to unload model: {settings.ollama_model}")
    except Exception as e:
        logger.warning(f"Ollama model unload request failed (non-critical): {e}")


def _clear_all_caches() -> None:
    """Clear all @st.cache_resource caches to release GPU-holding Python objects."""
    # Local caches in this module
    _get_ollama_client.clear()
    _get_chromadb_client.clear()

    # ChromaDB client in app.py (holds the HuggingFace embedding model)
    try:
        from src.ui.app import get_chromadb_client
        get_chromadb_client.clear()
    except Exception as e:
        logger.warning(f"Could not clear app.py chromadb cache: {e}")

    # HITL service cache
    try:
        from src.ui.components.hitl_panel import _get_hitl_service
        _get_hitl_service.clear()
    except Exception as e:
        logger.warning(f"Could not clear hitl_service cache: {e}")

    logger.info("All @st.cache_resource caches cleared")


def render_connection_status() -> None:
    """Render connection status indicators in sidebar."""
    st.sidebar.subheader("System Status")

    # Check Ollama (using cached client)
    try:
        client = _get_ollama_client()
        if client.is_available():
            st.sidebar.success("Ollama: Connected")
        else:
            st.sidebar.error("Ollama: Not Available")
    except Exception as e:
        st.sidebar.error(f"Ollama: Error - {e}")

    # Check ChromaDB (using cached client)
    try:
        client = _get_chromadb_client()
        collections = client.list_available_collections()
        st.sidebar.success(f"ChromaDB: {len(collections)} collections")
    except Exception as e:
        st.sidebar.error(f"ChromaDB: Error - {e}")
