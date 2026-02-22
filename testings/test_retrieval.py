"""
Retrieval Quality Testing GUI

A standalone Streamlit application for evaluating retrieval quality across
different similarity metrics and database configurations.

Run:
    uv run streamlit run testings/test_retrieval.py --server.port 8512
"""

import json
from datetime import datetime
from pathlib import Path

import chromadb
import streamlit as st

from src.config import settings
from src.models.research import VectorResult
from src.services.chromadb_client import ChromaDBClient

# Page configuration
st.set_page_config(
    page_title="Retrieval Quality Tester",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Session State Initialization
# ============================================================================


def init_session_state():
    """Initialize session state variables."""
    if "test_chromadb_client" not in st.session_state:
        st.session_state.test_chromadb_client = ChromaDBClient()

    if "selected_db" not in st.session_state:
        st.session_state.selected_db = None

    if "k_results" not in st.session_state:
        st.session_state.k_results = 4

    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = ["cosine"]

    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = None


# ============================================================================
# Multi-Metric Search Implementation
# ============================================================================


def search_with_metrics(
    query: str,
    db_name: str,
    top_k: int,
    metrics: list[str],
) -> dict[str, list[VectorResult]]:
    """Execute search with multiple similarity metrics.

    Args:
        query: Search query text
        db_name: Database directory name
        top_k: Number of results to return
        metrics: List of metric names ("cosine", "l2", "inner_product")

    Returns:
        Dictionary mapping metric name to list of VectorResult objects
    """
    client = st.session_state.test_chromadb_client

    # Determine database path
    db_path = client.base_path / db_name
    if not db_path.exists():
        raise ValueError(f"Database not found: {db_path}")

    # Try default subdirectory first (ChromaDB structure)
    actual_path = db_path / "default" if (db_path / "default").exists() else db_path

    # Get raw ChromaDB collection
    raw_client = chromadb.PersistentClient(path=str(actual_path))
    collections = raw_client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in database: {db_name}")

    collection_name = collections[0].name
    collection = raw_client.get_collection(collection_name)

    # Get embeddings using the same method as production code
    embedding_model = client._resolve_embedding_model(db_name)
    embeddings = client._get_embeddings(embedding_model)

    # Generate query embedding
    query_embedding = embeddings.embed_query(query)

    results_by_metric = {}

    # Map to ChromaDB distance function names
    distance_map = {"cosine": "cosine", "l2": "l2", "inner_product": "ip"}

    for metric in metrics:
        # ChromaDB collections have a fixed distance function - we can't change it at query time
        # We'll need to use the collection's native distance and convert appropriately
        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to VectorResult objects
        vector_results = []
        for i in range(len(raw_results["ids"][0])):
            doc_id = raw_results["ids"][0][i]
            distance = raw_results["distances"][0][i]
            document = raw_results["documents"][0][i]
            metadata = raw_results["metadatas"][0][i]

            # Convert distance to similarity score based on metric
            # NOTE: ChromaDB collection has a fixed distance function (usually L2)
            # We display the raw distance value for now
            if metric == "cosine":
                # For L2 distance with normalized vectors: cos_sim ≈ 1 - (L2² / 2)
                similarity = max(0.0, min(1.0, 1.0 - (distance**2 / 2.0)))
            elif metric == "l2":
                # L2 distance - lower is better, convert to similarity for consistency
                similarity = max(0.0, min(1.0, 1.0 - (distance**2 / 2.0)))
            elif metric == "inner_product":
                # If vectors are normalized, IP ≈ cosine similarity
                # For now, use the same conversion as cosine
                similarity = max(0.0, min(1.0, 1.0 - (distance**2 / 2.0)))
            else:
                similarity = distance

            vector_results.append(
                VectorResult(
                    doc_id=doc_id,
                    doc_name=metadata.get("source", "unknown"),
                    chunk_text=document,
                    page_number=metadata.get("page"),
                    relevance_score=similarity,
                    collection=collection_name,
                    query_used=query,
                )
            )

        # Sort by relevance (descending)
        vector_results.sort(key=lambda x: x.relevance_score, reverse=True)
        results_by_metric[metric] = vector_results

    return results_by_metric


# ============================================================================
# Database Explorer Functions
# ============================================================================


def get_all_chunks_from_db(db_name: str, limit: int = 1000) -> list[dict]:
    """Retrieve all chunks from a database for browsing.

    Args:
        db_name: Database directory name
        limit: Maximum number of chunks to retrieve

    Returns:
        List of chunk dictionaries with id, text, and metadata
    """
    client = st.session_state.test_chromadb_client

    # Determine database path
    db_path = client.base_path / db_name
    if not db_path.exists():
        raise ValueError(f"Database not found: {db_path}")

    # Try default subdirectory first
    actual_path = db_path / "default" if (db_path / "default").exists() else db_path

    # Get raw ChromaDB collection
    raw_client = chromadb.PersistentClient(path=str(actual_path))
    collections = raw_client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in database: {db_name}")

    collection_name = collections[0].name
    collection = raw_client.get_collection(collection_name)

    # Get all documents (ChromaDB get() without filters)
    all_data = collection.get(include=["documents", "metadatas"], limit=limit)

    chunks = []
    for i in range(len(all_data["ids"])):
        chunks.append(
            {
                "id": all_data["ids"][i],
                "text": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            }
        )

    return chunks


# ============================================================================
# Export Functions
# ============================================================================


def export_chunk_to_text(chunk: dict) -> str:
    """Export a chunk with metadata to formatted text.

    Args:
        chunk: Chunk dictionary with id, text, and metadata

    Returns:
        Formatted text content
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    content = f"""Chunk Export
================
Timestamp: {timestamp}
ID: {chunk['id']}
Source: {chunk['metadata'].get('source', 'N/A')}
Page: {chunk['metadata'].get('page', 'N/A')}
Collection: {chunk['metadata'].get('collection', 'N/A')}

Metadata:
{json.dumps(chunk['metadata'], indent=2)}

Text:
{chunk['text']}
"""
    return content


# ============================================================================
# UI Rendering Functions
# ============================================================================


def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("Configuration")

    # Database Selection
    st.sidebar.subheader("Database Selection")
    client = st.session_state.test_chromadb_client

    try:
        available_dbs = client.list_database_directories()
        if not available_dbs:
            st.sidebar.warning("No databases found in kb/database/")
            return False

        selected_db = st.sidebar.selectbox(
            "Select Database",
            options=available_dbs,
            index=(
                available_dbs.index(st.session_state.selected_db)
                if st.session_state.selected_db in available_dbs
                else 0
            ),
        )

        st.session_state.selected_db = selected_db

        # Display database info
        if selected_db:
            st.sidebar.info(f"**Database:** {selected_db}")

            try:
                embedding_model = client.extract_embedding_model(selected_db)
                st.sidebar.info(f"**Embedding Model:** {embedding_model}")
            except Exception as e:
                st.sidebar.warning(f"Could not extract embedding model: {e}")

    except Exception as e:
        st.sidebar.error(f"Error loading databases: {e}")
        return False

    # Retrieval Configuration
    st.sidebar.subheader("Retrieval Configuration")

    st.session_state.k_results = st.sidebar.slider(
        "Number of Results (k)",
        min_value=1,
        max_value=20,
        value=st.session_state.k_results,
        help="Number of top results to retrieve per query",
    )

    st.session_state.selected_metrics = st.sidebar.multiselect(
        "Similarity Metrics",
        options=["cosine", "l2", "inner_product"],
        default=st.session_state.selected_metrics,
        help="Note: All metrics currently use the collection's native distance function (L2)",
    )

    return True


def render_search_test_tab():
    """Render the Search Test tab."""
    st.header("Search Test")

    if not st.session_state.selected_db:
        st.warning("Please select a database from the sidebar")
        return

    if not st.session_state.selected_metrics:
        st.warning("Please select at least one similarity metric from the sidebar")
        return

    # Information about distance functions
    st.info(
        """
        **Note on Similarity Metrics**: ChromaDB collections have a fixed distance function
        (usually L2). This tool uses the collection's native distance and converts it to
        similarity scores using the formula: `similarity = 1 - (distance² / 2)` for
        normalized embeddings. True multi-metric comparison would require creating separate
        collections with different distance functions.
        """
    )

    # Search interface
    query = st.text_input(
        "Search Query",
        placeholder="Enter your search query...",
        help="Enter a query to search the selected database",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("Execute Search", type="primary")

    if search_button and query:
        with st.spinner("Executing search..."):
            try:
                results = search_with_metrics(
                    query=query,
                    db_name=st.session_state.selected_db,
                    top_k=st.session_state.k_results,
                    metrics=st.session_state.selected_metrics,
                )
                st.session_state.search_results = results
                st.success(f"Found {len(results[st.session_state.selected_metrics[0]])} results")
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(e)

    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader("Search Results")

        # Create tabs for each metric
        metric_tabs = st.tabs([m.upper() for m in st.session_state.selected_metrics])

        for i, metric in enumerate(st.session_state.selected_metrics):
            with metric_tabs[i]:
                results = st.session_state.search_results.get(metric, [])

                st.write(f"**Top {len(results)} Results - {metric.upper()} Similarity**")

                for rank, result in enumerate(results, 1):
                    with st.expander(
                        f"#{rank} | Score: {result.relevance_score:.3f} | "
                        f"{result.doc_name} (p.{result.page_number or 'N/A'})",
                        expanded=(rank == 1),
                    ):
                        # Display full chunk text
                        st.markdown("**Chunk Text:**")
                        st.text_area(
                            "Content",
                            value=result.chunk_text,
                            height=200,
                            key=f"{metric}_{rank}_text",
                            disabled=True,
                        )

                        # Metadata
                        st.markdown("**Metadata:**")
                        cols = st.columns(4)
                        cols[0].metric("Document", result.doc_name)
                        cols[1].metric("Page", result.page_number or "N/A")
                        cols[2].metric("Collection", result.collection)
                        cols[3].metric("Score", f"{result.relevance_score:.4f}")


def render_database_explorer_tab():
    """Render the Database Explorer tab."""
    st.header("Database Explorer")

    if not st.session_state.selected_db:
        st.warning("Please select a database from the sidebar")
        return

    # Load chunks button
    if st.button("Load Chunks from Database", type="primary"):
        with st.spinner("Loading chunks..."):
            try:
                chunks = get_all_chunks_from_db(
                    st.session_state.selected_db, limit=1000
                )
                st.session_state.all_chunks = chunks
                st.success(f"Loaded {len(chunks)} chunks")
            except Exception as e:
                st.error(f"Failed to load chunks: {e}")
                st.exception(e)

    if st.session_state.all_chunks:
        chunks = st.session_state.all_chunks
        st.info(f"**Total chunks in collection:** {len(chunks)}")

        # Pagination controls
        col1, col2 = st.columns([1, 3])
        with col1:
            page_size = st.selectbox("Chunks per page", [10, 25, 50, 100], index=1)
        with col2:
            total_pages = (len(chunks) - 1) // page_size + 1
            page_num = st.number_input(
                "Page", min_value=1, max_value=total_pages, value=1
            )

        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(chunks))

        st.write(f"Showing chunks {start_idx + 1}-{end_idx} of {len(chunks)}")

        # Display chunks
        for chunk in chunks[start_idx:end_idx]:
            with st.expander(
                f"ID: {chunk['id']} | "
                f"{chunk['metadata'].get('source', 'N/A')} "
                f"p.{chunk['metadata'].get('page', 'N/A')}"
            ):
                st.text_area(
                    "Text",
                    chunk["text"],
                    height=150,
                    disabled=True,
                    key=f"chunk_text_{chunk['id']}",
                )
                st.json(chunk["metadata"])

                # Export button
                export_text = export_chunk_to_text(chunk)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chunk_export_{chunk['id']}_{timestamp}.txt"

                st.download_button(
                    label="📥 Export This Chunk",
                    data=export_text,
                    file_name=filename,
                    mime="text/plain",
                    key=f"export_{chunk['id']}",
                )


def render_chunk_export_tab():
    """Render the Chunk Export tab."""
    st.header("Chunk Export")

    st.markdown(
        """
    This tab provides batch export functionality for chunks.

    **Current Features:**
    - Export individual chunks from the Database Explorer tab

    **Planned Features:**
    - Batch export of search results
    - Export with custom filtering
    - Multiple format options (JSON, CSV, Markdown)
    """
    )

    if st.session_state.search_results:
        st.subheader("Export Search Results")

        # Select metric
        metric = st.selectbox(
            "Select Metric",
            options=st.session_state.selected_metrics,
        )

        results = st.session_state.search_results.get(metric, [])

        if st.button("Export All Results to JSON"):
            # Convert VectorResult objects to dicts
            results_dicts = [
                {
                    "doc_id": r.doc_id,
                    "doc_name": r.doc_name,
                    "chunk_text": r.chunk_text,
                    "page_number": r.page_number,
                    "relevance_score": r.relevance_score,
                    "collection": r.collection,
                    "query_used": r.query_used,
                }
                for r in results
            ]

            export_data = {
                "metric": metric,
                "query": results[0].query_used if results else "",
                "timestamp": datetime.now().isoformat(),
                "results_count": len(results),
                "results": results_dicts,
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{metric}_{timestamp}.json"

            st.download_button(
                label="📥 Download JSON Export",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=filename,
                mime="application/json",
            )


# ============================================================================
# Main Application
# ============================================================================


def main():
    """Main application entry point."""
    init_session_state()

    st.title("🔍 Retrieval Quality Testing Tool")
    st.markdown(
        """
    Test and evaluate retrieval quality across different similarity metrics and configurations.
    """
    )

    # Render sidebar
    if not render_sidebar():
        st.error("Failed to initialize. Please check your database configuration.")
        return

    # Main content area - tabs
    tab1, tab2, tab3 = st.tabs(["🔎 Search Test", "📚 Database Explorer", "📥 Export"])

    with tab1:
        render_search_test_tab()

    with tab2:
        render_database_explorer_tab()

    with tab3:
        render_chunk_export_tab()


if __name__ == "__main__":
    main()
