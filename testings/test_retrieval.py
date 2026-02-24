"""
Retrieval Quality Testing GUI

A standalone Streamlit application for evaluating retrieval quality across
different similarity metrics and database configurations.

Run:
    uv run streamlit run testings/test_retrieval.py --server.port 8512
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import chromadb
import streamlit as st

from src.config import settings
from src.models.query import TaskSearchQueries
from src.models.research import VectorResult
from src.models.results import (
    RerankerBatchOutput,
    RerankerRecallBatchOutput,
    TaskSummaryOutput,
)
from src.prompts import (
    RERANKER_PRECISION_PROMPT_HUMAN,
    RERANKER_PRECISION_PROMPT_SYSTEM,
    RERANKER_RECALL_PROMPT_HUMAN,
    RERANKER_RECALL_PROMPT_SYSTEM,
    TASK_SEARCH_QUERIES_PROMPT_HUMAN,
    TASK_SEARCH_QUERIES_PROMPT_SYSTEM,
    TASK_SUMMARY_PROMPT_HUMAN,
    TASK_SUMMARY_PROMPT_SYSTEM,
)
from src.services.chromadb_client import ChromaDBClient
from src.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


RERANKER_MODELS = [
    "qwen3:1.7b",
    "qwen3:8b",
    "ministral-3:8b",
    "qwen3:14b",
    "ministral-3:14b",
    "gpt-oss:20b",
]


@st.cache_resource
def _get_ollama_client(model: str = settings.ollama_model) -> OllamaClient:
    """Cached OllamaClient — one instance per model name."""
    return OllamaClient(model=model)

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
        st.session_state.k_results = 10

    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = ["cosine"]

    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = None

    if "additional_context" not in st.session_state:
        st.session_state.additional_context = ""

    if "language" not in st.session_state:
        st.session_state.language = "German"

    if "reranker_enabled" not in st.session_state:
        st.session_state.reranker_enabled = False

    if "reranker_strategy" not in st.session_state:
        st.session_state.reranker_strategy = "precision"

    if "reranked_results" not in st.session_state:
        st.session_state.reranked_results = None

    if "show_reranked" not in st.session_state:
        st.session_state.show_reranked = True

    if "reranker_model" not in st.session_state:
        st.session_state.reranker_model = "qwen3:14b"

    if "export_data" not in st.session_state:
        st.session_state.export_data = None

    if "synthesis_result" not in st.session_state:
        st.session_state.synthesis_result = None

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    if "last_additional_context" not in st.session_state:
        st.session_state.last_additional_context = ""


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
                    doc_name=metadata.get("original_filename", metadata.get("source", "unknown")),
                    chunk_text=document,
                    page_number=metadata.get("page"),
                    chunk_id=metadata.get("chunk_id"),
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

        prev_selected = st.session_state.selected_db

        selected_db = st.sidebar.selectbox(
            "Select Database",
            options=available_dbs,
            index=(
                available_dbs.index(st.session_state.selected_db)
                if st.session_state.selected_db in available_dbs
                else 0
            ),
        )

        if prev_selected != selected_db:
            st.session_state.search_results = None
            st.session_state.reranked_results = None
            st.session_state.synthesis_result = None
            st.session_state.all_chunks = None

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

    # Language selector
    st.sidebar.subheader("Language")
    st.session_state.language = st.sidebar.selectbox(
        "Output Language",
        options=["German", "English"],
        index=0 if st.session_state.language == "German" else 1,
        help="Language for LLM-generated search queries",
    )

    # LLM Reranker
    st.sidebar.subheader("LLM Reranker")
    st.session_state.reranker_enabled = st.sidebar.checkbox(
        "Enable LLM Reranking",
        value=st.session_state.reranker_enabled,
        help="Re-score retrieved chunks using an LLM (pointwise, 5-point scale)",
    )

    if st.session_state.reranker_enabled:
        st.session_state.reranker_strategy = st.sidebar.radio(
            "Strategy",
            options=["precision", "recall"],
            index=0 if st.session_state.reranker_strategy == "precision" else 1,
            help="Precision: strict relevance, drops tangential. Recall: broader retention.",
        )
        if st.session_state.reranker_strategy == "precision":
            st.sidebar.info(
                "**Precision**: 5-point scale, anti-tangential bias. "
                "Mentioning a topic without answering the question = score 2 or lower."
            )
        else:
            st.sidebar.info(
                "**Recall**: 4-dimension composite (answerability, depth, novelty, specificity). "
                "When in doubt, scores higher to retain more chunks."
            )

        st.session_state.reranker_model = st.sidebar.selectbox(
            "Reranker Model",
            options=RERANKER_MODELS,
            index=(
                RERANKER_MODELS.index(st.session_state.reranker_model)
                if st.session_state.reranker_model in RERANKER_MODELS
                else 3
            ),
            help="Ollama model used for LLM reranking",
        )

    return True


def _generate_additional_queries(query: str, additional_context: str) -> list[str]:
    """Generate additional search queries using LLM when additional context is provided.

    Args:
        query: Original search query
        additional_context: HITL-style context for query generation

    Returns:
        List of 2 queries: [query_1, query_2] where query_1 == original (per prompt rule 1). Falls back to [original] on error.
    """
    lang_label = st.session_state.language
    ollama = _get_ollama_client()

    system_prompt = TASK_SEARCH_QUERIES_PROMPT_SYSTEM.format(language=lang_label)
    human_prompt = TASK_SEARCH_QUERIES_PROMPT_HUMAN.format(
        task=query,
        original_query=query,
        hitl_context=additional_context,
        key_entities="None",
        language=lang_label,
    )

    result = ollama.generate_structured_messages(
        system_prompt, human_prompt, TaskSearchQueries
    )
    return [result.query_1, result.query_2]


# ============================================================================
# LLM Reranker Functions
# ============================================================================


def _build_reranker_batches(
    chunks: list[VectorResult], batch_size: int = 6
) -> list[list[tuple[int, VectorResult]]]:
    """Round-robin distribute chunks into batches.

    Args:
        chunks: Flat list of VectorResult
        batch_size: Max chunks per batch (default 6)

    Returns:
        List of batches, each batch is a list of (global_index, VectorResult)
    """
    if not chunks:
        return []
    n_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)
    batches: list[list[tuple[int, VectorResult]]] = [[] for _ in range(n_batches)]
    for idx, chunk in enumerate(chunks):
        batches[idx % n_batches].append((idx, chunk))
    return batches


def _format_chunks_for_prompt(batch: list[tuple[int, VectorResult]]) -> str:
    """Format a batch of chunks for the reranker HUMAN prompt."""
    lines = []
    for global_idx, result in batch:
        text_preview = result.chunk_text[:1500]
        page_display = result.page_number if result.page_number is not None else "N/A"
        lines.append(
            f"[Chunk {global_idx}]\n"
            f"Document: {result.doc_name}, Page: {page_display}\n"
            f"Text: {text_preview}\n"
        )
    return "\n".join(lines)


def _rerank_batch_precision(
    batch: list[tuple[int, VectorResult]],
    query: str,
    additional_context: str,
    language: str,
    model: str = settings.ollama_model,
) -> list[dict]:
    """Rerank a batch using precision strategy.

    Returns list of {id, score, reason, original_result} dicts.
    Falls back to vector-score-based 1-5 mapping on error.
    """
    ollama = _get_ollama_client(model)
    chunks_text = _format_chunks_for_prompt(batch)

    system_prompt = RERANKER_PRECISION_PROMPT_SYSTEM.format(language=language)
    human_prompt = RERANKER_PRECISION_PROMPT_HUMAN.format(
        query=query,
        additional_context=additional_context or "N/A",
        chunks=chunks_text,
        language=language,
    )

    try:
        result = ollama.generate_structured_messages(
            system_prompt, human_prompt, RerankerBatchOutput
        )
        # Build lookup by id
        score_map = {r.id: (r.score, r.reason) for r in result.results}
    except Exception as e:
        logger.warning(f"Precision reranker failed: {e}, using fallback scores")
        score_map = {}

    output = []
    for global_idx, vr in batch:
        if global_idx in score_map:
            score, reason = score_map[global_idx]
        else:
            # Fallback: map cosine score (0-1) to 1-5
            score = max(1, min(5, round(vr.relevance_score * 5)))
            reason = "fallback from vector score"
        output.append({
            "id": global_idx,
            "score": score,
            "reason": reason,
            "original_result": vr,
        })
    return output


def _rerank_batch_recall(
    batch: list[tuple[int, VectorResult]],
    query: str,
    additional_context: str,
    known_info: str,
    language: str,
    model: str = settings.ollama_model,
) -> list[dict]:
    """Rerank a batch using recall strategy.

    Returns list of {id, score, reason, original_result} dicts.
    """
    ollama = _get_ollama_client(model)
    chunks_text = _format_chunks_for_prompt(batch)

    system_prompt = RERANKER_RECALL_PROMPT_SYSTEM.format(language=language)
    human_prompt = RERANKER_RECALL_PROMPT_HUMAN.format(
        query=query,
        known_info=known_info or "N/A",
        additional_context=additional_context or "N/A",
        chunks=chunks_text,
        language=language,
    )

    try:
        result = ollama.generate_structured_messages(
            system_prompt, human_prompt, RerankerRecallBatchOutput
        )
        score_map = {r.id: (r.s, r.r) for r in result.results}
    except Exception as e:
        logger.warning(f"Recall reranker failed: {e}, using fallback scores")
        score_map = {}

    output = []
    for global_idx, vr in batch:
        if global_idx in score_map:
            score, reason = score_map[global_idx]
        else:
            score = max(1, min(5, round(vr.relevance_score * 5)))
            reason = "fallback from vector score"
        output.append({
            "id": global_idx,
            "score": score,
            "reason": reason,
            "original_result": vr,
        })
    return output


def _normalize_batch_scores(
    batch_results: list[list[dict]],
) -> list[dict]:
    """Per-batch zero-mean normalization, merge and sort descending.

    Each entry has: {id, score, reason, original_result, normalized_score}
    """
    all_items = []
    for batch in batch_results:
        if not batch:
            continue
        batch_mean = sum(item["score"] for item in batch) / len(batch)
        for item in batch:
            item["normalized_score"] = item["score"] - batch_mean
            all_items.append(item)

    all_items.sort(key=lambda x: x["normalized_score"], reverse=True)
    return all_items


def _rerank_results(
    grouped_results: dict[str, list[tuple[str, list[VectorResult]]]],
    query: str,
    additional_context: str,
    language: str,
    strategy: str,
) -> dict[str, list[tuple[str, list[dict]]]]:
    """Orchestrate reranking for all metrics and query groups.

    Returns same structure as grouped_results but with reranked dicts
    instead of VectorResult objects. Each dict has:
    {id, score, reason, original_result, normalized_score}
    """
    reranked: dict[str, list[tuple[str, list[dict]]]] = {}

    for metric, query_groups in grouped_results.items():
        reranked_groups = []
        for q_text, results in query_groups:
            if not results:
                reranked_groups.append((q_text, []))
                continue

            batches = _build_reranker_batches(results, batch_size=6)
            batch_results = []
            model = st.session_state.get("reranker_model", settings.ollama_model)

            for batch in batches:
                if strategy == "precision":
                    scored = _rerank_batch_precision(
                        batch, query, additional_context, language, model
                    )
                else:
                    scored = _rerank_batch_recall(
                        batch, query, additional_context, "", language, model
                    )
                batch_results.append(scored)

            normalized = _normalize_batch_scores(batch_results)
            reranked_groups.append((q_text, normalized))

        reranked[metric] = reranked_groups

    return reranked


# ============================================================================
# Synthesis Helpers
# ============================================================================


def _build_ranked_findings() -> tuple[str, int, int]:
    """Collect reranked chunks scoring >= 4, deduplicate, and format for TASK_SUMMARY_PROMPT.

    Returns:
        (ranked_findings_str, count_5, count_4) where counts are deduplicated chunk counts.
    """
    reranked: dict = st.session_state.reranked_results or {}

    # Deduplicate by (doc_name, page_number, chunk_id); keep highest score entry
    seen: dict[tuple, dict] = {}
    for metric_groups in reranked.values():
        for _q_text, items in metric_groups:
            for item in items:
                if item["score"] < 4:
                    continue
                vr = item["original_result"]
                key = (vr.doc_name, vr.page_number, vr.chunk_id)
                if key not in seen or item["score"] > seen[key]["score"]:
                    seen[key] = item

    # Sort: 5/5 first, then 4/5; within same score by normalized_score desc
    top = sorted(seen.values(), key=lambda x: (-x["score"], -x.get("normalized_score", 0.0)))

    count_5 = sum(1 for x in top if x["score"] == 5)
    count_4 = sum(1 for x in top if x["score"] == 4)

    # Score mapping: 5/5 → 90/100, 4/5 → 70/100
    score_map = {5: 90, 4: 70}

    lines = []
    for rank, item in enumerate(top, 1):
        vr = item["original_result"]
        page_str = str(vr.page_number) if vr.page_number is not None else "N/A"
        chunk_str = f" chunk {vr.chunk_id}" if vr.chunk_id is not None else ""
        source_ref = f"[{vr.doc_name} ({chunk_str.strip() or 'N/A'}), Page {page_str}]"
        score_100 = score_map.get(item["score"], 60)
        reason = (item.get("reason") or "").strip() or "LLM scored highly relevant"
        text = vr.chunk_text[:1500]
        lines.append(f"Rank {rank} | Score {score_100}/100 | {source_ref} | {reason} | {text}")

    return "\n".join(lines), count_5, count_4


def _synthesize_from_reranked(query: str, additional_context: str, language: str) -> TaskSummaryOutput:
    """Run TASK_SUMMARY_PROMPT on top-rated reranked chunks."""
    ranked_findings, _, _ = _build_ranked_findings()
    ollama = _get_ollama_client(st.session_state.reranker_model)
    system_prompt = TASK_SUMMARY_PROMPT_SYSTEM.format(language=language)
    human_prompt = TASK_SUMMARY_PROMPT_HUMAN.format(
        task="",
        original_query=query,
        hitl_smry=additional_context or "N/A",
        ranked_findings=ranked_findings,
        preserved_quotes="[]",
        language=language,
    )
    return ollama.generate_structured_messages(system_prompt, human_prompt, TaskSummaryOutput)


# ============================================================================
# Chunk Rendering
# ============================================================================


def _render_chunk_expander(
    result: VectorResult, rank: int, metric: str, query_idx: int = 0,
    rerank_score: int | None = None, rerank_reason: str | None = None,
):
    """Render a single chunk result as an expander.

    Args:
        result: VectorResult to display
        rank: 1-based rank within this query group
        metric: Metric name for unique key generation
        query_idx: Query index for unique key generation (0 for single-query mode)
        rerank_score: Optional LLM reranker score (1-5)
        rerank_reason: Optional one-sentence reason from reranker
    """
    # Build title with optional reranker badge
    page_str = result.page_number if result.page_number is not None else "N/A"
    chunk_str = f" chunk {result.chunk_id}" if result.chunk_id is not None else ""
    title = (
        f"#{rank} | Score: {result.relevance_score:.3f} | "
        f"{result.doc_name} (p.{page_str}{chunk_str})"
    )
    if rerank_score is not None:
        title = f"#{rank} | [LLM: {rerank_score}/5] | Vec: {result.relevance_score:.3f} | {result.doc_name} (p.{page_str}{chunk_str})"

    # Key suffix to distinguish original vs reranked view
    view_tag = "rr" if rerank_score is not None else "orig"

    with st.expander(title, expanded=(rank == 1 and query_idx == 0)):
        # Show reranker info if present
        if rerank_score is not None:
            score_colors = {5: "green", 4: "blue", 3: "orange", 2: "red", 1: "red"}
            color = score_colors.get(rerank_score, "gray")
            st.markdown(
                f"**LLM Reranker:** :{color}[{rerank_score}/5] — {rerank_reason or 'N/A'}"
            )

        st.markdown("**Chunk Text:**")
        st.text_area(
            "Content",
            value=result.chunk_text,
            height=200,
            key=f"{view_tag}_{metric}_q{query_idx}_{rank}_text",
            disabled=True,
        )

        st.markdown("**Metadata:**")
        cols = st.columns(6)
        cols[0].metric("Document", result.doc_name)
        cols[1].metric("Page", result.page_number if result.page_number is not None else "N/A")
        cols[2].metric("Chunk ID", result.chunk_id if result.chunk_id is not None else "N/A")
        cols[3].metric("Collection", result.collection)
        cols[4].metric("Score", f"{result.relevance_score:.4f}")
        cols[5].metric("Query Used", result.query_used[:40] + "..." if len(result.query_used) > 40 else result.query_used)


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

    additional_context = st.text_area(
        "Additional Context (HITL)",
        placeholder="Optional context to refine search...",
        help=(
            "When set, the LLM generates 2 additional search queries "
            "(mirrors TASK_SEARCH_QUERIES_PROMPT in the research pipeline). "
            "Results are grouped by source query."
        ),
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("Execute Search", type="primary")

    if search_button and query:
        use_multi_query = bool(additional_context and additional_context.strip())

        # Generate queries
        all_queries = [query]
        if use_multi_query:
            with st.spinner("Generating additional queries via LLM..."):
                try:
                    all_queries = _generate_additional_queries(
                        query, additional_context.strip()
                    )
                    st.success(
                        f"Generated {len(all_queries)} queries: "
                        + " | ".join(f'"{q}"' for q in all_queries)
                    )
                except Exception as e:
                    st.warning(f"LLM query generation failed ({e}), using original query only")
                    all_queries = [query]

        # Execute search for each query
        with st.spinner("Executing search..."):
            try:
                # Results structure: metric -> list of (query_text, list[VectorResult])
                grouped_results: dict[str, list[tuple[str, list[VectorResult]]]] = {}
                for q in all_queries:
                    per_metric = search_with_metrics(
                        query=q,
                        db_name=st.session_state.selected_db,
                        top_k=st.session_state.k_results,
                        metrics=st.session_state.selected_metrics,
                    )
                    for metric, results in per_metric.items():
                        grouped_results.setdefault(metric, []).append((q, results))

                st.session_state.search_results = grouped_results
                st.session_state.reranked_results = None  # Clear previous reranking
                st.session_state.synthesis_result = None  # Clear previous synthesis
                st.session_state.last_query = query
                st.session_state.last_additional_context = additional_context.strip() if additional_context else ""
                first_metric = st.session_state.selected_metrics[0]
                total_chunks = sum(
                    len(r) for _, r in grouped_results.get(first_metric, [])
                )
                st.success(
                    f"Found {total_chunks} results across "
                    f"{len(all_queries)} {'queries' if len(all_queries) > 1 else 'query'}"
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(e)

        # LLM Reranking (after search)
        if (
            st.session_state.search_results
            and st.session_state.reranker_enabled
            and st.session_state.reranked_results is None
        ):
            strategy = st.session_state.reranker_strategy
            with st.spinner(f"Reranking with LLM ({strategy})..."):
                try:
                    reranked = _rerank_results(
                        st.session_state.search_results,
                        query,
                        additional_context.strip() if additional_context else "",
                        st.session_state.language,
                        strategy,
                    )
                    st.session_state.reranked_results = reranked
                    st.success(f"Reranking complete ({strategy} strategy)")
                except Exception as e:
                    st.error(f"Reranking failed: {e}")
                    logger.exception("Reranking failed")

    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.subheader("Search Results")

        # Toggle for original vs reranked view
        has_reranked = st.session_state.reranked_results is not None
        if has_reranked:
            st.session_state.show_reranked = st.toggle(
                "Show LLM-reranked order",
                value=st.session_state.show_reranked,
                help="Toggle between original vector similarity order and LLM-reranked order",
            )

        show_reranked = has_reranked and st.session_state.show_reranked
        results_data = (
            st.session_state.reranked_results
            if show_reranked
            else st.session_state.search_results
        )

        # Detect format: new grouped format (dict -> list[tuple]) vs legacy (dict -> list[VectorResult])
        first_metric = st.session_state.selected_metrics[0]
        first_val = results_data.get(first_metric, [])
        is_grouped = first_val and isinstance(first_val[0], tuple)

        metric_tabs = st.tabs([m.upper() for m in st.session_state.selected_metrics])

        for i, metric in enumerate(st.session_state.selected_metrics):
            with metric_tabs[i]:
                if is_grouped:
                    query_groups = results_data.get(metric, [])
                    multi_query = len(query_groups) > 1

                    for q_idx, (q_text, items) in enumerate(query_groups):
                        if multi_query:
                            st.markdown(
                                f"### Query {q_idx + 1}: {q_text}"
                            )

                        if show_reranked:
                            # Reranked view: items are dicts with score/reason/original_result
                            st.write(
                                f"**Top {len(items)} Results - LLM Reranked ({st.session_state.reranker_strategy})**"
                            )
                            for rank, item in enumerate(items, 1):
                                _render_chunk_expander(
                                    item["original_result"],
                                    rank,
                                    metric,
                                    q_idx,
                                    rerank_score=item["score"],
                                    rerank_reason=item["reason"],
                                )
                        else:
                            # Original view: items are VectorResult
                            st.write(
                                f"**Top {len(items)} Results - {metric.upper()} Similarity**"
                            )
                            for rank, result in enumerate(items, 1):
                                _render_chunk_expander(result, rank, metric, q_idx)

                        if multi_query and q_idx < len(query_groups) - 1:
                            st.markdown("---")
                else:
                    # Legacy flat format (backward compat)
                    results = results_data.get(metric, [])
                    st.write(
                        f"**Top {len(results)} Results - {metric.upper()} Similarity**"
                    )
                    for rank, result in enumerate(results, 1):
                        _render_chunk_expander(result, rank, metric)

        # ── Synthesize Report ──────────────────────────────────────────────
        if has_reranked:
            st.markdown("---")
            ranked_findings_str, count_5, count_4 = _build_ranked_findings()
            eligible = count_5 + count_4
            if eligible == 0:
                st.info("No chunks with LLM score 4 or 5 available for synthesis.")
            else:
                st.info(
                    f"**Synthesis pool:** {count_5} main resources (5/5) + "
                    f"{count_4} supplementing resources (4/5) = {eligible} chunks"
                )
                if st.button("Synthesize Report", type="primary", key="synthesize_btn"):
                    q = st.session_state.get("last_query", "")
                    ctx = st.session_state.get("last_additional_context", "")
                    with st.spinner("Synthesizing report..."):
                        try:
                            result = _synthesize_from_reranked(q, ctx, st.session_state.language)
                            st.session_state.synthesis_result = result
                        except Exception as e:
                            st.error(f"Synthesis failed: {e}")
                            logger.exception("Synthesis failed")

        if st.session_state.synthesis_result:
            result = st.session_state.synthesis_result
            st.markdown("### Research Report")
            st.markdown(f"**Relevance Score:** {result.relevance_score}/100")
            if result.relevance_assessment:
                st.markdown(f"*{result.relevance_assessment}*")
            st.markdown("**Summary:**")
            st.markdown(result.summary)
            if result.key_findings:
                st.markdown("**Key Findings:**")
                for f in result.key_findings:
                    st.markdown(f"- {f}")
            if result.gaps:
                with st.expander("Gaps"):
                    for g in result.gaps:
                        st.markdown(f"- {g}")
            if result.irrelevant_findings:
                with st.expander("Irrelevant Findings"):
                    for irr in result.irrelevant_findings:
                        st.markdown(f"- {irr}")


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

        raw_metric_data = st.session_state.search_results.get(metric, [])

        if st.button("Export All Results to JSON"):
            # Handle both grouped (new) and flat (legacy) formats
            is_grouped = raw_metric_data and isinstance(raw_metric_data[0], tuple)

            if is_grouped:
                all_results: list[VectorResult] = []
                query_groups_export = []
                for q_text, results_list in raw_metric_data:
                    all_results.extend(results_list)
                    query_groups_export.append({
                        "query": q_text,
                        "results_count": len(results_list),
                        "results": [
                            {
                                "doc_id": r.doc_id,
                                "doc_name": r.doc_name,
                                "chunk_text": r.chunk_text,
                                "page_number": r.page_number,
                                "relevance_score": r.relevance_score,
                                "collection": r.collection,
                                "query_used": r.query_used,
                            }
                            for r in results_list
                        ],
                    })
                export_data = {
                    "metric": metric,
                    "timestamp": datetime.now().isoformat(),
                    "total_results": len(all_results),
                    "query_groups": query_groups_export,
                }
            else:
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
                    for r in raw_metric_data
                ]
                export_data = {
                    "metric": metric,
                    "query": raw_metric_data[0].query_used if raw_metric_data else "",
                    "timestamp": datetime.now().isoformat(),
                    "results_count": len(raw_metric_data),
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

            st.session_state.export_data = export_data

    if st.session_state.export_data is not None:
        st.divider()
        st.subheader("Save to testings/ Directory")
        save_filename = st.text_input(
            "Filename",
            value="rerank.json",
            help="File will be saved inside the testings/ directory",
        )
        if st.button("Save to testings/"):
            save_path = Path(__file__).parent / save_filename
            try:
                save_path.write_text(
                    json.dumps(st.session_state.export_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                st.success(f"Saved to {save_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")


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
