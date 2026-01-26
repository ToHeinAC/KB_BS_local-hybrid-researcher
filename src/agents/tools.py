"""Tool definitions for the research agent."""

import logging
import re

from src.config import settings
from src.models.research import ChunkWithInfo, DetectedReference, NestedChunk
from src.models.results import VectorResult
from src.services.chromadb_client import ChromaDBClient
from src.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Initialize clients
_chromadb_client: ChromaDBClient | None = None
_ollama_client: OllamaClient | None = None


def get_chromadb_client() -> ChromaDBClient:
    """Get or create ChromaDB client singleton."""
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = ChromaDBClient()
    return _chromadb_client


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def vector_search(
    query: str,
    collections: list[str] | None = None,
    top_k: int | None = None,
) -> list[VectorResult]:
    """Search vector database for relevant chunks.

    Args:
        query: Search query text
        collections: Collections to search (defaults to all)
        top_k: Number of results per collection

    Returns:
        List of VectorResult objects sorted by relevance
    """
    client = get_chromadb_client()
    top_k = top_k or settings.m_chunks_per_query

    if collections:
        return client.search_multi_collection(query, collections, top_k)
    return client.search_multi_collection(query, top_k=top_k)


def extract_info(
    chunk_text: str,
    query: str,
) -> str:
    """Extract relevant information from chunk relative to query.

    Args:
        chunk_text: The chunk text to extract from
        query: The search query for context

    Returns:
        Extracted relevant information
    """
    client = get_ollama_client()

    prompt = f"""Given this search query: "{query}"

Extract only the relevant passages from this text chunk. Be concise and focus on information that directly answers or relates to the query.

Text chunk:
{chunk_text}

Extracted relevant information (in the same language as the chunk):"""

    try:
        return client.generate(prompt)
    except Exception as e:
        logger.warning(f"Failed to extract info: {e}")
        # Return truncated original if extraction fails
        return chunk_text[:500]


# Reference detection patterns (German and English)
REFERENCE_PATTERNS = [
    # German section references
    (r"(?:siehe|gem[äa][ß]|nach|in)\s*§\s*(\d+(?:\s*Abs\.\s*\d+)?)", "section"),
    (r"§\s*(\d+(?:\s*Abs\.\s*\d+)?)\s*(?:StrlSchV|StrlSchG)", "section"),
    # German document references
    (r"(?:siehe|gemäß)\s+(?:Dokument|Unterlage)\s+[\"']?([^\"'\n,]+)[\"']?", "document"),
    (r"(?:EU|EG)\s*(\d+(?:\.\d+)?)", "document"),
    # English references
    (r"(?:see|cf\.)\s+section\s+(\d+(?:\.\d+)?)", "section"),
    (r"(?:see|refer to)\s+document\s+[\"']?([^\"'\n,]+)[\"']?", "document"),
    # External references
    (r"(https?://[^\s<>\"]+)", "external"),
]


def detect_references(text: str) -> list[DetectedReference]:
    """Detect references within text using regex patterns.

    Args:
        text: Text to search for references

    Returns:
        List of detected references
    """
    references = []
    seen_targets = set()

    for pattern, ref_type in REFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            target = match.group(1).strip()

            # Skip duplicates
            ref_key = f"{ref_type}:{target}"
            if ref_key in seen_targets:
                continue
            seen_targets.add(ref_key)

            references.append(
                DetectedReference(
                    type=ref_type,
                    target=target,
                    original_text=match.group(0),
                )
            )

    logger.debug(f"Detected {len(references)} references in text")
    return references


def resolve_reference(
    ref: DetectedReference,
    current_doc: str,
    visited: set[str] | None = None,
    depth: int = 0,
) -> list[NestedChunk]:
    """Resolve a reference and retrieve its content.

    Args:
        ref: The reference to resolve
        current_doc: Current document context
        visited: Set of visited reference keys
        depth: Current recursion depth

    Returns:
        List of chunks from resolved reference
    """
    visited = visited or set()

    # Check depth limit
    if depth >= settings.reference_follow_depth:
        logger.debug(f"Max depth reached for ref: {ref.target}")
        return []

    # Check if already visited
    ref_key = f"{ref.type}:{ref.target}"
    if ref_key in visited:
        logger.debug(f"Already visited ref: {ref_key}")
        return []

    visited.add(ref_key)

    # Resolve based on type
    if ref.type == "section":
        return _resolve_section_ref(ref, current_doc)
    elif ref.type == "document":
        return _resolve_document_ref(ref)
    else:
        # External references not resolved in baseline
        return []


def _resolve_section_ref(
    ref: DetectedReference,
    current_doc: str,
) -> list[NestedChunk]:
    """Resolve a section reference within documents."""
    # Search for the section number across all collections
    query = f"§ {ref.target}"
    results = vector_search(query, top_k=3)

    chunks = []
    for result in results:
        if result.relevance_score >= settings.reference_relevance_threshold:
            chunks.append(
                NestedChunk(
                    chunk=result.chunk_text,
                    document=result.doc_name,
                    relevance_score=result.relevance_score,
                )
            )

    return chunks


def _resolve_document_ref(ref: DetectedReference) -> list[NestedChunk]:
    """Resolve a document reference."""
    # Search for the document name
    query = ref.target
    results = vector_search(query, top_k=3)

    chunks = []
    for result in results:
        # Look for matching document name
        if (
            ref.target.lower() in result.doc_name.lower()
            and result.relevance_score >= settings.reference_relevance_threshold
        ):
            chunks.append(
                NestedChunk(
                    chunk=result.chunk_text,
                    document=result.doc_name,
                    relevance_score=result.relevance_score,
                )
            )

    return chunks


def score_relevance(
    chunk_text: str,
    query: str,
) -> float:
    """Score relevance of chunk to query (0.0 to 1.0).

    Args:
        chunk_text: Text to score
        query: Original query

    Returns:
        Relevance score between 0.0 and 1.0
    """
    # Simple keyword-based scoring for baseline
    query_words = set(query.lower().split())
    chunk_words = set(chunk_text.lower().split())

    if not query_words:
        return 0.0

    overlap = len(query_words & chunk_words)
    return min(1.0, overlap / len(query_words))


def filter_by_relevance(
    chunks: list[ChunkWithInfo],
    query: str,
    threshold: float | None = None,
) -> list[ChunkWithInfo]:
    """Filter chunks by relevance threshold.

    Args:
        chunks: Chunks to filter
        query: Original query for relevance scoring
        threshold: Minimum relevance score (defaults to settings)

    Returns:
        Filtered list of chunks
    """
    threshold = threshold or settings.reference_relevance_threshold

    filtered = []
    for chunk in chunks:
        # Use existing score or compute new one
        score = chunk.relevance_score or score_relevance(chunk.chunk, query)

        if score >= threshold:
            chunk.relevance_score = score
            filtered.append(chunk)

    return filtered


def create_chunk_with_info(
    result: VectorResult,
    query: str,
    extract: bool = True,
) -> ChunkWithInfo:
    """Create ChunkWithInfo from VectorResult.

    Args:
        result: Vector search result
        query: Original query
        extract: Whether to extract info using LLM

    Returns:
        ChunkWithInfo instance
    """
    extracted = None
    if extract:
        extracted = extract_info(result.chunk_text, query)

    return ChunkWithInfo(
        chunk=result.chunk_text,
        document=result.doc_name,
        page=result.page_number,
        extracted_info=extracted,
        relevance_score=result.relevance_score,
    )
