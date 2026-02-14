"""Tool definitions for the research agent."""

import difflib
import json
import logging
import re
from collections import Counter
from pathlib import Path

from src.config import settings
from src.models.research import (
    ChunkWithInfo,
    DetectedReference,
    ExtractedReferenceList,
    InfoExtractionWithQuotes,
    NestedChunk,
    PreservedQuote,
)
from src.models.results import VectorResult
from src.prompts import (
    INFO_EXTRACTION_PROMPT,
    INFO_EXTRACTION_WITH_QUOTES_PROMPT,
    REFERENCE_EXTRACTION_PROMPT,
)
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
    selected_database: str | None = None,
) -> list[VectorResult]:
    """Search vector database for relevant chunks.

    Args:
        query: Search query text
        collections: Collections to search (defaults to all)
        top_k: Number of results per collection
        selected_database: Specific database directory name to search

    Returns:
        List of VectorResult objects sorted by relevance
    """
    client = get_chromadb_client()
    top_k = top_k or settings.m_chunks_per_query

    # If a specific database is selected, use it directly
    if selected_database:
        return client.search_by_database_name(query, selected_database, top_k)

    if collections:
        return client.search_multi_collection(query, collections, top_k)
    return client.search_multi_collection(query, top_k=top_k)


def extract_info(
    chunk_text: str,
    query: str,
    language: str = "de",
) -> str:
    """Extract relevant information from chunk relative to query.

    Args:
        chunk_text: The chunk text to extract from
        query: The search query for context
        language: Target language for extraction ('de' or 'en')

    Returns:
        Extracted relevant information
    """
    client = get_ollama_client()
    lang_label = "German" if language == "de" else "English"

    prompt = INFO_EXTRACTION_PROMPT.format(query=query, chunk_text=chunk_text, language=lang_label)

    try:
        return client.generate(prompt)
    except Exception as e:
        logger.warning(f"Failed to extract info: {e}")
        # Return truncated original if extraction fails
        return chunk_text[:500]


def extract_info_with_quotes(
    chunk_text: str,
    query: str,
    query_anchor: dict,
    source_doc: str = "",
    page: int = 0,
    language: str = "de",
) -> dict:
    """Extract info while preserving critical verbatim quotes.

    Args:
        chunk_text: The chunk text to extract from
        query: The search query for context
        query_anchor: Query anchor with key_entities
        source_doc: Source document name for quote attribution
        page: Page number for quote attribution
        language: Target language for extraction ('de' or 'en')

    Returns:
        Dict with "extracted_info" and "preserved_quotes"
    """
    client = get_ollama_client()
    key_entities = query_anchor.get("key_entities", [])
    lang_label = "German" if language == "de" else "English"

    prompt = INFO_EXTRACTION_WITH_QUOTES_PROMPT.format(
        query=query,
        key_entities=", ".join(key_entities) if key_entities else "none specified",
        chunk_text=chunk_text[:3000],  # Limit input
        language=lang_label,
    )

    try:
        result = client.generate_structured(prompt, InfoExtractionWithQuotes)

        # Add source attribution to quotes
        for quote in result.preserved_quotes:
            quote.source = source_doc
            quote.page = page

        return result.model_dump()
    except Exception as e:
        logger.warning(f"Failed to extract info with quotes: {e}")
        # Fallback to simple extraction
        return {
            "extracted_info": extract_info(chunk_text, query, language=language),
            "preserved_quotes": [],
        }


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


def get_context_window(text: str, mention: str, window_size: int = 400) -> str:
    """Extract a focused window of text around a mention.

    Args:
        text: The full text context
        mention: The verbatim mention to center on
        window_size: Total target size of the window

    Returns:
        A snippet of text containing the mention with surrounding context
    """
    if not text or not mention:
        return text[:window_size] if text else ""

    try:
        idx = text.lower().find(mention.lower())
        if idx == -1:
            return text[:window_size]

        half_window = window_size // 2
        start = max(0, idx - half_window)
        end = min(len(text), idx + len(mention) + half_window)

        # Try to expand to sentence boundaries if possible
        snippet = text[start:end]
        
        # Add ellipsis if truncated
        result = ""
        if start > 0:
            result += "..."
        result += snippet
        if end < len(text):
            result += "..."
            
        return result
    except Exception:
        return text[:window_size]


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
    min_results: int = 0,
) -> list[ChunkWithInfo]:
    """Filter chunks by relevance threshold, guaranteeing min_results.

    Args:
        chunks: Chunks to filter
        query: Original query for relevance scoring
        threshold: Minimum relevance score (defaults to settings)
        min_results: Minimum number of chunks to return. If threshold
            filtering yields fewer, backfill with top-scoring rejected chunks.

    Returns:
        Filtered list of chunks (at least min_results if enough input chunks)
    """
    threshold = threshold or settings.reference_relevance_threshold

    filtered = []
    rejected = []
    for chunk in chunks:
        score = chunk.relevance_score or score_relevance(chunk.chunk, query)
        chunk.relevance_score = score

        if score >= threshold:
            filtered.append(chunk)
        else:
            rejected.append(chunk)

    if len(filtered) < min_results and rejected:
        rejected.sort(key=lambda c: c.relevance_score or 0.0, reverse=True)
        needed = min_results - len(filtered)
        filtered.extend(rejected[:needed])

    return filtered


def create_chunk_with_info(
    result: VectorResult,
    query: str,
    extract: bool = True,
    query_anchor: dict | None = None,
    preserve_quotes: bool = False,
    language: str = "de",
) -> tuple[ChunkWithInfo, list[dict]]:
    """Create ChunkWithInfo from VectorResult, optionally with preserved quotes.

    Args:
        result: Vector search result
        query: Original query
        extract: Whether to extract info using LLM
        query_anchor: Query anchor for quote extraction (required if preserve_quotes=True)
        preserve_quotes: Whether to extract and preserve verbatim quotes
        language: Target language for extraction ('de' or 'en')

    Returns:
        Tuple of (ChunkWithInfo instance, list of preserved quote dicts)
    """
    extracted = None
    preserved_quotes = []

    if extract:
        if preserve_quotes and query_anchor:
            # Use enhanced extraction with quote preservation
            extraction_result = extract_info_with_quotes(
                chunk_text=result.chunk_text,
                query=query,
                query_anchor=query_anchor,
                source_doc=result.doc_name,
                page=result.page_number or 0,
                language=language,
            )
            extracted = extraction_result.get("extracted_info", "")
            preserved_quotes = extraction_result.get("preserved_quotes", [])
        else:
            # Simple extraction
            extracted = extract_info(result.chunk_text, query, language=language)

    chunk = ChunkWithInfo(
        chunk=result.chunk_text,
        document=result.doc_name,
        page=result.page_number,
        extracted_info=extracted,
        relevance_score=result.relevance_score,
    )

    return chunk, preserved_quotes


# =============================================================================
# Graded Context Classification (Phase B)
# =============================================================================


def classify_context_tier(
    chunk: ChunkWithInfo,
    query_anchor: dict,
    depth: int = 0,
    source_type: str = "vector_search",
) -> tuple[int, float]:
    """Classify chunk into context tier with weight.

    Tier 1 (Primary): Direct search, high relevance, matches query entities
    Tier 2 (Secondary): Depth-1 references or medium relevance
    Tier 3 (Tertiary): Depth-2+ or HITL retrieval

    Args:
        chunk: The chunk to classify
        query_anchor: Immutable query reference with key_entities
        depth: Current recursion depth (0 = direct search)
        source_type: How this chunk was obtained ("vector_search", "reference", "hitl")

    Returns:
        (tier, weight): 1-3 tier and 0.0-1.0 weight
    """
    relevance = chunk.relevance_score or 0.0
    key_entities = query_anchor.get("key_entities", [])

    # Check for entity match (boosts tier 1 eligibility)
    entity_match = False
    chunk_text_lower = chunk.chunk.lower()
    for entity in key_entities:
        if entity.lower() in chunk_text_lower:
            entity_match = True
            break

    # Tier 1: Direct search, high relevance (≥0.85), or entity match
    if (source_type == "vector_search" and depth == 0):
        if relevance >= 0.85 or entity_match:
            weight = 1.0 if entity_match else 0.95
            return (1, weight)
        elif relevance >= 0.7:
            # High-medium relevance direct search still tier 1
            return (1, 0.85 * relevance)

    # Tier 2: Depth-1 references or medium relevance (0.6-0.85)
    if depth == 1 or (0.6 <= relevance < 0.85 and source_type == "vector_search"):
        weight = 0.7 * relevance
        return (2, max(weight, 0.42))  # Minimum 0.42 for tier 2

    # Tier 3: Depth-2+, HITL retrieval, or lower relevance
    if source_type == "hitl":
        weight = 0.5  # HITL retrieval has fixed weight
    else:
        weight = 0.4 * relevance

    return (3, max(weight, 0.2))  # Minimum 0.2 for tier 3


def create_tiered_context_entry(
    chunk: ChunkWithInfo,
    tier: int,
    weight: float,
    depth: int = 0,
    source_type: str = "vector_search",
    task_id: int | None = None,
) -> dict:
    """Create a context entry dict with tier metadata.

    Args:
        chunk: The source chunk
        tier: Context tier (1, 2, or 3)
        weight: Context weight (0.0-1.0)
        depth: Recursion depth when found
        source_type: How this chunk was obtained
        task_id: Optional task ID for per-task filtering in UI

    Returns:
        Dict representation for tiered context storage
    """
    entry = {
        "chunk": chunk.chunk[:2000],  # Limit chunk size
        "document": chunk.document,
        "page": chunk.page,
        "extracted_info": chunk.extracted_info,
        "relevance_score": chunk.relevance_score,
        "context_tier": tier,
        "context_weight": weight,
        "depth": depth,
        "source_type": source_type,
    }
    if task_id is not None:
        entry["task_id"] = task_id
    return entry


# =============================================================================
# Document Registry
# =============================================================================

_document_registry: dict | None = None


def load_document_registry() -> dict:
    """Load document registry from JSON file (singleton).

    Returns:
        Registry dict with 'collections' key, or empty dict on error.
    """
    global _document_registry
    if _document_registry is not None:
        return _document_registry

    registry_path = Path(settings.document_registry_path)
    if not registry_path.exists():
        logger.warning(f"Document registry not found: {registry_path}")
        _document_registry = {"collections": {}}
        return _document_registry

    try:
        with open(registry_path) as f:
            _document_registry = json.load(f)
        logger.info(f"Loaded document registry from {registry_path}")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load document registry: {e}")
        _document_registry = {"collections": {}}

    return _document_registry


def resolve_document_name(
    doc_ref: str,
    collection_hint: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve a document reference to (filename, collection_key).

    Uses 3-stage greedy matching:
    1. Exact synonym match (case-insensitive)
    2. Fuzzy match via SequenceMatcher (threshold 0.7)
    3. Substring match on filename

    Args:
        doc_ref: Document reference text to resolve
        collection_hint: Optional collection to search first

    Returns:
        (filename, collection_key) or (None, None) if unresolved
    """
    registry = load_document_registry()
    collections = registry.get("collections", {})
    if not collections:
        return None, None

    doc_ref_lower = doc_ref.lower().strip()

    # Order collections: hint first if given
    ordered_keys = list(collections.keys())
    if collection_hint and collection_hint in ordered_keys:
        ordered_keys.remove(collection_hint)
        ordered_keys.insert(0, collection_hint)

    # Stage 1: Exact synonym match (case-insensitive)
    for coll_key in ordered_keys:
        for doc in collections[coll_key].get("documents", []):
            for synonym in doc.get("synonyms", []):
                if synonym.lower() == doc_ref_lower:
                    return doc["filename"], coll_key

    # Stage 2: Fuzzy match (threshold 0.7)
    best_score = 0.0
    best_match: tuple[str | None, str | None] = (None, None)
    for coll_key in ordered_keys:
        for doc in collections[coll_key].get("documents", []):
            for synonym in doc.get("synonyms", []):
                ratio = difflib.SequenceMatcher(
                    None, doc_ref_lower, synonym.lower()
                ).ratio()
                if ratio > best_score and ratio >= 0.7:
                    best_score = ratio
                    best_match = (doc["filename"], coll_key)

    if best_match[0] is not None:
        return best_match

    # Stage 3: Substring match on filename
    for coll_key in ordered_keys:
        for doc in collections[coll_key].get("documents", []):
            fname_lower = doc["filename"].lower()
            if doc_ref_lower in fname_lower or fname_lower.startswith(doc_ref_lower):
                return doc["filename"], coll_key

    logger.debug(f"Could not resolve document reference: {doc_ref}")
    return None, None


# =============================================================================
# LLM Reference Extraction
# =============================================================================

# Mapping from LLM reference types to DetectedReference types
_LLM_TYPE_MAP = {
    "legal_section": "legal_section",
    "academic_numbered": "academic_numbered",
    "academic_shortform": "academic_shortform",
    "document_mention": "document_mention",
}


def extract_references_llm(text: str) -> list[DetectedReference]:
    """Extract references using LLM structured output.

    Args:
        text: Text to extract references from

    Returns:
        List of DetectedReference objects with extraction_method="llm"
    """
    client = get_ollama_client()
    prompt = REFERENCE_EXTRACTION_PROMPT.format(text=text[:3000])  # Limit input

    try:
        result = client.generate_structured(prompt, ExtractedReferenceList)
    except Exception as e:
        logger.warning(f"LLM reference extraction failed: {e}")
        return []

    refs = []
    for ext_ref in result.references:
        ref_type = _LLM_TYPE_MAP.get(ext_ref.reference_type, ext_ref.reference_type)
        doc_context = ext_ref.target_document_hint or None

        refs.append(
            DetectedReference(
                type=ref_type,
                target=ext_ref.reference_mention,
                original_text=ext_ref.reference_mention,
                extraction_method="llm",
                document_context=doc_context,
            )
        )

    logger.debug(f"LLM extracted {len(refs)} references")
    return refs


# =============================================================================
# Hybrid Detection
# =============================================================================


def detect_references_hybrid(text: str) -> list[DetectedReference]:
    """Detect references using both regex and LLM, deduplicated.

    Runs regex first (fast), then LLM (thorough), deduplicates by type:target key.

    Args:
        text: Text to search for references

    Returns:
        Deduplicated list of detected references
    """
    regex_refs = detect_references(text)
    llm_refs = extract_references_llm(text)

    # Build seen set from regex results
    seen = set()
    combined = []
    for ref in regex_refs:
        key = f"{ref.type}:{ref.target}".lower()
        seen.add(key)
        combined.append(ref)

    # Add LLM refs that aren't duplicates
    for ref in llm_refs:
        key = f"{ref.type}:{ref.target}".lower()
        if key not in seen:
            # Also check if the target text appears in any existing ref
            is_dup = any(
                ref.target.lower() in existing.target.lower()
                or existing.target.lower() in ref.target.lower()
                for existing in combined
            )
            if not is_dup:
                seen.add(key)
                combined.append(ref)

    logger.debug(
        f"Hybrid detection: {len(regex_refs)} regex + {len(llm_refs)} llm "
        f"= {len(combined)} unique"
    )
    return combined


# =============================================================================
# Enhanced Resolution
# =============================================================================


def resolve_reference_enhanced(
    ref: DetectedReference,
    current_doc: str,
    visited: set[str] | None = None,
    depth: int = 0,
    token_count: int = 0,
) -> list[NestedChunk]:
    """Resolve a reference with scoped search when document is known.

    Routes by reference type:
    - legal_section/section: resolve doc via registry -> scoped search
    - document/document_mention: registry lookup -> scoped search
    - academic_*: broad vector search with citation query
    - external: not resolved

    Args:
        ref: The reference to resolve
        current_doc: Current document context
        visited: Set of visited reference keys
        depth: Current recursion depth
        token_count: Running token budget usage

    Returns:
        List of chunks from resolved reference
    """
    visited = visited or set()

    if depth >= settings.reference_follow_depth:
        return []

    if token_count >= settings.reference_token_budget:
        logger.debug("Token budget exhausted for reference following")
        return []

    ref_key = f"{ref.type}:{ref.target}"
    if ref_key in visited:
        return []
    visited.add(ref_key)

    if ref.type in ("legal_section", "section"):
        return _resolve_legal_ref_enhanced(ref, current_doc)
    elif ref.type in ("document", "document_mention"):
        return _resolve_document_ref_enhanced(ref)
    elif ref.type in ("academic_numbered", "academic_shortform"):
        return _resolve_academic_ref(ref)
    else:
        return []


def _resolve_legal_ref_enhanced(
    ref: DetectedReference,
    current_doc: str,
) -> list[NestedChunk]:
    """Resolve a legal section reference with scoped search."""
    # Try to resolve the target document from hint or text
    doc_hint = ref.document_context or ""
    if not doc_hint:
        # Extract document name from target text (e.g., "§ 133 StrlSchG" -> "StrlSchG")
        parts = ref.target.split()
        for part in parts:
            if len(part) > 2 and not part.startswith("§") and not part.isdigit():
                doc_hint = part.strip(".,;:()")
                break

    if doc_hint:
        filename, collection_key = resolve_document_name(doc_hint)
        if filename and collection_key:
            return _vector_search_scoped(
                f"§ {ref.target}", 
                filename, 
                collection_key,
                center_mention=ref.original_text,
                window_size=5000
            )

    # Fallback to broad search
    return _resolve_section_ref(ref, current_doc)


def _resolve_document_ref_enhanced(ref: DetectedReference) -> list[NestedChunk]:
    """Resolve a document mention with registry-based scoping."""
    doc_hint = ref.document_context or ref.target
    filename, collection_key = resolve_document_name(doc_hint)

    if filename and collection_key:
        return _vector_search_scoped(
            ref.target, 
            filename, 
            collection_key,
            center_mention=ref.original_text,
            window_size=5000
        )

    # Fallback to broad search
    return _resolve_document_ref(ref)


def _resolve_academic_ref(ref: DetectedReference) -> list[NestedChunk]:
    """Resolve academic citation via broad vector search."""
    query = ref.target
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


def _vector_search_scoped(
    query: str,
    document_filename: str,
    collection_key: str,
    top_k: int = 5,
) -> list[NestedChunk]:
    """Search within a specific collection, post-filtered by document name.

    Args:
        query: Search query
        document_filename: Target document filename to filter by
        collection_key: Collection to search in
        top_k: Number of results to retrieve (before filtering)
        center_mention: Optional text to center the context window on
        window_size: Size of the context window (default 3000)

    Returns:
        Filtered list of NestedChunk objects
    """
    client = get_chromadb_client()
    try:
        results = client.search(query, collection_key, top_k=top_k)
    except Exception as e:
        logger.warning(f"Scoped search failed for {collection_key}: {e}")
        return []

    # Post-filter by document name
    doc_base = document_filename.lower().replace(".pdf", "")
    chunks = []
    for result in results:
        result_doc = result.doc_name.lower()
        if (
            doc_base in result_doc
            or result_doc in doc_base
            or document_filename.lower() in result_doc
        ):
            if result.relevance_score >= settings.reference_relevance_threshold:
                chunk_text = result.chunk_text
                
                # Apply windowing if center_mention is provided
                if center_mention:
                    chunk_text = get_context_window(
                        chunk_text, 
                        center_mention, 
                        window_size=window_size
                    )

                chunks.append(
                    NestedChunk(
                        chunk=chunk_text,
                        document=result.doc_name,
                        relevance_score=result.relevance_score,
                    )
                )

    logger.debug(
        f"Scoped search for '{query}' in {collection_key}/{document_filename}: "
        f"{len(chunks)} results"
    )
    return chunks


# =============================================================================
# Convergence Detection
# =============================================================================


def detect_convergence(doc_history: list[str]) -> bool:
    """Check if any document appears >= convergence threshold times.

    Indicates the rabbithole is circling back to the same documents.

    Args:
        doc_history: List of document names encountered during traversal

    Returns:
        True if convergence detected
    """
    if not doc_history:
        return False

    counts = Counter(doc_history)
    threshold = settings.convergence_same_doc_threshold
    return any(count >= threshold for count in counts.values())
