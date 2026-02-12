"""Node functions for the LangGraph research agent."""

import json
import logging
from datetime import datetime

from pydantic import BaseModel

from src.agents.state import AgentState, get_next_task_id
from src.agents.tools import (
    classify_context_tier,
    create_chunk_with_info,
    create_tiered_context_entry,
    detect_convergence,
    detect_references,
    detect_references_hybrid,
    filter_by_relevance,
    resolve_reference,
    resolve_reference_enhanced,
    vector_search,
)
from src.config import settings
from src.models.hitl import HITLDecision
from src.models.query import QueryAnalysis, TaskSearchQueries, ToDoItem, ToDoList
from src.models.research import (
    ChunkWithInfo,
    ResearchContext,
    ResearchContextMetadata,
    SearchQueryResult,
)
from src.models.results import (
    FinalReport,
    Finding,
    LinkedSource,
    QualityAssessment,
    RelevanceScoreOutput,
    Source,
    SynthesisOutputEnhanced,
    TaskSummaryOutput,
)
from src.prompts import (
    HITL_SUMMARY_PROMPT,
    QUALITY_CHECK_PROMPT,
    RELEVANCE_SCORING_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_PROMPT_ENHANCED,
    TASK_SEARCH_QUERIES_PROMPT,
    TASK_SUMMARY_PROMPT,
    TODO_GENERATION_PROMPT,
)
from src.services.hitl_service import HITLService
from src.services.ollama_client import OllamaClient
from src.utils.debug_state import dump_state_markdown

logger = logging.getLogger(__name__)

# Service instances
_ollama_client: OllamaClient | None = None
_hitl_service: HITLService | None = None


def get_ollama_client() -> OllamaClient:
    """Get Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_hitl_service() -> HITLService:
    """Get HITL service singleton."""
    global _hitl_service
    if _hitl_service is None:
        _hitl_service = HITLService(max_questions=settings.max_clarification_questions)
    return _hitl_service


# --- Phase 2: ToDo List Generation ---


class ToDoListOutput(BaseModel):
    """LLM output for todo list generation."""

    items: list[dict]


def generate_todo(state: AgentState) -> dict:
    """Generate research task list based on query analysis.

    Prioritizes research_queries from HITL if available,
    otherwise falls back to LLM generation.

    Args:
        state: Current agent state

    Returns:
        State update with todo_list
    """
    # Dump state at Phase 1→2 boundary (fires whether HITL ran or was skipped)
    dump_state_markdown(state, {}, "tests/debugging/state_1hitl.md", "Phase 1: Pre-Todo")

    analysis = QueryAnalysis.model_validate(state["query_analysis"])

    # Check for research_queries from HITL first
    research_queries = state.get("research_queries", [])
    additional_context = state.get("additional_context", "")
    hitl_smry = state.get("hitl_smry", "")

    if research_queries:
        # Convert HITL research queries directly to ToDoItems
        # Prefer citation-aware hitl_smry over plain additional_context
        item_context = hitl_smry or additional_context or "From HITL conversation"
        items = [
            ToDoItem(
                id=i + 1,
                task=query,
                context=item_context if i == 0 else "From HITL conversation",
            )
            for i, query in enumerate(research_queries)
        ]
        logger.info(f"Using {len(items)} research queries from HITL")
    else:
        # Fallback to LLM generation
        client = get_ollama_client()

        lang_label = "German" if analysis.detected_language == "de" else "English"
        prompt = TODO_GENERATION_PROMPT.format(
            original_query=analysis.original_query,
            key_concepts=analysis.key_concepts,
            entities=analysis.entities,
            scope=analysis.scope,
            assumed_context=analysis.assumed_context,
            hitl_smry=hitl_smry or "No prior findings",
            num_items=settings.initial_todo_items,
            language=lang_label,
        )

        try:
            result = client.generate_structured(prompt, ToDoListOutput)
            items = [
                ToDoItem(
                    id=item.get("id", i + 1),
                    task=item.get("task", ""),
                    context=item.get("context", ""),
                )
                for i, item in enumerate(result.items)
            ]
        except Exception as e:
            logger.warning(f"ToDo generation failed: {e}")
            items = [
                ToDoItem(
                    id=1,
                    task=f"Research: {analysis.original_query}",
                    context="Fallback task from failed generation",
                )
            ]

    # Prepend original query as Task 0 for direct vector search
    task_zero = ToDoItem(
        id=0,
        task=analysis.original_query,
        context="Direct search for original user query",
    )
    items.insert(0, task_zero)

    todo_list = ToDoList(items=items, max_items=settings.todo_max_items)

    return {
        "todo_list": [item.model_dump() for item in todo_list.items],
        "phase": "hitl_approve_todo",
        "messages": [f"Generated {len(items)} research tasks"],
    }


def hitl_approve_todo(state: AgentState) -> dict:
    """Create HITL checkpoint for todo list approval.

    Args:
        state: Current agent state

    Returns:
        State update with HITL checkpoint
    """
    hitl_service = get_hitl_service()
    items = [ToDoItem.model_validate(item) for item in state.get("todo_list", [])]
    todo_list = ToDoList(items=items)

    checkpoint = hitl_service.create_todo_checkpoint(todo_list)

    return {
        "hitl_pending": True,
        "hitl_checkpoint": checkpoint.model_dump(),
        "messages": ["Awaiting user approval of research tasks"],
    }


def process_hitl_todo(state: AgentState) -> dict:
    """Process HITL todo approval response.

    Args:
        state: Current agent state with hitl_decision

    Returns:
        State update with approved todo list
    """
    hitl_service = get_hitl_service()
    items = [ToDoItem.model_validate(item) for item in state.get("todo_list", [])]
    todo_list = ToDoList(items=items)
    decision = HITLDecision.model_validate(state.get("hitl_decision", {}))

    if decision.approved:
        todo_list = hitl_service.apply_todo_modifications(todo_list, decision)
        # Renumber tasks sequentially after removals/additions
        for idx, item in enumerate(todo_list.items):
            item.id = idx

    return_dict = {
        "todo_list": [item.model_dump() for item in todo_list.items],
        "phase": "execute_tasks",
        "hitl_pending": False,
        "hitl_checkpoint": None,
        "hitl_decision": None,
        "current_task_id": get_next_task_id(state),
        "hitl_history": state.get("hitl_history", [])
        + [{"type": "todo_approve", "decision": decision.model_dump()}],
    }

    if settings.enable_state_dump:
        dump_state_markdown(state, return_dict, "tests/debugging/state_2todo.md", "Phase 2: ToDo Approved")

    return return_dict


# --- Phase 3: Task Execution ---


def execute_task(state: AgentState) -> dict:
    """Execute a single research task with graded context classification.

    Args:
        state: Current agent state

    Returns:
        State update with research results and tiered context
    """
    task_id = state.get("current_task_id")
    if task_id is None:
        return {"phase": "validate_relevance"}

    # Find current task
    todo_list = state.get("todo_list", [])
    current_task = None
    for item in todo_list:
        if item.get("id") == task_id:
            current_task = ToDoItem.model_validate(item)
            break

    if not current_task or current_task.completed:
        return {"phase": "validate_relevance"}

    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    context = ResearchContext.model_validate(
        state.get(
            "research_context",
            {"search_queries": [], "metadata": ResearchContextMetadata().model_dump()},
        )
    )

    # Get query anchor for context classification
    query_anchor = state.get("query_anchor", {
        "original_query": analysis.original_query,
        "key_entities": analysis.entities,
    })

    # Determine language for extraction and search queries
    language = query_anchor.get("detected_language", "de")
    lang_label = "German" if language == "de" else "English"

    # Generate 2 dedicated search queries for this task
    hitl_context = state.get("hitl_smry", "")
    key_entities = query_anchor.get("key_entities", [])
    client = get_ollama_client()

    prompt = TASK_SEARCH_QUERIES_PROMPT.format(
        task=current_task.task,
        original_query=analysis.original_query,
        hitl_context=hitl_context or "None",
        key_entities=", ".join(key_entities) if key_entities else "None",
        language=lang_label,
    )

    try:
        search_queries_out = client.generate_structured(
            prompt, TaskSearchQueries
        )
        generated_queries = [
            search_queries_out.query_1,
            search_queries_out.query_2,
        ]
    except Exception as e:
        logger.warning("Search query generation failed: %s, using fallback", e)
        generated_queries = []

    # Build query list: original concatenation + 2 LLM-generated
    base_query = f"{current_task.task} {' '.join(analysis.key_concepts)}"
    all_queries = [base_query] + generated_queries

    # Execute all queries and deduplicate results by chunk identity
    top_k = state.get("k_results") or settings.m_chunks_per_query
    selected_database = state.get("selected_database")
    seen_chunks: set[str] = set()
    results = []

    for q in all_queries:
        q_results = vector_search(
            q, top_k=top_k, selected_database=selected_database
        )
        for r in q_results:
            chunk_key = f"{r.doc_name}:{r.page_number}:{r.chunk_text[:100]}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                results.append(r)

    # Get existing tiered context and preserved quotes
    primary_context = list(state.get("primary_context", []))
    secondary_context = list(state.get("secondary_context", []))
    tertiary_context = list(state.get("tertiary_context", []))
    preserved_quotes = list(state.get("preserved_quotes", []))

    # Track per-task tier boundaries for task summary
    primary_start = len(primary_context)
    secondary_start = len(secondary_context)
    tertiary_start = len(tertiary_context)
    quotes_start = len(preserved_quotes)

    # Process results into chunks with info and classify into tiers
    chunks: list[ChunkWithInfo] = []
    doc_history: list[str] = []
    token_count = 0

    for result in results:
        # Use enhanced extraction with quote preservation (Phase C)
        chunk, quotes = create_chunk_with_info(
            result,
            analysis.original_query,
            query_anchor=query_anchor,
            preserve_quotes=True,
            language=language,
        )
        preserved_quotes.extend(quotes)

        # Classify into tier (Phase B)
        tier, weight = classify_context_tier(
            chunk=chunk,
            query_anchor=query_anchor,
            depth=0,
            source_type="vector_search",
        )
        context_entry = create_tiered_context_entry(
            chunk=chunk,
            tier=tier,
            weight=weight,
            depth=0,
            source_type="vector_search",
        )

        # Add to appropriate tier
        if tier == 1:
            primary_context.append(context_entry)
        elif tier == 2:
            secondary_context.append(context_entry)
        else:
            tertiary_context.append(context_entry)

        # Detect and follow references (configurable method)
        if chunk.extracted_info:
            if settings.reference_extraction_method == "hybrid":
                refs = detect_references_hybrid(chunk.extracted_info)
            elif settings.reference_extraction_method == "llm":
                from src.agents.tools import extract_references_llm
                refs = extract_references_llm(chunk.extracted_info)
            else:
                refs = detect_references(chunk.extracted_info)

            current_depth = state.get("current_depth", 0)
            for ref in refs:
                ref_key = f"{ref.type}:{ref.target}"
                if not context.has_visited_ref(ref_key):
                    nested = resolve_reference_enhanced(
                        ref,
                        chunk.document,
                        visited=set(context.metadata.visited_refs),
                        depth=current_depth + 1,
                        token_count=token_count,
                    )
                    ref.nested_chunks = nested
                    ref.found = len(nested) > 0
                    context.mark_ref_visited(ref_key)

                    # Classify nested chunks into tiers (Phase B)
                    for nc in nested:
                        token_count += len(nc.chunk) // 4
                        doc_history.append(nc.document)

                        # Create a ChunkWithInfo for classification
                        nested_chunk = ChunkWithInfo(
                            chunk=nc.chunk,
                            document=nc.document,
                            extracted_info=nc.extracted_info,
                            relevance_score=nc.relevance_score,
                        )
                        nc_tier, nc_weight = classify_context_tier(
                            chunk=nested_chunk,
                            query_anchor=query_anchor,
                            depth=current_depth + 1,
                            source_type="reference",
                        )
                        nc_entry = create_tiered_context_entry(
                            chunk=nested_chunk,
                            tier=nc_tier,
                            weight=nc_weight,
                            depth=current_depth + 1,
                            source_type="reference",
                        )

                        # Nested chunks go to tier 2 or 3
                        if nc_tier == 2:
                            secondary_context.append(nc_entry)
                        else:
                            tertiary_context.append(nc_entry)

            chunk.references = refs

        chunks.append(chunk)
        doc_history.append(chunk.document)
        context.add_document_reference(chunk.document)

        # Check convergence across document history
        if detect_convergence(doc_history):
            logger.info("Convergence detected in reference following, stopping early")
            break

    # Filter by relevance
    pre_filter_count = len(chunks)
    chunks = filter_by_relevance(chunks, analysis.original_query, min_results=top_k)
    logger.info(
        "Task %s: %d queries (%d generated), %d unique results, %d post-filter",
        task_id,
        len(all_queries),
        len(generated_queries),
        len(results),
        len(chunks),
    )

    # Add to research context
    search_result = SearchQueryResult(
        query=" | ".join(all_queries),
        chunks=chunks,
    )
    context.search_queries.append(search_result)
    context.metadata.total_iterations += 1

    # Generate task summary (Phase D) — pass per-task tiered findings
    task_summaries = list(state.get("task_summaries", []))
    task_summary = _generate_task_summary(
        task=current_task,
        task_primary=primary_context[primary_start:],
        task_secondary=secondary_context[secondary_start:],
        task_tertiary=tertiary_context[tertiary_start:],
        preserved_quotes=preserved_quotes[quotes_start:],
        query_anchor=query_anchor,
        hitl_smry=hitl_context,
    )
    task_summaries.append(task_summary)

    # Mark task completed (use list comprehension to avoid mutating during iteration)
    todo_list = [
        {**item, "completed": True} if item.get("id") == task_id else item
        for item in todo_list
    ]

    # Get next task
    next_task_id = None
    for item in todo_list:
        if not item.get("completed"):
            next_task_id = item.get("id")
            break

    return_dict = {
        "research_context": context.model_dump(),
        "todo_list": todo_list,
        "current_task_id": next_task_id,
        "current_depth": 0,  # Reset depth after each task
        "phase": "execute_tasks" if next_task_id else "validate_relevance",
        # Graded context (Phase B)
        "primary_context": primary_context,
        "secondary_context": secondary_context,
        "tertiary_context": tertiary_context,
        # Preserved quotes (Phase C)
        "preserved_quotes": preserved_quotes,
        # Task summaries (Phase D)
        "task_summaries": task_summaries,
        "messages": [f"Completed task {task_id}: found {len(chunks)} relevant chunks, {len(preserved_quotes)} quotes"],
    }

    if settings.enable_state_dump and next_task_id is None:
        dump_state_markdown(state, return_dict, "tests/debugging/state_3rabbithole.md", "Phase 3: Rabbithole Complete")

    return return_dict


# --- Phase 3.5: Pre-Synthesis Relevance Validation (Phase G) ---


def validate_relevance(state: AgentState) -> dict:
    """Validate accumulated context is relevant to original query.

    Runs before synthesis to filter drift. Scores each context item
    against the original query and filters out low-relevance items.

    Args:
        state: Current agent state

    Returns:
        State update with filtered tiered context
    """
    query_anchor = state.get("query_anchor", {})
    original_query = query_anchor.get("original_query", state.get("query", ""))
    key_entities = query_anchor.get("key_entities", [])

    # Score and filter primary context
    primary_context = state.get("primary_context", [])
    scored_primary = _score_and_filter_context(
        primary_context, original_query, key_entities, threshold=0.5
    )

    # Score and filter secondary context (slightly lower threshold)
    secondary_context = state.get("secondary_context", [])
    scored_secondary = _score_and_filter_context(
        secondary_context, original_query, key_entities, threshold=0.4
    )

    # Tertiary context: light filtering (keep most of it)
    tertiary_context = state.get("tertiary_context", [])
    scored_tertiary = _score_and_filter_context(
        tertiary_context, original_query, key_entities, threshold=0.3
    )

    # Log drift detection
    original_count = len(primary_context) + len(secondary_context)
    filtered_count = len(scored_primary) + len(scored_secondary)
    if original_count > 0 and filtered_count < original_count * 0.7:
        logger.warning(
            f"Query drift detected: filtered {original_count - filtered_count} "
            f"items ({100 * (original_count - filtered_count) / original_count:.0f}%) "
            f"from accumulated context"
        )

    return {
        "primary_context": scored_primary,
        "secondary_context": scored_secondary,
        "tertiary_context": scored_tertiary,
        "phase": "synthesize",
        "messages": [
            f"Relevance validation: {len(scored_primary)} primary, "
            f"{len(scored_secondary)} secondary, {len(scored_tertiary)} tertiary items retained"
        ],
    }


def _score_and_filter_context(
    context_items: list[dict],
    query: str,
    key_entities: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    """Score context items against query and filter by threshold.

    Uses simple keyword/entity matching for efficiency.
    For high-stakes filtering, could use LLM scoring.

    Args:
        context_items: List of context dicts to score
        query: Original query
        key_entities: Key entities from query
        threshold: Minimum relevance score to keep (0.0-1.0)

    Returns:
        Filtered and sorted list of context items
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_items = []
    for item in context_items:
        # Get text to score
        text = item.get("extracted_info") or item.get("chunk", "") or item.get("text", "")
        text_lower = text.lower()

        # Word overlap score
        text_words = set(text_lower.split())
        if query_words:
            word_score = len(query_words & text_words) / len(query_words)
        else:
            word_score = 0.0

        # Entity match score
        entity_score = 0.0
        if key_entities:
            matches = sum(1 for e in key_entities if e.lower() in text_lower)
            entity_score = matches / len(key_entities)

        # Combined relevance score
        relevance = 0.6 * word_score + 0.4 * entity_score

        # Boost by existing context weight if present
        existing_weight = item.get("context_weight", 0.5)
        final_relevance = 0.7 * relevance + 0.3 * existing_weight

        if final_relevance >= threshold:
            item["final_relevance"] = final_relevance
            scored_items.append(item)

    # Sort by final relevance (highest first)
    scored_items.sort(key=lambda x: x.get("final_relevance", 0), reverse=True)

    return scored_items


# --- Phase 4: Synthesis ---


class SynthesisOutput(BaseModel):
    """LLM output for synthesis."""

    summary: str
    key_findings: list[str]


def synthesize(state: AgentState) -> dict:
    """Synthesize research findings with graded context and query anchoring.

    Uses tiered context (primary, secondary, tertiary) and includes
    HITL context summary and preserved quotes for comprehensive synthesis.

    Args:
        state: Current agent state

    Returns:
        State update with synthesized results
    """
    context = ResearchContext.model_validate(state["research_context"])
    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    client = get_ollama_client()

    # Get query anchor for language enforcement
    query_anchor = state.get("query_anchor", {
        "original_query": analysis.original_query,
        "detected_language": analysis.detected_language,
    })
    language = query_anchor.get("detected_language", analysis.detected_language)

    # Get graded context presence for legacy fallback check
    primary_context = state.get("primary_context", [])
    secondary_context = state.get("secondary_context", [])

    # Get task summaries (Phase D) — sole evidence source for enhanced synthesis
    task_summaries = state.get("task_summaries", [])
    summaries_text = _format_task_summaries(task_summaries) if task_summaries else "No task summaries available"

    # Get HITL context summary (Phase A)
    hitl_smry = state.get("hitl_smry", "")

    # Fallback to old synthesis if no graded context available
    if not primary_context and not secondary_context:
        # Use legacy synthesis
        all_info = []
        for search_result in context.search_queries:
            for chunk in search_result.chunks:
                if chunk.extracted_info:
                    all_info.append({
                        "text": chunk.extracted_info,
                        "source": chunk.document,
                        "page": chunk.page,
                    })

        if not all_info:
            total_chunks = sum(len(sq.chunks) for sq in context.search_queries)
            logger.warning(
                "Synthesize: no extracted_info found. search_queries=%d, total_chunks=%d",
                len(context.search_queries),
                total_chunks,
            )
            # Fall through with raw chunk text instead of returning empty
            for search_result in context.search_queries:
                for chunk in search_result.chunks:
                    if chunk.chunk:
                        all_info.append({
                            "text": chunk.chunk[:500],
                            "source": chunk.document,
                            "page": chunk.page,
                        })

        if not all_info:
            # Truly nothing available — produce minimal summary
            result = SynthesisOutput(
                summary="Insufficient data found in the knowledge base to answer this query.",
                key_findings=[],
            )
            for i, sq in enumerate(context.search_queries):
                if i == 0:
                    sq.summary = result.summary
            return {
                "research_context": context.model_dump(),
                "phase": "quality_check",
                "messages": ["No data available for synthesis"],
            }

        max_synthesis_docs = settings.max_docs * 4
        info_text = json.dumps(all_info[:max_synthesis_docs], ensure_ascii=False, indent=2)

        prompt = SYNTHESIS_PROMPT.format(
            original_query=analysis.original_query,
            findings=info_text,
            language=language,
        )

        try:
            result = client.generate_structured(prompt, SynthesisOutput)
            for i, sq in enumerate(context.search_queries):
                if i == 0:
                    sq.summary = result.summary
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            result = SynthesisOutput(
                summary="Synthesis failed. Please review individual findings.",
                key_findings=[],
            )

        return {
            "research_context": context.model_dump(),
            "phase": "quality_check",
            "messages": [f"Synthesized {len(all_info)} findings (legacy mode)"],
        }

    # Use enhanced synthesis with pre-digested task summaries (Phase E)
    prompt = SYNTHESIS_PROMPT_ENHANCED.format(
        original_query=query_anchor.get("original_query", analysis.original_query),
        hitl_smry=hitl_smry or "No clarification conversation recorded",
        task_summaries=summaries_text,
        language=language,
    )

    try:
        # Use language-enforced generation (Phase F)
        result = client.generate_structured_with_language(
            prompt,
            SynthesisOutputEnhanced,
            target_language=language,
        )

        # Update search queries with summary
        for i, sq in enumerate(context.search_queries):
            if i == 0:
                sq.summary = result.summary

        logger.info(
            f"Enhanced synthesis complete: query_coverage={result.query_coverage}%, "
            f"gaps={len(result.remaining_gaps)}"
        )

    except Exception as e:
        logger.warning(f"Enhanced synthesis failed: {e}")
        result = SynthesisOutputEnhanced(
            summary="Synthesis failed. Please review individual findings.",
            key_findings=[],
            query_coverage=0,
            remaining_gaps=["Synthesis failed due to error"],
        )

    return {
        "research_context": context.model_dump(),
        "phase": "quality_check",
        "messages": [
            f"Synthesized from {len(task_summaries)} task summaries",
            f"Query coverage: {result.query_coverage}%",
        ],
    }


def _format_tiered_findings(context_items: list[dict], max_chars: int = 8000) -> str:
    """Format tiered context items into a string for synthesis.

    Args:
        context_items: List of context dicts with extracted_info
        max_chars: Maximum characters to include

    Returns:
        Formatted string with findings
    """
    if not context_items:
        return ""

    findings = []
    total_chars = 0

    for item in context_items:
        text = item.get("extracted_info") or item.get("text") or item.get("chunk", "")
        if not text:
            continue

        source = item.get("document", "Unknown")
        page = item.get("page", "?")
        weight = item.get("context_weight", 0.5)

        finding = f"[{source}, Page {page}] (weight: {weight:.2f}): {text[:500]}"

        if total_chars + len(finding) > max_chars:
            break

        findings.append(finding)
        total_chars += len(finding)

    return "\n\n".join(findings)


def _format_task_summaries(task_summaries: list[dict]) -> str:
    """Format task summaries with key findings and gaps for synthesis.

    Args:
        task_summaries: List of task summary dicts from Phase D

    Returns:
        Formatted string with summaries, findings, gaps, and quotes
    """
    parts = []
    for ts in task_summaries:
        tid = ts.get("task_id", "?")
        task_text = ts.get("task_text", "")
        summary = ts.get("summary", "No summary")
        key_findings = ts.get("key_findings", [])
        gaps = ts.get("gaps", [])
        quotes = ts.get("preserved_quotes", [])

        lines = [f"--- Task {tid}: {task_text} ---", f"Summary: {summary}"]

        if key_findings:
            lines.append("Key findings:")
            for f in key_findings:
                lines.append(f"  - {f}")

        if gaps:
            lines.append("Gaps:")
            for g in gaps:
                lines.append(f"  - {g}")

        if quotes:
            lines.append("Preserved quotes:")
            for q in quotes[:3]:
                text = q.get("quote_text", "") if isinstance(q, dict) else str(q)
                source = q.get("source_document", "") if isinstance(q, dict) else ""
                if text:
                    lines.append(f'  - "{text[:200]}" [{source}]')

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


# --- Phase 4b: Quality Check ---


class QualityCheckOutput(BaseModel):
    """LLM output for quality check."""

    factual_accuracy: int
    semantic_validity: int
    structural_integrity: int
    citation_correctness: int
    query_relevance: int = 0
    issues_found: list[str]


def quality_check(state: AgentState) -> dict:
    """Check quality of synthesized results.

    Args:
        state: Current agent state

    Returns:
        State update with quality assessment
    """
    if not settings.enable_quality_checker:
        return {"phase": "attribute_sources"}

    context = ResearchContext.model_validate(state["research_context"])
    client = get_ollama_client()

    # Get the synthesis summary
    summary = ""
    for sq in context.search_queries:
        if sq.summary:
            summary = sq.summary
            break

    if not summary:
        return {"phase": "attribute_sources"}

    # Get original query for relevance scoring
    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    query_anchor = state.get("query_anchor", {})
    original_query = query_anchor.get("original_query", analysis.original_query)

    language = query_anchor.get("detected_language", "de")
    lang_label = "German" if language == "de" else "English"
    prompt = QUALITY_CHECK_PROMPT.format(
        summary=summary, original_query=original_query, language=lang_label
    )

    try:
        result = client.generate_structured(prompt, QualityCheckOutput)
        overall_score = (
            result.factual_accuracy
            + result.semantic_validity
            + result.structural_integrity
            + result.citation_correctness
            + result.query_relevance
        )

        assessment = QualityAssessment(
            overall_score=overall_score,
            factual_accuracy=result.factual_accuracy,
            semantic_validity=result.semantic_validity,
            structural_integrity=result.structural_integrity,
            citation_correctness=result.citation_correctness,
            query_relevance=result.query_relevance,
            passes_quality=overall_score >= settings.quality_threshold,
            issues_found=result.issues_found,
        )

    except Exception as e:
        logger.warning(f"Quality check failed: {e}")
        # On failure, set passes_quality=False and overall_score=0
        assessment = QualityAssessment(
            overall_score=0,
            factual_accuracy=0,
            semantic_validity=0,
            structural_integrity=0,
            citation_correctness=0,
            passes_quality=False,
            issues_found=[f"Quality check failed: {e}"],
        )

    return {
        "quality_assessment": assessment.model_dump(),
        "phase": "attribute_sources",
        "messages": [f"Quality score: {assessment.overall_score}/500"],
    }


# --- Phase 5: Source Attribution ---


def attribute_sources(state: AgentState) -> dict:
    """Add source citations and create final report.

    Args:
        state: Current agent state

    Returns:
        State update with final report
    """
    context = ResearchContext.model_validate(state["research_context"])
    analysis = QueryAnalysis.model_validate(state["query_analysis"])

    # Collect all sources
    sources: list[LinkedSource] = []
    for sq in context.search_queries:
        for chunk in sq.chunks:
            source = Source(
                doc_id=chunk.document,
                doc_name=chunk.document,
                chunk_text=chunk.chunk[:500],
                page_number=chunk.page,
                relevance_score=chunk.relevance_score,
                category="vector_db",
            )
            linked = LinkedSource(
                source=source,
                resolved_path=f"kb/*__db_inserted/{chunk.document}",
                link_html=f'<a href="file://{chunk.document}">{chunk.document}</a>',
            )
            sources.append(linked)

    # Get synthesis
    answer = ""
    for sq in context.search_queries:
        if sq.summary:
            answer = sq.summary
            break

    # Create findings
    findings = []
    for sq in context.search_queries:
        for chunk in sq.chunks[:3]:  # Top 3 per query
            if chunk.extracted_info:
                findings.append(
                    Finding(
                        claim=chunk.extracted_info[:200],
                        evidence=chunk.chunk[:300],
                        sources=[
                            Source(
                                doc_id=chunk.document,
                                doc_name=chunk.document,
                                chunk_text=chunk.chunk[:200],
                                category="vector_db",
                            )
                        ],
                        confidence="medium",
                    )
                )

    # Count completed tasks
    todo_list = state.get("todo_list", [])
    completed_count = sum(1 for item in todo_list if item.get("completed"))

    # Get quality assessment from state (or use defaults if not available)
    qa = state.get("quality_assessment")
    if qa:
        quality_score = qa.get("overall_score", 0)
        quality_breakdown = {
            "factual_accuracy": qa.get("factual_accuracy", 0),
            "semantic_validity": qa.get("semantic_validity", 0),
            "structural_integrity": qa.get("structural_integrity", 0),
            "citation_correctness": qa.get("citation_correctness", 0),
            "query_relevance": qa.get("query_relevance", 0),
        }
    else:
        # Default values if quality checker was disabled or not run
        quality_score = 0
        quality_breakdown = {
            "factual_accuracy": 0,
            "semantic_validity": 0,
            "structural_integrity": 0,
            "citation_correctness": 0,
            "query_relevance": 0,
        }

    report = FinalReport(
        query=analysis.original_query,
        answer=answer or "No synthesis available",
        findings=findings[:10],  # Limit findings
        sources=sources[:settings.max_docs * 4],  # Use config for limit
        quality_score=quality_score,
        quality_breakdown=quality_breakdown,
        todo_items_completed=completed_count,
        research_iterations=context.metadata.total_iterations,
        metadata={
            "documents_referenced": context.metadata.documents_referenced,
            "timestamp": datetime.now().isoformat(),
        },
    )

    return {
        "final_report": report.model_dump(),
        "phase": "complete",
        "messages": [f"Report complete: {len(findings)} findings, {len(sources)} sources"],
    }


# --- Enhanced Phase 1: Iterative HITL Nodes ---


def hitl_init(state: AgentState) -> dict:
    """Initialize iterative HITL conversation.

    Step 1 of Enhanced Phase 1. Sets up conversation state and detects language.

    Args:
        state: Current agent state

    Returns:
        State update with initialized HITL state
    """
    from src.services.hitl_service import initialize_hitl_state

    query = state["query"]

    # Initialize HITL conversation state
    hitl_state = initialize_hitl_state(query)

    return {
        "hitl_state": hitl_state,
        "hitl_active": True,
        "hitl_iteration": 0,
        "hitl_conversation_history": hitl_state.get("conversation_history", []),
        "detected_language": hitl_state.get("detected_language", "de"),
        "phase": "hitl_generate_questions",
        "messages": [f"HITL initialized, language: {hitl_state.get('detected_language', 'de')}"],
    }


def hitl_generate_questions(state: AgentState) -> dict:
    """Generate follow-up questions for current HITL iteration.

    Step 2 of Enhanced Phase 1. Generates 2-3 contextual follow-up questions.

    Args:
        state: Current agent state

    Returns:
        State update with generated questions (hitl_pending=True)
    """
    from src.services.hitl_service import process_initial_query, process_human_feedback

    hitl_state = state.get("hitl_state", {})
    iteration = state.get("hitl_iteration", 0)

    # Pass query_retrieval from state to hitl_state for context-aware questions
    query_retrieval = state.get("query_retrieval", "")
    hitl_state["query_retrieval"] = query_retrieval

    if iteration == 0:
        # First iteration: generate initial questions
        hitl_state = process_initial_query(hitl_state)
    else:
        # Subsequent iterations: questions already generated in process_response
        pass

    questions = hitl_state.get("follow_up_questions", "")

    # Create checkpoint for UI
    # Prefer retrieval-based query_analysis (from hitl_analyze_retrieval) over hitl_state analysis
    retrieval_analysis = state.get("query_analysis", {})
    hitl_analysis = hitl_state.get("analysis", {})
    # Merge: use retrieval analysis as base, fill gaps from hitl_analysis
    merged_analysis = {**hitl_analysis, **retrieval_analysis} if retrieval_analysis else hitl_analysis

    checkpoint = {
        "checkpoint_type": "iterative_hitl",
        "content": {
            "questions": questions,
            "iteration": iteration,
            "max_iterations": state.get("hitl_max_iterations", 5),
            "analysis": merged_analysis,
            "coverage_score": state.get("coverage_score", 0.0),
            "knowledge_gaps": state.get("knowledge_gaps", []),
            "retrieval_stats": {
                "dedup_ratios": state.get("retrieval_dedup_ratios", []),
                "iteration_queries": state.get("iteration_queries", []),
            },
        },
        "requires_approval": True,
        "phase": "Phase 1: Query Refinement",
    }

    return {
        "hitl_state": hitl_state,
        "hitl_pending": True,
        "hitl_checkpoint": checkpoint,
        "hitl_conversation_history": hitl_state.get("conversation_history", []),
        "messages": [f"HITL iteration {iteration + 1}: awaiting user response"],
    }


def hitl_process_response(state: AgentState) -> dict:
    """Process user response in HITL conversation.

    Step 3 of Enhanced Phase 1. Analyzes response and decides next action:
    - If user typed '/end': finalize HITL
    - If max iterations reached: finalize HITL
    - Otherwise: generate new questions and continue

    Args:
        state: Current agent state with user response in hitl_decision

    Returns:
        State update with processed response
    """
    from src.services.hitl_service import process_human_feedback

    hitl_state = state.get("hitl_state", {})
    decision = state.get("hitl_decision", {})
    iteration = state.get("hitl_iteration", 0)
    max_iterations = state.get("hitl_max_iterations", 5)

    # Extract user response from decision
    user_response = ""
    if decision:
        if decision.get("approved"):
            # User provided a response
            mods = decision.get("modifications", {})
            user_response = mods.get("user_response", "")
        else:
            # User rejected/ended
            user_response = "/end"

    # Check for termination
    if user_response.strip().lower() == "/end":
        return {
            "hitl_active": False,
            "hitl_termination_reason": "user_end",
            "hitl_pending": False,
            "hitl_checkpoint": None,
            "hitl_decision": None,
            "phase": "hitl_finalize",
            "messages": ["User ended HITL conversation"],
        }

    # Check max iterations
    if iteration >= max_iterations - 1:
        # Process final response before finalizing
        if user_response:
            hitl_state = process_human_feedback(hitl_state, user_response)
        return {
            "hitl_state": hitl_state,
            "hitl_active": False,
            "hitl_termination_reason": "max_iterations",
            "hitl_pending": False,
            "hitl_checkpoint": None,
            "hitl_decision": None,
            "phase": "hitl_finalize",
            "messages": [f"HITL max iterations ({max_iterations}) reached"],
        }

    # Process user feedback and generate new questions
    if user_response:
        hitl_state = process_human_feedback(hitl_state, user_response)

    # Check convergence using retrieval-based coverage (from hitl_analyze_retrieval)
    # This is more reliable than conversation-based _estimate_coverage
    coverage = state.get("coverage_score", 0.0)

    # Fallback to conversation-based estimate if retrieval coverage not available
    if coverage == 0.0:
        analysis = hitl_state.get("analysis", {})
        coverage = _estimate_coverage(analysis)

    # Use 0.9 threshold for local convergence check (matches route_after_hitl_process_response 0.8 + margin)
    if coverage >= 0.9:
        return {
            "hitl_state": hitl_state,
            "hitl_active": False,
            "hitl_termination_reason": "convergence",
            "hitl_pending": False,
            "hitl_checkpoint": None,
            "hitl_decision": None,
            "coverage_score": coverage,
            "phase": "hitl_finalize",
            "messages": [f"HITL converged with coverage {coverage:.0%}"],
        }

    # Continue to next iteration
    return {
        "hitl_state": hitl_state,
        "hitl_iteration": iteration + 1,
        "hitl_pending": False,
        "hitl_checkpoint": None,
        "hitl_decision": None,
        "hitl_conversation_history": hitl_state.get("conversation_history", []),
        "coverage_score": coverage,
        "phase": "hitl_generate_questions",
        "messages": [f"HITL iteration {iteration + 1} complete, continuing..."],
    }


def _estimate_coverage(analysis: dict) -> float:
    """Estimate information coverage from HITL analysis.

    Args:
        analysis: Analysis dict from HITL service

    Returns:
        Coverage score 0-1
    """
    if not analysis:
        return 0.0

    score = 0.0

    # Check for entities (30%)
    entities = analysis.get("entities", [])
    if entities:
        score += min(len(entities) / 3, 1.0) * 0.3

    # Check for scope (30%)
    scope = analysis.get("scope", "")
    if scope:
        score += 0.3

    # Check for context (20%)
    context = analysis.get("context", "")
    if context:
        score += 0.2

    # Check for refined query (20%)
    refined = analysis.get("refined_query", "")
    if refined and refined != analysis.get("user_query", ""):
        score += 0.2

    return min(score, 1.0)


def hitl_finalize(state: AgentState) -> dict:
    """Finalize HITL conversation and prepare for Phase 2.

    Step 4 of Enhanced Phase 1. Generates research queries and hands off to Phase 2.
    Creates immutable query_anchor and summarizes HITL context for synthesis.

    Args:
        state: Current agent state

    Returns:
        State update with research_queries ready for Phase 2
    """
    from src.services.hitl_service import finalize_hitl_conversation

    hitl_state = state.get("hitl_state", {})

    # Finalize and get research queries
    result = finalize_hitl_conversation(hitl_state, max_queries=5)
    language = result.get("detected_language", "de")

    # Build query analysis from HITL results
    analysis_dict = {
        "original_query": state["query"],
        "key_concepts": result.get("entities", []),
        "entities": result.get("entities", []),
        "scope": result.get("scope", ""),
        "assumed_context": [result.get("context", "")] if result.get("context") else [],
        "clarification_needed": False,
        "detected_language": language,
        "hitl_refinements": [
            f"Summary: {result.get('summary', '')}",
        ],
    }

    # Create immutable query anchor (Phase A)
    hitl_conversation_history = state.get("hitl_conversation_history", [])
    query_anchor = {
        "original_query": state["query"],
        "detected_language": language,
        "key_entities": result.get("entities", []),
        "scope": result.get("scope", ""),
        "hitl_refinements": [
            msg["content"] for msg in hitl_conversation_history
            if msg.get("role") == "user"
        ],
        "created_at": datetime.now().isoformat(),
    }

    # Generate citation-aware HITL summary for synthesis (Phase A)
    hitl_smry = _generate_hitl_summary(
        query=state["query"],
        conversation=hitl_conversation_history,
        retrieval=state.get("query_retrieval", ""),
        knowledge_gaps=state.get("knowledge_gaps", []),
        language=language,
    )

    # Move HITL retrieval to tertiary context (Phase A)
    tertiary_context = _convert_hitl_retrieval_to_context(
        state.get("query_retrieval", ""),
        state.get("retrieval_history", {}),
    )

    return_dict = {
        "research_queries": result.get("research_queries", []),
        "additional_context": result.get("summary", ""),
        "query_analysis": analysis_dict,
        "detected_language": language,
        "hitl_active": False,
        "hitl_pending": False,
        "hitl_state": None,  # Clear HITL state
        "phase": "generate_todo",
        # Graded context management fields
        "query_anchor": query_anchor,
        "hitl_smry": hitl_smry,
        "tertiary_context": tertiary_context,
        "messages": [
            f"HITL finalized: {len(result.get('research_queries', []))} research queries generated",
            f"Termination reason: {state.get('hitl_termination_reason', 'unknown')}",
        ],
    }

    if settings.enable_state_dump:
        dump_state_markdown(state, return_dict, "tests/debugging/state_1hitl.md", "Phase 1: HITL Finalize")

    return return_dict


# --- Phase D Helpers: Per-Task Structured Summary ---


def _generate_task_summary(
    task: ToDoItem,
    task_primary: list[dict],
    task_secondary: list[dict],
    task_tertiary: list[dict],
    preserved_quotes: list[dict],
    query_anchor: dict,
    hitl_smry: str = "",
) -> dict:
    """Generate structured summary for completed task using tiered findings.

    Args:
        task: The completed task
        task_primary: Tier 1 context entries for this task
        task_secondary: Tier 2 context entries for this task
        task_tertiary: Tier 3 context entries for this task
        preserved_quotes: Quotes extracted from this task's chunks
        query_anchor: Immutable query reference
        hitl_smry: HITL findings summary for dedup context

    Returns:
        Task summary dict
    """
    client = get_ollama_client()
    language = query_anchor.get("detected_language", "de")
    original_query = query_anchor.get("original_query", "")

    # Format tiered findings using existing helper
    primary_text = _format_tiered_findings(task_primary[:15])
    secondary_text = _format_tiered_findings(task_secondary[:10])
    tertiary_text = _format_tiered_findings(task_tertiary[:5])

    # Collect sources from all tiers
    sources = []
    for item in task_primary + task_secondary + task_tertiary:
        doc = item.get("document")
        if doc:
            sources.append(doc)

    # Format quotes
    quotes_text = json.dumps(preserved_quotes[:5], ensure_ascii=False) if preserved_quotes else "[]"

    # Calculate relevance score
    relevance_score = _calculate_task_relevance(
        task_text=task.task,
        original_query=original_query,
        key_entities=query_anchor.get("key_entities", []),
    )

    prompt = TASK_SUMMARY_PROMPT.format(
        task=task.task,
        original_query=original_query,
        primary_findings=primary_text or "No primary findings",
        secondary_findings=secondary_text or "No secondary findings",
        tertiary_findings=tertiary_text or "No tertiary findings",
        preserved_quotes=quotes_text,
        hitl_smry=hitl_smry or "No prior findings",
        language=language,
    )

    try:
        result = client.generate_structured(prompt, TaskSummaryOutput)
        return {
            "task_id": task.id,
            "task_text": task.task,
            "summary": result.summary,
            "key_findings": result.key_findings,
            "gaps": result.gaps,
            "preserved_quotes": preserved_quotes,
            "sources": list(set(sources)),
            "relevance_to_query": relevance_score,
        }
    except Exception as e:
        logger.warning(f"Failed to generate task summary: {e}")
        return {
            "task_id": task.id,
            "task_text": task.task,
            "summary": f"Completed task: {task.task}",
            "key_findings": [],
            "gaps": [],
            "preserved_quotes": preserved_quotes,
            "sources": list(set(sources)),
            "relevance_to_query": relevance_score,
        }


def _calculate_task_relevance(
    task_text: str,
    original_query: str,
    key_entities: list[str],
) -> float:
    """Calculate relevance of task to original query.

    Uses simple keyword overlap scoring.

    Args:
        task_text: The task description
        original_query: Original user query
        key_entities: Key entities from query

    Returns:
        Relevance score 0.0-1.0
    """
    task_lower = task_text.lower()
    query_lower = original_query.lower()

    # Word overlap score
    query_words = set(query_lower.split())
    task_words = set(task_lower.split())
    if not query_words:
        return 0.5
    word_overlap = len(query_words & task_words) / len(query_words)

    # Entity match score
    entity_score = 0.0
    if key_entities:
        matches = sum(1 for e in key_entities if e.lower() in task_lower)
        entity_score = matches / len(key_entities)

    # Combined score (weighted average)
    return min(1.0, 0.6 * word_overlap + 0.4 * entity_score)


# --- Phase A Helpers: HITL Context Preservation ---


def _generate_hitl_summary(
    query: str,
    conversation: list[dict],
    retrieval: str,
    knowledge_gaps: list[str],
    language: str,
) -> str:
    """Generate citation-aware HITL summary for synthesis.

    Produces a summary with [Source_filename] annotations so downstream
    synthesis can trace facts back to source documents.

    Args:
        query: Original user query
        conversation: HITL conversation history
        retrieval: Accumulated retrieval text from HITL (with [doc, p.N] prefixes)
        knowledge_gaps: Identified knowledge gaps
        language: Target language (de/en)

    Returns:
        Citation-aware HITL summary string
    """
    if not conversation and not retrieval:
        return ""

    client = get_ollama_client()

    # Format conversation
    conv_text = "\n".join(
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
        for msg in conversation
    )

    # Format gaps
    gaps_text = "\n".join(f"- {gap}" for gap in knowledge_gaps) if knowledge_gaps else "None identified"

    # Truncate retrieval if too long — keep 8000 chars to preserve [doc, p.N] prefixes
    retrieval_truncated = retrieval[:8000] if len(retrieval) > 8000 else retrieval

    prompt = HITL_SUMMARY_PROMPT.format(
        query=query,
        conversation=conv_text or "No conversation recorded",
        retrieval=retrieval_truncated or "No retrieval performed",
        gaps=gaps_text,
        language=language,
    )

    try:
        return client.generate(prompt)
    except Exception as e:
        logger.warning(f"Failed to generate HITL summary: {e}")
        return f"HITL Summary: Query '{query}' with {len(conversation)} conversation turns."


def _convert_hitl_retrieval_to_context(
    query_retrieval: str,
    retrieval_history: dict,
) -> list[dict]:
    """Convert HITL retrieval text to tertiary context entries.

    Args:
        query_retrieval: Accumulated retrieval text
        retrieval_history: Per-iteration retrieval metadata

    Returns:
        List of context dicts for tertiary_context
    """
    if not query_retrieval:
        return []

    # Create a single context entry from the accumulated retrieval
    # In the future, this could be parsed to extract individual chunks
    return [
        {
            "source_type": "hitl_retrieval",
            "text": query_retrieval[:10000],  # Limit size
            "context_tier": 3,
            "context_weight": 0.4,
            "iteration_count": len(retrieval_history),
        }
    ]


# --- Enhanced Phase 1: Multi-Query Retrieval Nodes ---


def hitl_generate_queries(state: AgentState) -> dict:
    """Generate 3 search queries for current HITL iteration.

    Node 1 of the Enhanced Phase 1 retrieval loop.
    - Iteration 0: original + broader + alternative angle
    - Iteration N>0: refined based on user feedback + knowledge gaps

    Args:
        state: Current agent state

    Returns:
        State update with iteration_queries
    """
    from src.services.hitl_service import (
        generate_alternative_queries_llm,
        generate_refined_queries_llm,
    )

    iteration = state.get("hitl_iteration", 0)
    query = state["query"]
    analysis = state.get("query_analysis", {})
    conversation = state.get("hitl_conversation_history", [])
    language = state.get("detected_language", "de")

    if iteration == 0:
        # Initial: original + broader + alternative angle
        queries = generate_alternative_queries_llm(
            query, {}, iteration, language=language
        )
    else:
        # Refined: based on user feedback + knowledge gaps
        gaps = state.get("knowledge_gaps", [])
        last_response = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_response = msg.get("content", "")
                break
        queries = generate_refined_queries_llm(
            query, last_response, gaps, language=language
        )

    # Track all queries per iteration
    iteration_queries = list(state.get("iteration_queries", []))
    iteration_queries.append(queries)

    return {
        "iteration_queries": iteration_queries,
        "phase": "hitl_retrieve_chunks",
        "messages": [f"Generated {len(queries)} queries for iteration {iteration}"],
    }


def hitl_retrieve_chunks(state: AgentState) -> dict:
    """Execute vector search and deduplicate results.

    Node 2 of the Enhanced Phase 1 retrieval loop.
    Retrieves 3 chunks per query (9 total), deduplicates against existing retrieval.

    Args:
        state: Current agent state

    Returns:
        State update with query_retrieval, retrieval_dedup_ratios
    """
    from src.services.hitl_service import calculate_dedup_ratio, format_chunks_for_state

    iteration = state.get("hitl_iteration", 0)
    iteration_queries = state.get("iteration_queries", [[]])
    queries = iteration_queries[-1] if iteration_queries else []
    selected_database = state.get("selected_database")
    k_per_query = 3

    all_chunks = []
    for q in queries:
        try:
            results = vector_search(q, top_k=k_per_query, selected_database=selected_database)
            all_chunks.extend(results)
        except Exception as e:
            logger.warning(f"Vector search failed for query '{q}': {e}")

    # Deduplicate against existing retrieval
    existing = state.get("query_retrieval", "")
    unique_chunks, dedup_stats = calculate_dedup_ratio(all_chunks, existing)

    # Format and append to query_retrieval
    formatted = format_chunks_for_state(unique_chunks, queries)
    new_retrieval = existing + "\n\n" + formatted if existing else formatted

    # Track dedup ratio for convergence detection
    dedup_ratios = list(state.get("retrieval_dedup_ratios", []))
    dedup_ratios.append(dedup_stats["dedup_ratio"])

    # Update retrieval history
    retrieval_history = dict(state.get("retrieval_history", {}))
    retrieval_history[f"iteration_{iteration}"] = {
        "queries": queries,
        "new_chunks": dedup_stats["new_count"],
        "duplicates": dedup_stats["dup_count"],
    }

    return {
        "query_retrieval": new_retrieval,
        "retrieval_dedup_ratios": dedup_ratios,
        "retrieval_history": retrieval_history,
        "phase": "hitl_analyze_retrieval",
        "messages": [
            f"Retrieved {dedup_stats['new_count']} new chunks, "
            f"{dedup_stats['dup_count']} duplicates skipped"
        ],
    }


def hitl_analyze_retrieval(state: AgentState) -> dict:
    """Analyze accumulated retrieval for concepts, gaps, coverage.

    Node 3 of the Enhanced Phase 1 retrieval loop.
    LLM analyzes query + retrieval to extract entities, gaps, coverage score.

    Args:
        state: Current agent state

    Returns:
        State update with query_analysis, knowledge_gaps, coverage_score
    """
    from src.services.hitl_service import analyze_retrieval_context_llm

    query = state["query"]
    retrieval = state.get("query_retrieval", "")
    language = state.get("detected_language", "de")

    # LLM analysis
    analysis = analyze_retrieval_context_llm(query, retrieval, language=language)

    return {
        "query_analysis": analysis,
        "knowledge_gaps": analysis.get("knowledge_gaps", []),
        "coverage_score": analysis.get("coverage_score", 0.0),
        "phase": "hitl_generate_questions",
        "messages": [
            f"Coverage: {analysis.get('coverage_score', 0):.0%}, "
            f"gaps: {len(analysis.get('knowledge_gaps', []))}"
        ],
    }

