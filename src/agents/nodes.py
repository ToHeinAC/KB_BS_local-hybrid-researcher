"""Node functions for the LangGraph research agent."""

import json
import logging
from datetime import datetime

from pydantic import BaseModel

from src.agents.state import AgentState, get_next_task_id
from src.agents.tools import (
    create_chunk_with_info,
    detect_references,
    filter_by_relevance,
    resolve_reference,
    vector_search,
)
from src.config import settings
from src.models.hitl import HITLDecision
from src.models.query import QueryAnalysis, ToDoItem, ToDoList
from src.models.research import (
    ChunkWithInfo,
    ResearchContext,
    ResearchContextMetadata,
    SearchQueryResult,
)
from src.models.results import FinalReport, Finding, LinkedSource, QualityAssessment, Source
from src.prompts import (
    QUALITY_CHECK_PROMPT,
    SYNTHESIS_PROMPT,
    TODO_GENERATION_PROMPT,
)
from src.services.hitl_service import HITLService
from src.services.ollama_client import OllamaClient

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
    analysis = QueryAnalysis.model_validate(state["query_analysis"])

    # Check for research_queries from HITL first
    research_queries = state.get("research_queries", [])
    additional_context = state.get("additional_context", "")

    if research_queries:
        # Convert HITL research queries directly to ToDoItems
        items = [
            ToDoItem(
                id=i + 1,
                task=query,
                context=additional_context if i == 0 else "From HITL conversation",
            )
            for i, query in enumerate(research_queries)
        ]
        logger.info(f"Using {len(items)} research queries from HITL")
    else:
        # Fallback to LLM generation
        client = get_ollama_client()

        prompt = TODO_GENERATION_PROMPT.format(
            original_query=analysis.original_query,
            key_concepts=analysis.key_concepts,
            entities=analysis.entities,
            scope=analysis.scope,
            assumed_context=analysis.assumed_context,
            num_items=settings.initial_todo_items,
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

    return {
        "todo_list": [item.model_dump() for item in todo_list.items],
        "phase": "execute_tasks",
        "hitl_pending": False,
        "hitl_checkpoint": None,
        "hitl_decision": None,
        "current_task_id": get_next_task_id(state),
        "hitl_history": state.get("hitl_history", [])
        + [{"type": "todo_approve", "decision": decision.model_dump()}],
    }


# --- Phase 3: Task Execution ---


def execute_task(state: AgentState) -> dict:
    """Execute a single research task.

    Args:
        state: Current agent state

    Returns:
        State update with research results
    """
    task_id = state.get("current_task_id")
    if task_id is None:
        return {"phase": "synthesize"}

    # Find current task
    todo_list = state.get("todo_list", [])
    current_task = None
    for item in todo_list:
        if item.get("id") == task_id:
            current_task = ToDoItem.model_validate(item)
            break

    if not current_task or current_task.completed:
        return {"phase": "synthesize"}

    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    context = ResearchContext.model_validate(
        state.get(
            "research_context",
            {"search_queries": [], "metadata": ResearchContextMetadata().model_dump()},
        )
    )

    # Perform vector search
    query = f"{current_task.task} {' '.join(analysis.key_concepts)}"
    top_k = state.get("k_results") or settings.m_chunks_per_query
    selected_database = state.get("selected_database")
    results = vector_search(
        query,
        top_k=top_k,
        selected_database=selected_database,
    )

    # Process results into chunks with info
    chunks: list[ChunkWithInfo] = []
    for result in results:
        chunk = create_chunk_with_info(result, analysis.original_query)

        # Detect and follow references
        if chunk.extracted_info:
            refs = detect_references(chunk.extracted_info)
            current_depth = state.get("current_depth", 0)
            for ref in refs:
                ref_key = f"{ref.type}:{ref.target}"
                if not context.has_visited_ref(ref_key):
                    # Pass depth+1 since we're following a reference from the initial search
                    nested = resolve_reference(
                        ref,
                        chunk.document,
                        visited=set(context.metadata.visited_refs),
                        depth=current_depth + 1,
                    )
                    ref.nested_chunks = nested
                    ref.found = len(nested) > 0
                    context.mark_ref_visited(ref_key)

            chunk.references = refs

        chunks.append(chunk)
        context.add_document_reference(chunk.document)

    # Filter by relevance
    chunks = filter_by_relevance(chunks, analysis.original_query)

    # Add to research context
    search_result = SearchQueryResult(
        query=query,
        chunks=chunks,
    )
    context.search_queries.append(search_result)
    context.metadata.total_iterations += 1

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

    return {
        "research_context": context.model_dump(),
        "todo_list": todo_list,
        "current_task_id": next_task_id,
        "current_depth": 0,  # Reset depth after each task
        "phase": "execute_tasks" if next_task_id else "synthesize",
        "messages": [f"Completed task {task_id}: found {len(chunks)} relevant chunks"],
    }


# --- Phase 4: Synthesis ---


class SynthesisOutput(BaseModel):
    """LLM output for synthesis."""

    summary: str
    key_findings: list[str]


def synthesize(state: AgentState) -> dict:
    """Synthesize research findings into a coherent summary.

    Args:
        state: Current agent state

    Returns:
        State update with synthesized results
    """
    context = ResearchContext.model_validate(state["research_context"])
    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    client = get_ollama_client()

    # Collect all extracted info
    all_info = []
    for search_result in context.search_queries:
        for chunk in search_result.chunks:
            if chunk.extracted_info:
                all_info.append(
                    {
                        "text": chunk.extracted_info,
                        "source": chunk.document,
                        "page": chunk.page,
                    }
                )

    if not all_info:
        return {
            "phase": "quality_check",
            "messages": ["No information to synthesize"],
        }

    # Truncate if too long (use config setting for max docs in synthesis)
    max_synthesis_docs = settings.max_docs * 4  # 4x for synthesis context
    info_text = json.dumps(all_info[:max_synthesis_docs], ensure_ascii=False, indent=2)

    prompt = SYNTHESIS_PROMPT.format(
        original_query=analysis.original_query,
        findings=info_text,
        language=analysis.detected_language,
    )

    try:
        result = client.generate_structured(prompt, SynthesisOutput)

        # Update search queries with summaries
        for i, sq in enumerate(context.search_queries):
            if i == 0:  # Add overall summary to first query
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
        "messages": [f"Synthesized {len(all_info)} findings"],
    }


# --- Phase 4b: Quality Check ---


class QualityCheckOutput(BaseModel):
    """LLM output for quality check."""

    factual_accuracy: int
    semantic_validity: int
    structural_integrity: int
    citation_correctness: int
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

    prompt = QUALITY_CHECK_PROMPT.format(summary=summary)

    try:
        result = client.generate_structured(prompt, QualityCheckOutput)
        overall_score = (
            result.factual_accuracy
            + result.semantic_validity
            + result.structural_integrity
            + result.citation_correctness
        )

        assessment = QualityAssessment(
            overall_score=overall_score,
            factual_accuracy=result.factual_accuracy,
            semantic_validity=result.semantic_validity,
            structural_integrity=result.structural_integrity,
            citation_correctness=result.citation_correctness,
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
        "messages": [f"Quality score: {assessment.overall_score}/400"],
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
        }
    else:
        # Default values if quality checker was disabled or not run
        quality_score = 0
        quality_breakdown = {
            "factual_accuracy": 0,
            "semantic_validity": 0,
            "structural_integrity": 0,
            "citation_correctness": 0,
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

    Args:
        state: Current agent state

    Returns:
        State update with research_queries ready for Phase 2
    """
    from src.services.hitl_service import finalize_hitl_conversation

    hitl_state = state.get("hitl_state", {})

    # Finalize and get research queries
    result = finalize_hitl_conversation(hitl_state, max_queries=5)

    # Build query analysis from HITL results
    analysis_dict = {
        "original_query": state["query"],
        "key_concepts": result.get("entities", []),
        "entities": result.get("entities", []),
        "scope": result.get("scope", ""),
        "assumed_context": [result.get("context", "")] if result.get("context") else [],
        "clarification_needed": False,
        "detected_language": result.get("detected_language", "de"),
        "hitl_refinements": [
            f"Summary: {result.get('summary', '')}",
        ],
    }

    return {
        "research_queries": result.get("research_queries", []),
        "additional_context": result.get("summary", ""),
        "query_analysis": analysis_dict,
        "detected_language": result.get("detected_language", "de"),
        "hitl_active": False,
        "hitl_pending": False,
        "hitl_state": None,  # Clear HITL state
        "phase": "generate_todo",
        "messages": [
            f"HITL finalized: {len(result.get('research_queries', []))} research queries generated",
            f"Termination reason: {state.get('hitl_termination_reason', 'unknown')}",
        ],
    }


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

    if iteration == 0:
        # Initial: original + broader + alternative angle
        queries = generate_alternative_queries_llm(query, {}, iteration)
    else:
        # Refined: based on user feedback + knowledge gaps
        gaps = state.get("knowledge_gaps", [])
        last_response = ""
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                last_response = msg.get("content", "")
                break
        queries = generate_refined_queries_llm(query, last_response, gaps)

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

    # LLM analysis
    analysis = analyze_retrieval_context_llm(query, retrieval)

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

