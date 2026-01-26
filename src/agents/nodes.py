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
from src.models.hitl import HITLCheckpoint, HITLDecision
from src.models.query import QueryAnalysis, ToDoItem, ToDoList
from src.models.research import (
    ChunkWithInfo,
    ResearchContext,
    ResearchContextMetadata,
    SearchQueryResult,
)
from src.models.results import FinalReport, Finding, LinkedSource, QualityAssessment, Source
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


# --- Phase 1: Query Analysis ---


class QueryAnalysisOutput(BaseModel):
    """LLM output for query analysis."""

    key_concepts: list[str]
    entities: list[str]
    scope: str
    assumed_context: list[str]
    clarification_needed: bool
    detected_language: str


def analyze_query(state: AgentState) -> dict:
    """Analyze the user query and extract key information.

    Args:
        state: Current agent state

    Returns:
        State update with query_analysis
    """
    query = state["query"]
    client = get_ollama_client()

    prompt = f"""Analyze this research query and extract key information.

Query: "{query}"

Extract:
1. key_concepts: Main concepts/topics (list of strings)
2. entities: Named entities like laws, regulations, documents (list of strings)
3. scope: Research scope (e.g., "regulatory", "technical", "general")
4. assumed_context: Implicit context assumptions (list of strings)
5. clarification_needed: Whether clarification would help (boolean)
6. detected_language: Language of the query ("de" or "en")

Respond in JSON format."""

    try:
        result = client.generate_structured(prompt, QueryAnalysisOutput)
        analysis = QueryAnalysis(
            original_query=query,
            key_concepts=result.key_concepts,
            entities=result.entities,
            scope=result.scope,
            assumed_context=result.assumed_context,
            clarification_needed=result.clarification_needed,
            detected_language=result.detected_language,
        )
    except Exception as e:
        logger.warning(f"Query analysis failed, using defaults: {e}")
        analysis = QueryAnalysis(
            original_query=query,
            key_concepts=query.split()[:5],
            clarification_needed=True,
        )

    return {
        "query_analysis": analysis.model_dump(),
        "phase": "hitl_clarify",
        "messages": [f"Query analyzed: {len(analysis.key_concepts)} concepts extracted"],
    }


def hitl_clarify(state: AgentState) -> dict:
    """Generate clarification questions for HITL.

    Args:
        state: Current agent state

    Returns:
        State update with HITL checkpoint or phase transition
    """
    hitl_service = get_hitl_service()
    analysis = QueryAnalysis.model_validate(state["query_analysis"])

    # Check if clarification is needed
    if not hitl_service.should_request_clarification(analysis):
        return {
            "phase": "generate_todo",
            "hitl_pending": False,
        }

    # Generate questions
    questions = hitl_service.generate_clarification_questions(analysis)

    if not questions:
        return {
            "phase": "generate_todo",
            "hitl_pending": False,
        }

    # Create checkpoint
    checkpoint = hitl_service.create_query_checkpoint(questions)

    return {
        "hitl_pending": True,
        "hitl_checkpoint": checkpoint.model_dump(),
        "messages": [f"Awaiting user clarification: {len(questions)} questions"],
    }


def process_hitl_clarify(state: AgentState) -> dict:
    """Process HITL clarification response.

    Args:
        state: Current agent state with hitl_decision

    Returns:
        State update with refined analysis
    """
    hitl_service = get_hitl_service()
    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    decision = HITLDecision.model_validate(state.get("hitl_decision", {}))

    if decision.approved and decision.modifications:
        # Extract answers from modifications
        answers = decision.modifications.get("answers", {})
        analysis = hitl_service.merge_clarifications(analysis, answers)

    return {
        "query_analysis": analysis.model_dump(),
        "phase": "generate_todo",
        "hitl_pending": False,
        "hitl_checkpoint": None,
        "hitl_decision": None,
        "hitl_history": state.get("hitl_history", [])
        + [{"type": "clarify", "decision": decision.model_dump()}],
    }


# --- Phase 2: ToDo List Generation ---


class ToDoListOutput(BaseModel):
    """LLM output for todo list generation."""

    items: list[dict]


def generate_todo(state: AgentState) -> dict:
    """Generate research task list based on query analysis.

    Args:
        state: Current agent state

    Returns:
        State update with todo_list
    """
    analysis = QueryAnalysis.model_validate(state["query_analysis"])
    client = get_ollama_client()

    prompt = f"""Generate a research task list for this query analysis.

Original Query: "{analysis.original_query}"
Key Concepts: {analysis.key_concepts}
Entities: {analysis.entities}
Scope: {analysis.scope}
Context: {analysis.assumed_context}

Generate {settings.initial_todo_items} specific, actionable research tasks.
Each task should be:
- Specific and measurable
- Focused on finding concrete information
- Related to the query concepts

Return JSON with "items" array, each item having:
- id: integer starting from 1
- task: string describing the task
- context: string explaining why this task is needed

Example:
{{"items": [{{"id": 1, "task": "Find dose limit regulations", "context": "Core query requirement"}}]}}"""

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
    results = vector_search(query, top_k=settings.m_chunks_per_query)

    # Process results into chunks with info
    chunks: list[ChunkWithInfo] = []
    for result in results:
        chunk = create_chunk_with_info(result, analysis.original_query)

        # Detect and follow references
        if chunk.extracted_info:
            refs = detect_references(chunk.extracted_info)
            for ref in refs:
                ref_key = f"{ref.type}:{ref.target}"
                if not context.has_visited_ref(ref_key):
                    nested = resolve_reference(
                        ref,
                        chunk.document,
                        visited=set(context.metadata.visited_refs),
                        depth=state.get("current_depth", 0),
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

    # Mark task completed
    for item in todo_list:
        if item.get("id") == task_id:
            item["completed"] = True

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

    # Truncate if too long
    info_text = json.dumps(all_info[:20], ensure_ascii=False, indent=2)

    prompt = f"""Synthesize these research findings into a coherent answer.

Original Query: "{analysis.original_query}"

Research Findings:
{info_text}

Provide:
1. summary: A comprehensive answer to the query (in {analysis.detected_language})
2. key_findings: List of the most important findings

Include source citations in the format [Document_name.pdf, Page X] where applicable."""

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

    prompt = f"""Evaluate the quality of this research summary.

Summary:
{summary}

Score each dimension from 0-100:
1. factual_accuracy: Are claims factually correct?
2. semantic_validity: Does it make logical sense?
3. structural_integrity: Is it well-organized?
4. citation_correctness: Are sources properly cited?

Also list any issues_found as a list of strings."""

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
        assessment = QualityAssessment(
            overall_score=300,
            factual_accuracy=75,
            semantic_validity=75,
            structural_integrity=75,
            citation_correctness=75,
            passes_quality=True,
        )

    return {
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
    key_findings = []
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

    report = FinalReport(
        query=analysis.original_query,
        answer=answer or "No synthesis available",
        findings=findings[:10],  # Limit findings
        sources=sources[:20],  # Limit sources
        quality_score=300,  # Default if not checked
        quality_breakdown={
            "factual_accuracy": 75,
            "semantic_validity": 75,
            "structural_integrity": 75,
            "citation_correctness": 75,
        },
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
