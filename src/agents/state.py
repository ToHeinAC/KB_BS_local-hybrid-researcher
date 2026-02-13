"""Agent state definition using TypedDict (LangChain v1.0 requirement)."""

from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """State for the research agent.

    Uses TypedDict as required by LangGraph v1.0.
    Pydantic models are serialized to dicts when stored here.
    """

    # Core fields
    query: str  # User's research question
    query_analysis: dict  # Serialized QueryAnalysis
    todo_list: list[dict]  # Serialized list of ToDoItems
    research_context: dict  # Serialized ResearchContext
    final_report: dict  # Serialized FinalReport

    # Task tracking
    current_task_id: int | None
    completed_task_ids: list[int]

    # Phase control
    phase: str  # Current phase name
    iteration: int  # Current iteration count

    # Reference tracking (for loop prevention)
    visited_refs: set[str]
    current_depth: int

    # Quality assessment from quality_check phase
    quality_assessment: dict | None

    # HITL checkpoint fields (for graph interrupts)
    hitl_pending: bool
    hitl_checkpoint: dict | None  # Serialized HITLCheckpoint
    hitl_decision: dict | None  # Serialized HITLDecision

    # Phase 1: Iterative HITL fields
    hitl_state: dict | None  # Chat-style HITL conversation state
    hitl_iteration: int  # Current HITL iteration count (0-indexed)
    hitl_max_iterations: int  # Maximum HITL iterations (default 5)
    hitl_conversation_history: list[dict]  # Full conversation history
    hitl_active: bool  # Whether iterative HITL is active
    hitl_termination_reason: str | None  # Why HITL ended: user_end, max_iterations, convergence

    # Enhanced Phase 1: Multi-query retrieval tracking
    retrieval_history: dict  # Per-iteration retrieval results
    # Structure: {"iteration_N": {"queries": [...], "retrieved_chunks": [...], "dedup_stats": {...}}}
    query_retrieval: str  # Accumulated retrieval results (filtered)

    # Enhanced Phase 1: Convergence tracking
    coverage_score: float  # 0-1 estimate of information coverage
    convergence_score: float  # 0-1 convergence to stable state
    retrieval_quality_history: list[float]  # Dedup ratios per iteration

    # Enhanced Phase 1: Token budget tracking
    total_tokens_used: int  # Estimated tokens used
    max_tokens_allowed: int  # Budget constraint (default 4000)

    # Enhanced Phase 1: Multi-vector retrieval tracking
    iteration_queries: list[list[str]]  # Queries per iteration [[q1, q2, q3], ...]
    knowledge_gaps: list[str]  # Identified gaps from retrieval analysis
    retrieval_dedup_ratios: list[float]  # Dedup ratio per iteration for convergence

    # Message accumulation (for debugging/logging)
    messages: Annotated[list, add]

    # Error handling
    error: str | None
    warnings: list[str]

    # UI settings (passed from Streamlit)
    selected_database: str | None  # Specific database to search
    k_results: int  # Number of results per search query

    # HITL handoff fields (output from iterative HITL)
    research_queries: list[str]  # From HITL conversation
    additional_context: str  # From HITL analysis summary
    detected_language: str  # Language detected from user query

    # Agentic decision fields
    synthesis_retry_count: int  # Number of synthesis retries (max 1)
    quality_remediation_focus: str  # Focus instructions for re-synthesis

    # Graded Context Management (Phase A-E improvements)
    query_anchor: dict  # Immutable reference to original intent
    # Structure: {"original_query": str, "detected_language": str,
    #             "key_entities": list[str], "scope": str,
    #             "hitl_refinements": list[str], "created_at": str}
    hitl_smry: str  # Citation-aware HITL summary with [Source_filename] annotations
    primary_context: list[dict]  # Tier 1: Direct, high-relevance findings
    secondary_context: list[dict]  # Tier 2: Reference-followed, medium-relevance
    tertiary_context: list[dict]  # Tier 3: Deep references, HITL retrieval
    task_summaries: list[dict]  # Per-task structured summaries
    # Structure: {"task_id": int, "summary": str, "key_findings": list[str],
    #             "preserved_quotes": list[dict], "sources": list[str],
    #             "relevance_to_query": float}
    preserved_quotes: list[dict]  # Critical verbatim quotes
    # Structure: {"quote": str, "source": str, "page": int,
    #             "relevance_reason": str}


def create_initial_state(query: str) -> AgentState:
    """Create initial agent state for a new research session.

    Args:
        query: User's research question

    Returns:
        Initialized AgentState
    """
    return AgentState(
        query=query,
        query_analysis={},
        todo_list=[],
        research_context={
            "search_queries": [],
            "metadata": {
                "total_iterations": 0,
                "documents_referenced": [],
                "external_sources_used": False,
                "visited_refs": [],
            },
        },
        final_report={},
        current_task_id=None,
        completed_task_ids=[],
        phase="hitl_init",
        iteration=0,
        visited_refs=set(),
        current_depth=0,
        quality_assessment=None,
        # HITL checkpoint fields
        hitl_pending=False,
        hitl_checkpoint=None,
        hitl_decision=None,
        # Phase 1: Iterative HITL
        hitl_state=None,
        hitl_iteration=0,
        hitl_max_iterations=5,
        hitl_conversation_history=[],
        hitl_active=True,
        hitl_termination_reason=None,
        # Phase 1: Multi-query retrieval
        retrieval_history={},
        query_retrieval="",
        # Phase 1: Convergence
        coverage_score=0.0,
        convergence_score=0.0,
        retrieval_quality_history=[],
        # Phase 1: Token budget
        total_tokens_used=0,
        max_tokens_allowed=4000,
        # Phase 1: Multi-vector retrieval tracking
        iteration_queries=[],
        knowledge_gaps=[],
        retrieval_dedup_ratios=[],
        # General
        messages=[],
        error=None,
        warnings=[],
        research_queries=[],
        additional_context="",
        detected_language="de",
        # Agentic decision fields
        synthesis_retry_count=0,
        quality_remediation_focus="",
        # Graded Context Management
        query_anchor={},
        hitl_smry="",
        primary_context=[],
        secondary_context=[],
        tertiary_context=[],
        task_summaries=[],
        preserved_quotes=[],
    )


def get_phase(state: AgentState) -> str:
    """Get current phase from state."""
    return state.get("phase", "analyze")


def set_phase(state: AgentState, phase: str) -> AgentState:
    """Set phase in state and return updated state."""
    return {**state, "phase": phase}


def is_hitl_pending(state: AgentState) -> bool:
    """Check if HITL interaction is pending."""
    return state.get("hitl_pending", False)


def get_pending_tasks(state: AgentState) -> list[dict]:
    """Get pending tasks from todo_list."""
    todo_list = state.get("todo_list", [])
    return [item for item in todo_list if not item.get("completed", False)]


def get_next_task_id(state: AgentState) -> int | None:
    """Get ID of next pending task."""
    pending = get_pending_tasks(state)
    if pending:
        return pending[0].get("id")
    return None


# --- Phase 1: Iterative HITL Helpers ---


def is_iterative_hitl_active(state: AgentState) -> bool:
    """Check if iterative HITL is currently active."""
    return state.get("hitl_active", False)


def get_hitl_iteration(state: AgentState) -> int:
    """Get current HITL iteration count."""
    return state.get("hitl_iteration", 0)


def should_continue_hitl(state: AgentState) -> bool:
    """Determine if HITL iteration should continue.

    Returns False if:
    - hitl_active is False
    - Max iterations reached
    - Convergence detected (coverage > 0.9 and high dedup ratio)
    """
    if not state.get("hitl_active", False):
        return False

    iteration = state.get("hitl_iteration", 0)
    max_iterations = state.get("hitl_max_iterations", 5)
    if iteration >= max_iterations:
        return False

    # Check convergence
    coverage = state.get("coverage_score", 0.0)
    quality_history = state.get("retrieval_quality_history", [])
    if quality_history and coverage >= 0.9:
        recent_dedup = quality_history[-1] if quality_history else 0
        if recent_dedup >= 0.75:
            return False

    return True


def get_hitl_conversation(state: AgentState) -> list[dict]:
    """Get HITL conversation history."""
    return state.get("hitl_conversation_history", [])


def add_hitl_message(state: AgentState, role: str, content: str) -> list[dict]:
    """Add a message to HITL conversation history.

    Args:
        state: Current state
        role: Message role (user/assistant)
        content: Message content

    Returns:
        Updated conversation history
    """
    history = list(state.get("hitl_conversation_history", []))
    history.append({
        "role": role,
        "content": content,
        "iteration": state.get("hitl_iteration", 0),
    })
    return history
