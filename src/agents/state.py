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

    # HITL fields
    hitl_pending: bool
    hitl_checkpoint: dict | None  # Serialized HITLCheckpoint
    hitl_history: list[dict]  # History of HITL interactions
    hitl_decision: dict | None  # Serialized HITLDecision

    # Message accumulation (for debugging/logging)
    messages: Annotated[list, add]

    # Error handling
    error: str | None
    warnings: list[str]

    # UI settings (passed from Streamlit)
    selected_database: str | None  # Specific database to search
    k_results: int  # Number of results per search query


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
        phase="analyze",
        iteration=0,
        visited_refs=set(),
        current_depth=0,
        hitl_pending=False,
        hitl_checkpoint=None,
        hitl_history=[],
        hitl_decision=None,
        messages=[],
        error=None,
        warnings=[],
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
