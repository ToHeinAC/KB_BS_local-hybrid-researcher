"""LangGraph StateGraph definition for the research agent."""

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    attribute_sources,
    execute_task,
    generate_todo,
    hitl_approve_todo,
    process_hitl_todo,
    quality_check,
    synthesize,
    validate_relevance,
    # Enhanced Phase 1: Iterative HITL nodes
    hitl_init,
    hitl_generate_questions,
    hitl_process_response,
    hitl_finalize,
    # Enhanced Phase 1: Multi-vector retrieval nodes
    hitl_generate_queries,
    hitl_retrieve_chunks,
    hitl_analyze_retrieval,
)
from src.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_entry_point(
    state: AgentState,
) -> Literal["hitl_init", "hitl_process_response", "generate_todo", "process_hitl_todo"]:
    """Route at entry point: iterative HITL, todo processing, or skip to todo.

    Routes to:
    - process_hitl_todo: if resuming from todo approval (hitl_decision present, HITL not active)
    - generate_todo: if research_queries already populated (from UI chat-based HITL)
    - hitl_process_response: if resuming HITL with a decision (user responded)
    - hitl_init: if starting new iterative HITL (default)
    """
    # Check if resuming from todo approval (hitl_decision present, not iterative HITL)
    if state.get("hitl_decision") and not state.get("hitl_active", False):
        return "process_hitl_todo"

    # Check for HITL resume: decision present + hitl_active means user responded
    if state.get("hitl_decision") and state.get("hitl_active", False):
        return "hitl_process_response"

    # Check if HITL results are already available (from UI chat-based HITL)
    if state.get("research_queries"):
        return "generate_todo"

    # Check if phase is explicitly set to skip to todo
    phase = state.get("phase", "")
    if phase == "generate_todo":
        return "generate_todo"

    # Default: start iterative HITL
    return "hitl_init"


def route_after_hitl_init(state: AgentState) -> Literal["hitl_generate_queries"]:
    """Route after HITL initialization - go to query generation."""
    return "hitl_generate_queries"


def route_after_hitl_generate_queries(
    state: AgentState,
) -> Literal["hitl_retrieve_chunks"]:
    """Route after query generation - go to retrieval."""
    return "hitl_retrieve_chunks"


def route_after_hitl_retrieve_chunks(
    state: AgentState,
) -> Literal["hitl_analyze_retrieval"]:
    """Route after retrieval - go to analysis."""
    return "hitl_analyze_retrieval"


def route_after_hitl_analyze_retrieval(
    state: AgentState,
) -> Literal["hitl_generate_questions"]:
    """Route after analysis - go to question generation."""
    return "hitl_generate_questions"


def route_after_hitl_generate_questions(
    state: AgentState,
) -> Literal["__end__"]:
    """Route after HITL question generation - always wait for user."""
    # Always end to wait for user response
    return "__end__"


def route_after_hitl_process_response(
    state: AgentState,
) -> Literal["hitl_generate_queries", "hitl_finalize"]:
    """Route after processing HITL response.
    
    Routes to:
    - hitl_finalize: if user typed /end, max iterations, or convergence
    - hitl_generate_queries: to continue the retrieval loop
    """
    phase = state.get("phase", "")
    
    # Check termination conditions
    if phase == "hitl_finalize":
        return "hitl_finalize"
    
    if not state.get("hitl_active", False):
        return "hitl_finalize"
    
    # Check for convergence
    coverage = state.get("coverage_score", 0.0)
    dedup_ratios = state.get("retrieval_dedup_ratios", [])
    recent_dedup = dedup_ratios[-1] if dedup_ratios else 0.0
    gaps = len(state.get("knowledge_gaps", []))
    
    if coverage >= 0.80 and recent_dedup >= 0.70 and gaps <= 2:
        # Convergence detected - suggest finalization
        return "hitl_finalize"
    
    # Continue to next iteration via query generation
    return "hitl_generate_queries"


def route_after_hitl_finalize(state: AgentState) -> Literal["generate_todo"]:
    """Route after HITL finalization."""
    return "generate_todo"


def route_after_generate_todo(
    state: AgentState,
) -> Literal["hitl_approve_todo"]:
    """Route after todo generation."""
    return "hitl_approve_todo"


def route_after_hitl_todo(
    state: AgentState,
) -> Literal["process_hitl_todo", "execute_task", "__end__"]:
    """Route after HITL todo approval checkpoint."""
    if state.get("hitl_pending"):
        # Wait for user input - end this invocation
        return "__end__"
    if state.get("hitl_decision"):
        return "process_hitl_todo"
    return "execute_task"


def route_after_execute(
    state: AgentState,
) -> Literal["execute_task", "validate_relevance"]:
    """Route after task execution."""
    phase = state.get("phase", "")
    if phase == "execute_tasks" and state.get("current_task_id") is not None:
        return "execute_task"
    return "validate_relevance"


def route_after_validate_relevance(
    state: AgentState,
) -> Literal["synthesize"]:
    """Route after relevance validation - always go to synthesis."""
    return "synthesize"


def route_after_synthesize(state: AgentState) -> Literal["quality_check"]:
    """Route after synthesis."""
    return "quality_check"


def route_after_quality(
    state: AgentState,
) -> Literal["attribute_sources"]:
    """Route after quality check."""
    return "attribute_sources"


def _entry_router(state: AgentState) -> dict:
    """Entry router node - just passes through state."""
    return {}


def create_research_graph() -> StateGraph:
    """Create the research agent StateGraph.

    Returns:
        Compiled StateGraph with checkpointer for HITL support
    """
    # Create graph with AgentState
    graph = StateGraph(AgentState)

    # Add entry router node for conditional routing
    graph.add_node("entry_router", _entry_router)

    # Add nodes - Phase 1: Iterative HITL
    graph.add_node("hitl_init", hitl_init)
    graph.add_node("hitl_generate_queries", hitl_generate_queries)
    graph.add_node("hitl_retrieve_chunks", hitl_retrieve_chunks)
    graph.add_node("hitl_analyze_retrieval", hitl_analyze_retrieval)
    graph.add_node("hitl_generate_questions", hitl_generate_questions)
    graph.add_node("hitl_process_response", hitl_process_response)
    graph.add_node("hitl_finalize", hitl_finalize)

    # Add nodes - Phase 2 onwards
    graph.add_node("generate_todo", generate_todo)
    graph.add_node("hitl_approve_todo", hitl_approve_todo)
    graph.add_node("process_hitl_todo", process_hitl_todo)
    graph.add_node("execute_task", execute_task)
    graph.add_node("validate_relevance", validate_relevance)  # Phase G
    graph.add_node("synthesize", synthesize)
    graph.add_node("quality_check", quality_check)
    graph.add_node("attribute_sources", attribute_sources)

    # Set entry point to router
    graph.set_entry_point("entry_router")

    # Route from entry based on state
    graph.add_conditional_edges(
        "entry_router",
        route_entry_point,
        {
            "hitl_init": "hitl_init",
            "hitl_process_response": "hitl_process_response",
            "generate_todo": "generate_todo",
            "process_hitl_todo": "process_hitl_todo",
        },
    )

    # === Enhanced Phase 1: Iterative HITL Flow with Vector Retrieval ===
    # Flow: hitl_init → hitl_generate_queries → hitl_retrieve_chunks →
    #       hitl_analyze_retrieval → hitl_generate_questions → END (wait for user)
    # Resume: hitl_process_response → hitl_generate_queries (loop) OR hitl_finalize

    # hitl_init → hitl_generate_queries
    graph.add_conditional_edges(
        "hitl_init",
        route_after_hitl_init,
        {
            "hitl_generate_queries": "hitl_generate_queries",
        },
    )

    # hitl_generate_queries → hitl_retrieve_chunks
    graph.add_conditional_edges(
        "hitl_generate_queries",
        route_after_hitl_generate_queries,
        {
            "hitl_retrieve_chunks": "hitl_retrieve_chunks",
        },
    )

    # hitl_retrieve_chunks → hitl_analyze_retrieval
    graph.add_conditional_edges(
        "hitl_retrieve_chunks",
        route_after_hitl_retrieve_chunks,
        {
            "hitl_analyze_retrieval": "hitl_analyze_retrieval",
        },
    )

    # hitl_analyze_retrieval → hitl_generate_questions
    graph.add_conditional_edges(
        "hitl_analyze_retrieval",
        route_after_hitl_analyze_retrieval,
        {
            "hitl_generate_questions": "hitl_generate_questions",
        },
    )

    # hitl_generate_questions → END (wait for user)
    graph.add_conditional_edges(
        "hitl_generate_questions",
        route_after_hitl_generate_questions,
        {
            "__end__": END,
        },
    )

    # hitl_process_response → hitl_generate_queries (continue loop) OR hitl_finalize
    graph.add_conditional_edges(
        "hitl_process_response",
        route_after_hitl_process_response,
        {
            "hitl_generate_queries": "hitl_generate_queries",
            "hitl_finalize": "hitl_finalize",
        },
    )

    # hitl_finalize → generate_todo
    graph.add_conditional_edges(
        "hitl_finalize",
        route_after_hitl_finalize,
        {
            "generate_todo": "generate_todo",
        },
    )

    # Phase 2: ToDo Generation
    graph.add_conditional_edges(
        "generate_todo",
        route_after_generate_todo,
        {
            "hitl_approve_todo": "hitl_approve_todo",
        },
    )

    # HITL ToDo Approval
    graph.add_conditional_edges(
        "hitl_approve_todo",
        route_after_hitl_todo,
        {
            "__end__": END,
            "process_hitl_todo": "process_hitl_todo",
            "execute_task": "execute_task",
        },
    )
    graph.add_edge("process_hitl_todo", "execute_task")

    # Phase 3: Task Execution
    graph.add_conditional_edges(
        "execute_task",
        route_after_execute,
        {
            "execute_task": "execute_task",
            "validate_relevance": "validate_relevance",
        },
    )

    # Phase 3.5: Pre-Synthesis Relevance Validation (Phase G)
    graph.add_conditional_edges(
        "validate_relevance",
        route_after_validate_relevance,
        {
            "synthesize": "synthesize",
        },
    )

    # Phase 4: Synthesis and Quality
    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesize,
        {
            "quality_check": "quality_check",
        },
    )

    graph.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {
            "attribute_sources": "attribute_sources",
        },
    )

    # Phase 5: Source Attribution -> END
    graph.add_edge("attribute_sources", END)

    # Compile with memory checkpointer for HITL resume capability
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info("Research graph compiled successfully")
    return compiled


def run_research(
    query: str,
    config: dict | None = None,
) -> dict:
    """Run a complete research session.

    Args:
        query: User's research question
        config: Optional LangGraph config (for thread_id, etc.)

    Returns:
        Final agent state with report
    """
    from src.agents.state import create_initial_state

    graph = create_research_graph()
    initial_state = create_initial_state(query)

    config = config or {"configurable": {"thread_id": "default"}}

    # Run graph
    result = graph.invoke(initial_state, config)

    return result


def resume_research(
    thread_id: str,
    hitl_decision: dict,
) -> dict:
    """Resume research after HITL interaction.

    Args:
        thread_id: Thread ID from previous invocation
        hitl_decision: User's decision from HITL checkpoint

    Returns:
        Updated agent state
    """
    graph = create_research_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Resume iterative HITL: process the response
    state_update = {
        "hitl_decision": hitl_decision,
        "hitl_pending": False,
    }
    # Invoke hitl_process_response directly
    result = graph.invoke(state_update, config, interrupt_before=["hitl_process_response"])
    if result.get("hitl_pending"):
        return result
    # Continue from process_response
    return graph.invoke(None, config)


def resume_iterative_hitl(
    thread_id: str,
    user_response: str,
) -> dict:
    """Resume iterative HITL with user's response.

    This is the primary way to continue an iterative HITL conversation.

    Args:
        thread_id: Thread ID from previous invocation
        user_response: User's text response (or '/end' to finalize)

    Returns:
        Updated agent state
    """
    graph = create_research_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Create decision with user response
    hitl_decision = {
        "approved": True,
        "modifications": {
            "user_response": user_response,
        },
    }

    # Update state with user response
    state_update = {
        "hitl_decision": hitl_decision,
        "hitl_pending": False,
    }

    # Get current state and find next node
    current_state = graph.get_state(config)

    if current_state and current_state.values:
        # Add the decision to resume
        merged_state = {**current_state.values, **state_update}

        # Invoke the process_response node
        result = graph.invoke(merged_state, config)
        return result

    # Fallback: direct invoke
    return graph.invoke(state_update, config)
