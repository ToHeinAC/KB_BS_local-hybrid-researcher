"""LangGraph StateGraph definition for the research agent."""

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.nodes import (
    analyze_query,
    attribute_sources,
    execute_task,
    generate_todo,
    hitl_approve_todo,
    hitl_clarify,
    process_hitl_clarify,
    process_hitl_todo,
    quality_check,
    synthesize,
)
from src.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_after_analyze(state: AgentState) -> Literal["hitl_clarify", "generate_todo"]:
    """Route after query analysis."""
    phase = state.get("phase", "")
    if phase == "hitl_clarify":
        return "hitl_clarify"
    return "generate_todo"


def route_after_hitl_clarify(
    state: AgentState,
) -> Literal["process_hitl_clarify", "generate_todo", "__end__"]:
    """Route after HITL clarification checkpoint."""
    if state.get("hitl_pending"):
        # Wait for user input - end this invocation
        return "__end__"
    if state.get("hitl_decision"):
        return "process_hitl_clarify"
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
) -> Literal["execute_task", "synthesize"]:
    """Route after task execution."""
    phase = state.get("phase", "")
    if phase == "execute_tasks" and state.get("current_task_id") is not None:
        return "execute_task"
    return "synthesize"


def route_after_synthesize(state: AgentState) -> Literal["quality_check"]:
    """Route after synthesis."""
    return "quality_check"


def route_after_quality(
    state: AgentState,
) -> Literal["attribute_sources"]:
    """Route after quality check."""
    return "attribute_sources"


def create_research_graph() -> StateGraph:
    """Create the research agent StateGraph.

    Returns:
        Compiled StateGraph with checkpointer for HITL support
    """
    # Create graph with AgentState
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("hitl_clarify", hitl_clarify)
    graph.add_node("process_hitl_clarify", process_hitl_clarify)
    graph.add_node("generate_todo", generate_todo)
    graph.add_node("hitl_approve_todo", hitl_approve_todo)
    graph.add_node("process_hitl_todo", process_hitl_todo)
    graph.add_node("execute_task", execute_task)
    graph.add_node("synthesize", synthesize)
    graph.add_node("quality_check", quality_check)
    graph.add_node("attribute_sources", attribute_sources)

    # Set entry point
    graph.set_entry_point("analyze_query")

    # Add edges with routing

    # Phase 1: Query Analysis
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {
            "hitl_clarify": "hitl_clarify",
            "generate_todo": "generate_todo",
        },
    )

    # HITL Clarification
    graph.add_conditional_edges(
        "hitl_clarify",
        route_after_hitl_clarify,
        {
            "__end__": END,
            "process_hitl_clarify": "process_hitl_clarify",
            "generate_todo": "generate_todo",
        },
    )
    graph.add_edge("process_hitl_clarify", "generate_todo")

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


def run_research(query: str, config: dict | None = None) -> dict:
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

    # Update state with decision
    state_update = {
        "hitl_decision": hitl_decision,
        "hitl_pending": False,
    }

    # Resume from checkpoint
    result = graph.invoke(state_update, config)

    return result
