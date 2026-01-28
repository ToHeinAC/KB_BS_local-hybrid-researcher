# Agent Design

## LangChain v1.0+ Compatibility

This project targets **LangChain v1.0+** and **LangGraph v1.0+**.

**Key v1.0 changes:**
- Agent state MUST use `TypedDict` (not Pydantic models for state)
- Pydantic models are still used for tool inputs/outputs and data structures
- LangGraph is now the foundational runtime for stateful orchestration
- Legacy chains moved to `langchain-classic` (not needed here)

## LangGraph StateGraph Pattern

We use **LangGraph StateGraph** for explicit state management and phase control:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# Agent state MUST be TypedDict (LangChain v1.0 requirement)
class AgentState(TypedDict):
    query: str
    query_analysis: dict  # Serialized QueryAnalysis
    todo_list: list[dict]  # Serialized ToDoItems
    research_context: dict  # Serialized ResearchContext
    final_report: dict  # Serialized FinalReport
    current_task_id: int | None
    phase: str
    messages: Annotated[list, add]  # For message accumulation

# Build graph
graph = StateGraph(AgentState)

# Nodes (as implemented in src/agents/graph.py)

# Entry router for conditional routing
graph.add_node("entry_router", entry_router)

# Enhanced Phase 1: Iterative HITL nodes (NEW)
graph.add_node("hitl_init", hitl_init)
graph.add_node("hitl_generate_questions", hitl_generate_questions)
graph.add_node("hitl_process_response", hitl_process_response)
graph.add_node("hitl_finalize", hitl_finalize)

# Legacy Phase 1 nodes
graph.add_node("analyze_query", analyze_query)
graph.add_node("hitl_clarify", hitl_clarify)
graph.add_node("process_hitl_clarify", process_hitl_clarify)

# Phase 2+ nodes
graph.add_node("generate_todo", generate_todo)
graph.add_node("hitl_approve_todo", hitl_approve_todo)
graph.add_node("process_hitl_todo", process_hitl_todo)
graph.add_node("execute_task", execute_task)
graph.add_node("synthesize", synthesize)
graph.add_node("quality_check", quality_check)
graph.add_node("attribute_sources", attribute_sources)

# Compile (with checkpointer for HITL resume)
app = graph.compile()
```

### State Serialization Pattern

Since agent state must be TypedDict, serialize Pydantic models:

```python
from src.models.query import QueryAnalysis

def analyze_query_node(state: AgentState) -> dict:
    """Analyze query and return updated state."""
    # Create Pydantic model for validation
    analysis = QueryAnalysis(
        original_query=state["query"],
        key_concepts=extract_concepts(state["query"]),
        # ...
    )
    # Serialize to dict for state
    return {"query_analysis": analysis.model_dump()}

def get_query_analysis(state: AgentState) -> QueryAnalysis:
    """Deserialize QueryAnalysis from state."""
    return QueryAnalysis.model_validate(state["query_analysis"])
```

## Ollama Structured Outputs

For Ollama models <30B parameters, use `method="json_mode"`:

```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel

class QueryAnalysis(BaseModel):
    key_concepts: list[str]
    entities: list[str]
    scope: str
    clarification_needed: bool

llm = ChatOllama(model="qwen3:14b", temperature=0)

# Correct for Ollama
structured_llm = llm.with_structured_output(QueryAnalysis, method="json_mode")

result = structured_llm.invoke("Analyze this query: ...")
# result is a QueryAnalysis instance
```

## The Rabbithole Magic (Phase 3)

The core innovation: iterative reference following with context enrichment.

### Step A: Information Extraction

For each chunk retrieved from vector DB:
- Extract only passages relevant to the search query
- Condense (don't copy verbatim)
- Store in `chunk.extracted_info`

```python
def extract_info(chunk: str, query: str, llm: ChatOllama) -> str:
    """Extract relevant information from chunk relative to query."""
    prompt = f"""Given this search query: "{query}"

Extract only the relevant passages from this text chunk:
{chunk}

Return condensed, relevant information only."""

    return llm.invoke(prompt).content
```

### Step B: Reference Detection

Within `extracted_info`, detect reference patterns:
- "see section X" / "siehe Abschnitt X"
- "cf. § Y" / "gemäß § Y"
- "see document Z" / "siehe Dokument Z"
- "Article N of [regulation]"

```python
class DetectedReference(BaseModel):
    type: Literal["section", "document", "external"]
    target: str
    original_text: str

def detect_references(text: str, llm: ChatOllama) -> list[DetectedReference]:
    """Detect references within extracted info."""
    structured_llm = llm.with_structured_output(
        list[DetectedReference], method="json_mode"
    )
    # ...
```

### Step C: Reference Resolution & Following

For each detected reference:

| Reference Type | Resolution Strategy |
|---------------|---------------------|
| Section | Regex search in same document PDF |
| Document | Use document_mapping.json or filename grep |
| External | Mark for optional web search phase |

```python
def resolve_reference(
    ref: DetectedReference,
    current_doc: str,
    doc_mapping: dict,
) -> list[ResolvedChunk]:
    """Resolve reference and retrieve content."""

    if ref.type == "section":
        # Search same document for section
        return search_section_in_doc(current_doc, ref.target)

    elif ref.type == "document":
        # Map to file and retrieve
        target_file = doc_mapping.get(ref.target)
        if target_file:
            return retrieve_chunks_from_doc(target_file)
        # Fallback: grep for similar filenames
        return grep_for_document(ref.target)

    else:  # external
        return []  # Handled in Phase 4 web search
```

### Step D: Relevance Filtering

For each nested chunk from resolved references:
- Score relevance to original search query (0.0-1.0)
- Include only if score > REFERENCE_RELEVANCE_THRESHOLD (0.6)
- Extract and condense relevant information

```python
class ScoredChunk(BaseModel):
    chunk: str
    document: str
    relevance_score: float
    extracted_info: str

def filter_by_relevance(
    chunks: list[str],
    query: str,
    threshold: float = 0.6,
) -> list[ScoredChunk]:
    """Filter chunks by relevance to query."""
    scored = []
    for chunk in chunks:
        score = compute_relevance(chunk, query)
        if score >= threshold:
            scored.append(ScoredChunk(
                chunk=chunk,
                relevance_score=score,
                extracted_info=extract_info(chunk, query),
            ))
    return scored
```

### Step E: ToDoList Re-evaluation

After reference resolution:
1. Mark current task as `completed=True`
2. Check if new information suggests additional research tasks
3. Add new tasks if < TODO_MAX_ITEMS

```python
def reevaluate_todo(
    current_task: ToDoItem,
    research_context: ResearchContext,
    todo_list: ToDoList,
    llm: ChatOllama,
) -> ToDoList:
    """Re-evaluate ToDoList based on new findings."""

    # Mark current as done
    current_task.completed = True

    # Check for new tasks
    if len(todo_list.items) < todo_list.max_items:
        new_tasks = identify_new_tasks(research_context, llm)
        todo_list.items.extend(new_tasks)

    return todo_list
```

## Tool Definitions

### Vector Search Tool

```python
@tool
def vector_search(
    query: str,
    collections: list[str] | None = None,
    top_k: int | None = None,
    selected_database: str | None = None,
) -> list[VectorResult]:
    """Search ChromaDB collection for relevant chunks.

    Args:
        query: Search query text
        collections: Collections to search (defaults to all)
        top_k: Number of results per collection
        selected_database: Specific database directory name to search

    Returns:
        List of matching chunks with metadata
    """
```

## Streamlit UI Execution Model

The Streamlit UI runs the graph using LangGraph streaming so the user sees live progress updates and preliminary results:

```python
for state in graph.stream(input_state, config, stream_mode="values"):
    update_agent_state(state)
    render_research_status()
```

### Extract References Tool

```python
@tool
def extract_references(text: str) -> list[DetectedReference]:
    """Detect references within text.

    Identifies patterns like:
    - "see section X"
    - "cf. § Y"
    - "see document Z"

    Returns:
        List of detected references with type and target
    """
```

### Resolve Reference Tool

```python
@tool
def resolve_reference(
    ref: DetectedReference,
    current_doc: str,
) -> list[ResolvedChunk]:
    """Find content for detected reference.

    Args:
        ref: The detected reference
        current_doc: Document where reference was found

    Returns:
        Retrieved chunks from resolved reference
    """
```

### Generate Summary Tool

```python
@tool
def generate_summary(
    chunks: list[ChunkWithInfo],
    query: str,
) -> QuerySummary:
    """Synthesize comprehensive answer from chunks.

    Maintains original terminology and includes citations.

    Returns:
        Summary with text and source references
    """
```

### Quality Check Tool

```python
@tool
def quality_check(summary: QuerySummary) -> QualityAssessment:
    """Validate summary completeness and accuracy.

    Scoring dimensions (0-100 each, 0-400 total):
    - Factual accuracy
    - Semantic validity
    - Structural integrity
    - Citation correctness

    Returns:
        Quality assessment with scores and issues
    """
```

### Web Search Tool (Optional)

```python
@tool
def web_search(query: str, max_results: int = 2) -> list[WebResult]:
    """Search external sources via Tavily.

    Only enabled when ENABLE_WEB_SEARCH=true.
    Used to fill gaps from unresolved external references.
    """
```

**Note:** Web search is currently not wired into the baseline graph/UI in this repository.

## HITL Integration Points

### Enhanced: Iterative Retrieval-HITL (Phase 1) - NEW

The iterative HITL flow provides conversational query refinement **integrated with vector DB retrieval**:

```
hitl_init → hitl_generate_queries → hitl_retrieve_chunks → hitl_analyze_retrieval → hitl_generate_questions ↔ hitl_process_response → hitl_finalize
```

**Nodes:**

1. **hitl_init**: Initialize conversation, detect language (de/en)
   - Sets `hitl_active=True`, `hitl_iteration=0`

2. **hitl_generate_queries**: Generate 3 queries (original + 2 alternatives)
   - Iteration 0: standard discovery queries
   - Iteration N>0: refined queries based on user feedback + gaps

3. **hitl_retrieve_chunks**: Execute vector search for all 3 queries
   - Retrieves ~9 chunks per iteration
   - Deduplicates findings against existing `query_retrieval` context

4. **hitl_analyze_retrieval**: LLM analysis of retrieved context
   - Identifies `knowledge_gaps` and computes `coverage_score` (0-1)

5. **hitl_generate_questions**: Generate 2-3 contextual follow-ups
   - Targeted questions focused on filling identified `knowledge_gaps`
   - Creates `hitl_checkpoint` with `checkpoint_type="iterative_hitl"`
   - Sets `hitl_pending=True`, routes to END

6. **hitl_process_response**: Analyze user response
   - Extracts insights from user feedback
   - Checks termination: `/end`, max_iterations, convergence
   - Increments `hitl_iteration` and loops back to query generation

7. **hitl_finalize**: Generate research_queries for Phase 2
   - Builds `query_analysis` from final conversation and retrieval state
   - Sets `research_queries[]`, `additional_context`

**Termination (Convergence) Conditions:**
- User types `/end` → `hitl_termination_reason="user_end"`
- `hitl_iteration >= hitl_max_iterations` → `hitl_termination_reason="max_iterations"`
- **Convergence Detection (3-tier)**:
  - `coverage_score >= 0.8`
  - `retrieval_dedup_ratio >= 0.7` (high overlap indicates search stability)
  - `len(knowledge_gaps) <= 2`
  - Sets `hitl_termination_reason="convergence"`

**Entry Routing:**

The `route_entry_point` function decides which flow to use:

```python
def route_entry_point(state):
    # Skip to Phase 2 if HITL already done
    if state.get("research_queries"):
        return "generate_todo"
    # Use iterative HITL
    if state.get("hitl_active", False):
        return "hitl_init"
    # Legacy flow
    return "analyze_query"
```

### Legacy: Query Clarification (Phase 1)

The graph emits a HITL checkpoint when clarification is needed:

- **Node:** `hitl_clarify`
- **State:** `hitl_pending=True` and `hitl_checkpoint` populated

The Streamlit UI renders the checkpoint and resumes the graph with a `hitl_decision`.

### 2. ToDoList Approval (Phase 2)

The graph emits a checkpoint for ToDo approval:

- **Node:** `hitl_approve_todo`
- **State:** `hitl_pending=True` and `hitl_checkpoint` populated

The Streamlit UI uses the ToDo approval component and resumes with `process_hitl_todo`.

## Loop Prevention

To prevent infinite loops:

```python
# Configuration constants
TODO_MAX_ITEMS = 15           # Max tasks in list
REFERENCE_FOLLOW_DEPTH = 2    # Max nesting levels
MAX_ITERATIONS_PER_TASK = 3   # Prevent loops

# Track visited references
visited_refs: set[str] = set()

def should_follow_reference(ref: DetectedReference, depth: int) -> bool:
    """Check if reference should be followed."""
    ref_key = f"{ref.type}:{ref.target}"

    if ref_key in visited_refs:
        return False  # Already visited

    if depth >= REFERENCE_FOLLOW_DEPTH:
        return False  # Too deep

    visited_refs.add(ref_key)
    return True
```
