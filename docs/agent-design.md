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
    current_task_id: int | None
    phase: str
    messages: Annotated[list, add]  # For message accumulation

# Build graph
graph = StateGraph(AgentState)

# Add nodes for each phase
graph.add_node("analyze_query", analyze_query_node)
graph.add_node("hitl_refinement", hitl_refinement_node)
graph.add_node("generate_todo", generate_todo_node)
graph.add_node("hitl_approve_todo", hitl_approve_todo_node)
graph.add_node("extract_context", extract_context_node)
graph.add_node("synthesize", synthesize_node)
graph.add_node("quality_check", quality_check_node)
graph.add_node("source_attribution", source_attribution_node)

# Add edges with conditional routing
graph.add_edge("analyze_query", "hitl_refinement")
graph.add_conditional_edges(
    "hitl_refinement",
    should_continue_hitl,
    {"continue": "hitl_refinement", "done": "generate_todo"}
)
# ... more edges

# Compile
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
def vector_search(query: str, collection: str, top_k: int = 5) -> list[VectorResult]:
    """Search ChromaDB collection for relevant chunks.

    Args:
        query: Search query text
        collection: ChromaDB collection name
        top_k: Number of results to return

    Returns:
        List of matching chunks with metadata
    """
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

## HITL Integration Points

### 1. Query Refinement (Phase 1)

```python
def hitl_refinement_node(state: AgentState) -> AgentState:
    """Present clarification questions, collect answers."""

    # Generate questions based on QueryAnalysis
    questions = generate_clarification_questions(state["query_analysis"])

    # Display in Streamlit, await response
    answers = st.form("clarification")
    for q in questions:
        answers[q.id] = st.text_input(q.text)

    # Merge answers into QueryAnalysis
    state["query_analysis"] = merge_clarifications(
        state["query_analysis"],
        answers,
    )

    return state
```

### 2. ToDoList Approval (Phase 2)

```python
def hitl_approve_todo_node(state: AgentState) -> AgentState:
    """User approves or modifies ToDoList before research."""

    # Display ToDoList in editable form
    edited_tasks = st.data_editor(state["todo_list"])

    if st.button("Approve"):
        state["todo_list"] = edited_tasks
        state["phase"] = "extract"

    return state
```

### 3. Source Verification (Phase 5)

```python
def source_verification_node(state: AgentState) -> AgentState:
    """User can inspect and filter sources before final report."""

    sources = state["research_context"].all_sources

    # Display sources with checkboxes
    verified = []
    for src in sources:
        if st.checkbox(f"{src.doc_name}: {src.excerpt[:100]}..."):
            verified.append(src)

    state["verified_sources"] = verified
    return state
```

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
