# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Web UI                                │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │Query Input │ │To-Do List  │ │Results View  │ │HITL Panel          │  │
│  └────────────┘ └────────────┘ └──────────────┘ └────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│            Rabbithole-Agent (LangGraph StateGraph)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Enhanced Query Analysis + Iterative HITL                       │
│  ├─ hitl_init: initialize conversation, detect language                  │
│  ├─ hitl_generate_queries: 3 search queries per iteration                │
│  ├─ hitl_retrieve_chunks: vector search + deduplication                  │
│  ├─ hitl_analyze_retrieval: LLM context analysis & gaps                  │
│  ├─ hitl_generate_questions: gap-informed follow-ups                     │
│  ├─ hitl_process_response: analyze user feedback                         │
│  └─ hitl_finalize: transition to Phase 2                                 │
│                                                                          │
│  Phase 2: Research Planning                                              │
│  ├─ generate_todo: generate ToDo items                                   │
│  └─ hitl_approve_todo: checkpoint (user approves/modifies)               │
│                                                                          │
│  Phase 3: Deep Context Extraction (Rabbithole Magic)                    │
│  ├─ execute_task: vector search → extracted_info → reference following   │
│  ├─ Filter by relevance (threshold 0.6)                                  │
│  └─ Loop until all tasks completed                                       │
│                                                                          │
│  Phase 4: Synthesis & Quality Assurance                                  │
│  ├─ synthesize: combine extracted_info into an answer                    │
│  └─ quality_check: optional QA scoring (0-400)                           │
│                                                                          │
│  Phase 5: Source Attribution                                             │
│  └─ attribute_sources: build FinalReport with sources                    │
│                                                                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    │                             │                             │
    ▼                             ▼                             ▼
┌────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ PDF Corpus     │    │ ChromaDB         │    │ Ollama LLM          │
│ (kb/*__db_inserted)│ │ (kb/database/)   │    │ (qwen3:14b)         │
└────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Key State Objects

The LangGraph agent maintains state as a `TypedDict` (LangChain v1.0 requirement).
Pydantic models are used for validation but serialized to dicts in state.

### AgentState (TypedDict)

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    query: str                    # User's research question
    query_analysis: dict          # Serialized QueryAnalysis
    todo_list: list[dict]         # Serialized ToDoItems
    research_context: dict        # Serialized ResearchContext
    final_report: dict            # Serialized FinalReport
    current_task_id: int | None
    phase: str
    messages: Annotated[list, add]  # Accumulated messages

    # Reference tracking
    visited_refs: set[str]        # Visited reference keys (loop prevention)
    current_depth: int            # Current recursion depth

    # Quality assessment
    quality_assessment: dict | None  # Serialized QualityAssessment

    # HITL checkpoint support
    hitl_pending: bool
    hitl_checkpoint: dict | None
    hitl_decision: dict | None

    # Phase 1: Iterative HITL
    hitl_state: dict | None       # Chat-style HITL conversation state
    hitl_iteration: int           # Current iteration count (0-indexed)
    hitl_max_iterations: int      # Max iterations (default 5)
    hitl_conversation_history: list[dict]  # Full conversation
    hitl_active: bool             # Whether iterative HITL is active
    hitl_termination_reason: str | None  # user_end, max_iterations, convergence

    # Enhanced Phase 1: Multi-vector retrieval & Convergence
    iteration_queries: list[list[str]]  # Queries per iteration [[q1, q2, q3], ...]
    knowledge_gaps: list[str]           # Gaps identified from retrieval analysis
    retrieval_dedup_ratios: list[float] # Dedup ratio per iteration
    coverage_score: float               # 0-1 information coverage estimate
    retrieval_history: dict             # Per-iteration retrieval metadata
    query_retrieval: str                # Accumulated retrieval results (context)

    # HITL handoff fields
    research_queries: list[str]   # Generated queries from HITL
    additional_context: str       # Summary from HITL analysis
    detected_language: str        # de or en

    # UI settings
    selected_database: str | None
    k_results: int
```

### QueryAnalysis (Pydantic model, serialized to state)
```python
class QueryAnalysis(BaseModel):
    original_query: str
    key_concepts: list[str]
    entities: list[str]
    scope: str
    assumed_context: list[str]
    clarification_needed: bool
    hitl_refinements: list[str]  # Accumulated from HITL clarification
```

### ToDoList (Pydantic model, serialized to state)
```python
class ToDoItem(BaseModel):
    id: int
    task: str
    context: str
    completed: bool
    subtasks: list[str] = []

class ToDoList(BaseModel):
    items: list[ToDoItem]
    max_items: int = 15  # TODO_MAX_ITEMS
```

### state['ResearchContext']

The growing JSON structure that accumulates all research findings:

```json
{
  "search_queries": [
    {
      "query": "original search query string",
      "chunks": [
        {
          "chunk": "original chunk text from vector DB",
          "document": "document_name.pdf",
          "extracted_info": "condensed relevant passages",
          "references": [
            {
              "type": "section|document|external",
              "target": "section_number/document_name/url",
              "found": true,
              "nested_chunks": [
                {
                  "chunk": "reference content",
                  "document": "document_name.pdf",
                  "extracted_info": "relevant to original query",
                  "relevance_score": 0.85
                }
              ]
            }
          ]
        }
      ],
      "summary": "comprehensive answer to search query",
      "summary_references": ["document1.pdf", "document2.pdf"],
      "quality_score": 0.9,
      "web_search_results": null
    }
  ],
  "metadata": {
    "total_iterations": 3,
    "documents_referenced": ["StrlSchG.pdf", "StrlSchV.pdf"],
    "external_sources_used": false
  }
}
```

## Phase Transitions

### Iterative HITL Flow

```
┌──────────────────┐
│ START            │
└────────┬─────────┘
         │
         ▼
┌───────────────────────────┐
│ entry_router               │  (routes based on state)
└────────┬──────────────────┘
         │ hitl_active=True
         ▼
┌───────────────────────────┐
│ hitl_init                  │  (detect language, init state)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_generate_queries     │  (Node 1: original + alternatives)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_retrieve_chunks      │  (Node 2: vector search + dedup)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_analyze_retrieval    │  (Node 3: concepts, gaps, coverage)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_generate_questions    │  (Node 4: gap-informed questions)
└────────┬──────────────────┘
         │ (→ END, wait for user)
         ▼
┌───────────────────────────┐
│ hitl_process_response      │  (Node 5: analyze feedback)
└────────┬──────────────────┘
         │
    ┌────┴────┐
    │ loop?   │
    └────┬────┘
         │ no (termination or /end)
         ▼
┌───────────────────────────┐
│ hitl_finalize              │  (generate research_queries)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ generate_todo              │
└────────┬──────────────────┘
```

## Data Flow Details

### Phase 1: Enhanced Query Analysis + Iterative HITL

1. **User Query** → Streamlit UI captures research question
2. **hitl_init**: Detect language (de/en), initialize conversation state
3. **hitl_generate_queries**: Generate 3 search queries (original + 2 alternatives)
4. **hitl_retrieve_chunks**: Search ChromaDB, deduplicate, append to `query_retrieval`
5. **hitl_analyze_retrieval**: LLM analysis for concepts, gaps, and coverage score
6. **hitl_generate_questions**: Generate 2-3 contextual follow-ups based on knowledge gaps
7. **Graph interrupts** (→ END), awaits user response
8. **hitl_process_response**: Analyze user response, check termination:
   - `/end` typed → terminate with `user_end`
   - Max iterations reached → terminate with `max_iterations`
   - Convergence criteria met (coverage ≥ 0.8, dedup ≥ 0.7, gaps ≤ 2) → terminate with `convergence`
   - Otherwise → loop back to `hitl_generate_queries`
9. **hitl_finalize**: Generate research_queries list, build query_analysis
10. **Output**: `research_queries[]`, `query_analysis`, `coverage_score`, `query_retrieval` (as context)

### Phase 2: Research Planning

1. **Input**: QueryAnalysis
2. **ToDoList Generation**:
   - 3-5 initial items
   - Each item: specific, measurable task
   - Constraints: max TODO_MAX_ITEMS (15)
3. **HITL Checkpoint**: User approves/modifies tasks
4. **Output**: Approved ToDoList

### Phase 3: Deep Context Extraction

For each ToDoList item:

1. **Vector Search**: Search ChromaDB (either a selected database directory or all collections)
2. **Information Extraction**: Condense relevant passages into `extracted_info`
3. **Reference Detection**: Identify section/document/external refs
4. **Reference Following**: Resolve and retrieve nested chunks
5. **Relevance Filtering**: Keep only chunks above threshold
6. **ToDoList Update**: Mark task complete and continue to next task

Output: Fully populated ResearchContext

### Phase 4: Synthesis + Quality Assurance

1. **Synthesis**: Build a single coherent answer from extracted findings
2. **Quality Check** (optional): Score 0-400 across 4 dimensions

### Phase 5: Source Attribution

1. **Source List**: Collect sources from extracted chunks
2. **Report Assembly**: Build `FinalReport` (answer, findings, sources, quality)

## Streamlit Runtime Model

The UI runs the compiled LangGraph using streaming for live progress updates:

```python
for state in graph.stream(input_state, config, stream_mode="values"):
    update_agent_state(state)
    render_research_status()
```

HITL checkpoints are supported via a persisted `thread_id` stored in session state and reused when resuming.

### UI Data Flow for Chat-Based HITL

The chat-based HITL (`render_chat_hitl`) runs independently from the LangGraph, using standalone services:

```
┌─────────────────────────────────────────────────────────────────┐
│  Chat-Based HITL (hitl_panel.py)                                 │
├─────────────────────────────────────────────────────────────────┤
│  User Query                                                      │
│      │                                                           │
│      ▼                                                           │
│  HITLService.detect_language()                                   │
│      │                                                           │
│      ▼                                                           │
│  _perform_hitl_retrieval(query, session)  ──┐                    │
│      │                                      │                    │
│      │  ┌───────────────────────────────────┘                    │
│      │  │  Uses session.selected_database                        │
│      │  │  Stores in session.hitl_state["retrieval_history"]     │
│      │  └───────────────────────────────────┐                    │
│      │                                      │                    │
│      ▼                                      ▼                    │
│  HITLService.generate_follow_up_questions() │                    │
│      │                                      │                    │
│      ▼                                      │                    │
│  _render_retrieval_history()  ◄─────────────┘                    │
│      │  (reads from hitl_state or agent_state)                   │
│      │                                                           │
│      ▼                                                           │
│  [User Feedback Loop]                                            │
│      │                                                           │
│      ▼                                                           │
│  On /end: create_hitl_result() → research_queries                │
└─────────────────────────────────────────────────────────────────┘
```

**Key Data Sources:**
- `session.hitl_state["retrieval_history"]`: Populated during HITL phase
- `session.agent_state["retrieval_history"]`: Populated during graph execution
- `_render_retrieval_history()` reads from both, preferring `hitl_state`

### Cached Service Clients

To improve performance on Streamlit reruns, service clients are cached:

```python
@st.cache_resource
def _get_chromadb_client():    # safe_exit.py
def get_chromadb_client():     # app.py
def _get_ollama_client():      # safe_exit.py
def _get_hitl_service():       # hitl_panel.py
```

This prevents re-loading the embedding model and reconnecting to services on every UI interaction.

### Graph Entry Point Routing

The `route_entry_point()` function in `graph.py` handles multiple entry scenarios:

```python
def route_entry_point(state) -> Literal["hitl_init", "hitl_process_response", "generate_todo", "process_hitl_todo"]:
    # 1. hitl_decision + !hitl_active → process_hitl_todo (post-approval resume)
    # 2. hitl_decision + hitl_active → hitl_process_response (iterative HITL resume)
    # 3. research_queries present → generate_todo (skip HITL)
    # 4. phase == "generate_todo" → generate_todo
    # 5. else → hitl_init (start new)
```

This enables:
- **Resume after todo approval**: When user approves/modifies tasks (`hitl_decision` present, `hitl_active=False`)
- **Skip HITL**: When UI chat-based HITL has already produced research_queries
- **Resume HITL**: When user responds to an interrupted iterative HITL session
- **New HITL**: Default behavior when starting fresh

**Key invariant**: `_start_research_from_hitl()` sets `hitl_active=False` before entering the graph, so post-approval resume never misroutes to `hitl_process_response`.
