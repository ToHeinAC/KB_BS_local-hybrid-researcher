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
│  Phase 1: Query Analysis + HITL Clarification                            │
│  ├─ analyze_query: extract key_concepts, entities, scope, assumed_context│
│  └─ hitl_clarify: optional checkpoint (user clarifies)                   │
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

```
┌──────────────────┐
│ START            │
└────────┬─────────┘
         │
         ▼
┌───────────────────────────┐
│ analyze_query              │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_clarify               │  (optional checkpoint)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ process_hitl_clarify       │  (if user answered)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ generate_todo              │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ hitl_approve_todo          │  (checkpoint)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ process_hitl_todo          │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ execute_task (loop)        │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ synthesize                 │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ quality_check (optional)   │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ attribute_sources          │
└────────┬──────────────────┘
         │
         ▼
┌──────────────────┐
│ END: Final Report│
└──────────────────┘
```

## Data Flow Details

### Phase 1: Query Analysis + HITL

1. **User Query** → Streamlit UI captures research question
2. **Query Analysis** → LLM extracts key_concepts, entities, scope
3. **Optional HITL Clarification Checkpoint**:
   - If clarification is needed, the graph emits a checkpoint (`hitl_pending=True`)
   - The user answers clarification questions once
   - The graph resumes via `process_hitl_clarify`
4. **Output**: QueryAnalysis updated with any user clarifications

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
