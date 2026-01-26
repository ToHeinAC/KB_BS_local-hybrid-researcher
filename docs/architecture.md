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
│  Phase 1: Query Analysis & HITL Refinement                              │
│  ├─ Receive initial query                                                │
│  ├─ Extract: key_concepts, entities, scope, assumed_context             │
│  └─ HITL loop: clarification questions → merge answers → refine         │
│                                                                          │
│  Phase 2: Research Planning                                              │
│  └─ Generate ToDoList (3-5 items, max TODO_MAX_ITEMS=15)                │
│                                                                          │
│  Phase 3: Deep Context Extraction (Rabbithole Magic)                    │
│  ├─ For each ToDoList item:                                              │
│  │   ├─ Generate N search queries (2-4 per task)                        │
│  │   ├─ Vector DB search → M chunks per query (3-5)                     │
│  │   ├─ Extract relevant info from each chunk                           │
│  │   ├─ Detect references (section/document/external)                   │
│  │   ├─ Follow references → nested chunks                               │
│  │   ├─ Filter by relevance (threshold 0.6)                             │
│  │   └─ Re-evaluate ToDoList (mark done / add new tasks)                │
│  └─ Repeat until all tasks done or max_iterations reached               │
│                                                                          │
│  Phase 4: Synthesis & Quality Assurance                                  │
│  ├─ Generate per-query summaries with source citations                  │
│  ├─ Rerank by relevance to original query                               │
│  ├─ Quality check (0-400 score, 4 dimensions)                           │
│  ├─ Optional: web search for gaps                                        │
│  └─ Generate draft report                                                │
│                                                                          │
│  Phase 5: Source Attribution                                             │
│  └─ Add inline citations → resolve paths → clickable links              │
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
    current_task_id: int | None
    phase: str
    messages: Annotated[list, add]  # Accumulated messages
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
    hitl_refinements: list[str]  # Accumulated from HITL loop
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
┌──────────────────┐     clarification_needed?
│ Phase 1: Query   │────────────────────────┐
│ Analysis + HITL  │◄───────────────────────┘
└────────┬─────────┘     (loop until user exits)
         │
         ▼
┌──────────────────┐
│ HITL Checkpoint  │  User approves/modifies ToDoList
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Phase 2: Plan    │
│ (Generate ToDo)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     all tasks done?
│ Phase 3: Extract │─────────────────────────────────┐
│ (Rabbithole)     │◄────────────────────────────────┤
└────────┬─────────┘     (loop per task with refs)   │
         │                                           │
         ▼ (all done)                                │
┌──────────────────┐     quality < threshold?        │
│ Phase 4: Synth   │────────────────────────────────►│
│ + Quality Check  │     (regenerate / gap fill)     │
└────────┬─────────┘                                 │
         │                                           │
         ▼                                           │
┌──────────────────┐                                 │
│ Phase 5: Source  │                                 │
│ Attribution      │                                 │
└────────┬─────────┘                                 │
         │                                           │
         ▼                                           │
┌──────────────────┐                                 │
│ END: Final Report│                                 │
└──────────────────┘                                 │
```

## Data Flow Details

### Phase 1: Query Analysis + HITL

1. **User Query** → Streamlit UI captures research question
2. **Query Analysis** → LLM extracts key_concepts, entities, scope
3. **HITL Loop**:
   - Present 3 clarification questions based on QueryAnalysis
   - Collect user answers
   - Merge into QueryAnalysis
   - Offer exit point
4. **Output**: Refined QueryAnalysis with user context

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

1. **Generate Queries**: N search queries per task (2-4)
2. **Vector Search**: M chunks per query (3-5)
3. **Information Extraction**: Condense relevant passages
4. **Reference Detection**: Identify section/document/external refs
5. **Reference Following**: Resolve and retrieve nested chunks
6. **Relevance Filtering**: Include only if score > 0.6
7. **ToDoList Update**: Mark complete / spawn new tasks
8. **Loop**: Continue until task resolved or max_iterations

Output: Fully populated ResearchContext

### Phase 4: Synthesis + Quality Assurance

1. **Per-Query Summaries**: Synthesize from all extracted_info
2. **Reranking**: Order by relevance to original query
3. **Quality Check**: Score 0-400 across 4 dimensions
4. **Web Search** (optional): Fill gaps from external sources
5. **Draft Report**: Compile ordered summaries

### Phase 5: Source Attribution

1. **Citation Insertion**: Add `[Source_filename.pdf]` to all claims
2. **Path Resolution**: Map filenames to full paths
3. **Link Generation**: Create clickable HTML links
4. **Final Report**: Structured JSON with linked sources
