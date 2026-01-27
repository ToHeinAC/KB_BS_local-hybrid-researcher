# Rabbithole Magic Agent - Deep Context Extraction (Phase 3)

This document provides a detailed explanation of the Rabbithole Magic implementation - the core differentiator of this research system.

## 1. Core Concept

The Rabbithole Magic is the heart of the system - it performs **iterative deep context extraction** by:
- Executing vector searches for each research task
- Detecting references within retrieved chunks (sections, documents, URLs)
- Recursively following those references up to a configurable depth
- Filtering results by relevance to prevent context pollution
- Accumulating all findings into a unified research context

---

## 2. State Architecture

**AgentState (TypedDict)** manages the entire flow:

| Field | Purpose |
|-------|---------|
| `current_depth` | Integer tracking recursion level (0 = initial search) |
| `visited_refs` | Set of "type:target" strings preventing infinite loops |
| `current_task_id` | Currently executing ToDoItem |
| `completed_task_ids` | List of finished task IDs |
| `research_context` | Accumulated chunks, metadata, documents |
| `todo_list` | Dynamic task list (3-15 items) |

---

## 3. Execution Flow (execute_task Node)

Located in `src/agents/nodes.py` (lines 312-413):

```
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH TASK IN TODO_LIST:                                     │
│                                                                  │
│  1. Vector Search ──────────────────────────────────────────────│
│     └─ Query = task.task + query_analysis.concepts              │
│     └─ Returns top-k VectorResult objects (default k=4)         │
│                                                                  │
│  2. Chunk Processing ───────────────────────────────────────────│
│     └─ create_chunk_with_info() for each result                 │
│     └─ LLM extracts relevant passages via extract_info()        │
│     └─ Creates ChunkWithInfo with extracted_info, relevance     │
│                                                                  │
│  3. Reference Detection ────────────────────────────────────────│
│     └─ Regex patterns match section/document/external refs      │
│     └─ Returns list[DetectedReference]                          │
│                                                                  │
│  4. Reference Resolution (RECURSIVE) ───────────────────────────│
│     └─ For each detected reference:                             │
│         └─ Check if already visited (loop prevention)           │
│         └─ Check if depth < max_depth (default: 2)              │
│         └─ Resolve via vector search for target                 │
│         └─ Store results as NestedChunk objects                 │
│         └─ Mark reference as visited                            │
│                                                                  │
│  5. Relevance Filtering ────────────────────────────────────────│
│     └─ filter_by_relevance() removes chunks < threshold (0.6)   │
│     └─ Combines vector score + keyword overlap                  │
│                                                                  │
│  6. Context Accumulation ───────────────────────────────────────│
│     └─ Add SearchQueryResult to research_context                │
│     └─ Update metadata (docs_referenced, visited_refs)          │
│                                                                  │
│  7. Task Completion & Loop ─────────────────────────────────────│
│     └─ Mark current task completed                              │
│     └─ Find next pending task                                   │
│     └─ Reset depth to 0                                         │
│     └─ Loop or transition to synthesize                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Reference Detection System

**Three Reference Types** detected via regex in `src/agents/tools.py`:

### Section References
```python
r"(?:siehe|gemäß|nach|in)\s*§\s*(\d+(?:\s*Abs\.\s*\d+)?)"
r"§\s*(\d+(?:\s*Abs\.\s*\d+)?)\s*(?:StrlSchV|StrlSchG)"
```
Examples: `§123`, `§123 Abs. 1`, `gemäß §45 StrlSchV`

### Document References
```python
r"(?:siehe|gemäß)\s+Dokument\s+([A-Za-z0-9_-]+)"
r"EU\s*(?:Verordnung|Richtlinie)?\s*(\d{4}/\d+)"
```
Examples: `siehe Dokument ABC-123`, `EU 2024/123`

### External References
```python
r"https?://[^\s<>\"']+"
```
Examples: URLs

**DetectedReference Model**:
```python
class DetectedReference:
    type: Literal["section", "document", "external"]
    target: str              # Extracted reference target
    original_text: str       # Full matched text
    found: bool              # Resolution success
    nested_chunks: list[NestedChunk]  # Resolved content
```

---

## 5. Depth Tracking & Loop Prevention

**Depth Mechanism**:
- `current_depth` starts at 0 for initial vector search
- Incremented by 1 when following each reference level
- Checked against `reference_follow_depth` (default: 2)
- **Reset to 0** after completing each task

```
Depth 0: Initial task search results
    └─ Depth 1: References found in depth-0 chunks
        └─ Depth 2: References found in depth-1 chunks (MAX)
            └─ Depth 3: BLOCKED - returns empty list
```

**Loop Prevention**:
```python
# Before resolving
ref_key = f"{ref.type}:{ref.target}"
if context.has_visited_ref(ref_key):
    continue  # Skip already visited

# After resolving
context.mark_ref_visited(ref_key)
```

---

## 6. Relevance Filtering

**Two-Tier Scoring** in `filter_by_relevance()`:

1. **Vector Similarity Score**: From ChromaDB embedding distance
2. **Keyword Overlap Score**:
   ```python
   overlap = len(query_words & chunk_words)
   score = min(1.0, overlap / len(query_words))
   ```

**Filtering**:
- Threshold: `reference_relevance_threshold` (default: 0.6)
- Chunks scoring below threshold are discarded
- Applies to both initial results AND nested reference chunks

---

## 7. ToDoList Dynamic Management

**ToDoList Model**:
```python
class ToDoList:
    items: list[ToDoItem]    # 3-15 tasks
    max_items: int = 15
    current_item_id: int | None

class ToDoItem:
    id: int
    task: str                # What to research
    context: str             # Why it matters
    completed: bool
    subtasks: list[str]
```

**Lifecycle**:
1. **Generated** in Phase 2 via LLM (5 initial tasks)
2. **Approved** via HITL checkpoint (user can modify)
3. **Executed** one-by-one in Phase 3
4. **Marked completed** after each task finishes
5. **Displayed** in Streamlit UI for visibility

---

## 8. Context Accumulation

**ResearchContext Structure**:
```python
class ResearchContext:
    search_queries: list[SearchQueryResult]  # One per task
    metadata: ResearchContextMetadata

class SearchQueryResult:
    query: str
    chunks: list[ChunkWithInfo]  # Includes nested refs
    summary: str | None

class ChunkWithInfo:
    chunk_text: str
    extracted_info: str      # LLM-condensed info
    document: str
    page: int
    relevance_score: float
    references: list[DetectedReference]  # With nested_chunks
```

**Metadata Tracking**:
- `total_iterations`: Count of tasks executed
- `documents_referenced`: Unique document list
- `visited_refs`: All resolved reference keys

---

## 9. Loop Termination Conditions

The execute_task loop exits when ANY of these occur:

| Condition | Code Location | Result |
|-----------|---------------|--------|
| No current task | Line 322-323 | → synthesize |
| Task already completed | Line 333-334 | → synthesize |
| All tasks completed | Line 401-404 | → synthesize |
| Depth limit reached | tools.py:166-168 | Returns empty (no more refs) |

**Phase Transition**:
```python
return {
    "phase": "execute_tasks" if next_task_id else "synthesize",
    ...
}
```

---

## 10. Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reference_follow_depth` | 2 | Max recursion levels |
| `reference_relevance_threshold` | 0.6 | Min score to keep chunk |
| `m_chunks_per_query` | 4 | Results per vector search |
| `todo_max_items` | 15 | Max tasks in ToDoList |
| `initial_todo_items` | 5 | Starting task count |

---

## 11. Key Files

| File | Content |
|------|---------|
| `src/agents/nodes.py` | execute_task node (lines 312-413) |
| `src/agents/tools.py` | detect_references, resolve_reference, filter_by_relevance |
| `src/agents/graph.py` | LangGraph StateGraph definition, routing |
| `src/models/research.py` | ResearchContext, ChunkWithInfo, DetectedReference |
| `src/models/query.py` | ToDoList, ToDoItem |
| `src/config.py` | All threshold/limit settings |

---

## 12. Mindmap Summary Structure

```
RABBITHOLE MAGIC (Phase 3)
├── STATE
│   ├── current_depth (0→2 max)
│   ├── visited_refs (loop prevention set)
│   ├── current_task_id
│   └── research_context (accumulated findings)
│
├── EXECUTION LOOP
│   ├── Vector Search (ChromaDB)
│   ├── Chunk Processing (LLM extraction)
│   ├── Reference Detection (regex patterns)
│   │   ├── Section refs (§123)
│   │   ├── Document refs (Dokument X)
│   │   └── External refs (URLs)
│   ├── Reference Resolution (recursive)
│   │   ├── Depth check
│   │   ├── Visit check
│   │   └── NestedChunk creation
│   ├── Relevance Filtering (threshold 0.6)
│   └── Context Accumulation
│
├── LOOP CONTROL
│   ├── ToDoList iteration
│   ├── Task completion marking
│   ├── Depth reset per task
│   └── Exit conditions
│
└── OUTPUT
    ├── SearchQueryResult per task
    ├── ChunkWithInfo with nested refs
    └── Metadata (docs, refs, iterations)
```

---

## 13. Key Insight

The Rabbithole Magic creates **depth-controlled recursive context expansion** - it doesn't just do a single vector search, but actively discovers and follows the citation/reference graph within your document collection. This enables:

- **Deep understanding**: Following references reveals implicit relationships
- **Comprehensive coverage**: No relevant document is missed if it's referenced
- **Quality control**: Relevance filtering prevents noise accumulation
- **Safe exploration**: Depth limits and loop prevention ensure termination
