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
│  3. Reference Detection (Hybrid) ───────────────────────────────│
│     └─ Configurable: "regex", "llm", or "hybrid" (default)     │
│     └─ Regex: 7 patterns (§ refs, documents, URLs)              │
│     └─ LLM: REFERENCE_EXTRACTION_PROMPT → ExtractedReferenceList│
│     └─ Hybrid: regex + LLM, deduplicated by type:target key    │
│     └─ Returns list[DetectedReference] with extraction_method   │
│                                                                  │
│  4. Agentic Reference Gate (NEW) ─────────────────────────────────│
│     └─ For each detected reference:                             │
│         └─ LLM evaluates via REFERENCE_DECISION_PROMPT          │
│         └─ ReferenceDecision: {follow: bool, reason: str}       │
│         └─ Skip if not relevant to query (logged for debug)     │
│         └─ Fallback: follow on LLM error                        │
│                                                                  │
│  5. Reference Resolution (Enhanced, RECURSIVE) ─────────────────│
│     └─ For each followed reference:                             │
│         └─ Check visited + depth + token budget                 │
│         └─ Route by type via resolve_reference_enhanced():      │
│           └─ legal_section/section → registry → scoped search   │
│           └─ document/document_mention → registry → scoped      │
│           └─ academic_* → broad vector search                   │
│         └─ Track token usage + document history                 │
│         └─ Mark reference as visited                            │
│                                                                  │
│  6. Convergence Check ──────────────────────────────────────────│
│     └─ detect_convergence(doc_history) after each chunk         │
│     └─ Stops early if same doc appears >= threshold times (3)   │
│                                                                  │
│  7. Relevance Filtering ────────────────────────────────────────│
│     └─ filter_by_relevance() removes chunks < threshold (0.6)   │
│     └─ Combines vector score + keyword overlap                  │
│                                                                  │
│  8. Context Accumulation ───────────────────────────────────────│
│     └─ Add SearchQueryResult to research_context                │
│     └─ Update metadata (docs_referenced, visited_refs)          │
│                                                                  │
│  9. Task Completion & Loop ─────────────────────────────────────│
│     └─ Mark current task completed                              │
│     └─ Find next pending task                                   │
│     └─ Reset depth to 0                                         │
│     └─ Loop or transition to synthesize                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Reference Detection System

Reference detection is configurable via `reference_extraction_method` setting (`regex`, `llm`, or `hybrid`).

### 4a. Regex Detection (`detect_references()`)

**Three Reference Types** detected via 7 regex patterns in `src/agents/tools.py`:

#### Section References
```python
r"(?:siehe|gemäß|nach|in)\s*§\s*(\d+(?:\s*Abs\.\s*\d+)?)"
r"§\s*(\d+(?:\s*Abs\.\s*\d+)?)\s*(?:StrlSchV|StrlSchG)"
r"(?:see|cf\.)\s+section\s+(\d+(?:\.\d+)?)"
```
Examples: `§123`, `§123 Abs. 1`, `gemäß §45 StrlSchV`, `see section 5.2`

#### Document References
```python
r"(?:siehe|gemäß)\s+(?:Dokument|Unterlage)\s+[\"']?([^\"'\n,]+)[\"']?"
r"(?:EU|EG)\s*(\d+(?:\.\d+)?)"
r"(?:see|refer to)\s+document\s+[\"']?([^\"'\n,]+)[\"']?"
```
Examples: `siehe Dokument "Sicherheitsbericht"`, `EU 2024/123`

#### External References
```python
r"(https?://[^\s<>\"]+)"
```

### 4b. LLM Detection (`extract_references_llm()`)

Uses `REFERENCE_EXTRACTION_PROMPT` (in `src/prompts.py`) with 4 few-shot examples to classify references into 4 types:
- `legal_section`: German/English legal paragraph references (e.g., `§ 133 des Strahlenschutzgesetzes`)
- `academic_numbered`: Numbered citations (e.g., `[253]`)
- `academic_shortform`: Author-year citations (e.g., `[Townsend79]`)
- `document_mention`: Named document references (e.g., `KTA 1401`, `Kreislaufwirtschaftsgesetz`)

LLM output is parsed via `generate_structured()` into `ExtractedReferenceList` → converted to `DetectedReference` objects with `extraction_method="llm"`.

### 4c. Hybrid Detection (`detect_references_hybrid()`)

Default mode. Runs regex first (fast), then LLM (thorough), then deduplicates:
1. Collect regex results, add to `seen` set by `type:target` key
2. For each LLM result, skip if key already seen OR if target text overlaps with existing ref
3. Return combined, deduplicated list

### DetectedReference Model (Enhanced)

```python
class DetectedReference:
    type: Literal["section", "document", "external",
                   "legal_section", "academic_numbered",
                   "academic_shortform", "document_mention"]
    target: str              # Extracted reference target
    original_text: str       # Full matched text
    found: bool              # Resolution success
    nested_chunks: list[NestedChunk]  # Resolved content
    document_context: str | None  # Resolved document name hint (NEW)
    extraction_method: Literal["regex", "llm"]  # How detected (NEW)
```

---

## 4d. Document Registry (`kb/document_registry.json`)

Maps PDF filenames to human-readable synonyms across all 4 collections. Used by `resolve_document_name()` for scoped reference resolution.

**Structure:**
```json
{
  "collections": {
    "StrlSch": {
      "documents": [
        {"filename": "StrlSchG.pdf", "synonyms": ["Strahlenschutzgesetz", "StrlSchG", "StrSchG"]},
        {"filename": "KTA 1401_2017-11.pdf", "synonyms": ["KTA 1401"]}
      ]
    }
  }
}
```

**Resolution (`resolve_document_name()`)** - 3-stage greedy matching:
1. **Exact synonym** (case-insensitive): `"StrlSchG"` -> `StrlSchG.pdf` in `StrlSch`
2. **Fuzzy match** (`difflib.SequenceMatcher`, threshold 0.7): catches typos like `"Strahlenschutzverordung"`
3. **Substring match** on filename: `"AtG"` matches `AtG.pdf`

Returns `(filename, collection_key)` or `(None, None)` if unresolved.

## 4e. Enhanced Reference Resolution (`resolve_reference_enhanced()`)

Routes by reference type with scoped search when document is known:

| Reference Type | Resolution Strategy |
|---------------|---------------------|
| `legal_section` / `section` | Extract doc hint from `document_context` or target text -> registry resolve -> scoped vector search in that collection/document. Fallback: broad search. |
| `document` / `document_mention` | Registry lookup of doc name -> scoped vector search. Fallback: broad search. |
| `academic_numbered` / `academic_shortform` | Broad vector search with citation text as query. |
| `external` | Not resolved. |

**Scoped search** (`_vector_search_scoped()`):
- Calls `chromadb_client.search(query, collection_key, top_k=5)`
- Post-filters results where `doc_name` contains the target filename
- Applies `reference_relevance_threshold` (0.6)

**Additional guards:**
- **Token budget**: `token_count >= reference_token_budget` (default 50K) -> stop following
- **Depth limit**: unchanged (`reference_follow_depth`, default 2)
- **Visited set**: unchanged (`ref_key` in `visited`)

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
| No current task | nodes.py execute_task | → synthesize |
| Task already completed | nodes.py execute_task | → synthesize |
| All tasks completed | nodes.py execute_task | → synthesize |
| Depth limit reached | tools.py resolve_reference_enhanced | Returns empty (no more refs) |
| Token budget exhausted | tools.py resolve_reference_enhanced | Returns empty (no more refs) |
| Convergence detected | nodes.py execute_task | Breaks chunk processing loop early |

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
| `reference_extraction_method` | `"hybrid"` | Detection method: `"regex"`, `"llm"`, `"hybrid"` |
| `reference_token_budget` | 50000 | Max tokens for reference following per task |
| `convergence_same_doc_threshold` | 3 | Stop when same doc appears N times |
| `document_registry_path` | `"./kb/document_registry.json"` | Path to document-synonym registry |
| `m_chunks_per_query` | 4 | Results per vector search |
| `todo_max_items` | 15 | Max tasks in ToDoList |
| `initial_todo_items` | 5 | Starting task count |

---

## 11. Key Files

| File | Content |
|------|---------|
| `src/agents/nodes.py` | execute_task node with hybrid detection + enhanced resolution |
| `src/agents/tools.py` | detect_references, detect_references_hybrid, extract_references_llm, resolve_reference_enhanced, load_document_registry, resolve_document_name, detect_convergence |
| `src/agents/graph.py` | LangGraph StateGraph definition, routing |
| `src/models/research.py` | ResearchContext, ChunkWithInfo, DetectedReference, ExtractedReference, ExtractedReferenceList, ReferenceDecision, QualityRemediationDecision |
| `src/models/query.py` | ToDoList, ToDoItem |
| `src/prompts.py` | REFERENCE_EXTRACTION_PROMPT, REFERENCE_DECISION_PROMPT, QUALITY_REMEDIATION_PROMPT |
| `src/config.py` | All threshold/limit settings (incl. enhanced reference following) |
| `kb/document_registry.json` | Document-to-synonym mapping across 4 collections |
| `tests/test_reference_extraction.py` | 42 tests for reference extraction, resolution, and agentic gate logic |

---

## 12. Mindmap Summary Structure

```
RABBITHOLE MAGIC (Phase 3)
├── STATE
│   ├── current_depth (0→2 max)
│   ├── visited_refs (loop prevention set)
│   ├── current_task_id
│   ├── doc_history (convergence tracking)
│   ├── token_count (budget tracking)
│   └── research_context (accumulated findings)
│
├── REFERENCE DETECTION (configurable)
│   ├── Regex (7 patterns)
│   │   ├── Section refs (§123, see section X)
│   │   ├── Document refs (Dokument X, EU 2024/123)
│   │   └── External refs (URLs)
│   ├── LLM (REFERENCE_EXTRACTION_PROMPT)
│   │   ├── legal_section (§ 133 des StrlSchG)
│   │   ├── academic_numbered ([253])
│   │   ├── academic_shortform ([Townsend79])
│   │   └── document_mention (KTA 1401)
│   └── Hybrid (regex + LLM, deduplicated)
│
├── DOCUMENT REGISTRY (kb/document_registry.json)
│   ├── 4 collections (StrlSch, StrlSchExt, NORM, GLageKon)
│   ├── Filename → synonym mapping
│   └── 3-stage resolution (exact > fuzzy > substring)
│
├── ENHANCED RESOLUTION (resolve_reference_enhanced)
│   ├── legal_section → registry → scoped search
│   ├── document_mention → registry → scoped search
│   ├── academic_* → broad vector search
│   └── Fallback: original broad search
│
├── EXECUTION LOOP
│   ├── Vector Search (ChromaDB)
│   ├── Chunk Processing (LLM extraction)
│   ├── Hybrid Reference Detection
│   ├── Agentic Reference Gate (LLM decides follow/skip)
│   ├── Enhanced Reference Resolution (scoped)
│   ├── Convergence Check (doc_history)
│   ├── Relevance Filtering (threshold 0.6)
│   └── Context Accumulation
│
├── LOOP CONTROL
│   ├── ToDoList iteration
│   ├── Task completion marking
│   ├── Depth reset per task
│   ├── Token budget limit (50K default)
│   ├── Convergence detection (same doc >= 3)
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
- **Precise resolution**: Document registry enables scoped search within specific documents/collections
- **Hybrid detection**: Regex catches structured patterns fast; LLM catches nuanced references regex misses
- **Quality control**: Relevance filtering prevents noise accumulation
- **Intelligent selection**: Agentic reference gate lets the LLM skip tangential references, preserving token budget for high-value refs
- **Self-correcting quality**: Agentic remediation loop detects weak synthesis and retries with focused instructions (max 1 retry)
- **Safe exploration**: Depth limits, token budget, convergence detection, and loop prevention ensure termination
