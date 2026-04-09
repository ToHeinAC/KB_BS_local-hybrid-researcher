# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Web UI                                │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │Query Input │ │To-Do List  │ │Results View  │ │HITL Panel          │  │
│  └────────────┘ └────────────┘ └──────────────┘ └────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  GPU Widget (sidebar): live temp/fan/load via Tornado route inject  ││
│  └─────────────────────────────────────────────────────────────────────┘│
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
│  └─ hitl_finalize: build query_anchor/hitl_smry → assess_query          │
│                                                                          │
│  Phase 2.5: Query Assessment (Agentic Gate)                              │
│  └─ assess_query: LLM decides proceed/reject + num_tasks (3-6)          │
│     ├─ proceed=False → __end__ with rejection FinalReport                │
│     └─ proceed=True  → generate_todo (with num_tasks)                    │
│                                                                          │
│  Phase 2: Research Planning                                              │
│  ├─ generate_todo: LLM generates num_tasks items (fallback: HITL queries)│
│  └─ hitl_approve_todo: checkpoint (user approves/modifies)               │
│                                                                          │
│  Phase 3: Deep Context Extraction (with Graded Classification)           │
│  ├─ execute_task: LLM multi-query (3) → vector search → extract info    │
│  │   + quotes → classify tier                                            │
│  ├─ Agentic ref gate → Reference following → classify nested chunks     │
│  ├─ Accumulate by tier (primary/secondary/tertiary)                      │
│  └─ Loop until all tasks completed                                       │
│                                                                          │
│  Phase 3.5: Pre-Synthesis Relevance Validation                           │
│  └─ validate_relevance: filter drift against query_anchor                │
│                                                                          │
│  Phase 3.6: Task Summary Reranking                                       │
│  └─ rerank_task_summaries: sort by relevance_to_query desc, stamp rank   │
│     → synthesis prompt weights high-relevance findings over low ones     │
│                                                                          │
│  Phase 3.8: Reference Provenance                                         │
│  Nested chunks carry parent_document + reference_surrounding_context     │
│  _rerank_task_chunks(): parent_context → CHUNK_RERANKER (−20-40 penalty) │
│  _format_ranked_findings(): [via ref "..."] + Parent context: headers    │
│  TASK_SUMMARY rule 2e: cap off-topic ref chunks at effective score 49    │
│                                                                          │
│  Phase 3.9: Batch Chunk Reranking                                        │
│  └─ _rerank_task_chunks(): batch LLM scoring (batch_size=6)              │
│     ├─ Precision/recall strategies (RERANKER_PRECISION/RECALL_PROMPT)     │
│     ├─ Round-robin batching → cross-batch zero-mean normalization        │
│     ├─ Hard-filter below reranker_min_score (default 4)                  │
│     └─ Raw 1-5 → 0-100 mapping (SCORE_TO_100) for downstream compat     │
│                                                                          │
│  Phase 4: Query-Anchored Synthesis & Quality Assurance                   │
│  ├─ synthesize: pre-digested task summaries + HITL summary               │
│  ├─ Language enforcement (generate_structured_with_language)             │
│  ├─ quality_check: optional QA scoring (0-500, 5 dimensions)             │
│  └─ Agentic remediation: LLM decides accept/retry (max 1 retry)         │
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

    # Agentic decision fields
    query_assessment: dict | None # From assess_query: {proceed, num_tasks, reason, explanation}
    synthesis_retry_count: int    # Number of synthesis retries (max 1)
    quality_remediation_focus: str  # Focus instructions for re-synthesis

    # Graded Context Management
    query_anchor: dict            # Immutable reference to original intent
    hitl_smry: str                # Citation-aware HITL summary with [Source_filename] annotations
    primary_context: list[dict]   # Tier 1: Direct, high-relevance findings
    secondary_context: list[dict] # Tier 2: Reference-followed, medium-relevance
    tertiary_context: list[dict]  # Tier 3: Deep references, HITL retrieval
    task_summaries: list[dict]    # Per-task structured summaries
    preserved_quotes: list[dict]  # Critical verbatim quotes

    # UI settings
    selected_database: str | None
    k_results: int
```

### Query Anchor Structure (NEW)

Created in `hitl_finalize` (graph-based HITL) or `_start_research_from_hitl` (chat-based HITL), immutable throughout execution:

```python
query_anchor = {
    "original_query": str,        # User's original question
    "detected_language": str,     # "de" or "en"
    "key_entities": list[str],    # Extracted entities from HITL
    "scope": str,                 # Research scope
    "hitl_refinements": list[str],# User's clarifications during HITL
    "created_at": str,            # ISO timestamp
}
```

### Tiered Context Entry Structure (NEW)

Each entry in `primary_context`, `secondary_context`, `tertiary_context`:

```python
{
    "chunk": str,                 # Text content (limited to 2000 chars)
    "document": str,              # Source document name
    "page": int | None,           # Page number
    "extracted_info": str,        # Condensed relevant passages
    "relevance_score": float,     # Original vector search score
    "context_tier": int,          # 1, 2, or 3
    "context_weight": float,      # 0.0-1.0 weight for synthesis
    "depth": int,                 # Recursion depth when found
    "source_type": str,           # "vector_search", "reference", "hitl"
    "task_id": int | None,        # Task ID for per-task UI filtering (optional)
    "backfilled": bool,           # True if kept despite low relevance (Phase 3.5)
    "backfill_reason": str,       # Explanation for backfill (when backfilled=True)
    "final_relevance": float,     # Computed relevance score from validate_relevance
    # Reference provenance (Phase 3.8) — present only when depth > 0 and parent known
    "parent_document": str,       # Source document where the reference appeared
    "parent_page": int | None,    # Page in parent document
    "reference_original_text": str,  # Exact reference text ("§21 KrWG")
    "reference_type": str,           # Reference type (e.g. "legal_section")
    "reference_surrounding_context": str,  # ≤500 chars around reference in parent
}
```

**Notes:**
- `task_id` is included when entries are created during `execute_task()`, enabling per-task grouping in the UI. Entries without `task_id` (backward compat) trigger flat chunk rendering instead of tiered display.
- `backfilled` and `backfill_reason` are added by `validate_relevance` when chunk doesn't meet threshold but is kept to ensure minimum chunks per task (transparency feature).

### Preserved Quote Structure (NEW)

```python
{
    "quote": str,                 # Exact verbatim text
    "source": str,                # Source document name
    "page": int,                  # Page number
    "relevance_reason": str,      # Why this must be preserved verbatim
}
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

Accumulates all research findings. Top-level keys: `search_queries` (list of `SearchQueryResult` with query, chunks, summary, references) and `metadata` (total_iterations, documents_referenced, external_sources_used). Each chunk contains `extracted_info`, `references` (with `nested_chunks` from rabbithole traversal), and `relevance_score`. See `src/models/research.py` for full schema.

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
│ hitl_finalize              │  (build query_anchor/hitl_smry, supplementary queries)
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│ assess_query               │  (LLM gate: proceed? + num_tasks)
└────────┬──────────────────┘
         │
    ┌────┴──────────────┐
    │ proceed?           │
    └────┬──────────┬───┘
         │ yes      │ no
         ▼          ▼
┌──────────────┐  ┌────────────────────────────────┐
│ generate_todo │  │ __end__ (rejection FinalReport) │
└──────────────┘  └────────────────────────────────┘
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
9. **hitl_finalize**: Build query_analysis, query_anchor, hitl_smry; generate supplementary research_queries via `_build_diverse_queries()` (question-shaped, excludes original query since Task 0 covers it)
10. **Output**: `research_queries[]` (supplementary only), `query_analysis`, `query_anchor`, `hitl_smry`, `coverage_score`, `query_retrieval`

### Phase 2.5: Query Assessment

1. **Input**: `query_analysis`, `hitl_smry`, `knowledge_gaps`, `detected_language`
2. **LLM Assessment** via `QUERY_ASSESSMENT_PROMPT` → `QueryAssessmentDecision`:
   - `proceed: bool` — whether to run deep research
   - `num_tasks: int` (3-6) — how many ToDo tasks to generate
   - `reason` — rejection code if `proceed=False`: `no_live_data` | `out_of_context` | `no_clear_conversation_steering`
   - `explanation` — human-readable explanation
3. **Rejection path** (`proceed=False`): writes `FinalReport` with apology + reason → `__end__`
4. **Approval path** (`proceed=True`): passes `query_assessment` (with `num_tasks`) to `generate_todo`
5. **Fallback** on LLM error: `proceed=True, num_tasks=5`

### Phase 2: Research Planning

1. **Input**: `QueryAnalysis`, `query_assessment` (with `num_tasks`), `hitl_smry`
2. **ToDoList Generation** (LLM primary):
   - Generates exactly `num_tasks` items (clamped 3-6) using rich query analysis + hitl_smry
   - Fallback 1: `research_queries[:num_tasks]` from HITL if LLM fails
   - Fallback 2: single task from `original_query` as last resort
   - Each item: specific, measurable task anchored to query entities
   - Constraints: max TODO_MAX_ITEMS (15)
3. **HITL Checkpoint**: User approves/modifies tasks
4. **Output**: Approved ToDoList

### Phase 3: Deep Context Extraction (with Graded Classification)

For each ToDoList item (starting from Task 0 = original query):

1. **Multi-Query Generation**: LLM generates 2 targeted queries via `TASK_SEARCH_QUERIES_PROMPT` + 1 base concatenation query
2. **Vector Search**: Execute all 3 queries against ChromaDB, deduplicate by chunk identity
3. **Information Extraction**: Condense relevant passages into `extracted_info` + preserve critical quotes (language-aware)
4. **Context Classification**: Classify each chunk into Tier 1/2/3 based on relevance, depth, entity match
4. **Reference Detection**: Identify section/document/external refs
5. **Agentic Reference Gate**: LLM decides per-reference whether to follow (`ReferenceDecision` model). Skips tangential refs to preserve token budget.
6. **Reference Following**: Resolve and retrieve nested chunks (classified into Tier 2/3)
7. **Task Summary**: Generate structured summary with key findings and LLM-scored relevance (0-100)
7. **ToDoList Update**: Mark task complete and continue to next task

Output: Fully populated ResearchContext + tiered context (primary/secondary/tertiary) + task_summaries (with per-task tiered evidence) + preserved_quotes

### Phase 3.5: Pre-Synthesis Relevance Validation + Backfill

1. **validate_relevance node**: Scores accumulated context against query_anchor
2. **Drift Detection**: Filters items below relevance threshold (0.5 for primary, 0.4 secondary, 0.3 tertiary)
3. **Chunk Backfill** (NEW): Guarantees minimum chunks per task even if below threshold:
   - Primary context: minimum 3 chunks (configurable via `PRIMARY_MIN_CHUNKS`)
   - Secondary context: minimum 2 chunks (configurable via `SECONDARY_MIN_CHUNKS`)
   - Backfilled chunks marked with `backfilled=True` flag + reason for transparency
   - Top-scoring rejected chunks selected when backfill needed
4. **Warning Log**: Logs when >30% of accumulated context is filtered as drift

Output: Filtered tiered context with guaranteed minimums, ready for reranking

### Phase 3.8: Reference Provenance

Nested chunks carry parent document + surrounding context of the reference (reuses `surrounding_window` from the agentic gate — zero extra LLM calls). Stored via `create_tiered_context_entry()` when `depth > 0`. Downstream: reranker penalises off-topic parent context (−20–40 pts), task summariser caps at score 49. See [docs/mindmap_rabbithole_provenance.md](mindmap_rabbithole_provenance.md).

### Phase 3.6: Task Summary Reranking

1. **rerank_task_summaries node**: Deterministic sort — no LLM call
2. **Sort key**: descending `relevance_to_query` float; ascending `task_id` breaks ties
3. **Rank stamping**: adds `rank` int (1 = most relevant) to each summary dict
4. **Low-relevance warning**: logs task IDs with `relevance_to_query < 0.3`
5. **_format_task_summaries()**: renders each header as `--- Task N: ... [Rank: N/total] [Relevance: N/100] ---`
6. **Synthesis prompt rule**: tasks with Relevance ≥70/100 = primary evidence; ≤30/100 = supplementary context only

Output: Sorted task_summaries with rank metadata, ready for synthesis

### Phase 3.9: Batch Chunk Reranking

Batch LLM scoring (~3-4 calls for 20 chunks) with precision/recall strategies. Round-robin batching (`reranker_batch_size=6`) → cross-batch zero-mean normalization → hard-filter below `reranker_min_score` (default 4) → raw 1-5 mapped to 0-100 via `SCORE_TO_100`. Fallback on LLM error: `raw_score = round(relevance_score * 5)`.

### Phase 4: Query-Anchored Synthesis + Quality Assurance

1. **Deep Report Synthesis**: Uses `SYNTHESIS_PROMPT_ENHANCED` to produce extensive, structured deep reports:
   - LLM acts as "expert report writer" — no sentence cap, markdown-formatted output
   - Task summaries (sole evidence source, formatted with key_findings, gaps, preserved quotes via `_format_task_summaries()`)
   - HITL context summary
   - Preserves exact figures, verbatim quotes, section/paragraph references from sources
   - Tiered evidence is resolved at the task summary level, not at synthesis level
2. **Language Enforcement**: `generate_structured_with_language()` validates output language
3. **Quality Check** (optional): Score 0-500 across 5 dimensions (factual accuracy, semantic validity, structural integrity, citation correctness, query relevance)
4. **Agentic Quality Remediation**: If score < `quality_threshold` (375) and `synthesis_retry_count < 1`:
   - LLM evaluates quality scores via `QUALITY_REMEDIATION_PROMPT` → `QualityRemediationDecision`
   - If `action == "retry"`: sets `phase="retry_synthesis"`, increments `synthesis_retry_count`, stores `quality_remediation_focus` (specific improvement instructions)
   - Graph routes back to `synthesize` node, which appends focus instructions to prompt
   - If `action == "accept"` or max retries reached: proceeds to source attribution
   - Max 1 retry to prevent infinite loops

### Phase 5: Source Attribution + Numbered Citations

1. **Source List**: Collect sources from extracted chunks
2. **Report Assembly**: Build `FinalReport` (answer, findings, sources, quality)
3. **Numbered Citations**: `numberify_citations(answer, language)` post-processes the answer text:
   - Scans for all `[Document.pdf, Page N]` / `[Document.pdf, Seite N]` / `[Document.pdf]` patterns
   - Assigns sequential numbers in reading order; same `(doc, page)` pair reuses same number
   - Replaces inline citations with `[N]` markers
   - Appends a `### Quellenverzeichnis` / `### References` block with clickable PDF links
   - PDF links use `/_api/pdf?path=<encoded>` served by the injected Tornado route
4. **PDF Route**: `ensure_pdf_route()` (called from `render_results_view()`) injects `/_api/pdf` Tornado handler once; security validated to serve only files within `kb/` directory

## Streamlit Runtime Model

The UI runs the compiled LangGraph using streaming for live progress updates:

```python
for state in graph.stream(input_state, config, stream_mode="values"):
    update_agent_state(state)
    render_research_status()
```

HITL checkpoints are supported via a persisted `thread_id` stored in session state and reused when resuming.

### UI Data Flow for Chat-Based HITL

The chat-based HITL (`render_chat_hitl`) runs independently from the LangGraph:

1. `HITLService.detect_language()` → `_perform_hitl_retrieval(query, session)` (uses `session.selected_database`)
2. `HITLService.generate_follow_up_questions()` → `_render_retrieval_history()` → user feedback loop
3. On `/end`: `create_hitl_result()` → `_start_research_from_hitl()` (builds `query_anchor`, generates `hitl_smry`, sets `initial_state`)

**Key Data Sources:** `session.hitl_state["retrieval_history"]` (HITL phase) and `session.agent_state["retrieval_history"]` (graph execution). `_render_retrieval_history()` reads both, preferring `hitl_state`.

### Cached Service Clients

To improve performance on Streamlit reruns, service clients are cached:

```python
@st.cache_resource
def _get_chromadb_client():    # safe_exit.py
def get_chromadb_client():     # app.py
def _get_ollama_client():      # safe_exit.py
def _get_hitl_service():       # hitl_panel.py
def _ensure_gpu_route():       # gpu_widget.py (one-time Tornado route injection)
```

This prevents re-loading the embedding model and reconnecting to services on every UI interaction.

**Runtime reset on model switch**: When the user changes the research depth selector, `_apply_research_depth()` calls `reset_ollama_client()` on all 3 consumer modules (`nodes.py`, `tools.py`, `hitl_service.py`) and clears the `@st.cache_resource` for `_get_hitl_service` and `_get_ollama_client`. This ensures new `OllamaClient` instances pick up the changed `settings.ollama_model`.

### Research Depth Selector (Sidebar)

The "Erweiterte Einstellungen" expander in the sidebar contains a selectbox for runtime model switching:

| Label | Model | Family | Default |
|-------|-------|--------|---------|
| basic (gemma4:e4b) | `gemma4:e4b` | gemma4 | ✓ |
| einfach (qwen3:8b) | `qwen3:8b` | qwen | |
| standard (qwen3:14b) | `qwen3:14b` | qwen | |
| erhöht (gpt-oss:20b) | `gpt-oss:20b` | gpt-oss | |
| tief (qwen3:30b) | `qwen3:30b` | qwen | |

On change, `_apply_research_depth()`:
1. Sets `settings.ollama_model` to the new model name
2. Resets all module-level `OllamaClient` singletons via `reset_ollama_client()`
3. Clears `@st.cache_resource` cached services
4. Prompt routing auto-adapts via PEP 562 `__getattr__` in `src/prompts/__init__.py`

The selectbox is disabled during active research (`workflow_phase == "research"`) and persists across reruns via `session.research_depth`.

### GPU Widget (Sidebar)

Live GPU temperature, fan speed, and utilization — plus elapsed research time — are displayed in the sidebar via a Tornado route injection pattern:

1. **`_ensure_gpu_route()`** (`@st.cache_resource`): One-time injection that:
   - Checks `nvidia-smi` availability; returns `False` if no GPU
   - Discovers the live `tornado.web.Application` via `gc.get_objects()` (Streamlit ≥1.53 removed `Server.get_current()`)
   - Registers a `GPUStatsHandler` at `/_api/gpu` via `tornado_app.add_handlers()`
   - Double-injection guard checks `default_router.rules` (where `add_handlers` writes)
2. **`render_gpu_sidebar()`**: Renders a `components.v1.html()` snippet in the sidebar whose JS fetches `/_api/gpu` every 1s
3. **Why Tornado**: Tornado's I/O loop runs independently of Streamlit's script-runner thread, so GPU stats keep updating even while `graph.stream()` blocks for 30s+. `@st.fragment(run_every=...)` is not viable because fragments queue on the same single thread.

**Response format** (as of Phase 6.11):
```json
{"gpus": [{"name": "...", "fan": "33", "temp": "48", "util": "88"}], "elapsed": 42, "is_running": true}
```
`elapsed` is `null` until the user approves the todo list; `is_running` turns `false` when the report is generated.

**Elapsed research time** (`gpu_widget.py` module-level globals, three public setters):
- `set_research_start()` — called in `app.py` immediately after todo approval
- `set_research_end()` — called when `session.final_report` is detected
- `reset_research_timer()` — called on "Neue Recherche starten"

Display format: two GPU lines + `llm: <model>` + optional `t: Xs...` (green, running) / `t: Xs` (grey, done):
```
RTX 4090    48°C|Fan:33%|Load: 88%
llm: qwen3:14b
t: 127s
```
Color coding: temp green/orange/red at 70/80°C; load green/orange/red at 50/80%; elapsed green while running, grey when complete.

### Graph Entry Point Routing

The `route_entry_point()` function in `graph.py` handles multiple entry scenarios:

```python
def route_entry_point(state) -> Literal["hitl_init", "hitl_process_response", "assess_query", "process_hitl_todo"]:
    # 1. hitl_decision + !hitl_active → process_hitl_todo (post-approval resume)
    # 2. hitl_decision + hitl_active → hitl_process_response (iterative HITL resume)
    # 3. research_queries present → assess_query (skip HITL, go straight to assessment)
    # 4. phase == "generate_todo" → assess_query
    # 5. else → hitl_init (start new)
```

This enables:
- **Resume after todo approval**: When user approves/modifies tasks (`hitl_decision` present, `hitl_active=False`)
- **Skip HITL**: When UI chat-based HITL has already produced research_queries → goes to `assess_query`
- **Resume HITL**: When user responds to an interrupted iterative HITL session
- **New HITL**: Default behavior when starting fresh

**Key invariants**:
- `_start_research_from_hitl()` sets `hitl_active=False` before entering the graph, so post-approval resume never misroutes to `hitl_process_response`.
- `_start_research_from_hitl()` generates `hitl_smry` (via `_generate_hitl_summary()`) and builds `query_anchor`, since the chat-based HITL UI bypasses `hitl_finalize` entirely.

### Completed Results View

When `workflow_phase == "completed"`, `render_results_view()` shows the final report **plus** persisted HITL and task data via two private helpers in `results_view.py`:

- **`_render_hitl_expander(session)`**: Expanded expander with conversation history (`st.chat_message`), `hitl_smry`, and numbered research queries
- **`_render_task_expanders(session)`**: One expanded expander per task with summary (via `render_task_summary_markdown()`), and chunks grouped by tier via `render_tiered_chunks()` (Tier 1 expanded, Tier 2/3 collapsed, empty tiers hidden). Falls back to flat chunk rendering for old states without `task_id` entries.

Data sources:
- `session.hitl_conversation_history` — persists across phase transitions, cleared only on "Neue Recherche starten"
- `session.agent_state["hitl_smry"]`, `["task_summaries"]`, `["todo_list"]`, `["research_context"]`, `["primary_context"]`, `["secondary_context"]`, `["tertiary_context"]` — set via `update_agent_state()`, never cleared until reset
- `session.hitl_result["research_queries"]` — set at end of HITL phase
