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
│  Phase 3.10: Optional Web Search (Tavily API)                            │
│  └─ web_search node (runs only when user enables per session)            │
│     ├─ LLM generates search query from key_findings + gaps               │
│     ├─ Tavily REST API call → web results                                │
│     ├─ LLM summarizes with [Title](URL) citations                        │
│     ├─ Contradiction detection against KB findings                        │
│     └─ Strictly separated: appended in attribute_sources, not synthesize │
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

`hitl_init` → `hitl_generate_queries` (3/iteration) → `hitl_retrieve_chunks` → `hitl_analyze_retrieval` → `hitl_generate_questions` → `hitl_process_response` [loop or terminate] → `hitl_finalize`. Termination: `/end`, max 5 iterations, or convergence (coverage ≥ 0.8, dedup ≥ 0.7, gaps ≤ 2). Output: `query_anchor`, `hitl_smry`, supplementary `research_queries[]`.

### Phase 2.5: Query Assessment

`QUERY_ASSESSMENT_PROMPT` → `QueryAssessmentDecision(proceed, num_tasks 3-6, reason, explanation)`. `proceed=False` → `__end__` with rejection FinalReport. `proceed=True` → `generate_todo(num_tasks)`. Rejection reasons: `no_live_data | out_of_context | no_clear_conversation_steering`. Fallback: `proceed=True, num_tasks=5`.

### Phase 2: Research Planning

LLM generates `num_tasks` ToDoItems from `QueryAnalysis` + `hitl_smry` (clamped 3–6). Fallback 1: `research_queries[:num_tasks]`. Fallback 2: single task from `original_query`. HITL checkpoint for user approval/modification.

### Phase 3: Deep Context Extraction (with Graded Classification)

For each task: multi-query (3) → vector search → dedup → extract info + quotes → classify Tier 1/2/3 → reference detection → agentic gate (`ReferenceDecision`) → scoped resolution → classify nested chunks → convergence check → task summary (0-100 relevance score).

Output: ResearchContext + tiered context (primary/secondary/tertiary) + task_summaries + preserved_quotes.

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

### Phase 3.10: Optional Web Search (Tavily API)

Runs only when user enables web search via the GUI checkbox. Inserted between `rerank_task_summaries` and `synthesize` in the graph.

1. **Guard**: Returns `{}` immediately if `enable_web_search` is `False`
2. **Query generation**: LLM generates one concise search query (4-8 keywords) from `key_findings` + `gaps` across task summaries via `WEB_SEARCH_QUERY_PROMPT`
3. **Tavily API call**: `tavily_search(query)` POSTs to `https://api.tavily.com/search`, returns raw result dicts
4. **Result formatting**: `format_tavily_results()` converts to `WebResult` instances; `format_results_for_prompt()` builds numbered text blocks
5. **LLM summarization**: `WEB_SEARCH_SUMMARIZE_PROMPT` → `WebSearchSummaryOutput` with `[Title](URL)` citations and contradiction detection against KB `key_findings`
6. **Contradiction notice**: If contradictions found, prepended as `**Widersprüche**` (de) / `**Contradictions**` (en) block

**Strict separation**: Web summary is stored in `web_search_summary` state field and appended in `attribute_sources()` as a distinct markdown section (`### Ergänzende Webrecherche` / `### Supplementary Web Research`). It is **never** passed into the `synthesize()` prompt.

Output: `web_search_results` (list of WebResult dicts), `web_search_summary` (formatted markdown)

### Phase 4: Query-Anchored Synthesis + Quality Assurance

`SYNTHESIS_PROMPT_ENHANCED` → markdown deep report from task summaries + hitl_smry. Language enforcement via `generate_structured_with_language()`. Optional quality check (0-500, 5 dimensions). Agentic remediation: if score < 375 and `synthesis_retry_count < 1`, `QUALITY_REMEDIATION_PROMPT` → `QualityRemediationDecision(accept|retry)` → retry appends `quality_remediation_focus` to prompt (max 1 retry).

### Phase 5: Source Attribution + Numbered Citations

`attribute_sources()` builds `FinalReport`. `numberify_citations(answer, language)` replaces `[Doc.pdf, Page N]` patterns with sequential `[N]` markers + appended `### Quellenverzeichnis/References` block with `/_api/pdf` links. `ensure_pdf_route()` injects the Tornado PDF handler once (path validated to `kb/` dir).

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

**"Free GPU & Reset" button** (`safe_exit.py`): Graceful GPU release without stopping the server — keeps Cloudflare tunnel alive.
1. `POST /api/generate` with `keep_alive: 0` → Ollama evicts current model from VRAM
2. `.clear()` on all four cached resource functions → releases Python refs to HuggingFace embedding model
3. `torch.cuda.empty_cache()` + `gc.collect()` → frees CUDA allocator fragment cache
4. `reset_ollama_client()` on `nodes`, `tools`, `hitl_service` modules
5. `reset_session_state()` + `reset_research_timer()` → fresh UI state
6. `st.rerun()` → app reloads to HITL screen; port process untouched

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

**Model sync on first load**: `_apply_research_depth(depth)` is now called unconditionally on every sidebar render. Its guard (`if model_name == settings.ollama_model: return`) keeps repeated calls cheap. This ensures `settings.ollama_model` is always aligned with the UI selection even on first page load (when `.env` may carry a different model than the session default `gemma4:e4b`).

### Wissensdatenbank Panel (Sidebar)

A **"Dokumente anzeigen"** button below the embedding caption opens a native `@st.dialog` modal listing all unique document filenames in the selected database. Closes via ✕; no app state affected.

- `ChromaDBClient.get_document_names(db_name)` — raw `chromadb.PersistentClient` metadata-only query (no embeddings); extracts `original_filename` / `source` / `filename`, deduplicates, returns sorted list.
- Button visible only when a specific database is selected (`use_ext_db=True`).

### GPU Widget (Sidebar)

Live GPU temp/fan/load + elapsed research time via Tornado route injection (`/_api/gpu`), updating every 1s independently of Streamlit's script-runner thread.

- **`_ensure_gpu_route()`**: One-time injection via `gc.get_objects()` → `tornado_app.add_handlers()` with double-injection guard
- **`render_gpu_sidebar()`**: `components.v1.html()` with JS polling; color-coded thresholds (temp 70/80°C, load 50/80%)
- **Elapsed time**: `set_research_start()` on todo approval, `set_research_end()` on report, `reset_research_timer()` on new session
- **Response**: `{"gpus": [...], "elapsed": int|null, "is_running": bool}`
- **Why Tornado**: I/O loop is independent — `@st.fragment(run_every=...)` blocks during `graph.stream()`

### Remote Access (Cloudflare Tunnel)

The `login/` directory provides remote access via Cloudflare quick tunnels:

```
┌──────────────────────────────────────────────────────────────────┐
│  Remote User                                                      │
│  ↓ (HTTPS via *.trycloudflare.com)                               │
├──────────────────────────────────────────────────────────────────┤
│  cloudflared tunnel (port 8522)    cloudflared tunnel (port 8511)│
│  ↓                                  ↓                            │
│  Launcher App (login/launcher_app.py)  Main Streamlit App        │
│  Port 8522                              Port 8511                │
│  ├─ Password gate (LAUNCHER_PASSWORD)                            │
│  ├─ Start/Stop/Restart controls                                  │
│  ├─ Process monitoring (psutil)                                  │
│  └─ Log viewer                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Key files:**
- `login/launcher_app.py` — password-gated Streamlit control panel
- `login/start-quick-tunnels.sh` — creates two quick tunnels (temporary URLs)
- `login/start-launcher.sh` — starts the launcher via `uv run`
- `login/cloudflared-config.yml` — template for persistent tunnel (requires domain)

**Tunnel coexistence:** Scripts use targeted `pkill` by port URL to avoid killing other tunnels (e.g., `brain-nw1`). URL files are project-specific (`/tmp/hybrid-*-url.txt`).

**Quick tunnels vs. persistent:** Quick tunnels generate temporary `*.trycloudflare.com` URLs. For permanent URLs, a Cloudflare-managed domain is required (see `login/README.md` for upgrade path).

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
