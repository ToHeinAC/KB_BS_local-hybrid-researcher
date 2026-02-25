# Implementation

## Implementation Phases

### Phase 1: Core Infrastructure
- [x] Project setup: pyproject.toml, directory structure
- [x] Config management with pydantic-settings
- [x] Pydantic models: QueryAnalysis, ToDoList, ResearchContext
- [x] Pydantic models: VectorResult, DocumentFinding, WebResult
- [x] Pydantic models: QualityAssessment, FinalReport
- [x] ChromaDB client service
- [x] Ollama LLM client service (with `json_mode` structured output)
- [x] PDF reader service (PyMuPDF)
- [x] Basic tests for services and models

### Phase 2: HITL + ToDoList (Research Phase 1-2)
- [x] Query analysis with NER/keyword extraction
- [x] HITL conversational interface for refinement
- [x] ToDoList generation (3-5 tasks)
- [x] HITL checkpoint for task approval/modification
- [x] Tests for HITL flow

### Phase 2.5: Enhanced Iterative HITL + Query Assessment Gate
- [x] **Iterative Retrieval-HITL Nodes** in `nodes.py`:
  - `hitl_init()`: Initialize conversation, detect language
  - `hitl_generate_queries()`: Generate 3 queries (original + 2 alternatives)
  - `hitl_retrieve_chunks()`: Search ChromaDB, deduplicate, append to context
  - `hitl_analyze_retrieval()`: LLM analysis for concepts, gaps, coverage
  - `hitl_generate_questions()`: Generate contextual follow-up questions (uses `query_retrieval` from state)
  - `hitl_process_response()`: Analyze user feedback, check termination
  - `hitl_finalize()`: Generate research_queries for Phase 2
- [x] **Convergence Detection**: Coverage ≥ 0.8, dedup ratio ≥ 0.7, gaps ≤ 2
- [x] **Query Assessment Gate** (`assess_query` node):
  - LLM decides proceed/reject + num_tasks (3-6) via `QueryAssessmentDecision`
  - Rejection routes to `__end__` with explanation (no research run)
  - `generate_todo` priority flip: LLM generation is **primary** (informed by `hitl_smry` + `num_tasks`); `research_queries[:num_items]` is **Fallback 1**
- [x] **Graph Entry Routing**: Conditional entry point (`route_entry_point`)
- [x] **UI Support**: Live display of retrieval stats and coverage during HITL phase
- [x] **Centralized Prompts**: All LLM prompts in `src/prompts/` package (`hitl.py`, `research.py`, `synthesis.py`)


### Phase 3: LangGraph Agent (Research Phase 3)
- [x] LangGraph StateGraph setup with TypedDict state (v1.0 pattern)
- [x] State serialization helpers (Pydantic <-> dict)
- [x] `vector_search` tool implementation
- [x] `extract_references` tool
- [x] `resolve_reference` tool
- [x] Reference following with depth tracking
- [x] Relevance filtering (threshold 0.6)
- [x] ToDoList re-evaluation after each task
- [x] Loop prevention (visited refs, max iterations)
- [x] Tests for agent and tools

### Phase 3.5: Enhanced Reference Following
- [x] **Hybrid Reference Detection** (`detect_references_hybrid()`):
  - Regex (7 patterns) + LLM (`REFERENCE_EXTRACTION_PROMPT`) with deduplication
  - Configurable via `reference_extraction_method` setting: `"regex"`, `"llm"`, `"hybrid"`
- [x] **Document Registry** (`kb/document_registry.json`):
  - Maps PDF filenames to synonyms across 4 collections
  - `load_document_registry()`: singleton loader
  - `resolve_document_name()`: 3-stage matching (exact > fuzzy 0.7 > substring)
- [x] **Enhanced Resolution** (`resolve_reference_enhanced()`):
  - Routes by ref type: legal -> registry scoped, document -> registry scoped, academic -> broad
  - `_vector_search_scoped()`: searches specific collection, post-filters by document name
- [x] **Traversal Controls**:
  - Token budget tracking (`reference_token_budget`, default 50K)
  - Convergence detection (`detect_convergence()`, threshold 3)
- [x] **New Models**: `ExtractedReference`, `ExtractedReferenceList` in `src/models/research.py`
- [x] **Extended `DetectedReference`**: `document_context`, `extraction_method` fields
- [x] **39 Tests**: `tests/test_reference_extraction.py`

- [x] **Graded Context Management**: Tiered context classification (Tier 1/2/3), query_anchor, hitl_smry, preserved quotes, per-task summaries, pre-synthesis relevance validation, deep report synthesis

- [x] **Prompt Standardization & Multi-Query Execution**: SYSTEM/HUMAN split, universal {language} enforcement, multi-query (3/task), Task 0 prepend, 5-dimension quality scoring (0-500), relevance filter backfill, language-aware extraction

- [x] **Agentic Decision Points**: Reference following gate (ReferenceDecision), quality remediation loop (max 1 retry with focused instructions)

- [x] **Tiered Context & HITL Pipeline Fixes**: L2→cosine conversion fix, chat HITL generates hitl_smry/query_anchor, sync conversation history on all termination paths


### Phase 3.13: Database Selection Propagation Fix

Bug fix: the user-selected database was respected for initial vector searches but bypassed in all three reference-resolution fallback paths.

- [x] **Root cause**: `execute_task()` extracted `selected_database` from state and passed it to the main `vector_search()` call, but did **not** pass it to `resolve_reference_enhanced()`. Inside that function, three fallback helpers (`_resolve_section_ref`, `_resolve_document_ref`, `_resolve_academic_ref`) called `vector_search()` without `selected_database`, causing them to search all databases.

- [x] **Fix** (`src/agents/tools.py`):
  - `_resolve_section_ref(ref, current_doc, selected_database=None)`: passes `selected_database` to `vector_search()`
  - `_resolve_document_ref(ref, selected_database=None)`: passes `selected_database` to `vector_search()`
  - `_resolve_academic_ref(ref, selected_database=None)`: passes `selected_database` to `vector_search()`
  - `_resolve_legal_ref_enhanced(ref, current_doc, selected_database=None)`: passes to `_resolve_section_ref()` fallback
  - `_resolve_document_ref_enhanced(ref, selected_database=None)`: passes to `_resolve_document_ref()` fallback
  - `resolve_reference_enhanced(..., selected_database=None)`: accepts and threads `selected_database` to all three sub-resolvers

- [x] **Caller update** (`src/agents/nodes.py`): `resolve_reference_enhanced()` call now includes `selected_database=selected_database`

- [x] **Note**: `_vector_search_scoped()` is **intentionally unchanged** — it uses a registry-resolved `collection_key` (e.g. `"StrlSch"`) which is a targeted cross-document search within the KB. That scoped path does not override user intent; only the broad fallback searches do.

- [x] **179 Unit Tests**: All pass (3 new in `TestSelectedDatabasePropagation`: section ref, document ref, academic ref)

### Phase 4: Synthesis + Quality (Research Phase 4)
- [x] `synthesize` node (LLM synthesis from extracted findings)
- [x] Enhanced `synthesize` with pre-digested task summaries, HITL summary, language enforcement
- [x] `quality_check` node (optional, 0-500 scoring, 5 dimensions)
- [x] Tests for synthesis + QA

### Phase 5: Source Attribution (Research Phase 5)
- [x] `attribute_sources` node (FinalReport assembly)
- [x] Source list generation (linked sources)
- [x] Tests for attribution

### Phase 6: Streamlit UI
- [x] Basic app layout with query input
- [x] HITL panel (clarification questions)
- [x] ToDoList component (real-time updates)
- [x] Live progress updates via LangGraph streaming
- [x] Results view with linked sources
- [x] Session state management
- [x] Safe exit button (port-aware kill)
- [x] Source inspection view

### Phase 6.5: UI Enhancements
- [x] Retrieval history panel, database selection fix, cached service clients, graph entry routing, coverage metrics in checkpoints

### Phase 6.7: Todo Side Panel & Streaming Improvements
- [x] Expander-based task list, verbose task spinner, simplified graph streaming, sequential task ID renumbering

### Phase 6.8: Persistent Results View
- [x] HITL expander (conversation, hitl_smry, research queries), per-task expanders with tiered chunks, shared task rendering helpers

### Phase 6.9: Tiered Chunk Rendering in Task Expanders
- [x] `task_id` in tiered context entries, tiered rendering helpers with backward compat fallback

- [x] **Prompt Optimization for Local LLMs**: 5 SYSTEM prompts → XML tag format (see docs/prompt-opt-guide.md for principles); output format placed 2nd, HARD CONSTRAINTS separated, realistic domain examples

### Phase 3.6: Pre-Synthesis Task Summary Reranker
- [x] `rerank_task_summaries()` node: sort by relevance_to_query desc, stamp rank int; warnings for relevance < 0.3
- [x] Graph wiring: `validate_relevance` → `rerank_task_summaries` → `synthesize`
- [x] Synthesis weighting: `[Relevance ≥70/100]` = primary evidence, `≤30/100` = supplementary only

### Phase 3.7: Chunk Filtering with Minimum Guarantees (NEW)
- [x] **Configuration**: Added `primary_min_chunks` (default: 3) and `secondary_min_chunks` (default: 2) settings
- [x] **Core Logic** (`src/agents/nodes.py`):
  - Enhanced `_score_and_filter_context()` with `min_results` parameter
  - Implemented backfill logic: if fewer than `min_results` pass threshold, backfill with top-scoring rejected chunks
  - Updated `validate_relevance()` to use guaranteed minimums for primary/secondary context
  - Added logging for backfill events
- [x] **Transparency Markers**:
  - Backfilled chunks marked with `backfilled=True` flag
  - `backfill_reason` field explains why chunk was kept
- [x] **UI Enhancements** (`src/ui/components/`):
  - `render_chunk_expander()`: displays ⚠️ badge for backfilled chunks
  - Info box shows backfill reason (e.g., "Below threshold 0.50, kept for visibility")
  - `results_view.py`: per-task backfill statistics display
- [x] **Test Coverage** (`tests/test_chunk_filtering.py`):
  - 11 comprehensive tests for backfill logic
  - All tests passing (11/11 new + 65/65 existing)
- [x] **Documentation**: Updated `.env.example`, `CLAUDE.md`, `docs/architecture.md`, `docs/configuration.md`

**Rationale**: Ensures primary retrieval results remain visible even if they don't meet strict relevance thresholds, preventing silent suppression while maintaining transparency through visual indicators.

### Phase 3.8: Reference Provenance (NEW)

Solves the lost-context problem: nested chunks were scored in isolation, allowing off-topic
references to pollute synthesis. The `surrounding_window` already computed for the agentic gate
is now attached to each `NestedChunk` and propagated through the pipeline — zero new LLM calls.

- [x] **`NestedChunk` model** (`src/models/research.py`): 5 new optional provenance fields:
  `parent_document`, `parent_page`, `reference_original_text`, `reference_type`,
  `reference_surrounding_context` (all backward-compatible, default `""` / `None`)
- [x] **`create_tiered_context_entry()`** (`src/agents/tools.py`): accepts 5 provenance kwargs;
  writes them to the dict only when `depth > 0 and parent_document` non-empty
- [x] **`execute_task()`** (`src/agents/nodes.py`):
  - Initialises `surrounding_window = ""` before the `try:` block (safe default)
  - After `resolve_reference_enhanced()`, loops over nested chunks to attach provenance
  - Forwards provenance kwargs to `create_tiered_context_entry()`
- [x] **`_rerank_task_chunks()`** (`src/agents/nodes.py`): reads `reference_surrounding_context`
  from the chunk dict; passes as `parent_context` to `CHUNK_RERANKER_PROMPT_HUMAN`
- [x] **`_format_ranked_findings()`** (`src/agents/nodes.py`): for depth>0 chunks with
  `parent_document`, appends `[via ref_type ref "..." in Parent.pdf, Page N]` +
  `Parent context: "..."` lines after the standard header
- [x] **`CHUNK_RERANKER_PROMPT_SYSTEM`** (`src/prompts/research.py`): new rule 2 — penalise
  off-topic parent context by 20-40 pts; old rules renumbered 3-5
- [x] **`CHUNK_RERANKER_PROMPT_HUMAN`**: adds `parent_context: {parent_context}` field
- [x] **`TASK_SUMMARY_PROMPT_SYSTEM`** processing rule 2e: chunks with off-topic parent context
  capped at effective score 49 (supplementary, never elevated to `key_findings`)
- [x] **Tests** (`tests/test_provenance.py`): 11 new unit tests, all passing
- [x] **Design doc**: `docs/mindmap_rabbithole_provenance.md`

**212 tests total, all passing.**

### Phase 7: Polish
- [x] Multi-collection search
- [ ] Query history and caching
- [ ] Export results (JSON, Markdown)
- [x] Error handling and recovery
- [x] Logging and observability

### Phase 6.6: UI Localization & Layout Fixes
- [x] German localization, layout fixes

### Phase 3.9: Batch Chunk Reranking
- [x] **Batch architecture**: `_build_reranker_batches()` splits chunks via round-robin into batches of `reranker_batch_size` (default 6)
- [x] **Dual strategy**: `_rerank_batch()` supports `precision` and `recall` strategies with separate prompt pairs (`RERANKER_PRECISION_PROMPT`, `RERANKER_RECALL_PROMPT`)
- [x] **Cross-batch normalization**: `_normalize_batch_scores()` applies zero-mean normalization for comparability across batches
- [x] **Hard filtering**: Drops chunks with raw score < `reranker_min_score` (default 4)
- [x] **Score mapping**: Raw 1-5 → 0-100 via `SCORE_TO_100` dict for downstream `_format_ranked_findings` / `TASK_SUMMARY` compatibility
- [x] **New models**: `RerankerChunkResult`, `RerankerBatchOutput`, `RerankerRecallChunkResult`, `RerankerRecallBatchOutput` in `src/models/results.py`
- [x] **New prompts**: `RERANKER_PRECISION_PROMPT_{SYSTEM,HUMAN}`, `RERANKER_RECALL_PROMPT_{SYSTEM,HUMAN}` in `src/prompts/research.py`
- [x] **Configuration**: `reranker_strategy`, `reranker_batch_size`, `reranker_min_score` in `src/config.py`
- [x] **Tests**: `tests/test_batch_reranking.py` (11 tests) + updated `TestChunkReranker` in `tests/test_agents.py`

### Phase 3.10: Diverse Research Queries (Question-Shaped, No Duplication)
- [x] **`_build_diverse_queries()`** rewritten in `src/services/hitl_service.py`:
  - Excludes `user_query` from output (Task 0 already covers it)
  - Accepts `language` parameter for German/English question templates
  - Produces question-shaped queries: `"Welche Regelungen gelten für {entity} im Bereich {scope}?"`
  - Refined query passed through as-is (usually already a sentence from LLM)
- [x] **Removed**: `KNOWLEDGE_BASE_QUESTIONS_PROMPT` (unused LLM-based query generation), `_generate_knowledge_base_questions_llm()`, `HITLService.generate_knowledge_base_questions()`
- [x] **Removed**: `max_search_queries` UI slider (query count now driven by `num_tasks` from assess_query)
- [x] **Callsites updated**: 3 in `hitl_panel.py` + 1 in `finalize_hitl_conversation()` — all pass `language`
- [x] **Tests**: `tests/test_diverse_queries.py` (12 tests)

**238 tests total, all passing.**

### Phase 8: Testing Improvements
- [x] `TestRouteEntryPoint` class for graph routing logic
  - `test_route_to_hitl_init_on_new_session`
  - `test_route_to_hitl_process_response_on_resume`
  - `test_route_to_generate_todo_with_research_queries`
  - `test_route_to_generate_todo_with_phase`
  - `test_decision_without_hitl_active_routes_to_process_hitl_todo`

