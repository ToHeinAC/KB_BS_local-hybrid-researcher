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
- [x] "Free GPU & Reset" button: Ollama `keep_alive=0` unload + clear all `@st.cache_resource` caches + `torch.cuda.empty_cache()` + session reset; server stays alive (Cloudflare tunnel preserved)
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

### Phase 6.10: Live GPU Widget (Sidebar)
- [x] **Tornado route injection** (`src/ui/components/gpu_widget.py`): `_api/gpu` endpoint serving live `nvidia-smi` stats as JSON, registered under `server.baseUrlPath`
- [x] **gc-based discovery**: `gc.get_objects()` finds the live `tornado.web.Application` (Streamlit ≥1.53 removed `Server.get_current()`)
- [x] **Double-injection guard**: Checks `default_router.rules` (where `add_handlers` writes) to prevent duplicate registration
- [x] **Sidebar rendering**: `components.v1.html()` with JS polling `./_api/gpu` (relative — survives the /brain/ proxy prefix) every 1s; fixed-width monospace layout
- [x] **Color coding**: Temp (green <70°C, orange <80°C, red ≥80°C), Load (green <50%, orange <80%, red ≥80%)
- [x] **Graceful degradation**: No GPU / no `nvidia-smi` → widget not rendered, no errors
- [x] **Why not `@st.fragment`**: Fragments queue on the same script-runner thread and block during `graph.stream()`; Tornado I/O loop is independent

### Phase 6.11: Elapsed Research Time in GPU Widget
- [x] **Module-level timing state** (`gpu_widget.py`): `_research_start_time` / `_research_end_time` (float | None) — safe for single-user local app
- [x] **Three public setters**: `set_research_start()`, `set_research_end()`, `reset_research_timer()` — called from `app.py`
- [x] **Updated `_api/gpu` response**: `{"gpus": [...], "elapsed": int|null, "is_running": bool}` (backward-compatible within widget)
- [x] **JS rendering**: `t: Xs...` in green (`#21c354`) while running; `t: Xs` in grey (`#aaa`) when done; line hidden before first approval
- [x] **Lifecycle hooks in `app.py`**:
  - `set_research_start()` — called right after todo approval (`_resume_with_decision`)
  - `set_research_end()` — called when `session.final_report` is detected before `set_workflow_phase("completed")`
  - `reset_research_timer()` — called in "Neue Recherche starten" reset handler
- [x] **Component height**: 70 → 85 px to accommodate the extra line

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

### Phase 3.11: Model-Conditional Prompt Routing (gpt-oss Support)
- [x] **`src/config.py`**: Added `ollama_temperature: float = 0.0` field and `model_family` property (returns `"gpt-oss"` or `"qwen"` based on `ollama_model` prefix)
- [x] **`src/services/ollama_client.py`**: gpt-oss runtime adaptations:
  - `is_gpt_oss` property for model detection
  - `_HARMONY_PREAMBLE` constant prepended to system prompts for gpt-oss models
  - `_prepare_system_prompt()` applies preamble conditionally
  - `_extract_json_from_tags()` regex extracts content between `<json>...</json>` tags
  - `generate_messages()` and `generate_structured_messages()` apply `_prepare_system_prompt()`
  - `generate_structured_messages()` error handler: on gpt-oss, falls back to raw invoke + tag extraction + manual validation
  - `temperature=0` replaced with `settings.ollama_temperature` in both `llm` and `fallback_llm`
- [x] **gpt-oss prompt files** (Harmony format: `# headers`, flat rules, `<json>` tags, no `/no_think`):
  - `src/prompts/hitl_gpt.py`: 16 constants (8 SYSTEM/HUMAN pairs)
  - `src/prompts/research_gpt.py`: 18 constants (9 pairs)
  - `src/prompts/synthesis_gpt.py`: 14 constants (7 pairs)
  - All export identical constant names as Qwen counterparts — zero consumer changes
- [x] **`src/prompts/__init__.py`**: Conditional routing via `settings.model_family`
- [x] **Tests**: `tests/test_prompt_routing.py` (15 tests), `tests/test_ollama_gpt_oss.py` (12 tests)

**265 tests total, all passing.**

### Phase 3.11 addendum: gemma4 Support
- [x] **`src/config.py`**: `model_family` property extended — returns `"gemma4"` via case-insensitive substring match (`"gemma4"` or `"gemma-4"` in lowercased model string); both `"gemma4"` and `"qwen"` resolve to the Qwen prompt set
- [x] **`src/prompts/__init__.py`**: `__getattr__` treats `"gemma4"` as alias for Qwen prompts (comment added)
- [x] **`src/services/ollama_client.py`**: `is_gemma4` property uses case-insensitive check; `_prepare_system_prompt()` strips `/no_think` tokens for gemma4 models (Qwen3-specific directive unsupported by Gemma)
- [x] **`src/ui/state.py`**: `k_results` default raised 3→6
- [x] **Tests**: `tests/test_prompt_routing.py` — `TestGemma4Support` + `TestGemma4E4BSupport` classes

### Phase 3.12: Runtime Model Selection (Research Depth Selector)
- [x] **Dynamic prompt routing** (`src/prompts/__init__.py`): Rewrote from static wildcard imports to PEP 562 `__getattr__`:
  - Both Qwen and gpt-oss prompt sets eagerly loaded into `_qwen_prompts` / `_gptoss_prompts` dicts
  - `__getattr__(name)` checks `settings.model_family` **at access time**, enabling runtime model switching
  - `__all__` lists union of all prompt constant names
- [x] **Consumer migration** — replaced `from src.prompts import X` with `from src import prompts` + `prompts.X` at call sites:
  - `src/agents/nodes.py`: 29 prompt constants, added `reset_ollama_client()` (clears `_ollama_client` + `_hitl_service`)
  - `src/agents/tools.py`: 6 prompt constants, added `reset_ollama_client()`
  - `src/services/hitl_service.py`: 14 prompt constants, added `reset_ollama_client()`
- [x] **UI depth selector** (`src/ui/app.py`):
  - `st.selectbox` in "Erweiterte Einstellungen" with 6 options: basic (gemma4:e4b), einfach (granite4.1:3b), standard (qwen3:14b), erhöht (gpt-oss:20b), tief (qwen3:30b), specialized (north-mini-code-1.0)
  - Disabled during active research (`workflow_phase == "research"`)
  - `_apply_research_depth()` coordinator: updates `settings.ollama_model`, calls `reset_ollama_client()` on all 3 modules, clears `@st.cache_resource` for HITLService and OllamaClient
- [x] **Session state** (`src/ui/state.py`): Added `research_depth` field (default: `"basic (gemma4:e4b)"`)

### Phase 6.12: Todo Approval Multi-Task + UI Polish
- [x] **Dynamic new-task list** (`src/ui/components/todo_approval.py`): Replaced single text input with dynamic list of pending tasks (add/remove rows)
- [x] **HITL summary hiding**: Summary hidden during active task execution (`todo_approved` flag)
- [x] **Streaming cleanup**: Activity log hidden during streaming; clean "Recherche-Ergebnisse" header shown instead
- [x] **`todo_approved` flag** (`src/ui/state.py`): Tracks whether user has approved tasks, reset in `reset_hitl_conversation()`

### Phase 6.13: Numbered Citation Transformation (Post-Processing)

Replaces verbose inline `[Document.pdf, Page N]` citations with sequential `[1]`, `[2]`, … markers and appends a formatted reference list with clickable PDF links — zero LLM calls.

- [x] **`numberify_citations(answer, language)`** (`src/agents/tools.py`):
  - Regex `_CITATION_RE` matches `[*.pdf, Page N]`, `[*.pdf, Seite N]`, `[*.pdf]`; negative lookahead avoids markdown links
  - Assigns numbers in reading order; same `(doc_name.lower(), page)` key reuses same number
  - Replaces right-to-left (preserves string offsets)
  - Appends `### Quellenverzeichnis` / `### References` block with `[N] Doc.pdf, Page X — [PDF öffnen](_api/pdf?path=...)` lines
- [x] **`resolve_pdf_path(doc_name)`** (`src/agents/tools.py`):
  - Scans `kb/*__db_inserted/` folders (exact match, then case-insensitive fallback)
  - Returns absolute path string or `None`; used by `numberify_citations()` to build PDF links
- [x] **`attribute_sources()` integration** (`src/agents/nodes.py`):
  - After extracting `answer` from `search_queries[0].summary`, calls `numberify_citations(answer, language)`
  - Result stored in `final_report["answer"]`
- [x] **`pdf_route.py`** (`src/ui/components/pdf_route.py`) — new file:
  - `ensure_pdf_route()` (`@st.cache_resource`): one-time Tornado route injection, same gc-based discovery as `gpu_widget.py`
  - `PDFHandler.get()`: validates path is within `kb/` (prevents traversal), serves PDF with `Content-Type: application/pdf`
  - Double-injection guard checks `default_router.rules`
- [x] **`render_results_view()`** (`src/ui/components/results_view.py`): calls `ensure_pdf_route()` on entry
- [x] **Prompt updates** (`src/prompts/synthesis.py`, `src/prompts/synthesis_gpt.py`): citation rules now say "using the EXACT filename from task_summaries/research_findings" to reduce hallucinated filenames
- [x] **Tests** (`tests/test_citations.py`): 14 tests covering `resolve_pdf_path`, `numberify_citations`, PDF handler security, and `attribute_sources` integration

**295 tests total, all passing** (2 ChromaDB GPU-OOM failures unrelated to these changes).

### Phase 6.14: Depth Selector — gemma4:e4b as Default

- [x] **`src/ui/app.py`**: `DEPTH_OPTIONS` updated — `"basic (gemma4:e4b)"` added at index 0 (first/default); old `batiai/gemma4-26b:q3` entry removed
- [x] **`src/config.py`**: `ollama_model` default changed to `"gemma4:e4b"`; `model_family` uses case-insensitive match (`"gemma4"` or `"gemma-4"` in lowercased model string)
- [x] **`src/services/ollama_client.py`**: `is_gemma4` property likewise uses case-insensitive check
- [x] **`src/ui/state.py`**: `research_depth` default set to `"basic (gemma4:e4b)"`
- [x] **Tests**: `TestGemma4E4BSupport` class (4 tests) in `tests/test_prompt_routing.py`

### Phase 6.15b: Depth Selector — einfach swapped to granite4.1:3b

- [x] **`src/ui/app.py`**: `"einfach (gemma4:e2b)"` → `"einfach (granite4.1:3b)"` in `DEPTH_OPTIONS` and `_DEPTH_TO_MODEL`
- [x] **`src/config.py`**: `ollama_fallback_model` default changed from `"gemma4:e2b"` to `"granite4.1:3b"`
- [x] **`src/services/ollama_client.py`**: added `is_granite4` property (case-insensitive `"granite4"` substring check); `_prepare_system_prompt()` now strips `/no_think` for both `is_gemma4` and `is_granite4` (Qwen3-specific directive not supported by Granite)
- `granite4.1` `model_family` resolves to `"qwen"` (falls through gpt-oss/gemma4 checks) → uses Qwen prompt set; Granite chat template (`<|start_of_role|>...<|end_of_role|>`) is handled by Ollama server-side, no code changes needed

### Phase 6.15c: Depth Selector — specialized (north-mini-code-1.0) added

- [x] **`src/ui/app.py`**: added `"specialized (north-mini-code-1.0)"` as the 6th entry in `DEPTH_OPTIONS` and `_DEPTH_TO_MODEL` (mapped to model `north-mini-code-1.0`)
- `north-mini-code-1.0` `model_family` resolves to `"qwen"` (no `gpt-oss`/`gemma4` match) → uses Qwen prompt set; no prompt-routing, config-default, or session-default changes needed (`research_depth` default stays `"basic (gemma4:e4b)"`)
- Model must be available in Ollama (`ollama pull` / `ollama list`) for the selection to run

### Phase 3.14: Resilient Task Summaries (Graceful Degradation)

Fixes inconsistent per-task output (most tasks rendered as `"Completed task: <task>"` with no
findings) on small/code-specialized models. Root cause: `_generate_task_summary()` had a single
strict-schema attempt — any `TaskSummaryOutput` parse failure (schema mismatch like
`relevance_score: "60%"`, missing required field, unrepairable truncation) fell straight to the
useless placeholder. Model-agnostic fix; Tier 1 unchanged for already-working models.

- [x] **`src/models/results.py`**: new `TaskSummarySimple` (4 fields: `summary` required, `key_findings`, `gaps`, `relevance_score`) with a `field_validator(mode="before")` coercing loose `relevance_score` inputs (`"60%"`, `"60"`, `60.0` → 60; junk → 50; clamped 0-100)
- [x] **`src/prompts/research.py`** + **`research_gpt.py`**: new `TASK_SUMMARY_SIMPLE_PROMPT_{SYSTEM,HUMAN}` (Qwen markdown + gpt-oss `<json>`-tag variants), auto-collected by PEP 562 routing — name parity verified (54 constants per set)
- [x] **`src/agents/nodes.py`**: `_generate_task_summary()` wraps Tier 1 in try/except; delegates failures to new `_generate_task_summary_degraded()` helper:
  - Tier 2: retry `generate_structured_messages(..., TaskSummarySimple)`, map → full dict
  - Tier 3: `generate_messages()` plain-text prose summary (no JSON)
  - Last resort: keyword-overlap relevance + `"Completed task:"` placeholder only if all 3 LLM calls raise
- [x] **Tests** (`tests/test_agents.py`): updated total-failure test (now requires all tiers to fail); added Tier 2 degradation, Tier 3 prose fallback, and loose-`relevance_score` coercion tests

**344 tests total, all passing.**

### Phase 9: Remote Access via Cloudflare Tunnel

- [x] **`login/launcher_app.py`** (new): Password-gated Streamlit control panel on port 8522
  - `LAUNCHER_PASSWORD` env var (default: `changeme123`)
  - Start/stop/restart controls for the main app (port 8511)
  - Process monitoring via `psutil` (CPU, memory metrics)
  - Tunnel URL display (reads from `/tmp/hybrid-*-url.txt`)
  - Log viewer (`/tmp/hybrid_researcher_app.log`, last 50 lines)
  - Safe exit guard: refuses to kill ports < 1024 or the launcher port
- [x] **`login/start-launcher.sh`** (new): Starts launcher via `uv run streamlit run`
  - Password prompt if `LAUNCHER_PASSWORD` not set
  - Port 8522 availability check with optional kill
- [x] **`login/start-quick-tunnels.sh`** (new): Creates two Cloudflare quick tunnels
  - Targeted `pkill` by port URL (preserves `brain-nw1` and other tunnels)
  - Backup/restore `~/.cloudflared/config.yml` to force quick tunnel mode
  - Extracts `*.trycloudflare.com` URLs from logs, saves to `/tmp/hybrid-*-url.txt`
  - Separate log files: `/tmp/hybrid-launcher-tunnel.log`, `/tmp/hybrid-app-tunnel.log`
- [x] **`login/cloudflared-config.yml`** (new): Template for future persistent tunnel setup (requires Cloudflare-managed domain)
- [x] **`login/README.md`** (new): Quick start, port assignments, coexistence with `brain-nw1`, upgrade path to permanent URLs
- [x] **`pyproject.toml`**: Added `psutil>=5.9.0` dependency

**Port assignments:** Launcher 8522, Main app 8511. Quick tunnel URLs are temporary (`*.trycloudflare.com`).

### Phase 3.10: Tavily Web Search Integration (Optional)

Adds an optional web search step between `rerank_task_summaries` and `synthesize`. KB results and web results are **strictly separated** — web summary is appended in `attribute_sources()`, never passed into `synthesize()`.

- [x] **`src/services/tavily_client.py`** (new): Tavily REST API client
  - `tavily_search(query, max_results)`: POST to `https://api.tavily.com/search`, returns raw dicts; `[]` on failure
  - `format_tavily_results(results, query)`: converts raw dicts to `WebResult` instances
  - `format_results_for_prompt(web_results)`: numbered text blocks for LLM prompt
- [x] **`src/models/results.py`**: Added `WebSearchSummaryOutput` model (`web_summary`, `contradictions`); extended `FinalReport` with `web_search_section` and `web_sources`
- [x] **`src/agents/state.py`**: Added `enable_web_search`, `web_search_results`, `web_search_summary` to `AgentState`; initialized in `create_initial_state()`
- [x] **`src/prompts/synthesis.py`**: 4 new constants — `WEB_SEARCH_QUERY_PROMPT_{SYSTEM,HUMAN}` (generates search term from gaps) + `WEB_SEARCH_SUMMARIZE_PROMPT_{SYSTEM,HUMAN}` (summarizes with `[Title](URL)` citations + contradiction detection)
- [x] **`src/prompts/synthesis_gpt.py`**: Same 4 constants in Harmony format
- [x] **`src/agents/nodes.py`**:
  - `web_search(state)` node: guard on `enable_web_search`, LLM query generation, Tavily API call, LLM summarization with `WebSearchSummaryOutput`, contradiction notice prepending
  - `attribute_sources()` modified: appends web section with language-appropriate header (`### Ergänzende Webrecherche` / `### Supplementary Web Research`), populates `FinalReport.web_search_section` and `web_sources`
- [x] **`src/agents/graph.py`**: `web_search` node added; `route_after_rerank()` routes to `web_search` if enabled, else `synthesize`; `route_after_web_search()` always → `synthesize`
- [x] **`src/ui/app.py`**: Checkbox `disabled=is_researching` (was `disabled=True`); `initial_state["enable_web_search"]` set in both research paths; phase/subtask labels added
- [x] **`src/ui/components/results_view.py`**: Web sources rendering + markdown export inclusion
- [x] **`tests/test_web_search.py`** (new): 23 tests covering models, Tavily client, graph routing, web_search node, attribute_sources integration, state defaults

**315 tests total, all passing.**

### Phase 6.15: Wissensdatenbank Document Browser + Model Sync Fix

Two sidebar improvements in `src/ui/app.py` and `src/services/chromadb_client.py`.

**Document Browser:**
- [x] **`ChromaDBClient.get_document_names(db_name)`** (`src/services/chromadb_client.py`): metadata-only query via raw `chromadb.PersistentClient` (no embeddings); calls `collection.get(include=["metadatas"])`, extracts `original_filename` / `source` / `filename`, deduplicates, returns sorted `list[str]`.
- [x] **`@st.dialog` modal** (`src/ui/app.py`): `_show_documents_dialog(db_name, doc_names)` — native Streamlit modal showing DB name, document count, and one filename per line; closes via ✕ without affecting app state.
- [x] **"Dokumente anzeigen" button** added to "Wissensdatenbank" expander after the embedding caption; only visible when a specific database is selected.

**Model Sync Fix (GPU widget showed stale `.env` model):**
- [x] **Root cause**: `_apply_research_depth()` was only called when `depth != session.research_depth`, so on first page load (where both equal the session default `"basic (gemma4:e4b)"`), `settings.ollama_model` was never synced from the UI default — it kept the `.env` value (e.g., `qwen3:14b`), which the GPU widget displayed.
- [x] **Fix**: `_apply_research_depth(depth)` is now called unconditionally after the selectbox on every render. Its guard (`if model_name == settings.ollama_model: return`) makes it a no-op when already in sync.

### Phase 8: Testing Improvements
- [x] `TestRouteEntryPoint` class for graph routing logic
  - `test_route_to_hitl_init_on_new_session`
  - `test_route_to_hitl_process_response_on_resume`
  - `test_route_to_generate_todo_with_research_queries`
  - `test_route_to_generate_todo_with_phase`
  - `test_decision_without_hitl_active_routes_to_process_hitl_todo`

### Phase 10: GUI Login Gate

Role-free login screen in front of the Streamlit app, inspired by the sibling project `KB_BS_local-wiki-he` (stripped of its roles / per-DB access / admin UI).

- [x] **`src/ui/auth.py`** (new): self-contained auth layer + login UI
  - `USERS_JSON_PATH` → gitignored `data/users.json`; `_DEFAULT_USERS` = `{"T. Hein": "#BrAIn1", "Gast": "2026_BrAIn"}` (seed-only plaintext, hashed immediately)
  - `_hash()` uses stdlib `hashlib.pbkdf2_hmac("sha256", …, 200_000)` with a per-user random salt — no new dependency (reference repo used `bcrypt`)
  - `ensure_seeded()` (idempotent — no-op if file exists), `verify()` (`hmac.compare_digest`), `is_authenticated()`, `current_user()`, `logout()`, `render_login()` (German centered `st.form`)
- [x] **`src/ui/app.py`**: gate in `main()` immediately after `st.set_page_config()` (`ensure_seeded()` + `if not auth.is_authenticated(): render_login(); return`); sidebar "Angemeldet als …" caption + "Abmelden" button in `render_sidebar()`
- [x] **Auth state isolation**: stored in top-level `st.session_state["auth_user"]`, NOT on the `SessionState` dataclass, so `reset_session_state()` (safe-exit / new research) does not log the user out
- [x] **`.gitignore`**: `data/users.json` (seeded password hashes never committed)
- [x] **Tests** (`tests/test_auth.py`): 7 tests — seeding (+ idempotency), correct/wrong/unknown credentials, hash-not-plaintext storage, distinct salts

