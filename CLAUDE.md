# Rabbithole-Agent: Local Hybrid Researcher

A fully local, privacy-first research system that performs **deep reference-following** across document collections using Ollama LLMs, ChromaDB, and LangGraph.

## Core Problem

Classical RAG lacks deep contextual understanding and cannot follow inter-document relationships. This agent solves it by iteratively "digging into the rabbithole" - following references, building context, and discovering document interconnections.

## Architecture (5 Phases + Graded Context)

```
┌────────────────────────────────────────────────────────────────────┐
│  Phase 1: Enhanced Query Analysis + Iterative HITL                  │
│  User Query → Language Detection → Iterative Clarification Loop     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  hitl_init → hitl_generate_questions ↔ hitl_process_response │  │
│  │  → hitl_finalize (on /end, max_iterations, or convergence)   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: research_queries[], query_anchor, hitl_smry     │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2: Research Planning                                         │
│  QueryAnalysis → ToDoList (3-5 tasks, max 15)                       │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3: Deep Context Extraction (with Graded Classification)      │
│  For each task:                                                      │
│    LLM Multi-Query (3) → Vector Search → Extract Info + Quotes →    │
│    Classify Tier →                                                   │
│    Hybrid Ref Detection → **Agentic Ref Gate** →                    │
│    Registry-Scoped Resolution →                                      │
│    Token Budget → Convergence Check → Generate Task Summary →       │
│    Accumulate by Tier (primary/secondary/tertiary) → Next Task      │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3.5: Pre-Synthesis Relevance Validation (NEW)                │
│  validate_relevance: Filter drift against query_anchor              │
├────────────────────────────────────────────────────────────────────┤
│  Phase 4: Deep Report Synthesis + Quality Assurance                   │
│  Pre-Digested Task Summaries + HITL Summary → Deep Report            │
│  Language Enforcement → Quality Check → **Agentic Remediation** →   │
│  Re-Synthesis (max 1 retry) OR Accept → Report                      │
├────────────────────────────────────────────────────────────────────┤
│  Phase 5: Source Attribution                                        │
│  Add citations → Resolve paths → Generate clickable links           │
└────────────────────────────────────────────────────────────────────┘
```

### Graded Context Management (NEW)

The system now uses **tiered context classification** to prevent query drift and ensure synthesis quality:

```
┌──────────────────────────────────────────────────────────────────┐
│  TIER 1: Primary Context (weight 1.0)                            │
│  ├─ Direct vector search results for current task                │
│  ├─ Highest relevance score chunks (≥0.85)                       │
│  └─ Explicitly matches key entities from query_anchor            │
├──────────────────────────────────────────────────────────────────┤
│  TIER 2: Secondary Context (weight 0.7)                          │
│  ├─ Rabbithole depth-1 references (direct citations)             │
│  └─ Medium relevance score chunks (0.6-0.85)                     │
├──────────────────────────────────────────────────────────────────┤
│  TIER 3: Tertiary Context (weight 0.4)                           │
│  ├─ Rabbithole depth-2 references                                │
│  └─ HITL retrieval chunks (query_retrieval)                      │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Query Anchor**: Immutable reference to original intent created in `hitl_finalize`
- **Preserved Quotes**: Verbatim extraction of legal/technical language
- **Task Summaries**: Per-task structured summaries with relevance scoring
- **Drift Detection**: Pre-synthesis filtering warns when >30% of context is irrelevant
- **Language Enforcement**: Strict single-language output with retry on mismatch

### Agentic Decision Points (NEW)

Two LLM-driven decision points where the orchestrator is no longer deterministic:

1. **Reference Following Gate** (Phase 3, `execute_task`):
   - Before following each detected reference, LLM evaluates: "Is this reference worth following given the query?"
   - Uses `REFERENCE_DECISION_PROMPT` → `ReferenceDecision(follow: bool, reason: str)`
   - Gate receives full context: `original_query`, `key_entities`, `scope`, `current_task`
   - Bias toward following: "when uncertain, FOLLOW" (skipping relevant refs is costlier)
   - Prevents tangential references from wasting token budget and diluting context
   - Falls back to following on LLM error (safe default)

2. **Quality Remediation Loop** (Phase 4, `quality_check`):
   - If quality score < threshold (375), LLM decides: accept as-is or retry synthesis with focused instructions
   - Uses `QUALITY_REMEDIATION_PROMPT` → `QualityRemediationDecision(action: "accept"|"retry", focus_instructions: str)`
   - Max 1 retry to prevent infinite loops (tracked via `synthesis_retry_count`)
   - On retry, `quality_remediation_focus` is appended to the synthesis prompt
   - `route_after_quality` routes to `synthesize` (retry) or `attribute_sources` (accept)

**Agentic State Fields:**
- `synthesis_retry_count`: int (default 0, max 1)
- `quality_remediation_focus`: str (cleared after use)

### Enhanced Phase 1: Iterative HITL with Multi-Vector Retrieval

The enhanced iterative HITL system provides intelligent query refinement through conversation **with integrated vector DB retrieval at each iteration**:

```
┌──────────────────────────────────────────────────────────────────┐
│  hitl_init → hitl_generate_queries → hitl_retrieve_chunks →      │
│  hitl_analyze_retrieval → hitl_generate_questions → [wait] →     │
│  hitl_process_response → [loop back or hitl_finalize]           │
└──────────────────────────────────────────────────────────────────┘
```

**Node Descriptions:**

1. **hitl_init**: Initialize conversation, detect language (de/en)
2. **hitl_generate_queries** (NEW): Generate 3 search queries per iteration
   - Iteration 0: original + broader_scope + alternative_angle
   - Iteration N>0: refined based on user feedback + knowledge gaps
3. **hitl_retrieve_chunks** (NEW): Execute vector search with deduplication
   - 3 chunks per query (~9 total per iteration)
   - Deduplicates against accumulated `query_retrieval`
4. **hitl_analyze_retrieval** (NEW): LLM analysis of retrieval context
   - Extracts: key_concepts, entities, scope, knowledge_gaps, coverage_score
5. **hitl_generate_questions**: Generate 2-3 contextual follow-up questions
   - Now informed by retrieval analysis and identified gaps
   - **Uses `query_retrieval` from state** to provide retrieval context to LLM
6. **hitl_process_response**: Analyze user response, check termination conditions
7. **hitl_finalize**: Generate research_queries and hand off to Phase 2

**Termination Conditions**:
- User types `/end` → `user_end`
- Max iterations reached (default: 5) → `max_iterations`
- **Convergence** (coverage ≥ 0.8 AND dedup_ratio ≥ 0.7 AND gaps ≤ 2) → `convergence`

**State Tracking**:
- `hitl_iteration`: Current iteration count (0-indexed)
- `coverage_score`: 0-1 estimate of information coverage
- `iteration_queries`: List of query triples per iteration
- `knowledge_gaps`: Identified gaps from retrieval analysis
- `retrieval_dedup_ratios`: Dedup ratio per iteration for convergence detection
- `hitl_conversation_history`: Full conversation for context
- `query_retrieval`: Accumulated retrieval text (converted to tertiary_context in finalize)

**Graded Context State Fields** (NEW):
- `query_anchor`: Immutable reference to original intent (created in hitl_finalize)
- `hitl_smry`: Synthesized HITL findings for final synthesis
- `primary_context`: Tier 1 high-confidence findings (list of dicts)
- `secondary_context`: Tier 2 supporting findings (list of dicts)
- `tertiary_context`: Tier 3 background context (list of dicts)
- `task_summaries`: Per-task structured summaries with relevance scores
- `preserved_quotes`: Critical verbatim quotes for legal/technical precision

## Tech Stack (LangChain v1.0+)

| Component | Technology |
|-----------|------------|
| Framework | LangChain v1.0+, LangGraph v1.0+ |
| LLM | Ollama (qwen3:14b, qwen3:8b fallback) |
| Embeddings | Qwen/Qwen3-Embedding-0.6B via HuggingFace |
| Vector DB | ChromaDB (local persistent) |
| Orchestration | LangGraph StateGraph (TypedDict state) |
| Structured Output | `llm.with_structured_output(Model, method="json_mode")` |
| PDF Processing | PyMuPDF |
| UI | Streamlit (port >8510) |
| Python | >=3.10 (v1.0 requirement) |

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # Edit .env if needed

# Pull required Ollama models (for LLM generation)
ollama pull qwen3:14b           # Primary model (14B)
ollama pull qwen3:8b            # Fallback model
# Note: Embeddings use Qwen/Qwen3-Embedding-0.6B via HuggingFace
# (downloaded automatically on first run, requires GPU)

# Run Streamlit UI
uv run streamlit run src/ui/app.py --server.port 8511 --server.headless false

# Or run via CLI
python -m src.main --ui --port 8511

# Run single query (non-interactive)
python -m src.main --query "Was sind die Grenzwerte für Strahlenexposition?"

# Run tests
pytest tests/ -v
```

## Key Configuration

Edit `.env` for your setup:
- `OLLAMA_NUM_CTX=131072`: 128K context for dual 4090s (adjust if needed)
- `OLLAMA_SAFE_LIMIT=0.9`: Stop at 90% to prevent OOM
- `QUALITY_THRESHOLD=375`: Minimum quality score (0-500, 5 dimensions)
- `REFERENCE_EXTRACTION_METHOD=hybrid`: Reference detection method (`regex`, `llm`, `hybrid`)
- `REFERENCE_TOKEN_BUDGET=50000`: Max tokens for reference following per task
- `CONVERGENCE_SAME_DOC_THRESHOLD=3`: Stop following when same doc appears N times

## Directory Structure

```
KB_BS_local-hybrid-researcher/
├── CLAUDE.md              # This file
├── docs/                  # Detailed documentation
│   ├── architecture.md    # Full system design
│   ├── agent-design.md    # ReAct + LangGraph patterns
│   ├── data-models.md     # Pydantic schemas
│   ├── data-sources.md    # PDF corpus + ChromaDB
│   ├── configuration.md   # .env + pyproject.toml
│   ├── implementation.md  # Phases + coding standards
│   ├── rabbithole-magic.md # Deep reference-following algorithm
│   └── references.md      # External resources
├── src/                   # Source code
│   ├── agents/            # LangGraph agents + tools
│   ├── models/            # Pydantic data models
│   ├── services/          # ChromaDB, Ollama, PDF
│   └── ui/                # Streamlit app
├── tests/                 # Pytest tests
└── kb/                    # Knowledge base (pre-existing)
    ├── database/          # ChromaDB collections
    ├── document_registry.json   # Document-to-synonym mapping for scoped search
    ├── GLageKon__db_inserted/   # Source PDFs for GLageKon
    ├── NORM__db_inserted/       # Source PDFs for NORM
    ├── StrlSch__db_inserted/    # Source PDFs for StrlSch
    └── StrlSchExt__db_inserted/ # Source PDFs for StrlSchExt
```

## MUST-HAVE Requirements

1. **Human-In-The-Loop**: User validation at query refinement and task approval
2. **ToDoList Tracking**: Visible task progress with dynamic updates
3. **Structured JSON Outputs**: All LLM responses via Pydantic + `json_mode`
4. **Fully Local**: Ollama-only, no external API calls
5. **Safe Exit**: Streamlit button to cleanly terminate (port-aware)
6. **Reference Following**: Deep rabbithole traversal with hybrid detection (regex+LLM), document registry scoping, and relevance filtering

## Coding Standards

### Prompt Management
- **All LLM prompts MUST be defined in `src/prompts.py`**
- Never inline prompt strings in node functions or services
- Use template variables for dynamic content (e.g., `{query}`, `{context}`)
- Group prompts by category (HITL, Research, Quality)
- **Every content-bearing prompt MUST include `{language}`** to enforce output language
  - Only exceptions: `LANGUAGE_DETECTION_PROMPT` (outputs code) and `REFERENCE_EXTRACTION_PROMPT` (copies verbatim)
- All prompts follow a strict 4-section format: `### Task`, `### Input`, `### Rules`, `### Output format`

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Full architecture diagram, state objects, data flow |
| [docs/agent-design.md](docs/agent-design.md) | ReAct+LangGraph patterns, tools |
| [docs/data-models.md](docs/data-models.md) | All Pydantic models with JSON schemas |
| [docs/data-sources.md](docs/data-sources.md) | PDF corpus, ChromaDB collections, embeddings |
| [docs/configuration.md](docs/configuration.md) | Environment variables, pyproject.toml |
| [docs/implementation.md](docs/implementation.md) | Implementation phases, coding standards |
| [docs/rabbithole-magic.md](docs/rabbithole-magic.md) | Deep reference-following algorithm |
| [docs/references.md](docs/references.md) | External repos, LangGraph docs, examples |

## Implementation Status

### Baseline Complete (Week 1)
- [x] Core infrastructure (config, models, services)
- [x] Pydantic models for all data structures
- [x] ChromaDB multi-collection search
- [x] Ollama client with structured output + retry
- [x] HITL service (clarification + approval)
- [x] LangGraph StateGraph with 5 phases
- [x] Reference detection and following (depth=2)
- [x] Relevance filtering (threshold=0.6)
- [x] Streamlit UI with HITL panels
- [x] Safe exit button
- [x] Basic tests

### Enhanced Phase 1 (Week 2) - COMPLETE
- [x] Iterative HITL with conversation loop
- [x] Enhanced AgentState with retrieval tracking fields (`iteration_queries`, `knowledge_gaps`, etc.)
- [x] Multi-vector retrieval nodes (`hitl_generate_queries`, `hitl_retrieve_chunks`, `hitl_analyze_retrieval`)
- [x] Convergence detection (coverage ≥ 0.8, dedup ≥ 0.7, gaps ≤ 2)
- [x] Interactive query refinement powered by real-time retrieval context
- [x] UI support for iterative retrieval statistics and coverage metrics
- [x] Full graph integration with retrieval loop back edges
- [x] Unit tests for enhanced state and graph flow (17/17 passed)

### UI Enhancements (Week 2.5) - COMPLETE
- [x] **Retrieval History Panel**: Real-time display of vector search results during HITL
  - Shows queries, chunk counts, dedup ratios per iteration
  - Nested expanders with chunk details (doc, page, score, text preview)
- [x] **Database Selection Fix**: User's collection choice now respected in HITL retrieval
  - `_perform_hitl_retrieval()` uses `search_by_database_name()` when DB selected
- [x] **Active Database Indicator**: Sidebar shows currently selected database
- [x] **Cached Service Clients**: `@st.cache_resource` for ChromaDB/Ollama clients (faster reloads)
- [x] **Graph Entry Point Enhancement**: 4-way router (`hitl_init`, `hitl_process_response`, `generate_todo`, `process_hitl_todo`)
  - Todo-approval resume: `hitl_decision + !hitl_active` -> `process_hitl_todo`
  - Iterative HITL resume: `hitl_decision + hitl_active` -> `hitl_process_response`
  - `_start_research_from_hitl()` sets `hitl_active=False` to prevent misrouting
- [x] **Coverage Metrics in Checkpoints**: Knowledge gaps and dedup ratios shown in UI
- [x] **German Localization**: Todo approval panel fully translated (buttons, labels, messages)
- [x] **Layout Fix**: HITL summary/checkpoints moved inside column layout for consistent rendering
- [x] **Todo Side Panel Expanders**: Task list uses `st.expander` per task instead of truncated text
  - Header: icon + sequential number + truncated label (40 chars)
  - Body: full untruncated task text via `st.markdown`
  - Currently executing task auto-expanded (`expanded=is_current`)
  - Unicode emoji chars (shortcodes don't render in expander labels)
- [x] **Verbose Task Spinner**: `execute_tasks` phase shows `Aufgabe {n}/{total}: {task_text}` in spinner
- [x] **Simplified Graph Streaming**: Removed inline column layout from `_run_graph_stream()` (UI renders via rerun)
- [x] **Sequential Task ID Renumbering**: `process_hitl_todo` renumbers task IDs after user modifications

### Enhanced Reference Following (Week 3) - COMPLETE
- [x] **Hybrid Reference Detection**: Configurable `regex`, `llm`, or `hybrid` extraction method
  - Regex: 7 hardcoded patterns (German/English sections, documents, URLs)
  - LLM: `REFERENCE_EXTRACTION_PROMPT` with few-shot examples via `generate_structured()`
  - Hybrid: runs both, deduplicates by `type:target` key + substring overlap
- [x] **Document Registry** (`kb/document_registry.json`): Maps PDFs to synonyms across 4 collections
  - `load_document_registry()`: singleton loader
  - `resolve_document_name()`: 3-stage matching (exact synonym > fuzzy 0.7 > substring)
- [x] **Scoped Passage Retrieval**: `_vector_search_scoped()` searches within resolved document's collection
  - Post-filters by `doc_name` matching the resolved filename
- [x] **Enhanced Resolution**: `resolve_reference_enhanced()` routes by ref type
  - `legal_section`/`section`: registry resolve -> scoped search (fallback: broad)
  - `document`/`document_mention`: registry lookup -> scoped search (fallback: broad)
  - `academic_numbered`/`academic_shortform`: broad vector search
- [x] **Token Budget Tracking**: `reference_token_budget` (default 50K) stops reference following when exhausted
- [x] **Convergence Detection**: `detect_convergence()` stops when same document appears >= threshold times
- [x] **New Pydantic Models**: `ExtractedReference`, `ExtractedReferenceList` for LLM structured output
- [x] **Extended `DetectedReference`**: Added `document_context`, `extraction_method` fields
- [x] **39 Unit Tests**: `tests/test_reference_extraction.py` (registry, regex, LLM mock, hybrid, resolution, convergence)

### Graded Context Management (Week 4) - COMPLETE
- [x] **Query Anchor & HITL Context Preservation** (Phase A)
  - `query_anchor`: Immutable reference to original intent
  - `hitl_smry`: LLM-synthesized HITL conversation for synthesis
  - `HITL_SUMMARY_PROMPT` in `src/prompts.py`
- [x] **Strict Language Enforcement** (Phase F)
  - `generate_structured_with_language()` in OllamaClient
  - Language validation with German/English marker heuristics
  - Automatic retry with stronger language instruction on mismatch
- [x] **Graded Context Classification** (Phase B)
  - `classify_context_tier()`: Assigns Tier 1/2/3 based on source, depth, relevance
  - `create_tiered_context_entry()`: Creates weighted context dicts with optional `task_id` for per-task UI filtering
  - Chunks accumulated into `primary_context`, `secondary_context`, `tertiary_context`
- [x] **Verbatim Quote Preservation** (Phase C)
  - `PreservedQuote`, `InfoExtractionWithQuotes` models in `src/models/research.py`
  - `extract_info_with_quotes()`: Returns condensed info + critical verbatim quotes
  - `INFO_EXTRACTION_WITH_QUOTES_PROMPT` for legal/technical precision
- [x] **Per-Task Structured Summary** (Phase D)
  - `_generate_task_summary()`: Receives per-task tiered findings (primary/secondary/tertiary) + preserved quotes
  - `TASK_SUMMARY_PROMPT` receives `{primary_findings}`, `{secondary_findings}`, `{tertiary_findings}`, `{preserved_quotes}`
  - Tier priority rule: primary > secondary > tertiary (conflicts noted in gaps)
  - `task_summaries` state field accumulated during task execution
- [x] **Pre-Synthesis Relevance Validation** (Phase G)
  - `validate_relevance()` node: Filters drift before synthesis
  - `_score_and_filter_context()`: Scores against query_anchor entities
  - Logs warning when >30% of accumulated context is filtered
- [x] **Deep Report Synthesis** (Phase E)
  - `SYNTHESIS_PROMPT_ENHANCED`: Expert report writer producing extensive markdown-formatted deep reports
  - No sentence cap — structured sections, exact figures, verbatim quotes, section references
  - Receives only `{original_query}`, `{hitl_smry}`, `{task_summaries}`, `{language}`
  - `_format_task_summaries()` enriches summaries with key_findings, gaps, and preserved quotes
  - Falls back to legacy `SYNTHESIS_PROMPT` (also deep-report style) if no graded context available
  - UI heading: `:microscope: Detailbericht` (Key Findings section removed from results view)
- [x] **New Pydantic Models**: `SynthesisOutputEnhanced`, `TaskSummaryOutput`, `RelevanceScoreOutput`
- [x] **Graph Update**: Added `validate_relevance` node between `execute_task` and `synthesize`
- [x] **84+ Unit Tests**: All existing tests pass (22 model, 23 agent, 39 reference extraction)

### HITL-Aware Summaries & Prompt Annotations (Week 4.7) - COMPLETE
- [x] **Rename `hitl_context_summary` → `hitl_smry`**: Shorter name, consistent across state/prompts/nodes
  - `HITL_CONTEXT_SUMMARY_PROMPT` → `HITL_SUMMARY_PROMPT` with citation-aware instructions
  - `_summarize_hitl_context()` → `_generate_hitl_summary()` with `[Source_filename]` annotations
  - Retrieval truncation raised from 4K to 8K chars to preserve `[doc, p.N]` prefixes
- [x] **`hitl_smry` fed into `generate_todo()`**: Both research_queries path (as `item_context`) and LLM fallback path
- [x] **`hitl_smry` fed into `_generate_task_summary()`**: New `{hitl_smry}` variable in `TASK_SUMMARY_PROMPT`
  - Rule: "Do not repeat findings already covered in hitl_findings"
  - Fallback: `"No prior findings"` when `hitl_smry` is empty
- [x] **Comprehensive Prompt Annotations**: Every prompt in `src/prompts.py` now has a docstring header
  - Phase, graph node, caller, workflow position, input/output, consumption
  - Several prompts gained `### Role` section for small-model clarity
- [x] **Debug State Dumps**: `enable_state_dump` config + `src/utils/debug_state.py`
  - `dump_state_markdown()` writes flat key→value phase snapshots to `tests/debugging/`
  - Trigger points at correct phase boundaries: `generate_todo()` (state_1hitl), `execute_task()` task_id==0 (state_2todo), `validate_relevance()` (state_3rabbithole)
- [x] **116 Unit Tests**: All pass (22 model, 35 agent, 39 reference, 20 task search + other)

### Prompt Standardization & Multi-Query Task Execution (Week 4.5) - COMPLETE
- [x] **Prompt 4-Section Format**: All 19 prompts reformatted to `### Task / ### Input / ### Rules / ### Output format`
  - Merged `FOLLOW_UP_QUESTIONS_DE` + `FOLLOW_UP_QUESTIONS_EN` into single `FOLLOW_UP_QUESTIONS_PROMPT` with `{language}`
- [x] **Universal `{language}` Enforcement**: All 17 content-bearing prompts include `{language}` template variable
  - Callers in `hitl_service.py` (5 functions) and `nodes.py` (5 functions) compute `lang_label` and pass it
  - Only exceptions: `LANGUAGE_DETECTION_PROMPT` (outputs code), `REFERENCE_EXTRACTION_PROMPT` (copies verbatim)
- [x] **Multi-Query Task Execution**: `execute_task()` generates 3 queries per task
  - New `TASK_SEARCH_QUERIES_PROMPT` generates 2 LLM-targeted queries + 1 base concatenation query
  - New `TaskSearchQueries` Pydantic model in `src/models/query.py`
  - Chunk deduplication by `doc_name:page:text[:100]` across all 3 query results
- [x] **Task 0 Prepend**: `generate_todo()` prepends original query as Task 0 for direct vector search
- [x] **5-Dimension Quality Scoring**: Added `query_relevance` (0-100) to `QualityAssessment`
  - Total score now 0-500 (was 0-400), threshold raised to 375
  - UI and CLI updated to display `/500`
- [x] **Enhanced `TaskSummaryOutput`**: Added `relevance_assessment` and `irrelevant_findings` fields
- [x] **Relevance Filter Backfill**: `filter_by_relevance()` accepts `min_results` param
  - Guarantees minimum chunk count by backfilling from top-scoring rejected chunks
- [x] **Language-Aware Extraction**: `extract_info()`, `extract_info_with_quotes()`, `create_chunk_with_info()` accept `language` parameter
- [x] **Config Tuning**: `quality_threshold` 300→375, `k_results` 5→3
- [x] **106 Unit Tests**: All pass (22 model, 23 agent, 39 reference, 22 task search + other)

### Persistent Results View (Week 4.8) - COMPLETE
- [x] **HITL Expander in Results**: `_render_hitl_expander()` in `results_view.py`
  - Conversation history rendered as `st.chat_message` bubbles
  - `hitl_smry` (LLM-synthesized HITL summary with `[Source]` annotations)
  - Numbered research queries from `hitl_result`
  - Guarded: skips when all three data sources are empty
- [x] **Per-Task Expanders in Results**: `_render_task_expanders()` in `results_view.py`
  - Task summary, key findings (bullets), gaps (bullets), relevance assessment
  - Per-chunk expanders with full original vector DB text + LLM extraction (via shared `task_rendering.py`)
  - Matches `task_summaries` by `task_id` lookup, chunks by index into `search_queries`
  - Guarded: skips when `todo_list` is empty
- [x] **Shared Task Rendering Helpers**: `src/ui/components/task_rendering.py`
  - `render_task_summary_markdown()`: Formatted summary with findings, gaps, relevance
  - `render_chunk_expander()`: Per-chunk expander with full extraction + original text
  - `render_tiered_chunks()`: Renders chunks grouped by tier as nested expanders (Tier 1 expanded, others collapsed)
  - `filter_tiered_context_by_task()`: Filters tiered context lists by `task_id`
  - `has_task_id_entries()`: Backward compat check for old states without `task_id`
  - Used by both live view (`app.py`) and persistent results view (`results_view.py`)
- [x] **Debug State Dump Enhancement**: `dump_state_markdown()` outputs full values (removed 500-char truncation)
  - Phase 1→2 boundary dump in `generate_todo()`
- [x] **124 Unit Tests**: All pass

### Tiered Chunk Rendering in Task Expanders (Week 4.9) - COMPLETE
- [x] **`task_id` in Tiered Context Entries**: `create_tiered_context_entry()` accepts optional `task_id` param
  - `execute_task()` passes `task_id` for both direct chunks and nested reference chunks
  - Backward compatible: `task_id=None` default, old callers unaffected
- [x] **Tiered Rendering Helpers** in `src/ui/components/task_rendering.py`:
  - `render_tiered_chunks(primary, secondary, tertiary)`: Nested tier expanders (German labels), Tier 1 expanded by default, empty tiers hidden
  - `filter_tiered_context_by_task(primary_ctx, secondary_ctx, tertiary_ctx, task_id)`: Per-task filtering
  - `has_task_id_entries(context_lists)`: Backward compat detection for old states without `task_id`
- [x] **Persistent Results View**: `_render_task_expanders()` uses tiered rendering when `task_id` entries exist, falls back to flat chunk rendering for old states
- [x] **Live View**: `_render_task_result_expander()` accepts `tiered_context` param; `_run_graph_with_live_updates()` extracts and passes per-task tiered context during streaming
- [x] **124 Unit Tests**: All pass (no new tests needed — rendering helpers are UI-only)

### True LLM Agency at Decision Points (Week 5) - COMPLETE
- [x] **Agentic Reference Following Gate**: LLM decides per-reference whether to follow
  - `ReferenceDecision` Pydantic model in `src/models/research.py`
  - `REFERENCE_DECISION_PROMPT` in `src/prompts.py` (4-section format with `{language}`)
  - Gate in `execute_task()` before `resolve_reference_enhanced()` — skips tangential refs
  - Falls back to following on LLM error (safe default)
  - Logged: "Skipped ref: {target} ({type}) — {reason}"
- [x] **Quality-Gated Re-Synthesis Loop**: LLM decides accept/retry on low-quality synthesis
  - `QualityRemediationDecision` Pydantic model in `src/models/research.py`
  - `QUALITY_REMEDIATION_PROMPT` in `src/prompts.py` (4-section format with `{language}`)
  - Remediation logic in `quality_check()` — triggers when score < threshold and retry_count < 1
  - `synthesis_retry_count` and `quality_remediation_focus` state fields in `AgentState`
  - `route_after_quality()` updated: conditional routing to `synthesize` on `phase == "retry_synthesis"`
  - `synthesize()` appends `quality_remediation_focus` to prompt on retry, clears after use
  - Graph edge map updated: `quality_check → synthesize` (retry) or `quality_check → attribute_sources` (normal)
- [x] **142 Unit Tests**: All pass (22 model, 47 agent, 42 reference, 22 task search + other)

### LLM-Based Task Relevance Scoring (Week 5.1) - COMPLETE
- [x] **LLM relevance_score in TaskSummaryOutput**: Replaced broken keyword-overlap `_calculate_task_relevance()` with LLM-generated `relevance_score` (0-100)
  - `TaskSummaryOutput` gains `relevance_score: int` field (0-100, default 50)
  - `TASK_SUMMARY_PROMPT` updated with 4-tier scoring rubric (80-100 / 50-79 / 20-49 / 0-19)
  - `_generate_task_summary()` uses `result.relevance_score / 100.0` on success
  - `_calculate_task_relevance()` retained as fallback on LLM error only
- [x] **144 Unit Tests**: All pass (22 model, 49 agent, 42 reference, 22 task search + other)

### Debug & Reference Gate Improvements (Week 5.2) - COMPLETE
- [x] **Simplified Debug State Dumps**: `src/utils/debug_state.py` rewritten
  - Removed `_FIELD_GROUPS` dict and grouped formatting
  - `_format_value()` simplified: no `key` param, no special `messages` truncation
  - `dump_state_markdown()` emits flat alphabetical `## key` + code block per value
- [x] **Fixed State Dump Trigger Points**: Each dump fires at the correct phase boundary
  - `state_1hitl.md`: Start of `generate_todo()` (Phase 1 finalized)
  - `state_2todo.md`: Start of `execute_task()` when `task_id == 0` (Phase 2 finalized)
  - `state_3rabbithole.md`: Start of `validate_relevance()` (Phase 3 finalized)
  - Removed incorrect dumps from `process_hitl_todo()`, `execute_task()` end, `hitl_finalize()`
- [x] **Enhanced Reference Decision Gate**: `REFERENCE_DECISION_PROMPT` improvements
  - `query_anchor` now includes `scope` and `current_task` for better context
  - New rule: "When uncertain, FOLLOW" — skipping relevant refs is costlier than following tangential ones
- [x] **144 Unit Tests**: All pass

### Deferred to Week 6+
- [ ] Orchestrator-worker parallelization
- [ ] RAG Triad automated validation
- [ ] CI/CD integration
- [ ] Security hardening (PII redaction)
