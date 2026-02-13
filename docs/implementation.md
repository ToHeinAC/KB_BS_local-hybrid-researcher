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

### Phase 2.5: Enhanced Iterative HITL with Multi-Vector Retrieval
- [x] **Iterative Retrieval-HITL Nodes** in `nodes.py`:
  - `hitl_init()`: Initialize conversation, detect language
  - `hitl_generate_queries()`: Generate 3 queries (original + 2 alternatives)
  - `hitl_retrieve_chunks()`: Search ChromaDB, deduplicate, append to context
  - `hitl_analyze_retrieval()`: LLM analysis for concepts, gaps, coverage
  - `hitl_generate_questions()`: Generate contextual follow-up questions
    - Passes `query_retrieval` to LLM prompt via `{retrieval}` template variable
    - Questions focus on gaps in retrieved information
  - `hitl_process_response()`: Analyze user feedback, check termination
  - `hitl_finalize()`: Generate research_queries for Phase 2
- [x] **Convergence Detection**: Improved 3-tier criteria
  - Coverage score ≥ 0.8
  - Retrieval deduplication ratio ≥ 0.7
  - Knowledge gaps count ≤ 2
- [x] **AgentState Tracking**:
  - `iteration_queries`, `knowledge_gaps`, `retrieval_dedup_ratios`
  - `coverage_score`, `retrieval_history`, `query_retrieval`

**Note on `query_retrieval` Lifecycle**:
- Accumulated during HITL via `hitl_retrieve_chunks()`
- Passed to follow-up question prompts for context-aware questions
- Persists in state after `hitl_finalize()` (LangGraph only overwrites returned keys)
- **Not currently used in Phase 2+** (Phase 3 does independent vector searches into `research_context`)
- [x] **Graph Entry Routing**: Conditional entry point (`route_entry_point`)
- [x] **UI Support**: Live display of retrieval stats and coverage during HITL phase
- [x] **Centralized Prompts**: All LLM prompts in `src/prompts.py`


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

### Phase 3.7: Graded Context Management (NEW)

Prevents query drift and ensures synthesis quality through tiered context classification:

- [x] **Query Anchor & HITL Context Preservation** (Phase A):
  - `query_anchor`: Immutable reference to original intent created in `hitl_finalize`
  - `hitl_smry`: Citation-aware HITL summary with `[Source_filename]` annotations
  - `HITL_SUMMARY_PROMPT` in `src/prompts.py`
  - `_generate_hitl_summary()` helper in `nodes.py` (retrieval truncation: 8K chars)

- [x] **Strict Language Enforcement** (Phase F):
  - `generate_structured_with_language()` in `OllamaClient`
  - `_validate_language()`: German/English marker heuristics
  - Automatic retry with stronger language instruction on mismatch

- [x] **Graded Context Classification** (Phase B):
  - `classify_context_tier()`: Assigns Tier 1/2/3 based on source, depth, relevance
  - `create_tiered_context_entry()`: Creates weighted context dicts with optional `task_id` for per-task UI filtering
  - Chunks accumulated into `primary_context`, `secondary_context`, `tertiary_context`
  - Tier 1: Direct search, relevance ≥0.85 or entity match (weight 1.0)
  - Tier 2: Depth-1 refs or medium relevance 0.6-0.85 (weight 0.7)
  - Tier 3: Depth-2+ or HITL retrieval (weight 0.4)

- [x] **Verbatim Quote Preservation** (Phase C):
  - `PreservedQuote`, `InfoExtractionWithQuotes` models in `src/models/research.py`
  - `extract_info_with_quotes()`: Returns condensed info + critical verbatim quotes
  - `INFO_EXTRACTION_WITH_QUOTES_PROMPT` for legal/technical precision
  - Modified `create_chunk_with_info()` returns (chunk, quotes) tuple

- [x] **Per-Task Structured Summary** (Phase D):
  - `_generate_task_summary()`: Receives per-task tiered findings (`task_primary`, `task_secondary`, `task_tertiary`) + `preserved_quotes`
  - `execute_task()` tracks per-task tier boundaries (start indices) and slices per-task additions
  - Uses `_format_tiered_findings()` to format each tier for the prompt
  - `TASK_SUMMARY_PROMPT` receives `{primary_findings}`, `{secondary_findings}`, `{tertiary_findings}`, `{preserved_quotes}`, `{hitl_smry}`
  - Tier priority rule: primary > secondary > tertiary (conflicts noted in gaps)
  - LLM generates `relevance_score` (0-100) as part of `TaskSummaryOutput`, converted to 0.0-1.0
  - `_calculate_task_relevance()` retained as keyword-overlap fallback on LLM error only
  - `task_summaries` state field accumulated during task execution

- [x] **Pre-Synthesis Relevance Validation** (Phase G):
  - `validate_relevance()` node: Filters drift before synthesis
  - `_score_and_filter_context()`: Scores against query_anchor entities
  - Threshold: 0.5 primary, 0.4 secondary, 0.3 tertiary
  - Logs warning when >30% of accumulated context is filtered
  - `RELEVANCE_SCORING_PROMPT` in `src/prompts.py`

- [x] **Deep Report Synthesis** (Phase E):
  - `SYNTHESIS_PROMPT_ENHANCED`: Expert report writer producing extensive markdown-formatted deep reports
  - Instructs structured sections (####), exact figures, verbatim quotes, section references
  - No sentence cap — produces comprehensive thematic reports instead of brief summaries
  - Receives only `{original_query}`, `{hitl_smry}`, `{task_summaries}`, `{language}`
  - `_format_task_summaries()` enriches summaries with key_findings, gaps, and preserved quotes
  - Falls back to legacy `SYNTHESIS_PROMPT` (also deep-report style) if no graded context available

- [x] **New Pydantic Models** in `src/models/results.py`:
  - `SynthesisOutputEnhanced`: With query_coverage (0-100) and remaining_gaps
  - `TaskSummaryOutput`: For per-task summary generation
  - `RelevanceScoreOutput`: For relevance scoring

- [x] **Graph Update** in `src/agents/graph.py`:
  - Added `validate_relevance` node between `execute_task` and `synthesize`
  - `route_after_execute()` now routes to `validate_relevance` instead of `synthesize`
  - `route_after_validate_relevance()` always routes to `synthesize`

### Phase 3.8: Prompt Standardization & Multi-Query Task Execution (NEW)

Universal language enforcement and improved search quality:

- [x] **Prompt 4-Section Format**: All 19 prompts reformatted to `### Task / ### Input / ### Rules / ### Output format`
  - Merged `FOLLOW_UP_QUESTIONS_DE` + `FOLLOW_UP_QUESTIONS_EN` into single `FOLLOW_UP_QUESTIONS_PROMPT` with `{language}`
  - JSON keys stay in English (structural); only JSON *values* must be in `{language}`

- [x] **Universal `{language}` Enforcement**: All 17 content-bearing prompts include `{language}` template variable
  - `hitl_service.py` callers (5 functions): `_analyse_user_feedback_llm`, `_generate_knowledge_base_questions_llm`, `generate_alternative_queries_llm`, `analyze_retrieval_context_llm`, `generate_refined_queries_llm`
  - `nodes.py` callers (5 functions): `generate_todo`, `execute_task`, `quality_check`, `hitl_generate_queries`, `hitl_analyze_retrieval`
  - 3 functions gained `language: str = "de"` parameter: `generate_alternative_queries_llm`, `analyze_retrieval_context_llm`, `generate_refined_queries_llm`
  - Only exceptions: `LANGUAGE_DETECTION_PROMPT` (outputs code), `REFERENCE_EXTRACTION_PROMPT` (copies verbatim)

- [x] **Multi-Query Task Execution**: `execute_task()` generates 3 queries per task:
  - `TASK_SEARCH_QUERIES_PROMPT`: LLM generates 2 targeted queries (core + complementary angle)
  - `TaskSearchQueries` model in `src/models/query.py`: `query_1`, `query_2` fields
  - Base query: task text + key concepts (concatenation, no LLM)
  - Chunk deduplication: `doc_name:page_number:chunk_text[:100]` identity key across all results
  - Fallback: if LLM generation fails, uses only the base query

- [x] **Task 0 Prepend**: `generate_todo()` always prepends original user query as Task 0
  - Ensures direct vector search for the original query regardless of LLM-generated tasks
  - `process_hitl_todo()` renumbers from 0 (was from 1)

- [x] **5-Dimension Quality Scoring**:
  - Added `query_relevance` (0-100) to `QualityCheckOutput` and `QualityAssessment`
  - `QualityAssessment.overall_score`: 0-500 (was 0-400)
  - `quality_threshold` default: 375 (was 300)
  - `QUALITY_CHECK_PROMPT` includes `{language}` for issue descriptions
  - UI (`results_view.py`) and CLI (`main.py`) display `/500`

- [x] **Enhanced `TaskSummaryOutput`**:
  - `relevance_assessment: str`: Whether findings actually match query intent
  - `irrelevant_findings: list[str]`: Findings superficially related but not answering the query
  - `relevance_score: int` (0-100): LLM-generated relevance score with 4-tier rubric

- [x] **Relevance Filter Backfill**: `filter_by_relevance()` accepts `min_results: int = 0`
  - When threshold filtering yields fewer results, backfills from top-scoring rejected chunks
  - Prevents empty result sets for tasks with low-relevance corpus

- [x] **Language-Aware Extraction**:
  - `extract_info(chunk_text, query, language="de")`: passes `{language}` to `INFO_EXTRACTION_PROMPT`
  - `extract_info_with_quotes(...)`: passes `{language}` to `INFO_EXTRACTION_WITH_QUOTES_PROMPT`
  - `create_chunk_with_info(...)`: accepts `language` param, passes through to extraction functions

- [x] **Config Tuning**: `k_results` default 5→3 in `SessionState`

- [x] **106 Unit Tests**: All pass including new `tests/test_task_search_queries.py`

### Phase 3.9: True LLM Agency at Decision Points (NEW)

Two agentic decision points where the LLM autonomously decides control flow:

- [x] **Agentic Reference Following Gate** (in `execute_task()`):
  - `REFERENCE_DECISION_PROMPT` in `src/prompts.py` (4-section format with `{language}`)
  - `ReferenceDecision` model in `src/models/research.py`: `{follow: bool, reason: str}`
  - Gate receives full context: `original_query`, `key_entities`, `scope`, `current_task`
  - Biased toward following when uncertain (skipping relevant refs is costlier)
  - Skips tangential, repetitive, or vague references (logged for transparency)
  - Falls back to following on LLM error (safe default)

- [x] **Quality-Gated Re-Synthesis Loop** (in `quality_check()` + `synthesize()`):
  - `QUALITY_REMEDIATION_PROMPT` in `src/prompts.py` (4-section format with `{language}`)
  - `QualityRemediationDecision` model: `{action: "accept"|"retry", focus_instructions: str}`
  - Triggered when `total_score < quality_threshold` (375) and `synthesis_retry_count < 1`
  - If `action == "retry"`: increments retry count, stores focus instructions, sets `phase="retry_synthesis"`
  - `synthesize()` appends `quality_remediation_focus` to prompt on retry, clears after use
  - `route_after_quality()` conditional: `"retry_synthesis"` → `synthesize`, else → `attribute_sources`
  - Max 1 retry to prevent infinite loops

- [x] **State Fields** in `AgentState` + `create_initial_state()`:
  - `synthesis_retry_count: int` (default 0)
  - `quality_remediation_focus: str` (default "")

- [x] **Graph Update**: `route_after_quality()` returns `Literal["synthesize", "attribute_sources"]`
  - Edge map: `{"synthesize": "synthesize", "attribute_sources": "attribute_sources"}`

- [x] **142 Unit Tests**: All pass (18 new agentic tests + 3 prompt/model tests)

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

### Phase 6.5: UI Enhancements (NEW)
- [x] **Retrieval History Panel** (`hitl_panel.py`):
  - `_perform_hitl_retrieval()`: Vector search during chat-based HITL
  - `_render_retrieval_history()`: Expander with queries, chunks, dedup stats
  - Nested expanders showing chunk details (doc, page, score, text)
  - Reads from `hitl_state` during HITL, `agent_state` during research
- [x] **Database Selection Fix**:
  - HITL retrieval now respects `session.selected_database`
  - Uses `search_by_database_name()` when specific DB selected
  - Falls back to `search_all_collections()` when no selection
- [x] **Active Database Indicator**: Sidebar shows "Aktive DB: {name}"
- [x] **Cached Service Clients** (`@st.cache_resource`):
  - `_get_chromadb_client()` in `safe_exit.py`
  - `_get_ollama_client()` in `safe_exit.py`
  - `_get_hitl_service()` in `hitl_panel.py`
  - `get_chromadb_client()` in `app.py`
- [x] **Graph Entry Point Enhancement** (`graph.py`):
  - `route_entry_point()` returns 4 targets: `hitl_init`, `hitl_process_response`, `generate_todo`, `process_hitl_todo`
  - Priority: todo-approval resume (`hitl_decision + !hitl_active`) checked first
  - Iterative HITL resume (`hitl_decision + hitl_active`) checked second
  - `_start_research_from_hitl()` sets `hitl_active=False` to prevent misrouting
- [x] **Coverage Metrics in Checkpoints** (`nodes.py`):
  - `hitl_generate_questions()` includes coverage, gaps, dedup ratios
  - UI displays knowledge gaps in expander

### Phase 6.7: Todo Side Panel & Streaming Improvements
- [x] **Expander-based task list** (`todo_side_panel.py`):
  - Each task rendered as `st.expander` with icon + sequential number + truncated header
  - Full task text shown inside expander body (previously truncated to 40 chars)
  - Currently executing task auto-expanded; others collapsed
  - Unicode emoji chars used directly (`:colon_emoji:` shortcodes don't render in expander labels)
- [x] **Verbose task spinner** during `execute_tasks` phase:
  - Shows `Aufgabe {position}/{total}: {task_text}` with up to 80 chars
  - Phase description shown via `st.caption` for all other phases
- [x] **Simplified graph streaming** (`app.py`):
  - Removed inline column layout from `_run_graph_stream()` — UI renders via Streamlit rerun cycle
- [x] **Sequential task ID renumbering** (`nodes.py`):
  - `process_hitl_todo` renumbers task IDs sequentially after user removals/additions

### Phase 6.8: Persistent Results View
- [x] **HITL Expander** (`results_view.py: _render_hitl_expander()`):
  - Renders conversation history as `st.chat_message` bubbles
  - Shows `hitl_smry` (LLM summary with `[Source]` annotations)
  - Lists numbered research queries from `hitl_result`
  - Guard: skips when conversation, hitl_smry, and research_queries are all empty
- [x] **Per-Task Expanders** (`results_view.py: _render_task_expanders()`):
  - One `st.expander` per task with formatted summary via shared `render_task_summary_markdown()`
  - Tiered chunk rendering via `render_tiered_chunks()` when `task_id` entries exist, flat fallback for old states
  - `task_summaries` matched by `task_id` dict lookup
  - Guard: skips when `todo_list` is empty
- [x] **Shared Task Rendering** (`src/ui/components/task_rendering.py`):
  - `render_task_summary_markdown()`: Summary, key findings, gaps, relevance (% + text)
  - `render_chunk_expander()`: Per-chunk expander with header (doc, page, score), full extraction, divider, full original text
  - `render_tiered_chunks()`: Renders chunks grouped by tier as nested expanders (Tier 1 expanded, others collapsed, empty tiers hidden)
  - `filter_tiered_context_by_task()`: Filters tiered context lists to entries matching a specific `task_id`
  - `has_task_id_entries()`: Backward compat check — returns True if any entry has `task_id` key
  - Handles both dict and object access patterns
  - Used by live view (`app.py: _render_task_result_expander`) and results view (`results_view.py: _render_task_expanders`)
- [x] **Insertion point**: After metrics row + first divider, before `### :microscope: Detailbericht`
- [x] **Data sources**: All data from `session.agent_state` and `session.hitl_conversation_history` (persisted across phase transitions, only cleared on reset)

### Phase 6.9: Tiered Chunk Rendering in Task Expanders
- [x] **`task_id` in Tiered Context Entries** (`tools.py`, `nodes.py`):
  - `create_tiered_context_entry()` accepts optional `task_id: int | None = None`
  - `execute_task()` passes `task_id` for both direct vector search chunks and nested reference chunks
  - Backward compatible: old callers without `task_id` unaffected
- [x] **Tiered Rendering Helpers** (`task_rendering.py`):
  - `render_tiered_chunks(primary, secondary, tertiary)`: 3 nested tier expanders with German labels
  - `filter_tiered_context_by_task()`: Filters each context list to entries matching a specific `task_id`
  - `has_task_id_entries()`: Detects old states without `task_id` for backward compat fallback
- [x] **Results View Update** (`results_view.py`):
  - `_render_task_expanders()` reads `primary_context`, `secondary_context`, `tertiary_context` from state
  - Uses tiered rendering when `task_id` entries detected, falls back to flat chunk rendering otherwise
- [x] **Live View Update** (`app.py`):
  - `_render_task_result_expander()` accepts optional `tiered_context` tuple
  - `_run_graph_with_live_updates()` extracts tiered context per task during streaming

### Phase 7: Polish
- [x] Multi-collection search
- [ ] Query history and caching
- [ ] Export results (JSON, Markdown)
- [x] Error handling and recovery
- [x] Logging and observability

### Phase 6.6: UI Localization & Layout Fixes
- [x] **German localization** of todo approval panel (`todo_approval.py`):
  - All labels, buttons, and messages translated to German
- [x] **Layout fix**: HITL summary and checkpoints moved inside column layout for consistent rendering
- [x] **Removed unused** `render_connection_status()` import and sidebar call

### Phase 8: Testing Improvements
- [x] `TestRouteEntryPoint` class for graph routing logic
  - `test_route_to_hitl_init_on_new_session`
  - `test_route_to_hitl_process_response_on_resume`
  - `test_route_to_generate_todo_with_research_queries`
  - `test_route_to_generate_todo_with_phase`
  - `test_decision_without_hitl_active_routes_to_process_hitl_todo`

---

## Coding Standards

### From Global CLAUDE.md
- **Tests first** when changing behavior
- **Functions under ~40 lines** when possible
- **Small commits**, imperative messages, never commit `.env` or secrets
- **Streamlit on ports >8510**

### Project-Specific
- **Type hints required** on all functions
- **Pydantic models** for all data structures
- **Structured JSON output** via `method="json_mode"` for Ollama
- **LangGraph** for agent orchestration (NOT AgentExecutor)
- **Docstrings** on public functions (Google style)
- **Prompt format**: All prompts in `src/prompts.py` use 4-section format (`### Task / ### Input / ### Rules / ### Output format`)
- **Language enforcement**: Every content-bearing prompt includes `{language}` template variable; callers compute `lang_label = "German" if language == "de" else "English"`

### Example Function

```python
from pydantic import BaseModel
from langchain_ollama import ChatOllama

class SearchResult(BaseModel):
    """A single search result from vector DB."""
    text: str
    score: float
    source: str

def search_vectors(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search ChromaDB collection for similar documents.

    Args:
        query: Search query text
        collection_name: Name of ChromaDB collection
        top_k: Number of results to return

    Returns:
        List of search results with scores
    """
    client = get_chromadb_client()
    collection = client.get_collection(collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return [
        SearchResult(
            text=doc,
            score=1 - dist,  # Convert distance to similarity
            source=meta.get("source", "unknown"),
        )
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]
```

---

## Testing Patterns

### Unit Tests

```python
import pytest
from src.models.query import QueryAnalysis

def test_query_analysis_creation():
    """Test QueryAnalysis model creation."""
    analysis = QueryAnalysis(
        original_query="What are dose limits?",
        key_concepts=["dose limits", "radiation"],
        entities=["StrlSchV"],
        scope="regulatory",
        assumed_context=["German law"],
        clarification_needed=True,
    )

    assert analysis.original_query == "What are dose limits?"
    assert len(analysis.key_concepts) == 2
    assert analysis.clarification_needed is True
```

### Integration Tests

```python
import pytest
from src.services.ollama_client import OllamaClient
from src.models.query import QueryAnalysis

@pytest.fixture
def ollama_client():
    return OllamaClient()

def test_structured_output(ollama_client):
    """Test Ollama structured output with json_mode."""
    result = ollama_client.analyze_query(
        "What are the dose limits for occupational exposure?"
    )

    assert isinstance(result, QueryAnalysis)
    assert len(result.key_concepts) > 0
```

### Agent Tests

```python
import pytest
from src.agents.orchestrator import create_research_graph

def test_graph_execution():
    """Test full graph execution."""
    graph = create_research_graph()

    result = graph.invoke({
        "query": "Test query",
        "phase": "analyze",
    })

    assert "query_analysis" in result
    assert "todo_list" in result
```

---

## Known Challenges & Mitigations

| Challenge | Root Cause | Mitigation |
|-----------|-----------|-----------|
| **Infinite reference loops** | Circular cross-references | Maintain `visited_refs` set, track recursion depth, convergence detection |
| **Reference resolution ambiguity** | Multiple matches | Document registry with 3-stage matching, scoped search within collection |
| **Hallucinated references** | LLM invents citations | Hybrid: regex provides baseline, LLM adds coverage, dedup filters noise |
| **Over-following tangential refs** | Poor relevance filter | Agentic reference gate (LLM decides per-ref), token budget (50K), convergence (same doc >= 3) |
| **Query drift during synthesis** | Accumulating irrelevant context | Tiered evidence resolved at task summary level, pre-synthesis relevance validation, query_anchor |
| **Lost HITL context** | HITL findings not used in synthesis | `hitl_smry` fed into todo generation, task summaries, and synthesis + `tertiary_context` from HITL retrieval |
| **Mixed language output** | LLM ignores language instruction | `generate_structured_with_language()` with validation and retry |
| **Lost legal/technical precision** | Summarization paraphrases quotes | `preserved_quotes` extracted verbatim during info extraction |
| **Low-quality synthesis passed through** | No remediation | Agentic quality remediation loop (LLM decides accept/retry, max 1 retry with focused instructions) |
| **Report bloat** | Including everything | Strict extractive summarization |
| **Long execution times** | Deep recursion | Default N=3, M=4, depth=2 |
| **Ollama structured output failures** | Wrong method | Use `method="json_mode"` for <30B models |

---

## File Organization

```
src/
├── __init__.py
├── main.py                    # CLI entry point
├── config.py                  # Settings via pydantic-settings
│
├── models/                    # Pydantic data models
│   ├── __init__.py
│   ├── query.py              # QueryAnalysis, ToDoList
│   ├── research.py           # ResearchContext, ResearchTask
│   ├── results.py            # VectorResult, DocumentFinding
│   └── report.py             # FinalReport, QualityAssessment
│
├── agents/                    # LangGraph agents
│   ├── __init__.py
│   ├── graph.py              # StateGraph definition
│   ├── nodes.py              # Node functions
│   └── tools.py              # Tool definitions
│
├── services/                  # Core services
│   ├── __init__.py
│   ├── chromadb_client.py
│   ├── ollama_client.py
│   ├── embedding_service.py
│   └── pdf_reader.py
│
└── ui/                        # Streamlit application
    ├── __init__.py
    ├── app.py
    ├── state.py
    └── components/
        ├── __init__.py
        ├── query_input.py
        ├── hitl_panel.py
        ├── todo_list.py
        ├── results_view.py
        ├── task_rendering.py
        └── safe_exit.py
```
