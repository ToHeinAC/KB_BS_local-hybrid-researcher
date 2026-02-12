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
  - `create_tiered_context_entry()`: Creates weighted context dicts
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
  - `_generate_task_summary()`: Creates summary with key findings and relevance score
  - Accepts `hitl_smry` parameter to avoid repeating HITL-established findings
  - `_calculate_task_relevance()`: Word/entity overlap scoring
  - `task_summaries` state field accumulated during task execution
  - `TASK_SUMMARY_PROMPT` in `src/prompts.py` (includes `{hitl_smry}` input)

- [x] **Pre-Synthesis Relevance Validation** (Phase G):
  - `validate_relevance()` node: Filters drift before synthesis
  - `_score_and_filter_context()`: Scores against query_anchor entities
  - Threshold: 0.5 primary, 0.4 secondary, 0.3 tertiary
  - Logs warning when >30% of accumulated context is filtered
  - `RELEVANCE_SCORING_PROMPT` in `src/prompts.py`

- [x] **Query-Anchored Synthesis** (Phase E):
  - `SYNTHESIS_PROMPT_ENHANCED`: Tiered context structure with explicit instructions
  - Modified `synthesize()` uses graded context + language enforcement
  - Includes `hitl_smry`, `preserved_quotes`, `task_summaries`
  - Falls back to legacy synthesis if no graded context available

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

- [x] **Relevance Filter Backfill**: `filter_by_relevance()` accepts `min_results: int = 0`
  - When threshold filtering yields fewer results, backfills from top-scoring rejected chunks
  - Prevents empty result sets for tasks with low-relevance corpus

- [x] **Language-Aware Extraction**:
  - `extract_info(chunk_text, query, language="de")`: passes `{language}` to `INFO_EXTRACTION_PROMPT`
  - `extract_info_with_quotes(...)`: passes `{language}` to `INFO_EXTRACTION_WITH_QUOTES_PROMPT`
  - `create_chunk_with_info(...)`: accepts `language` param, passes through to extraction functions

- [x] **Config Tuning**: `k_results` default 5→3 in `SessionState`

- [x] **106 Unit Tests**: All pass including new `tests/test_task_search_queries.py`

### Phase 4: Synthesis + Quality (Research Phase 4)
- [x] `synthesize` node (LLM synthesis from extracted findings)
- [x] Enhanced `synthesize` with tiered context, preserved quotes, language enforcement
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
| **Over-following tangential refs** | Poor relevance filter | Set threshold=0.6+, token budget (50K), convergence (same doc >= 3) |
| **Query drift during synthesis** | Accumulating irrelevant context | Graded context tiers, pre-synthesis relevance validation, query_anchor |
| **Lost HITL context** | HITL findings not used in synthesis | `hitl_smry` fed into todo generation, task summaries, and synthesis + `tertiary_context` from HITL retrieval |
| **Mixed language output** | LLM ignores language instruction | `generate_structured_with_language()` with validation and retry |
| **Lost legal/technical precision** | Summarization paraphrases quotes | `preserved_quotes` extracted verbatim during info extraction |
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
        └── safe_exit.py
```
