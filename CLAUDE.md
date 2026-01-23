# AI-LLM Agentic Researcher with Hybrid Approach

A fully local, privacy-first research system combining agentic document search with semantic vector DB retrieval using Ollama LLMs.

## Project Overview

This researcher performs deep analytical research on fully private data through a **hybrid approach**:
- **Agentic Document Search**: ReAct agent dynamically navigates PDF documents, extracts relevant information, and discovers interconnections
- **Semantic Vector Search**: ChromaDB-based retrieval from enriched document embeddings

All components run locally:
- **LLM**: Ollama (no API calls, full privacy)
- **Vector DB**: ChromaDB (persistent local storage)
- **Data**: Local PDF corpus + embedded vectors
- **Orchestration**: LangChain with ReAct agent pattern
- **UI**: Streamlit web interface

**Target Users:** Technical researchers and analysts seeking project intelligence on private document collections.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Web UI                                │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │Query Input │ │To-Do List  │ │Results View  │ │HITL Panel          │  │
│  └────────────┘ └────────────┘ └──────────────┘ └────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                    Phase 1: Human-in-the-Loop                            │
│  User Query → Conversational Refinement → To-Do List (3-5 items)        │
│                              ↓                                           │
│           HITL Checkpoint: User approves/modifies tasks                  │
│           (Tasks start with NO queries - generated in Phase 2)          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                    Phase 2: ReAct Agent Loop                             │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Per Todo-Item Loop                               │ │
│  │                                                                     │ │
│  │  ┌──────────────────┐                                               │ │
│  │  │ generate_queries │ ◄───────────────────────────────────┐        │ │
│  │  └────────┬─────────┘                                     │        │ │
│  │           ▼                                               │        │ │
│  │  ┌──────────────────────────────────────────────────────┐ │        │ │
│  │  │              research (parallel)                      │ │        │ │
│  │  │  ┌────────────┐ ┌─────────────────┐ ┌────────────┐   │ │        │ │
│  │  │  │ Vector DB  │ │ Doc Agent       │ │ Web Search │   │ │        │ │
│  │  │  │ Search     │ │ (scan/preview/  │ │ (Tavily)   │   │ │        │ │
│  │  │  │            │ │  parse/read/    │ │            │   │ │        │ │
│  │  │  │            │ │  grep/glob)     │ │            │   │ │        │ │
│  │  │  │            │ │  + ref tracking │ │            │   │ │        │ │
│  │  │  └────────────┘ └─────────────────┘ └────────────┘   │ │        │ │
│  │  └──────────────────────────┬───────────────────────────┘ │        │ │
│  │                             ▼                             │        │ │
│  │  ┌──────────────────┐                                     │        │ │
│  │  │    summarize     │ (per category, preserves sources)   │        │ │
│  │  └────────┬─────────┘                                     │        │ │
│  │           ▼                                               │        │ │
│  │  ┌──────────────────┐     gaps + queries < MAX?           │        │ │
│  │  │    critique      │ ────────────────────────────────────┘        │ │
│  │  └────────┬─────────┘                                              │ │
│  │           │ no gaps OR queries >= MAX                              │ │
│  │           ▼                                                        │ │
│  │     Mark todo-item DONE → Next todo-item                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ All todo-items done
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 3: Report Generation                            │
│                                                                          │
│  ┌────────┐   ┌───────────┐   ┌─────────────────┐   ┌──────────────┐   │
│  │ Rerank │ → │ GenAnswer │ → │ Quality Checker │ → │ Source Linker│   │
│  └────────┘   └─────┬─────┘   └────────┬────────┘   └──────────────┘   │
│                     ▲                  │                                 │
│                     └──────────────────┘                                 │
│                      (reflection if fails)                               │
│                                                                          │
│                              ▼                                           │
│              ┌───────────────────────────────────────┐                  │
│              │     Final Report (Structured JSON)    │                  │
│              │  - Answer with [Source] citations     │                  │
│              │  - Clickable source links             │                  │
│              │  - Quality metrics                    │                  │
│              └───────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    │                                                           │
    ▼                             ▼                             ▼
┌────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ PDF Corpus     │    │ ChromaDB         │    │ Ollama LLM          │
│ (kb/insert_data)│    │ (kb/database/)   │    │ (qwen3:14b)         │
└────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Data Flow

**Phase 1: HITL + To-Do List Generation**
1. **User Query** → Streamlit UI captures research question
2. **HITL Phase** → Conversational refinement with user validation
3. **To-Do List Created** → 3-5 research tasks (mutable, max MAX_TODO_ITEMS)
   - Tasks start with NO queries (generated on-demand in Phase 2)
4. **HITL Checkpoint** → User approves/modifies tasks before research begins

**Phase 2: ReAct Agent with Ordered Tool Loop**

For each todo-item, execute in order:
```
generate_queries → research → summarize → critique
        ▲                                    │
        └────────────────────────────────────┘
                  (if gaps found AND queries < MAX)
```

5. **generate_queries** → Create MIN_QUERY_ITEMS to MAX_QUERY_ITEMS queries per category
   - `vector_queries`: Semantic queries for ChromaDB
   - `doc_keywords`: Keywords for document search (3× multiplier)
   - `web_queries`: Contextual queries for web (if enabled)
6. **research** → Parallel execution across all sources:
   - Vector DB: Query ChromaDB, return MAX_VDB_ITEMS results per query
   - Document Agent: Agentic navigation with reference tracking (up to MAX_DOCS)
   - Web Search: Tavily results (WEB_RESULTS_PER_QUERY per query)
7. **summarize** → One summary per category preserving source attribution
8. **critique** → Gap analysis with cross-document/cross-todo relationship detection
   - If significant gaps AND queries < MAX → generate new queries, loop back
   - Else → mark todo-item DONE, proceed to next

**Phase 3: Report Generation**
9. **Rerank** → Cross-todo-item deduplication and relevance re-scoring
10. **GenAnswer** → Synthesize response with `[Source_filename]` citations
11. **Quality Check** → Validate (0-400 score), reflection if fails
12. **Source Linker** → Convert citations to clickable PDF links
13. **Final Report** → Structured JSON with linked sources and quality metrics

---

## Directory Structure

```
KB_BS_local-hybrid-researcher/
├── CLAUDE.md                 # This file - implementation blueprint
├── pyproject.toml            # Project dependencies and config
├── .env.example              # Environment template
├── .env                      # Local config (git-ignored)
│
├── src/
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   ├── config.py             # Settings via pydantic-settings
│   │
│   ├── models/               # Pydantic data models
│   │   ├── __init__.py
│   │   ├── query.py          # ResearchQuery, QueryPlan
│   │   ├── response.py       # ResearchResponse, Source, Finding
│   │   └── document.py       # DocumentChunk, PDFMetadata
│   │
│   ├── agents/               # LangChain agents
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # ReAct main agent
│   │   ├── doc_search.py     # PDF navigation agent
│   │   ├── vector_search.py  # ChromaDB search agent
│   │   ├── folder_researcher.py  # Document folder navigation tools
│   │   └── tools.py          # Tool definitions
│   │
│   ├── services/             # Core services
│   │   ├── __init__.py
│   │   ├── chromadb_client.py    # ChromaDB connection
│   │   ├── ollama_client.py      # Ollama LLM wrapper
│   │   ├── embedding_service.py  # Embedding generation
│   │   └── pdf_reader.py         # PDF text extraction
│   │
│   └── ui/                   # Streamlit application
│       ├── __init__.py
│       ├── app.py            # Main Streamlit app
│       ├── state.py          # Session state management
│       └── components/
│           ├── __init__.py
│           ├── query_input.py
│           ├── results_view.py
│           ├── hitl_panel.py     # Human-in-the-loop
│           ├── todo_list.py      # Research task tracker
│           └── safe_exit.py      # Safe exit button
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── test_models.py
│   ├── test_agents.py
│   └── test_services.py
│
└── kb/                       # Knowledge base (pre-existing)
    ├── database/             # ChromaDB persistent stores
    │   ├── GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000/
    │   ├── NORM__Qwen--Qwen3-Embedding-0.6B--3000--600/
    │   ├── StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600/
    │   └── StrlSchExt__Qwen--Qwen3-Embedding-0.6B--3000--600/
    ├── insert_data/          # Source PDF documents (761 files)
    ├── GLageKon__db_inserted/    # Tracking: inserted docs
    ├── NORM__db_inserted/
    ├── StrlSch__db_inserted/
    └── StrlSchExt__db_inserted/
```

---

## Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | >=3.11 | Runtime |
| uv | latest | Package manager |
| langchain | >=0.3.0 | Agent orchestration |
| langchain-ollama | >=0.2.0 | Ollama integration |
| langchain-chroma | >=0.1.0 | ChromaDB integration |
| chromadb | >=0.5.0 | Vector database |
| streamlit | >=1.28.0 | Web UI |
| pydantic | >=2.0.0 | Data validation |
| pydantic-settings | >=2.0.0 | Config management |
| pymupdf | >=1.24.0 | PDF text extraction |

### Ollama Models

| Model | Purpose | RAM Required |
|-------|---------|--------------|
| qwen3:14b | Primary LLM (reasoning) | ~10GB |
| llama3.1:8b | Fallback LLM | ~6GB |
| Qwen/Qwen3-Embedding-0.6B | Embeddings (via Ollama) | ~1GB |

---

## Agent Architecture

### ReAct Orchestrator

The main agent follows the ReAct (Reasoning + Acting) pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                        ReAct Loop                            │
│                                                              │
│  1. THOUGHT: "I need to find information about X..."         │
│  2. ACTION: call tool (doc_search OR vector_search)          │
│  3. OBSERVATION: tool returns results                        │
│  4. THOUGHT: "Based on this, I should now..."               │
│  5. [Repeat until sufficient information OR max iterations]  │
│  6. FINAL ANSWER: structured JSON response                   │
└─────────────────────────────────────────────────────────────┘
```

### Main Orchestrator Tools

The ReAct orchestrator uses four tools executed in order per todo-item:

| Tool | Purpose | When Called |
|------|---------|-------------|
| `generate_queries` | Create MIN to MAX_QUERY_ITEMS queries per category | Only if no queries exist for current todo-item |
| `research` | Execute searches across all sources | For each available query |
| `summarize` | Create per-category summary with sources | After research completes for todo-item |
| `critique` | Gap analysis, spawn new queries if needed | After summarize, triggers loop back |

```python
# Tool 1: Generate Queries
def generate_queries(
    todo_item: ResearchTask,
    query_context: QueryContext,
) -> QuerySet:
    """Create research queries for a todo-item.

    Input: todo-item description, query_context (original query + HITL history)
    Output: QuerySet with queries for each category

    Only generates if todo_item.query_sets is empty or critique requested new queries.
    Generates diverse queries covering different aspects.
    For doc_keywords: 3× multiplier ensures sufficient keyword coverage.
    """

# Tool 2: Research (delegates to parallel sub-components)
def research(
    query_set: QuerySet,
    todo_item: ResearchTask,
) -> ResearchResults:
    """Execute searches across all sources in parallel.

    Delegates to:
    - Vector DB Search: Query ChromaDB, return MAX_VDB_ITEMS results
    - Document Research Agent: Agentic navigation with reference tracking
    - Web Search: Tavily results (if enabled)

    Returns combined results with full source tracking.
    """

# Tool 3: Summarize
def summarize(
    research_results: ResearchResults,
    todo_item: ResearchTask,
) -> list[CategorySummary]:
    """Create per-category summary preserving source attribution.

    Scope: One summary per category (vector_db, documents, web) per todo-item.
    Preserves exact quotes and figures from sources.
    Tracks which source contributed which finding.
    Includes source_quotes for later citation verification.
    """

# Tool 4: Critique
def critique(
    summaries: list[CategorySummary],
    todo_item: ResearchTask,
) -> CritiqueResult:
    """Gap analysis with relationship detection.

    1. Compare summaries against todo-item requirements
    2. Identify gaps in coverage
    3. If severity == "significant" AND current_queries < MAX_QUERY_ITEMS:
       - Generate new queries focusing on:
         - Cross-document relationships (documents referencing each other)
         - Cross-todo-item relationships (findings relevant to other tasks)
       - Return should_continue = True → loop back to generate_queries
    4. Else: Mark todo-item done, proceed to next
    """
```

### Document Research Agent (Sub-Agent)

The Document Research Agent is an agentic sub-agent called by the `research` tool
for navigating and extracting information from PDF documents:

**Behavior:**
1. Start with doc_keywords from QuerySet
2. Use grep/glob to find initial matches
3. For each match:
   - Read surrounding context (up to DOC_WORD_LIMIT words per doc, MAX_DOCS total)
   - **Check for references to other documents** (e.g., "siehe Dokument 003_...", "gemäß StrlSchV")
   - Mark referenced documents for analysis in next iteration
4. **Iterative expansion:**
   - Use relevant passages as new search terms
   - Follow document references up to DOC_REFERENCE_HOPS hops
   - Accumulate findings with source tracking

**Relevance scoring (weighted average):**
```python
score = KEYWORD_DENSITY_WEIGHT * keyword_density + DOC_FREQUENCY_WEIGHT * document_frequency
# keyword_density: matches_in_doc / total_words_in_doc
# document_frequency: docs_with_keyword / total_docs_searched
```

**Sub-Agent Tools:**

```python
# Tool 1: Scan Folder
def scan_folder(path: str = "kb/insert_data") -> FolderStructure:
    """Scan folder structure, return tree of PDFs with metadata."""

# Tool 2: Preview Document
def preview(doc_path: str, pages: int = 3) -> DocumentPreview:
    """Quick preview - first N pages, TOC if available, metadata.
    Use to decide if document is worth full parsing.
    """

# Tool 3: Parse Document
def parse(doc_path: str) -> ParsedDocument:
    """Full parse - extract all text, tables, structure.
    More expensive than preview; use after preview confirms relevance.
    """

# Tool 4: Read Section
def read(doc_path: str, page: int | None = None, section: str | None = None) -> str:
    """Read specific page or section from document."""

# Tool 5: Grep Documents
def grep(pattern: str, doc_filter: str | None = None) -> list[GrepMatch]:
    """Search for regex pattern across documents."""

# Tool 6: Glob Documents
def glob(pattern: str) -> list[str]:
    """Find documents by filename pattern.
    Examples: '*Strl*.pdf', '003_*.pdf', '*Gebäude*.pdf'
    """
```

### Web Search Tool (Optional)

```python
def web_search(query: str, max_results: int = 2) -> list[WebSearchResult]:
    """Search the web for supplementary information.

    Only used when ENABLE_WEB_SEARCH=true in configuration.
    Returns WEB_RESULTS_PER_QUERY results per query.
    Supplements local document search, does not replace it.
    """
```

### Human-in-the-Loop Integration (MUST-HAVE)

Integration points for user validation:

1. **Query Refinement**: User confirms/modifies interpreted research question
2. **Tool Selection Approval**: User approves agent's planned search strategy
3. **Intermediate Review**: User validates key findings before final synthesis
4. **Source Verification**: User can inspect and filter retrieved sources

```python
# HITL checkpoint example
class HITLCheckpoint(BaseModel):
    checkpoint_type: Literal["query", "strategy", "findings", "sources"]
    content: dict
    requires_approval: bool = True

async def await_user_approval(checkpoint: HITLCheckpoint) -> HITLDecision:
    """Block until user approves, modifies, or rejects."""
```

### To-Do List Component (MUST-HAVE)

The To-Do List is a **mutable object** visible in the UI showing research task breakdown:

- Created in Phase 1 with 3-5 items (max MAX_TODO_ITEMS)
- Tasks start with **no queries** (generated on-demand in Phase 2)
- Can grow during Phase 2 via critique tool identifying gaps

**Core Data Models:**

```python
class QueryContext(BaseModel):
    """Accumulated context for query generation from original query + HITL."""
    original_query: str                    # User's initial question
    hitl_conversation: list[str]           # Full HITL conversation history
    user_feedback_analysis: str | None     # Key insights from HITL feedback
    detected_language: str = "de"          # For German document corpus

class QuerySet(BaseModel):
    """Queries generated for a todo-item by generate_queries tool."""
    todo_item_id: str
    vector_queries: list[str]              # Min MIN_QUERY_ITEMS, max MAX_QUERY_ITEMS
    doc_keywords: list[str]                # 3 × MAX_QUERY_ITEMS keywords
    web_queries: list[str]                 # Min MIN_QUERY_ITEMS (if enabled)
    iteration: int = 1                     # Tracks critique loop iterations
    generated_from_critique: bool = False  # True if spawned by critique

class CategorySummary(BaseModel):
    """Summary for one research category from summarize tool."""
    category: Literal["vector_db", "documents", "web"]
    summary_text: str
    key_findings: list[str]
    sources_used: list[Source]             # Full source objects for citation
    source_quotes: dict[str, str]          # source_id → relevant quote
    todo_item_id: str
    query_context: str                     # Original query for context

class CritiqueResult(BaseModel):
    """Gap analysis result from critique tool."""
    todo_item_id: str
    gaps_found: list[str]                  # Identified information gaps
    severity: Literal["none", "minor", "significant"]
    suggested_queries: list[str]           # New queries focusing on interrelationships
    cross_document_refs: list[str]         # Document references to explore
    cross_todo_refs: list[str]             # Connections to other todo-items
    should_continue: bool                  # True if significant gaps AND queries < MAX

class ResearchTask(BaseModel):
    """A single research task with tracking for tool execution."""
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "blocked"]

    # Query tracking (populated by generate_queries tool)
    query_sets: list[QuerySet] = []
    current_iteration: int = 0

    # Research results (populated by research tool)
    vector_results: list[VectorResult] = []
    doc_results: list[DocumentFinding] = []
    web_results: list[WebResult] = []

    # Summaries (populated by summarize tool)
    summaries: list[CategorySummary] = []

    # Critique tracking (populated by critique tool)
    critiques: list[CritiqueResult] = []

    # Source tracking for citations
    all_sources: list[Source] = []

    # Lineage
    spawned_from: str | None = None        # Parent task ID if spawned
    error_message: str | None = None

class ResearchTodoList(BaseModel):
    """Research task tracker (mutable during Phase 2)."""
    query: str
    query_context: QueryContext
    tasks: list[ResearchTask]
    current_task_id: str | None
    progress_percent: float
    started_at: datetime

    # Configuration limits
    max_tasks: int = 10                    # MAX_TODO_ITEMS
    max_queries_per_task: int = 5          # MAX_QUERY_ITEMS
    min_queries_per_task: int = 3          # MIN_QUERY_ITEMS
```

**UI Workflow:**

1. **Phase 1 Complete** → To-Do List displayed with 3-5 tasks (no queries yet)
2. **HITL Checkpoint** → User approves/modifies tasks before research begins
3. **Phase 2 Start** → First task marked in_progress
4. **Real-time Updates** → Tasks update as tools execute:
   - `generate_queries` → queries populated
   - `research` → results accumulated
   - `summarize` → summaries created
   - `critique` → may spawn new queries or mark done
5. **Loop Detection** → If critique returns `should_continue=True`, loop visible in UI
6. **Task Completion** → Status changes to completed, next task starts

**Streamlit Integration:**

```python
# In src/ui/components/todo_list.py
def render_todo_list(todo_list: ResearchTodoList) -> None:
    """Render the research to-do list in Streamlit sidebar."""
    st.sidebar.header(f"Research Progress: {todo_list.progress_percent:.0f}%")
    st.sidebar.progress(todo_list.progress_percent / 100)

    for task in todo_list.tasks:
        icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "blocked": "❌"}
        with st.sidebar.expander(f"{icon[task.status]} {task.description}"):
            # Show iteration count if looping
            if task.current_iteration > 1:
                st.write(f"Iteration: {task.current_iteration}")

            # Show query count
            if task.query_sets:
                latest = task.query_sets[-1]
                st.write(f"Queries: {len(latest.vector_queries)} vector, "
                         f"{len(latest.doc_keywords)} doc keywords")

            # Show summaries if available
            for summary in task.summaries:
                st.write(f"**{summary.category}**: {len(summary.key_findings)} findings")
```

**Tool Execution Flow (per todo-item):**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Per Todo-Item Loop                                │
│                                                                      │
│   generate_queries ──► research ──► summarize ──► critique          │
│         ▲                                            │               │
│         │                                            ▼               │
│         │                                  ┌─────────────────┐       │
│         │                                  │ Gaps found AND  │       │
│         │                                  │ queries < MAX?  │       │
│         │                                  └────────┬────────┘       │
│         │                                     yes   │   no           │
│         └───────────────────────────────────────────┘    │           │
│                                                          ▼           │
│                                              Mark todo-item DONE     │
│                                              → Next todo-item        │
└─────────────────────────────────────────────────────────────────────┘
```

**Critique-Driven Query Generation:**

The critique tool can spawn new queries focusing on relationships:

```python
def handle_critique_result(
    critique: CritiqueResult,
    task: ResearchTask,
    query_context: QueryContext,
    llm: BaseLLM,
) -> bool:
    """Handle critique result - spawn new queries or mark done.

    Returns True if should continue (loop back to generate_queries).
    """
    if not critique.should_continue:
        return False  # No gaps or at query limit

    # Generate new QuerySet focusing on identified gaps
    new_query_set = QuerySet(
        todo_item_id=task.id,
        vector_queries=generate_gap_focused_queries(
            critique.gaps_found,
            critique.cross_document_refs,
            query_context,
            llm,
        ),
        doc_keywords=extract_keywords_from_refs(critique.cross_document_refs),
        web_queries=[] if not web_enabled else generate_web_queries(...),
        iteration=task.current_iteration + 1,
        generated_from_critique=True,
    )

    task.query_sets.append(new_query_set)
    task.current_iteration += 1
    return True  # Continue loop
```

### Quality Checker (MUST-HAVE)

The Quality Checker validates generated answers before final output:

**Scoring Dimensions (0-400 total):**
- **Factual Accuracy** (0-100): Claims match source content
- **Semantic Validity** (0-100): Logical coherence and relevance
- **Structural Integrity** (0-100): Proper formatting and organization
- **Citation Correctness** (0-100): All claims have `[Source]` references

```python
class QualityAssessment(BaseModel):
    """Quality check result."""
    overall_score: int  # 0-400
    factual_accuracy: int  # 0-100
    semantic_validity: int  # 0-100
    structural_integrity: int  # 0-100
    citation_correctness: int  # 0-100
    passes_quality: bool  # True if overall_score >= threshold
    issues_found: list[str]
    improvement_suggestions: list[str]

# Quality router logic
MAX_REFLECTIONS = 1  # Prevent infinite loops

def quality_router(assessment: QualityAssessment, reflection_count: int) -> str:
    """Route based on quality check result.

    Returns:
        'source_linker' if passes OR max reflections reached
        'generate_answer' if needs improvement AND within reflection limit
    """
    if assessment.passes_quality or reflection_count >= MAX_REFLECTIONS:
        return "source_linker"
    return "generate_answer"
```

**Workflow Integration:**
1. Answer generated with `[Source_filename]` citations
2. Quality checker evaluates across 4 dimensions
3. If fails and reflections < MAX_REFLECTIONS → regenerate answer
4. If passes OR max reflections reached → proceed to source linker

### Source Linker (MUST-HAVE)

The Source Linker converts text citations to clickable PDF references:

**Citation Format:**
- All facts MUST include `[Source_filename.pdf]` citation
- Example: "The building height is 25m [003_EG_025_K1.pdf]"

**Linkification Process:**
```python
class LinkedResponse(BaseModel):
    """Response with linkified sources."""
    raw_answer: str  # Original with [filename] references
    linked_answer: str  # HTML with clickable links
    sources_linked: list[str]  # List of resolved source paths

def linkify_sources(answer: str, source_dir: str) -> LinkedResponse:
    """Convert [filename] references to clickable HTML links.

    1. Find all [filename.pdf] patterns in answer
    2. Resolve each filename to full path in source_dir
    3. Generate HTML anchor tags or base64-embedded links
    4. Return both raw and linked versions
    """

def resolve_source_directory(collection_name: str) -> str:
    """Map ChromaDB collection to source PDF folder.

    Example:
        'GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000'
        → 'kb/insert_data/GLageKon/'
    """
```

**Citation Requirements (enforced by Quality Checker):**
- Preserve original wording from sources
- Include exact figures, numbers, statistics only from sources
- Use direct quotes for key definitions
- All claims require proper `[Source]` attribution
- No external knowledge should augment source information

---

## Data Models

### Query Models

```python
class ResearchQuery(BaseModel):
    """User's research query with metadata."""
    question: str
    collections: list[str] = []  # ChromaDB collections to search
    doc_filter: str | None = None  # PDF filename pattern
    max_sources: int = 10

class QueryPlan(BaseModel):
    """Agent's decomposed search plan."""
    sub_queries: list[str]
    search_strategy: Literal["vector_first", "doc_first", "parallel"]
    estimated_steps: int
```

### Research Result Models (Phase 2 Outputs)

```python
class VectorResult(BaseModel):
    """Result from Vector DB search."""
    doc_id: str
    doc_name: str
    chunk_text: str
    page_number: int | None
    relevance_score: float
    collection: str
    query_used: str  # Which query produced this result

class DocumentFinding(BaseModel):
    """Finding from Document Research Agent."""
    doc_path: str
    doc_name: str
    passage: str                        # Extracted text (up to DOC_WORD_LIMIT)
    page_numbers: list[int]
    keywords_matched: list[str]
    relevance_score: float              # Weighted average score
    references_found: list[str]         # Other docs referenced in this passage
    search_chain: list[str]             # How we got here (for tracing)

class WebResult(BaseModel):
    """Result from web search."""
    title: str
    url: str
    snippet: str
    content: str | None
    query_used: str

class ResearchResults(BaseModel):
    """Combined results from all research sources."""
    todo_item_id: str
    vector_results: list[VectorResult]
    doc_results: list[DocumentFinding]
    web_results: list[WebResult]
    total_sources: int
```

### Response Models (MUST-HAVE: Structured JSON Output)

```python
class Source(BaseModel):
    """A source document or chunk."""
    doc_id: str
    doc_name: str
    chunk_text: str
    page_number: int | None
    relevance_score: float
    collection: str | None  # ChromaDB collection if from vector search
    category: Literal["vector_db", "documents", "web"]

class LinkedSource(BaseModel):
    """Source with resolved path for linking."""
    source: Source
    resolved_path: str                  # Full path to PDF
    link_html: str                      # Clickable HTML link

class Finding(BaseModel):
    """A discrete finding from the research."""
    claim: str
    evidence: str
    sources: list[Source]
    source_quotes: dict[str, str]       # source_id → exact quote used
    confidence: Literal["high", "medium", "low"]

class RankedFindings(BaseModel):
    """Reranked findings from all todo-items."""
    findings: list[Finding]
    unified_sources: list[Source]       # Deduplicated source list
    relevance_scores: dict[str, float]  # finding_id → score vs original query

class DraftAnswer(BaseModel):
    """Answer before quality check."""
    answer_text: str                    # With [Source_filename.pdf] citations
    findings_used: list[str]            # Finding IDs referenced
    citation_map: dict[str, str]        # citation_key → source_id

class FinalReport(BaseModel):
    """Final structured output."""
    query: str
    answer: str                         # With clickable source links
    findings: list[Finding]             # Discrete findings with sources
    sources: list[LinkedSource]         # All sources with paths
    quality_score: int                  # 0-400
    quality_breakdown: dict[str, int]   # Per-dimension scores
    todo_items_completed: int
    research_iterations: int            # Total across all tasks
    metadata: dict                      # Timing, tokens, etc.
```

### Document Models

```python
class PDFMetadata(BaseModel):
    """PDF document metadata."""
    filename: str
    path: str
    page_count: int
    file_size_bytes: int

class DocumentChunk(BaseModel):
    """A chunk of text from a PDF."""
    text: str
    page: int
    bbox: tuple[float, float, float, float] | None
    metadata: PDFMetadata
```

### Folder Researcher Models

```python
class FolderStructure(BaseModel):
    """Tree structure of document folder."""
    path: str
    file_count: int
    total_size_bytes: int
    files: list[PDFMetadata]
    subfolders: list["FolderStructure"] = []

class DocumentPreview(BaseModel):
    """Quick preview of a document."""
    metadata: PDFMetadata
    toc: list[str] | None  # Table of contents if available
    first_pages_text: str  # Text from first N pages
    summary: str | None  # LLM-generated summary if requested

class ParsedDocument(BaseModel):
    """Fully parsed document content."""
    metadata: PDFMetadata
    pages: list[str]  # Text content per page
    tables: list[dict] | None  # Extracted tables if any
    sections: dict[str, str] | None  # Section headings mapped to content

class GrepMatch(BaseModel):
    """A search match in documents."""
    doc_path: str
    doc_name: str
    page: int
    line_number: int
    context: str  # Surrounding text for context
    match: str  # The matched text
    score: float = 1.0  # Relevance score (1.0 for exact match)
```

### Web Search Models

```python
class WebSearchResult(BaseModel):
    """Web search result from Tavily or similar API."""
    title: str
    url: str
    snippet: str  # Brief excerpt
    content: str | None  # Full page content if available
    source: str = "web"  # Distinguish from local sources
```

---

## Data Sources

### PDF Document Corpus

- **Location**: `kb/insert_data/`
- **Count**: 761 PDF documents
- **Language**: German
- **Domain**: Technical/regulatory documentation (construction, engineering, nuclear safety)
- **Naming**: Sequential numbering with descriptive names (e.g., `003_EG_025_K1,_Verw.-_u._Sozialgebäude.pdf`)

### ChromaDB Collections

| Collection | Chunk Size | Overlap | Documents | Purpose |
|------------|------------|---------|-----------|---------|
| GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000 | 10000 | 2000 | ~450 | General layout/construction |
| NORM__Qwen--Qwen3-Embedding-0.6B--3000--600 | 3000 | 600 | ~50 | Standards/norms |
| StrlSch__Qwen--Qwen3-Embedding-0.6B--3000--600 | 3000 | 600 | ~100 | Radiation protection |
| StrlSchExt__Qwen--Qwen3-Embedding-0.6B--3000--600 | 3000 | 600 | ~150 | Extended radiation protection |

**Naming Convention**: `{category}__{embedding_model}--{chunk_size}--{overlap}`

**Embedding Model**: Qwen3-Embedding-0.6B (via Ollama)

---

## Key Commands

### Development

```bash
# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run Streamlit app (port >8510 per global config)
streamlit run src/ui/app.py --server.port 8511

# Run with auto-reload during development
streamlit run src/ui/app.py --server.port 8511 --server.runOnSave true
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_agents.py -v
```

### Ollama

```bash
# Pull required models
ollama pull qwen3:14b
ollama pull llama3.1:8b

# Check running models
ollama list

# Run embedding model
ollama pull qwen3:0.6b  # For embeddings
```

### ChromaDB Management

```bash
# List collections (Python one-liner)
python -c "import chromadb; c=chromadb.PersistentClient('kb/database/GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000'); print([col.name for col in c.list_collections()])"
```

---

## Implementation Phases

### Impl Phase 1: Core Infrastructure
- [ ] Project setup: pyproject.toml, directory structure
- [ ] Config management with pydantic-settings (all new env vars)
- [ ] Pydantic models: QueryContext, QuerySet, CategorySummary, CritiqueResult
- [ ] Pydantic models: ResearchTask, ResearchTodoList, DocumentFinding
- [ ] Pydantic models: VectorResult, WebResult, ResearchResults
- [ ] Pydantic models: FinalReport, LinkedSource, RankedFindings
- [ ] ChromaDB client service
- [ ] Ollama LLM client service
- [ ] PDF reader service (PyMuPDF)
- [ ] Basic tests for services and models

### Impl Phase 2: HITL + To-Do List (Research Phase 1)
- [ ] HITL conversational interface for query refinement
- [ ] To-Do List generation (3-5 tasks, no queries yet)
- [ ] HITL checkpoint for task approval/modification
- [ ] QueryContext accumulation from HITL conversation
- [ ] Tests for HITL flow

### Impl Phase 3: ReAct Agent Tools (Research Phase 2)
- [ ] `generate_queries` tool implementation
- [ ] `research` tool with parallel execution
- [ ] Vector DB search component (MAX_VDB_ITEMS per query)
- [ ] Document Research Agent (sub-agent with 6 tools)
  - [ ] scan_folder, preview, parse, read, grep, glob
  - [ ] Reference tracking across documents (DOC_REFERENCE_HOPS)
  - [ ] Relevance scoring (weighted keyword density + doc frequency)
- [ ] Web search component (if enabled)
- [ ] `summarize` tool (per-category, source preservation)
- [ ] `critique` tool (gap analysis, cross-document/cross-todo refs)
- [ ] Critique → generate_queries loop logic
- [ ] Tests for each tool and the loop

### Impl Phase 4: Report Generation (Research Phase 3)
- [ ] Rerank: cross-todo-item deduplication, relevance scoring
- [ ] GenAnswer: synthesis with [Source_filename] citations
- [ ] Quality Checker (0-400 scoring, reflection loop)
- [ ] Source Linker (resolve paths, generate clickable links)
- [ ] FinalReport assembly
- [ ] Tests for report generation pipeline

### Impl Phase 5: Streamlit UI
- [ ] Basic app layout with query input
- [ ] HITL panel (conversational refinement)
- [ ] To-Do List component (real-time updates, loop visualization)
- [ ] Results view component (linked sources)
- [ ] Session state management
- [ ] Safe exit button (find port dynamically, kill process)
- [ ] Source inspection view

### Impl Phase 6: Polish & Advanced Features
- [ ] Multi-collection search
- [ ] Query history and caching
- [ ] Export results (JSON, Markdown)
- [ ] Performance optimizations
- [ ] Error handling and recovery
- [ ] Logging and observability

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
- **Structured JSON output** from all LLM calls (MUST-HAVE)
- **Docstrings** on public functions (Google style)

### Example Function

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
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
    # Implementation...
```

---

## Environment Setup

### Prerequisites

1. **Python 3.11+** installed
2. **uv** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Ollama** installed and running: [ollama.ai](https://ollama.ai)

### Installation

```bash
# Clone repository
cd /path/to/KB_BS_local-hybrid-researcher

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Pull Ollama models
ollama pull qwen3:14b
ollama pull qwen3:0.6b
```

### Configuration (.env)

```ini
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
OLLAMA_EMBEDDING_MODEL=qwen3:0.6b

# ChromaDB
CHROMADB_PATH=./kb/database

# Streamlit
STREAMLIT_PORT=8511

# Logging
LOG_LEVEL=INFO

# Optional Features
ENABLE_WEB_SEARCH=false          # Enable Tavily web search
TAVILY_API_KEY=                  # Required if web search enabled

# Quality Checker
ENABLE_QUALITY_CHECKER=true      # Enable answer quality validation
QUALITY_THRESHOLD=300            # Minimum score (0-400) to pass
MAX_REFLECTIONS=1                # Max regeneration attempts on failure

# Phase 1: To-Do List
MAX_TODO_ITEMS=10                # Maximum research tasks (prevents runaway)
INITIAL_TODO_ITEMS=5             # Initial tasks created (3-5 typical)

# Phase 2: Query Generation
MAX_QUERY_ITEMS=5                # Max queries per todo-item per category
MIN_QUERY_ITEMS=3                # Minimum queries to generate

# Phase 2: Research Limits
MAX_VDB_ITEMS=10                 # Vector DB results per query
MAX_DOCS=5                       # Documents to fully analyze per todo-item
DOC_WORD_LIMIT=5000              # Max words extracted per document
DOC_REFERENCE_HOPS=2             # Max hops following document references

# Phase 2: Document Relevance Scoring
KEYWORD_DENSITY_WEIGHT=0.7       # Weight for keyword density in scoring
DOC_FREQUENCY_WEIGHT=0.3         # Weight for document frequency in scoring

# Phase 2: Web Search
WEB_RESULTS_PER_QUERY=2          # Web results per query (if enabled)
```

---

## pyproject.toml

```toml
[project]
name = "kb-bs-local-hybrid-researcher"
version = "0.1.0"
description = "AI-LLM agentic researcher with hybrid document and vector search"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Your Name", email = "you@example.com" }
]
dependencies = [
    "langchain>=0.3.0",
    "langchain-ollama>=0.2.0",
    "langchain-chroma>=0.1.0",
    "langchain-community>=0.3.0",
    "chromadb>=0.5.0",
    "streamlit>=1.28.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pymupdf>=1.24.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[project.scripts]
researcher = "src.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## Reference Implementations

| # | Reference | URL | Extract |
|---|-----------|-----|---------|
| 1 | Local vectorstore researcher | [KB_BS_local-rag-he](https://github.com/ToHeinAC/KB_BS_local-rag-he) | GUI patterns, ChromaDB integration, HITL implementation |
| 2 | Vectorstore | `kb/` folder | Pre-populated ChromaDB instances, embedding configuration |
| 3 | ReAct Agent | [deepagents](https://github.com/langchain-ai/deepagents), [deepagents_ollama](https://github.com/ToHeinAC/deepagents_ollama) | ReAct pattern, local Ollama integration |
| 4 | Document File Search | [agentic-file-search](https://github.com/PromtEngineer/agentic-file-search), [YouTube](https://www.youtube.com/watch?v=rMADSuus6jg) | Dynamic document navigation, section extraction |
| 5 | Human-In-The-Loop | [KB_BS_local-rag-he](https://github.com/ToHeinAC/KB_BS_local-rag-he) | **MUST-HAVE**: User validation checkpoints |
| 6 | To-Do List | See HITL reference | **MUST-HAVE**: Task tracking after HITL start |
| 7 | Structured Outputs | LangChain docs | **MUST-HAVE**: All LLM outputs as JSON |

---

## MUST-HAVE Requirements Summary

1. **Human-In-The-Loop**: User validation at key decision points
2. **To-Do List Approach**: Task tracking visible to user after research starts
3. **Structured JSON Outputs**: All LLM responses use Pydantic models
4. **Fully Local**: No external API calls (Ollama + ChromaDB)
5. **Safe Exit**: Streamlit button to cleanly terminate (port-aware)
