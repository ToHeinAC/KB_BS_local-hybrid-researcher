# Data Models

All data structures use Pydantic v2 for validation and JSON schema generation.

## LangChain v1.0 Compatibility Note

**Agent State vs Data Models:**
- **Agent state** (in LangGraph): Must be `TypedDict` (LangChain v1.0 requirement)
- **Data models** (for tools, validation, serialization): Use Pydantic as shown below

Pydantic models are serialized to/from dicts when stored in agent state:
```python
# Store in state
state["query_analysis"] = analysis.model_dump()

# Retrieve from state
analysis = QueryAnalysis.model_validate(state["query_analysis"])
```

## Core State Models

### QueryAnalysis

```python
class QueryAnalysis(BaseModel):
    """Extracted analysis of user's research query."""
    original_query: str
    key_concepts: list[str]
    entities: list[str]
    scope: str
    assumed_context: list[str]
    clarification_needed: bool
    hitl_refinements: list[str] = []
    detected_language: str = "de"
```

### ToDoList Models

```python
class ToDoItem(BaseModel):
    """A single research task."""
    id: int
    task: str
    context: str
    completed: bool = False
    subtasks: list[str] = []

class ToDoList(BaseModel):
    """Research task tracker (mutable during Phase 2-3)."""
    items: list[ToDoItem]
    max_items: int = 15  # TODO_MAX_ITEMS
    current_item_id: int | None = None
```

### ResearchContext

```python
class NestedChunk(BaseModel):
    """Chunk retrieved from following a reference."""
    chunk: str
    document: str
    extracted_info: str
    relevance_score: float


class PreservedQuote(BaseModel):
    """A verbatim quote preserved for legal/technical precision (NEW)."""
    quote: str                    # Exact verbatim text
    relevance_reason: str = ""    # Why this must be preserved verbatim
    source: str = ""              # Source document name
    page: int = 0                 # Page number if available


class InfoExtractionWithQuotes(BaseModel):
    """Result of info extraction with preserved quotes (NEW)."""
    extracted_info: str           # Condensed relevant information
    preserved_quotes: list[PreservedQuote] = []  # Critical verbatim quotes


class ExtractedReference(BaseModel):
    """Reference extracted by LLM (structured output model)."""
    reference_mention: str           # Exact text as it appears
    reference_type: Literal["legal_section", "academic_numbered",
                            "academic_shortform", "document_mention"]
    target_document_hint: str = ""   # Best guess at target document name
    confidence: float = 0.9          # 0.0 to 1.0

class ExtractedReferenceList(BaseModel):
    """Container for LLM extraction output."""
    references: list[ExtractedReference] = []

class DetectedReference(BaseModel):
    """A reference detected within text."""
    type: Literal["section", "document", "external",
                   "legal_section", "academic_numbered",
                   "academic_shortform", "document_mention"]
    target: str
    original_text: str = ""
    found: bool = False
    nested_chunks: list[NestedChunk] = []
    document_context: str | None = None   # Resolved document name hint (NEW)
    extraction_method: Literal["regex", "llm"] = "regex"  # How detected (NEW)

class ChunkWithInfo(BaseModel):
    """A chunk from vector DB with extracted info and references."""
    chunk: str
    document: str
    page: int | None = None
    extracted_info: str | None = None
    references: list[DetectedReference] = []
    relevance_score: float = 0.0

class SearchQueryResult(BaseModel):
    """Results for a single search query."""
    query: str
    chunks: list[ChunkWithInfo]
    summary: str | None = None
    summary_references: list[str] = []
    quality_score: float | None = None
    web_search_results: list[WebResult] | None = None

class ResearchContextMetadata(BaseModel):
    """Metadata about the research process."""
    total_iterations: int = 0
    documents_referenced: list[str] = []
    external_sources_used: bool = False
    visited_refs: list[str] = []

class ResearchContext(BaseModel):
    """Growing JSON accumulating all research findings."""
    search_queries: list[SearchQueryResult] = []
    metadata: ResearchContextMetadata = ResearchContextMetadata()
```

## Research Task Models

### QueryContext

```python
class QueryContext(BaseModel):
    """Accumulated context for query generation from original query + HITL."""
    original_query: str
    hitl_conversation: list[str]
    user_feedback_analysis: str | None = None
    detected_language: str = "de"
```

### QuerySet

```python
class QuerySet(BaseModel):
    """Queries generated for a todo-item."""
    todo_item_id: str
    vector_queries: list[str]      # 3-5 queries
    doc_keywords: list[str]        # 3x multiplier for doc search
    web_queries: list[str] = []    # If web search enabled
    iteration: int = 1
    generated_from_critique: bool = False
```

### ResearchTask

```python
class ResearchTask(BaseModel):
    """A single research task with full tracking."""
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "blocked"]

    # Query tracking
    query_sets: list[QuerySet] = []
    current_iteration: int = 0

    # Research results
    vector_results: list[VectorResult] = []
    doc_results: list[DocumentFinding] = []
    web_results: list[WebResult] = []

    # Summaries
    summaries: list[CategorySummary] = []

    # Critique tracking
    critiques: list[CritiqueResult] = []

    # Source tracking
    all_sources: list[Source] = []

    # Lineage
    spawned_from: str | None = None
    error_message: str | None = None
```

## Search Result Models

### VectorResult

```python
class VectorResult(BaseModel):
    """Result from ChromaDB vector search."""
    doc_id: str
    doc_name: str
    chunk_text: str
    page_number: int | None
    relevance_score: float
    collection: str
    query_used: str
```

### DocumentFinding

```python
class DocumentFinding(BaseModel):
    """Finding from document navigation agent."""
    doc_path: str
    doc_name: str
    passage: str
    page_numbers: list[int]
    keywords_matched: list[str]
    relevance_score: float
    references_found: list[str]
    search_chain: list[str]  # How we got here
```

### WebResult

```python
class WebResult(BaseModel):
    """Result from web search."""
    title: str
    url: str
    snippet: str
    content: str | None = None
    query_used: str
    source: str = "web"
```

## Summary Models

### CategorySummary

```python
class CategorySummary(BaseModel):
    """Summary for one research category."""
    category: Literal["vector_db", "documents", "web"]
    summary_text: str
    key_findings: list[str]
    sources_used: list[Source]
    source_quotes: dict[str, str]  # source_id → quote
    todo_item_id: str
    query_context: str
```

### CritiqueResult

```python
class CritiqueResult(BaseModel):
    """Gap analysis result."""
    todo_item_id: str
    gaps_found: list[str]
    severity: Literal["none", "minor", "significant"]
    suggested_queries: list[str]
    cross_document_refs: list[str]
    cross_todo_refs: list[str]
    should_continue: bool
```

## Quality Models

### QualityAssessment

```python
class QualityAssessment(BaseModel):
    """Quality check result with 5-dimension scoring."""
    overall_score: int  # 0-500
    factual_accuracy: int  # 0-100
    semantic_validity: int  # 0-100
    structural_integrity: int  # 0-100
    citation_correctness: int  # 0-100
    query_relevance: int  # 0-100 (NEW)
    passes_quality: bool
    issues_found: list[str]
    improvement_suggestions: list[str]
```

## Source Models

### Source

```python
class Source(BaseModel):
    """A source document or chunk."""
    doc_id: str
    doc_name: str
    chunk_text: str
    page_number: int | None
    relevance_score: float
    collection: str | None = None
    category: Literal["vector_db", "documents", "web"]
```

### LinkedSource

```python
class LinkedSource(BaseModel):
    """Source with resolved path for linking."""
    source: Source
    resolved_path: str
    link_html: str
```

## Report Models

### Finding

```python
class Finding(BaseModel):
    """A discrete finding from research."""
    claim: str
    evidence: str
    sources: list[Source]
    source_quotes: dict[str, str]
    confidence: Literal["high", "medium", "low"]
```

### FinalReport

```python
class FinalReport(BaseModel):
    """Final structured output."""
    query: str
    answer: str  # With clickable source links
    findings: list[Finding]
    sources: list[LinkedSource]
    quality_score: int
    quality_breakdown: dict[str, int]
    todo_items_completed: int
    research_iterations: int
    metadata: dict
```

## Document Models

### SynthesisOutputEnhanced (NEW)

```python
class SynthesisOutputEnhanced(BaseModel):
    """Enhanced synthesis output with query coverage tracking."""
    summary: str                  # Comprehensive answer in target language
    key_findings: list[str] = []  # List of most important findings
    query_coverage: int = 50      # How completely query was answered (0-100)
    remaining_gaps: list[str] = [] # Unanswered aspects of the query
```

### TaskSearchQueries (NEW)

```python
class TaskSearchQueries(BaseModel):
    """LLM output: 2 dedicated search queries for a task."""
    query_1: str   # First search query focusing on core aspects
    query_2: str   # Second search query exploring related/complementary angle
```

### TaskSummaryOutput (NEW)

```python
class TaskSummaryOutput(BaseModel):
    """Output for per-task structured summary."""
    summary: str                           # Concise task summary
    key_findings: list[str] = []           # List of discrete findings
    gaps: list[str] = []                   # Identified gaps or limitations
    relevance_assessment: str = ""         # Whether findings match query intent
    irrelevant_findings: list[str] = []    # Findings superficially related but not answering the query
```

### RelevanceScoreOutput (NEW)

```python
class RelevanceScoreOutput(BaseModel):
    """Output for relevance scoring."""
    relevance_score: int          # Relevance score 0-100
    reasoning: str = ""           # Brief explanation of the score
```

### PDFMetadata

```python
class PDFMetadata(BaseModel):
    """PDF document metadata."""
    filename: str
    path: str
    page_count: int
    file_size_bytes: int
```

### GrepMatch

```python
class GrepMatch(BaseModel):
    """A search match in documents."""
    doc_path: str
    doc_name: str
    page: int
    line_number: int
    context: str
    match: str
    score: float = 1.0
```

## HITL Models

### HITLCheckpoint

```python
class HITLCheckpoint(BaseModel):
    """Checkpoint requiring user validation."""
    checkpoint_type: Literal["query", "strategy", "findings", "sources", "iterative_hitl"]
    content: dict
    requires_approval: bool = True

class HITLDecision(BaseModel):
    """User's decision at HITL checkpoint."""
    approved: bool
    modifications: dict | None = None
    feedback: str | None = None
```

## Enhanced Phase 1: Iterative HITL State (NEW)

The following fields are added to AgentState for the iterative HITL flow:

### Iterative HITL State Fields

```python
# In AgentState (TypedDict):

# Chat-style HITL conversation state
hitl_state: dict | None  # {user_query, language, conversation_history, analysis}

# Iteration tracking
hitl_iteration: int           # Current iteration (0-indexed)
hitl_max_iterations: int      # Max iterations (default 5)
hitl_active: bool             # Whether iterative HITL is active
hitl_termination_reason: str | None  # "user_end", "max_iterations", "convergence"

# Conversation history
hitl_conversation_history: list[dict]  # [{role, content, iteration}, ...]

# Enhanced Multi-query retrieval tracking (NEW)
iteration_queries: list[list[str]]   # [[q1, q2, q3], ...] per iteration
knowledge_gaps: list[str]            # Gaps identified from retrieval analysis
retrieval_dedup_ratios: list[float]  # Dedup ratio per iteration for convergence

# Convergence tracking
coverage_score: float         # 0-1 information coverage estimate
retrieval_history: dict       # {"iteration_N": {queries, retrieved_chunks, dedup_stats}}
query_retrieval: str          # Accumulated filtered retrieval results (context)

# Token budget
total_tokens_used: int        # Estimated tokens consumed
max_tokens_allowed: int       # Budget constraint (default 4000)

# HITL handoff (output for Phase 2)
research_queries: list[str]   # Generated search queries
additional_context: str       # Summary from analysis
detected_language: str        # "de" or "en"

# Graded Context Management (NEW)
query_anchor: dict            # Immutable reference to original intent
hitl_context_summary: str     # Synthesized HITL findings for synthesis
primary_context: list[dict]   # Tier 1: Direct, high-relevance findings
secondary_context: list[dict] # Tier 2: Reference-followed, medium-relevance
tertiary_context: list[dict]  # Tier 3: Deep references, HITL retrieval
task_summaries: list[dict]    # Per-task structured summaries
preserved_quotes: list[dict]  # Critical verbatim quotes
```

### Query Anchor Structure (NEW)

Created in `hitl_finalize`, immutable throughout execution:

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
context_entry = {
    "chunk": str,                 # Text content (limited to 2000 chars)
    "document": str,              # Source document name
    "page": int | None,           # Page number
    "extracted_info": str,        # Condensed relevant passages
    "relevance_score": float,     # Original vector search score
    "context_tier": int,          # 1, 2, or 3
    "context_weight": float,      # 0.0-1.0 weight for synthesis
    "depth": int,                 # Recursion depth when found
    "source_type": str,           # "vector_search", "reference", "hitl"
}
```

### Task Summary Structure (NEW)

```python
task_summary = {
    "task_id": int,               # Task ID
    "task_text": str,             # Task description
    "summary": str,               # Generated summary
    "key_findings": list[str],    # Discrete findings
    "gaps": list[str],            # Identified gaps
    "preserved_quotes": list[dict], # Quotes from this task
    "sources": list[str],         # Source documents
    "relevance_to_query": float,  # 0.0-1.0 relevance score
}
```

### Iterative HITL Checkpoint Content

When `checkpoint_type == "iterative_hitl"`:

```python
checkpoint = {
    "checkpoint_type": "iterative_hitl",
    "content": {
        "questions": str,           # Follow-up questions text
        "iteration": int,           # Current iteration number
        "max_iterations": int,      # Maximum iterations allowed
        "analysis": {               # Current understanding
            "entities": list[str],
            "scope": str,
            "context": str,
            "refined_query": str,
        },
    },
    "requires_approval": True,
    "phase": "Phase 1: Query Refinement",
}
```

### Alternative Queries Output

```python
class AlternativeQueriesOutput(BaseModel):
    """LLM output for multi-query generation."""
    broader_scope: str        # Query for broader context
    alternative_angle: str    # Query for different perspective
    rationale: str            # Why these alternatives matter
```

## JSON Schema Example

All models generate JSON schemas automatically:

```python
print(ResearchContext.model_json_schema())
```

Output:
```json
{
  "title": "ResearchContext",
  "type": "object",
  "properties": {
    "search_queries": {
      "type": "array",
      "items": {"$ref": "#/$defs/SearchQueryResult"}
    },
    "metadata": {"$ref": "#/$defs/ResearchContextMetadata"}
  },
  "$defs": {
    "SearchQueryResult": {...},
    "ResearchContextMetadata": {...}
  }
}
```
