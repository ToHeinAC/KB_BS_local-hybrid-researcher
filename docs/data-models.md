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

class DetectedReference(BaseModel):
    """A reference detected within text."""
    type: Literal["section", "document", "external"]
    target: str
    original_text: str = ""
    found: bool = False
    nested_chunks: list[NestedChunk] = []

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
    """Quality check result with 4-dimension scoring."""
    overall_score: int  # 0-400
    factual_accuracy: int  # 0-100
    semantic_validity: int  # 0-100
    structural_integrity: int  # 0-100
    citation_correctness: int  # 0-100
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
    checkpoint_type: Literal["query", "strategy", "findings", "sources"]
    content: dict
    requires_approval: bool = True

class HITLDecision(BaseModel):
    """User's decision at HITL checkpoint."""
    approved: bool
    modifications: dict | None = None
    feedback: str | None = None
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
