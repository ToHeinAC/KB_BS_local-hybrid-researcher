# Implementation

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Project setup: pyproject.toml, directory structure
- [ ] Config management with pydantic-settings
- [ ] Pydantic models: QueryAnalysis, ToDoList, ResearchContext
- [ ] Pydantic models: VectorResult, DocumentFinding, WebResult
- [ ] Pydantic models: QualityAssessment, FinalReport
- [ ] ChromaDB client service
- [ ] Ollama LLM client service (with `json_mode` structured output)
- [ ] PDF reader service (PyMuPDF)
- [ ] Basic tests for services and models

### Phase 2: HITL + ToDoList (Research Phase 1-2)
- [ ] Query analysis with NER/keyword extraction
- [ ] HITL conversational interface for refinement
- [ ] ToDoList generation (3-5 tasks)
- [ ] HITL checkpoint for task approval/modification
- [ ] Tests for HITL flow

### Phase 3: LangGraph Agent (Research Phase 3)
- [ ] LangGraph StateGraph setup with TypedDict state (v1.0 pattern)
- [ ] State serialization helpers (Pydantic <-> dict)
- [ ] `vector_search` tool implementation
- [ ] `extract_references` tool
- [ ] `resolve_reference` tool
- [ ] Reference following with depth tracking
- [ ] Relevance filtering (threshold 0.6)
- [ ] ToDoList re-evaluation after each task
- [ ] Loop prevention (visited refs, max iterations)
- [ ] Tests for agent and tools

### Phase 4: Synthesis + Quality (Research Phase 4)
- [ ] `synthesize` node (LLM synthesis from extracted findings)
- [ ] `quality_check` node (optional, 0-400 scoring)
- [ ] Tests for synthesis + QA

### Phase 5: Source Attribution (Research Phase 5)
- [ ] `attribute_sources` node (FinalReport assembly)
- [ ] Source list generation (linked sources)
- [ ] Tests for attribution

### Phase 6: Streamlit UI
- [ ] Basic app layout with query input
- [ ] HITL panel (clarification questions)
- [ ] ToDoList component (real-time updates)
- [ ] Live progress updates via LangGraph streaming
- [ ] Results view with linked sources
- [ ] Session state management
- [ ] Safe exit button (port-aware kill)
- [ ] Source inspection view

### Phase 7: Polish
- [ ] Multi-collection search
- [ ] Query history and caching
- [ ] Export results (JSON, Markdown)
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
- **Structured JSON output** via `method="json_mode"` for Ollama
- **LangGraph** for agent orchestration (NOT AgentExecutor)
- **Docstrings** on public functions (Google style)

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
| **Infinite reference loops** | Circular cross-references | Maintain `visited_refs` set, track recursion depth |
| **Reference resolution ambiguity** | Multiple matches | Use document_mapping.json, fallback to scoring |
| **Hallucinated references** | LLM invents citations | Strict pattern matching, validate all refs |
| **Over-following tangential refs** | Poor relevance filter | Set threshold=0.6+, manual review |
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
