# Rabbithole-Agent: Local Hybrid Researcher

A fully local, privacy-first research system that performs **deep reference-following** across document collections using Ollama LLMs, ChromaDB, and LangGraph.

## Core Problem

Classical RAG lacks deep contextual understanding and cannot follow inter-document relationships. This agent solves it by iteratively "digging into the rabbithole" - following references, building context, and discovering document interconnections.

## Architecture (5 Phases)

```
┌────────────────────────────────────────────────────────────────────┐
│  Phase 1: Enhanced Query Analysis + Iterative HITL                  │
│  User Query → Language Detection → Iterative Clarification Loop     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  hitl_init → hitl_generate_questions ↔ hitl_process_response │  │
│  │  → hitl_finalize (on /end, max_iterations, or convergence)   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Output: research_queries[], coverage_score, query_analysis         │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2: Research Planning                                         │
│  QueryAnalysis → ToDoList (3-5 tasks, max 15)                       │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3: Deep Context Extraction (Rabbithole Magic)                │
│  For each task:                                                      │
│    Vector Search → Extract Info → Detect Refs → Follow Refs         │
│    → Filter by Relevance → Update ToDoList → Loop until done        │
├────────────────────────────────────────────────────────────────────┤
│  Phase 4: Synthesis + Quality Assurance                             │
│  Summaries → Rerank → Quality Check → Gap Analysis → Report         │
├────────────────────────────────────────────────────────────────────┤
│  Phase 5: Source Attribution                                        │
│  Add citations → Resolve paths → Generate clickable links           │
└────────────────────────────────────────────────────────────────────┘
```

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
- `query_retrieval`: Accumulated retrieval text (persists after HITL, not used in Phase 2+)

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
- `QUALITY_THRESHOLD=300`: Minimum quality score (0-400)

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
6. **Reference Following**: Deep rabbithole traversal with relevance filtering

## Coding Standards

### Prompt Management
- **All LLM prompts MUST be defined in `src/prompts.py`**
- Never inline prompt strings in node functions or services
- Use template variables for dynamic content (e.g., `{query}`, `{context}`)
- Group prompts by category (HITL, Research, Quality)

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
- [x] **Graph Entry Point Enhancement**: Supports HITL resume via `hitl_process_response` routing
- [x] **Coverage Metrics in Checkpoints**: Knowledge gaps and dedup ratios shown in UI

### Deferred to Week 3+
- [ ] Progressive disclosure / knowledge pyramid
- [ ] Three-tier memory architecture
- [ ] Orchestrator-worker parallelization
- [ ] RAG Triad automated validation
- [ ] CI/CD integration
- [ ] Security hardening (PII redaction)
