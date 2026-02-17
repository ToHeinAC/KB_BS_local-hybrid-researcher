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

**Termination Conditions** (all paths sync `hitl_conversation_history` to agent state):
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
- `query_anchor`: Immutable reference to original intent (created in `hitl_finalize` for graph-based HITL, or in `_start_research_from_hitl` for chat-based HITL)
- `hitl_smry`: Citation-aware HITL summary (generated in `hitl_finalize` or `_start_research_from_hitl`)
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
| Structured Output | `llm.with_structured_output(Model, method="json_mode")` via SystemMessage/HumanMessage |
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
│   ├── references.md      # External resources
│   └── prompts-design.md  # Prompt design and management
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


## Prompt Management
**All LLM prompts MUST be defined in `src/prompts.py` @src/prompts.py**.
Every prompt is split into a `_SYSTEM` / `_HUMAN` pair (e.g. `TODO_GENERATION_PROMPT_SYSTEM` + `TODO_GENERATION_PROMPT_HUMAN`).
All callers use `OllamaClient.generate_structured_messages()` or `generate_messages()` with separate system/human arguments.
For specific prompt rules, see @docs/prompts-design.md [docs/prompts-design.md](docs/prompts-design.md).

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | @docs/architecture.md Full architecture diagram, state objects, data flow |
| [docs/agent-design.md](docs/agent-design.md) | @docs/agent-design.md ReAct+LangGraph patterns, tools |
| [docs/data-models.md](docs/data-models.md) | @docs/data-models.md All Pydantic models with JSON schemas |
| [docs/data-sources.md](docs/data-sources.md) | @docs/data-sources.md PDF corpus, ChromaDB collections, embeddings |
| [docs/configuration.md](docs/configuration.md) | @docs/configuration.md Environment variables, pyproject.toml |
| [docs/implementation.md](docs/implementation.md) | @docs/implementation.md Implementation phases, coding standards |
| [docs/rabbithole-magic.md](docs/rabbithole-magic.md) | @docs/rabbithole-magic.md Deep reference-following algorithm |
| [docs/references.md](docs/references.md) | @docs/references.md External repos, LangGraph docs, examples |
| [docs/prompts-design.md](docs/prompts-design.md) | @docs/prompts-design.md Prompt design and management |

## Implementation Status

See @docs/implementation.md [docs/implementation.md](docs/implementation.md)