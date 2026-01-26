# Rabbithole-Agent: Local Hybrid Researcher

A fully local, privacy-first research system that performs **deep reference-following** across document collections using Ollama LLMs, ChromaDB, and LangGraph.

## Core Problem

Classical RAG lacks deep contextual understanding and cannot follow inter-document relationships. This agent solves it by iteratively "digging into the rabbithole" - following references, building context, and discovering document interconnections.

## Architecture (5 Phases)

```
┌────────────────────────────────────────────────────────────────────┐
│  Phase 1: Query Analysis + HITL                                     │
│  User Query → NER/Keywords → Clarification Loop → QueryAnalysis     │
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

## Tech Stack (LangChain v1.0+)

| Component | Technology |
|-----------|------------|
| Framework | LangChain v1.0+, LangGraph v1.0+ |
| LLM | Ollama (qwen3:14b, llama3.1:8b fallback) |
| Embeddings | Qwen3-Embedding-0.6B via Ollama |
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
cp .env.example .env

# Pull models
ollama pull qwen3:14b
ollama pull qwen3:0.6b

# Run
streamlit run src/ui/app.py --server.port 8511
```

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

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Full architecture diagram, state objects, data flow |
| [docs/agent-design.md](docs/agent-design.md) | ReAct+LangGraph patterns, Rabbithole Magic, tools |
| [docs/data-models.md](docs/data-models.md) | All Pydantic models with JSON schemas |
| [docs/data-sources.md](docs/data-sources.md) | PDF corpus, ChromaDB collections, embeddings |
| [docs/configuration.md](docs/configuration.md) | Environment variables, pyproject.toml |
| [docs/implementation.md](docs/implementation.md) | Implementation phases, coding standards |
| [docs/references.md](docs/references.md) | External repos, LangGraph docs, examples |
