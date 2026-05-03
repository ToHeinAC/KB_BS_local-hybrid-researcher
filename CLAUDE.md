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
│  Output: research_queries[] (supplementary), query_anchor, hitl_smry │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2.5: Query Assessment (Agentic Gate)                         │
│  assess_query: LLM decides proceed/reject + num_tasks (3-6)         │
│  → Rejected queries → __end__ with explanation (no research run)    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 2: Research Planning                                         │
│  LLM generates ToDoList (num_tasks items) anchored on hitl_smry     │
│  Fallback: research_queries[:num_tasks] if LLM fails               │
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
│  Phase 3.5: Pre-Synthesis Relevance Validation + Backfill           │
│  validate_relevance: Filter drift against query_anchor              │
│  Guarantee min chunks (3 primary, 2 secondary) for transparency     │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3.6: Task Summary Reranking                                  │
│  rerank_task_summaries: Sort summaries by relevance_to_query desc   │
│  Stamps rank int; synthesis prompt weights by [Relevance: N/100]    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3.8: Reference Provenance                                    │
│  Nested chunks carry parent document + surrounding context of ref   │
│  Reranker penalises off-topic parent context (−20-40 pts)           │
│  Formatted findings show [via ref "..."] header for traceability    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3.9: Batch Chunk Reranking                                   │
│  _rerank_task_chunks(): batch LLM scoring (6 chunks/call)           │
│  Precision/recall strategies; raw 1-5 → 0-100 mapping               │
│  Hard-filter below reranker_min_score; cross-batch normalization    │
├────────────────────────────────────────────────────────────────────┤
│  Phase 3.10: Optional Web Search (Tavily API)                       │
│  web_search node: LLM generates query → Tavily REST API → LLM      │
│  summarizes with [Title](URL) citations + contradiction detection   │
│  Strictly separated from KB results; appended in attribute_sources  │
│  Disabled by default; user enables per session via GUI checkbox     │
├────────────────────────────────────────────────────────────────────┤
│  Phase 4: Deep Report Synthesis + Quality Assurance                   │
│  Pre-Digested Task Summaries + HITL Summary → Deep Report            │
│  Language Enforcement → Quality Check → **Agentic Remediation** →   │
│  Re-Synthesis (max 1 retry) OR Accept → Report                      │
├────────────────────────────────────────────────────────────────────┤
│  Phase 5: Source Attribution + Numbered Citations                   │
│  Add citations → numberify_citations() → [N] references + PDF links │
│  PDF served via /_api/pdf Tornado route (pdf_route.py)             │
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
- **Chunk Backfill**: Guarantees minimum chunks per task (3 primary, 2 secondary) even if below relevance threshold; backfilled chunks marked with ⚠️ badge for transparency
- **Task Summary Reranking**: Deterministic sort by `relevance_to_query` before synthesis; `[Rank: N/total]` / `[Relevance: N/100]` headers visible in formatted summaries
- **Reference Provenance**: Nested chunks carry parent document + surrounding context of the reference; reranker penalises off-topic parent context; `[via ref "..."]` header in formatted findings for traceability
- **Batch Chunk Reranking**: `_rerank_task_chunks()` uses batch LLM scoring (~3-4 calls for 20 chunks) with precision/recall strategies, cross-batch normalization, and hard-filtering below `reranker_min_score`
- **Optional Web Search**: Tavily API integration (Phase 3.10) — LLM generates search query from gaps, summarizes results with `[Title](URL)` citations, detects contradictions against KB findings; strictly separated from KB synthesis; disabled by default, user-enabled per session
- **Language Enforcement**: Strict single-language output with retry on mismatch
- **Numbered Citations**: `numberify_citations()` replaces inline `[Doc.pdf, Page N]` with sequential `[1]`, `[2]`, … + appended reference list with PDF links served via `/_api/pdf` Tornado route

### Agentic Decision Points

Three LLM-driven gates where the orchestrator is non-deterministic:

0. **Query Assessment Gate** (Phase 2.5): `assess_query` → `QueryAssessmentDecision(proceed, num_tasks, reason, explanation)`. Reject → `__end__` with FinalReport; approve → `generate_todo` with `num_tasks` (3-6). Fallback: `proceed=True, num_tasks=5`.

1. **Reference Following Gate** (Phase 3): Per-reference LLM evaluation via `REFERENCE_DECISION_PROMPT` → `ReferenceDecision(follow, reason)`. Context: `original_query`, `key_entities`, `scope`, `current_task`. Biased toward following when uncertain. Falls back to follow on error.

2. **Quality Remediation Loop** (Phase 4): If quality < 375, LLM decides accept/retry via `QualityRemediationDecision`. Max 1 retry; `quality_remediation_focus` appended to synthesis prompt on retry.

### Enhanced Phase 1: Iterative HITL with Multi-Vector Retrieval

The enhanced iterative HITL system provides intelligent query refinement through conversation **with integrated vector DB retrieval at each iteration**:

```
┌──────────────────────────────────────────────────────────────────┐
│  hitl_init → hitl_generate_queries → hitl_retrieve_chunks →      │
│  hitl_analyze_retrieval → hitl_generate_questions → [wait] →     │
│  hitl_process_response → [loop back or hitl_finalize]           │
└──────────────────────────────────────────────────────────────────┘
```

**Nodes**: `hitl_init` → `hitl_generate_queries` (3/iteration) → `hitl_retrieve_chunks` (deduplicated) → `hitl_analyze_retrieval` (coverage_score, gaps) → `hitl_generate_questions` → `hitl_process_response` → `hitl_finalize`. Full node specs in [docs/architecture.md](docs/architecture.md).

**Termination**: `/end` → `user_end`; max 5 iterations → `max_iterations`; coverage ≥ 0.8 AND dedup ≥ 0.7 AND gaps ≤ 2 → `convergence`.

**Graded Context State Fields**:
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
| LLM | Ollama (runtime-selectable via UI depth selector: gemma4:e4b / gemma4:e2b / qwen3:14b / gpt-oss:20b / qwen3:30b; default: gemma4:e4b) |
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
ollama pull gemma4:e4b          # Default model
ollama pull gemma4:e2b               # Fallback model
# Note: Embeddings use Qwen/Qwen3-Embedding-0.6B via HuggingFace
# (downloaded automatically on first run, requires GPU)

# Run Streamlit UI (local access)
uv run streamlit run src/ui/app.py --server.port 8511 --server.headless false

# Or run via CLI
python -m src.main --ui --port 8511

# Run single query (non-interactive)
python -m src.main --query "Was sind die Grenzwerte für Strahlenexposition?"

# Run tests
pytest tests/ -v

# Remote access via Cloudflare Tunnel (see login/README.md)
export LAUNCHER_PASSWORD="your-password"
./login/start-quick-tunnels.sh   # Terminal 1: creates temporary public URLs
./login/start-launcher.sh        # Terminal 2: password-gated launcher on port 8522
```

## Key Configuration

Edit `.env` for your setup:
- `OLLAMA_MODEL=gemma4:e4b`: Default LLM model (overridden at runtime by the UI depth selector)
- `OLLAMA_TEMPERATURE=0.0`: LLM sampling temperature
- `OLLAMA_NUM_CTX=131072`: 128K context for dual 4090s (adjust if needed)
- `OLLAMA_SAFE_LIMIT=0.9`: Stop at 90% to prevent OOM
- `QUALITY_THRESHOLD=375`: Minimum quality score (0-500, 5 dimensions)
- `REFERENCE_EXTRACTION_METHOD=hybrid`: Reference detection method (`regex`, `llm`, `hybrid`)
- `REFERENCE_TOKEN_BUDGET=50000`: Max tokens for reference following per task
- `CONVERGENCE_SAME_DOC_THRESHOLD=3`: Stop following when same doc appears N times
- `PRIMARY_MIN_CHUNKS=3`: Minimum primary chunks to keep per task (ensures transparency)
- `SECONDARY_MIN_CHUNKS=2`: Minimum secondary chunks to keep per task
- `RERANKER_STRATEGY=precision`: Chunk reranking strategy (`precision` or `recall`)
- `RERANKER_BATCH_SIZE=6`: Chunks per LLM reranking call
- `RERANKER_MIN_SCORE=4`: Minimum raw score (1-5) to keep after reranking

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
├── login/                 # Remote access via Cloudflare Tunnel
│   ├── launcher_app.py    # Password-gated Streamlit control panel (port 8522)
│   ├── start-launcher.sh  # Start launcher
│   ├── start-quick-tunnels.sh  # Create temporary Cloudflare quick tunnels
│   ├── cloudflared-config.yml  # Template for persistent tunnel (requires domain)
│   └── README.md          # Setup & usage docs
├── src/                   # Source code
│   ├── agents/            # LangGraph agents + tools
│   ├── models/            # Pydantic data models
│   ├── prompts/           # LLM prompt constants (model-conditional: hitl.py/hitl_gpt.py, etc.)
│   ├── services/          # ChromaDB, Ollama, PDF
│   └── ui/                # Streamlit app (incl. gpu_widget.py for live GPU stats + elapsed timer)
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
4. **Fully Local**: Ollama-only, no external API calls (exception: optional Tavily web search, disabled by default)
5. **Free GPU & Reset**: Unloads Ollama model (`keep_alive=0`), clears all `@st.cache_resource` caches, runs `torch.cuda.empty_cache()`, resets session — server stays alive (Cloudflare tunnel preserved)
6. **Reference Following**: Deep rabbithole traversal with hybrid detection (regex+LLM), document registry scoping, relevance filtering, and database-selection propagation (broad fallback searches respect `selected_database` from the UI)
7. **Runtime Model Selection**: UI depth selector in sidebar ("Erweiterte Einstellungen") with 5 levels — prompts auto-adapt via dynamic routing, all cached clients reset on switch; `settings.ollama_model` always synced with UI selection on every render (fixes stale `.env` model shown in GPU widget)
8. **Document Browser**: "Dokumente anzeigen" button in "Wissensdatenbank" sidebar panel opens a native `@st.dialog` modal listing all unique PDF filenames in the selected ChromaDB database (metadata-only query via `ChromaDBClient.get_document_names()`, no embeddings loaded)


## Prompt Management
**All LLM prompts MUST be defined in `src/prompts/` package** (split by phase: `hitl.py`, `research.py`, `synthesis.py`).
Every prompt is split into a `_SYSTEM` / `_HUMAN` pair (e.g. `TODO_GENERATION_PROMPT_SYSTEM` + `TODO_GENERATION_PROMPT_HUMAN`).
All callers use `OllamaClient.generate_structured_messages()` or `generate_messages()` with separate system/human arguments.

### Dynamic Prompt Routing (Runtime Model Switching)

The `src/prompts/__init__.py` uses **PEP 562 `__getattr__`** for runtime-dynamic prompt resolution:
- Both Qwen and gpt-oss prompt sets are eagerly loaded into `_qwen_prompts` / `_gptoss_prompts` dicts at import time
- `__getattr__(name)` checks `settings.model_family` **at access time** and returns from the correct dict
- Consumers use `from src import prompts` then `prompts.X` (module-level access, not `from src.prompts import X`)
- This enables switching models at runtime (via the UI depth selector) without restarting

The `model_family` property returns `"gpt-oss"` when `ollama_model.startswith("gpt-oss")`, `"gemma4"` when `"gemma4"` or `"gemma-4"` is found in the lowercased model string (case-insensitive substring match, covers `gemma4:e4b`), else `"qwen"`. Both `"gemma4"` and `"qwen"` resolve to the Qwen prompt set.
Both variants export **identical constant names** (48 total).

**Consumer pattern** (used in `nodes.py`, `tools.py`, `hitl_service.py`):
```python
from src import prompts
system = prompts.SYNTHESIS_PROMPT_ENHANCED_SYSTEM.format(language=lang)
```

**Singleton reset**: Each consumer module exposes `reset_ollama_client()` to clear cached `OllamaClient` instances when the model changes. Called by `_apply_research_depth()` in `app.py`.

### Qwen Prompt Format (hitl.py, research.py, synthesis.py)

Two SYSTEM prompt formats co-exist (see `docs/prompts-design.md` for full rules):
- **XML tag format** (`<role>`, `<output_format>`, `<constraints>`, `<content_rules>`, `<input_definitions>`, `<example>`): used by the 5 synthesis/summary prompts (`SYNTHESIS_PROMPT_ENHANCED_SYSTEM`, `SYNTHESIS_PROMPT_SYSTEM`, `QUERY_ASSESSMENT_PROMPT_SYSTEM`, `HITL_SUMMARY_PROMPT_SYSTEM`, `TASK_SUMMARY_PROMPT_SYSTEM`). Output format is placed 2nd so the LLM sees the schema before the rules.
- **Markdown section format** (`### Role / ### Goal / ### Rules / ### Output format`): used by all other prompts.

### gpt-oss Prompt Format (hitl_gpt.py, research_gpt.py, synthesis_gpt.py)

Adapted for Harmony format conventions:
- **`# Role` / `# Goal` / `# Rules`**: Top-level markdown headers (not `###` or XML tags)
- **Flat numbered rules**: No nested sub-lists (a, b, c) — everything at one level
- **`<json>...</json>` wrapper tags**: Structured output prompts instruct the model to wrap JSON in `<json>` tags
- **No `/no_think`**: Qwen3-specific directives removed
- **Output-only examples**: Trimmed input portions (~200 tokens saved per prompt)
- **`{language}` placeholder**: Retained in all content-bearing prompts (unchanged from Qwen variants)

### OllamaClient Adaptations for gpt-oss

- **Harmony preamble**: `_prepare_system_prompt()` prepends `"You are a helpful assistant.\nReasoning: high\n---\n"` for gpt-oss models
- **JSON tag extraction**: `_extract_json_from_tags()` regex-extracts content between `<json>` and `</json>` tags as fallback when structured output parsing fails
- **Temperature**: `ChatOllama` instances use `settings.ollama_temperature` (configurable, default `0.0`)

### OllamaClient Adaptations for gemma4

- **`/no_think` stripping**: `_prepare_system_prompt()` removes `/no_think` tokens from system prompts for gemma4 models (Qwen3-specific directive not supported by Gemma)
- Prompt set: uses the Qwen prompt set unchanged (no separate prompt files needed)

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
| [docs/prompt-opt-guide.md](docs/prompt-opt-guide.md) | @docs/prompt-opt-guide.md 12-principle optimization guide for ≤32B local LLMs |
| [login/README.md](login/README.md) | Remote access setup via Cloudflare Tunnel (launcher + quick tunnels) |

## Implementation Status

See @docs/implementation.md [docs/implementation.md](docs/implementation.md)