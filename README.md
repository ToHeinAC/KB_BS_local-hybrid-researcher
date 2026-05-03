# Rabbithole-Agent: Local Hybrid Researcher

A fully local, privacy-first research system that performs deep reference-following across document collections using Ollama LLMs, ChromaDB, and LangGraph.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env

# Pull Ollama models (for LLM generation)
ollama pull qwen3:14b            # Default model
ollama pull granite4.1:8b        # Fallback model
# Optional: ollama pull gpt-oss:20b  # Alternative model (auto-detected)

# Note: Embeddings use HuggingFace Qwen/Qwen3-Embedding-0.6B
# (downloaded automatically on first run)

# Run UI (local)
streamlit run src/ui/app.py --server.port 8511

# Remote access via Cloudflare Tunnel
export LAUNCHER_PASSWORD="your-password"
./login/start-quick-tunnels.sh   # Creates temporary public HTTPS URLs
./login/start-launcher.sh        # Password-gated launcher on port 8522
```

## Features

- **Iterative Retrieval-HITL Loop**: Integrated vector search during the clarification phase to provide smarter, context-aware follow-up questions.
- **Convergence Detection**: Automated loop termination based on information coverage, knowledge gaps, and content deduplication.
- **Multi-Angle Search**: Generates original, broader, and alternative queries in parallel to ensure maximum document coverage.
- **Query Assessment Gate**: After HITL, an LLM gate (`assess_query`) decides whether the query is answerable from the knowledge base, sets the number of research tasks (3-6), and routes unanswerable queries to an immediate rejection response — no wasted compute.
- **Task Summary Reranking**: Deterministic relevance-based sort before synthesis; high-relevance findings ([Relevance: ≥70/100]) weighted as primary evidence, low-relevance as supplementary only
- **Batch Chunk Reranking**: Efficient batch LLM scoring (~3-4 calls for 20 chunks) with precision/recall strategies, cross-batch normalization, and hard-filtering — replaces per-chunk scoring.
- **Multi-Query Task Execution**: Each research task generates 3 deduplicated search queries (1 base + 2 LLM-targeted) for comprehensive retrieval.
- **Deep Reference Following**: Hybrid regex+LLM detection with document registry-based scoped resolution, token budget tracking, convergence detection, and full `selected_database` propagation (broad fallback searches respect the user's database selection). Agentic reference gate lets the LLM skip tangential references.
- **Agentic Quality Remediation**: LLM evaluates synthesis quality and autonomously retries with focused instructions when below threshold.
- **Graded Context Management**: Tiered classification (primary/secondary/tertiary) prevents query drift and ensures synthesis quality.
- **Verbatim Quote Preservation**: Critical legal/technical quotes extracted and preserved for precision.
- **Deep Report Synthesis**: Produces extensive, structured deep reports (not brief summaries) from pre-digested task summaries, with exact figures, verbatim quotes, and section references, anchored to original intent with HITL context.
- **Runtime Research Depth Selector**: Switch LLM models at runtime via a sidebar dropdown (basic/einfach/standard/erhöht/tief; default: gemma4:e4b) — no restart required. Prompts auto-adapt via PEP 562 dynamic routing; all cached clients reset transparently.
- **Model-Conditional Prompt Routing**: Prompts automatically swap between Qwen (XML-tag/`###` formats) and Harmony-adapted gpt-oss variants (`# headers`, `<json>` tags, no `/no_think`) based on the active model — zero consumer-code changes.
- **Optimized Prompt Architecture**: All prompts in `src/prompts/` package (split by phase: `hitl.py`, `research.py`, `synthesis.py` + `*_gpt.py` variants), each as SYSTEM/HUMAN pairs. The 5 key synthesis/summary prompts use an XML-tag format (`<role>`, `<output_format>`, `<constraints>`, `<content_rules>`, `<example>`) optimized for Qwen3:14b — output schema placed before rules so the LLM anchors on structure first, HARD CONSTRAINTS separated from writing rules, realistic domain examples replacing placeholders.
- **Language Enforcement**: All 17 content-bearing prompts enforce `{language}`, with validation and retry on mismatch.
- **Pre-Synthesis Drift Detection**: Filters irrelevant accumulated context before synthesis.
- **Transparent Chunk Filtering**: Guarantees minimum chunks per task (3 primary, 2 secondary) even if below relevance threshold; backfilled chunks marked with ⚠️ badge to maintain transparency while preventing silent suppression of retrieval results.
- **Full Human-In-The-Loop**: Checkpoints for query refinement, task list approval, and final result verification.
- **Privacy-First & Local**: Powered by Ollama and local ChromaDB, ensuring all research data stays on your machine.
- **Numbered Citation Transformation**: Inline `[Document.pdf, Page N]` citations are post-processed into sequential `[1]`, `[2]`, … markers with an appended reference list. PDFs open directly in the browser via the `/_api/pdf` Tornado route (same injection pattern as the GPU widget).
- **Persistent Results View**: Completed report page shows HITL conversation, task summaries with findings/gaps, and per-task tiered chunk expanders (primary/secondary/tertiary) with full original vector DB text + LLM extraction alongside the final answer.
- **Retrieval History Panel**: Real-time display of vector search results during HITL with chunk details.
- **Remote Access via Cloudflare Tunnel**: Password-protected launcher app (`login/`) with start/stop/restart controls, accessible via temporary `*.trycloudflare.com` quick tunnel URLs. Coexists safely with other tunnels. See `login/README.md`.
- **Database Selection**: Choose specific knowledge base collections or search all.
- **Cached Service Clients**: Fast UI reloads via `@st.cache_resource` for ChromaDB/Ollama clients.
- **Live GPU Widget**: Sidebar shows real-time GPU temp/fan/load + elapsed research time via Tornado route injection (`/_api/gpu`), updating every 1s even during blocking `graph.stream()` calls. Timer starts at todo approval, freezes on report completion, resets on new session. Color-coded thresholds for temp, load, and elapsed time. Graceful degradation when no GPU is available.

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation.
