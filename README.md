# Rabbithole-Agent: Local Hybrid Researcher

A fully local, privacy-first research system that performs deep reference-following across document collections using Ollama LLMs, ChromaDB, and LangGraph.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env

# Pull Ollama models (for LLM generation)
ollama pull qwen3:14b
ollama pull qwen3:8b

# Note: Embeddings use HuggingFace Qwen/Qwen3-Embedding-0.6B
# (downloaded automatically on first run)

# Run UI
streamlit run src/ui/app.py --server.port 8511
```

## Features

- **Iterative Retrieval-HITL Loop**: Integrated vector search during the clarification phase to provide smarter, context-aware follow-up questions.
- **Convergence Detection**: Automated loop termination based on information coverage, knowledge gaps, and content deduplication.
- **Multi-Angle Search**: Generates original, broader, and alternative queries in parallel to ensure maximum document coverage.
- **Deep Reference Following**: Hybrid regex+LLM detection with document registry-based scoped resolution, token budget tracking, and convergence detection.
- **Graded Context Management**: Tiered classification (primary/secondary/tertiary) prevents query drift and ensures synthesis quality.
- **Verbatim Quote Preservation**: Critical legal/technical quotes extracted and preserved for precision.
- **Query-Anchored Synthesis**: Final answers stay anchored to original intent with HITL context included.
- **Language Enforcement**: Strict single-language output with validation and retry on mismatch.
- **Pre-Synthesis Drift Detection**: Filters irrelevant accumulated context before synthesis.
- **Full Human-In-The-Loop**: Checkpoints for query refinement, task list approval, and final result verification.
- **Privacy-First & Local**: Powered by Ollama and local ChromaDB, ensuring all research data stays on your machine.
- **Source Attribution**: Detailed citations with clickable PDF links and page numbers.
- **Retrieval History Panel**: Real-time display of vector search results during HITL with chunk details.
- **Database Selection**: Choose specific knowledge base collections or search all.
- **Cached Service Clients**: Fast UI reloads via `@st.cache_resource` for ChromaDB/Ollama clients.

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation.
