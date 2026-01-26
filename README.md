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

- **Human-In-The-Loop**: Query clarification and task approval checkpoints
- **Deep Reference Following**: Automatically follows cross-document references
- **Multi-Collection Search**: Searches across all ChromaDB collections
- **Structured Output**: JSON-mode structured responses via Pydantic
- **Quality Assessment**: 4-dimension quality scoring (0-400)
- **Source Attribution**: Clickable PDF links with page numbers

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation.
