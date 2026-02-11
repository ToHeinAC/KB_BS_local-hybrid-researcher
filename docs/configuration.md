# Configuration

## Environment Variables (.env)

```ini
# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
OLLAMA_FALLBACK_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b

# =============================================================================
# CHROMADB CONFIGURATION
# =============================================================================
CHROMADB_PATH=./kb/database

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================
STREAMLIT_PORT=8511

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# PHASE 1: QUERY ANALYSIS + HITL
# =============================================================================
MAX_CLARIFICATION_QUESTIONS=3

# =============================================================================
# PHASE 2: TODO LIST
# =============================================================================
TODO_MAX_ITEMS=15                # Max tasks (prevents runaway)
INITIAL_TODO_ITEMS=5             # Initial tasks (3-5 typical)

# =============================================================================
# PHASE 3: RESEARCH EXECUTION
# =============================================================================
# Search parameters
N_SEARCH_QUERIES=3               # Queries per task
M_CHUNKS_PER_QUERY=4             # Chunks per query from vector DB

# Reference following
REFERENCE_FOLLOW_DEPTH=2         # Max nesting levels
REFERENCE_RELEVANCE_THRESHOLD=0.6
MAX_ITERATIONS_PER_TASK=3        # Prevent loops per task

# Enhanced reference following
DOCUMENT_REGISTRY_PATH=./kb/document_registry.json  # Document-synonym mapping
REFERENCE_EXTRACTION_METHOD=hybrid   # "regex", "llm", or "hybrid"
REFERENCE_TOKEN_BUDGET=50000         # Max tokens for reference following per task
CONVERGENCE_SAME_DOC_THRESHOLD=3     # Stop when same doc appears N times

# Document search limits
MAX_DOCS=5                       # Documents to analyze per task
DOC_WORD_LIMIT=5000              # Max words per document

# =============================================================================
# PHASE 4: QUALITY ASSURANCE
# =============================================================================
ENABLE_QUALITY_CHECKER=true
QUALITY_THRESHOLD=375            # Min score (0-500, 5 dimensions) to pass
MAX_REFLECTIONS=1                # Max regeneration attempts

# =============================================================================
# OPTIONAL: WEB SEARCH
# =============================================================================
ENABLE_WEB_SEARCH=false
TAVILY_API_KEY=                  # Required if web search enabled
WEB_RESULTS_PER_QUERY=2
```

---

## pyproject.toml

```toml
[project]
name = "kb-bs-local-hybrid-researcher"
version = "2.2.0"
description = "Local hybrid researcher with deep reference-following using Ollama and ChromaDB"
readme = "README.md"
requires-python = ">=3.10"  # LangChain v1.0 dropped Python 3.9
license = { text = "Apache-2.0" }
authors = [{ name = "Rabbithole-Agent Team" }]
dependencies = [
    # LangChain v1.0+ ecosystem
    "langchain>=1.0.0",
    "langchain-core>=1.0.0",
    "langchain-ollama>=0.3.0",
    "langchain-chroma>=0.2.0",
    "langchain-huggingface>=1.0.0",
    "langgraph>=1.0.0",
    # Embeddings (Qwen3-Embedding-0.6B via HuggingFace)
    "sentence-transformers>=3.0.0",
    "torch>=2.0.0",
    # Data & Storage
    "chromadb>=0.5.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    # PDF & UI
    "streamlit>=1.28.0",
    "pymupdf>=1.24.0",
    "python-dotenv>=1.0.0",
    # Retry logic
    "tenacity>=8.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]
web = [
    "tavily-python>=0.3.0",
]

[project.scripts]
researcher = "src.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## Configuration Constants (src/config.py)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_fallback_model: str = "qwen3:8b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_num_ctx: int = 131072
    ollama_safe_limit: float = 0.9

    # ChromaDB
    chromadb_path: str = "./kb/database"

    # Streamlit
    streamlit_port: int = 8511

    # Logging
    log_level: str = "INFO"

    # Phase 1
    max_clarification_questions: int = 3

    # Phase 2
    todo_max_items: int = 15
    initial_todo_items: int = 5

    # Phase 3
    n_search_queries: int = 3
    m_chunks_per_query: int = 4
    reference_follow_depth: int = 2
    reference_relevance_threshold: float = 0.6
    max_iterations_per_task: int = 3
    max_docs: int = 5
    doc_word_limit: int = 5000

    # Enhanced Reference Following
    document_registry_path: str = "./kb/document_registry.json"
    reference_extraction_method: str = "hybrid"  # "regex", "llm", "hybrid"
    reference_token_budget: int = 50000
    convergence_same_doc_threshold: int = 3

    # Phase 4
    enable_quality_checker: bool = True
    quality_threshold: int = 375
    max_reflections: int = 1

    # Web search (optional)
    enable_web_search: bool = False
    tavily_api_key: str = ""
    web_results_per_query: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Singleton
settings = Settings()
```

---

## Development Commands

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
```

### Ollama Models

```bash
# Pull required models
ollama pull qwen3:14b
ollama pull qwen3:8b

# Check models
ollama list

# Test model
ollama run qwen3:14b "Hello"
```

### Run Application

```bash
# Run Streamlit (port >8510 per global config)
streamlit run src/ui/app.py --server.port 8511

# Run with auto-reload
streamlit run src/ui/app.py --server.port 8511 --server.runOnSave true
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_agents.py -v
```

### Linting

```bash
# Check code style
ruff check src/

# Fix auto-fixable issues
ruff check src/ --fix

# Type checking
mypy src/
```

### ChromaDB Management

```bash
# List collections
python -c "
import chromadb
c = chromadb.PersistentClient('kb/database/GLageKon__Qwen--Qwen3-Embedding-0.6B--10000--2000')
print([col.name for col in c.list_collections()])
"
```
