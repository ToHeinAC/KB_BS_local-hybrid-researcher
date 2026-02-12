"""Application configuration using pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_fallback_model: str = "qwen3:8b"
    default_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Critical: 128K context for dual 4090s
    ollama_num_ctx: int = 131072
    # Safety: stop at 90% of max context to prevent OOM
    ollama_safe_limit: float = 0.9

    # ChromaDB Configuration
    chromadb_path: str = "./kb/database"

    # Streamlit Configuration
    streamlit_port: int = 8511

    # Logging
    log_level: str = "INFO"

    # Debug state dumps (writes markdown snapshots to tests/debugging/)
    enable_state_dump: bool = False

    # Phase 1: Query Analysis + HITL
    max_clarification_questions: int = 3

    # Phase 2: ToDo List
    todo_max_items: int = 15
    initial_todo_items: int = 5

    # Phase 3: Research Execution
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

    # Phase 4: Quality Assurance
    enable_quality_checker: bool = True
    quality_threshold: int = 375
    max_reflections: int = 1

    # Web Search (optional)
    enable_web_search: bool = False
    tavily_api_key: str = ""
    web_results_per_query: int = 2

    @property
    def safe_context_limit(self) -> int:
        """Calculate safe context token limit (90% of max)."""
        return int(self.ollama_num_ctx * self.ollama_safe_limit)

    @property
    def chromadb_path_resolved(self) -> Path:
        """Return resolved ChromaDB path."""
        return Path(self.chromadb_path).resolve()


# Singleton instance
settings = Settings()
