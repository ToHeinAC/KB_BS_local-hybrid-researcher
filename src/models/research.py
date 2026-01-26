"""Research context and task tracking models."""

from typing import Literal

from pydantic import BaseModel, Field

from src.models.results import (
    CategorySummary,
    CritiqueResult,
    DocumentFinding,
    Source,
    VectorResult,
    WebResult,
)


class NestedChunk(BaseModel):
    """Chunk retrieved from following a reference."""

    chunk: str = Field(description="The chunk text")
    document: str = Field(description="Source document name")
    extracted_info: str = Field(
        default="",
        description="Condensed relevant information",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score to original query (0-1)",
    )


class DetectedReference(BaseModel):
    """A reference detected within text."""

    type: Literal["section", "document", "external"] = Field(
        description="Type of reference",
    )
    target: str = Field(description="Reference target (section number, doc name, URL)")
    original_text: str = Field(
        default="",
        description="Original text containing the reference",
    )
    found: bool = Field(
        default=False,
        description="Whether the reference was resolved",
    )
    nested_chunks: list[NestedChunk] = Field(
        default_factory=list,
        description="Chunks retrieved from following this reference",
    )


class ChunkWithInfo(BaseModel):
    """A chunk from vector DB with extracted info and references."""

    chunk: str = Field(description="Original chunk text from vector DB")
    document: str = Field(description="Source document name")
    page: int | None = Field(
        default=None,
        description="Page number if available",
    )
    extracted_info: str | None = Field(
        default=None,
        description="Condensed relevant passages",
    )
    references: list[DetectedReference] = Field(
        default_factory=list,
        description="References detected in this chunk",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score from vector search",
    )


class SearchQueryResult(BaseModel):
    """Results for a single search query."""

    query: str = Field(description="The search query used")
    chunks: list[ChunkWithInfo] = Field(
        default_factory=list,
        description="Retrieved chunks with extracted info",
    )
    summary: str | None = Field(
        default=None,
        description="Summary synthesized from chunks",
    )
    summary_references: list[str] = Field(
        default_factory=list,
        description="Documents referenced in summary",
    )
    quality_score: float | None = Field(
        default=None,
        description="Quality score for this query result",
    )
    web_search_results: list[WebResult] | None = Field(
        default=None,
        description="Web search results if used",
    )


class ResearchContextMetadata(BaseModel):
    """Metadata about the research process."""

    total_iterations: int = Field(
        default=0,
        description="Total number of research iterations",
    )
    documents_referenced: list[str] = Field(
        default_factory=list,
        description="All documents referenced",
    )
    external_sources_used: bool = Field(
        default=False,
        description="Whether external sources were used",
    )
    visited_refs: list[str] = Field(
        default_factory=list,
        description="References already visited (for loop prevention)",
    )


class ResearchContext(BaseModel):
    """Growing JSON accumulating all research findings."""

    search_queries: list[SearchQueryResult] = Field(
        default_factory=list,
        description="Results for each search query",
    )
    metadata: ResearchContextMetadata = Field(
        default_factory=ResearchContextMetadata,
        description="Research process metadata",
    )

    def add_document_reference(self, doc_name: str) -> None:
        """Add a document to referenced list if not present."""
        if doc_name not in self.metadata.documents_referenced:
            self.metadata.documents_referenced.append(doc_name)

    def has_visited_ref(self, ref_key: str) -> bool:
        """Check if a reference has been visited."""
        return ref_key in self.metadata.visited_refs

    def mark_ref_visited(self, ref_key: str) -> None:
        """Mark a reference as visited."""
        if ref_key not in self.metadata.visited_refs:
            self.metadata.visited_refs.append(ref_key)


class ResearchTask(BaseModel):
    """A single research task with full tracking."""

    id: str = Field(description="Unique task identifier")
    description: str = Field(description="Task description")
    status: Literal["pending", "in_progress", "completed", "blocked"] = Field(
        default="pending",
        description="Current task status",
    )

    # Query tracking
    query_sets: list["QuerySet"] = Field(
        default_factory=list,
        description="Generated query sets",
    )
    current_iteration: int = Field(
        default=0,
        description="Current iteration number",
    )

    # Research results
    vector_results: list[VectorResult] = Field(
        default_factory=list,
        description="Results from vector search",
    )
    doc_results: list[DocumentFinding] = Field(
        default_factory=list,
        description="Results from document search",
    )
    web_results: list[WebResult] = Field(
        default_factory=list,
        description="Results from web search",
    )

    # Summaries
    summaries: list[CategorySummary] = Field(
        default_factory=list,
        description="Generated summaries",
    )

    # Critique tracking
    critiques: list[CritiqueResult] = Field(
        default_factory=list,
        description="Gap analysis results",
    )

    # Source tracking
    all_sources: list[Source] = Field(
        default_factory=list,
        description="All sources used",
    )

    # Lineage
    spawned_from: str | None = Field(
        default=None,
        description="Parent task ID if spawned",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )


# Import here to avoid circular import
from src.models.query import QuerySet  # noqa: E402

ResearchTask.model_rebuild()
