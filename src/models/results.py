"""Result and report models."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VectorResult(BaseModel):
    """Result from ChromaDB vector search."""

    doc_id: str = Field(description="Document ID in ChromaDB")
    doc_name: str = Field(description="Document filename")
    chunk_text: str = Field(description="Retrieved chunk text")
    page_number: int | None = Field(
        default=None,
        description="Page number if available",
    )
    chunk_id: int | None = Field(
        default=None,
        description="Chunk sequence number within page",
    )
    relevance_score: float = Field(description="Similarity score (0-1)")
    collection: str = Field(description="ChromaDB collection name")
    query_used: str = Field(description="Query that retrieved this result")


class DocumentFinding(BaseModel):
    """Finding from document navigation."""

    doc_path: str = Field(description="Full path to document")
    doc_name: str = Field(description="Document filename")
    passage: str = Field(description="Relevant passage text")
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Page numbers where found",
    )
    keywords_matched: list[str] = Field(
        default_factory=list,
        description="Keywords that matched",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score",
    )
    references_found: list[str] = Field(
        default_factory=list,
        description="References found in passage",
    )
    search_chain: list[str] = Field(
        default_factory=list,
        description="How we got here (navigation path)",
    )


class WebResult(BaseModel):
    """Result from web search."""

    title: str = Field(description="Page title")
    url: str = Field(description="Page URL")
    snippet: str = Field(description="Search result snippet")
    content: str | None = Field(
        default=None,
        description="Full page content if fetched",
    )
    query_used: str = Field(description="Query that found this result")
    source: str = Field(
        default="web",
        description="Source identifier",
    )


class Source(BaseModel):
    """A source document or chunk."""

    doc_id: str = Field(description="Document identifier")
    doc_name: str = Field(description="Document filename")
    chunk_text: str = Field(description="Source text chunk")
    page_number: int | None = Field(
        default=None,
        description="Page number if available",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score",
    )
    collection: str | None = Field(
        default=None,
        description="ChromaDB collection",
    )
    category: Literal["vector_db", "documents", "web"] = Field(
        default="vector_db",
        description="Source category",
    )


class LinkedSource(BaseModel):
    """Source with resolved path for linking."""

    source: Source = Field(description="The source")
    resolved_path: str = Field(description="Full resolved file path")
    link_html: str = Field(description="Clickable HTML link")


class CategorySummary(BaseModel):
    """Summary for one research category."""

    category: Literal["vector_db", "documents", "web"] = Field(
        description="Category of sources",
    )
    summary_text: str = Field(description="Summary text")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings",
    )
    sources_used: list[Source] = Field(
        default_factory=list,
        description="Sources used in summary",
    )
    source_quotes: dict[str, str] = Field(
        default_factory=dict,
        description="source_id -> quote mapping",
    )
    todo_item_id: str = Field(description="Associated todo item ID")
    query_context: str = Field(
        default="",
        description="Query context for this summary",
    )


class CritiqueResult(BaseModel):
    """Gap analysis result."""

    todo_item_id: str = Field(description="Associated todo item ID")
    gaps_found: list[str] = Field(
        default_factory=list,
        description="Identified gaps",
    )
    severity: Literal["none", "minor", "significant"] = Field(
        default="none",
        description="Gap severity",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Queries to fill gaps",
    )
    cross_document_refs: list[str] = Field(
        default_factory=list,
        description="Cross-document references to follow",
    )
    cross_todo_refs: list[str] = Field(
        default_factory=list,
        description="Related todo items",
    )
    should_continue: bool = Field(
        default=False,
        description="Whether to continue researching",
    )


class QualityAssessment(BaseModel):
    """Quality check result with 5-dimension scoring."""

    overall_score: int = Field(
        ge=0,
        le=500,
        description="Overall quality score (0-500)",
    )
    factual_accuracy: int = Field(
        ge=0,
        le=100,
        description="Factual accuracy score (0-100)",
    )
    semantic_validity: int = Field(
        ge=0,
        le=100,
        description="Semantic validity score (0-100)",
    )
    structural_integrity: int = Field(
        ge=0,
        le=100,
        description="Structural integrity score (0-100)",
    )
    citation_correctness: int = Field(
        ge=0,
        le=100,
        description="Citation correctness score (0-100)",
    )
    query_relevance: int = Field(
        ge=0,
        le=100,
        default=0,
        description="Query relevance score (0-100)",
    )
    passes_quality: bool = Field(description="Whether passes quality threshold")
    issues_found: list[str] = Field(
        default_factory=list,
        description="Issues identified",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement",
    )


class Finding(BaseModel):
    """A discrete finding from research."""

    claim: str = Field(description="The finding/claim")
    evidence: str = Field(description="Supporting evidence")
    sources: list[Source] = Field(
        default_factory=list,
        description="Supporting sources",
    )
    source_quotes: dict[str, str] = Field(
        default_factory=dict,
        description="source_id -> quote mapping",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level",
    )


class FinalReport(BaseModel):
    """Final structured output."""

    query: str = Field(description="Original query")
    answer: str = Field(description="Final answer with citations")
    findings: list[Finding] = Field(
        default_factory=list,
        description="Discrete findings",
    )
    sources: list[LinkedSource] = Field(
        default_factory=list,
        description="All sources with links",
    )
    quality_score: int = Field(description="Overall quality score")
    quality_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Quality dimension scores",
    )
    todo_items_completed: int = Field(description="Number of tasks completed")
    research_iterations: int = Field(description="Total iterations")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    # Web search (supplementary, strictly separated from KB results)
    web_search_section: str = Field(
        default="",
        description="Separate web search results section (markdown)",
    )
    web_sources: list[WebResult] = Field(
        default_factory=list,
        description="Web search source entries",
    )


class WebSearchSummaryOutput(BaseModel):
    """LLM output for web search result summarization."""

    web_summary: str = Field(
        description="Markdown summary of web results with [Title](URL) citations",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictions found between web and KB findings",
    )


class SynthesisOutputEnhanced(BaseModel):
    """Enhanced synthesis output with query coverage tracking."""

    summary: str = Field(
        ...,
        description="Comprehensive answer in the specified language only"
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="List of most important findings"
    )
    query_coverage: int = Field(
        default=50,
        ge=0,
        le=100,
        description="How completely the query was answered (0-100)"
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        description="Unanswered aspects of the query"
    )


class TaskSummaryOutput(BaseModel):
    """Output for per-task structured summary."""

    summary: str = Field(description="Comprehensive task synthesis")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Findings with [Document.pdf, Page N] citations and verbatim quotes"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Identified gaps or limitations"
    )
    relevance_assessment: str = Field(
        default="",
        description="Whether findings actually match query intent"
    )
    irrelevant_findings: list[str] = Field(
        default_factory=list,
        description="Findings superficially related but not answering the query"
    )
    relevance_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Relevance of findings to the original query (0-100)"
    )


class TaskSummarySimple(BaseModel):
    """Minimal, lenient per-task summary schema for graceful degradation.

    Used as a Tier-2 retry when the strict ``TaskSummaryOutput`` fails to parse
    (common with small/code-specialized models that emit schema-mismatched JSON,
    e.g. ``relevance_score`` as ``"60%"``). Only ``summary`` is required; the
    relevance score coerces loosely and defaults to 50 on any failure.
    """

    summary: str = Field(description="Task synthesis text")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Findings with [Document.pdf, Page N] citations"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Identified gaps or limitations"
    )
    relevance_score: int = Field(
        default=50,
        description="Relevance of findings to the original query (0-100)"
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_relevance_score(cls, v: object) -> int:
        """Coerce loose inputs ("60", 60.0, "60%", "score: 60") to a 0-100 int."""
        if isinstance(v, bool):
            return 50
        if isinstance(v, (int, float)):
            num = int(v)
        elif isinstance(v, str):
            match = re.search(r"-?\d+", v)
            num = int(match.group()) if match else 50
        else:
            return 50
        return max(0, min(100, num))


class RelevanceScoreOutput(BaseModel):
    """Output for relevance scoring."""

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="Relevance score 0-100"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the score"
    )


class ChunkRankingOutput(BaseModel):
    """LLM output for scoring one chunk against the query + HITL context."""

    relevance_score: int = Field(
        ge=0,
        le=100,
        description="Relevance score 0-100 (100=directly answers, 0=irrelevant)",
    )
    reasoning: str = Field(
        default="",
        description="One-sentence explanation of the score in the target language",
    )


class RerankerChunkResult(BaseModel):
    """Single chunk score from batch reranker (precision strategy)."""

    id: int = Field(ge=0, description="Chunk ID within the batch")
    score: int = Field(ge=1, le=5, description="Relevance score 1-5")
    reason: str = Field(default="", description="One-sentence micro-CoT")


class RerankerBatchOutput(BaseModel):
    """Batch reranker output (precision strategy)."""

    results: list[RerankerChunkResult] = Field(default_factory=list)


class RerankerRecallChunkResult(BaseModel):
    """Single chunk score from recall-focused reranker (short field names)."""

    id: int = Field(ge=0, description="Chunk ID within the batch")
    s: int = Field(ge=1, le=5, description="Relevance score 1-5")
    r: str = Field(default="", description="Reason, max 15 words")


class RerankerRecallBatchOutput(BaseModel):
    """Batch reranker output (recall strategy)."""

    results: list[RerankerRecallChunkResult] = Field(default_factory=list)


class PDFMetadata(BaseModel):
    """PDF document metadata."""

    filename: str = Field(description="PDF filename")
    path: str = Field(description="Full file path")
    page_count: int = Field(description="Number of pages")
    file_size_bytes: int = Field(description="File size in bytes")


class GrepMatch(BaseModel):
    """A search match in documents."""

    doc_path: str = Field(description="Document path")
    doc_name: str = Field(description="Document name")
    page: int = Field(description="Page number")
    line_number: int = Field(description="Line number")
    context: str = Field(description="Surrounding context")
    match: str = Field(description="Matched text")
    score: float = Field(
        default=1.0,
        description="Match score",
    )
