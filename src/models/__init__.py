"""Pydantic data models for the research agent."""

from src.models.hitl import (
    ClarificationQuestion,
    HITLCheckpoint,
    HITLDecision,
)
from src.models.query import (
    QueryAnalysis,
    QueryContext,
    QuerySet,
    ToDoItem,
    ToDoList,
)
from src.models.research import (
    ChunkWithInfo,
    DetectedReference,
    NestedChunk,
    ResearchContext,
    ResearchContextMetadata,
    ResearchTask,
    SearchQueryResult,
)
from src.models.results import (
    CategorySummary,
    CritiqueResult,
    DocumentFinding,
    FinalReport,
    Finding,
    GrepMatch,
    LinkedSource,
    PDFMetadata,
    QualityAssessment,
    Source,
    VectorResult,
    WebResult,
)

__all__ = [
    # Query models
    "QueryAnalysis",
    "QueryContext",
    "QuerySet",
    "ToDoItem",
    "ToDoList",
    # Research models
    "ChunkWithInfo",
    "DetectedReference",
    "NestedChunk",
    "ResearchContext",
    "ResearchContextMetadata",
    "ResearchTask",
    "SearchQueryResult",
    # Result models
    "CategorySummary",
    "CritiqueResult",
    "DocumentFinding",
    "Finding",
    "FinalReport",
    "GrepMatch",
    "LinkedSource",
    "PDFMetadata",
    "QualityAssessment",
    "Source",
    "VectorResult",
    "WebResult",
    # HITL models
    "ClarificationQuestion",
    "HITLCheckpoint",
    "HITLDecision",
]
