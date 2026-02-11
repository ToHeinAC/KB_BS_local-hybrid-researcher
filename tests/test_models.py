"""Tests for Pydantic models."""

import pytest

from src.models.hitl import (
    ClarificationQuestion,
    HITLCheckpoint,
    HITLDecision,
    HITLState,
)
from src.models.query import QueryAnalysis, QuerySet, ToDoItem, ToDoList
from src.models.research import (
    ChunkWithInfo,
    DetectedReference,
    NestedChunk,
    ResearchContext,
)
from src.models.results import (
    Finding,
    FinalReport,
    QualityAssessment,
    Source,
    VectorResult,
)


class TestQueryModels:
    """Test query-related models."""

    def test_query_analysis_creation(self):
        """Test QueryAnalysis model creation."""
        analysis = QueryAnalysis(
            original_query="Was sind die Grenzwerte für Strahlenexposition?",
            key_concepts=["Grenzwerte", "Strahlenexposition"],
            entities=["StrlSchV"],
            scope="regulatory",
            assumed_context=["German law"],
            clarification_needed=True,
        )

        assert analysis.original_query == "Was sind die Grenzwerte für Strahlenexposition?"
        assert len(analysis.key_concepts) == 2
        assert analysis.clarification_needed is True
        assert analysis.detected_language == "de"

    def test_query_analysis_defaults(self):
        """Test QueryAnalysis with minimal fields."""
        analysis = QueryAnalysis(original_query="test query")

        assert analysis.original_query == "test query"
        assert analysis.key_concepts == []
        assert analysis.clarification_needed is False

    def test_todo_item_creation(self):
        """Test ToDoItem model creation."""
        item = ToDoItem(
            id=1,
            task="Research dose limits",
            context="Need to find legal requirements",
        )

        assert item.id == 1
        assert item.task == "Research dose limits"
        assert item.completed is False

    def test_todo_list_operations(self):
        """Test ToDoList model operations."""
        todo = ToDoList(
            items=[
                ToDoItem(id=1, task="Task 1"),
                ToDoItem(id=2, task="Task 2", completed=True),
                ToDoItem(id=3, task="Task 3"),
            ]
        )

        # Test get_pending_tasks
        pending = todo.get_pending_tasks()
        assert len(pending) == 2
        assert all(not t.completed for t in pending)

        # Test get_next_task
        next_task = todo.get_next_task()
        assert next_task is not None
        assert next_task.id == 1

        # Test mark_completed
        todo.mark_completed(1)
        assert todo.items[0].completed is True

        # Test add_task
        new_item = todo.add_task("Task 4", "context")
        assert new_item is not None
        assert new_item.id == 4
        assert len(todo.items) == 4

    def test_todo_list_max_items(self):
        """Test ToDoList max items limit."""
        todo = ToDoList(
            items=[ToDoItem(id=i, task=f"Task {i}") for i in range(15)],
            max_items=15,
        )

        # Should not add new task at limit
        result = todo.add_task("Should not add")
        assert result is None
        assert len(todo.items) == 15

    def test_query_set_creation(self):
        """Test QuerySet model creation."""
        query_set = QuerySet(
            todo_item_id="1",
            vector_queries=["query 1", "query 2", "query 3"],
            doc_keywords=["keyword 1", "keyword 2"],
        )

        assert query_set.todo_item_id == "1"
        assert len(query_set.vector_queries) == 3
        assert query_set.iteration == 1
        assert query_set.generated_from_critique is False


class TestResearchModels:
    """Test research-related models."""

    def test_nested_chunk_creation(self):
        """Test NestedChunk model creation."""
        chunk = NestedChunk(
            chunk="Some text from document",
            document="StrlSchV.pdf",
            extracted_info="Relevant info extracted",
            relevance_score=0.85,
        )

        assert chunk.document == "StrlSchV.pdf"
        assert chunk.relevance_score == 0.85

    def test_detected_reference_creation(self):
        """Test DetectedReference model creation."""
        ref = DetectedReference(
            type="section",
            target="§ 5",
            original_text="siehe § 5",
        )

        assert ref.type == "section"
        assert ref.found is False

    def test_chunk_with_info_creation(self):
        """Test ChunkWithInfo model creation."""
        chunk = ChunkWithInfo(
            chunk="Original chunk text",
            document="test.pdf",
            page=5,
            extracted_info="Extracted relevant content",
            relevance_score=0.9,
        )

        assert chunk.page == 5
        assert chunk.relevance_score == 0.9
        assert chunk.references == []

    def test_research_context_operations(self):
        """Test ResearchContext model operations."""
        ctx = ResearchContext()

        # Test add_document_reference
        ctx.add_document_reference("doc1.pdf")
        ctx.add_document_reference("doc1.pdf")  # Duplicate
        ctx.add_document_reference("doc2.pdf")
        assert len(ctx.metadata.documents_referenced) == 2

        # Test ref tracking
        assert not ctx.has_visited_ref("section:§5")
        ctx.mark_ref_visited("section:§5")
        assert ctx.has_visited_ref("section:§5")


class TestResultModels:
    """Test result-related models."""

    def test_vector_result_creation(self):
        """Test VectorResult model creation."""
        result = VectorResult(
            doc_id="123",
            doc_name="document.pdf",
            chunk_text="Some chunk text",
            page_number=10,
            relevance_score=0.95,
            collection="GLageKon",
            query_used="test query",
        )

        assert result.doc_name == "document.pdf"
        assert result.relevance_score == 0.95

    def test_source_creation(self):
        """Test Source model creation."""
        source = Source(
            doc_id="1",
            doc_name="source.pdf",
            chunk_text="Source text",
            category="vector_db",
        )

        assert source.category == "vector_db"
        assert source.page_number is None

    def test_quality_assessment_creation(self):
        """Test QualityAssessment model creation."""
        qa = QualityAssessment(
            overall_score=350,
            factual_accuracy=90,
            semantic_validity=85,
            structural_integrity=88,
            citation_correctness=87,
            passes_quality=True,
        )

        assert qa.overall_score == 350
        assert qa.passes_quality is True

    def test_quality_assessment_validation(self):
        """Test QualityAssessment score validation."""
        with pytest.raises(ValueError):
            QualityAssessment(
                overall_score=600,  # Invalid: max is 500
                factual_accuracy=90,
                semantic_validity=85,
                structural_integrity=88,
                citation_correctness=87,
                passes_quality=False,
            )

    def test_finding_creation(self):
        """Test Finding model creation."""
        finding = Finding(
            claim="Dose limit is 20 mSv/year",
            evidence="StrlSchV § 78 states...",
            confidence="high",
        )

        assert finding.confidence == "high"
        assert finding.sources == []

    def test_final_report_creation(self):
        """Test FinalReport model creation."""
        report = FinalReport(
            query="What are dose limits?",
            answer="The dose limits are...",
            quality_score=350,
            todo_items_completed=5,
            research_iterations=3,
        )

        assert report.quality_score == 350
        assert report.findings == []


class TestHITLModels:
    """Test HITL-related models."""

    def test_clarification_question_creation(self):
        """Test ClarificationQuestion model creation."""
        question = ClarificationQuestion(
            id="q1",
            question="Which regulation are you interested in?",
            options=["StrlSchV", "StrlSchG", "Both"],
            context="Multiple regulations exist",
        )

        assert question.id == "q1"
        assert len(question.options) == 3
        assert question.answer is None

    def test_hitl_checkpoint_creation(self):
        """Test HITLCheckpoint model creation."""
        checkpoint = HITLCheckpoint(
            checkpoint_type="query_clarify",
            content={"questions": ["q1", "q2"]},
            phase="Phase 1",
        )

        assert checkpoint.checkpoint_type == "query_clarify"
        assert checkpoint.requires_approval is True

    def test_hitl_decision_creation(self):
        """Test HITLDecision model creation."""
        decision = HITLDecision(
            approved=True,
            feedback="Looks good",
        )

        assert decision.approved is True
        assert decision.modifications is None

    def test_hitl_state_operations(self):
        """Test HITLState model operations."""
        state = HITLState()

        assert state.pending is False
        assert state.history == []

        # Test add_to_history
        decision = HITLDecision(approved=True, feedback="OK")
        state.add_to_history("query_clarify", decision)
        assert len(state.history) == 1

        # Test clear_pending
        state.pending = True
        state.checkpoint = HITLCheckpoint(checkpoint_type="query_clarify")
        state.clear_pending()
        assert state.pending is False
        assert state.checkpoint is None


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_query_analysis_serialization(self):
        """Test QueryAnalysis serialization round-trip."""
        original = QueryAnalysis(
            original_query="test",
            key_concepts=["concept1", "concept2"],
            clarification_needed=True,
        )

        # Serialize to dict (for TypedDict state)
        data = original.model_dump()
        assert isinstance(data, dict)

        # Deserialize back
        restored = QueryAnalysis.model_validate(data)
        assert restored.original_query == original.original_query
        assert restored.key_concepts == original.key_concepts

    def test_research_context_json_schema(self):
        """Test ResearchContext JSON schema generation."""
        schema = ResearchContext.model_json_schema()

        assert "properties" in schema
        assert "search_queries" in schema["properties"]
        assert "metadata" in schema["properties"]
