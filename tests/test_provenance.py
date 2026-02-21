"""Tests for reference provenance tracking (Phase: Rabbithole Provenance)."""

import pytest

from src.agents.nodes import _format_ranked_findings
from src.agents.tools import create_tiered_context_entry
from src.models.research import ChunkWithInfo, NestedChunk
from src.prompts import CHUNK_RERANKER_PROMPT_HUMAN


class TestNestedChunkProvenance:
    """Test that NestedChunk carries provenance fields with correct defaults."""

    def test_nested_chunk_default_provenance_fields(self):
        """NestedChunk defaults all provenance fields to empty/None."""
        nc = NestedChunk(chunk="text", document="doc.pdf")
        assert nc.parent_document == ""
        assert nc.parent_page is None
        assert nc.reference_original_text == ""
        assert nc.reference_type == ""
        assert nc.reference_surrounding_context == ""

    def test_nested_chunk_provenance_fields_set(self):
        """NestedChunk accepts and stores provenance fields."""
        nc = NestedChunk(
            chunk="radioactive waste storage text",
            document="KrWG.pdf",
            parent_document="StrlSchV.pdf",
            parent_page=45,
            reference_original_text="§21 KrWG",
            reference_type="legal_section",
            reference_surrounding_context="For waste disposal see Kreislaufwirtschaftsgesetz §21",
        )
        assert nc.parent_document == "StrlSchV.pdf"
        assert nc.parent_page == 45
        assert nc.reference_original_text == "§21 KrWG"
        assert nc.reference_type == "legal_section"
        assert "Kreislaufwirtschaftsgesetz" in nc.reference_surrounding_context


class TestCreateTieredContextEntryProvenance:
    """Test that create_tiered_context_entry stores provenance for depth>0 chunks."""

    def _make_chunk(self, text: str = "sample text", doc: str = "doc.pdf") -> ChunkWithInfo:
        return ChunkWithInfo(
            chunk=text,
            document=doc,
            page=1,
            extracted_info=text,
            relevance_score=0.8,
        )

    def test_depth0_has_no_provenance(self):
        """Depth-0 (direct vector search) entries must not have provenance keys."""
        chunk = self._make_chunk()
        entry = create_tiered_context_entry(
            chunk=chunk,
            tier=1,
            weight=1.0,
            depth=0,
            source_type="vector_search",
            parent_document="SomeParent.pdf",
            reference_original_text="§ 78",
        )
        assert "parent_document" not in entry
        assert "reference_original_text" not in entry

    def test_depth1_with_parent_document_stores_provenance(self):
        """Depth-1 entries with parent_document must have all provenance keys."""
        chunk = self._make_chunk("radioactive waste", "KrWG.pdf")
        entry = create_tiered_context_entry(
            chunk=chunk,
            tier=2,
            weight=0.7,
            depth=1,
            source_type="reference",
            parent_document="StrlSchV.pdf",
            parent_page=45,
            reference_original_text="§21 KrWG",
            reference_type="legal_section",
            reference_surrounding_context="For waste disposal see Kreislaufwirtschaftsgesetz §21 in this context",
        )
        assert entry["parent_document"] == "StrlSchV.pdf"
        assert entry["parent_page"] == 45
        assert entry["reference_original_text"] == "§21 KrWG"
        assert entry["reference_type"] == "legal_section"
        assert "Kreislaufwirtschaftsgesetz" in entry["reference_surrounding_context"]

    def test_depth1_without_parent_document_no_provenance(self):
        """Depth-1 entries without parent_document must not write provenance keys."""
        chunk = self._make_chunk()
        entry = create_tiered_context_entry(
            chunk=chunk,
            tier=2,
            weight=0.7,
            depth=1,
            source_type="reference",
            # parent_document is default ""
        )
        assert "parent_document" not in entry

    def test_surrounding_context_truncated_to_500(self):
        """reference_surrounding_context is truncated to 500 chars in the entry."""
        long_context = "x" * 1000
        chunk = self._make_chunk()
        entry = create_tiered_context_entry(
            chunk=chunk,
            tier=2,
            weight=0.7,
            depth=1,
            source_type="reference",
            parent_document="Parent.pdf",
            reference_surrounding_context=long_context,
        )
        assert len(entry["reference_surrounding_context"]) == 500


class TestFormatRankedFindingsProvenance:
    """Test that _format_ranked_findings shows provenance for depth>0 chunks."""

    def _make_ranked_chunk(self, with_provenance: bool) -> dict:
        base = {
            "_llm_score": 80,
            "_llm_reasoning": "Directly relevant to dose limits",
            "document": "KrWG.pdf",
            "page": 21,
            "extracted_info": "§21 KrWG: radioactive waste must be stored in certified facilities",
            "chunk": "radioactive waste chunk",
            "depth": 1 if with_provenance else 0,
        }
        if with_provenance:
            base["parent_document"] = "StrlSchV.pdf"
            base["parent_page"] = 45
            base["reference_original_text"] = "§21 KrWG"
            base["reference_type"] = "legal_section"
            base["reference_surrounding_context"] = (
                "Annual dose limit is 20 mSv. For waste disposal see §21 KrWG."
            )
        return base

    def test_format_ranked_findings_shows_via_ref_header(self):
        """Chunks with parent_document must show '[via ...' in output."""
        chunk = self._make_ranked_chunk(with_provenance=True)
        result = _format_ranked_findings([chunk])
        assert "[via" in result
        assert "StrlSchV.pdf" in result
        assert "§21 KrWG" in result

    def test_format_ranked_findings_shows_parent_context(self):
        """Chunks with reference_surrounding_context must show 'Parent context:' in output."""
        chunk = self._make_ranked_chunk(with_provenance=True)
        result = _format_ranked_findings([chunk])
        assert "Parent context:" in result
        assert "Annual dose limit" in result

    def test_format_ranked_findings_no_via_for_direct_chunks(self):
        """Direct vector search chunks (no parent_document) must NOT show '[via' header."""
        chunk = self._make_ranked_chunk(with_provenance=False)
        result = _format_ranked_findings([chunk])
        assert "[via" not in result
        assert "Parent context:" not in result


class TestChunkRerankerPromptParentContext:
    """Test that CHUNK_RERANKER_PROMPT_HUMAN accepts parent_context variable."""

    def test_reranker_human_prompt_with_parent_context(self):
        """CHUNK_RERANKER_PROMPT_HUMAN must format with parent_context field."""
        surrounding = "Annual dose limit is 20 mSv. For waste disposal see §21 KrWG."
        formatted = CHUNK_RERANKER_PROMPT_HUMAN.format(
            query="What are dose limits for radiation workers?",
            hitl_context="Focus on occupational exposure",
            text="§21 KrWG radioactive waste certified facilities",
            parent_context=surrounding,
            language="German",
        )
        assert surrounding in formatted
        assert "{parent_context}" not in formatted  # template variable must be substituted

    def test_reranker_human_prompt_with_na_parent_context(self):
        """CHUNK_RERANKER_PROMPT_HUMAN works with 'N/A (direct vector search result)'."""
        formatted = CHUNK_RERANKER_PROMPT_HUMAN.format(
            query="dose limits",
            hitl_context="No context",
            text="some text",
            parent_context="N/A (direct vector search result)",
            language="English",
        )
        assert "N/A (direct vector search result)" in formatted
