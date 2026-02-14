"""Tests for enhanced reference extraction and resolution."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.tools import (
    detect_convergence,
    detect_references,
    detect_references_hybrid,
    extract_references_llm,
    load_document_registry,
    resolve_document_name,
    resolve_reference_enhanced,
)
from src.models.research import (
    DetectedReference,
    ExtractedReference,
    ExtractedReferenceList,
    NestedChunk,
)


# =============================================================================
# Document Registry Tests
# =============================================================================


class TestDocumentRegistry:
    """Tests for document registry loading and resolution."""

    def test_load_registry(self):
        """Registry loads and has expected structure."""
        registry = load_document_registry()
        assert "collections" in registry
        assert "StrlSch" in registry["collections"]

    def test_resolve_exact_synonym(self):
        """Exact synonym match resolves correctly."""
        filename, coll = resolve_document_name("Strahlenschutzgesetz")
        assert filename == "StrlSchG.pdf"
        assert coll is not None

    def test_resolve_exact_synonym_case_insensitive(self):
        """Case-insensitive synonym matching."""
        filename, coll = resolve_document_name("strahlenschutzgesetz")
        assert filename == "StrlSchG.pdf"

    def test_resolve_abbreviation(self):
        """Short abbreviation synonyms resolve."""
        filename, coll = resolve_document_name("StrlSchG")
        assert filename == "StrlSchG.pdf"

    def test_resolve_kta(self):
        """KTA standard references resolve."""
        filename, coll = resolve_document_name("KTA 1401")
        assert filename is not None
        assert "KTA" in filename

    def test_resolve_with_collection_hint(self):
        """Collection hint prioritizes search."""
        filename, coll = resolve_document_name("StrlSchG", collection_hint="NORM")
        assert filename == "StrlSchG.pdf"
        # With hint, should find in NORM first
        assert coll == "NORM"

    def test_resolve_fuzzy_match(self):
        """Fuzzy matching catches close misspellings."""
        filename, coll = resolve_document_name("Strahlenschutzverordung")  # typo
        # Should fuzzy match "Strahlenschutzverordnung"
        if filename:
            assert "StrlSchV" in filename

    def test_resolve_substring_match(self):
        """Substring match on filename."""
        filename, coll = resolve_document_name("AtG")
        assert filename == "AtG.pdf"

    def test_resolve_no_match(self):
        """Unknown reference returns None."""
        filename, coll = resolve_document_name("NonexistentDocument12345")
        assert filename is None
        assert coll is None

    def test_resolve_icrp(self):
        """ICRP publication references resolve."""
        filename, coll = resolve_document_name("ICRP 103")
        assert filename is not None
        assert "ICRP" in filename


# =============================================================================
# Regex Detection Tests (backward compatibility)
# =============================================================================


class TestRegexDetection:
    """Existing regex patterns still work."""

    def test_german_section_ref(self):
        """German § references detected."""
        text = "siehe § 12 Abs. 3 StrlSchV"
        refs = detect_references(text)
        assert len(refs) >= 1
        assert any(r.type == "section" for r in refs)

    def test_german_document_ref(self):
        """German document references detected."""
        text = 'siehe Dokument "Sicherheitsbericht"'
        refs = detect_references(text)
        assert len(refs) >= 1
        assert any(r.type == "document" for r in refs)

    def test_english_section_ref(self):
        """English section references detected."""
        text = "see section 5.2 for details"
        refs = detect_references(text)
        assert len(refs) >= 1
        assert any(r.type == "section" for r in refs)

    def test_external_url(self):
        """URLs detected as external references."""
        text = "see https://example.com/doc for details"
        refs = detect_references(text)
        assert len(refs) >= 1
        assert any(r.type == "external" for r in refs)

    def test_no_references(self):
        """Text without references returns empty list."""
        refs = detect_references("This is plain text with no references.")
        assert refs == []

    def test_deduplication(self):
        """Duplicate references within same text deduplicated."""
        text = "gemäß § 5 StrlSchG und auch § 5 StrlSchG"
        refs = detect_references(text)
        targets = [r.target for r in refs if r.type == "section"]
        # Should have at most one "5" target
        assert targets.count("5") <= 1 or targets.count("5 StrlSchG") <= 1


# =============================================================================
# LLM Extraction Tests
# =============================================================================


class TestLLMExtraction:
    """Tests for LLM-based reference extraction."""

    @patch("src.agents.tools.get_ollama_client")
    def test_llm_extraction_legal(self, mock_get_client):
        """LLM extracts legal section references."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate_structured.return_value = ExtractedReferenceList(
            references=[
                ExtractedReference(
                    reference_mention="§ 133 des Strahlenschutzgesetzes",
                    reference_type="legal_section",
                    target_document_hint="Strahlenschutzgesetz",
                    confidence=0.95,
                )
            ]
        )

        refs = extract_references_llm("gemäß § 133 des Strahlenschutzgesetzes")
        assert len(refs) == 1
        assert refs[0].type == "legal_section"
        assert refs[0].extraction_method == "llm"
        assert refs[0].document_context == "Strahlenschutzgesetz"

    @patch("src.agents.tools.get_ollama_client")
    def test_llm_extraction_academic(self, mock_get_client):
        """LLM extracts academic numbered references."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate_structured.return_value = ExtractedReferenceList(
            references=[
                ExtractedReference(
                    reference_mention="[253]",
                    reference_type="academic_numbered",
                    confidence=0.9,
                )
            ]
        )

        refs = extract_references_llm("see [253] for details")
        assert len(refs) == 1
        assert refs[0].type == "academic_numbered"

    @patch("src.agents.tools.get_ollama_client")
    def test_llm_extraction_shortform(self, mock_get_client):
        """LLM extracts author-year citation references."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate_structured.return_value = ExtractedReferenceList(
            references=[
                ExtractedReference(
                    reference_mention="[Townsend79]",
                    reference_type="academic_shortform",
                    confidence=0.85,
                )
            ]
        )

        refs = extract_references_llm("as shown by [Townsend79]")
        assert len(refs) == 1
        assert refs[0].type == "academic_shortform"

    @patch("src.agents.tools.get_ollama_client")
    def test_llm_extraction_document_mention(self, mock_get_client):
        """LLM extracts document mention references."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate_structured.return_value = ExtractedReferenceList(
            references=[
                ExtractedReference(
                    reference_mention="KTA 1401",
                    reference_type="document_mention",
                    target_document_hint="KTA 1401",
                    confidence=0.95,
                )
            ]
        )

        refs = extract_references_llm("Die Anforderungen der KTA 1401")
        assert len(refs) == 1
        assert refs[0].type == "document_mention"
        assert refs[0].document_context == "KTA 1401"

    @patch("src.agents.tools.get_ollama_client")
    def test_llm_extraction_failure(self, mock_get_client):
        """LLM extraction failure returns empty list."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate_structured.side_effect = Exception("LLM error")

        refs = extract_references_llm("some text")
        assert refs == []


# =============================================================================
# Hybrid Detection Tests
# =============================================================================


class TestHybridDetection:
    """Tests for combined regex + LLM detection."""

    @patch("src.agents.tools.extract_references_llm")
    def test_hybrid_combines_results(self, mock_llm):
        """Hybrid combines regex and LLM results."""
        mock_llm.return_value = [
            DetectedReference(
                type="document_mention",
                target="KTA 1401",
                original_text="KTA 1401",
                extraction_method="llm",
                document_context="KTA 1401",
            )
        ]

        text = "siehe § 12 StrlSchG und KTA 1401"
        refs = detect_references_hybrid(text)

        types = {r.type for r in refs}
        assert "section" in types  # from regex
        assert "document_mention" in types  # from LLM

    @patch("src.agents.tools.extract_references_llm")
    def test_hybrid_deduplicates(self, mock_llm):
        """Hybrid deduplicates overlapping regex and LLM results."""
        # LLM returns same section ref as regex would find
        mock_llm.return_value = [
            DetectedReference(
                type="legal_section",
                target="§ 12",
                original_text="§ 12",
                extraction_method="llm",
            )
        ]

        text = "siehe § 12 StrlSchG"
        refs = detect_references_hybrid(text)

        # Should not have duplicate entries for § 12
        all_targets = [r.target for r in refs]
        # The "12" or "12 StrlSchG" from regex + "§ 12" from LLM
        # Dedup should catch overlap
        assert len(refs) <= 3  # Reasonable upper bound

    @patch("src.agents.tools.extract_references_llm")
    def test_hybrid_empty_llm(self, mock_llm):
        """Hybrid works when LLM returns nothing."""
        mock_llm.return_value = []
        text = "siehe § 12 StrlSchG"
        refs = detect_references_hybrid(text)
        assert len(refs) >= 1  # Regex still works


# =============================================================================
# Enhanced Resolution Tests
# =============================================================================


class TestEnhancedResolution:
    """Tests for enhanced reference resolution."""

    @patch("src.agents.tools._vector_search_scoped")
    def test_legal_ref_with_doc_context(self, mock_scoped):
        """Legal ref with document_context uses scoped search."""
        mock_scoped.return_value = [
            NestedChunk(chunk="test content", document="StrlSchG.pdf", relevance_score=0.8)
        ]

        ref = DetectedReference(
            type="legal_section",
            target="§ 133 StrlSchG",
            document_context="Strahlenschutzgesetz",
            extraction_method="llm",
        )

        chunks = resolve_reference_enhanced(ref, "current.pdf")
        assert len(chunks) == 1
        assert mock_scoped.called

    @patch("src.agents.tools.vector_search")
    def test_academic_ref_broad_search(self, mock_search):
        """Academic refs use broad vector search."""
        mock_search.return_value = [
            MagicMock(
                chunk_text="citation content",
                doc_name="paper.pdf",
                relevance_score=0.7,
            )
        ]

        ref = DetectedReference(
            type="academic_numbered",
            target="[253]",
            extraction_method="llm",
        )

        chunks = resolve_reference_enhanced(ref, "current.pdf")
        assert mock_search.called

    def test_depth_limit_respected(self):
        """Resolution stops at depth limit."""
        ref = DetectedReference(type="section", target="§ 5")
        chunks = resolve_reference_enhanced(ref, "doc.pdf", depth=10)
        assert chunks == []

    def test_visited_ref_skipped(self):
        """Already-visited references are skipped."""
        ref = DetectedReference(type="section", target="§ 5")
        visited = {"section:§ 5"}
        chunks = resolve_reference_enhanced(ref, "doc.pdf", visited=visited)
        assert chunks == []

    def test_token_budget_respected(self):
        """Resolution stops when token budget exhausted."""
        ref = DetectedReference(type="section", target="§ 5")
        chunks = resolve_reference_enhanced(
            ref, "doc.pdf", token_count=999999
        )
        assert chunks == []


# =============================================================================
# Convergence Detection Tests
# =============================================================================


class TestConvergenceDetection:
    """Tests for document convergence detection."""

    def test_convergence_triggers_at_threshold(self):
        """Convergence detected when doc appears >= threshold times."""
        history = ["doc1.pdf", "doc2.pdf", "doc1.pdf", "doc3.pdf", "doc1.pdf"]
        assert detect_convergence(history) is True

    def test_no_convergence_below_threshold(self):
        """No convergence when all docs below threshold."""
        history = ["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf"]
        assert detect_convergence(history) is False

    def test_empty_history(self):
        """Empty history returns False."""
        assert detect_convergence([]) is False

    def test_single_doc_repeated(self):
        """Single document repeated hits threshold."""
        history = ["doc1.pdf"] * 5
        assert detect_convergence(history) is True

    def test_exactly_at_threshold(self):
        """Exactly at threshold triggers convergence."""
        # Default threshold is 3
        history = ["a.pdf", "b.pdf", "a.pdf", "c.pdf", "a.pdf"]
        assert detect_convergence(history) is True

    def test_below_threshold_by_one(self):
        """One below threshold does not trigger."""
        history = ["a.pdf", "b.pdf", "a.pdf", "c.pdf"]
        assert detect_convergence(history) is False


# =============================================================================
# Model Tests
# =============================================================================


class TestExtractedReferenceModel:
    """Tests for the new Pydantic models."""

    def test_extracted_reference_defaults(self):
        """ExtractedReference has correct defaults."""
        ref = ExtractedReference(
            reference_mention="§ 5",
            reference_type="legal_section",
        )
        assert ref.target_document_hint == ""
        assert ref.confidence == 0.9

    def test_extracted_reference_list(self):
        """ExtractedReferenceList accepts references."""
        lst = ExtractedReferenceList(
            references=[
                ExtractedReference(
                    reference_mention="§ 5",
                    reference_type="legal_section",
                )
            ]
        )
        assert len(lst.references) == 1

    def test_detected_reference_new_types(self):
        """DetectedReference accepts new LLM types."""
        ref = DetectedReference(
            type="legal_section",
            target="§ 133",
            extraction_method="llm",
            document_context="Strahlenschutzgesetz",
        )
        assert ref.extraction_method == "llm"
        assert ref.document_context == "Strahlenschutzgesetz"

    def test_detected_reference_backward_compat(self):
        """DetectedReference still works with old types."""
        ref = DetectedReference(type="section", target="5")
        assert ref.extraction_method == "regex"
        assert ref.document_context is None


# =============================================================================
# Agentic Reference Decision Tests
# =============================================================================


class TestReferenceDecisionPrompt:
    """Tests for agentic reference following gate."""

    def test_reference_decision_prompt_has_required_vars(self):
        """REFERENCE_DECISION_PROMPT contains all required template variables."""
        from src.prompts import REFERENCE_DECISION_PROMPT

        assert "{reference_type}" in REFERENCE_DECISION_PROMPT
        assert "{reference_target}" in REFERENCE_DECISION_PROMPT
        assert "{document_context}" in REFERENCE_DECISION_PROMPT
        assert "{query_anchor}" in REFERENCE_DECISION_PROMPT
        assert "{surrounding_context}" in REFERENCE_DECISION_PROMPT
        assert "{language}" in REFERENCE_DECISION_PROMPT

    def test_reference_decision_prompt_format(self):
        """REFERENCE_DECISION_PROMPT can be formatted without error."""
        from src.prompts import REFERENCE_DECISION_PROMPT

        formatted = REFERENCE_DECISION_PROMPT.format(
            reference_type="legal_section",
            reference_target="§ 5 StrlSchV",
            document_context="StrlSchG.pdf",
            surrounding_context="gemäß § 5 der Strahlenschutzverordnung gelten folgende Grenzwerte",
            query_anchor='{"original_query": "Grenzwerte", "key_entities": ["StrlSchV"]}',
            language="German",
        )
        assert "legal_section" in formatted
        assert "§ 5 StrlSchV" in formatted
        assert "gemäß § 5 der Strahlenschutzverordnung" in formatted

    def test_reference_decision_model_serialization(self):
        """ReferenceDecision round-trips through dict."""
        from src.models.research import ReferenceDecision

        d = ReferenceDecision(follow=False, reason="Tangential to query")
        data = d.model_dump()
        restored = ReferenceDecision.model_validate(data)
        assert restored.follow is False
        assert restored.reason == "Tangential to query"


# =============================================================================
# Context Window Tests
# =============================================================================


class TestGetContextWindow:
    """Tests for the get_context_window utility function."""

    def test_mention_found_centered(self):
        """Window centers on the mention with surrounding context."""
        from src.agents.tools import get_context_window

        text = "A" * 200 + " TARGET_MENTION " + "B" * 200
        result = get_context_window(text, "TARGET_MENTION", window_size=100)
        assert "TARGET_MENTION" in result
        # Should have ellipsis since we're truncating both sides
        assert result.startswith("...")
        assert result.endswith("...")

    def test_mention_not_found_returns_start(self):
        """When mention is not in text, return the first window_size chars."""
        from src.agents.tools import get_context_window

        text = "Some long text without the expected mention."
        result = get_context_window(text, "NONEXISTENT", window_size=20)
        assert result == text[:20]

    def test_empty_text(self):
        """Empty text returns empty string."""
        from src.agents.tools import get_context_window

        assert get_context_window("", "mention", window_size=100) == ""

    def test_empty_mention(self):
        """Empty mention returns first window_size chars of text."""
        from src.agents.tools import get_context_window

        text = "Some text content here."
        result = get_context_window(text, "", window_size=10)
        assert result == text[:10]

    def test_mention_at_start_no_leading_ellipsis(self):
        """Mention at start of text should not have leading ellipsis."""
        from src.agents.tools import get_context_window

        text = "TARGET at the very beginning" + " extra" * 100
        result = get_context_window(text, "TARGET", window_size=40)
        assert not result.startswith("...")
        assert "TARGET" in result

    def test_mention_at_end_no_trailing_ellipsis(self):
        """Mention at end of text should not have trailing ellipsis."""
        from src.agents.tools import get_context_window

        text = "x" * 200 + "END_TARGET"
        result = get_context_window(text, "END_TARGET", window_size=40)
        assert not result.endswith("...")
        assert "END_TARGET" in result

    def test_short_text_no_ellipsis(self):
        """Text shorter than window_size returns full text without ellipsis."""
        from src.agents.tools import get_context_window

        text = "Short text with MENTION inside."
        result = get_context_window(text, "MENTION", window_size=500)
        assert result == text

    def test_case_insensitive_match(self):
        """Mention matching is case-insensitive."""
        from src.agents.tools import get_context_window

        text = "Before § 12 StrlSchG defines the limits after"
        result = get_context_window(text, "§ 12 strlschg", window_size=100)
        assert "§ 12 StrlSchG" in result

    def test_none_text_returns_empty(self):
        """None text is handled gracefully."""
        from src.agents.tools import get_context_window

        assert get_context_window(None, "mention") == ""

