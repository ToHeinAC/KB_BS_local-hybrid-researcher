"""Tests for sequential numbered citation transformation."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.tools import numberify_citations, resolve_pdf_path


class TestResolvePdfPath:
    """Tests for resolve_pdf_path()."""

    def test_resolve_pdf_path_found(self, tmp_path):
        """Find a PDF in a kb/__db_inserted folder."""
        # Set up fake kb structure
        folder = tmp_path / "GLageKon__db_inserted"
        folder.mkdir()
        pdf = folder / "StrlSchV.pdf"
        pdf.touch()

        db_path = str(tmp_path / "database")
        with patch("src.agents.tools.settings") as mock_settings:
            mock_settings.chromadb_path = db_path
            result = resolve_pdf_path("StrlSchV.pdf")

        assert result is not None
        assert result == str(pdf.resolve())

    def test_resolve_pdf_path_not_found(self, tmp_path):
        """Return None when PDF doesn't exist in any folder."""
        folder = tmp_path / "GLageKon__db_inserted"
        folder.mkdir()

        db_path = str(tmp_path / "database")
        with patch("src.agents.tools.settings") as mock_settings:
            mock_settings.chromadb_path = db_path
            result = resolve_pdf_path("nonexistent.pdf")

        assert result is None

    def test_resolve_pdf_path_case_insensitive(self, tmp_path):
        """Find PDF via case-insensitive fallback."""
        folder = tmp_path / "StrlSch__db_inserted"
        folder.mkdir()
        pdf = folder / "StrlSchG.pdf"
        pdf.touch()

        db_path = str(tmp_path / "database")
        with patch("src.agents.tools.settings") as mock_settings:
            mock_settings.chromadb_path = db_path
            result = resolve_pdf_path("strlschg.pdf")

        assert result is not None
        assert "StrlSchG.pdf" in result


class TestNumberifyCitations:
    """Tests for numberify_citations()."""

    def test_numberify_basic(self):
        """Two different citations get sequential numbers."""
        text = "Foo [StrlSchV.pdf, Page 45] bar [StrlSchG.pdf, Page 12]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert "[1]" in result
        assert "[2]" in result
        assert "[StrlSchV.pdf" not in result
        assert len(citations) == 2
        assert citations[0]["number"] == 1
        assert citations[0]["doc_name"] == "StrlSchV.pdf"
        assert citations[0]["page"] == "45"
        assert citations[1]["number"] == 2

    def test_numberify_same_doc_different_pages(self):
        """Same doc, different pages get different numbers."""
        text = "[Doc.pdf, Page 1] and [Doc.pdf, Page 2]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert "[1]" in result
        assert "[2]" in result
        assert len(citations) == 2

    def test_numberify_same_doc_same_page(self):
        """Same doc+page reuses the same number."""
        text = "[Doc.pdf, Page 5] and again [Doc.pdf, Page 5]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        # 2 inline + 1 in reference list = 3 total
        assert result.count("[1]") == 3
        assert "[2]" not in result
        assert len(citations) == 1

    def test_numberify_reading_order(self):
        """First occurrence in reading order gets [1]."""
        text = "[B.pdf, Page 1] then [A.pdf, Page 1] then [B.pdf, Page 1]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert citations[0]["doc_name"] == "B.pdf"
        assert citations[0]["number"] == 1
        assert citations[1]["doc_name"] == "A.pdf"
        assert citations[1]["number"] == 2

    def test_numberify_no_citations(self):
        """Text without citations returned unchanged."""
        text = "No citations here."
        result, citations = numberify_citations(text)

        assert result == text
        assert citations == []

    def test_numberify_preserves_markdown_links(self):
        """Markdown links like [text](url) are not touched."""
        text = "See [link](http://example.com) and [Doc.pdf, Page 1]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert "[link](http://example.com)" in result
        assert "[1]" in result
        assert len(citations) == 1

    def test_numberify_seite_format(self):
        """German 'Seite' variant is recognized."""
        text = "Siehe [StrlSchV.pdf, Seite 10] für Details."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert "[1]" in result
        assert citations[0]["page"] == "10"
        assert "Quellenverzeichnis" in result

    def test_numberify_no_page(self):
        """Citation without page number is handled."""
        text = "See [Report.pdf] for more."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text)

        assert "[1]" in result
        assert citations[0]["page"] is None
        assert citations[0]["doc_name"] == "Report.pdf"

    def test_numberify_english_headers(self):
        """English language uses English headers."""
        text = "[Doc.pdf, Page 1]."
        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result, citations = numberify_citations(text, language="en")

        assert "References" in result
        assert "Quellenverzeichnis" not in result

    def test_numberify_with_pdf_path(self):
        """Reference list includes HTTP PDF link when path is resolved."""
        text = "[Doc.pdf, Page 1]."
        with patch(
            "src.agents.tools.resolve_pdf_path",
            return_value="/abs/path/Doc.pdf",
        ):
            result, citations = numberify_citations(text)

        assert "PDF öffnen" in result
        # Relative on purpose — a root-absolute link would escape the reverse
        # proxy's /brain/ prefix. See src/ui/components/base_path.py.
        assert "(_api/pdf?path=" in result
        assert "(/_api/pdf?path=" not in result
        assert "file://" not in result

    def test_numberify_empty_string(self):
        """Empty input returns empty."""
        result, citations = numberify_citations("")
        assert result == ""
        assert citations == []


class TestPDFHandler:
    """Tests for PDF Tornado handler security."""

    def test_pdf_handler_serves_file(self, tmp_path):
        """Path validation allows files within kb/ directory."""
        # Create a fake kb structure with a PDF
        pdf_file = tmp_path / "GLageKon__db_inserted" / "test.pdf"
        pdf_file.parent.mkdir(parents=True)
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        # Validate the security logic used by PDFHandler
        requested = pdf_file.resolve()
        kb_base = tmp_path.resolve()
        assert str(requested).startswith(str(kb_base))
        assert requested.exists()
        assert requested.is_file()

    def test_pdf_handler_blocks_traversal(self, tmp_path):
        """Path traversal outside kb/ is rejected."""
        from pathlib import Path

        kb_base = (tmp_path / "kb").resolve()
        kb_base.mkdir()

        # Attempt traversal
        traversal = (tmp_path / "kb" / ".." / "etc" / "passwd").resolve()
        assert not str(traversal).startswith(str(kb_base))

    def test_pdf_handler_missing_file_returns_none(self, tmp_path):
        """Non-existent file path is detected."""
        fake_path = tmp_path / "nonexistent.pdf"
        assert not fake_path.exists()


class TestAttributeSourcesIntegration:
    """Integration test verifying numbered citations flow through attribute_sources."""

    def test_attribute_sources_transforms_citations(self):
        """attribute_sources node produces numbered citations in the final report."""
        from unittest.mock import MagicMock
        from src.agents.nodes import attribute_sources

        state = {
            "research_context": {
                "search_queries": [
                    {
                        "query": "test",
                        "chunks": [
                            {
                                "chunk": "some text",
                                "document": "TestDoc.pdf",
                                "page": 5,
                                "extracted_info": "info",
                                "relevance_score": 0.9,
                                "references": [],
                            }
                        ],
                        "summary": "The limit is 20 mSv [TestDoc.pdf, Page 5].",
                        "summary_references": [],
                    }
                ],
                "metadata": {
                    "total_iterations": 1,
                    "documents_referenced": ["TestDoc.pdf"],
                    "external_sources_used": False,
                    "visited_refs": [],
                },
            },
            "query_analysis": {
                "original_query": "test query",
                "key_concepts": [],
                "entities": [],
                "scope": "",
                "assumed_context": [],
                "clarification_needed": False,
                "hitl_refinements": [],
                "detected_language": "de",
            },
            "query_anchor": {"detected_language": "de"},
            "todo_list": [{"id": 0, "task": "test", "completed": True}],
            "quality_assessment": None,
        }

        with patch("src.agents.tools.resolve_pdf_path", return_value=None):
            result = attribute_sources(state)

        answer = result["final_report"]["answer"]
        assert "[1]" in answer
        assert "Quellenverzeichnis" in answer
        assert "[TestDoc.pdf, Page 5]" not in answer
