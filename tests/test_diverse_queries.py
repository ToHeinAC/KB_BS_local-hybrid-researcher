"""Tests for _build_diverse_queries in hitl_service."""

from src.services.hitl_service import _build_diverse_queries


class TestBuildDiverseQueries:
    """Tests for _build_diverse_queries."""

    def test_basic_derivation(self):
        """Should combine refined_query and entity+scope question combos."""
        analysis = {
            "entities": ["StrlSchG", "StrlSchV"],
            "scope": "Dosisgrenzwerte",
            "refined_query": "Grenzwerte fuer berufliche Strahlenexposition",
        }
        result = _build_diverse_queries(analysis, "Strahlenexposition Grenzwerte")
        assert len(result) >= 3
        assert "Grenzwerte fuer berufliche Strahlenexposition" in result

    def test_excludes_user_query(self):
        """Result should NOT contain the user_query itself (Task 0 handles it)."""
        analysis = {
            "entities": ["StrlSchG"],
            "scope": "Dosisgrenzwerte",
            "refined_query": "Refined query about dose limits",
        }
        user_query = "Strahlenexposition Grenzwerte"
        result = _build_diverse_queries(analysis, user_query)
        assert user_query not in result

    def test_entity_scope_combinations_are_questions(self):
        """Entity + scope combos should be question-shaped (contain '?')."""
        analysis = {
            "entities": ["AtG", "KTA 1401"],
            "scope": "Genehmigung",
            "refined_query": "",
        }
        result = _build_diverse_queries(analysis, "Atomrecht")
        # Should be questions, not bare keyword combos
        for q in result:
            assert "?" in q, f"Expected question mark in: {q}"
        # Check entity appears in questions
        entity_texts = " ".join(result)
        assert "AtG" in entity_texts
        assert "KTA 1401" in entity_texts

    def test_dedup(self):
        """Duplicate queries (case-insensitive) should be removed."""
        analysis = {
            "entities": [],
            "scope": "",
            "refined_query": "strahlenexposition grenzwerte",
        }
        # refined_query matches user_query case-insensitively → both excluded
        result = _build_diverse_queries(analysis, "Strahlenexposition Grenzwerte")
        assert len(result) == 0

    def test_empty_analysis_fallback(self):
        """Empty analysis with user_query excluded should return empty list."""
        result = _build_diverse_queries({}, "test query")
        assert result == []

    def test_max_cap(self):
        """Should respect max_queries cap."""
        analysis = {
            "entities": ["A", "B", "C"],
            "scope": "scope",
            "refined_query": "refined",
        }
        result = _build_diverse_queries(analysis, "query", max_queries=3)
        assert len(result) <= 3

    def test_empty_user_query(self):
        """Empty user_query with analysis should still produce queries."""
        analysis = {
            "entities": ["StrlSchG"],
            "scope": "Grenzwerte",
            "refined_query": "Dosisgrenzwerte StrlSchG",
        }
        result = _build_diverse_queries(analysis, "")
        assert len(result) >= 1
        assert "Dosisgrenzwerte StrlSchG" in result

    def test_entities_without_scope_are_questions(self):
        """Entities without scope should appear as question-shaped queries."""
        analysis = {
            "entities": ["StrlSchG", "AtG"],
            "scope": "",
            "refined_query": "",
        }
        result = _build_diverse_queries(analysis, "query")
        assert len(result) >= 2
        for q in result:
            assert "?" in q, f"Expected question mark in: {q}"

    def test_questions_formulated_as_questions_de(self):
        """German output should contain '?' and German question words."""
        analysis = {
            "entities": ["StrlSchG"],
            "scope": "Dosisgrenzwerte",
            "refined_query": "",
        }
        result = _build_diverse_queries(analysis, "query", language="de")
        assert len(result) >= 1
        for q in result:
            assert "?" in q

    def test_questions_formulated_as_questions_en(self):
        """English output should contain '?' and English question words."""
        analysis = {
            "entities": ["StrlSchG"],
            "scope": "Dosisgrenzwerte",
            "refined_query": "",
        }
        result = _build_diverse_queries(analysis, "query", language="en")
        assert len(result) >= 1
        for q in result:
            assert "?" in q
        # Check English templates used
        combined = " ".join(result)
        assert "What" in combined

    def test_language_parameter_de_vs_en(self):
        """German and English should produce different templates."""
        analysis = {
            "entities": ["StrlSchG"],
            "scope": "Dosisgrenzwerte",
            "refined_query": "",
        }
        de_result = _build_diverse_queries(analysis, "query", language="de")
        en_result = _build_diverse_queries(analysis, "query", language="en")
        assert de_result != en_result
        # German should have "Welche" or "Was"
        de_combined = " ".join(de_result)
        assert "Welche" in de_combined or "Was" in de_combined
        # English should have "What"
        en_combined = " ".join(en_result)
        assert "What" in en_combined

    def test_scope_question_included(self):
        """Bare scope should become a question."""
        analysis = {
            "entities": [],
            "scope": "Dosisgrenzwerte",
            "refined_query": "",
        }
        result = _build_diverse_queries(analysis, "query", language="de")
        assert len(result) >= 1
        assert any("Dosisgrenzwerte" in q and "?" in q for q in result)
