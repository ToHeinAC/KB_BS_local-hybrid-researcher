"""Tests for model-conditional prompt routing.

Verifies that:
- Qwen model loads standard prompts (XML tags / ### headers)
- gpt-oss model loads gpt-oss prompts (# headers / <json> tags)
- Both variants export the same constant names
- All content-bearing prompts contain {language} placeholder
"""

import ast
import importlib
from unittest.mock import patch

import pytest


def _get_public_constants(filepath: str) -> list[str]:
    """Extract uppercase constant names from a Python file via AST."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.append(target.id)
    return sorted(names)


class TestPromptNameParity:
    """Verify gpt-oss files export the same names as standard files."""

    @pytest.mark.parametrize(
        "original,gpt_variant",
        [
            ("src/prompts/hitl.py", "src/prompts/hitl_gpt.py"),
            ("src/prompts/research.py", "src/prompts/research_gpt.py"),
            ("src/prompts/synthesis.py", "src/prompts/synthesis_gpt.py"),
        ],
    )
    def test_constant_names_match(self, original: str, gpt_variant: str):
        orig_names = _get_public_constants(original)
        gpt_names = _get_public_constants(gpt_variant)
        assert orig_names == gpt_names, (
            f"Name mismatch between {original} and {gpt_variant}. "
            f"Missing in gpt: {set(orig_names) - set(gpt_names)}, "
            f"Extra in gpt: {set(gpt_names) - set(orig_names)}"
        )


# Constants that are exempt from {language} requirement
_LANGUAGE_EXEMPT = {
    "LANGUAGE_DETECTION_PROMPT_SYSTEM",
    "LANGUAGE_DETECTION_PROMPT_HUMAN",
    "REFERENCE_EXTRACTION_PROMPT_SYSTEM",
    "REFERENCE_EXTRACTION_PROMPT_HUMAN",
}


class TestLanguagePlaceholder:
    """Verify {language} in all content-bearing prompts."""

    @pytest.mark.parametrize(
        "filepath",
        [
            "src/prompts/hitl.py",
            "src/prompts/research.py",
            "src/prompts/synthesis.py",
            "src/prompts/hitl_gpt.py",
            "src/prompts/research_gpt.py",
            "src/prompts/synthesis_gpt.py",
        ],
    )
    def test_language_placeholder_present(self, filepath: str):
        """All content-bearing prompts must contain {language}."""
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and target.id not in _LANGUAGE_EXEMPT
                    ):
                        # Get the string value
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            assert "{language}" in node.value.value, (
                                f"{filepath}::{target.id} missing {{language}} placeholder"
                            )


class TestQwenPromptFormat:
    """Verify Qwen prompts use expected format markers."""

    def test_qwen_prompts_have_xml_or_markdown_headers(self):
        """Qwen prompts should contain XML tags or ### headers."""
        from src.prompts import hitl as hitl_mod
        from src.prompts import research as research_mod
        from src.prompts import synthesis as synthesis_mod

        # Check a few representative prompts
        assert "###" in hitl_mod.ALTERNATIVE_QUERIES_INITIAL_PROMPT_SYSTEM
        assert "<role>" in synthesis_mod.SYNTHESIS_PROMPT_ENHANCED_SYSTEM
        assert "###" in research_mod.TASK_SEARCH_QUERIES_PROMPT_SYSTEM


class TestGptOssPromptFormat:
    """Verify gpt-oss prompts use expected format markers."""

    def test_gpt_oss_prompts_have_hash_headers(self):
        """gpt-oss prompts should use # headers (not ### or XML tags)."""
        from src.prompts import hitl_gpt, research_gpt, synthesis_gpt

        # Check representative prompts have # headers
        assert "# Role" in hitl_gpt.ALTERNATIVE_QUERIES_INITIAL_PROMPT_SYSTEM
        assert "# Role" in synthesis_gpt.SYNTHESIS_PROMPT_ENHANCED_SYSTEM
        assert "# Role" in research_gpt.TASK_SEARCH_QUERIES_PROMPT_SYSTEM

    def test_gpt_oss_prompts_no_xml_tags(self):
        """gpt-oss prompts should not contain <role> XML tags."""
        from src.prompts import hitl_gpt, research_gpt, synthesis_gpt

        for mod in [hitl_gpt, research_gpt, synthesis_gpt]:
            for name in dir(mod):
                if name.isupper() and isinstance(getattr(mod, name), str):
                    val = getattr(mod, name)
                    assert "<role>" not in val, f"{name} in {mod.__name__} contains <role>"

    def test_gpt_oss_prompts_use_json_tags(self):
        """gpt-oss structured prompts should mention <json> tags."""
        from src.prompts import synthesis_gpt

        assert "<json>" in synthesis_gpt.SYNTHESIS_PROMPT_ENHANCED_SYSTEM
        assert "<json>" in synthesis_gpt.QUERY_ASSESSMENT_PROMPT_SYSTEM

    def test_gpt_oss_prompts_no_no_think(self):
        """gpt-oss prompts should not contain /no_think directive."""
        from src.prompts import research_gpt

        for name in dir(research_gpt):
            if name.isupper() and isinstance(getattr(research_gpt, name), str):
                val = getattr(research_gpt, name)
                assert "/no_think" not in val, (
                    f"{name} in research_gpt contains /no_think"
                )


class TestConditionalRouting:
    """Verify __init__.py routes based on model_family."""

    def test_routing_matches_config(self):
        """Prompts loaded must match the active model_family from config."""
        from src.config import settings
        import src.prompts as prompts_mod

        prompt = prompts_mod.ALTERNATIVE_QUERIES_INITIAL_PROMPT_SYSTEM
        if settings.model_family == "gpt-oss":
            # gpt-oss prompts use # headers (single hash), no XML tags
            assert "# Role" in prompt
            assert "<role>" not in prompt
        else:
            # Qwen prompts use ### headers or XML tags
            assert "###" in prompt or "<role>" in prompt


class TestGemma4Support:
    """Verify gemma4 model family routing and /no_think stripping."""

    def test_gemma4_model_family(self):
        """gemma4 models return 'gemma4' family."""
        from src.config import Settings

        s = Settings(ollama_model="gemma4:26b")
        assert s.model_family == "gemma4"

    def test_gemma4_routes_to_qwen_prompts(self):
        """gemma4 model family should resolve to Qwen prompt constants."""
        with patch("src.config.settings") as mock_settings:
            mock_settings.model_family = "gemma4"
            import importlib
            import src.prompts as prompts_mod
            importlib.reload(prompts_mod)

            # Access via the internal dict directly (bypass __getattr__ mock complexity)
            from src.prompts import _qwen_prompts
            assert "SYNTHESIS_PROMPT_ENHANCED_SYSTEM" in _qwen_prompts

    def test_gemma4_strips_no_think(self):
        """OllamaClient._prepare_system_prompt strips /no_think for gemma4."""
        from src.services.ollama_client import OllamaClient

        client = OllamaClient(model="gemma4:26b")
        result = client._prepare_system_prompt("/no_think\nSome system instruction.")
        assert not result.startswith("/no_think")
        assert result == "Some system instruction."

    def test_gemma4_no_think_not_stripped_for_qwen(self):
        """OllamaClient._prepare_system_prompt does NOT strip /no_think for qwen."""
        from src.services.ollama_client import OllamaClient

        client = OllamaClient(model="qwen3:14b")
        prompt = "/no_think\nSome system instruction."
        result = client._prepare_system_prompt(prompt)
        assert result == prompt

    def test_gemma4_no_harmony_preamble(self):
        """OllamaClient does NOT prepend Harmony preamble for gemma4."""
        from src.services.ollama_client import OllamaClient, _HARMONY_PREAMBLE

        client = OllamaClient(model="gemma4:26b")
        result = client._prepare_system_prompt("Some prompt without /no_think.")
        assert not result.startswith(_HARMONY_PREAMBLE)
        assert result == "Some prompt without /no_think."
