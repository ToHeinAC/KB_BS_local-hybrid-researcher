"""Tests for gpt-oss support in OllamaClient.

Verifies:
- Harmony preamble injection for gpt-oss models
- No preamble injection for Qwen models
- JSON tag extraction from <json>...</json> wrappers
- Temperature from config
- is_gpt_oss detection
"""

from unittest.mock import patch

import pytest

from src.services.ollama_client import (
    OllamaClient,
    _HARMONY_PREAMBLE,
    _JSON_TAG_RE,
)


class TestIsGptOss:
    """Test is_gpt_oss property."""

    def test_gpt_oss_20b(self):
        client = OllamaClient(model="gpt-oss:20b")
        assert client.is_gpt_oss is True

    def test_gpt_oss_variant(self):
        client = OllamaClient(model="gpt-oss:13b-q4")
        assert client.is_gpt_oss is True

    def test_qwen_model(self):
        client = OllamaClient(model="qwen3:14b")
        assert client.is_gpt_oss is False

    def test_other_model(self):
        client = OllamaClient(model="llama3:8b")
        assert client.is_gpt_oss is False


class TestHarmonyPreamble:
    """Test _prepare_system_prompt Harmony preamble."""

    def test_preamble_injected_for_gpt_oss(self):
        client = OllamaClient(model="gpt-oss:20b")
        prompt = "# Role\nYou are a test assistant."
        result = client._prepare_system_prompt(prompt)
        assert result.startswith(_HARMONY_PREAMBLE)
        assert prompt in result

    def test_preamble_not_injected_for_qwen(self):
        client = OllamaClient(model="qwen3:14b")
        prompt = "### Role\nYou are a test assistant."
        result = client._prepare_system_prompt(prompt)
        assert result == prompt
        assert _HARMONY_PREAMBLE not in result


class TestJsonTagExtraction:
    """Test _extract_json_from_tags static method."""

    def test_extracts_json_from_tags(self):
        text = 'Some preamble\n<json>{"key": "value"}</json>\nSome trailer'
        result = OllamaClient._extract_json_from_tags(text)
        assert result == '{"key": "value"}'

    def test_extracts_json_with_whitespace(self):
        text = '<json>\n  {"key": "value"}\n</json>'
        result = OllamaClient._extract_json_from_tags(text)
        assert result == '{"key": "value"}'

    def test_no_tags_returns_original(self):
        text = '{"key": "value"}'
        result = OllamaClient._extract_json_from_tags(text)
        assert result == text

    def test_empty_string(self):
        result = OllamaClient._extract_json_from_tags("")
        assert result == ""

    def test_multiline_json(self):
        text = '<json>{\n  "a": 1,\n  "b": [2, 3]\n}</json>'
        result = OllamaClient._extract_json_from_tags(text)
        assert '"a": 1' in result
        assert '"b": [2, 3]' in result


class TestTemperatureFromConfig:
    """Test that ChatOllama uses settings.ollama_temperature."""

    @patch("src.services.ollama_client.settings")
    def test_temperature_from_config(self, mock_settings):
        mock_settings.ollama_model = "qwen3:14b"
        mock_settings.ollama_fallback_model = "granite4.1:8b"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.ollama_num_ctx = 8192
        mock_settings.safe_context_limit = 7372
        mock_settings.ollama_temperature = 0.7

        client = OllamaClient()
        # Force LLM creation
        llm = client.llm
        assert llm.temperature == 0.7
