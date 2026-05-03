"""Tests for synthesis reliability fixes across Ollama models.

Covers:
- _repair_truncated_json() utility
- num_predict_override parameter threading
- gpt-oss primary strategy (raw invoke + tag extraction)
- Truncation detection logging
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.ollama_client import OllamaClient


class TestRepairTruncatedJson:
    """Tests for OllamaClient._repair_truncated_json()."""

    def test_valid_json_passthrough(self):
        """Valid JSON is returned unchanged."""
        valid = '{"summary": "hello", "key_findings": ["a"]}'
        result = OllamaClient._repair_truncated_json(valid)
        assert json.loads(result) == json.loads(valid)

    def test_empty_input(self):
        """Empty input is returned as-is."""
        assert OllamaClient._repair_truncated_json("") == ""
        assert OllamaClient._repair_truncated_json("  ") == "  "

    def test_truncated_string_value(self):
        """Truncated string value is repaired by closing at last complete string."""
        truncated = '{"summary": "This is a report about radia'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert "summary" in parsed

    def test_unclosed_object(self):
        """Unclosed object brace is appended."""
        truncated = '{"summary": "hello"'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["summary"] == "hello"

    def test_unclosed_array(self):
        """Unclosed array bracket is appended."""
        truncated = '{"items": ["a", "b"'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b"]

    def test_unclosed_nested_structures(self):
        """Multiple unclosed nesting levels are closed."""
        truncated = '{"outer": {"inner": ["val"'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == ["val"]

    def test_truncated_mid_key(self):
        """JSON truncated mid-key still produces parseable result."""
        truncated = '{"summary": "done", "key_fin'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["summary"] == "done"

    def test_escaped_quotes_not_confused(self):
        """Escaped quotes inside strings don't confuse the parser."""
        valid = '{"text": "He said \\"hello\\" to me"}'
        result = OllamaClient._repair_truncated_json(valid)
        parsed = json.loads(result)
        assert "hello" in parsed["text"]

    def test_trailing_comma_removed(self):
        """Trailing comma after truncated string is cleaned up."""
        truncated = '{"a": "x", "b": "truncated value here...'
        result = OllamaClient._repair_truncated_json(truncated)
        parsed = json.loads(result)
        assert parsed["a"] == "x"


class TestNumPredictOverride:
    """Tests for num_predict_override parameter threading."""

    @patch("src.services.ollama_client.ChatOllama")
    def test_override_creates_temporary_llm(self, mock_chat_ollama):
        """num_predict_override creates a temporary ChatOllama instance."""
        from pydantic import BaseModel

        class SimpleModel(BaseModel):
            text: str

        mock_instance = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SimpleModel(text="hello")
        mock_instance.with_structured_output.return_value = mock_structured
        mock_chat_ollama.return_value = mock_instance

        client = OllamaClient.__new__(OllamaClient)
        client.model = "qwen3:14b"
        client.fallback_model = "nemotron-3-nano:4b"
        client.base_url = "http://localhost:11434"
        client.num_ctx = 131072
        client.safe_limit = 117964
        client._llm = MagicMock()
        client._fallback_llm = None

        # Call with override
        try:
            client.generate_structured_messages(
                "system", "human", SimpleModel,
                num_predict_override=16384,
            )
        except Exception:
            pass  # We just want to check ChatOllama was called

        # Verify a new ChatOllama was created with the override
        mock_chat_ollama.assert_called_once()
        call_kwargs = mock_chat_ollama.call_args[1]
        assert call_kwargs["num_predict"] == 16384
        assert call_kwargs["timeout"] == 300

    def test_no_override_uses_default_llm(self):
        """Without num_predict_override, the default self.llm is used."""
        from pydantic import BaseModel

        class SimpleModel(BaseModel):
            text: str

        client = OllamaClient.__new__(OllamaClient)
        client.model = "qwen3:14b"
        client.fallback_model = "nemotron-3-nano:4b"
        client.base_url = "http://localhost:11434"
        client.num_ctx = 131072
        client.safe_limit = 117964

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SimpleModel(text="hello")
        mock_llm.with_structured_output.return_value = mock_structured
        client._llm = mock_llm
        client._fallback_llm = None

        result = client.generate_structured_messages(
            "system", "human", SimpleModel,
        )
        assert result.text == "hello"
        # Verify the default llm was used (with_structured_output called on it)
        mock_llm.with_structured_output.assert_called_once()


class TestGptOssPrimaryStrategy:
    """Tests for gpt-oss raw invoke + tag extraction as primary strategy."""

    def test_gpt_oss_uses_raw_invoke_first(self):
        """gpt-oss models try raw invoke + tag extraction before structured_llm."""
        from pydantic import BaseModel

        class SimpleModel(BaseModel):
            text: str

        client = OllamaClient.__new__(OllamaClient)
        client.model = "gpt-oss:20b"
        client.fallback_model = "nemotron-3-nano:4b"
        client.base_url = "http://localhost:11434"
        client.num_ctx = 131072
        client.safe_limit = 117964

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '<json>{"text": "from tags"}</json>'
        mock_response.response_metadata = {"done_reason": "stop"}
        mock_llm.invoke.return_value = mock_response
        client._llm = mock_llm
        client._fallback_llm = None

        result = client.generate_structured_messages(
            "system", "human", SimpleModel,
        )
        assert result.text == "from tags"
        # raw invoke should be called, NOT with_structured_output
        mock_llm.invoke.assert_called_once()
        mock_llm.with_structured_output.assert_not_called()


class TestTruncationDetection:
    """Tests for truncation detection logging."""

    def test_log_truncation_warns_on_length(self):
        """_log_truncation logs warning when done_reason is 'length'."""
        client = OllamaClient.__new__(OllamaClient)
        client.model = "test-model"

        mock_response = MagicMock()
        mock_response.response_metadata = {"done_reason": "length"}

        with patch("src.services.ollama_client.logger") as mock_logger:
            client._log_truncation(mock_response)
            mock_logger.warning.assert_called_once()
            assert "truncated" in mock_logger.warning.call_args[0][0].lower()

    def test_log_truncation_silent_on_stop(self):
        """_log_truncation does not log when done_reason is 'stop'."""
        client = OllamaClient.__new__(OllamaClient)
        client.model = "test-model"

        mock_response = MagicMock()
        mock_response.response_metadata = {"done_reason": "stop"}

        with patch("src.services.ollama_client.logger") as mock_logger:
            client._log_truncation(mock_response)
            mock_logger.warning.assert_not_called()

    def test_log_truncation_handles_missing_metadata(self):
        """_log_truncation handles responses without response_metadata."""
        client = OllamaClient.__new__(OllamaClient)
        client.model = "test-model"

        mock_response = MagicMock(spec=[])  # no attributes

        with patch("src.services.ollama_client.logger") as mock_logger:
            client._log_truncation(mock_response)
            mock_logger.warning.assert_not_called()


class TestSynthesisNumPredictConfig:
    """Tests for the synthesis-specific num_predict configuration."""

    def test_config_has_synthesis_num_predict(self):
        """Settings includes ollama_num_predict_synthesis field."""
        from src.config import Settings
        s = Settings()
        assert hasattr(s, "ollama_num_predict_synthesis")
        assert s.ollama_num_predict_synthesis == 16384

    def test_synthesis_higher_than_default(self):
        """Synthesis num_predict is higher than the default."""
        from src.config import Settings
        s = Settings()
        assert s.ollama_num_predict_synthesis > s.ollama_num_predict
