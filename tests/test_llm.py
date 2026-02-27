"""Tests for mnemory.llm — LLM client helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mnemory.config import LLMConfig
from mnemory.llm import LLMClient, _clean_response, parse_json_response

# ── _clean_response ──────────────────────────────────────────────────


class TestCleanResponse:
    def test_plain_json(self):
        assert _clean_response('{"key": "value"}') == '{"key": "value"}'

    def test_strips_markdown_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert _clean_response(text) == '{"key": "value"}'

    def test_strips_markdown_fence_no_language(self):
        text = '```\n{"key": "value"}\n```'
        assert _clean_response(text) == '{"key": "value"}'

    def test_strips_think_blocks(self):
        text = '<think>Some reasoning here</think>{"key": "value"}'
        assert _clean_response(text) == '{"key": "value"}'

    def test_strips_multiline_think_blocks(self):
        text = '<think>\nStep 1\nStep 2\n</think>\n{"key": "value"}'
        result = _clean_response(text)
        assert '"key"' in result
        assert "<think>" not in result

    def test_strips_whitespace(self):
        assert _clean_response("  hello  ") == "hello"

    def test_empty_string(self):
        assert _clean_response("") == ""


# ── parse_json_response ──────────────────────────────────────────────


class TestParseJsonResponse:
    def test_valid_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"key": "value"} done.'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_response("not json at all")

    def test_array_json_raises(self):
        """parse_json_response expects a dict, not a list."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_response("[1, 2, 3]")

    def test_nested_json(self):
        text = '{"memories": [{"text": "hello", "action": "ADD"}]}'
        result = parse_json_response(text)
        assert "memories" in result
        assert len(result["memories"]) == 1

    def test_json_with_think_block(self):
        text = '<think>reasoning</think>{"key": "value"}'
        result = parse_json_response(text)
        assert result == {"key": "value"}


# ── LLMClient._build_params reasoning_effort ────────────────────────


class TestBuildParamsReasoningEffort:
    """Test per-call reasoning_effort override in _build_params."""

    @staticmethod
    def _make_client(reasoning_effort: str | None = None) -> LLMClient:
        config = LLMConfig(
            model="test-model",
            base_url="http://localhost",
            api_key="test-key",
            reasoning_effort=reasoning_effort,
        )
        with patch("mnemory.llm.OpenAI"):
            return LLMClient(config)

    def test_no_effort_set(self):
        """No instance or per-call effort → param not in output."""
        client = self._make_client(reasoning_effort=None)
        params = client._build_params([], None, 0.1, 2000)
        assert "reasoning_effort" not in params

    def test_instance_effort_used(self):
        """Instance-level effort is used when no per-call override."""
        client = self._make_client(reasoning_effort="minimal")
        params = client._build_params([], None, 0.1, 2000)
        assert params["reasoning_effort"] == "minimal"

    def test_per_call_overrides_instance(self):
        """Per-call effort overrides instance-level effort."""
        client = self._make_client(reasoning_effort="minimal")
        params = client._build_params([], None, 0.1, 2000, reasoning_effort="high")
        assert params["reasoning_effort"] == "high"

    def test_per_call_with_no_instance(self):
        """Per-call effort works even when instance has no default."""
        client = self._make_client(reasoning_effort=None)
        params = client._build_params([], None, 0.1, 2000, reasoning_effort="medium")
        assert params["reasoning_effort"] == "medium"

    def test_param_fix_suppresses_effort(self):
        """If model rejected reasoning_effort, it's omitted even if set."""
        client = self._make_client(reasoning_effort="high")
        client._param_fixes["reasoning_effort"] = "remove"
        params = client._build_params([], None, 0.1, 2000)
        assert "reasoning_effort" not in params

    def test_param_fix_suppresses_per_call_effort(self):
        """If model rejected reasoning_effort, per-call override is also omitted."""
        client = self._make_client(reasoning_effort=None)
        client._param_fixes["reasoning_effort"] = "remove"
        params = client._build_params([], None, 0.1, 2000, reasoning_effort="high")
        assert "reasoning_effort" not in params
