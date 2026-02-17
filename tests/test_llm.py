"""Tests for mnemory.llm — LLM client helpers."""

from __future__ import annotations

import pytest

from mnemory.llm import _clean_response, parse_json_response

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
