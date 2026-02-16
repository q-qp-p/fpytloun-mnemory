"""Tests for mnemory.classify module — classification logic and category cache."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from mnemory.classify import (
    CategoryCache,
    _build_system_prompt,
    _validate_classification,
    classify_memory,
)
from mnemory.config import LLMConfig

# ── _build_system_prompt ───────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_all_fields(self):
        prompt = _build_system_prompt(
            {"memory_type", "categories", "importance", "pinned"},
            ["personal", "work", "project:myapp"],
        )
        assert "memory_type" in prompt
        assert "categories" in prompt
        assert "importance" in prompt
        assert "pinned" in prompt
        assert "project:myapp" in prompt

    def test_single_field(self):
        prompt = _build_system_prompt(
            {"memory_type"},
            ["personal", "work"],
        )
        assert "memory_type" in prompt
        assert "categories" not in prompt
        assert "importance" not in prompt
        assert "pinned" not in prompt

    def test_categories_includes_dynamic(self):
        prompt = _build_system_prompt(
            {"categories"},
            ["personal", "work", "project:domecek", "project:myapp"],
        )
        assert "project:domecek" in prompt
        assert "project:myapp" in prompt

    def test_empty_fields(self):
        prompt = _build_system_prompt(set(), ["personal"])
        # Should still have the header but no field instructions
        assert "Classify" in prompt


# ── _validate_classification ───────────────────────────────────────────


class TestValidateClassification:
    def test_valid_full_response(self):
        result = _validate_classification(
            {
                "memory_type": "preference",
                "categories": ["work", "technical"],
                "importance": "high",
                "pinned": True,
            },
            {"memory_type", "categories", "importance", "pinned"},
        )
        assert result["memory_type"] == "preference"
        assert result["categories"] == ["work", "technical"]
        assert result["importance"] == "high"
        assert result["pinned"] is True

    def test_invalid_memory_type_falls_back(self):
        result = _validate_classification(
            {"memory_type": "invalid_type"},
            {"memory_type"},
        )
        assert result["memory_type"] == "fact"

    def test_invalid_importance_falls_back(self):
        result = _validate_classification(
            {"importance": "super_important"},
            {"importance"},
        )
        assert result["importance"] == "normal"

    def test_invalid_categories_falls_back(self):
        result = _validate_classification(
            {"categories": ["invented_category"]},
            {"categories"},
        )
        assert result["categories"] == []

    def test_non_list_categories_falls_back(self):
        result = _validate_classification(
            {"categories": "work"},
            {"categories"},
        )
        assert result["categories"] == []

    def test_missing_fields_use_defaults(self):
        result = _validate_classification(
            {},
            {"memory_type", "categories", "importance", "pinned"},
        )
        assert result["memory_type"] == "fact"
        assert result["categories"] == []
        assert result["importance"] == "normal"
        assert result["pinned"] is False

    def test_pinned_truthy_values(self):
        result = _validate_classification({"pinned": 1}, {"pinned"})
        assert result["pinned"] is True

    def test_pinned_falsy_values(self):
        result = _validate_classification({"pinned": 0}, {"pinned"})
        assert result["pinned"] is False

    def test_only_requested_fields_returned(self):
        result = _validate_classification(
            {
                "memory_type": "fact",
                "categories": ["work"],
                "importance": "high",
                "pinned": True,
            },
            {"memory_type"},  # Only request memory_type
        )
        assert "memory_type" in result
        assert "categories" not in result
        assert "importance" not in result
        assert "pinned" not in result

    def test_case_insensitive_memory_type(self):
        result = _validate_classification(
            {"memory_type": "PREFERENCE"},
            {"memory_type"},
        )
        assert result["memory_type"] == "preference"

    def test_case_insensitive_importance(self):
        result = _validate_classification(
            {"importance": "HIGH"},
            {"importance"},
        )
        assert result["importance"] == "high"


# ── classify_memory (with mocked LLM) ─────────────────────────────────


class TestClassifyMemory:
    @staticmethod
    def _make_llm_config():
        return LLMConfig(
            model="test-model",
            base_url="http://localhost:1234/v1",
            api_key="test-key",
        )

    def test_empty_missing_fields_returns_empty(self):
        result = classify_memory(
            "test content",
            missing_fields=set(),
            llm_config=self._make_llm_config(),
        )
        assert result == {}

    @patch("mnemory.classify._get_openai_client")
    def test_successful_classification(self, mock_get_client):
        mock_response = MagicMock()
        classify_json = (
            '{"memory_type": "preference", '
            '"categories": ["work"], '
            '"importance": "high", "pinned": false}'
        )
        mock_response.choices = [MagicMock(message=MagicMock(content=classify_json))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_memory(
            "I prefer Python over JavaScript",
            missing_fields={"memory_type", "categories", "importance", "pinned"},
            llm_config=self._make_llm_config(),
        )
        assert result["memory_type"] == "preference"
        assert result["categories"] == ["work"]
        assert result["importance"] == "high"
        assert result["pinned"] is False

    @patch("mnemory.classify._get_openai_client")
    def test_llm_failure_returns_defaults(self, mock_get_client):
        mock_get_client.side_effect = Exception("Connection refused")

        result = classify_memory(
            "test content",
            missing_fields={"memory_type", "importance"},
            llm_config=self._make_llm_config(),
        )
        assert result["memory_type"] == "fact"
        assert result["importance"] == "normal"

    @patch("mnemory.classify._get_openai_client")
    def test_empty_response_returns_defaults(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=""))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_memory(
            "test content",
            missing_fields={"memory_type", "categories"},
            llm_config=self._make_llm_config(),
        )
        assert result["memory_type"] == "fact"
        assert result["categories"] == []

    @patch("mnemory.classify._get_openai_client")
    def test_invalid_json_returns_defaults(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_memory(
            "test content",
            missing_fields={"memory_type"},
            llm_config=self._make_llm_config(),
        )
        assert result["memory_type"] == "fact"

    @patch("mnemory.classify._get_openai_client")
    def test_available_categories_passed_to_prompt(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"categories": ["project:myapp"]}'))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = classify_memory(
            "Fixed the login bug in myapp",
            missing_fields={"categories"},
            llm_config=self._make_llm_config(),
            available_categories=["personal", "work", "project:myapp"],
        )
        assert result["categories"] == ["project:myapp"]

        # Verify the system prompt included project:myapp
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "project:myapp" in system_msg


# ── CategoryCache ──────────────────────────────────────────────────────


class TestCategoryCache:
    def test_get_miss(self):
        cache = CategoryCache(ttl_seconds=60)
        assert cache.get("user1") is None

    def test_set_and_get(self):
        cache = CategoryCache(ttl_seconds=60)
        cache.set("user1", ["personal", "work", "project:foo"])
        result = cache.get("user1")
        assert result == ["personal", "work", "project:foo"]

    def test_ttl_expiry(self):
        cache = CategoryCache(ttl_seconds=0)  # Immediate expiry
        cache.set("user1", ["personal"])
        # Sleep briefly to ensure monotonic clock advances
        time.sleep(0.01)
        assert cache.get("user1") is None

    def test_invalidate(self):
        cache = CategoryCache(ttl_seconds=60)
        cache.set("user1", ["personal"])
        cache.invalidate("user1")
        assert cache.get("user1") is None

    def test_invalidate_nonexistent(self):
        cache = CategoryCache(ttl_seconds=60)
        cache.invalidate("nonexistent")  # Should not raise

    def test_separate_users(self):
        cache = CategoryCache(ttl_seconds=60)
        cache.set("user1", ["personal"])
        cache.set("user2", ["work"])
        assert cache.get("user1") == ["personal"]
        assert cache.get("user2") == ["work"]

    def test_overwrite(self):
        cache = CategoryCache(ttl_seconds=60)
        cache.set("user1", ["personal"])
        cache.set("user1", ["work", "technical"])
        assert cache.get("user1") == ["work", "technical"]
