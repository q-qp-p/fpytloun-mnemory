"""Tests for mnemory.prompts — unified extraction, classification, and dedup prompts."""

from __future__ import annotations

import json

from mnemory.prompts import (
    _validate_categories,
    _validate_importance,
    _validate_memory_type,
    build_classification_prompt,
    build_extraction_prompt,
    parse_extraction_response,
)

# ── Validation helpers ───────────────────────────────────────────────


class TestValidateMemoryType:
    def test_valid_types(self):
        for t in ("preference", "fact", "episodic", "procedural", "context"):
            assert _validate_memory_type(t) == t

    def test_invalid_type_falls_back(self):
        assert _validate_memory_type("invalid") == "fact"

    def test_none_falls_back(self):
        assert _validate_memory_type(None) == "fact"


class TestValidateCategories:
    def test_valid_list(self):
        assert _validate_categories(["work", "technical"]) == ["work", "technical"]

    def test_non_list_returns_empty(self):
        assert _validate_categories("work") == []

    def test_strips_whitespace(self):
        assert _validate_categories([" work ", " tech "]) == ["work", "tech"]

    def test_filters_empty_strings(self):
        assert _validate_categories(["work", "", "  "]) == ["work"]

    def test_none_returns_empty(self):
        assert _validate_categories(None) == []


class TestValidateImportance:
    def test_valid_levels(self):
        for level in ("low", "normal", "high", "critical"):
            assert _validate_importance(level) == level

    def test_invalid_falls_back(self):
        assert _validate_importance("super") == "normal"

    def test_none_falls_back(self):
        assert _validate_importance(None) == "normal"


# ── build_extraction_prompt ──────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_returns_three_elements(self):
        messages, schema, id_mapping = build_extraction_prompt("test content")
        assert isinstance(messages, list)
        assert isinstance(schema, dict)
        assert isinstance(id_mapping, dict)

    def test_user_role_uses_user_prompt(self):
        messages, _, _ = build_extraction_prompt("test content", role="user")
        system = messages[0]["content"]
        assert "user" in system.lower()
        # Content should be passed as-is without role prefix
        assert messages[1]["content"] == "test content"

    def test_assistant_role_uses_agent_prompt(self):
        messages, _, _ = build_extraction_prompt("test content", role="assistant")
        system = messages[0]["content"]
        assert "assistant" in system.lower()
        # Content should be passed as-is without role prefix
        assert messages[1]["content"] == "test content"

    def test_existing_memories_mapped_to_integer_ids(self):
        existing = [
            {"id": "uuid-abc-123", "text": "Likes Python"},
            {"id": "uuid-def-456", "text": "Works at Google"},
        ]
        messages, _, id_mapping = build_extraction_prompt(
            "test", existing_memories=existing
        )
        assert id_mapping == {"0": "uuid-abc-123", "1": "uuid-def-456"}
        # System prompt should contain the integer IDs
        system = messages[0]["content"]
        assert '"id": "0"' in system
        assert '"id": "1"' in system

    def test_no_existing_memories(self):
        messages, _, id_mapping = build_extraction_prompt("test")
        assert id_mapping == {}
        system = messages[0]["content"]
        assert "None yet" in system

    def test_available_categories_in_prompt(self):
        cats = ["personal", "work", "project:myapp"]
        messages, _, _ = build_extraction_prompt("test", available_categories=cats)
        system = messages[0]["content"]
        assert "project:myapp" in system

    def test_explicit_fields_in_prompt(self):
        messages, _, _ = build_extraction_prompt(
            "test", explicit_fields={"memory_type": "fact", "pinned": True}
        )
        system = messages[0]["content"]
        assert "Caller-Provided" in system
        assert '"memory_type"' in system
        assert '"fact"' in system

    def test_schema_has_required_structure(self):
        _, schema, _ = build_extraction_prompt("test")
        assert schema["name"] == "memory_extraction"
        assert "memories" in schema["schema"]["properties"]

    def test_today_date_in_prompt(self):
        messages, _, _ = build_extraction_prompt("test")
        system = messages[0]["content"]
        # Should contain a date like 2026-02-17
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", system)


# ── parse_extraction_response ────────────────────────────────────────


class TestParseExtractionResponse:
    def test_add_action(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Name is John",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": True,
                    }
                ]
            }
        )
        results = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["action"] == "ADD"
        assert results[0]["text"] == "Name is John"
        assert results[0]["target_id"] is None
        assert results[0]["pinned"] is True

    def test_update_action_maps_id(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Uses Neovim",
                        "action": "UPDATE",
                        "target_id": "0",
                        "old_memory": "Uses VS Code",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        id_mapping = {"0": "real-uuid-123"}
        results = parse_extraction_response(response, id_mapping)
        assert len(results) == 1
        assert results[0]["action"] == "UPDATE"
        assert results[0]["target_id"] == "real-uuid-123"
        assert results[0]["old_memory"] == "Uses VS Code"

    def test_delete_action_maps_id(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Old fact",
                        "action": "DELETE",
                        "target_id": "1",
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        id_mapping = {"0": "uuid-a", "1": "uuid-b"}
        results = parse_extraction_response(response, id_mapping)
        assert len(results) == 1
        assert results[0]["action"] == "DELETE"
        assert results[0]["target_id"] == "uuid-b"

    def test_none_action_filtered_out(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Already known",
                        "action": "NONE",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        results = parse_extraction_response(response, {})
        assert len(results) == 0

    def test_empty_text_filtered_out(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        results = parse_extraction_response(response, {})
        assert len(results) == 0

    def test_unknown_target_id_skipped(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Updated fact",
                        "action": "UPDATE",
                        "target_id": "99",
                        "old_memory": "Old",
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        id_mapping = {"0": "uuid-a"}
        results = parse_extraction_response(response, id_mapping)
        assert len(results) == 0  # Skipped because "99" not in mapping

    def test_invalid_json_returns_empty(self):
        results = parse_extraction_response("not json", {})
        assert results == []

    def test_empty_memories_list(self):
        response = json.dumps({"memories": []})
        results = parse_extraction_response(response, {})
        assert results == []

    def test_invalid_action_skipped(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "test",
                        "action": "INVALID",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        results = parse_extraction_response(response, {})
        assert len(results) == 0

    def test_multiple_actions(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "New fact",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": False,
                    },
                    {
                        "text": "Updated pref",
                        "action": "UPDATE",
                        "target_id": "0",
                        "old_memory": "Old pref",
                        "memory_type": "preference",
                        "categories": [],
                        "importance": "high",
                        "pinned": True,
                    },
                ]
            }
        )
        id_mapping = {"0": "uuid-existing"}
        results = parse_extraction_response(response, id_mapping)
        assert len(results) == 2
        assert results[0]["action"] == "ADD"
        assert results[1]["action"] == "UPDATE"
        assert results[1]["target_id"] == "uuid-existing"

    def test_invalid_classification_fields_get_defaults(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "test fact",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "invalid_type",
                        "categories": "not_a_list",
                        "importance": "super_important",
                        "pinned": False,
                    }
                ]
            }
        )
        results = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["memory_type"] == "fact"  # Default
        assert results[0]["categories"] == []  # Default
        assert results[0]["importance"] == "normal"  # Default


# ── build_classification_prompt ──────────────────────────────────────


class TestBuildClassificationPrompt:
    def test_empty_fields_returns_empty(self):
        messages, schema = build_classification_prompt("test", missing_fields=set())
        assert messages == []
        assert schema is None

    def test_single_field(self):
        messages, schema = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        assert len(messages) == 2
        assert schema is not None
        assert "memory_type" in schema["schema"]["properties"]
        assert "categories" not in schema["schema"]["properties"]

    def test_all_fields(self):
        messages, schema = build_classification_prompt(
            "test",
            missing_fields={"memory_type", "categories", "importance", "pinned"},
        )
        props = schema["schema"]["properties"]
        assert "memory_type" in props
        assert "categories" in props
        assert "importance" in props
        assert "pinned" in props

    def test_available_categories_in_prompt(self):
        messages, _ = build_classification_prompt(
            "test",
            missing_fields={"categories"},
            available_categories=["personal", "work", "project:myapp"],
        )
        system = messages[0]["content"]
        assert "project:myapp" in system

    def test_content_in_user_message(self):
        messages, _ = build_classification_prompt(
            "I prefer Python", missing_fields={"memory_type"}
        )
        assert messages[1]["content"] == "I prefer Python"
