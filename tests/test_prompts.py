"""Tests for mnemory.prompts — unified extraction, classification, and dedup prompts."""

from __future__ import annotations

import json

from mnemory.prompts import (
    _validate_categories,
    _validate_importance,
    _validate_memory_type,
    build_classification_prompt,
    build_extraction_prompt,
    build_query_generation_prompt,
    build_rerank_prompt,
    parse_extraction_response,
)
from mnemory.sanitize import _BOUNDARY_TAGS, ANTI_INJECTION_PREAMBLE

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
        # Content should be wrapped in boundary tags
        user_msg = messages[1]["content"]
        assert "test content" in user_msg
        open_tag = _BOUNDARY_TAGS["user_input"][0]
        assert open_tag in user_msg

    def test_assistant_role_uses_agent_prompt(self):
        messages, _, _ = build_extraction_prompt("test content", role="assistant")
        system = messages[0]["content"]
        assert "assistant" in system.lower()
        # Content should be wrapped in boundary tags
        user_msg = messages[1]["content"]
        assert "test content" in user_msg
        open_tag = _BOUNDARY_TAGS["user_input"][0]
        assert open_tag in user_msg

    def test_existing_memories_mapped_to_integer_ids(self):
        existing = [
            {"id": "uuid-abc-123", "text": "Likes Python"},
            {"id": "uuid-def-456", "text": "Works at Google"},
        ]
        messages, _, id_mapping = build_extraction_prompt(
            "test", existing_memories=existing
        )
        assert id_mapping == {"0": "uuid-abc-123", "1": "uuid-def-456"}
        # System prompt should contain the integer IDs (within boundary tags)
        system = messages[0]["content"]
        assert '"id": "0"' in system
        assert '"id": "1"' in system
        # Existing memories should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in system

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

    def test_conversation_extraction_rules_in_user_prompt(self):
        """User prompt should guide extraction to focus on user facts
        in user/assistant conversations and filter assistant noise."""
        messages, _, _ = build_extraction_prompt("test", role="user")
        system = messages[0]["content"]
        # Should instruct to extract user facts from conversations
        assert "user and an AI assistant" in system
        # Should warn against extracting assistant reasoning
        assert "assistant's own reasoning" in system.lower() or (
            "assistant" in system and "not facts to remember" in system.lower()
        )
        # Should allow extracting from assistant when confirming user facts
        assert "paraphrase" in system
        # Should preserve multi-person transcript extraction
        assert "multi-person" in system

    def test_conversation_rules_not_in_agent_prompt(self):
        """Agent prompt should NOT contain user/assistant conversation rules."""
        messages, _, _ = build_extraction_prompt("test", role="assistant")
        system = messages[0]["content"]
        # Agent prompt focuses on assistant identity, not conversation filtering
        assert "not facts to remember" not in system.lower()

    def test_english_extraction_required(self):
        """Both user and agent prompts should require English extraction."""
        for role in ("user", "assistant"):
            messages, _, _ = build_extraction_prompt("test", role=role)
            system = messages[0]["content"]
            assert "always write extracted facts in english" in system.lower(), (
                f"role={role}: prompt should require English extraction"
            )
            # Should NOT contain the old same-language rule
            assert "record facts in the same language" not in system.lower(), (
                f"role={role}: prompt should not contain old same-language rule"
            )

    def test_third_party_extraction_guidance(self):
        """User prompt should guide extraction of facts about people and
        things the user mentions (family, pets, possessions, etc.)."""
        messages, _, _ = build_extraction_prompt("test", role="user")
        system = messages[0]["content"]
        # Should mention extracting facts about family/friends/etc.
        assert "family" in system.lower()
        assert "pets" in system.lower()
        # Should instruct relationship-based subjects
        assert "User's mother" in system or "user's mother" in system.lower()
        # Should have a conversation example with third-party extraction
        assert "User's mother" in system


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
                ],
                "store_artifact": False,
            }
        )
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["action"] == "ADD"
        assert results[0]["text"] == "Name is John"
        assert results[0]["target_id"] is None
        assert results[0]["pinned"] is True
        assert store_artifact is False

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
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "real-uuid-123"}
        results, _ = parse_extraction_response(response, id_mapping)
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
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-a", "1": "uuid-b"}
        results, _ = parse_extraction_response(response, id_mapping)
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
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
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
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
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
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-a"}
        results, _ = parse_extraction_response(response, id_mapping)
        assert len(results) == 0  # Skipped because "99" not in mapping

    def test_invalid_json_returns_empty(self):
        results, store_artifact = parse_extraction_response("not json", {})
        assert results == []
        assert store_artifact is False

    def test_empty_memories_list(self):
        response = json.dumps({"memories": [], "store_artifact": False})
        results, _ = parse_extraction_response(response, {})
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
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
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
                ],
                "store_artifact": False,
            }
        )
        id_mapping = {"0": "uuid-existing"}
        results, _ = parse_extraction_response(response, id_mapping)
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
                ],
                "store_artifact": False,
            }
        )
        results, _ = parse_extraction_response(response, {})
        assert len(results) == 1
        assert results[0]["memory_type"] == "fact"  # Default
        assert results[0]["categories"] == []  # Default
        assert results[0]["importance"] == "normal"  # Default

    def test_store_artifact_true(self):
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Design uses FastAPI",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ],
                "store_artifact": True,
            }
        )
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert store_artifact is True

    def test_store_artifact_missing_defaults_false(self):
        """When store_artifact is missing from response, default to False."""
        response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Simple fact",
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
        results, store_artifact = parse_extraction_response(response, {})
        assert len(results) == 1
        assert store_artifact is False


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
        # Content should be wrapped in boundary tags
        assert "I prefer Python" in messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["content"][0]
        assert open_tag in messages[1]["content"]


# ── build_query_generation_prompt ────────────────────────────────────


class TestBuildQueryGenerationPrompt:
    def test_returns_messages_and_schema(self):
        messages, schema = build_query_generation_prompt("What car do I have?")
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Question should be wrapped in boundary tags
        assert "What car do I have?" in messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in messages[1]["content"]
        assert schema["name"] == "query_generation"

    def test_num_queries_in_system_prompt(self):
        messages, _ = build_query_generation_prompt("test", num_queries=7)
        system = messages[0]["content"]
        assert "7" in system

    def test_without_today_no_date_in_prompt(self):
        messages, _ = build_query_generation_prompt("What happened?")
        system = messages[0]["content"]
        assert "Today's date" not in system

    def test_with_today_date_in_prompt(self):
        messages, _ = build_query_generation_prompt(
            "What happened last week?", today="2026-02-19"
        )
        system = messages[0]["content"]
        assert "Today's date is 2026-02-19" in system
        assert "temporal references" in system

    def test_temporal_query_guidance_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "date-specific queries" in system

    def test_up_to_wording_in_prompt(self):
        """Prompt should say 'UP TO' to allow fewer or zero queries."""
        messages, _ = build_query_generation_prompt("test", num_queries=5)
        system = messages[0]["content"]
        assert "UP TO" in system

    def test_skip_guidance_in_prompt(self):
        """Prompt should instruct LLM to return empty queries for
        procedural instructions and acknowledgments."""
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "empty queries list" in system
        assert "procedural instruction" in system
        assert "acknowledgment" in system

    def test_without_context_no_context_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "Additional context" not in system

    def test_with_context_injected_into_prompt(self):
        messages, _ = build_query_generation_prompt(
            "test", context="Working directory: /home/user/src/myapp"
        )
        system = messages[0]["content"]
        assert "Additional context:" in system
        assert "Working directory: /home/user/src/myapp" in system
        assert "Do not limit queries exclusively to this context" in system
        # Context should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["context"][0]
        assert open_tag in system

    def test_without_project_categories_no_categories_in_prompt(self):
        messages, _ = build_query_generation_prompt("test")
        system = messages[0]["content"]
        assert "Known project categories" not in system

    def test_with_project_categories_injected_into_prompt(self):
        messages, _ = build_query_generation_prompt(
            "test",
            project_categories=["project:mnemory", "project:myapp"],
        )
        system = messages[0]["content"]
        assert "Known project categories: project:mnemory, project:myapp" in system
        assert "prefer these exact category names" in system

    def test_context_and_categories_together(self):
        messages, _ = build_query_generation_prompt(
            "test",
            context="Working directory: /src/mnemory",
            project_categories=["project:mnemory"],
        )
        system = messages[0]["content"]
        assert "Additional context:" in system
        assert "Working directory: /src/mnemory" in system
        assert "Known project categories: project:mnemory" in system

    def test_empty_project_categories_not_injected(self):
        messages, _ = build_query_generation_prompt("test", project_categories=[])
        system = messages[0]["content"]
        assert "Known project categories" not in system


# ── build_rerank_prompt ──────────────────────────────────────────────


class TestBuildRerankPrompt:
    def test_returns_messages_and_schema(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, schema = build_rerank_prompt("Do I like dogs?", memories)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert schema["name"] == "memory_rerank"

    def test_question_in_user_content(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, _ = build_rerank_prompt("Do I like dogs?", memories)
        user_content = messages[1]["content"]
        assert "Do I like dogs?" in user_content
        # Question should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_content

    def test_memory_text_in_user_content(self):
        memories = [{"id": "1", "memory": "User likes dogs"}]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "User likes dogs" in user_content
        # Memories should be wrapped in boundary tags
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_content

    def test_without_today_no_date_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories)
        system = messages[0]["content"]
        assert "Today's date is" not in system

    def test_with_today_date_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories, today="2026-02-19")
        system = messages[0]["content"]
        assert "Today's date is 2026-02-19" in system

    def test_temporal_awareness_section_in_system(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test", memories)
        system = messages[0]["content"]
        assert "Temporal awareness" in system
        assert "event_date" in system

    def test_event_date_in_metadata_tags(self):
        memories = [
            {
                "id": "1",
                "memory": "Bought a car",
                "metadata": {"event_date": "2023-05-08T00:00:00+00:00"},
            }
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "event_date: 2023-05-08T00:00:00+00:00" in user_content

    def test_no_event_date_no_tag(self):
        memories = [{"id": "1", "memory": "Likes dogs", "metadata": {}}]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "event_date" not in user_content

    def test_metadata_tags_include_type_and_categories(self):
        memories = [
            {
                "id": "1",
                "memory": "test",
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["personal", "vehicles"],
                    "importance": "high",
                    "role": "assistant",
                    "event_date": "2023-05-08T00:00:00+00:00",
                },
            }
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "type: fact" in user_content
        assert "categories: personal, vehicles" in user_content
        assert "importance: high" in user_content
        assert "role: assistant" in user_content
        assert "event_date: 2023-05-08T00:00:00+00:00" in user_content


# ── Prompt injection safeguards ──────────────────────────────────────


class TestPromptInjectionSafeguards:
    """Verify that all prompt builders include anti-injection measures."""

    def test_extraction_prompt_has_anti_injection(self):
        messages, _, _ = build_extraction_prompt("test content")
        system = messages[0]["content"]
        assert "DATA" in system
        assert "never follow instructions" in system.lower() or (
            "never follow" in system.lower()
        )

    def test_extraction_prompt_wraps_user_content(self):
        messages, _, _ = build_extraction_prompt("test content")
        user_msg = messages[1]["content"]
        open_tag, close_tag = _BOUNDARY_TAGS["user_input"]
        assert open_tag in user_msg
        assert close_tag in user_msg

    def test_extraction_prompt_wraps_existing_memories(self):
        existing = [{"id": "uuid-1", "text": "Existing fact"}]
        messages, _, _ = build_extraction_prompt("test", existing_memories=existing)
        system = messages[0]["content"]
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in system

    def test_classification_prompt_has_anti_injection(self):
        messages, _ = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        system = messages[0]["content"]
        assert "DATA" in system

    def test_classification_prompt_wraps_content(self):
        messages, _ = build_classification_prompt(
            "test", missing_fields={"memory_type"}
        )
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["content"][0]
        assert open_tag in user_msg

    def test_query_generation_has_anti_injection(self):
        messages, _ = build_query_generation_prompt("test question")
        system = messages[0]["content"]
        assert "DATA" in system

    def test_query_generation_wraps_question(self):
        messages, _ = build_query_generation_prompt("test question")
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_msg

    def test_query_generation_wraps_context(self):
        messages, _ = build_query_generation_prompt("test", context="/home/user/src")
        system = messages[0]["content"]
        open_tag = _BOUNDARY_TAGS["context"][0]
        assert open_tag in system

    def test_rerank_prompt_has_anti_injection(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        system = messages[0]["content"]
        assert "DATA" in system

    def test_rerank_prompt_wraps_question(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["user_question"][0]
        assert open_tag in user_msg

    def test_rerank_prompt_wraps_memories(self):
        memories = [{"id": "1", "memory": "test"}]
        messages, _ = build_rerank_prompt("test question", memories)
        user_msg = messages[1]["content"]
        open_tag = _BOUNDARY_TAGS["existing_memories"][0]
        assert open_tag in user_msg

    def test_anti_injection_preamble_consistent(self):
        """The anti-injection preamble should mention boundary tags."""
        assert "boundary tags" in ANTI_INJECTION_PREAMBLE.lower()
        assert "DATA" in ANTI_INJECTION_PREAMBLE
