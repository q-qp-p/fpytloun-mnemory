"""Unit tests for the consolidation service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mnemory.consolidation import ConsolidationService


def _make_service(vector_mock=None):
    """Create a ConsolidationService with mocked dependencies."""
    service = ConsolidationService.__new__(ConsolidationService)
    service._vector = vector_mock or MagicMock()
    service._llm = MagicMock()
    service._embedding = MagicMock()
    service._memory = MagicMock()
    service._sessions = MagicMock()
    service._collector = None
    service._config = MagicMock()
    return service


class TestFetchRawMemories:
    """Tests for ConsolidationService._fetch_raw_memories."""

    def test_includes_memories_without_layer(self):
        """Memories without memory_layer field should be treated as raw."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-1",
            "memory": "User likes Python",
            "metadata": {
                "memory_type": "preference",
                # No memory_layer field
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-1"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-1"

    def test_includes_explicit_raw(self):
        """Memories with memory_layer='raw' should be included."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-2",
            "memory": "User prefers dark mode",
            "metadata": {"memory_layer": "raw"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-2"], "user-1")
        assert len(result) == 1

    def test_excludes_consolidated(self):
        """Memories with memory_layer='consolidated' should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-3",
            "memory": "User lives in Prague",
            "metadata": {"memory_layer": "consolidated"},
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-3"], "user-1")
        assert len(result) == 0

    def test_excludes_superseded(self):
        """Superseded raw memories should be excluded."""
        vector = MagicMock()
        vector.get_by_id.return_value = {
            "id": "mem-4",
            "memory": "User likes Java",
            "metadata": {
                "memory_layer": "raw",
                "superseded_by": "consolidated-1",
            },
        }
        service = _make_service(vector)
        result = service._fetch_raw_memories(["mem-4"], "user-1")
        assert len(result) == 0

    def test_skips_missing_memories(self):
        """Memory IDs that don't exist should be skipped."""
        vector = MagicMock()
        vector.get_by_id.return_value = None
        service = _make_service(vector)
        result = service._fetch_raw_memories(["nonexistent"], "user-1")
        assert len(result) == 0

    def test_handles_exceptions(self):
        """Exceptions from vector.get_by_id should be caught and skipped."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            Exception("Connection error"),
            {
                "id": "mem-5",
                "memory": "User likes Rust",
                "metadata": {"memory_layer": "raw"},
            },
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["bad-id", "mem-5"], "user-1")
        assert len(result) == 1
        assert result[0]["id"] == "mem-5"

    def test_mixed_memories(self):
        """Test with a mix of raw, consolidated, superseded, and missing."""
        vector = MagicMock()
        vector.get_by_id.side_effect = [
            # raw, no layer field (should include)
            {"id": "m1", "memory": "fact 1", "metadata": {}},
            # explicit raw (should include)
            {"id": "m2", "memory": "fact 2", "metadata": {"memory_layer": "raw"}},
            # consolidated (should exclude)
            {
                "id": "m3",
                "memory": "fact 3",
                "metadata": {"memory_layer": "consolidated"},
            },
            # superseded raw (should exclude)
            {
                "id": "m4",
                "memory": "fact 4",
                "metadata": {"memory_layer": "raw", "superseded_by": "c1"},
            },
            # missing (should skip)
            None,
        ]
        service = _make_service(vector)
        result = service._fetch_raw_memories(["m1", "m2", "m3", "m4", "m5"], "user-1")
        assert len(result) == 2
        assert {r["id"] for r in result} == {"m1", "m2"}


class TestReConsolidationStateReset:
    """Tests for re-consolidation state reset in SessionSummaryStore.upsert()."""

    def test_state_resets_to_idle_on_new_memories_after_consolidation(self):
        """When new memories arrive after consolidation, state should reset to idle."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        # Simulate existing consolidated session
        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1", "m2"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="new summary",
            new_memory_ids=["m3"],
        )

        # Verify the upsert was called with consolidation_state="idle"
        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "idle"

    def test_state_preserved_when_no_new_memories(self):
        """When no new memories arrive, consolidated state should be preserved."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "old summary",
                    "turn_count": 5,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "consolidated",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="updated summary",
            new_memory_ids=None,
        )

        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "consolidated"

    def test_idle_state_stays_idle_with_new_memories(self):
        """Idle state should remain idle when new memories arrive."""
        from mnemory.storage.vector import SessionSummaryStore

        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()

        store._client.retrieve.return_value = [
            MagicMock(
                payload={
                    "session_id": "ses-1",
                    "user_id": "user-1",
                    "summary": "summary",
                    "turn_count": 3,
                    "memory_ids": ["m1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "consolidation_state": "idle",
                }
            )
        ]

        store.upsert(
            session_id="ses-1",
            user_id="user-1",
            summary="summary",
            new_memory_ids=["m2"],
        )

        call_args = store._client.upsert.call_args
        payload = call_args.kwargs["points"][0].payload
        assert payload["consolidation_state"] == "idle"


class TestConsolidationPromptWithPrevious:
    """Tests for build_consolidation_prompt with previous_consolidated."""

    def test_prompt_includes_previous_section(self):
        """When previous_consolidated is provided, prompt should include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
            previous_consolidated=[
                {
                    "id": "c1",
                    "memory": "User prefers Python for coding",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated Memories" in user_msg
        assert "User prefers Python for coding" in user_msg
        assert "COMPLETE replacement set" in user_msg

    def test_prompt_without_previous(self):
        """When no previous_consolidated, prompt should not include the section."""
        from mnemory.prompts import build_consolidation_prompt

        messages, schema = build_consolidation_prompt(
            summary="Test summary",
            raw_memories=[
                {
                    "id": "m1",
                    "memory": "User likes Python",
                    "metadata": {
                        "memory_type": "preference",
                        "role": "user",
                        "importance": "normal",
                        "categories": [],
                    },
                },
            ],
        )

        user_msg = messages[1]["content"]
        assert "Previously Consolidated" not in user_msg
