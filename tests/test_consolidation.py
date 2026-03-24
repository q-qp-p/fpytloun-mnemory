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
        vector.get.return_value = {
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
        vector.get.return_value = {
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
        vector.get.return_value = {
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
        vector.get.return_value = {
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
        vector.get.return_value = None
        service = _make_service(vector)
        result = service._fetch_raw_memories(["nonexistent"], "user-1")
        assert len(result) == 0

    def test_handles_exceptions(self):
        """Exceptions from vector.get should be caught and skipped."""
        vector = MagicMock()
        vector.get.side_effect = [
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
        vector.get.side_effect = [
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
