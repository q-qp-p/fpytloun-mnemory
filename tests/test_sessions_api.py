"""Tests for session summary listing APIs and store helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mnemory.api.deps import SessionContext
from mnemory.api.sessions import list_sessions
from mnemory.storage.vector import SessionSummaryStore


def _point(payload: dict | None) -> MagicMock:
    point = MagicMock()
    point.payload = payload
    point.id = (payload or {}).get("session_id", "point")
    return point


class TestListSessionsEndpoint:
    """Tests for GET /api/sessions endpoint behavior."""

    def test_list_sessions_passes_query_params_and_returns_metadata(self):
        """Endpoint should pass through paging/search/sort params."""
        mock_store = MagicMock()
        mock_store.list_for_user.return_value = {
            "sessions": [{"session_id": "ses-1", "summary": "hello"}],
            "total": 1,
            "offset": 20,
            "limit": 10,
            "has_more": False,
            "total_truncated": False,
        }
        mock_service = MagicMock(_session_summary_store=mock_store)
        ctx = SessionContext(user_id="user-1", agent_id=None, timezone=None)

        with patch("mnemory.api.sessions._get_service", return_value=mock_service):
            result = asyncio.run(
                list_sessions(
                    offset=20,
                    limit=10,
                    consolidation_state="idle",
                    q="hello",
                    sort_by="created_at",
                    sort_dir="asc",
                    ctx=ctx,
                )
            )

        assert result["total"] == 1
        assert result["offset"] == 20
        assert result["limit"] == 10
        mock_store.list_for_user.assert_called_once_with(
            "user-1",
            offset=20,
            limit=10,
            consolidation_state="idle",
            q="hello",
            sort_by="created_at",
            sort_dir="asc",
            include_metadata=True,
        )

    def test_list_sessions_returns_clean_503_on_store_error(self):
        """Endpoint should wrap store failures in a clean HTTP 503."""
        mock_store = MagicMock()
        mock_store.list_for_user.side_effect = RuntimeError("boom")
        mock_service = MagicMock(_session_summary_store=mock_store)
        ctx = SessionContext(user_id="user-1", agent_id=None, timezone=None)

        with patch("mnemory.api.sessions._get_service", return_value=mock_service):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(list_sessions(ctx=ctx))

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Failed to list sessions"


class TestSessionSummaryStoreListForUser:
    """Tests for SessionSummaryStore.list_for_user pagination helpers."""

    def test_list_for_user_filters_sorts_and_paginates(self):
        """Store should dedupe, filter, sort, and paginate after full scroll."""
        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()
        store._client.scroll.side_effect = [
            (
                [
                    _point(
                        {
                            "session_id": "ses-3",
                            "user_id": "user-1",
                            "summary": "Zebra project summary",
                            "created_at": "2026-03-03T00:00:00+00:00",
                            "updated_at": "2026-03-04T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    ),
                    _point({"session_id": "broken", "user_id": "user-1"}),
                ],
                "offset-2",
            ),
            (
                [
                    _point(
                        {
                            "session_id": "ses-2",
                            "user_id": "user-1",
                            "summary": "Beta project summary",
                            "created_at": "2026-03-02T00:00:00+00:00",
                            "updated_at": "2026-03-03T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-4",
                            "user_id": "user-1",
                            "summary": "Other topic",
                            "created_at": "2026-03-05T00:00:00+00:00",
                            "updated_at": "2026-03-05T00:00:00+00:00",
                        }
                    ),
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary duplicate",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    ),
                ],
                None,
            ),
        ]

        result = store.list_for_user(
            "user-1",
            q="project",
            sort_by="created_at",
            sort_dir="asc",
            offset=1,
            limit=2,
            include_metadata=True,
        )

        assert result["total"] == 3
        assert result["offset"] == 1
        assert result["limit"] == 2
        assert result["has_more"] is False
        assert result["total_truncated"] is False
        assert [s["session_id"] for s in result["sessions"]] == ["ses-2", "ses-3"]

    def test_list_for_user_scroll_failure_raises_runtime_error(self):
        """Store should raise a clean RuntimeError on scroll failure."""
        store = SessionSummaryStore.__new__(SessionSummaryStore)
        store._client = MagicMock()
        store._client.scroll.side_effect = [
            (
                [
                    _point(
                        {
                            "session_id": "ses-1",
                            "user_id": "user-1",
                            "summary": "Alpha project summary",
                            "created_at": "2026-03-01T00:00:00+00:00",
                            "updated_at": "2026-03-02T00:00:00+00:00",
                        }
                    )
                ],
                "offset-2",
            ),
            Exception("qdrant unavailable"),
        ]

        with pytest.raises(RuntimeError, match="Failed to list session summaries"):
            store.list_for_user("user-1", include_metadata=True)
