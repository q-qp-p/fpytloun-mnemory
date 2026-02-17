"""Tests for mnemory.ttl — TTL utility functions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from mnemory.ttl import (
    build_expiry_metadata,
    build_reinforcement_metadata,
    calculate_expiration,
    get_default_ttl,
    is_decayed,
    is_expired,
    should_exclude,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Create a mock MemoryConfig with TTL defaults."""
    config = MagicMock()
    config.ttl_fact = overrides.get("ttl_fact", None)
    config.ttl_preference = overrides.get("ttl_preference", None)
    config.ttl_episodic = overrides.get("ttl_episodic", 90)
    config.ttl_procedural = overrides.get("ttl_procedural", 60)
    config.ttl_context = overrides.get("ttl_context", 7)
    return config


def _make_memory(
    expires_at=None,
    pinned=False,
    decayed_at=None,
    ttl_days=None,
    access_count=0,
    last_accessed_at=None,
):
    """Create a memory dict with TTL metadata."""
    return {
        "id": "test-id",
        "memory": "test content",
        "metadata": {
            "memory_type": "fact",
            "importance": "normal",
            "pinned": pinned,
            "expires_at": expires_at,
            "decayed_at": decayed_at,
            "ttl_days": ttl_days,
            "access_count": access_count,
            "last_accessed_at": last_accessed_at,
        },
    }


# ── get_default_ttl ─────────────────────────────────────────────────


class TestGetDefaultTtl:
    def test_fact_permanent(self):
        config = _make_config()
        assert get_default_ttl("fact", config) is None

    def test_preference_permanent(self):
        config = _make_config()
        assert get_default_ttl("preference", config) is None

    def test_episodic_default(self):
        config = _make_config()
        assert get_default_ttl("episodic", config) == 90

    def test_procedural_default(self):
        config = _make_config()
        assert get_default_ttl("procedural", config) == 60

    def test_context_default(self):
        config = _make_config()
        assert get_default_ttl("context", config) == 7

    def test_custom_override(self):
        config = _make_config(ttl_fact=30)
        assert get_default_ttl("fact", config) == 30

    def test_none_memory_type(self):
        config = _make_config()
        assert get_default_ttl(None, config) is None

    def test_unknown_memory_type(self):
        config = _make_config()
        assert get_default_ttl("unknown", config) is None


# ── calculate_expiration ─────────────────────────────────────────────


class TestCalculateExpiration:
    def test_none_ttl_returns_none(self):
        assert calculate_expiration(None) is None

    def test_calculates_from_now(self):
        result = calculate_expiration(30)
        assert result is not None
        expires = datetime.fromisoformat(result)
        expected = datetime.now(timezone.utc) + timedelta(days=30)
        # Allow 2 second tolerance
        assert abs((expires - expected).total_seconds()) < 2

    def test_calculates_from_custom_time(self):
        base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = calculate_expiration(7, from_time=base)
        expected = datetime(2024, 1, 22, 10, 0, 0, tzinfo=timezone.utc)
        assert result == expected.isoformat()

    def test_zero_ttl(self):
        """TTL of 0 days means expires immediately."""
        base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = calculate_expiration(0, from_time=base)
        assert result == base.isoformat()


# ── is_expired ───────────────────────────────────────────────────────


class TestIsExpired:
    def test_no_expires_at_not_expired(self):
        mem = _make_memory(expires_at=None)
        assert is_expired(mem) is False

    def test_future_expires_at_not_expired(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        mem = _make_memory(expires_at=future)
        assert is_expired(mem) is False

    def test_past_expires_at_is_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mem = _make_memory(expires_at=past)
        assert is_expired(mem) is True

    def test_pinned_memory_never_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mem = _make_memory(expires_at=past, pinned=True)
        assert is_expired(mem) is False

    def test_pinned_without_expires_at(self):
        mem = _make_memory(pinned=True)
        assert is_expired(mem) is False

    def test_invalid_expires_at_not_expired(self):
        mem = _make_memory(expires_at="not-a-date")
        assert is_expired(mem) is False

    def test_empty_string_expires_at_not_expired(self):
        mem = _make_memory(expires_at="")
        assert is_expired(mem) is False

    def test_just_expired(self):
        """Memory that expired 1 second ago."""
        just_past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        mem = _make_memory(expires_at=just_past)
        assert is_expired(mem) is True

    def test_empty_metadata(self):
        """Memory with no metadata at all (legacy)."""
        mem = {"id": "test", "memory": "content", "metadata": {}}
        assert is_expired(mem) is False

    def test_no_metadata_key(self):
        """Memory with no metadata key at all."""
        mem = {"id": "test", "memory": "content"}
        assert is_expired(mem) is False


# ── is_decayed ───────────────────────────────────────────────────────


class TestIsDecayed:
    def test_not_decayed(self):
        mem = _make_memory(decayed_at=None)
        assert is_decayed(mem) is False

    def test_decayed(self):
        mem = _make_memory(decayed_at="2024-01-15T10:00:00+00:00")
        assert is_decayed(mem) is True

    def test_empty_metadata(self):
        mem = {"id": "test", "memory": "content", "metadata": {}}
        assert is_decayed(mem) is False


# ── should_exclude ───────────────────────────────────────────────────


class TestShouldExclude:
    def test_active_memory_not_excluded(self):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        mem = _make_memory(expires_at=future)
        assert should_exclude(mem) is False

    def test_permanent_memory_not_excluded(self):
        mem = _make_memory(expires_at=None)
        assert should_exclude(mem) is False

    def test_expired_memory_excluded(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mem = _make_memory(expires_at=past)
        assert should_exclude(mem) is True

    def test_decayed_memory_excluded(self):
        mem = _make_memory(decayed_at="2024-01-15T10:00:00+00:00")
        assert should_exclude(mem) is True

    def test_expired_not_excluded_with_include_decayed(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mem = _make_memory(expires_at=past)
        assert should_exclude(mem, include_decayed=True) is False

    def test_decayed_not_excluded_with_include_decayed(self):
        mem = _make_memory(decayed_at="2024-01-15T10:00:00+00:00")
        assert should_exclude(mem, include_decayed=True) is False

    def test_pinned_expired_not_excluded(self):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        mem = _make_memory(expires_at=past, pinned=True)
        assert should_exclude(mem) is False


# ── build_expiry_metadata ────────────────────────────────────────────


class TestBuildExpiryMetadata:
    def test_explicit_ttl(self):
        config = _make_config()
        result = build_expiry_metadata(30, "fact", config)
        assert result["ttl_days"] == 30
        assert result["expires_at"] is not None
        assert result["decayed_at"] is None
        assert result["last_accessed_at"] is None
        assert result["access_count"] == 0

    def test_default_ttl_from_type(self):
        config = _make_config(ttl_context=7)
        result = build_expiry_metadata(None, "context", config)
        assert result["ttl_days"] == 7
        assert result["expires_at"] is not None

    def test_permanent_memory(self):
        config = _make_config()
        result = build_expiry_metadata(None, "fact", config)
        assert result["ttl_days"] is None
        assert result["expires_at"] is None

    def test_explicit_ttl_overrides_default(self):
        config = _make_config(ttl_context=7)
        result = build_expiry_metadata(30, "context", config)
        assert result["ttl_days"] == 30

    def test_all_fields_present(self):
        config = _make_config()
        result = build_expiry_metadata(10, "fact", config)
        assert set(result.keys()) == {
            "ttl_days",
            "expires_at",
            "decayed_at",
            "last_accessed_at",
            "access_count",
        }


# ── build_reinforcement_metadata ─────────────────────────────────────


class TestBuildReinforcementMetadata:
    def test_increments_access_count(self):
        mem = _make_memory(access_count=5, ttl_days=None)
        result = build_reinforcement_metadata(mem)
        assert result["access_count"] == 6

    def test_sets_last_accessed_at(self):
        mem = _make_memory()
        result = build_reinforcement_metadata(mem)
        assert result["last_accessed_at"] is not None
        accessed = datetime.fromisoformat(result["last_accessed_at"])
        assert abs((accessed - datetime.now(timezone.utc)).total_seconds()) < 2

    def test_resets_ttl_on_access(self):
        past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        mem = _make_memory(expires_at=past, ttl_days=30)
        result = build_reinforcement_metadata(mem)
        assert result["expires_at"] is not None
        new_expires = datetime.fromisoformat(result["expires_at"])
        expected = datetime.now(timezone.utc) + timedelta(days=30)
        assert abs((new_expires - expected).total_seconds()) < 2

    def test_no_ttl_no_expires_reset(self):
        mem = _make_memory(ttl_days=None)
        result = build_reinforcement_metadata(mem)
        assert "expires_at" not in result

    def test_clears_decayed_at_on_access(self):
        mem = _make_memory(
            ttl_days=30,
            decayed_at="2024-01-15T10:00:00+00:00",
        )
        result = build_reinforcement_metadata(mem)
        assert result["decayed_at"] is None

    def test_zero_access_count_initializes(self):
        mem = _make_memory(access_count=0)
        result = build_reinforcement_metadata(mem)
        assert result["access_count"] == 1

    def test_none_access_count_initializes(self):
        """Legacy memory with no access_count field."""
        mem = {"id": "test", "memory": "content", "metadata": {}}
        result = build_reinforcement_metadata(mem)
        assert result["access_count"] == 1
