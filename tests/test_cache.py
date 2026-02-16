"""Tests for mnemory.cache module — generic TTLCache."""

from __future__ import annotations

import time

from mnemory.cache import TTLCache


class TestTTLCacheBasics:
    def test_get_miss(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        assert cache.get("key1") is None

    def test_set_and_get(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_ttl_expiry(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=0)
        cache.set("key1", "value1")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_overwrite(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_separate_keys(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        assert cache.get("a") == "1"
        assert cache.get("b") == "2"


class TestTTLCacheInvalidate:
    def test_invalidate_existing(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_nonexistent(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.invalidate("nonexistent")  # Should not raise

    def test_clear(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestTTLCacheInvalidatePrefix:
    def test_string_key_prefix(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("user1:data", "a")
        cache.set("user1:other", "b")
        cache.set("user2:data", "c")
        cache.invalidate_prefix("user1")
        assert cache.get("user1:data") is None
        assert cache.get("user1:other") is None
        assert cache.get("user2:data") == "c"

    def test_tuple_key_prefix(self):
        """invalidate_prefix matches the first element of tuple keys."""
        cache: TTLCache[tuple, str] = TTLCache(ttl_seconds=60)
        cache.set(("filip", "open-webui", 24), "core1")
        cache.set(("filip", "claude", 24), "core2")
        cache.set(("filip", "", 48), "core3")
        cache.set(("other", "open-webui", 24), "core4")
        cache.invalidate_prefix("filip")
        assert cache.get(("filip", "open-webui", 24)) is None
        assert cache.get(("filip", "claude", 24)) is None
        assert cache.get(("filip", "", 48)) is None
        assert cache.get(("other", "open-webui", 24)) == "core4"

    def test_prefix_no_match(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        cache.invalidate_prefix("nonexistent")
        assert cache.get("key1") == "value1"

    def test_empty_cache(self):
        cache: TTLCache[str, str] = TTLCache(ttl_seconds=60)
        cache.invalidate_prefix("anything")  # Should not raise
