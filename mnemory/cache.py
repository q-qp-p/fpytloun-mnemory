"""Generic TTL-based in-memory cache.

Used by MemoryService for caching category lists (auto-classification)
and core memory responses (get_core_memories).
"""

from __future__ import annotations

import time
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Simple TTL-based in-memory cache.

    Entries expire after ``ttl_seconds``. Expired entries are evicted
    lazily on access. Setting ``ttl_seconds=0`` effectively disables
    caching (every ``get`` returns ``None``).
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._cache: dict[K, tuple[V, float]] = {}

    def get(self, key: K) -> V | None:
        """Return cached value or ``None`` if missing/expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, timestamp = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        """Store a value with the current timestamp."""
        self._cache[key] = (value, time.monotonic())

    def invalidate(self, key: K) -> None:
        """Remove a single entry."""
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        """Remove all entries whose key starts with *prefix*.

        Works for string keys and tuple keys (matches the first element).
        This is useful for invalidating all cache entries for a given
        ``user_id`` regardless of other key components.
        """
        to_delete = []
        for k in self._cache:
            if isinstance(k, str) and k.startswith(prefix):
                to_delete.append(k)
            elif isinstance(k, tuple) and k and str(k[0]) == prefix:
                to_delete.append(k)
        for k in to_delete:
            del self._cache[k]

    def clear(self) -> None:
        """Remove all entries."""
        self._cache.clear()
