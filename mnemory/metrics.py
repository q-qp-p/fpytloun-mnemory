"""Prometheus metrics collection for mnemory.

Exposes two types of metrics:
- **Counters** (in-memory, reset on restart): operation counts per user/agent
- **Gauges** (from Qdrant, cached): memory counts by type/category/status

The MetricsCollector is a module-level singleton, initialized lazily.
When metrics are disabled (ENABLE_METRICS=false), get_collector() returns None
and all instrumentation calls are no-ops.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────

_collector: MetricsCollector | None = None
_collector_initialized = False


def get_collector() -> MetricsCollector | None:
    """Get the metrics collector singleton.

    Returns None if metrics are disabled. Safe to call before init
    (returns None).
    """
    return _collector


def init_collector(
    vector_store: Any,
    session_store: Any,
    config: Any,
) -> MetricsCollector | None:
    """Initialize the metrics collector singleton.

    Called once during server startup. Returns None if metrics are
    disabled in config.

    Args:
        vector_store: VectorStore instance for Qdrant aggregation.
        session_store: SessionStore instance for active session count.
        config: Config instance for metadata and cache TTL.
    """
    global _collector, _collector_initialized

    if _collector_initialized:
        return _collector

    _collector_initialized = True

    if not config.server.enable_metrics:
        logger.info("Prometheus metrics disabled (ENABLE_METRICS=false)")
        return None

    _collector = MetricsCollector(
        vector_store=vector_store,
        session_store=session_store,
        config=config,
    )
    logger.info(
        "Prometheus metrics enabled (cache_ttl=%ds)",
        config.server.metrics_cache_ttl,
    )
    return _collector


# ── MetricsCollector ──────────────────────────────────────────────────


class MetricsCollector:
    """Collects and exposes Prometheus metrics for mnemory.

    Uses a custom CollectorRegistry to avoid polluting the global
    default registry. Gauges from Qdrant are cached with a configurable
    TTL to avoid expensive scroll operations on every scrape.
    """

    def __init__(
        self,
        vector_store: Any,
        session_store: Any,
        config: Any,
    ):
        self._vector_store = vector_store
        self._session_store = session_store
        self._config = config
        self._cache_ttl = config.server.metrics_cache_ttl

        # Cache state
        self._cache_lock = threading.Lock()
        self._cache_timestamp: float = 0.0

        # Optional reference to MaintenanceService for schedule info.
        self._maintenance: Any = None

        # Custom registry (not the global default)
        self._registry = CollectorRegistry()

        # ── Info gauge (static metadata) ──────────────────────────
        self._info = Gauge(
            "mnemory_info",
            "mnemory server metadata (always 1)",
            ["version", "vector_backend", "artifact_backend"],
            registry=self._registry,
        )
        from mnemory import __version__

        self._info.labels(
            version=__version__,
            vector_backend="qdrant" if config.vector.is_remote else "qdrant-local",
            artifact_backend=config.artifact.backend,
        ).set(1)

        # ── Operation counter (in-memory) ─────────────────────────
        self._operations = Counter(
            "mnemory_operations_total",
            "Total number of MCP/REST operations",
            ["operation", "user_id", "agent_id"],
            registry=self._registry,
        )

        # ── Active sessions gauge (live) ──────────────────────────
        self._active_sessions = Gauge(
            "mnemory_active_sessions",
            "Number of active memory sessions",
            registry=self._registry,
        )

        # ── Memory gauges (from Qdrant, cached) ───────────────────
        self._memories_total = Gauge(
            "mnemory_memories_total",
            "Total memories by dimensions",
            ["user_id", "agent_id", "memory_type", "role"],
            registry=self._registry,
        )
        self._memories_decayed = Gauge(
            "mnemory_memories_decayed_total",
            "Total decayed (expired) memories",
            ["user_id", "agent_id"],
            registry=self._registry,
        )
        self._memories_pinned = Gauge(
            "mnemory_memories_pinned_total",
            "Total pinned memories",
            ["user_id", "agent_id"],
            registry=self._registry,
        )
        self._memories_by_category = Gauge(
            "mnemory_memories_by_category_total",
            "Total memories by category",
            ["user_id", "category"],
            registry=self._registry,
        )
        self._memories_with_artifacts = Gauge(
            "mnemory_memories_with_artifacts_total",
            "Total memories that have artifacts attached",
            ["user_id", "agent_id"],
            registry=self._registry,
        )

        # ── Auto-fsck counters (in-memory) ────────────────────────
        self._autofsck_runs = Counter(
            "mnemory_autofsck_runs_total",
            "Total number of automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_issues_found = Counter(
            "mnemory_autofsck_issues_found_total",
            "Total issues found by automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_fixes_applied = Counter(
            "mnemory_autofsck_fixes_applied_total",
            "Total fixes applied by automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )
        self._autofsck_fixes_failed = Counter(
            "mnemory_autofsck_fixes_failed_total",
            "Total fixes that failed during automatic fsck runs",
            ["user_id"],
            registry=self._registry,
        )

        # ── Auto-fsck last-run timestamp gauge ────────────────────
        self._autofsck_last_run = Gauge(
            "mnemory_autofsck_last_run_timestamp",
            "Unix timestamp of the last automatic fsck run per user",
            ["user_id"],
            registry=self._registry,
        )

    # ── Public API ────────────────────────────────────────────────

    def set_maintenance_service(self, maintenance: Any) -> None:
        """Set a reference to the MaintenanceService for schedule info.

        Called once during startup after both MetricsCollector and
        MaintenanceService are created.
        """
        self._maintenance = maintenance

    def record_operation(
        self,
        operation: str,
        user_id: str,
        agent_id: str | None = None,
    ) -> None:
        """Increment the operation counter.

        Args:
            operation: Operation name (e.g., "add_memory", "search_memories").
            user_id: User who performed the operation.
            agent_id: Agent scope (None → empty string label).
        """
        self._operations.labels(
            operation=operation,
            user_id=user_id,
            agent_id=agent_id or "",
        ).inc()

    def record_autofsck_run(
        self,
        *,
        user_id: str,
        issues_found: int,
        fixes_applied: int,
        fixes_failed: int,
    ) -> None:
        """Record the result of one automatic fsck run for a user.

        Args:
            user_id: User the check was run for.
            issues_found: Number of qualifying issues found.
            fixes_applied: Number of fixes successfully applied.
            fixes_failed: Number of fixes that failed.
        """
        self._autofsck_runs.labels(user_id=user_id).inc()
        self._autofsck_issues_found.labels(user_id=user_id).inc(issues_found)
        self._autofsck_fixes_applied.labels(user_id=user_id).inc(fixes_applied)
        self._autofsck_fixes_failed.labels(user_id=user_id).inc(fixes_failed)
        self._autofsck_last_run.labels(user_id=user_id).set(time.time())

    def collect_gauges(self) -> None:
        """Refresh gauge values from Qdrant (with caching).

        Scrolls all points in the collection (without vectors),
        aggregates by dimensions, and updates gauge values. Results
        are cached for ``metrics_cache_ttl`` seconds.

        Also updates the active sessions gauge (always fresh).
        """
        # Active sessions — always fresh (cheap)
        self._active_sessions.set(self._session_store.active_count)

        # Check cache
        now = time.monotonic()
        with self._cache_lock:
            if now - self._cache_timestamp < self._cache_ttl:
                return  # Cache is fresh
            # Mark as refreshing (prevents concurrent refreshes)
            self._cache_timestamp = now

        try:
            self._refresh_gauges_from_qdrant()
        except Exception:
            logger.warning("Failed to refresh metrics from Qdrant", exc_info=True)
            # Reset cache timestamp so next scrape retries
            with self._cache_lock:
                self._cache_timestamp = 0.0

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text exposition format output."""
        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        """Prometheus content type header value."""
        return CONTENT_TYPE_LATEST

    def get_stats_json(self) -> dict:
        """Return metrics as a JSON-serializable dict for the management UI.

        Calls collect_gauges() to ensure freshness, then reads the internal
        prometheus_client gauge/counter values to build a structured response.
        Reuses the cached Qdrant aggregation — no extra scroll.
        """
        self.collect_gauges()

        from mnemory import __version__

        totals = {"memories": 0, "pinned": 0, "decayed": 0, "with_artifacts": 0}
        by_type: dict[str, int] = defaultdict(int)
        by_category: dict[str, int] = defaultdict(int)
        by_role: dict[str, int] = defaultdict(int)
        users: set[str] = set()
        agents: set[str] = set()
        by_user: dict[str, dict] = {}

        def _ensure_user(uid: str) -> dict:
            if uid not in by_user:
                by_user[uid] = {
                    "total": 0,
                    "pinned": 0,
                    "decayed": 0,
                    "with_artifacts": 0,
                    "by_type": {},
                    "by_category": {},
                    "by_role": {},
                }
            return by_user[uid]

        # Read _memories_total gauge: labels are (user_id, agent_id, memory_type, role)
        try:
            for labels, metric in self._memories_total._metrics.items():
                uid, aid, mtype, role = labels
                count = int(metric._value.get())
                totals["memories"] += count
                by_type[mtype] += count
                by_role[role] += count
                users.add(uid)
                if aid:
                    agents.add(aid)
                u = _ensure_user(uid)
                u["total"] += count
                u["by_type"][mtype] = u["by_type"].get(mtype, 0) + count
                u["by_role"][role] = u["by_role"].get(role, 0) + count
        except Exception:
            logger.debug("Failed to read _memories_total gauge", exc_info=True)

        # Read pinned gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_pinned._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["pinned"] += count
                users.add(uid)
                _ensure_user(uid)["pinned"] += count
        except Exception:
            logger.debug("Failed to read _memories_pinned gauge", exc_info=True)

        # Read decayed gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_decayed._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["decayed"] += count
                users.add(uid)
                _ensure_user(uid)["decayed"] += count
        except Exception:
            logger.debug("Failed to read _memories_decayed gauge", exc_info=True)

        # Read with_artifacts gauge: labels are (user_id, agent_id)
        try:
            for labels, metric in self._memories_with_artifacts._metrics.items():
                uid, aid = labels
                count = int(metric._value.get())
                totals["with_artifacts"] += count
                users.add(uid)
                _ensure_user(uid)["with_artifacts"] += count
        except Exception:
            logger.debug("Failed to read _memories_with_artifacts gauge", exc_info=True)

        # Read by_category gauge: labels are (user_id, category)
        try:
            for labels, metric in self._memories_by_category._metrics.items():
                uid, cat = labels
                count = int(metric._value.get())
                by_category[cat] += count
                users.add(uid)
                u = _ensure_user(uid)
                u["by_category"][cat] = u["by_category"].get(cat, 0) + count
        except Exception:
            logger.debug("Failed to read _memories_by_category gauge", exc_info=True)

        # Read operations counter: labels are (operation, user_id, agent_id)
        operations: dict[str, dict] = {}
        try:
            for labels, metric in self._operations._metrics.items():
                op, uid, aid = labels
                count = int(metric._value.get())
                if op not in operations:
                    operations[op] = {"total": 0, "by_user": {}}
                operations[op]["total"] += count
                if uid:
                    operations[op]["by_user"][uid] = (
                        operations[op]["by_user"].get(uid, 0) + count
                    )
        except Exception:
            logger.debug("Failed to read _operations counter", exc_info=True)

        # Sort operations by total descending
        operations = dict(
            sorted(operations.items(), key=lambda x: x[1]["total"], reverse=True)
        )

        # Determine vector/artifact backend from info gauge
        vector_backend = "unknown"
        artifact_backend = "unknown"
        try:
            for labels, metric in self._info._metrics.items():
                version_label, vb, ab = labels
                vector_backend = vb
                artifact_backend = ab
                break
        except Exception:
            pass

        # Build autofsck section
        autofsck_cfg = self._config.memory
        autofsck_enabled = autofsck_cfg.fsck_auto_interval > 0
        autofsck_by_user: dict[str, dict] = {}

        try:
            for labels, metric in self._autofsck_runs._metrics.items():
                (uid,) = labels
                if uid not in autofsck_by_user:
                    autofsck_by_user[uid] = {
                        "runs": 0,
                        "issues_found": 0,
                        "fixes_applied": 0,
                        "fixes_failed": 0,
                        "last_run": None,
                    }
                autofsck_by_user[uid]["runs"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_runs counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_issues_found._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["issues_found"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_issues_found counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_fixes_applied._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["fixes_applied"] = int(metric._value.get())
        except Exception:
            logger.debug(
                "Failed to read _autofsck_fixes_applied counter", exc_info=True
            )

        try:
            for labels, metric in self._autofsck_fixes_failed._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    autofsck_by_user[uid]["fixes_failed"] = int(metric._value.get())
        except Exception:
            logger.debug("Failed to read _autofsck_fixes_failed counter", exc_info=True)

        try:
            for labels, metric in self._autofsck_last_run._metrics.items():
                (uid,) = labels
                if uid in autofsck_by_user:
                    ts = metric._value.get()
                    autofsck_by_user[uid]["last_run"] = ts if ts > 0 else None
        except Exception:
            logger.debug("Failed to read _autofsck_last_run gauge", exc_info=True)

        autofsck_totals = {
            "runs": sum(u["runs"] for u in autofsck_by_user.values()),
            "issues_found": sum(u["issues_found"] for u in autofsck_by_user.values()),
            "fixes_applied": sum(u["fixes_applied"] for u in autofsck_by_user.values()),
            "fixes_failed": sum(u["fixes_failed"] for u in autofsck_by_user.values()),
        }

        return {
            "version": __version__,
            "vector_backend": vector_backend,
            "artifact_backend": artifact_backend,
            "active_sessions": self._session_store.active_count,
            "users": sorted(users),
            "agents": sorted(agents),
            "totals": totals,
            "by_type": dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True)),
            "by_category": dict(
                sorted(by_category.items(), key=lambda x: x[1], reverse=True)
            ),
            "by_role": dict(by_role),
            "by_user": {
                uid: {
                    **udata,
                    "by_type": dict(
                        sorted(
                            udata["by_type"].items(), key=lambda x: x[1], reverse=True
                        )
                    ),
                    "by_category": dict(
                        sorted(
                            udata["by_category"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ),
                    "by_role": dict(udata["by_role"]),
                }
                for uid, udata in sorted(by_user.items())
            },
            "operations": operations,
            "autofsck": {
                "enabled": autofsck_enabled,
                "interval_hours": autofsck_cfg.fsck_auto_interval,
                "min_confidence": autofsck_cfg.fsck_auto_min_confidence,
                "min_severity": autofsck_cfg.fsck_auto_min_severity,
                "by_user": autofsck_by_user,
                "totals": autofsck_totals,
                "next_run_at": (
                    self._maintenance.next_run_at
                    if self._maintenance is not None
                    else None
                ),
                "running": (
                    self._maintenance.is_running
                    if self._maintenance is not None
                    else False
                ),
                "last_check_ids": (
                    self._maintenance.last_check_ids
                    if self._maintenance is not None
                    else {}
                ),
            },
        }

    # ── Internal ──────────────────────────────────────────────────

    def _refresh_gauges_from_qdrant(self) -> None:
        """Scroll all Qdrant points and aggregate into gauge values.

        Clears existing gauge label sets before updating to avoid
        stale labels from deleted users/agents/types.
        """
        client = self._vector_store._client
        collection = self._vector_store.collection_name

        # Aggregation accumulators
        # (user_id, agent_id, memory_type, role) -> count
        total: dict[tuple[str, str, str, str], int] = defaultdict(int)
        # (user_id, agent_id) -> count
        decayed: dict[tuple[str, str], int] = defaultdict(int)
        pinned: dict[tuple[str, str], int] = defaultdict(int)
        with_artifacts: dict[tuple[str, str], int] = defaultdict(int)
        # (user_id, category) -> count
        by_category: dict[tuple[str, str], int] = defaultdict(int)

        # Scroll all points (no vectors, payloads only)
        offset = None
        batch_size = 256

        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                user_id = payload.get("user_id", "unknown")
                agent_id = payload.get("agent_id", "") or ""
                memory_type = payload.get("memory_type", "unknown")
                role = payload.get("role", "user")

                # Total by dimensions
                total[(user_id, agent_id, memory_type, role)] += 1

                # Decayed
                if payload.get("decayed_at"):
                    decayed[(user_id, agent_id)] += 1

                # Pinned
                if payload.get("pinned"):
                    pinned[(user_id, agent_id)] += 1

                # Has artifacts
                artifacts = payload.get("artifacts")
                if artifacts and len(artifacts) > 0:
                    with_artifacts[(user_id, agent_id)] += 1

                # Categories
                categories = payload.get("categories") or []
                for cat in categories:
                    by_category[(user_id, cat)] += 1

            if next_offset is None:
                break
            offset = next_offset

        # Clear existing gauge label sets to remove stale entries
        self._memories_total._metrics.clear()
        self._memories_decayed._metrics.clear()
        self._memories_pinned._metrics.clear()
        self._memories_by_category._metrics.clear()
        self._memories_with_artifacts._metrics.clear()

        # Set new values
        for (uid, aid, mtype, role), count in total.items():
            self._memories_total.labels(
                user_id=uid,
                agent_id=aid,
                memory_type=mtype,
                role=role,
            ).set(count)

        for (uid, aid), count in decayed.items():
            self._memories_decayed.labels(
                user_id=uid,
                agent_id=aid,
            ).set(count)

        for (uid, aid), count in pinned.items():
            self._memories_pinned.labels(
                user_id=uid,
                agent_id=aid,
            ).set(count)

        for (uid, cat), count in by_category.items():
            self._memories_by_category.labels(
                user_id=uid,
                category=cat,
            ).set(count)

        for (uid, aid), count in with_artifacts.items():
            self._memories_with_artifacts.labels(
                user_id=uid,
                agent_id=aid,
            ).set(count)

        logger.debug(
            "Metrics refreshed: %d label sets across %d points",
            len(total)
            + len(decayed)
            + len(pinned)
            + len(by_category)
            + len(with_artifacts),
            sum(total.values()),
        )
