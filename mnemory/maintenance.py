"""Periodic background maintenance service for mnemory.

Runs automatic fsck (memory consistency checks) for all users at a
configurable interval and auto-applies fixes that meet confidence and
severity thresholds.

Usage::

    service = MaintenanceService(config=cfg, fsck=fsck_service, collector=collector)
    await service.start()
    # ... server runs ...
    await service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mnemory.config import Config
    from mnemory.fsck import FsckService
    from mnemory.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Severity ordering for threshold comparison.
_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class MaintenanceService:
    """Periodic background maintenance: auto-fsck for all users.

    When ``fsck_auto_interval`` is 0 (default), the service is disabled
    and ``start()`` / ``stop()`` are no-ops.

    The loop is sleep-first: it waits the full interval before running the
    first check. This avoids hammering the LLM immediately on startup.
    """

    def __init__(
        self,
        config: Config,
        fsck: FsckService,
        collector: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._fsck = fsck
        self._collector = collector
        self._task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background maintenance loop (no-op if disabled)."""
        interval = self._config.memory.fsck_auto_interval
        if interval <= 0:
            logger.debug("Auto-fsck disabled (FSCK_AUTO_INTERVAL=0)")
            return
        logger.info(
            "Auto-fsck enabled: interval=%dh, min_confidence=%.2f, min_severity=%s",
            interval,
            self._config.memory.fsck_auto_min_confidence,
            self._config.memory.fsck_auto_min_severity,
        )
        self._task = asyncio.create_task(self._loop(), name="maintenance-autofsck")

    async def stop(self) -> None:
        """Stop the background maintenance loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.debug("Auto-fsck maintenance loop stopped")

    # ── Loop ─────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main maintenance loop: sleep first, then run."""
        interval_seconds = self._config.memory.fsck_auto_interval * 3600
        while True:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                return
            try:
                await self._run_auto_fsck()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Auto-fsck run failed unexpectedly")

    # ── Core logic ───────────────────────────────────────────────────

    async def _run_auto_fsck(self) -> None:
        """Run fsck for every user and auto-apply qualifying fixes."""
        # list_user_ids() is a synchronous Qdrant scroll — run in thread pool
        # to avoid blocking the event loop.
        user_ids: list[str] = await asyncio.to_thread(self._fsck._vector.list_user_ids)

        if not user_ids:
            logger.debug("Auto-fsck: no users found, skipping run")
            return

        logger.info("Auto-fsck: starting run for %d user(s)", len(user_ids))
        t0 = time.monotonic()

        for user_id in user_ids:
            try:
                await self._run_for_user(user_id)
            except Exception:
                logger.exception(
                    "Auto-fsck: error processing user %s, continuing", user_id
                )

        elapsed = time.monotonic() - t0
        logger.info("Auto-fsck: run complete in %.1fs", elapsed)

    async def _run_for_user(self, user_id: str) -> None:
        """Run fsck for a single user and apply qualifying fixes."""
        min_confidence = self._config.memory.fsck_auto_min_confidence
        min_severity = self._config.memory.fsck_auto_min_severity

        # start_check() is synchronous (just creates a FsckCheck object)
        check = self._fsck.start_check(user_id=user_id)
        check_id = check.check_id

        logger.debug("Auto-fsck: running check %s for user %s", check_id, user_id)

        # run_check() is synchronous and CPU/IO-bound (LLM calls via requests).
        # Run in a thread pool to avoid blocking the event loop.
        await asyncio.to_thread(self._fsck.run_check, check_id)

        check = self._fsck.get_check(check_id)
        if check is None or check.status != "completed":
            status = check.status if check else "not found"
            logger.warning(
                "Auto-fsck: check %s for user %s ended with status=%s",
                check_id,
                user_id,
                status,
            )
            return

        # Filter issues that meet the thresholds
        qualifying_ids = [
            issue.issue_id
            for issue in check.issues
            if self._meets_thresholds(issue, min_confidence, min_severity)
        ]

        issues_found = len(qualifying_ids)
        fixes_applied = 0
        fixes_failed = 0

        if qualifying_ids:
            logger.info(
                "Auto-fsck: applying %d/%d qualifying fixes for user %s",
                issues_found,
                len(check.issues),
                user_id,
            )
            # apply_check() is synchronous (direct vector store mutations)
            result: dict[str, Any] = await asyncio.to_thread(
                self._fsck.apply_check, check_id, qualifying_ids
            )
            fixes_applied = result.get("applied", 0)
            fixes_failed = result.get("failed", 0)
            logger.info(
                "Auto-fsck: user %s — applied=%d, failed=%d",
                user_id,
                fixes_applied,
                fixes_failed,
            )
        else:
            logger.debug(
                "Auto-fsck: no qualifying fixes for user %s "
                "(%d issues below threshold)",
                user_id,
                len(check.issues),
            )

        if self._collector is not None:
            self._collector.record_autofsck_run(
                user_id=user_id,
                issues_found=issues_found,
                fixes_applied=fixes_applied,
                fixes_failed=fixes_failed,
            )

    # ── Helpers ──────────────────────────────────────────────────────

    @classmethod
    def _meets_thresholds(
        cls,
        issue: Any,
        min_confidence: float,
        min_severity: str,
    ) -> bool:
        """Return True if the issue meets both confidence and severity thresholds.

        Args:
            issue: FsckIssue object with ``confidence`` (float) and
                   ``severity`` (str) attributes.
            min_confidence: Minimum confidence score (0.0–1.0).
            min_severity: Minimum severity level ("low", "medium", "high").
        """
        confidence = getattr(issue, "confidence", 0.0) or 0.0
        severity = getattr(issue, "severity", "low") or "low"

        if confidence < min_confidence:
            return False

        issue_sev_order = _SEVERITY_ORDER.get(severity.lower(), 0)
        min_sev_order = _SEVERITY_ORDER.get(min_severity.lower(), 0)
        return issue_sev_order >= min_sev_order
