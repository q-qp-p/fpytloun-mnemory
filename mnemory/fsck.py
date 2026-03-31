"""Memory consistency check (fsck) — detect and fix quality issues.

Three-phase pipeline:
  Phase 0 — Security scan: regex-based prompt injection detection (no LLM)
  Phase 1 — Duplicate detection: vector similarity clustering + LLM evaluation
  Phase 2 — Quality check: LLM-based batch evaluation for spelling, sense,
             split candidates, metadata misclassification, and subtle injection

Results are cached in-memory with configurable TTL so the user can review
issues in the UI and then apply selected fixes without re-running the check.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from qdrant_client.models import PointStruct

from mnemory.categories import (
    PREDEFINED_CATEGORIES,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.config import Config
from mnemory.embeddings import EmbeddingClient
from mnemory.llm import LLMClient, parse_json_response
from mnemory.prompts import (
    build_fsck_content_quality_prompt,
    build_fsck_duplicate_prompt,
    build_fsck_metadata_normalization_prompt,
    build_fsck_security_reeval_prompt,
)
from mnemory.sanitize import detect_injection_patterns
from mnemory.storage.vector import VectorStore

if TYPE_CHECKING:
    from mnemory.memory import MemoryService

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Similarity threshold for duplicate detection (higher than dedup_similarity
# during ingestion because we want to flag clear duplicates, not just related).
_DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Maximum memories per LLM quality-check batch.
_QUALITY_BATCH_SIZE = 20

# Maximum memories per duplicate cluster sent to LLM.
_MAX_CLUSTER_SIZE = 15

# Maximum similar neighbors to check per memory during duplicate detection.
_DUPLICATE_NEIGHBORS = 5


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class FsckAction:
    """A single action to fix an issue."""

    action: str  # "update", "delete", "add"
    memory_id: str | None = None
    new_content: str | None = None
    new_metadata: dict | None = None


@dataclass
class FsckAffectedMemory:
    """A memory affected by an issue."""

    id: str
    content: str
    metadata: dict | None = None
    agent_id: str | None = None


@dataclass
class FsckIssue:
    """A single issue found during memory check."""

    issue_id: str
    type: str  # duplicate, quality, split, contradiction, reclassify, security
    severity: str  # low, medium, high
    reasoning: str
    affected_memories: list[FsckAffectedMemory]
    actions: list[FsckAction]
    confidence: float | None = None  # 0.0-1.0, LLM-reported confidence


@dataclass
class FsckProgress:
    """Progress of a running memory check."""

    phase: str = "starting"
    total_memories: int = 0
    processed: int = 0
    percent: int = 0
    issues_found: int = 0
    truncated: bool = False


@dataclass
class FsckSummary:
    """Summary of issues found."""

    duplicate: int = 0
    quality: int = 0
    split: int = 0
    contradiction: int = 0
    reclassify: int = 0
    security: int = 0
    total: int = 0


@dataclass
class FsckCheck:
    """State of a single memory check run."""

    check_id: str
    user_id: str
    agent_id: str | None = None
    status: str = "running"  # running, applying, completed, failed
    progress: FsckProgress = field(default_factory=FsckProgress)
    issues: list[FsckIssue] = field(default_factory=list)
    summary: FsckSummary | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_seconds: int = 1800
    budget_exhausted: bool = False
    llm_calls: int = 0
    # Track which issues have been applied to prevent double-apply.
    applied_issue_ids: set[str] = field(default_factory=set)

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at > self.ttl_seconds

    @property
    def expires_at_utc(self) -> str:
        """Compute expiration time as ISO 8601 UTC string."""
        created = datetime.fromisoformat(self.created_at_utc)
        expires = created + timedelta(seconds=self.ttl_seconds)
        return expires.isoformat()


# ── FsckStore — in-memory state for check runs ──────────────────────


class FsckStore:
    """Thread-safe in-memory store for fsck check results.

    Similar to SessionStore but for fsck check runs. Checks are stored
    with a TTL and cleaned up periodically via a background sweep task.
    """

    def __init__(self, default_ttl: int = 1800, sweep_interval: int = 300):
        self._checks: dict[str, FsckCheck] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._sweep_interval = sweep_interval
        self._sweep_task: asyncio.Task | None = None

    def create(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> FsckCheck:
        """Create a new check and return it."""
        check = FsckCheck(
            check_id=str(uuid.uuid4()),
            user_id=user_id,
            agent_id=agent_id,
            ttl_seconds=self._default_ttl,
        )
        with self._lock:
            self._checks[check.check_id] = check
        return check

    def get(self, check_id: str) -> FsckCheck | None:
        """Get a check by ID, or None if not found/expired."""
        with self._lock:
            check = self._checks.get(check_id)
            if check is None:
                return None
            if check.is_expired:
                del self._checks[check_id]
                return None
            return check

    def sweep(self) -> int:
        """Remove expired checks. Returns count removed."""
        with self._lock:
            expired = [cid for cid, c in self._checks.items() if c.is_expired]
            for cid in expired:
                del self._checks[cid]
            return len(expired)

    def start_cleanup_task(self) -> None:
        """Start periodic background sweep for expired checks.

        Safe to call multiple times — only starts one task.
        Must be called from within an async context (event loop running).
        """
        if self._sweep_task is not None:
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info("Fsck cleanup task started (interval=%ds)", self._sweep_interval)

    async def stop_cleanup_task(self) -> None:
        """Stop the background sweep task."""
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
            logger.info("Fsck cleanup task stopped")

    async def _sweep_loop(self) -> None:
        """Periodically remove expired checks."""
        while True:
            await asyncio.sleep(self._sweep_interval)
            removed = self.sweep()
            if removed > 0:
                logger.info("Fsck sweep: removed %d expired checks", removed)


# ── Union-Find for clustering ────────────────────────────────────────


class _UnionFind:
    """Simple union-find (disjoint set) for clustering similar memories."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def clusters(self) -> dict[str, list[str]]:
        """Return clusters as {root_id: [member_ids]}."""
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups


# ── FsckService — the check pipeline ────────────────────────────────


class FsckService:
    """Memory consistency check service.

    Runs a three-phase pipeline to detect and suggest fixes for memory
    quality issues. Results are stored in FsckStore for later application.
    """

    def __init__(
        self,
        config: Config,
        vector: VectorStore,
        llm: LLMClient,
        store: FsckStore,
        memory_service: MemoryService | None = None,
    ):
        self._config = config
        self._vector = vector
        self._llm = llm
        self._store = store
        self._memory_service = memory_service
        self._reasoning_effort = config.memory.fsck_reasoning_effort

    # ── Budget helpers ──────────────────────────────────────────────

    def _check_budget(self, check: FsckCheck) -> bool:
        """Return True if LLM budget is exhausted."""
        max_calls = self._config.memory.fsck_max_llm_calls
        return max_calls > 0 and check.llm_calls >= max_calls

    # ── Public API ───────────────────────────────────────────────────

    def start_check(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> FsckCheck:
        """Create a new check and return it (status=running).

        The caller is responsible for running run_check() in a background
        task after this returns, passing any filter parameters (categories,
        memory_type) directly to run_check().
        """
        check = self._store.create(user_id, agent_id)
        return check

    def run_check(
        self,
        check_id: str,
        *,
        categories: list[str] | None = None,
        memory_type: str | None = None,
        include_raw: bool = False,
        incremental: bool = False,
    ) -> None:
        """Execute the full check pipeline. Called in a background task.

        Updates the FsckCheck in-place with progress, issues, and status.

        Phase weight allocation for percent:
          Phase 0a (security_scan):    0 –  3%  (instant, regex only)
          Phase 0b (security_reeval):  3 –  8%  (LLM call per flagged memory)
          Phase 1a (duplicate_search): 8 – 30%  (vector search per memory)
          Phase 1b (duplicate_eval):  30 – 55%  (LLM call per cluster)
          Phase 2 (quality_check):    55 – 100% (LLM call per batch)
        """
        check = self._store.get(check_id)
        if check is None:
            logger.warning("Fsck check %s not found or expired", check_id)
            return

        try:
            # Fetch all memories
            filters: dict[str, Any] = {}
            if memory_type:
                filters["memory_type"] = memory_type

            # When agent_id is set, perform dual-scope scroll: fetch both
            # agent-scoped AND shared (no agent_id) memories. This allows
            # fsck to detect cross-scope duplicates (e.g., same fact stored
            # as both agent-scoped and shared).
            memories = self._vector.scroll_with_vectors(
                user_id=check.user_id,
                agent_id=check.agent_id,
                filters=filters,
                exclude_layers=None if include_raw else ["raw"],
                exclude_expired=True,
            )
            if check.agent_id:
                shared_memories = self._vector.scroll_with_vectors(
                    user_id=check.user_id,
                    agent_id=None,
                    shared_only=True,
                    filters=filters,
                    exclude_layers=None if include_raw else ["raw"],
                    exclude_expired=True,
                )
                # Merge and deduplicate by memory ID
                seen_ids = {m["id"] for m in memories}
                for m in shared_memories:
                    if m["id"] not in seen_ids:
                        seen_ids.add(m["id"])
                        memories.append(m)

            # Filter by categories if specified (client-side since Qdrant
            # MatchAny on array fields needs special handling)
            if categories:
                cat_set = set(categories)
                memories = [
                    m
                    for m in memories
                    if cat_set.intersection(
                        (m.get("metadata") or {}).get("categories", [])
                    )
                ]

            total = len(memories)

            # Apply memory count cap
            max_memories = self._config.memory.fsck_max_memories
            if max_memories > 0 and total > max_memories:
                import random

                logger.info(
                    "Fsck check %s: capping %d memories to %d (random sample)",
                    check_id,
                    total,
                    max_memories,
                )
                memories = random.sample(memories, max_memories)
                total = len(memories)
                check.progress.truncated = True

            # Incremental mode: only check memories changed since last maintenance.
            # A memory is "changed" if it has no checked_at or its updated_at_utc
            # is more recent than checked_at. checked_at is stamped all-or-nothing
            # after a memory passes through ALL enabled phases.
            if incremental:
                full_corpus = list(
                    memories
                )  # Keep full corpus for duplicate neighbor search
                memories = [
                    m
                    for m in memories
                    if not (m.get("metadata") or {}).get("checked_at")
                    or (m.get("metadata") or {}).get("updated_at_utc", "")
                    > (m.get("metadata") or {}).get("checked_at", "")
                ]
                logger.info(
                    "Fsck check %s: incremental mode — %d changed out of %d total",
                    check_id,
                    len(memories),
                    len(full_corpus),
                )
            else:
                full_corpus = memories

            total = len(memories)
            check.progress.total_memories = total
            logger.info("Fsck check %s: %d memories to check", check_id, total)

            if total == 0:
                check.status = "completed"
                check.summary = FsckSummary()
                check.progress.phase = "done"
                check.progress.percent = 100
                return

            # mem_by_id is the working set. In incremental mode, full_corpus
            # is retained above for potential future cross-referencing, but
            # duplicate neighbor filtering uses mem_by_id (working set only).
            mem_by_id: dict[str, dict] = {m["id"]: m for m in memories}

            # Phase 0a: Security scan (regex, no LLM) — 0-3%
            check.progress.phase = "security_scan"
            check.progress.processed = 0
            logger.info(
                "Fsck check %s phase 0a: scanning %d memories for injection patterns",
                check_id,
                total,
            )
            regex_flagged = self._phase_security_scan_regex(memories)
            check.progress.processed = total
            check.progress.percent = 3

            # Phase 0b: Security re-evaluation (LLM) — 3-8%
            # Re-evaluate regex hits to drop false positives before adding issues.
            logger.info(
                "Fsck check %s phase 0b: re-evaluating %d security flags with LLM",
                check_id,
                len(regex_flagged),
            )
            security_issues = self._phase_security_reeval(regex_flagged)
            check.issues.extend(security_issues)
            check.progress.percent = 8
            check.progress.issues_found = len(check.issues)

            # Estimate LLM calls for budget tracking
            check.llm_calls += len(regex_flagged)  # 1 call per flagged memory

            # Phases execute in order; under budget pressure, earlier phases
            # take precedence. Later phases catch up over subsequent
            # incremental runs.
            if self._check_budget(check):
                logger.info(
                    "Fsck check %s: LLM budget exhausted (%d calls), stopping after security phase",
                    check_id,
                    check.llm_calls,
                )
                check.budget_exhausted = True
                check.progress.phase = "done"
                check.summary = self._build_summary(check.issues)
                check.status = "completed"
                return

            # Phase 1: Duplicate detection (vector similarity + LLM) — 8-55%
            logger.info(
                "Fsck check %s phase 1: duplicate detection on %d memories",
                check_id,
                total,
            )
            dup_issues, clustered_ids = self._phase_duplicate_detection(
                memories, mem_by_id, check
            )
            check.issues.extend(dup_issues)
            check.progress.issues_found = len(check.issues)

            # Estimate LLM calls for duplicate evaluation (1 per cluster).
            # clustered_ids is a flat set; average cluster size ~2-3 members,
            # so number of clusters ≈ len(clustered_ids) / 2.
            estimated_dup_clusters = (
                max(len(clustered_ids) // 2, 1) if clustered_ids else 0
            )
            check.llm_calls += estimated_dup_clusters

            if self._check_budget(check):
                logger.info(
                    "Fsck check %s: LLM budget exhausted (%d calls), stopping after duplicate phase",
                    check_id,
                    check.llm_calls,
                )
                check.budget_exhausted = True
                check.progress.phase = "done"
                check.summary = self._build_summary(check.issues)
                check.status = "completed"
                return

            # Phase 2: Quality check (LLM batches) — 55-100%
            # Only exclude memories that were part of confirmed duplicate/contradiction
            # issues. Memories in clusters that the LLM declared clean should still
            # reach quality review.
            confirmed_issue_mem_ids: set[str] = set()
            for issue in dup_issues:
                for am in issue.affected_memories:
                    confirmed_issue_mem_ids.add(am.id)
            quality_memories = [
                m for m in memories if m["id"] not in confirmed_issue_mem_ids
            ]
            logger.info(
                "Fsck check %s phase 2: quality check on %d memories (%d skipped, in confirmed issues)",
                check_id,
                len(quality_memories),
                len(memories) - len(quality_memories),
            )
            # Load available categories for this user so the LLM only proposes valid ones
            available_categories = self._get_available_categories(check.user_id)
            quality_issues = self._phase_quality_check(
                quality_memories, check, available_categories=available_categories
            )
            check.issues.extend(quality_issues)
            check.progress.issues_found = len(check.issues)
            check.progress.percent = 100

            # Estimate LLM calls for quality check (2 passes per batch:
            # content quality + metadata normalization).
            quality_batches = (
                (len(quality_memories) + _QUALITY_BATCH_SIZE - 1) // _QUALITY_BATCH_SIZE
                if quality_memories
                else 0
            )
            check.llm_calls += (
                quality_batches * 2
            )  # 2 LLM calls per batch (Pass A + Pass B)

            # Stamp checked_at on memories WITHOUT issues.
            # Memories with unresolved issues are intentionally left
            # unstamped so the next incremental run re-examines them.
            # If an issue IS auto-applied later, the metadata update
            # refreshes updated_at_utc, making the memory eligible for
            # re-check anyway.
            now_utc = datetime.now(timezone.utc).isoformat()
            issue_memory_ids: set[str] = set()
            for issue in check.issues:
                for am in issue.affected_memories:
                    issue_memory_ids.add(am.id)
            clean_ids = [m["id"] for m in memories if m["id"] not in issue_memory_ids]
            if clean_ids:
                for mid in clean_ids:
                    try:
                        self._vector.update_metadata(mid, {"checked_at": now_utc})
                    except Exception:
                        logger.warning(
                            "Failed to stamp checked_at on memory %s",
                            mid,
                            exc_info=True,
                        )
                logger.info(
                    "Fsck check %s: stamped checked_at on %d/%d memories "
                    "(%d with issues left unstamped for re-check)",
                    check_id,
                    len(clean_ids),
                    len(memories),
                    len(issue_memory_ids),
                )

            # Build summary
            check.progress.phase = "done"
            check.summary = self._build_summary(check.issues)
            check.status = "completed"

            logger.info(
                "Fsck check %s completed: %d memories, %d issues found",
                check_id,
                total,
                len(check.issues),
            )

        except Exception as e:
            logger.exception("Fsck check %s failed", check_id)
            check.status = "failed"
            check.error = str(e)

    def get_check(self, check_id: str) -> FsckCheck | None:
        """Get a check by ID."""
        return self._store.get(check_id)

    def apply_check(
        self,
        check_id: str,
        issue_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply fixes from a completed check.

        Idempotent: issues that have already been applied are skipped
        (tracked in check.applied_issue_ids). Safe to call multiple times
        with different issue_ids to apply fixes incrementally.

        Args:
            check_id: The check to apply.
            issue_ids: Specific issue IDs to apply. None/empty = apply all.

        Returns:
            Dict with applied/skipped/failed counts and details.
        """
        check = self._store.get(check_id)
        if check is None:
            return {
                "error": True,
                "message": "Check not found or expired. Please re-run the check.",
            }
        if check.status not in ("completed", "applying"):
            return {
                "error": True,
                "message": f"Check is not completed (status: {check.status})",
            }

        # Select issues to apply
        if issue_ids:
            id_set = set(issue_ids)
            issues = [i for i in check.issues if i.issue_id in id_set]
        else:
            issues = list(check.issues)

        applied = 0
        skipped = 0
        failed = 0
        details: list[dict] = []

        for issue in issues:
            # Idempotency: skip issues that were already applied.
            if issue.issue_id in check.applied_issue_ids:
                skipped += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "skipped",
                        "actions_executed": 0,
                        "actions_skipped": 0,
                    }
                )
                continue

            try:
                actions_executed, actions_skipped = self._apply_issue(
                    issue, check.user_id
                )
                # Only mark as applied (and prevent retry) when at least one
                # action actually executed. If all actions were skipped (e.g.,
                # memory not found), leave the issue available for future apply
                # attempts so transient misses don't permanently block fixes.
                if actions_executed > 0:
                    check.applied_issue_ids.add(issue.issue_id)
                applied += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "applied",
                        "actions_executed": actions_executed,
                        "actions_skipped": actions_skipped,
                    }
                )
            except Exception as e:
                logger.warning(
                    "Failed to apply fsck issue %s: %s",
                    issue.issue_id,
                    e,
                )
                failed += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "failed",
                        "actions_executed": 0,
                        "actions_skipped": 0,
                        "error": str(e),
                    }
                )

        # Invalidate caches so mutations are reflected immediately.
        if self._memory_service is not None and (applied > 0 or failed > 0):
            try:
                self._memory_service._core_cache.invalidate_prefix(check.user_id)
                self._memory_service._category_cache.invalidate(check.user_id)
            except Exception:
                logger.warning(
                    "Fsck apply: failed to invalidate caches for user %s",
                    check.user_id,
                    exc_info=True,
                )

        return {
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    # ── Phase 0: Security scan ───────────────────────────────────────

    def _phase_security_scan_regex(
        self,
        memories: list[dict],
    ) -> list[tuple[dict, list[str]]]:
        """Scan all memories for prompt injection patterns using regex.

        Fast, no LLM cost. Uses the existing detect_injection_patterns()
        from sanitize.py.

        Returns a list of (memory, matched_patterns) tuples for flagged memories.
        """
        flagged: list[tuple[dict, list[str]]] = []

        for mem in memories:
            text = mem.get("memory", "")
            patterns = detect_injection_patterns(text)
            if patterns:
                flagged.append((mem, patterns))

        if flagged:
            logger.info(
                "Fsck security scan: %d memories flagged by regex (pending LLM re-eval)",
                len(flagged),
            )

        return flagged

    def _phase_security_reeval(
        self,
        flagged: list[tuple[dict, list[str]]],
    ) -> list[FsckIssue]:
        """Re-evaluate regex-flagged memories with LLM to drop false positives.

        Runs in parallel using the configured concurrency. Only confirmed
        threats become FsckIssue objects — false positives are silently dropped.
        """
        if not flagged:
            return []

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        issues: list[FsckIssue] = []
        lock = threading.Lock()

        def _reeval(mem: dict, patterns: list[str]) -> FsckIssue | None:
            return self._evaluate_security_flag(mem, patterns)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_reeval, mem, patterns): (mem, patterns)
                for mem, patterns in flagged
            }
            for future in as_completed(futures):
                mem, patterns = futures[future]
                try:
                    issue = future.result()
                    if issue is not None:
                        with lock:
                            issues.append(issue)
                except Exception:
                    logger.warning(
                        "Failed to re-evaluate security flag for memory %s",
                        mem.get("id"),
                        exc_info=True,
                    )
                    # On LLM failure, include the issue conservatively
                    with lock:
                        issues.append(
                            FsckIssue(
                                issue_id=str(uuid.uuid4()),
                                type="security",
                                severity="high",
                                reasoning=(
                                    f"Prompt injection patterns detected: {', '.join(patterns)}. "
                                    "LLM re-evaluation failed — flagged conservatively."
                                ),
                                affected_memories=[self._make_affected_memory(mem)],
                                actions=[
                                    FsckAction(action="delete", memory_id=mem["id"])
                                ],
                            )
                        )

        confirmed = len(issues)
        dropped = len(flagged) - confirmed
        logger.info(
            "Fsck security re-eval: %d confirmed threats, %d false positives dropped",
            confirmed,
            dropped,
        )
        return issues

    def _evaluate_security_flag(
        self,
        mem: dict,
        patterns: list[str],
    ) -> FsckIssue | None:
        """Ask the LLM whether a regex-flagged memory is a real threat.

        Returns an FsckIssue if confirmed threat, None if false positive.
        """
        # Security re-evaluation handles one memory at a time, so aliasing the
        # memory ID is unnecessary here unlike duplicate/quality batch prompts.
        messages, schema = build_fsck_security_reeval_prompt(mem, patterns)

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_security",
        )

        try:
            parsed = parse_json_response(response)
        except Exception:
            parsed = None

        if not parsed:
            # Parsing failed — include conservatively
            logger.warning(
                "Security re-eval: failed to parse LLM response for memory %s",
                mem.get("id"),
            )
            verdict = "threat"
            reasoning = "LLM response could not be parsed — flagged conservatively."
        else:
            verdict = parsed.get("verdict", "threat")
            reasoning = parsed.get("reasoning", "")

        if verdict == "false_positive":
            logger.debug(
                "Security re-eval: memory %s is a false positive: %s",
                mem.get("id"),
                reasoning,
            )
            return None

        # Confirmed threat
        return FsckIssue(
            issue_id=str(uuid.uuid4()),
            type="security",
            severity="high",
            reasoning=(
                f"Prompt injection patterns detected: {', '.join(patterns)}. "
                f"LLM confirmed: {reasoning}"
            ),
            affected_memories=[self._make_affected_memory(mem)],
            actions=[FsckAction(action="delete", memory_id=mem["id"])],
        )

    # ── Phase 1: Duplicate detection ─────────────────────────────────

    def _phase_duplicate_detection(
        self,
        memories: list[dict],
        mem_by_id: dict[str, dict],
        check: FsckCheck,
    ) -> tuple[list[FsckIssue], set[str]]:
        """Find duplicate clusters via vector similarity, then evaluate with LLM.

        Sub-phase 1a (duplicate_search): Build similarity graph — 5-30%
        Sub-phase 1b (duplicate_eval): Evaluate clusters with LLM — 30-55%

        Returns (issues, set of memory IDs that were part of clusters).
        """
        total = len(memories)

        # ── Sub-phase 1a: Build similarity graph ────────────────────
        check.progress.phase = "duplicate_search"
        check.progress.processed = 0
        uf = _UnionFind()

        for idx, mem in enumerate(memories):
            vector = mem.get("vector")
            if vector is not None:
                # Search for similar memories using stored vector.
                # When agent_id is set, also search shared memories
                # to detect cross-scope duplicates.
                similar = self._vector.search_by_vector_ids(
                    vector,
                    user_id=check.user_id,
                    agent_id=check.agent_id,
                    limit=_DUPLICATE_NEIGHBORS,
                    exclude_ids=[mem["id"]],
                )
                if check.agent_id:
                    shared_similar = self._vector.search_by_vector_ids(
                        vector,
                        user_id=check.user_id,
                        agent_id=None,
                        shared_only=True,
                        limit=_DUPLICATE_NEIGHBORS,
                        exclude_ids=[mem["id"]],
                    )
                    # Merge, deduplicate by ID, keep highest score
                    seen = {s["id"] for s in similar}
                    for s in shared_similar:
                        if s["id"] not in seen:
                            seen.add(s["id"])
                            similar.append(s)

                # Only neighbors that survived the main run_check() filters
                # may influence duplicate clustering. This prevents excluded
                # raw memories from acting as graph bridges between durable
                # memories during union-find.
                similar = [s for s in similar if s.get("id") in mem_by_id]

                for sim in similar:
                    score = sim.get("score", 0)
                    if score >= _DUPLICATE_SIMILARITY_THRESHOLD:
                        uf.union(mem["id"], sim["id"])

            check.progress.processed = idx + 1
            # 8-30% range
            check.progress.percent = 8 + int(22 * (idx + 1) / total) if total else 30

        # Extract clusters with 2+ members
        clusters = {
            root: members
            for root, members in uf.clusters().items()
            if len(members) >= 2
        }

        logger.info(
            "Fsck duplicate search: found %d clusters from %d memories",
            len(clusters),
            total,
        )

        if not clusters:
            check.progress.percent = 55
            return [], set()

        # ── Sub-phase 1b: Evaluate clusters with LLM (parallel) ────
        check.progress.phase = "duplicate_eval"
        check.progress.processed = 0
        issues: list[FsckIssue] = []
        clustered_ids: set[str] = set()
        cluster_list = list(clusters.items())
        total_clusters = len(cluster_list)

        # Prepare cluster data and collect clustered IDs
        cluster_inputs: list[tuple[str, list[dict]]] = []
        for _root, member_ids in cluster_list:
            member_ids = member_ids[:_MAX_CLUSTER_SIZE]
            clustered_ids.update(member_ids)
            cluster_mems = [mem_by_id[mid] for mid in member_ids if mid in mem_by_id]
            if len(cluster_mems) >= 2:
                cluster_inputs.append((_root, cluster_mems))

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        progress_lock = threading.Lock()
        completed_count = 0

        def _eval_cluster(root: str, mems: list[dict]) -> list[FsckIssue]:
            return self._evaluate_duplicate_cluster(mems)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_eval_cluster, root, mems): root
                for root, mems in cluster_inputs
            }

            for future in as_completed(futures):
                root = futures[future]
                try:
                    cluster_issues = future.result()
                    with progress_lock:
                        issues.extend(cluster_issues)
                        completed_count += 1
                        check.progress.processed = completed_count
                        check.progress.issues_found = len(check.issues) + len(issues)
                        check.progress.percent = (
                            30 + int(25 * completed_count / total_clusters)
                            if total_clusters
                            else 55
                        )
                except Exception:
                    logger.warning(
                        "Failed to evaluate duplicate cluster (root=%s)",
                        root,
                        exc_info=True,
                    )
                    with progress_lock:
                        completed_count += 1
                        check.progress.processed = completed_count
                        check.progress.percent = (
                            30 + int(25 * completed_count / total_clusters)
                            if total_clusters
                            else 55
                        )

        logger.info(
            "Fsck duplicate detection: %d clusters (%d evaluated), %d issues",
            len(clusters),
            len(cluster_inputs),
            len(issues),
        )

        return issues, clustered_ids

    def _evaluate_duplicate_cluster(
        self,
        cluster: list[dict],
    ) -> list[FsckIssue]:
        """Send a cluster of similar memories to LLM for duplicate evaluation."""
        messages, schema, id_mapping = build_fsck_duplicate_prompt(
            cluster,
            max_memory_length=self._config.memory.max_memory_length,
        )

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_dedup",
        )

        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in cluster}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            actions, action_target_ids = self._parse_issue_actions(
                raw_issue, id_mapping, mem_lookup
            )
            affected_mems = self._resolve_affected_memories(
                raw_issue, id_mapping, mem_lookup, action_target_ids
            )
            if not affected_mems:
                logger.warning(
                    "Fsck duplicate check: dropping issue with no resolvable affected memories (type=%s, raw_ids=%s)",
                    raw_issue.get("type", "duplicate"),
                    raw_issue.get("affected_memory_ids", []),
                )
                continue

            issue_type = raw_issue.get("type", "duplicate")
            if issue_type not in ("duplicate", "contradiction"):
                issue_type = "duplicate"

            severity = raw_issue.get("severity", "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"

            raw_confidence = raw_issue.get("confidence")
            confidence: float | None = None
            if raw_confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (TypeError, ValueError):
                    confidence = None

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=severity,
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                    confidence=confidence,
                )
            )

        return issues

    # ── Phase 2: Quality check ───────────────────────────────────────

    def _get_available_categories(self, user_id: str) -> list[str]:
        """Return valid category names for this user.

        Combines predefined categories with any dynamic project:* categories
        found in the user's memories. Falls back to predefined list on error.
        """
        try:
            if self._memory_service is not None:
                cats = self._memory_service.list_categories(user_id=user_id)
                return [c["name"] for c in cats.get("categories", []) if c.get("name")]
        except Exception:
            logger.warning("Failed to load categories for fsck, using predefined list")
        return list(PREDEFINED_CATEGORIES.keys())

    def _phase_quality_check(
        self,
        memories: list[dict],
        check: FsckCheck,
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Batch memories and send to LLM for quality evaluation.

        Checks for spelling, sense/completeness, split candidates,
        metadata misclassification, and subtle injection patterns.

        Progress: 55-100% range.
        """
        check.progress.phase = "quality_check"
        check.progress.processed = 0
        issues: list[FsckIssue] = []
        total = len(memories)
        total_batches = (total + _QUALITY_BATCH_SIZE - 1) // _QUALITY_BATCH_SIZE

        # Pre-split into batches
        batches: list[tuple[int, list[dict]]] = []
        for i in range(0, total, _QUALITY_BATCH_SIZE):
            batches.append((i, memories[i : i + _QUALITY_BATCH_SIZE]))

        concurrency = max(1, self._config.memory.fsck_llm_concurrency)
        progress_lock = threading.Lock()
        processed_mems = 0

        def _eval_batch(batch: list[dict]) -> list[FsckIssue]:
            return self._evaluate_quality_batch(
                batch, available_categories=available_categories
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_eval_batch, batch): (offset, batch)
                for offset, batch in batches
            }

            for future in as_completed(futures):
                offset, batch = futures[future]
                try:
                    batch_issues = future.result()
                    with progress_lock:
                        issues.extend(batch_issues)
                        processed_mems += len(batch)
                        check.progress.processed = processed_mems
                        check.progress.issues_found = len(check.issues) + len(issues)
                        check.progress.percent = (
                            55 + int(45 * processed_mems / total) if total else 100
                        )
                except Exception:
                    logger.warning(
                        "Failed to evaluate quality batch (offset=%d, size=%d)",
                        offset,
                        len(batch),
                        exc_info=True,
                    )
                    with progress_lock:
                        processed_mems += len(batch)
                        check.progress.processed = processed_mems
                        check.progress.percent = (
                            55 + int(45 * processed_mems / total) if total else 100
                        )

        logger.info(
            "Fsck quality check: %d memories in %d batches, %d issues",
            total,
            total_batches,
            len(issues),
        )

        return issues

    def _evaluate_quality_batch(
        self,
        batch: list[dict],
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Send a batch of memories through both quality passes.

        Runs Pass A (content quality) and Pass B (metadata normalization)
        sequentially and merges the results.
        """
        issues: list[FsckIssue] = []
        issues.extend(self._evaluate_content_quality_batch(batch))
        issues.extend(
            self._evaluate_metadata_normalization_batch(
                batch, available_categories=available_categories
            )
        )
        return issues

    def _evaluate_content_quality_batch(
        self,
        batch: list[dict],
    ) -> list[FsckIssue]:
        """Pass A: check a batch of memories for content quality issues."""
        messages, schema, id_mapping = build_fsck_content_quality_prompt(batch)

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_content_quality",
        )

        return self._parse_quality_response(
            response, id_mapping, batch, default_type="quality"
        )

    def _evaluate_metadata_normalization_batch(
        self,
        batch: list[dict],
        *,
        available_categories: list[str] | None = None,
    ) -> list[FsckIssue]:
        """Pass B: check a batch of memories for metadata issues."""
        messages, schema, id_mapping = build_fsck_metadata_normalization_prompt(
            batch, available_categories=available_categories
        )

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            reasoning_effort=self._reasoning_effort,
            operation="fsck_metadata_normalization",
        )

        return self._parse_quality_response(
            response, id_mapping, batch, default_type="reclassify"
        )

    def _parse_quality_response(
        self,
        response: str,
        id_mapping: dict[str, str],
        batch: list[dict],
        *,
        default_type: str = "quality",
    ) -> list[FsckIssue]:
        """Parse an LLM quality/metadata response into ``FsckIssue`` objects."""
        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in batch}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            actions, action_target_ids = self._parse_issue_actions(
                raw_issue, id_mapping, mem_lookup
            )
            affected_mems = self._resolve_affected_memories(
                raw_issue, id_mapping, mem_lookup, action_target_ids
            )
            if not affected_mems:
                logger.warning(
                    "Fsck quality check: dropping issue with no resolvable affected memories (type=%s, raw_ids=%s)",
                    raw_issue.get("type", default_type),
                    raw_issue.get("affected_memory_ids", []),
                )
                continue

            issue_type = raw_issue.get("type", default_type)
            if issue_type not in (
                "quality",
                "reclassify",
            ):
                issue_type = default_type

            severity = raw_issue.get("severity", "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"

            raw_confidence = raw_issue.get("confidence")
            confidence: float | None = None
            if raw_confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                except (TypeError, ValueError):
                    confidence = None

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=severity,
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                    confidence=confidence,
                )
            )

        return issues

    @staticmethod
    def _get_memory_agent_id(mem: dict[str, Any]) -> str | None:
        """Return the memory agent ID from the normalized memory shape."""
        agent_id = mem.get("agent_id")
        if agent_id is not None:
            return str(agent_id)
        metadata = mem.get("metadata") or {}
        fallback = metadata.get("agent_id")
        return str(fallback) if fallback is not None else None

    def _make_affected_memory(self, mem: dict[str, Any]) -> FsckAffectedMemory:
        """Convert a memory dict into an affected-memory payload."""
        return FsckAffectedMemory(
            id=str(mem.get("id", "")),
            content=mem.get("memory", ""),
            metadata=mem.get("metadata"),
            agent_id=self._get_memory_agent_id(mem),
        )

    @staticmethod
    def _resolve_llm_memory_id(
        raw_memory_id: Any,
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
    ) -> str | None:
        """Resolve an LLM-returned alias or direct ID to a real memory ID."""
        if raw_memory_id is None:
            return None
        candidate = str(raw_memory_id).strip()
        if not candidate:
            return None
        resolved = id_mapping.get(candidate, candidate)
        return resolved if resolved in mem_lookup else None

    def _parse_issue_actions(
        self,
        raw_issue: dict[str, Any],
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
    ) -> tuple[list[FsckAction], list[str]]:
        """Parse and validate fsck issue actions from LLM output."""
        actions: list[FsckAction] = []
        action_target_ids: list[str] = []

        for raw_action in raw_issue.get("actions", []):
            if not isinstance(raw_action, dict):
                continue
            action_type = raw_action.get("action", "")
            resolved_memory_id = self._resolve_llm_memory_id(
                raw_action.get("memory_id"), id_mapping, mem_lookup
            )
            if action_type != "add" and resolved_memory_id is None:
                continue
            actions.append(
                FsckAction(
                    action=action_type,
                    memory_id=resolved_memory_id,
                    new_content=raw_action.get("new_content"),
                    new_metadata=raw_action.get("new_metadata"),
                )
            )
            if resolved_memory_id and resolved_memory_id not in action_target_ids:
                action_target_ids.append(resolved_memory_id)

        return actions, action_target_ids

    def _resolve_affected_memories(
        self,
        raw_issue: dict[str, Any],
        id_mapping: dict[str, str],
        mem_lookup: dict[str, dict[str, Any]],
        action_target_ids: list[str],
    ) -> list[FsckAffectedMemory]:
        """Resolve issue affected memories from aliases and validated actions."""
        resolved_ids: list[str] = []
        for raw_memory_id in raw_issue.get("affected_memory_ids", []):
            resolved = self._resolve_llm_memory_id(
                raw_memory_id, id_mapping, mem_lookup
            )
            if resolved and resolved not in resolved_ids:
                resolved_ids.append(resolved)

        for memory_id in action_target_ids:
            if memory_id not in resolved_ids:
                resolved_ids.append(memory_id)

        return [self._make_affected_memory(mem_lookup[mid]) for mid in resolved_ids]

    # ── Apply logic ──────────────────────────────────────────────────

    def _apply_issue(self, issue: FsckIssue, user_id: str) -> tuple[int, int]:
        """Apply all actions for a single issue.

        Returns:
            Tuple of (actions_executed, actions_skipped).
        """
        executed = 0
        skipped = 0

        # Guard: skip split actions on memories with artifacts.
        # Splitting would delete the original (destroying artifacts) and
        # create new memories without them. The prompt should prevent this,
        # but this is a hard safety net.
        if issue.type == "split":
            for action in issue.actions:
                if action.action == "delete" and action.memory_id:
                    mem = self._vector.get_by_id(action.memory_id)
                    if mem and (mem.get("metadata") or {}).get("artifacts"):
                        logger.warning(
                            "Fsck apply: skipping split of memory %s — "
                            "has %d artifact(s) that would be destroyed",
                            action.memory_id,
                            len(mem["metadata"]["artifacts"]),
                        )
                        return 0, len(issue.actions)

        for action in issue.actions:
            if action.action == "delete" and action.memory_id:
                # Verify ownership before deleting
                existing = self._vector.get_by_id(action.memory_id)
                if existing is None:
                    logger.warning(
                        "Fsck apply: memory %s not found, skipping delete",
                        action.memory_id,
                    )
                    skipped += 1
                    continue
                if existing.get("user_id") != user_id:
                    logger.warning(
                        "Fsck apply: memory %s belongs to a different user, skipping delete",
                        action.memory_id,
                    )
                    skipped += 1
                    continue
                # Warn when deleting a memory with artifacts — the prompt
                # should prevent this in most cases, but log for visibility.
                existing_meta = existing.get("metadata") or {}
                if existing_meta.get("artifacts"):
                    logger.warning(
                        "Fsck apply: deleting memory %s which has %d artifact(s)",
                        action.memory_id,
                        len(existing_meta["artifacts"]),
                    )
                # Route through MemoryService to clean up artifacts and
                # invalidate caches. Fall back to direct delete in tests.
                if self._memory_service is not None:
                    self._memory_service.delete_memory(
                        memory_id=action.memory_id, user_id=user_id
                    )
                else:
                    self._vector.delete(action.memory_id)
                executed += 1

            elif action.action == "update" and action.memory_id:
                # Verify ownership before updating
                existing = self._vector.get_by_id(action.memory_id)
                if existing is None:
                    logger.warning(
                        "Fsck apply: memory %s not found, skipping update",
                        action.memory_id,
                    )
                    skipped += 1
                    continue
                if existing.get("user_id") != user_id:
                    logger.warning(
                        "Fsck apply: memory %s belongs to a different user, skipping update",
                        action.memory_id,
                    )
                    skipped += 1
                    continue
                if action.new_content:
                    # Generate sparse vector for hybrid search if available
                    sparse_vector = None
                    if self._memory_service is not None:
                        sparse_vector = self._memory_service._get_sparse_vector(
                            action.new_content
                        )
                    self._vector.update_content(
                        action.memory_id,
                        action.new_content,
                        sparse_vector=sparse_vector,
                    )
                    # Stamp updated_at_utc so incremental fsck picks up the change.
                    self._vector.update_metadata(
                        action.memory_id,
                        {"updated_at_utc": datetime.now(timezone.utc).isoformat()},
                    )
                    executed += 1
                if action.new_metadata:
                    # Filter to allowed metadata fields, dropping None values
                    # (None means "unchanged" in LLM output — don't overwrite).
                    allowed = {
                        "memory_type",
                        "categories",
                        "importance",
                        "pinned",
                        "role",
                        "labels",
                    }
                    clean_meta = {
                        k: v
                        for k, v in action.new_metadata.items()
                        if k in allowed and v is not None
                    }
                    if clean_meta:
                        # Validate memory_type — strip LLM-hallucinated values
                        # and reject promotions that contradict heuristic rules.
                        if "memory_type" in clean_meta:
                            try:
                                clean_meta["memory_type"] = validate_memory_type(
                                    clean_meta["memory_type"]
                                )
                            except ValueError:
                                logger.warning(
                                    "Fsck apply: stripping invalid memory_type '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["memory_type"],
                                    action.memory_id,
                                )
                                del clean_meta["memory_type"]
                        # Heuristic safety net: if the fsck model suggests
                        # promoting to "fact" or "preference", validate the
                        # memory text against our post-LLM heuristic patterns.
                        # We test as if the type were "fact" — if the heuristic
                        # demotes it (→ episodic/context), the text clearly
                        # shouldn't be permanent, so promoting to fact OR
                        # preference is wrong.
                        if "memory_type" in clean_meta and clean_meta[
                            "memory_type"
                        ] in ("fact", "preference"):
                            from mnemory.prompts import _correct_memory_type

                            mem_text = existing.get("memory", "")
                            corrected = _correct_memory_type("fact", mem_text)
                            if corrected != "fact":
                                logger.info(
                                    "Fsck apply: rejecting promotion to '%s' "
                                    "for memory %s — heuristic says '%s' "
                                    "(text: %.80s)",
                                    clean_meta["memory_type"],
                                    action.memory_id,
                                    corrected,
                                    mem_text,
                                )
                                del clean_meta["memory_type"]
                        # Validate importance — strip LLM-hallucinated values.
                        if "importance" in clean_meta:
                            try:
                                clean_meta["importance"] = validate_importance(
                                    clean_meta["importance"]
                                )
                            except ValueError:
                                logger.warning(
                                    "Fsck apply: stripping invalid importance '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["importance"],
                                    action.memory_id,
                                )
                                del clean_meta["importance"]
                        # Validate categories — strip any LLM-hallucinated
                        # categories that don't exist in the predefined set,
                        # keeping valid ones rather than dropping the whole field.
                        if (
                            "categories" in clean_meta
                            and clean_meta["categories"] is not None
                        ):
                            valid_cats = []
                            for cat in clean_meta["categories"]:
                                try:
                                    valid_cats.extend(validate_categories([cat]))
                                except ValueError as cat_err:
                                    logger.warning(
                                        "Fsck apply: stripping invalid category '%s' "
                                        "from reclassify action for memory %s: %s",
                                        cat,
                                        action.memory_id,
                                        cat_err,
                                    )
                            if valid_cats:
                                clean_meta["categories"] = valid_cats
                            else:
                                del clean_meta["categories"]
                        # Validate role — must be "user" or "assistant",
                        # and "assistant" requires the memory to have an
                        # agent_id (invariant from add_memory validation).
                        if "role" in clean_meta:
                            if clean_meta["role"] not in ("user", "assistant"):
                                logger.warning(
                                    "Fsck apply: stripping invalid role '%s' "
                                    "from reclassify action for memory %s",
                                    clean_meta["role"],
                                    action.memory_id,
                                )
                                del clean_meta["role"]
                            elif clean_meta["role"] == "assistant" and not existing.get(
                                "agent_id"
                            ):
                                logger.warning(
                                    "Fsck apply: cannot set role='assistant' "
                                    "on shared memory %s (no agent_id), "
                                    "stripping role change",
                                    action.memory_id,
                                )
                                del clean_meta["role"]
                        if clean_meta:
                            self._vector.update_metadata(action.memory_id, clean_meta)
                            executed += 1

            elif action.action == "add" and action.new_content:
                # Derive metadata from source memory when available.
                # For split issues, the source is the memory being deleted.
                source_meta: dict[str, Any] = {}
                source_agent_id: str | None = None
                for other_action in issue.actions:
                    if other_action.action == "delete" and other_action.memory_id:
                        source = self._vector.get_by_id(other_action.memory_id)
                        if source:
                            source_meta = source.get("metadata") or {}
                            source_agent_id = source.get("agent_id")
                        break

                if self._memory_service is not None:
                    meta = action.new_metadata or {}
                    self._memory_service.add_memory(
                        content=action.new_content,
                        user_id=user_id,
                        agent_id=source_agent_id or meta.get("agent_id"),
                        role=source_meta.get("role") or meta.get("role"),
                        memory_type=meta.get("memory_type")
                        or source_meta.get("memory_type"),
                        categories=meta.get("categories")
                        or source_meta.get("categories"),
                        importance=meta.get("importance")
                        or source_meta.get("importance"),
                        pinned=meta.get("pinned", source_meta.get("pinned")),
                        labels=meta.get("labels") or source_meta.get("labels"),
                        event_date=source_meta.get("event_date"),
                        ttl_days=source_meta.get("ttl_days"),
                        infer=False,
                    )
                else:
                    # Fallback: direct vector store insert when MemoryService
                    # is not available (e.g., in tests).
                    embed = EmbeddingClient(self._config.embed)
                    vector = embed.embed(action.new_content)

                    now = datetime.now(timezone.utc)
                    payload: dict[str, Any] = {
                        "data": action.new_content,
                        "hash": hashlib.sha256(action.new_content.encode()).hexdigest(),
                        "user_id": user_id,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                    if action.new_metadata:
                        for k, v in action.new_metadata.items():
                            if k in (
                                "memory_type",
                                "categories",
                                "importance",
                                "pinned",
                            ):
                                payload[k] = v

                    payload.setdefault("memory_type", "fact")
                    payload.setdefault("importance", "normal")
                    payload.setdefault("pinned", False)

                    point_id = str(uuid.uuid4())
                    self._vector._client.upsert(
                        collection_name=self._vector.collection_name,
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=vector,
                                payload=payload,
                            )
                        ],
                    )
                executed += 1

        return executed, skipped

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(issues: list[FsckIssue]) -> FsckSummary:
        """Build a summary from a list of issues."""
        summary = FsckSummary()
        for issue in issues:
            if issue.type == "duplicate":
                summary.duplicate += 1
            elif issue.type == "quality":
                summary.quality += 1
            elif issue.type == "split":
                summary.split += 1
            elif issue.type == "contradiction":
                summary.contradiction += 1
            elif issue.type == "reclassify":
                summary.reclassify += 1
            elif issue.type == "security":
                summary.security += 1
            summary.total += 1
        return summary
