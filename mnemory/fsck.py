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

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mnemory.config import Config
from mnemory.llm import LLMClient, parse_json_response
from mnemory.prompts import (
    build_fsck_duplicate_prompt,
    build_fsck_quality_prompt,
)
from mnemory.sanitize import detect_injection_patterns
from mnemory.storage.vector import VectorStore
from mnemory.ttl import is_expired

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Similarity threshold for duplicate detection (higher than dedup_similarity
# during ingestion because we want to flag clear duplicates, not just related).
_DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Maximum memories per LLM quality-check batch.
_QUALITY_BATCH_SIZE = 20

# Maximum memories per duplicate cluster sent to LLM.
_MAX_CLUSTER_SIZE = 15

# Maximum memories to process in a single fsck run.
_MAX_MEMORIES = 2000

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


@dataclass
class FsckIssue:
    """A single issue found during memory check."""

    issue_id: str
    type: str  # duplicate, quality, split, contradiction, reclassify, security
    severity: str  # low, medium, high
    reasoning: str
    affected_memories: list[FsckAffectedMemory]
    actions: list[FsckAction]


@dataclass
class FsckProgress:
    """Progress of a running memory check."""

    phase: str = "starting"
    total_memories: int = 0
    processed: int = 0
    percent: int = 0
    issues_found: int = 0


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
    status: str = "running"  # running, completed, failed
    progress: FsckProgress = field(default_factory=FsckProgress)
    issues: list[FsckIssue] = field(default_factory=list)
    summary: FsckSummary | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_seconds: int = 1800

    @property
    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at > self.ttl_seconds

    @property
    def expires_at_utc(self) -> str:
        """Compute expiration time as ISO 8601 UTC string."""
        created = datetime.fromisoformat(self.created_at_utc)
        from datetime import timedelta

        expires = created + timedelta(seconds=self.ttl_seconds)
        return expires.isoformat()


# ── FsckStore — in-memory state for check runs ──────────────────────


class FsckStore:
    """Thread-safe in-memory store for fsck check results.

    Similar to SessionStore but for fsck check runs. Checks are stored
    with a TTL and cleaned up lazily on access.
    """

    def __init__(self, default_ttl: int = 1800):
        self._checks: dict[str, FsckCheck] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl

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
    ):
        self._config = config
        self._vector = vector
        self._llm = llm
        self._store = store

    # ── Public API ───────────────────────────────────────────────────

    def start_check(
        self,
        user_id: str,
        agent_id: str | None = None,
        categories: list[str] | None = None,
        memory_type: str | None = None,
    ) -> FsckCheck:
        """Create a new check and return it (status=running).

        The caller is responsible for running run_check() in a background
        task after this returns.
        """
        check = self._store.create(user_id, agent_id)
        return check

    def run_check(
        self,
        check_id: str,
        *,
        categories: list[str] | None = None,
        memory_type: str | None = None,
    ) -> None:
        """Execute the full check pipeline. Called in a background task.

        Updates the FsckCheck in-place with progress, issues, and status.

        Phase weight allocation for percent:
          Phase 0 (security_scan):    0 –  5%  (instant, regex only)
          Phase 1a (duplicate_search): 5 – 30%  (vector search per memory)
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

            memories = self._vector.scroll_with_vectors(
                user_id=check.user_id,
                agent_id=check.agent_id,
                filters=filters,
                limit=_MAX_MEMORIES,
            )

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

            # Filter out expired/decayed memories
            memories = [
                m
                for m in memories
                if not is_expired(m) or (m.get("metadata") or {}).get("pinned")
            ]

            total = len(memories)
            check.progress.total_memories = total
            logger.info("Fsck check %s: %d memories to check", check_id, total)

            if total == 0:
                check.status = "completed"
                check.summary = FsckSummary()
                check.progress.phase = "done"
                check.progress.percent = 100
                return

            # Build lookup for quick access
            mem_by_id: dict[str, dict] = {m["id"]: m for m in memories}

            # Phase 0: Security scan (regex, no LLM) — 0-5%
            check.progress.phase = "security_scan"
            check.progress.processed = 0
            logger.info(
                "Fsck check %s phase 0: scanning %d memories for injection patterns",
                check_id,
                total,
            )
            security_issues = self._phase_security_scan(memories)
            check.issues.extend(security_issues)
            check.progress.processed = total
            check.progress.percent = 5
            check.progress.issues_found = len(check.issues)

            # Phase 1: Duplicate detection (vector similarity + LLM) — 5-55%
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

            # Phase 2: Quality check (LLM batches) — 55-100%
            logger.info(
                "Fsck check %s phase 2: quality check on %d memories",
                check_id,
                total,
            )
            quality_issues = self._phase_quality_check(memories, check)
            check.issues.extend(quality_issues)
            check.progress.issues_found = len(check.issues)
            check.progress.percent = 100

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
        if check.status != "completed":
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
            try:
                actions_executed = self._apply_issue(issue, check.user_id)
                applied += 1
                details.append(
                    {
                        "issue_id": issue.issue_id,
                        "status": "applied",
                        "actions_executed": actions_executed,
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
                        "error": str(e),
                    }
                )

        return {
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    # ── Phase 0: Security scan ───────────────────────────────────────

    def _phase_security_scan(
        self,
        memories: list[dict],
    ) -> list[FsckIssue]:
        """Scan all memories for prompt injection patterns using regex.

        Fast, no LLM cost. Uses the existing detect_injection_patterns()
        from sanitize.py.
        """
        issues: list[FsckIssue] = []

        for mem in memories:
            text = mem.get("memory", "")
            patterns = detect_injection_patterns(text)
            if not patterns:
                continue

            issue = FsckIssue(
                issue_id=str(uuid.uuid4()),
                type="security",
                severity="high",
                reasoning=(
                    f"Prompt injection patterns detected: {', '.join(patterns)}. "
                    "This memory contains content that appears to be an attempt "
                    "to manipulate AI behavior rather than a genuine memory."
                ),
                affected_memories=[
                    FsckAffectedMemory(
                        id=mem["id"],
                        content=text,
                        metadata=mem.get("metadata"),
                    )
                ],
                actions=[
                    FsckAction(action="delete", memory_id=mem["id"]),
                ],
            )
            issues.append(issue)

        if issues:
            logger.info(
                "Fsck security scan: found %d memories with injection patterns",
                len(issues),
            )

        return issues

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
                # Search for similar memories using stored vector
                similar = self._vector.search_by_vector(
                    vector,
                    user_id=check.user_id,
                    agent_id=check.agent_id,
                    limit=_DUPLICATE_NEIGHBORS,
                    exclude_ids=[mem["id"]],
                )

                for sim in similar:
                    score = sim.get("score", 0)
                    if score >= _DUPLICATE_SIMILARITY_THRESHOLD:
                        uf.union(mem["id"], sim["id"])

            check.progress.processed = idx + 1
            # 5-30% range
            check.progress.percent = 5 + int(25 * (idx + 1) / total) if total else 30

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
        messages, schema = build_fsck_duplicate_prompt(
            cluster,
            max_memory_length=self._config.memory.max_memory_length,
        )

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            max_tokens=4000,
        )

        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in cluster}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            affected_ids = raw_issue.get("affected_memory_ids", [])
            affected_mems = []
            for mid in affected_ids:
                m = mem_lookup.get(mid)
                if m:
                    affected_mems.append(
                        FsckAffectedMemory(
                            id=mid,
                            content=m.get("memory", ""),
                            metadata=m.get("metadata"),
                        )
                    )

            actions = []
            for raw_action in raw_issue.get("actions", []):
                actions.append(
                    FsckAction(
                        action=raw_action.get("action", ""),
                        memory_id=raw_action.get("memory_id"),
                        new_content=raw_action.get("new_content"),
                        new_metadata=raw_action.get("new_metadata"),
                    )
                )

            issue_type = raw_issue.get("type", "duplicate")
            if issue_type not in ("duplicate", "contradiction"):
                issue_type = "duplicate"

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=raw_issue.get("severity", "medium"),
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                )
            )

        return issues

    # ── Phase 2: Quality check ───────────────────────────────────────

    def _phase_quality_check(
        self,
        memories: list[dict],
        check: FsckCheck,
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
            return self._evaluate_quality_batch(batch)

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
    ) -> list[FsckIssue]:
        """Send a batch of memories to LLM for quality evaluation."""
        messages, schema = build_fsck_quality_prompt(batch)

        response = self._llm.generate(
            messages,
            json_schema=schema,
            temperature=0.1,
            max_tokens=4000,
        )

        parsed = parse_json_response(response)
        if not parsed or "issues" not in parsed:
            return []

        # Build lookup for affected memory content
        mem_lookup = {m["id"]: m for m in batch}

        issues: list[FsckIssue] = []
        for raw_issue in parsed["issues"]:
            affected_ids = raw_issue.get("affected_memory_ids", [])
            affected_mems = []
            for mid in affected_ids:
                m = mem_lookup.get(mid)
                if m:
                    affected_mems.append(
                        FsckAffectedMemory(
                            id=mid,
                            content=m.get("memory", ""),
                            metadata=m.get("metadata"),
                        )
                    )

            actions = []
            for raw_action in raw_issue.get("actions", []):
                actions.append(
                    FsckAction(
                        action=raw_action.get("action", ""),
                        memory_id=raw_action.get("memory_id"),
                        new_content=raw_action.get("new_content"),
                        new_metadata=raw_action.get("new_metadata"),
                    )
                )

            issue_type = raw_issue.get("type", "quality")
            if issue_type not in (
                "quality",
                "split",
                "reclassify",
                "security",
            ):
                issue_type = "quality"

            severity = raw_issue.get("severity", "medium")
            if severity not in ("low", "medium", "high"):
                severity = "medium"

            issues.append(
                FsckIssue(
                    issue_id=str(uuid.uuid4()),
                    type=issue_type,
                    severity=severity,
                    reasoning=raw_issue.get("reasoning", ""),
                    affected_memories=affected_mems,
                    actions=actions,
                )
            )

        return issues

    # ── Apply logic ──────────────────────────────────────────────────

    def _apply_issue(self, issue: FsckIssue, user_id: str) -> int:
        """Apply all actions for a single issue. Returns count of actions executed."""
        executed = 0

        for action in issue.actions:
            if action.action == "delete" and action.memory_id:
                self._vector.delete(action.memory_id)
                executed += 1

            elif action.action == "update" and action.memory_id:
                if action.new_content:
                    self._vector.update_content(action.memory_id, action.new_content)
                    executed += 1
                if action.new_metadata:
                    # Filter to allowed metadata fields
                    allowed = {
                        "memory_type",
                        "categories",
                        "importance",
                        "pinned",
                    }
                    clean_meta = {
                        k: v for k, v in action.new_metadata.items() if k in allowed
                    }
                    if clean_meta:
                        self._vector.update_metadata(action.memory_id, clean_meta)
                        executed += 1

            elif action.action == "add" and action.new_content:
                # For split actions: add new memory with infer=False
                # We need to go through the memory service for proper
                # embedding and metadata handling
                from mnemory.embeddings import EmbeddingClient

                embed = EmbeddingClient(self._config.embed)
                vector = embed.embed(action.new_content)

                # Build payload
                import hashlib

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

                # Set defaults if not provided
                payload.setdefault("memory_type", "fact")
                payload.setdefault("importance", "normal")
                payload.setdefault("pinned", False)

                from qdrant_client.models import PointStruct

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

        return executed

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
