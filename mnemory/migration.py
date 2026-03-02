"""Data migration framework for mnemory.

Migrations run automatically on startup before the server accepts
requests. State is tracked in a separate ``_mnemory_meta`` Qdrant
collection (not the filesystem) to support stateless Kubernetes
deployments where no persistent local storage is available.

Each migration must be **idempotent** — safe to re-run if interrupted
mid-way or if multiple instances start simultaneously. Progress
checkpoints are saved after each batch for resumable migrations.

Usage::

    runner = MigrationRunner(client, get_migrations(collection, sparse))
    runner.run_pending()
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Qdrant collection for migration state. Separate from the main memory
# collection to avoid polluting search results with metadata points.
META_COLLECTION = "_mnemory_meta"

# Well-known point ID for the migration state document.
META_POINT_ID = "00000000-0000-0000-0000-000000000000"


class Migration(ABC):
    """Base class for data migrations.

    Subclasses must define ``id`` (unique, e.g. ``001_add_sparse_vectors``)
    and ``description`` (human-readable), and implement ``run()``.

    Migrations receive the Qdrant client, optional progress checkpoint
    (for resuming), and a callback to persist state between batches.
    """

    id: str
    description: str

    @abstractmethod
    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        """Execute the migration.

        Args:
            client: Qdrant client for all operations.
            progress: Checkpoint from a previous interrupted run, or None.
            state_callback: Callable(state) to persist intermediate state.
            state: The full migration state dict (for checkpoint writes).
        """
        ...


class MigrationRunner:
    """Runs pending data migrations on startup.

    State is stored in a ``_mnemory_meta`` Qdrant collection as a single
    point with a well-known UUID. The payload tracks which migrations
    have been applied and optional per-migration progress checkpoints.
    """

    def __init__(
        self,
        client: QdrantClient,
        migrations: list[Migration],
    ):
        self._client = client
        self._migrations = migrations

    def _ensure_meta_collection(self) -> None:
        """Create the ``_mnemory_meta`` collection if it doesn't exist.

        Uses a 1-dimensional dummy vector config — the collection only
        stores migration metadata in payloads, not real vectors.
        """
        existing = [c.name for c in self._client.get_collections().collections]
        if META_COLLECTION not in existing:
            logger.debug("Creating migration metadata collection '%s'", META_COLLECTION)
            self._client.create_collection(
                collection_name=META_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def _load_state(self) -> dict[str, Any]:
        """Load migration state from ``_mnemory_meta``."""
        try:
            result = self._client.retrieve(
                collection_name=META_COLLECTION,
                ids=[META_POINT_ID],
                with_payload=True,
            )
            if result:
                return dict(result[0].payload or {})
        except Exception:
            logger.debug("No existing migration state found")
        return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        """Persist migration state to ``_mnemory_meta``."""
        self._client.upsert(
            collection_name=META_COLLECTION,
            points=[
                PointStruct(
                    id=META_POINT_ID,
                    vector=[0.0],  # dummy vector
                    payload=state,
                )
            ],
        )

    def run_pending(self) -> None:
        """Run all pending migrations in order.

        For each migration that hasn't been applied yet:
        1. Mark it as "running" (optimistic lock for multi-pod safety)
        2. Execute the migration
        3. Mark it as complete

        If a migration was previously interrupted (``running`` marker
        present), it is re-run since all migrations are idempotent.
        """
        self._ensure_meta_collection()
        state = self._load_state()
        applied = list(state.get("applied", []))

        for migration in self._migrations:
            if migration.id in applied:
                logger.debug("Migration %s already applied, skipping", migration.id)
                continue

            # Check for interrupted previous run
            running_key = f"{migration.id}:running"
            if running_key in applied:
                logger.warning(
                    "Migration %s was previously interrupted or is running "
                    "on another instance — re-running (idempotent)",
                    migration.id,
                )

            # Optimistic lock: mark as running before starting
            if running_key not in applied:
                applied.append(running_key)
            state["applied"] = applied
            self._save_state(state)

            logger.info(
                "Running migration: %s — %s",
                migration.id,
                migration.description,
            )

            try:
                progress = state.get(f"{migration.id}_progress")
                migration.run(
                    self._client,
                    progress=progress,
                    state_callback=self._save_state,
                    state=state,
                )

                # Mark complete: replace running marker with final ID
                applied = [m for m in state.get("applied", []) if m != running_key]
                applied.append(migration.id)
                state["applied"] = applied
                state.pop(f"{migration.id}_progress", None)
                state["last_run"] = datetime.now(timezone.utc).isoformat()
                self._save_state(state)
                logger.info("Migration %s completed successfully", migration.id)

            except Exception:
                logger.exception("Migration %s failed", migration.id)
                # Remove running marker on failure so it can be retried
                applied = [m for m in state.get("applied", []) if m != running_key]
                state["applied"] = applied
                self._save_state(state)
                raise


# ── Migration 001: Add BM25 sparse vectors ──────────────────────────


class AddSparseVectorsMigration(Migration):
    """Add BM25 sparse vectors to existing memories for hybrid search.

    Steps:
    1. Add sparse vector config to the collection (``update_collection``)
    2. Scroll all existing points in batches
    3. Generate BM25 sparse vectors from each memory's text
    4. Use ``update_vectors()`` to add sparse vectors without touching
       the existing dense vectors

    This migration is idempotent: ``update_collection`` is a no-op if
    the sparse config already exists, and ``update_vectors`` safely
    overwrites sparse vectors on already-migrated points.

    Performance: BM25 via FastEmbed runs at ~1000 texts/sec on CPU.
    10K memories ≈ 10s, 100K ≈ 2-3 minutes.
    """

    id = "001_add_sparse_vectors"
    description = "Add BM25 sparse vectors for hybrid search"

    def __init__(
        self,
        collection_name: str,
        sparse_client: Any,
    ):
        self._collection_name = collection_name
        self._sparse = sparse_client

    def run(
        self,
        client: QdrantClient,
        *,
        progress: dict[str, Any] | None,
        state_callback: Callable[[dict[str, Any]], None],
        state: dict[str, Any],
    ) -> None:
        from qdrant_client.models import (
            Modifier,
            PointVectors,
            SparseVectorParams,
        )

        if self._sparse is None or not self._sparse.available:
            logger.warning(
                "Skipping migration %s: fastembed not installed — "
                "hybrid search will use dense-only mode",
                self.id,
            )
            return

        # Step 1: Add sparse vector config to collection.
        # This is idempotent — Qdrant ignores if the config already exists.
        try:
            client.update_collection(
                collection_name=self._collection_name,
                sparse_vectors_config={
                    "bm25": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            logger.info(
                "Added sparse vector config 'bm25' to collection '%s'",
                self._collection_name,
            )
        except Exception:
            # May already exist (new collection created with sparse config)
            logger.debug(
                "Sparse vector config may already exist on '%s', continuing",
                self._collection_name,
            )

        # Step 2: Count total points for progress estimation
        collection_info = client.get_collection(self._collection_name)
        total_points = collection_info.points_count or 0
        if total_points == 0:
            logger.info(
                "Collection '%s' is empty, nothing to migrate", self._collection_name
            )
            return

        estimated_seconds = max(1, total_points / 1000)
        logger.info(
            "Migrating %d points — generating BM25 sparse vectors "
            "(estimated %.0f seconds)",
            total_points,
            estimated_seconds,
        )

        # Step 3: Resume from checkpoint if available
        resume_offset: str | None = None
        processed = 0
        if progress:
            resume_offset = progress.get("offset")
            processed = progress.get("processed", 0)
            if resume_offset:
                logger.info(
                    "Resuming migration from checkpoint: %d points already processed",
                    processed,
                )

        BATCH_SIZE = 256
        offset = resume_offset
        start_time = time.monotonic()

        while True:
            # Scroll batch — payload only (we need the text), no vectors
            points, next_offset = client.scroll(
                collection_name=self._collection_name,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=["data"],
                with_vectors=False,
            )

            if not points:
                break

            # Generate BM25 sparse vectors from memory text
            texts = [p.payload.get("data", "") if p.payload else "" for p in points]
            sparse_vectors = self._sparse.embed_batch(texts)

            if sparse_vectors:
                # Add sparse vectors to existing points without touching
                # the dense vectors. update_vectors() only modifies the
                # specified named vectors, leaving others intact.
                point_vectors = [
                    PointVectors(
                        id=p.id,
                        vector={"bm25": sv},
                    )
                    for p, sv in zip(points, sparse_vectors)
                ]
                client.update_vectors(
                    collection_name=self._collection_name,
                    points=point_vectors,
                )

            processed += len(points)
            elapsed = time.monotonic() - start_time
            total_batches = max(1, (total_points + BATCH_SIZE - 1) // BATCH_SIZE)
            batch_num = min(total_batches, (processed + BATCH_SIZE - 1) // BATCH_SIZE)

            logger.info(
                "Migration %s: batch %d/%d (%d/%d points, %.1fs elapsed)",
                self.id,
                batch_num,
                total_batches,
                processed,
                total_points,
                elapsed,
            )

            # Save checkpoint for resumable migration
            state[f"{self.id}_progress"] = {
                "offset": str(next_offset) if next_offset else None,
                "processed": processed,
            }
            state_callback(state)

            if next_offset is None:
                break
            offset = next_offset

        elapsed = time.monotonic() - start_time
        logger.info(
            "Migration %s complete: %d points updated in %.1fs",
            self.id,
            processed,
            elapsed,
        )


# ── Migration registry ───────────────────────────────────────────────


def get_migrations(
    collection_name: str,
    sparse_client: Any,
) -> list[Migration]:
    """Return the ordered list of all migrations.

    New migrations should be appended to this list. Never reorder or
    remove existing migrations — they are identified by their ``id``.
    """
    return [
        AddSparseVectorsMigration(collection_name, sparse_client),
    ]
