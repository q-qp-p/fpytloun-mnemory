"""Tests for the data migration framework."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mnemory.migration import (
    META_COLLECTION,
    AddSparseVectorsMigration,
    Migration,
    MigrationRunner,
    get_migrations,
)

# ── Helpers ───────────────────────────────────────────────────────────


class DummyMigration(Migration):
    """Simple migration for testing the runner."""

    id = "test_001"
    description = "Test migration"

    def __init__(self):
        self.run_count = 0
        self.last_progress = None

    def run(self, client, *, progress, state_callback, state):
        self.run_count += 1
        self.last_progress = progress


class FailingMigration(Migration):
    """Migration that always raises."""

    id = "test_fail"
    description = "Failing migration"

    def run(self, client, *, progress, state_callback, state):
        raise RuntimeError("Migration failed")


def _mock_client(existing_collections=None, state_payload=None):
    """Create a mock Qdrant client with configurable state."""
    client = MagicMock()

    # Mock get_collections
    collections = []
    if existing_collections:
        for name in existing_collections:
            col = MagicMock()
            col.name = name
            collections.append(col)
    client.get_collections.return_value.collections = collections

    # Mock retrieve for state loading
    if state_payload is not None:
        point = MagicMock()
        point.payload = state_payload
        client.retrieve.return_value = [point]
    else:
        client.retrieve.return_value = []

    return client


# ── MigrationRunner ──────────────────────────────────────────────────


class TestMigrationRunnerMetaCollection:
    """Test _mnemory_meta collection creation."""

    def test_creates_meta_collection_if_missing(self):
        """Should create _mnemory_meta when it doesn't exist."""
        client = _mock_client(existing_collections=[])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_called_once()
        args = client.create_collection.call_args
        assert args[1]["collection_name"] == META_COLLECTION

    def test_skips_creation_if_meta_exists(self):
        """Should not create _mnemory_meta if it already exists."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_not_called()

    def test_skips_creation_with_other_collections(self):
        """Should create meta even when other collections exist."""
        client = _mock_client(existing_collections=["memories"])
        runner = MigrationRunner(client, [])
        runner.run_pending()
        client.create_collection.assert_called_once()


class TestMigrationRunnerStateManagement:
    """Test state loading and saving."""

    def test_loads_empty_state_when_no_point(self):
        """Should return empty state when no meta point exists."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Migration should have run (no applied state)
        assert migration.run_count == 1

    def test_loads_existing_state(self):
        """Should load state from existing meta point."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Migration should be skipped (already applied)
        assert migration.run_count == 0

    def test_saves_state_after_migration(self):
        """Should persist state after successful migration."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()

        # Should have called upsert at least twice:
        # 1. Mark as running
        # 2. Mark as complete
        assert client.upsert.call_count >= 2

        # Last upsert should contain the migration ID in applied
        last_call = client.upsert.call_args
        point = last_call[1]["points"][0]
        assert migration.id in point.payload["applied"]
        assert "last_run" in point.payload


class TestMigrationRunnerExecution:
    """Test migration execution logic."""

    def test_runs_pending_migration(self):
        """Should run a migration that hasn't been applied."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.run_count == 1

    def test_skips_applied_migration(self):
        """Should skip a migration that's already applied."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.run_count == 0

    def test_runs_multiple_in_order(self):
        """Should run multiple migrations in order."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        m1 = DummyMigration()
        m1.id = "test_001"
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        runner.run_pending()
        assert m1.run_count == 1
        assert m2.run_count == 1

    def test_skips_first_runs_second(self):
        """Should skip applied migration but run the next one."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001"]},
        )
        m1 = DummyMigration()
        m1.id = "test_001"
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        runner.run_pending()
        assert m1.run_count == 0
        assert m2.run_count == 1

    def test_reruns_interrupted_migration(self):
        """Should re-run a migration that was interrupted (running marker)."""
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={"applied": ["test_001:running"]},
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        # Should re-run since it was interrupted
        assert migration.run_count == 1

    def test_passes_progress_checkpoint(self):
        """Should pass progress checkpoint to resumed migration."""
        checkpoint = {"offset": "abc-123", "processed": 50}
        client = _mock_client(
            existing_collections=[META_COLLECTION],
            state_payload={
                "applied": ["test_001:running"],
                "test_001_progress": checkpoint,
            },
        )
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.last_progress == checkpoint

    def test_no_progress_for_fresh_migration(self):
        """Fresh migration should receive None progress."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()
        assert migration.last_progress is None


class TestMigrationRunnerFailure:
    """Test failure handling."""

    def test_failure_removes_running_marker(self):
        """Failed migration should remove running marker from state."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = FailingMigration()

        runner = MigrationRunner(client, [migration])
        with pytest.raises(RuntimeError, match="Migration failed"):
            runner.run_pending()

        # Last upsert should NOT contain the running marker
        last_call = client.upsert.call_args
        point = last_call[1]["points"][0]
        assert "test_fail:running" not in point.payload.get("applied", [])
        assert "test_fail" not in point.payload.get("applied", [])

    def test_failure_propagates_exception(self):
        """Migration failure should propagate the exception."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        migration = FailingMigration()
        runner = MigrationRunner(client, [migration])
        with pytest.raises(RuntimeError, match="Migration failed"):
            runner.run_pending()

    def test_failure_stops_subsequent_migrations(self):
        """Failure in one migration should stop subsequent ones."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        m1 = FailingMigration()
        m2 = DummyMigration()
        m2.id = "test_002"
        runner = MigrationRunner(client, [m1, m2])
        with pytest.raises(RuntimeError):
            runner.run_pending()
        assert m2.run_count == 0


class TestMigrationRunnerOptimisticLock:
    """Test optimistic locking for multi-pod safety."""

    def test_running_marker_set_before_execution(self):
        """Should set running marker before executing migration."""
        client = _mock_client(existing_collections=[META_COLLECTION])
        upsert_payloads = []

        def capture_upsert(**kwargs):
            point = kwargs["points"][0]
            upsert_payloads.append(dict(point.payload))

        client.upsert.side_effect = capture_upsert

        migration = DummyMigration()
        runner = MigrationRunner(client, [migration])
        runner.run_pending()

        # First upsert should have the running marker
        assert "test_001:running" in upsert_payloads[0]["applied"]
        # Last upsert should have the final ID (not running)
        assert "test_001" in upsert_payloads[-1]["applied"]
        assert "test_001:running" not in upsert_payloads[-1]["applied"]


# ── AddSparseVectorsMigration ────────────────────────────────────────


class TestAddSparseVectorsMigration:
    """Test the BM25 sparse vector migration."""

    def test_skips_when_sparse_unavailable(self):
        """Should skip gracefully when fastembed is not installed."""
        client = MagicMock()
        sparse = MagicMock()
        sparse.available = False

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should not call update_collection or scroll
        client.update_collection.assert_not_called()
        client.scroll.assert_not_called()

    def test_skips_when_sparse_is_none(self):
        """Should skip gracefully when sparse client is None."""
        client = MagicMock()
        migration = AddSparseVectorsMigration("memories", None)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )
        client.update_collection.assert_not_called()

    def test_adds_sparse_config_to_collection(self):
        """Should call update_collection with sparse vector config."""
        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True
        sparse.embed_batch.return_value = []

        # Empty collection
        collection_info = MagicMock()
        collection_info.points_count = 0
        client.get_collection.return_value = collection_info

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.update_collection.assert_called_once()
        kwargs = client.update_collection.call_args[1]
        assert kwargs["collection_name"] == "memories"
        assert "bm25" in kwargs["sparse_vectors_config"]

    def test_skips_empty_collection(self):
        """Should skip scrolling when collection is empty."""
        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 0
        client.get_collection.return_value = collection_info

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.scroll.assert_not_called()
        sparse.embed_batch.assert_not_called()

    def test_processes_single_batch(self):
        """Should process a single batch of points."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        # Collection with 2 points
        collection_info = MagicMock()
        collection_info.points_count = 2
        client.get_collection.return_value = collection_info

        # Mock scroll: one batch, then done
        point1 = MagicMock()
        point1.id = "uuid-1"
        point1.payload = {"data": "User likes cats"}
        point2 = MagicMock()
        point2.id = "uuid-2"
        point2.payload = {"data": "User works at Acme"}
        client.scroll.return_value = ([point1, point2], None)

        # Mock sparse embeddings
        sv1 = SparseVector(indices=[1, 2], values=[0.5, 0.3])
        sv2 = SparseVector(indices=[3, 4], values=[0.7, 0.1])
        sparse.embed_batch.return_value = [sv1, sv2]

        state = {}
        state_callback = MagicMock()

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        # Should have scrolled once
        client.scroll.assert_called_once()
        # Should have embedded the texts
        sparse.embed_batch.assert_called_once_with(
            ["User likes cats", "User works at Acme"]
        )
        # Should have updated vectors
        client.update_vectors.assert_called_once()
        point_vectors = client.update_vectors.call_args[1]["points"]
        assert len(point_vectors) == 2
        assert point_vectors[0].id == "uuid-1"
        assert point_vectors[1].id == "uuid-2"

    def test_processes_multiple_batches(self):
        """Should process multiple batches with pagination."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 3
        client.get_collection.return_value = collection_info

        # Mock scroll: two batches
        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text 1"}
        p2 = MagicMock()
        p2.id = "uuid-2"
        p2.payload = {"data": "text 2"}
        p3 = MagicMock()
        p3.id = "uuid-3"
        p3.payload = {"data": "text 3"}

        client.scroll.side_effect = [
            ([p1, p2], "next-offset"),
            ([p3], None),
        ]

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv, sv]

        state = {}
        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state=state,
        )

        assert client.scroll.call_count == 2
        assert client.update_vectors.call_count == 2

    def test_saves_checkpoint_after_each_batch(self):
        """Should save progress checkpoint after each batch."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 2
        client.get_collection.return_value = collection_info

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text"}
        client.scroll.side_effect = [
            ([p1], "next-offset"),
            ([], None),
        ]

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv]

        state = {}
        state_callback = MagicMock()

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=state_callback,
            state=state,
        )

        # State callback should have been called with progress
        assert state_callback.call_count >= 1
        progress_key = f"{migration.id}_progress"
        assert progress_key in state
        # After completion, the last checkpoint should have offset=None
        last_progress = state[progress_key]
        assert last_progress["processed"] >= 1

    def test_resumes_from_checkpoint(self):
        """Should resume from checkpoint offset."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 10
        client.get_collection.return_value = collection_info

        p1 = MagicMock()
        p1.id = "uuid-5"
        p1.payload = {"data": "text"}
        client.scroll.return_value = ([p1], None)

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv]

        state = {}
        progress = {"offset": "resume-offset", "processed": 4}

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=progress,
            state_callback=MagicMock(),
            state=state,
        )

        # Should have scrolled starting from the resume offset
        scroll_kwargs = client.scroll.call_args[1]
        assert scroll_kwargs["offset"] == "resume-offset"

    def test_handles_missing_payload_data(self):
        """Should handle points with missing or empty data field."""
        from qdrant_client.models import SparseVector

        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 2
        client.get_collection.return_value = collection_info

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {}  # No "data" field
        p2 = MagicMock()
        p2.id = "uuid-2"
        p2.payload = None  # None payload
        client.scroll.return_value = ([p1, p2], None)

        sv = SparseVector(indices=[1], values=[0.5])
        sparse.embed_batch.return_value = [sv, sv]

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        # Should have embedded empty strings for missing data
        sparse.embed_batch.assert_called_once_with(["", ""])

    def test_skips_update_when_no_sparse_vectors(self):
        """Should skip update_vectors when embed_batch returns None."""
        client = MagicMock()
        sparse = MagicMock()
        sparse.available = True

        collection_info = MagicMock()
        collection_info.points_count = 1
        client.get_collection.return_value = collection_info

        p1 = MagicMock()
        p1.id = "uuid-1"
        p1.payload = {"data": "text"}
        client.scroll.return_value = ([p1], None)

        # embed_batch returns None (shouldn't happen, but defensive)
        sparse.embed_batch.return_value = None

        migration = AddSparseVectorsMigration("memories", sparse)
        migration.run(
            client,
            progress=None,
            state_callback=MagicMock(),
            state={},
        )

        client.update_vectors.assert_not_called()


# ── get_migrations registry ──────────────────────────────────────────


class TestGetMigrations:
    """Test the migration registry."""

    def test_returns_list(self):
        """Should return a list of migrations."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_first_migration_is_sparse_vectors(self):
        """First migration should be AddSparseVectorsMigration."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert isinstance(result[0], AddSparseVectorsMigration)
        assert result[0].id == "001_add_sparse_vectors"

    def test_passes_collection_name(self):
        """Should pass collection name to migrations."""
        sparse = MagicMock()
        result = get_migrations("custom_collection", sparse)
        assert result[0]._collection_name == "custom_collection"

    def test_passes_sparse_client(self):
        """Should pass sparse client to migrations."""
        sparse = MagicMock()
        result = get_migrations("memories", sparse)
        assert result[0]._sparse is sparse
