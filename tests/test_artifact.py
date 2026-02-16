"""Tests for artifact storage — path validation and filesystem."""

import pytest

from mnemory.config import ArtifactConfig
from mnemory.storage.artifact import (
    ArtifactMetadata,
    ArtifactStore,
    FilesystemBackend,
    _validate_path_component,
)

# ── _validate_path_component ──────────────────────────────────────────


class TestValidatePathComponent:
    def test_valid_simple(self):
        _validate_path_component("user123", "user_id")  # should not raise

    def test_valid_with_hyphens_dots(self):
        _validate_path_component("my-file.txt", "filename")

    def test_valid_with_colons(self):
        _validate_path_component("project:myapp", "category")

    def test_valid_with_slashes(self):
        _validate_path_component("project:domecek/k8s", "category")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_path_component("", "user_id")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            _validate_path_component("a" * 257, "user_id")

    def test_dotdot_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _validate_path_component("../etc/passwd", "filename")

    def test_dotdot_in_middle_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _validate_path_component("foo/../bar", "filename")

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file\x00name", "filename")

    def test_space_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file name", "filename")

    def test_backslash_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_path_component("file\\name", "filename")


# ── ArtifactMetadata ──────────────────────────────────────────────────


class TestArtifactMetadata:
    def test_to_dict(self):
        meta = ArtifactMetadata(
            artifact_id="abc123",
            filename="note.md",
            content_type="text/markdown",
            size=1024,
            created_at="2026-01-01T00:00:00+00:00",
        )
        d = meta.to_dict()
        assert d["id"] == "abc123"
        assert d["filename"] == "note.md"
        assert d["content_type"] == "text/markdown"
        assert d["size"] == 1024

    def test_from_dict_roundtrip(self):
        original = ArtifactMetadata(
            artifact_id="abc123",
            filename="note.md",
            content_type="text/markdown",
            size=1024,
            created_at="2026-01-01T00:00:00+00:00",
        )
        restored = ArtifactMetadata.from_dict(original.to_dict())
        assert restored.artifact_id == original.artifact_id
        assert restored.filename == original.filename
        assert restored.size == original.size


# ── FilesystemBackend path traversal ──────────────────────────────────


class TestFilesystemBackendSecurity:
    def test_path_traversal_in_filename(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        with pytest.raises(ValueError):
            backend._path("user1", "mem1", "art1", "../../etc/passwd")

    def test_path_traversal_in_user_id(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        with pytest.raises(ValueError):
            backend._path("../../etc", "mem1", "art1", "file.txt")

    def test_valid_path_works(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        backend = FilesystemBackend(config)

        path = backend._path("user1", "mem1", "art1", "note.md")
        assert str(tmp_path) in str(path)


# ── ArtifactStore ─────────────────────────────────────────────────────


class TestArtifactStore:
    def test_max_size_enforced(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=100)

        with pytest.raises(ValueError, match="Artifact too large"):
            store.save(
                user_id="user1",
                memory_id="mem1",
                content="x" * 200,
                filename="big.txt",
                content_type="text/plain",
            )

    def test_save_and_load_text(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            memory_id="mem1",
            content="Hello, world!",
            filename="note.txt",
            content_type="text/plain",
        )

        assert meta.filename == "note.txt"
        assert meta.size == len("Hello, world!".encode("utf-8"))

        result = store.load(
            user_id="user1",
            memory_id="mem1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )
        assert result["content"] == "Hello, world!"
        assert result["is_text"] is True
        assert result["has_more"] is False

    def test_load_with_pagination(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        content = "A" * 100
        meta = store.save(
            user_id="user1",
            memory_id="mem1",
            content=content,
            filename="big.txt",
            content_type="text/plain",
        )

        result = store.load(
            user_id="user1",
            memory_id="mem1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
            offset=0,
            limit=50,
        )
        assert len(result["content"]) == 50
        assert result["has_more"] is True

    def test_delete_artifact(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        meta = store.save(
            user_id="user1",
            memory_id="mem1",
            content="test",
            filename="note.txt",
            content_type="text/plain",
        )

        store.delete(
            user_id="user1",
            memory_id="mem1",
            artifact_id=meta.artifact_id,
            artifacts_meta=[meta.to_dict()],
        )

        with pytest.raises(FileNotFoundError):
            store.load(
                user_id="user1",
                memory_id="mem1",
                artifact_id=meta.artifact_id,
                artifacts_meta=[meta.to_dict()],
            )

    def test_delete_nonexistent_raises(self, tmp_path):
        config = ArtifactConfig()
        config.filesystem_path = str(tmp_path)
        config.backend = "filesystem"
        store = ArtifactStore(config, max_artifact_size=102400)

        with pytest.raises(ValueError, match="not found"):
            store.delete(
                user_id="user1",
                memory_id="mem1",
                artifact_id="nonexistent",
                artifacts_meta=[],
            )

    def test_invalid_backend_raises(self):
        config = ArtifactConfig()
        config.backend = "invalid"
        with pytest.raises(ValueError, match="Unsupported artifact backend"):
            ArtifactStore(config)
