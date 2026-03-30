"""Tests for mnemory.config module."""

import pytest

from mnemory.config import load_config


class TestConfigValidation:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            load_config()

    def test_invalid_artifact_backend_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("ARTIFACT_BACKEND", "invalid")
        with pytest.raises(ValueError, match="Unsupported ARTIFACT_BACKEND"):
            load_config()

    def test_s3_missing_credentials_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("ARTIFACT_BACKEND", "s3")
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="S3_ACCESS_KEY and S3_SECRET_KEY"):
            load_config()

    def test_valid_minimal_config(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = load_config()
        assert config.llm.api_key == "test-key"
        assert config.artifact.backend == "filesystem"

    def test_qdrant_local_mode_by_default(self, monkeypatch):
        """Without QDRANT_HOST, vector config should be local mode."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("QDRANT_HOST", raising=False)
        config = load_config()
        assert config.vector.is_remote is False

    def test_qdrant_remote_mode(self, monkeypatch):
        """With QDRANT_HOST set, vector config should be remote mode."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
        config = load_config()
        assert config.vector.is_remote is True

    def test_embed_config_separate(self, monkeypatch):
        """EmbedConfig should be separate from LLMConfig."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("EMBED_MODEL", "custom-embed-model")
        monkeypatch.setenv("EMBED_DIMS", "768")
        config = load_config()
        assert config.embed.model == "custom-embed-model"
        assert config.embed.dims == 768

    def test_embed_base_url_fallback(self, monkeypatch):
        """EMBED_BASE_URL should fall back to LLM_BASE_URL."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api.com/v1")
        monkeypatch.delenv("EMBED_BASE_URL", raising=False)
        config = load_config()
        assert config.embed.base_url == "https://custom.api.com/v1"

    def test_embed_api_key_explicit(self, monkeypatch):
        """EMBED_API_KEY should be used when set."""
        monkeypatch.setenv("LLM_API_KEY", "llm-key")
        monkeypatch.setenv("EMBED_API_KEY", "embed-key")
        config = load_config()
        assert config.embed.api_key == "embed-key"
        assert config.llm.api_key == "llm-key"

    def test_embed_api_key_falls_back_to_llm_key(self, monkeypatch):
        """EMBED_API_KEY should fall back to LLM_API_KEY."""
        monkeypatch.setenv("LLM_API_KEY", "llm-key")
        monkeypatch.delenv("EMBED_API_KEY", raising=False)
        config = load_config()
        assert config.embed.api_key == "llm-key"

    def test_embed_api_key_falls_back_to_openai_key(self, monkeypatch):
        """EMBED_API_KEY should fall back to OPENAI_API_KEY."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("EMBED_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        config = load_config()
        assert config.embed.api_key == "openai-key"

    def test_qdrant_api_key_optional(self, monkeypatch):
        """Qdrant API key should be optional."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        config = load_config()
        assert config.vector.qdrant_api_key == ""

    def test_invalid_instruction_mode_raises(self, monkeypatch):
        """Invalid INSTRUCTION_MODE should raise ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("INSTRUCTION_MODE", "invalid")
        with pytest.raises(ValueError, match="Unsupported INSTRUCTION_MODE"):
            load_config()

    # ── Session backend config tests ─────────────────────────────────

    def test_session_backend_defaults_to_sqlite(self, monkeypatch):
        """Without SESSION_BACKEND or REDIS_URL, should default to sqlite."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("SESSION_BACKEND", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        config = load_config()
        assert config.memory.session_backend == "sqlite"

    def test_session_backend_auto_detects_redis(self, monkeypatch):
        """With REDIS_URL set and no SESSION_BACKEND, should default to redis."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("SESSION_BACKEND", raising=False)
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        config = load_config()
        assert config.memory.session_backend == "redis"

    def test_session_backend_explicit_memory(self, monkeypatch):
        """Explicit SESSION_BACKEND=memory should be accepted."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("SESSION_BACKEND", "memory")
        config = load_config()
        assert config.memory.session_backend == "memory"

    def test_session_backend_invalid_raises(self, monkeypatch):
        """Invalid SESSION_BACKEND should raise ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("SESSION_BACKEND", "invalid")
        with pytest.raises(ValueError, match="Unsupported SESSION_BACKEND"):
            load_config()

    def test_session_backend_redis_without_url_raises(self, monkeypatch):
        """SESSION_BACKEND=redis without REDIS_URL should raise ValueError."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("SESSION_BACKEND", "redis")
        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(ValueError, match="REDIS_URL is required"):
            load_config()

    def test_session_ttl_default_24h(self, monkeypatch):
        """Default session TTL should be 86400 (24 hours)."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("MEMORY_SESSION_TTL", raising=False)
        config = load_config()
        assert config.memory.memory_session_ttl == 86400

    def test_session_ttl_custom(self, monkeypatch):
        """Custom MEMORY_SESSION_TTL should be respected."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("MEMORY_SESSION_TTL", "7200")
        config = load_config()
        assert config.memory.memory_session_ttl == 7200

    def test_session_path_default(self, monkeypatch):
        """Default session path should be in data dir."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("SESSION_PATH", raising=False)
        config = load_config()
        assert config.memory.session_path.endswith("sessions.db")

    def test_session_path_custom(self, monkeypatch):
        """Custom SESSION_PATH should be respected."""
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("SESSION_PATH", "/tmp/custom_sessions.db")
        config = load_config()
        assert config.memory.session_path == "/tmp/custom_sessions.db"

    def test_jwt_verifier_sources_are_mutually_exclusive(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        public_key = tmp_path / "public.pem"
        public_key.write_text("test", encoding="utf-8")
        monkeypatch.setenv("MNEMORY_JWT_PUBLIC_KEY", str(public_key))
        monkeypatch.setenv("MNEMORY_JWKS_URL", "https://example.com/jwks.json")
        with pytest.raises(ValueError, match="Configure only one JWT verifier source"):
            load_config()

    def test_missing_jwt_public_key_path_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("MNEMORY_JWT_PUBLIC_KEY", "/tmp/does-not-exist.pem")
        monkeypatch.delenv("MNEMORY_JWKS_URL", raising=False)
        with pytest.raises(ValueError, match="does not exist"):
            load_config()
