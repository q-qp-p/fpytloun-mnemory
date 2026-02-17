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
