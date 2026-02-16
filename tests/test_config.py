"""Tests for mnemory.config module."""


import pytest

from mnemory.config import Config, load_config


class TestConfigValidation:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="LLM_API_KEY is required"):
            load_config()

    def test_invalid_vector_backend_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "invalid")
        with pytest.raises(ValueError, match="Unsupported VECTOR_BACKEND"):
            load_config()

    def test_invalid_artifact_backend_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "chroma")
        monkeypatch.setenv("ARTIFACT_BACKEND", "invalid")
        with pytest.raises(ValueError, match="Unsupported ARTIFACT_BACKEND"):
            load_config()

    def test_s3_missing_credentials_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "chroma")
        monkeypatch.setenv("ARTIFACT_BACKEND", "s3")
        monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="S3_ACCESS_KEY and S3_SECRET_KEY"):
            load_config()

    def test_valid_minimal_config(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "chroma")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = load_config()
        assert config.llm.api_key == "test-key"
        assert config.vector.backend == "chroma"
        assert config.artifact.backend == "filesystem"


class TestBuildMem0Config:
    def test_qdrant_config(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = load_config()
        mem0_config = config.build_mem0_config()

        assert mem0_config["vector_store"]["provider"] == "qdrant"
        assert mem0_config["llm"]["config"]["api_key"] == "test-key"
        assert "embedding_dims" in mem0_config["embedder"]["config"]

    def test_chroma_config(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "chroma")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = load_config()
        mem0_config = config.build_mem0_config()

        assert mem0_config["vector_store"]["provider"] == "chroma"

    def test_invalid_backend_in_build(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = Config()
        config.vector.backend = "invalid"
        config.llm.api_key = "test-key"
        with pytest.raises(ValueError, match="Unsupported vector backend"):
            config.build_mem0_config()

    def test_embed_base_url_fallback(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api.com/v1")
        monkeypatch.delenv("EMBED_BASE_URL", raising=False)
        monkeypatch.setenv("VECTOR_BACKEND", "chroma")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        config = load_config()
        mem0_config = config.build_mem0_config()

        # embed_base_url should fall back to llm base_url
        assert (
            mem0_config["embedder"]["config"]["openai_base_url"]
            == "https://custom.api.com/v1"
        )

    def test_qdrant_api_key_optional(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("ARTIFACT_BACKEND", "filesystem")
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        config = load_config()
        mem0_config = config.build_mem0_config()

        assert "api_key" not in mem0_config["vector_store"]["config"]
