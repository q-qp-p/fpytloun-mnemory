"""Configuration management for mnemory.

All settings are loaded from environment variables with sensible defaults.
This allows running locally with minimal config or in Kubernetes with full
infrastructure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class VectorConfig:
    """Vector store configuration."""

    backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "qdrant"))

    # Qdrant settings
    qdrant_host: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection_name: str = field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "mnemory")
    )

    # Chroma settings (local alternative)
    chroma_path: str = field(
        default_factory=lambda: _env("CHROMA_PATH", "/data/chroma")
    )


@dataclass
class LLMConfig:
    """LLM and embedding configuration."""

    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-5-mini"))
    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    temperature: float = 0.1

    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small")
    )
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL"))
    embed_dims: int = field(default_factory=lambda: _env_int("EMBED_DIMS", 1536))


@dataclass
class ArtifactConfig:
    """Artifact store configuration."""

    backend: str = field(default_factory=lambda: _env("ARTIFACT_BACKEND", "s3"))

    # S3 / MinIO settings
    s3_endpoint: str = field(
        default_factory=lambda: _env("S3_ENDPOINT", "http://localhost:9000")
    )
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "mnemory"))
    s3_region: str = field(default_factory=lambda: _env("S3_REGION"))

    # Filesystem settings (local alternative)
    filesystem_path: str = field(
        default_factory=lambda: _env("ARTIFACT_PATH", "/data/artifacts")
    )


def _parse_api_keys(raw: str) -> dict[str, str]:
    """Parse MCP_API_KEYS JSON into a dict mapping key -> user_id.

    Format: {"api-key-1": "username", "api-key-2": "*"}
    A value of "*" means the key authenticates but does not bind to a user_id.
    """
    if not raw:
        return {}
    try:
        keys = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"MCP_API_KEYS is not valid JSON: {e}") from e
    if not isinstance(keys, dict):
        raise ValueError("MCP_API_KEYS must be a JSON object mapping key -> user_id")
    for k, v in keys.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("MCP_API_KEYS keys and values must be strings")
    return keys


@dataclass
class ServerConfig:
    """MCP server configuration."""

    host: str = field(default_factory=lambda: _env("MCP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("MCP_PORT", 8050))
    api_key: str = field(default_factory=lambda: _env("MCP_API_KEY"))
    api_keys: dict[str, str] = field(
        default_factory=lambda: _parse_api_keys(_env("MCP_API_KEYS"))
    )


@dataclass
class MemoryConfig:
    """Memory behavior configuration."""

    history_db_path: str = field(
        default_factory=lambda: _env("HISTORY_DB_PATH", "/data/history.db")
    )
    max_memory_length: int = field(
        default_factory=lambda: _env_int("MAX_MEMORY_LENGTH", 1000)
    )
    max_artifact_size: int = field(
        default_factory=lambda: _env_int("MAX_ARTIFACT_SIZE", 102400)
    )
    max_core_context_length: int = field(
        default_factory=lambda: _env_int("MAX_CORE_CONTEXT_LENGTH", 4000)
    )
    default_recent_hours: int = field(
        default_factory=lambda: _env_int("DEFAULT_RECENT_HOURS", 24)
    )


@dataclass
class Config:
    """Root configuration container."""

    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def build_mem0_config(self) -> dict:
        """Build the mem0 library configuration dict."""
        config: dict = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm.model,
                    "temperature": self.llm.temperature,
                    "openai_base_url": self.llm.base_url,
                    "api_key": self.llm.api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.llm.embed_model,
                    "openai_base_url": self.llm.embed_base_url or self.llm.base_url,
                    "api_key": self.llm.api_key,
                    "embedding_dims": self.llm.embed_dims,
                },
            },
            "history_db_path": self.memory.history_db_path,
        }

        if self.vector.backend == "qdrant":
            vs_config: dict = {
                "host": self.vector.qdrant_host,
                "port": self.vector.qdrant_port,
                "collection_name": self.vector.collection_name,
                "embedding_model_dims": self.llm.embed_dims,
            }
            if self.vector.qdrant_api_key:
                vs_config["api_key"] = self.vector.qdrant_api_key
            config["vector_store"] = {"provider": "qdrant", "config": vs_config}
        elif self.vector.backend == "chroma":
            config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": self.vector.collection_name,
                    "path": self.vector.chroma_path,
                },
            }
        else:
            raise ValueError(
                f"Unsupported vector backend: {self.vector.backend}. "
                "Use 'qdrant' or 'chroma'."
            )

        return config

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.llm.api_key:
            raise ValueError("LLM_API_KEY is required")
        if self.vector.backend not in ("qdrant", "chroma"):
            raise ValueError(f"Unsupported VECTOR_BACKEND: {self.vector.backend}")
        if self.artifact.backend not in ("s3", "filesystem"):
            raise ValueError(f"Unsupported ARTIFACT_BACKEND: {self.artifact.backend}")
        if self.artifact.backend == "s3":
            if not self.artifact.s3_access_key or not self.artifact.s3_secret_key:
                raise ValueError(
                    "S3_ACCESS_KEY and S3_SECRET_KEY are required "
                    "when ARTIFACT_BACKEND=s3"
                )


def load_config() -> Config:
    """Load and validate configuration from environment variables."""
    config = Config()
    config.validate()
    return config
