"""Configuration management for mnemory.

All settings are loaded from environment variables with sensible defaults.
Defaults are optimized for local development — just set OPENAI_API_KEY and
run. Override for production (Qdrant, S3, API keys).

Data is stored in ~/.mnemory by default (override with DATA_DIR env var).
In Docker, DATA_DIR is set to /data for volume mounting.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("mnemory")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


def _env_int_or_none(key: str, default: int | None = None) -> int | None:
    """Parse an env var as int, treating empty string and 'null'/'none' as None."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    if raw.lower() in ("null", "none"):
        return None
    return int(raw)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _data_dir() -> str:
    """Resolve base data directory.

    Priority: DATA_DIR env var > ~/.mnemory
    In Docker, the Dockerfile sets DATA_DIR=/data for volume mounting.
    Locally, defaults to ~/.mnemory for clean, predictable storage.
    """
    raw = os.environ.get("DATA_DIR", "")
    if raw:
        return raw
    return os.path.join(os.path.expanduser("~"), ".mnemory")


def _llm_api_key() -> str:
    """Resolve LLM API key with fallback chain.

    Priority: LLM_API_KEY > OPENAI_API_KEY
    """
    return _env("LLM_API_KEY") or _env("OPENAI_API_KEY")


def _llm_base_url() -> str:
    """Resolve LLM base URL with fallback chain.

    Priority: LLM_BASE_URL > OPENAI_API_BASE > default OpenAI URL
    """
    return (
        _env("LLM_BASE_URL") or _env("OPENAI_API_BASE") or "https://api.openai.com/v1"
    )


@dataclass
class VectorConfig:
    """Vector store configuration (Qdrant only).

    Mode detection:
    - If QDRANT_HOST is set → remote mode (connects to Qdrant server)
    - If QDRANT_HOST is not set → local mode (embedded Qdrant, data in QDRANT_PATH)
    """

    qdrant_host: str = field(default_factory=lambda: _env("QDRANT_HOST"))
    qdrant_port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection_name: str = field(
        default_factory=lambda: _env("QDRANT_COLLECTION", "mnemory")
    )

    # Local mode settings (used when QDRANT_HOST is not set)
    qdrant_path: str = field(
        default_factory=lambda: (
            _env("QDRANT_PATH") or os.path.join(_data_dir(), "qdrant")
        )
    )

    @property
    def is_remote(self) -> bool:
        """True if connecting to a remote Qdrant server."""
        return bool(self.qdrant_host)


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-5-mini"))
    base_url: str = field(default_factory=_llm_base_url)
    api_key: str = field(default_factory=_llm_api_key)
    temperature: float = 0.1
    reasoning_effort: str | None = field(
        default_factory=lambda: _env("LLM_REASONING_EFFORT") or None
    )


@dataclass
class EmbedConfig:
    """Embedding model configuration."""

    model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small")
    )
    base_url: str = field(
        default_factory=lambda: _env("EMBED_BASE_URL") or _llm_base_url()
    )
    api_key: str = field(default_factory=_llm_api_key)
    dims: int = field(default_factory=lambda: _env_int("EMBED_DIMS", 1536))


@dataclass
class ArtifactConfig:
    """Artifact store configuration."""

    backend: str = field(default_factory=lambda: _env("ARTIFACT_BACKEND", "filesystem"))

    # S3 / MinIO settings
    s3_endpoint: str = field(
        default_factory=lambda: _env("S3_ENDPOINT", "http://localhost:9000")
    )
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "mnemory"))
    s3_region: str = field(default_factory=lambda: _env("S3_REGION"))

    # Filesystem settings (default for local development)
    filesystem_path: str = field(
        default_factory=lambda: (
            _env("ARTIFACT_PATH") or os.path.join(_data_dir(), "artifacts")
        )
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
    enable_delete_all: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DELETE_ALL", False)
    )
    instruction_mode: str = field(
        default_factory=lambda: _env("INSTRUCTION_MODE", "proactive")
    )


@dataclass
class MemoryConfig:
    """Memory behavior configuration."""

    max_memory_length: int = field(
        default_factory=lambda: _env_int("MAX_MEMORY_LENGTH", 1000)
    )
    max_artifact_size: int = field(
        default_factory=lambda: _env_int("MAX_ARTIFACT_SIZE", 102400)
    )
    max_core_context_length: int = field(
        default_factory=lambda: _env_int("MAX_CORE_CONTEXT_LENGTH", 4000)
    )
    default_recent_days: int = field(
        default_factory=lambda: _env_int("DEFAULT_RECENT_DAYS", 7)
    )
    recent_limit_user: int = field(
        default_factory=lambda: _env_int("RECENT_LIMIT_USER", 25)
    )
    recent_limit_agent: int = field(
        default_factory=lambda: _env_int("RECENT_LIMIT_AGENT", 25)
    )
    auto_classify: bool = field(
        default_factory=lambda: _env_bool("AUTO_CLASSIFY", True)
    )
    classify_cache_ttl: int = field(
        default_factory=lambda: _env_int("CLASSIFY_CACHE_TTL", 300)
    )
    core_memories_cache_ttl: int = field(
        default_factory=lambda: _env_int("CORE_MEMORIES_CACHE_TTL", 300)
    )

    # TTL defaults by memory type (days, None = permanent)
    ttl_fact: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_FACT", None)
    )
    ttl_preference: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_PREFERENCE", None)
    )
    ttl_episodic: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_EPISODIC", 90)
    )
    ttl_procedural: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_PROCEDURAL", 60)
    )
    ttl_context: int | None = field(
        default_factory=lambda: _env_int_or_none("TTL_CONTEXT", 7)
    )

    # Access tracking
    track_memory_access: bool = field(
        default_factory=lambda: _env_bool("TRACK_MEMORY_ACCESS", True)
    )

    # Search quality thresholds
    search_score_threshold: float = field(
        default_factory=lambda: _env_float("SEARCH_SCORE_THRESHOLD", 0.30)
    )
    dedup_similarity_threshold: float = field(
        default_factory=lambda: _env_float("DEDUP_SIMILARITY_THRESHOLD", 0.4)
    )

    # Search ranking weights
    # Controls the balance between cosine similarity and importance in the
    # combined search score formula: score = similarity_weight * cosine_sim
    # + (1 - similarity_weight) * importance_weight.
    # Default 0.9 means 90% similarity, 10% importance — importance acts as
    # a tiebreaker rather than a primary ranking factor.
    search_similarity_weight: float = field(
        default_factory=lambda: _env_float("SEARCH_SIMILARITY_WEIGHT", 0.9)
    )

    # Post-retrieval keyword boost weight. After vector search returns
    # results, a keyword overlap score is blended in:
    # final = (1 - keyword_weight) * qdrant_score + keyword_weight * keyword_score
    # Set to 0.0 to disable keyword boosting entirely.
    search_keyword_weight: float = field(
        default_factory=lambda: _env_float("SEARCH_KEYWORD_WEIGHT", 0.2)
    )

    # find_memories: maximum number of search queries the LLM generates from
    # the user's message. The LLM may return fewer (or zero) based on input.
    find_memories_queries: int = field(
        default_factory=lambda: _env_int("FIND_MEMORIES_QUERIES", 5)
    )

    # Default timezone for naive event_date values (no timezone info).
    # IANA timezone name (e.g., "UTC", "Europe/Prague", "America/New_York").
    # Empty string = use server's local timezone.
    default_timezone: str = field(default_factory=lambda: _env("DEFAULT_TIMEZONE", ""))

    # Auto-artifact threshold for remember() endpoint.
    # When the LLM recommends storing an artifact AND content exceeds this
    # length (chars), the original content is saved as an artifact.
    # Set to 0 to disable auto-artifact in remember() entirely.
    remember_artifact_threshold: int = field(
        default_factory=lambda: _env_int("REMEMBER_ARTIFACT_THRESHOLD", 4000)
    )

    # Input length cap (chars) for infer=True and remember().
    # ~100k tokens for gpt-5-mini, leaving 50% headroom for output/reasoning.
    max_input_length: int = field(
        default_factory=lambda: _env_int("MAX_INPUT_LENGTH", 400000)
    )

    # Session settings
    memory_session_ttl: int = field(
        default_factory=lambda: _env_int("MEMORY_SESSION_TTL", 3600)
    )
    memory_session_sweep_interval: int = field(
        default_factory=lambda: _env_int("MEMORY_SESSION_SWEEP_INTERVAL", 300)
    )

    # Recall settings
    recall_max_results: int = field(
        default_factory=lambda: _env_int("RECALL_MAX_RESULTS", 10)
    )

    # Remember settings — rate limit per user per minute (0 = no limit)
    remember_rate_limit: int = field(
        default_factory=lambda: _env_int("REMEMBER_RATE_LIMIT", 10)
    )


@dataclass
class Config:
    """Root configuration container."""

    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        """Validate that required configuration is present."""
        if not self.llm.api_key:
            raise ValueError("API key is required. Set LLM_API_KEY or OPENAI_API_KEY.")
        if self.artifact.backend not in ("s3", "filesystem"):
            raise ValueError(f"Unsupported ARTIFACT_BACKEND: {self.artifact.backend}")
        if self.server.instruction_mode not in (
            "passive",
            "proactive",
            "personality",
        ):
            raise ValueError(
                f"Unsupported INSTRUCTION_MODE: "
                f"{self.server.instruction_mode}. "
                "Must be one of: passive, proactive, personality"
            )
        if self.artifact.backend == "s3":
            if not self.artifact.s3_access_key or not self.artifact.s3_secret_key:
                raise ValueError(
                    "S3_ACCESS_KEY and S3_SECRET_KEY are required "
                    "when ARTIFACT_BACKEND=s3"
                )


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Creates the data directory (~/.mnemory by default) if it doesn't exist.
    """
    config = Config()
    config.validate()

    # Ensure data directory exists
    data_dir = _data_dir()
    os.makedirs(data_dir, exist_ok=True)
    logger.debug("Data directory: %s", data_dir)

    return config
