"""Shared pytest fixtures.

The session-scoped ``memory_service`` fixture creates a real MemoryService
backed by embedded Qdrant and real LLM/embedding API calls.  It is used
exclusively by e2e tests (``@pytest.mark.e2e``) and requires an API key.
"""

from __future__ import annotations

import os

import pytest

from mnemory.config import (
    ArtifactConfig,
    Config,
    EmbedConfig,
    LLMConfig,
    MemoryConfig,
    ServerConfig,
    VectorConfig,
)
from mnemory.memory import MemoryService


def _has_api_key() -> bool:
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def memory_service(tmp_path_factory: pytest.TempPathFactory) -> MemoryService:
    """Create a real MemoryService with embedded Qdrant for e2e tests.

    Skips if no LLM API key is available.  Data is stored in a temporary
    directory that is cleaned up after the test session.
    """
    if not _has_api_key():
        pytest.skip("No LLM API key available for e2e tests")

    data_dir = str(tmp_path_factory.mktemp("mnemory_e2e"))

    config = Config(
        vector=VectorConfig(
            qdrant_host="",
            qdrant_path=os.path.join(data_dir, "qdrant"),
        ),
        llm=LLMConfig(),
        embed=EmbedConfig(),
        artifact=ArtifactConfig(
            backend="filesystem",
            filesystem_path=os.path.join(data_dir, "artifacts"),
        ),
        server=ServerConfig(),
        memory=MemoryConfig(
            auto_classify=True,
            track_memory_access=True,
            search_score_threshold=0.30,
            # Disable TTLs so memories persist for the whole session
            ttl_fact=None,
            ttl_preference=None,
            ttl_episodic=None,
            ttl_procedural=None,
            ttl_context=None,
            # Disable core-memory cache so mutations are visible immediately
            core_memories_cache_ttl=0,
        ),
    )
    config.validate()

    return MemoryService(config)
