"""
Mnemory plugin configuration.

All settings are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(key: str, default: bool) -> bool:
    """Parse a boolean environment variable (true/false/1/0/yes/no)."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class MnemoryConfig:
    """Mnemory plugin configuration parsed from environment variables."""

    # Connection
    url: str
    """Mnemory server URL (e.g. ``http://localhost:8050``)."""

    api_key: str
    """Bearer token for mnemory authentication."""

    user_id: str
    """User ID sent as ``X-User-Id`` header."""

    agent_prefix: str
    """Value sent as ``X-Agent-Id`` header. Default: ``hermes``."""

    # Behaviour
    auto_recall: bool
    """Automatically inject relevant memories into context."""

    auto_capture: bool
    """Automatically extract and store memories from conversations."""

    recall_find_first: bool
    """Use AI-powered ``find`` search on the first turn of each session."""

    recall_search_mode: str
    """Search mode for subsequent turns: ``find`` or ``search``."""

    score_threshold: float
    """Minimum relevance score for recalled memories (0.0-1.0)."""

    include_assistant: bool
    """Send assistant messages to mnemory for extraction."""

    managed: bool
    """Include mnemory behavioural instructions in the system prompt."""

    timeout: float
    """HTTP request timeout in seconds."""


def load_config() -> MnemoryConfig:
    """Load and validate configuration from environment variables.

    Raises:
        ValueError: If required variables are missing or values are invalid.
    """
    url = os.environ.get("MNEMORY_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("MNEMORY_URL environment variable is required")

    api_key = os.environ.get("MNEMORY_API_KEY", "").strip()
    user_id = os.environ.get("MNEMORY_USER_ID", "").strip()
    agent_prefix = os.environ.get("MNEMORY_AGENT_PREFIX", "hermes").strip()

    auto_recall = _bool_env("MNEMORY_AUTO_RECALL", True)
    auto_capture = _bool_env("MNEMORY_AUTO_CAPTURE", True)
    recall_find_first = _bool_env("MNEMORY_RECALL_FIND_FIRST", True)

    recall_search_mode = (
        os.environ.get("MNEMORY_RECALL_SEARCH_MODE", "search").strip().lower()
    )
    if recall_search_mode not in ("find", "search"):
        raise ValueError("MNEMORY_RECALL_SEARCH_MODE must be 'find' or 'search'")

    score_threshold_raw = os.environ.get("MNEMORY_SCORE_THRESHOLD", "0.5").strip()
    try:
        score_threshold = float(score_threshold_raw)
    except ValueError:
        raise ValueError(
            f"MNEMORY_SCORE_THRESHOLD must be a number, got '{score_threshold_raw}'"
        )
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("MNEMORY_SCORE_THRESHOLD must be between 0.0 and 1.0")

    include_assistant = _bool_env("MNEMORY_INCLUDE_ASSISTANT", True)
    managed = _bool_env("MNEMORY_MANAGED", True)

    timeout_raw = os.environ.get("MNEMORY_TIMEOUT", "60").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        raise ValueError(f"MNEMORY_TIMEOUT must be a number, got '{timeout_raw}'")
    if timeout < 1:
        raise ValueError("MNEMORY_TIMEOUT must be at least 1 (second)")

    return MnemoryConfig(
        url=url,
        api_key=api_key,
        user_id=user_id,
        agent_prefix=agent_prefix,
        auto_recall=auto_recall,
        auto_capture=auto_capture,
        recall_find_first=recall_find_first,
        recall_search_mode=recall_search_mode,
        score_threshold=score_threshold,
        include_assistant=include_assistant,
        managed=managed,
        timeout=timeout,
    )
