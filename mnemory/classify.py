"""Auto-classification of memory metadata using an LLM.

When the caller (LLM or API client) doesn't provide memory_type, categories,
importance, or pinned, this module classifies them with a single LLM call.

Uses the same LLM configured for mem0 (via LLM_MODEL, LLM_BASE_URL, LLM_API_KEY).
Categories are enriched with existing user categories (cached with TTL) so the
LLM can pick relevant project:<name> subcategories.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from mnemory.categories import (
    IMPORTANCE_WEIGHTS,
    PREDEFINED_CATEGORIES,
    VALID_MEMORY_TYPES,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.config import LLMConfig

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Raised when auto-classification fails after retry."""

    pass


# Defaults used when classification fails or is disabled.
# NOTE: "categories" uses a tuple to prevent accidental mutation of the
# shared default. _defaults_for() converts it to a fresh list for callers.
DEFAULTS = {
    "memory_type": "fact",
    "categories": (),
    "importance": "normal",
    "pinned": False,
}


def _defaults_for(fields: set[str]) -> dict[str, Any]:
    """Return default values for the given fields (with fresh list copies)."""
    result = {}
    for f in fields:
        val = DEFAULTS[f]
        result[f] = list(val) if isinstance(val, (list, tuple)) else val
    return result


def _build_system_prompt(
    missing_fields: set[str],
    available_categories: list[str],
) -> str:
    """Build a minimal system prompt requesting only the missing fields."""
    parts = [
        "Classify this memory content. Return a JSON object with ONLY "
        "the following fields:"
    ]

    field_instructions = []
    if "memory_type" in missing_fields:
        types = ", ".join(VALID_MEMORY_TYPES)
        field_instructions.append(
            f'"memory_type": one of [{types}]. '
            "preference=likes/dislikes/style, fact=biographical/factual, "
            "episodic=events/interactions/conclusions, "
            "procedural=workflows/habits/how-to, "
            "context=session/short-term notes"
        )
    if "categories" in missing_fields:
        cats = ", ".join(available_categories)
        field_instructions.append(
            f'"categories": list of applicable categories from [{cats}]. '
            "Use project:<name> for project-specific content. "
            "Empty list [] if no category fits."
        )
    if "importance" in missing_fields:
        levels = ", ".join(IMPORTANCE_WEIGHTS.keys())
        field_instructions.append(
            f'"importance": one of [{levels}]. '
            "low=minor details, normal=standard (default), "
            "high=important facts/decisions, critical=essential/always-relevant"
        )
    if "pinned" in missing_fields:
        field_instructions.append(
            '"pinned": boolean. true ONLY for essential identity facts, '
            "core preferences, or critical information that should always "
            "be loaded at conversation start. Most memories are false."
        )

    parts.append("\n".join(f"- {fi}" for fi in field_instructions))
    parts.append("Return ONLY valid JSON, no explanation.")
    return "\n\n".join(parts)


def _build_strict_retry_prompt(
    missing_fields: set[str],
    available_categories: list[str],
) -> str:
    """Build a stricter prompt for retry after first classification failure."""
    parts = [
        "IMPORTANT: Your previous response was invalid or empty. "
        "You MUST return valid JSON this time.",
        "Return a JSON object with EXACTLY these fields:",
    ]

    field_instructions = []
    if "memory_type" in missing_fields:
        types = ", ".join(VALID_MEMORY_TYPES)
        field_instructions.append(
            f'"memory_type": REQUIRED. Must be exactly one of: {types}'
        )
    if "categories" in missing_fields:
        cats = ", ".join(available_categories[:15])  # Limit to avoid huge prompt
        if len(available_categories) > 15:
            cats += ", ..."
        field_instructions.append(
            f'"categories": REQUIRED. Array of strings from: [{cats}]. '
            "Use [] if none apply."
        )
    if "importance" in missing_fields:
        levels = ", ".join(IMPORTANCE_WEIGHTS.keys())
        field_instructions.append(
            f'"importance": REQUIRED. Must be exactly one of: {levels}'
        )
    if "pinned" in missing_fields:
        field_instructions.append('"pinned": REQUIRED. Must be true or false')

    parts.append("\n".join(f"- {fi}" for fi in field_instructions))
    parts.append(
        "Return ONLY the JSON object. No markdown code blocks, no explanation, "
        "no additional text. Just the raw JSON."
    )
    return "\n\n".join(parts)


# Cached OpenAI client — avoids creating a new HTTP connection pool per
# classification call. Keyed by (base_url, api_key) to handle config changes.
_openai_client: OpenAI | None = None
_openai_client_key: tuple[str, str] | None = None


def _get_openai_client(llm_config: LLMConfig) -> OpenAI:
    """Get or create a cached OpenAI client from the LLM config."""
    global _openai_client, _openai_client_key
    key = (llm_config.base_url, llm_config.api_key)
    if _openai_client is None or _openai_client_key != key:
        _openai_client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        _openai_client_key = key
    return _openai_client


def classify_memory(
    content: str,
    *,
    missing_fields: set[str],
    llm_config: LLMConfig,
    available_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Classify memory metadata using an LLM call.

    Makes up to 2 LLM calls (retry on first failure with stricter prompt).
    Raises ClassificationError if both attempts fail.

    Args:
        content: The memory content to classify.
        missing_fields: Set of field names to classify
                        (memory_type, categories, importance, pinned).
        llm_config: LLM configuration (model, base_url, api_key).
        available_categories: List of valid category names including
                              dynamic project:* subcategories.

    Returns:
        Dict with classified values for the requested fields.

    Raises:
        ClassificationError: If classification fails after retry.
    """
    if not missing_fields:
        return {}

    if available_categories is None:
        available_categories = list(PREDEFINED_CATEGORIES.keys())

    # First attempt with standard prompt
    result = _attempt_classification(
        content,
        missing_fields,
        llm_config,
        available_categories,
        system_prompt=_build_system_prompt(missing_fields, available_categories),
        attempt=1,
    )
    if result is not None:
        return result

    # Retry with stricter prompt
    logger.warning(
        "Classification failed on first attempt, retrying with stricter prompt"
    )
    result = _attempt_classification(
        content,
        missing_fields,
        llm_config,
        available_categories,
        system_prompt=_build_strict_retry_prompt(missing_fields, available_categories),
        attempt=2,
    )
    if result is not None:
        return result

    # Both attempts failed
    logger.error("Classification failed after retry")
    raise ClassificationError(
        "Auto-classification failed after retry. Please provide metadata explicitly: "
        "memory_type (preference/fact/episodic/procedural/context), "
        "categories (list from predefined set or project:<name>), "
        "importance (low/normal/high/critical), "
        "pinned (true/false)."
    )


def _attempt_classification(
    content: str,
    missing_fields: set[str],
    llm_config: LLMConfig,
    available_categories: list[str],
    system_prompt: str,
    attempt: int,
) -> dict[str, Any] | None:
    """Single classification attempt. Returns None on failure."""
    try:
        client = _get_openai_client(llm_config)
        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            logger.warning("Empty classification response (attempt %d)", attempt)
            return None

        result = json.loads(raw)
        return _validate_classification(result, missing_fields)

    except json.JSONDecodeError as e:
        logger.warning(
            "Invalid JSON in classification response (attempt %d): %s", attempt, e
        )
        return None
    except Exception:
        logger.exception("Classification LLM call failed (attempt %d)", attempt)
        return None


def _validate_classification(
    result: dict[str, Any],
    missing_fields: set[str],
) -> dict[str, Any]:
    """Validate and sanitize the LLM classification response.

    Ensures all values are valid. Falls back to defaults for invalid fields.
    """
    validated: dict[str, Any] = {}

    if "memory_type" in missing_fields:
        raw_type = result.get("memory_type", DEFAULTS["memory_type"])
        try:
            validated["memory_type"] = validate_memory_type(str(raw_type))
        except ValueError:
            logger.warning(
                "Invalid classified memory_type '%s', using default", raw_type
            )
            validated["memory_type"] = DEFAULTS["memory_type"]

    if "categories" in missing_fields:
        raw_cats = result.get("categories")
        if isinstance(raw_cats, list):
            try:
                validated["categories"] = validate_categories(
                    [str(c) for c in raw_cats]
                )
            except ValueError:
                logger.warning(
                    "Invalid classified categories %s, using default", raw_cats
                )
                validated["categories"] = []
        else:
            validated["categories"] = []

    if "importance" in missing_fields:
        raw_imp = result.get("importance", DEFAULTS["importance"])
        try:
            validated["importance"] = validate_importance(str(raw_imp))
        except ValueError:
            logger.warning("Invalid classified importance '%s', using default", raw_imp)
            validated["importance"] = DEFAULTS["importance"]

    if "pinned" in missing_fields:
        raw_pinned = result.get("pinned", DEFAULTS["pinned"])
        validated["pinned"] = bool(raw_pinned)

    return validated
