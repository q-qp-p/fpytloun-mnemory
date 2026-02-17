"""Unified memory extraction, classification, and deduplication prompts.

Combines fact extraction, per-fact classification (memory_type, categories,
importance, pinned), and deduplication against existing memories into a
single LLM call.

This replaces three separate operations:
1. Fact extraction
2. Dedup/update detection
3. Classification
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from mnemory.categories import (
    IMPORTANCE_WEIGHTS,
    PREDEFINED_CATEGORIES,
    VALID_MEMORY_TYPES,
)

logger = logging.getLogger(__name__)

# JSON schema for structured output (OpenAI json_schema mode).
# Used when the provider supports it; falls back to json_object mode otherwise.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "name": "memory_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "required": ["memories"],
        "additionalProperties": False,
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "text",
                        "action",
                        "target_id",
                        "old_memory",
                        "memory_type",
                        "categories",
                        "importance",
                        "pinned",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The extracted fact text",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["ADD", "UPDATE", "DELETE", "NONE"],
                        },
                        "target_id": {
                            "type": ["string", "null"],
                            "description": (
                                "ID of existing memory for UPDATE/DELETE, "
                                "null for ADD/NONE"
                            ),
                        },
                        "old_memory": {
                            "type": ["string", "null"],
                            "description": (
                                "Previous text of the memory being updated, "
                                "null unless action is UPDATE"
                            ),
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": list(VALID_MEMORY_TYPES),
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "importance": {
                            "type": "string",
                            "enum": list(IMPORTANCE_WEIGHTS.keys()),
                        },
                        "pinned": {"type": "boolean"},
                    },
                },
            },
        },
    },
}


# ── Retry prompt for oversized facts ─────────────────────────────────

_SHORTEN_SYSTEM_PROMPT = """\
You are a memory manager. A previously extracted memory fact is too long.
Rewrite it more concisely or split it into multiple shorter facts.
Preserve ALL important information — do not lose detail.

Each fact must be under {max_length} characters.
Keep the same JSON schema as the original.

Return a JSON object with a "memories" array using the same format:
text, action, target_id, old_memory, memory_type, categories, importance, pinned.

Return ONLY the JSON object. No explanation, no markdown."""


def build_shorten_prompt(
    oversized_action: dict[str, Any],
    *,
    max_memory_length: int = 1000,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build a prompt to shorten or split an oversized extracted fact.

    Args:
        oversized_action: The action dict with text exceeding max length.
        max_memory_length: Maximum character length for each fact.

    Returns:
        Tuple of (messages, json_schema) for the LLM call.
    """
    system_prompt = _SHORTEN_SYSTEM_PROMPT.format(max_length=max_memory_length)

    user_content = json.dumps(
        {"memories": [oversized_action]},
        indent=2,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return messages, EXTRACTION_SCHEMA


# ── Prompt templates ─────────────────────────────────────────────────

_USER_SYSTEM_PROMPT = """\
You are a memory manager for an AI assistant. Your job is to:
1. Extract distinct facts from the user's input
2. Classify each fact
3. Compare against existing memories and decide what to do

## Fact Extraction Rules

- Extract facts ONLY from the user's messages. IGNORE assistant/system messages.
- Each fact should be a single, atomic piece of information.
- Write facts in third person, always including the subject
  explicitly (e.g., "User prefers dark mode",
  "User's partner likes hiking").
- Preserve all important information — do not over-compress
  at the cost of losing detail.
- Each fact must be under {max_length} characters. If content
  is too detailed for a single fact, split into multiple facts.
- Detect the language of the input and record facts in the
  same language.
- If no relevant facts can be extracted, return an empty list.
- Today's date is {today}.

### Examples

Input: "Hi, how are you?"
Output: {{"memories": []}}

Input: "My name is John and I'm a software engineer at Google"
Output: {{"memories": [
  {{"text": "User's name is John", "action": "ADD",
    "target_id": null, "old_memory": null,
    "memory_type": "fact", "categories": ["personal"],
    "importance": "normal", "pinned": true}},
  {{"text": "User is a software engineer at Google",
    "action": "ADD", "target_id": null, "old_memory": null,
    "memory_type": "fact", "categories": ["work"],
    "importance": "normal", "pinned": true}}
]}}

Input: "I switched from VS Code to Neovim last week"
Output: {{"memories": [
  {{"text": "User uses Neovim as primary editor",
    "action": "UPDATE", "target_id": "0",
    "old_memory": "User uses VS Code as primary editor",
    "memory_type": "preference",
    "categories": ["technical"],
    "importance": "normal", "pinned": false}}
]}}

## Classification Rules

For each extracted fact, classify:

- **memory_type**: {memory_types}
  - preference = likes, dislikes, style choices
  - fact = biographical, factual information
  - episodic = events, interactions, conclusions
  - procedural = workflows, habits, how-to
  - context = session/short-term notes

- **categories**: Pick from the available list below. Use [] if none fit.
  Use project:<name> for project-specific content.

- **importance**: {importance_levels}
  - low = minor details, temporary notes
  - normal = standard memories (default for most)
  - high = important facts, key decisions
  - critical = essential, always-relevant information

- **pinned**: true ONLY for essential identity facts (name, job, location),
  core preferences, or critical information that should always be loaded
  at conversation start. Most memories should be false.

{categories_section}

## Deduplication Rules

Compare each extracted fact against the existing memories below.

- **ADD**: New information not present in existing memories. Use target_id=null.
- **UPDATE**: Modifies, enriches, or replaces an existing memory. Set target_id to the existing memory's ID and old_memory to its current text. The text field should contain the NEW, updated content.
- **DELETE**: Contradicts an existing memory that should be removed. Set target_id to the existing memory's ID. The text field should contain the memory being deleted.
- **NONE**: Already captured in existing memories. Skip it (do not include in output).

### Subject preservation

- Only UPDATE when the new fact is about the SAME subject
  as the existing memory.
- "User's partner likes dogs" must NOT update
  "User does not like dogs" — different subjects.
- "User moved to Berlin" CAN update "User lives in Prague"
  — same subject (user's location).
- When in doubt, prefer ADD over UPDATE.

When updating, keep the same meaning but incorporate new
information. When facts overlap, merge them into a single
updated memory.

{existing_section}

## Output Format

Return a JSON object with a "memories" array. Each entry
must have ALL fields: text, action, target_id, old_memory,
memory_type, categories, importance, pinned.

Return ONLY the JSON object. No explanation, no markdown."""

_AGENT_SYSTEM_PROMPT = """\
You are a memory manager for an AI assistant. Your job is to:
1. Extract distinct facts about the assistant from its messages
2. Classify each fact
3. Compare against existing memories and decide what to do

## Fact Extraction Rules

- Extract facts ONLY from the assistant's messages. IGNORE user/system messages.
- Focus on the assistant's: personality traits, preferences, capabilities,
  knowledge areas, communication style, and self-descriptions.
- Write facts in third person, always including the subject
  explicitly (e.g., "Assistant prefers concise responses",
  "Assistant is expert in Python").
- Preserve all important information — do not over-compress
  at the cost of losing detail.
- Each fact must be under {max_length} characters. If content
  is too detailed for a single fact, split into multiple facts.
- Detect the language of the input and record facts in the
  same language.
- If no relevant facts can be extracted, return an empty list.
- Today's date is {today}.

### Examples

Input: "assistant: I prefer to give concise, direct answers."
Output: {{"memories": [
  {{"text": "Assistant prefers concise, direct answers",
    "action": "ADD", "target_id": null, "old_memory": null,
    "memory_type": "preference",
    "categories": ["preferences"],
    "importance": "normal", "pinned": true}}
]}}

Input: "assistant: I researched Kubernetes networking and \
concluded Cilium is the best CNI for our use case."
Output: {{"memories": [
  {{"text": "Assistant researched Kubernetes networking, \
concluded Cilium is the best CNI",
    "action": "ADD", "target_id": null, "old_memory": null,
    "memory_type": "episodic", "categories": ["technical"],
    "importance": "high", "pinned": false}}
]}}

## Classification Rules

For each extracted fact, classify:

- **memory_type**: {memory_types}
  - preference = assistant's likes, dislikes, style choices
  - fact = assistant's identity, name, capabilities
  - episodic = research conclusions, interaction outcomes
  - procedural = assistant's workflows, approaches
  - context = session/short-term notes

- **categories**: Pick from the available list below. Use [] if none fit.
  Use project:<name> for project-specific content.

- **importance**: {importance_levels}
  - low = minor details
  - normal = standard memories (default for most)
  - high = important knowledge, key conclusions
  - critical = core identity, always-relevant

- **pinned**: true for core identity facts (name, personality traits),
  key capabilities, and critical knowledge. false for most memories.

{categories_section}

## Deduplication Rules

Compare each extracted fact against the existing memories below.

- **ADD**: New information not present in existing memories. Use target_id=null.
- **UPDATE**: Modifies, enriches, or replaces an existing memory. Set target_id to the existing memory's ID and old_memory to its current text. The text field should contain the NEW, updated content.
- **DELETE**: Contradicts an existing memory that should be removed. Set target_id to the existing memory's ID. The text field should contain the memory being deleted.
- **NONE**: Already captured in existing memories. Skip it (do not include in output).

### Subject preservation

- Only UPDATE when the new fact is about the SAME subject
  as the existing memory.
- "Assistant learned to use Helm" must NOT update
  "Assistant is expert in Kubernetes" — different subjects.
- "Assistant now prefers brief responses" CAN update
  "Assistant prefers verbose responses" — same subject.
- When in doubt, prefer ADD over UPDATE.

When updating, keep the same meaning but incorporate new
information. When facts overlap, merge them into a single
updated memory.

{existing_section}

## Output Format

Return a JSON object with a "memories" array. Each entry
must have ALL fields: text, action, target_id, old_memory,
memory_type, categories, importance, pinned.

Return ONLY the JSON object. No explanation, no markdown."""


# ── Prompt builders ──────────────────────────────────────────────────


def build_extraction_prompt(
    content: str,
    *,
    role: str = "user",
    existing_memories: list[dict[str, Any]] | None = None,
    available_categories: list[str] | None = None,
    explicit_fields: dict[str, Any] | None = None,
    max_memory_length: int = 1000,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, str]]:
    """Build the unified extraction+classification+dedup prompt.

    Args:
        content: The raw content to process.
        role: "user" or "assistant" — determines which extraction prompt to use.
        existing_memories: Similar existing memories from vector search.
            Each dict should have "id" and "text" keys, and optionally
            "type" and "categories" for richer LLM context.
        available_categories: List of valid category names including
            dynamic project:* subcategories.
        explicit_fields: Fields explicitly provided by the caller that
            should NOT be classified by the LLM. The prompt will instruct
            the LLM to use these exact values.
        max_memory_length: Maximum character length for each extracted fact.
            Communicated to the LLM in the prompt.

    Returns:
        Tuple of (messages, json_schema, id_mapping) for the LLM call.
        messages: List of chat messages (system + user).
        json_schema: The structured output schema dict.
        id_mapping: Dict mapping integer IDs to real UUIDs.
    """
    if available_categories is None:
        available_categories = list(PREDEFINED_CATEGORIES.keys())

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_types = ", ".join(VALID_MEMORY_TYPES)
    importance_levels = ", ".join(IMPORTANCE_WEIGHTS.keys())

    # Build categories section
    cats_str = ", ".join(available_categories)
    categories_section = f"**Available categories**: [{cats_str}]"

    # Build existing memories section with integer ID mapping
    id_mapping: dict[str, str] = {}  # "0" -> real_uuid
    if existing_memories:
        mapped = []
        for idx, mem in enumerate(existing_memories):
            str_idx = str(idx)
            id_mapping[str_idx] = mem["id"]
            entry: dict[str, Any] = {"id": str_idx, "text": mem["text"]}
            if mem.get("type"):
                entry["type"] = mem["type"]
            if mem.get("categories"):
                entry["categories"] = mem["categories"]
            mapped.append(entry)

        existing_json = json.dumps(mapped, indent=2)
        existing_section = (
            f"**Existing memories** (compare against these):\n```\n{existing_json}\n```"
        )
    else:
        existing_section = (
            "**Existing memories**: None yet. "
            "All extracted facts should use action ADD."
        )

    # Build explicit fields instruction
    explicit_note = ""
    if explicit_fields:
        parts = []
        for field_name, value in explicit_fields.items():
            parts.append(
                f'- Set "{field_name}" to {json.dumps(value)} for ALL memories'
            )
        explicit_note = (
            "\n\n## Caller-Provided Values\n"
            "The following fields have been explicitly set. "
            "Use these exact values:\n" + "\n".join(parts)
        )

    # Select template
    template = _AGENT_SYSTEM_PROMPT if role == "assistant" else _USER_SYSTEM_PROMPT

    system_prompt = template.format(
        today=today,
        max_length=max_memory_length,
        memory_types=memory_types,
        importance_levels=importance_levels,
        categories_section=categories_section,
        existing_section=existing_section,
    )

    if explicit_note:
        system_prompt += explicit_note

    # Build user message
    if role == "assistant":
        user_content = f"assistant: {content}"
    else:
        user_content = f"user: {content}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return messages, EXTRACTION_SCHEMA, id_mapping


def parse_extraction_response(
    response_text: str,
    id_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Parse and validate the LLM's extraction response.

    Maps integer IDs back to real UUIDs and validates each memory entry.

    Args:
        response_text: Raw JSON string from the LLM.
        id_mapping: Mapping from integer IDs ("0", "1") to real UUIDs.

    Returns:
        List of validated memory action dicts, each with:
        - text: str
        - action: "ADD" | "UPDATE" | "DELETE"
        - target_id: str | None (real UUID for UPDATE/DELETE)
        - old_memory: str | None
        - memory_type: str
        - categories: list[str]
        - importance: str
        - pinned: bool

        NONE actions are filtered out. Invalid entries are skipped with warnings.
    """
    from mnemory.llm import parse_json_response

    try:
        data = parse_json_response(response_text)
    except ValueError:
        logger.warning("Failed to parse extraction response, returning empty list")
        return []

    raw_memories = data.get("memories", [])
    if not isinstance(raw_memories, list):
        logger.warning("'memories' is not a list in extraction response")
        return []

    results = []
    for entry in raw_memories:
        if not isinstance(entry, dict):
            continue

        action = entry.get("action", "").upper()
        text = entry.get("text", "").strip()

        # Skip NONE actions and empty text
        if action == "NONE" or not text:
            continue

        # Validate action
        if action not in ("ADD", "UPDATE", "DELETE"):
            logger.warning(
                "Invalid action '%s' in extraction response, skipping", action
            )
            continue

        # Map target_id back to real UUID
        target_id = None
        if action in ("UPDATE", "DELETE"):
            raw_target = entry.get("target_id")
            if raw_target is not None:
                raw_target = str(raw_target)
                target_id = id_mapping.get(raw_target)
                if target_id is None:
                    logger.warning(
                        "Unknown target_id '%s' in extraction response, "
                        "skipping %s action",
                        raw_target,
                        action,
                    )
                    continue
            else:
                logger.warning("%s action without target_id, skipping", action)
                continue

        # Validate and sanitize classification fields
        memory_type = _validate_memory_type(entry.get("memory_type"))
        categories = _validate_categories(entry.get("categories"))
        importance = _validate_importance(entry.get("importance"))
        pinned = bool(entry.get("pinned", False))

        results.append(
            {
                "text": text,
                "action": action,
                "target_id": target_id,
                "old_memory": entry.get("old_memory"),
                "memory_type": memory_type,
                "categories": categories,
                "importance": importance,
                "pinned": pinned,
            }
        )

    return results


# ── Classification-only prompt (for infer=False path) ────────────────

_CLASSIFY_SYSTEM_PROMPT = """\
Classify this memory content. Return a JSON object with ONLY the following fields:

{field_instructions}

Return ONLY valid JSON, no explanation.

{categories_section}"""


def build_classification_prompt(
    content: str,
    *,
    missing_fields: set[str],
    available_categories: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Build a classification-only prompt for the infer=False path.

    Only classifies the fields in missing_fields. Returns messages and
    an optional json_schema (None if no fields need classification).

    Args:
        content: The memory content to classify.
        missing_fields: Set of field names to classify
                        (memory_type, categories, importance, pinned).
        available_categories: List of valid category names.

    Returns:
        Tuple of (messages, json_schema).
    """
    if not missing_fields:
        return [], None

    if available_categories is None:
        available_categories = list(PREDEFINED_CATEGORIES.keys())

    field_instructions = []
    schema_props: dict[str, Any] = {}
    required: list[str] = []

    if "memory_type" in missing_fields:
        types = ", ".join(VALID_MEMORY_TYPES)
        field_instructions.append(
            f'"memory_type": one of [{types}]. '
            "preference=likes/dislikes/style, fact=biographical/factual, "
            "episodic=events/interactions/conclusions, "
            "procedural=workflows/habits/how-to, "
            "context=session/short-term notes"
        )
        schema_props["memory_type"] = {
            "type": "string",
            "enum": list(VALID_MEMORY_TYPES),
        }
        required.append("memory_type")

    if "categories" in missing_fields:
        cats = ", ".join(available_categories)
        field_instructions.append(
            f'"categories": list of applicable categories from [{cats}]. '
            "Use project:<name> for project-specific content. "
            "Empty list [] if no category fits."
        )
        schema_props["categories"] = {
            "type": "array",
            "items": {"type": "string"},
        }
        required.append("categories")

    if "importance" in missing_fields:
        levels = ", ".join(IMPORTANCE_WEIGHTS.keys())
        field_instructions.append(
            f'"importance": one of [{levels}]. '
            "low=minor details, normal=standard (default), "
            "high=important facts/decisions, critical=essential/always-relevant"
        )
        schema_props["importance"] = {
            "type": "string",
            "enum": list(IMPORTANCE_WEIGHTS.keys()),
        }
        required.append("importance")

    if "pinned" in missing_fields:
        field_instructions.append(
            '"pinned": boolean. true ONLY for essential identity facts, '
            "core preferences, or critical information that should always "
            "be loaded at conversation start. Most memories are false."
        )
        schema_props["pinned"] = {"type": "boolean"}
        required.append("pinned")

    cats_str = ", ".join(available_categories)
    categories_section = f"Available categories: [{cats_str}]"

    system_prompt = _CLASSIFY_SYSTEM_PROMPT.format(
        field_instructions="\n".join(f"- {fi}" for fi in field_instructions),
        categories_section=categories_section,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    json_schema = {
        "name": "memory_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "required": required,
            "additionalProperties": False,
            "properties": schema_props,
        },
    }

    return messages, json_schema


# ── Validation helpers ───────────────────────────────────────────────

# Defaults for invalid/missing classification values
_DEFAULT_MEMORY_TYPE = "fact"
_DEFAULT_IMPORTANCE = "normal"


def _validate_memory_type(value: Any) -> str:
    """Validate memory_type, returning default on invalid input."""
    if isinstance(value, str) and value in VALID_MEMORY_TYPES:
        return value
    if value is not None:
        logger.debug("Invalid memory_type '%s', using default", value)
    return _DEFAULT_MEMORY_TYPE


def _validate_categories(value: Any) -> list[str]:
    """Validate categories list, returning empty list on invalid input."""
    if not isinstance(value, list):
        return []
    result = []
    for cat in value:
        if isinstance(cat, str) and cat.strip():
            result.append(cat.strip())
    return result


def _validate_importance(value: Any) -> str:
    """Validate importance level, returning default on invalid input."""
    if isinstance(value, str) and value in IMPORTANCE_WEIGHTS:
        return value
    if value is not None:
        logger.debug("Invalid importance '%s', using default", value)
    return _DEFAULT_IMPORTANCE
