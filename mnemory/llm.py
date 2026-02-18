"""LLM client wrapper for mnemory.

Provides a thin, cached OpenAI-compatible chat completions client with
structured output support (json_schema) and automatic fallback to JSON mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import BadRequestError, OpenAI

from mnemory.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Cached OpenAI-compatible LLM client.

    Reuses a single HTTP connection pool across calls. Supports structured
    outputs (json_schema) with automatic fallback to plain JSON mode for
    providers that don't support it.
    """

    def __init__(self, config: LLMConfig):
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._supports_structured: bool | None = None
        # Tracks unsupported parameters for this model/provider.
        # Maps param name -> fix action. Populated on first BadRequestError.
        self._param_fixes: dict[str, str] = {}

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a chat completion, returning the content string.

        Args:
            messages: Chat messages (system + user).
            json_schema: If provided, attempts structured output first
                         (json_schema mode), falling back to json_object mode.
            temperature: Override default temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The raw content string from the LLM response.
        """
        temp = temperature if temperature is not None else self._temperature

        if json_schema and self._supports_structured is not False:
            try:
                result = self._call(
                    messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema,
                    },
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                self._supports_structured = True
                return result
            except Exception:
                if self._supports_structured is None:
                    logger.debug(
                        "Structured outputs not supported by provider, "
                        "falling back to JSON mode"
                    )
                    self._supports_structured = False
                else:
                    raise

        # JSON mode fallback (or no schema requested)
        response_format = {"type": "json_object"} if json_schema else None
        return self._call(
            messages,
            response_format=response_format,
            temperature=temp,
            max_tokens=max_tokens,
        )

    def _call(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Execute a single chat completion call.

        Handles parameter incompatibilities across model versions and
        providers. When a model rejects a parameter (e.g., max_tokens,
        temperature), the fix is cached so subsequent calls skip the
        unsupported parameter without any retry overhead.
        """
        params = self._build_params(messages, response_format, temperature, max_tokens)

        try:
            response = self._client.chat.completions.create(**params)
        except BadRequestError as e:
            if self._try_fix_params(e, params, max_tokens):
                response = self._client.chat.completions.create(**params)
            else:
                raise

        content = response.choices[0].message.content or ""
        return _clean_response(content)

    def _build_params(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build API call parameters, applying any cached fixes."""
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # temperature: omit if model doesn't support custom values
        if "temperature" not in self._param_fixes:
            params["temperature"] = temperature

        # max_tokens: swap to max_completion_tokens if needed
        if "max_tokens" in self._param_fixes:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        if response_format:
            params["response_format"] = response_format

        return params

    def _try_fix_params(
        self,
        error: BadRequestError,
        params: dict[str, Any],
        max_tokens: int,
    ) -> bool:
        """Try to fix params based on a BadRequestError.

        Parses the error to identify the unsupported parameter, applies
        the appropriate fix, caches it for future calls, and mutates
        params in-place for the retry.

        Returns True if a fix was applied and the call should be retried.
        """
        error_body = getattr(error, "body", None)
        if not isinstance(error_body, dict):
            return False

        param = error_body.get("param", "")
        code = error_body.get("code", "")

        if not param or code not in (
            "unsupported_parameter",
            "unsupported_value",
        ):
            return False

        # Already tried to fix this param — don't loop
        if param in self._param_fixes:
            return False

        if param == "max_tokens":
            logger.info(
                "Model %s requires max_completion_tokens instead of "
                "max_tokens — adapting",
                self._model,
            )
            self._param_fixes["max_tokens"] = "use_max_completion_tokens"
            params.pop("max_tokens", None)
            params["max_completion_tokens"] = max_tokens
            return True

        if param == "temperature":
            logger.info(
                "Model %s does not support custom temperature — omitting",
                self._model,
            )
            self._param_fixes["temperature"] = "omit"
            params.pop("temperature", None)
            return True

        # Generic: try omitting the unsupported parameter
        if param in params:
            logger.info(
                "Model %s does not support parameter '%s' — omitting",
                self._model,
                param,
            )
            self._param_fixes[param] = "omit"
            params.pop(param, None)
            return True

        return False


def _clean_response(text: str) -> str:
    """Strip markdown code fences and <think> blocks from LLM output.

    Some models (DeepSeek, etc.) wrap JSON in ```json...``` blocks or
    include <think>...</think> reasoning blocks. Strip these to get
    clean JSON.
    """
    # Remove <think>...</think> blocks (DeepSeek reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON response from the LLM, with fallback extraction.

    Tries direct JSON parse first. If that fails, attempts to extract
    JSON from within the text (e.g., surrounded by other text).

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = _clean_response(text)

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")
