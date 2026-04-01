"""
Hermes lifecycle hooks for auto-recall and auto-capture.

Hook mapping:
  on_session_start  -> pre-fetch instructions + core memories (threaded)
  pre_llm_call      -> per-turn search + inject context
  post_llm_call     -> extract last exchange and send to /api/remember
  on_session_end    -> clean up session state
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .client import MnemoryClient
from .config import MnemoryConfig
from .helpers import SessionStore, build_system_text, extract_last_exchange

logger = logging.getLogger("hermes_mnemory.hooks")

# Timeout (seconds) for waiting on the init-recall background thread.
INIT_RECALL_WAIT_TIMEOUT = 8.0


def create_hooks(
    client: MnemoryClient,
    config: MnemoryConfig,
    sessions: SessionStore,
) -> dict[str, Any]:
    """Return a dict of ``{hook_name: callback}`` ready for registration.

    Only hooks whose corresponding feature is enabled are included.
    """
    hooks: dict[str, Any] = {}

    agent_id = config.agent_prefix or None

    # ------------------------------------------------------------------
    # on_session_start — pre-fetch instructions + core memories
    # ------------------------------------------------------------------
    if config.auto_recall:

        def _on_session_start(session_id: str, **kwargs: Any) -> None:
            state = sessions.get_or_create(session_id)

            def _init_recall() -> None:
                try:
                    result = client.recall(
                        include_instructions=True,
                        managed=config.managed,
                        score_threshold=config.score_threshold,
                        agent_id=agent_id,
                    )
                    if result:
                        state.mnemory_session_id = result.get("session_id")
                        state.init_result = result
                        logger.debug(
                            "mnemory: init recall complete (session=%s)",
                            state.mnemory_session_id,
                        )
                except Exception:
                    logger.warning("mnemory: init recall failed", exc_info=True)
                finally:
                    state.init_event.set()

            thread = threading.Thread(target=_init_recall, daemon=True)
            thread.start()

        hooks["on_session_start"] = _on_session_start

    # ------------------------------------------------------------------
    # pre_llm_call — per-turn search + inject context
    # ------------------------------------------------------------------
    if config.auto_recall:

        def _on_pre_llm_call(
            session_id: str,
            user_message: str,
            conversation_history: list[dict[str, Any]],
            is_first_turn: bool,
            **kwargs: Any,
        ) -> dict[str, str] | None:
            state = sessions.get_or_create(session_id)

            # Compaction detection: if history shrank, the context was
            # compressed.  Reset the mnemory session so memories can be
            # re-discovered (the server tracks known_ids per session).
            current_len = len(conversation_history)
            if state.last_history_len > 0 and current_len < state.last_history_len:
                logger.info(
                    "mnemory: compaction detected (history %d -> %d), resetting session",
                    state.last_history_len,
                    current_len,
                )
                state.mnemory_session_id = None
                state.init_result = None
                state.init_event.clear()
                state.turn_count = 0
                state.last_search_results = None

                # Re-run init recall synchronously after compaction
                try:
                    result = client.recall(
                        include_instructions=True,
                        managed=config.managed,
                        score_threshold=config.score_threshold,
                        agent_id=agent_id,
                    )
                    if result:
                        state.mnemory_session_id = result.get("session_id")
                        state.init_result = result
                except Exception:
                    logger.warning(
                        "mnemory: post-compaction init recall failed", exc_info=True
                    )
                finally:
                    state.init_event.set()

            # Wait for init recall if it hasn't completed yet
            if not state.init_event.is_set():
                state.init_event.wait(timeout=INIT_RECALL_WAIT_TIMEOUT)

            # If init recall never ran (e.g. on_session_start didn't fire),
            # do a combined init + search now.
            if state.init_result is None:
                logger.debug("mnemory: init result missing, running combined recall")
                try:
                    result = client.recall(
                        session_id=state.mnemory_session_id,
                        query=user_message,
                        include_instructions=True,
                        managed=config.managed,
                        search_mode=_search_mode(state, config),
                        score_threshold=config.score_threshold,
                        agent_id=agent_id,
                    )
                    if result:
                        state.mnemory_session_id = result.get("session_id")
                        state.init_result = result
                        state.last_search_results = result.get("search_results")
                        state.turn_count += 1
                except Exception:
                    logger.warning("mnemory: combined recall failed", exc_info=True)

                state.last_history_len = current_len
                return _build_context(state)

            # Per-turn search with the user's message as query
            try:
                result = client.recall(
                    session_id=state.mnemory_session_id,
                    query=user_message,
                    search_mode=_search_mode(state, config),
                    score_threshold=config.score_threshold,
                    agent_id=agent_id,
                )
                if result:
                    # Update session ID in case it was assigned on first call
                    if not state.mnemory_session_id:
                        state.mnemory_session_id = result.get("session_id")
                    state.last_search_results = result.get("search_results")
                    state.turn_count += 1
            except Exception:
                logger.warning("mnemory: per-turn recall failed", exc_info=True)

            state.last_history_len = current_len
            return _build_context(state)

        hooks["pre_llm_call"] = _on_pre_llm_call

    # ------------------------------------------------------------------
    # post_llm_call — auto-capture
    # ------------------------------------------------------------------
    if config.auto_capture:

        def _on_post_llm_call(
            session_id: str,
            user_message: str,
            assistant_response: str,
            conversation_history: list[dict[str, Any]],
            **kwargs: Any,
        ) -> None:
            state = sessions.get_or_create(session_id)

            exchange = extract_last_exchange(
                conversation_history,
                after_index=max(0, state.last_history_len - 1)
                if state.last_history_len > 0
                else 0,
                include_assistant=config.include_assistant,
            )
            if not exchange:
                return

            messages: list[dict[str, str]] = [
                {"role": "user", "content": exchange["user"]}
            ]
            if exchange.get("assistant"):
                messages.append({"role": "assistant", "content": exchange["assistant"]})

            # Fire-and-forget
            try:
                client.remember(
                    session_id=state.mnemory_session_id,
                    messages=messages,
                    labels={"source": "hermes"},
                    agent_id=agent_id,
                )
            except Exception:
                logger.warning("mnemory: remember failed", exc_info=True)

        hooks["post_llm_call"] = _on_post_llm_call

    # ------------------------------------------------------------------
    # on_session_end — cleanup
    # ------------------------------------------------------------------

    def _on_session_end(session_id: str, **kwargs: Any) -> None:
        sessions.remove(session_id)

    hooks["on_session_end"] = _on_session_end

    return hooks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _search_mode(state: Any, config: MnemoryConfig) -> str:
    """Choose the search mode based on turn count and config."""
    if state.turn_count == 0 and config.recall_find_first:
        return "find"
    return config.recall_search_mode


def _build_context(state: Any) -> dict[str, str] | None:
    """Build the context dict to return from ``pre_llm_call``."""
    # Merge init result with per-turn search results
    merged: dict[str, Any] = {}
    if state.init_result:
        merged.update(state.init_result)
    if state.last_search_results is not None:
        merged["search_results"] = state.last_search_results

    text = build_system_text(merged)
    if not text:
        return None
    return {"context": text}
