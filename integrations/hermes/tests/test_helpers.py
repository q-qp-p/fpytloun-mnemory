"""Unit tests for mnemory Hermes plugin helpers."""

from __future__ import annotations

from helpers import (
    SessionStore,
    build_system_text,
    escape_for_prompt,
    extract_last_exchange,
)


class TestEscapeForPrompt:
    """Tests for ``escape_for_prompt()``."""

    def test_no_special_chars(self) -> None:
        assert escape_for_prompt("hello world") == "hello world"

    def test_html_entities(self) -> None:
        assert escape_for_prompt('<script>alert("xss")</script>') == (
            "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;"
        )

    def test_ampersand(self) -> None:
        assert escape_for_prompt("a & b") == "a &amp; b"

    def test_single_quotes(self) -> None:
        assert escape_for_prompt("it's") == "it&#x27;s"

    def test_empty_string(self) -> None:
        assert escape_for_prompt("") == ""

    def test_mixed(self) -> None:
        assert escape_for_prompt("a<b>c&d'e\"f") == "a&lt;b&gt;c&amp;d&#x27;e&quot;f"


class TestBuildSystemText:
    """Tests for ``build_system_text()``."""

    def test_empty_result(self) -> None:
        assert build_system_text({}) == ""

    def test_instructions_only(self) -> None:
        result = build_system_text({"instructions": "Be helpful."})
        assert result == "Be helpful."

    def test_core_memories_only(self) -> None:
        result = build_system_text({"core_memories": "User likes Python."})
        assert result == "User likes Python."

    def test_search_results_only(self) -> None:
        result = build_system_text(
            {
                "search_results": [
                    {"memory": "User lives in Prague", "id": "1"},
                    {"memory": "User prefers dark mode", "id": "2"},
                ],
            }
        )
        assert "## Recalled Memories" in result
        assert "- User lives in Prague" in result
        assert "- User prefers dark mode" in result

    def test_full_result(self) -> None:
        result = build_system_text(
            {
                "instructions": "Instructions here.",
                "core_memories": "Core memories here.",
                "search_results": [{"memory": "A fact", "id": "1"}],
            }
        )
        parts = result.split("\n\n")
        assert len(parts) == 3
        assert parts[0] == "Instructions here."
        assert parts[1] == "Core memories here."
        assert "## Recalled Memories" in parts[2]

    def test_empty_search_results_skipped(self) -> None:
        result = build_system_text(
            {
                "instructions": "Hello",
                "search_results": [
                    {"memory": "", "id": "1"},
                    {"memory": "  ", "id": "2"},
                ],
            }
        )
        assert "Recalled Memories" not in result

    def test_html_escaping_in_search_results(self) -> None:
        result = build_system_text(
            {
                "search_results": [{"memory": "<script>alert(1)</script>", "id": "1"}],
            }
        )
        assert "&lt;script&gt;" in result
        assert "<script>" not in result


class TestSessionStore:
    """Tests for ``SessionStore``."""

    def test_get_or_create_new(self) -> None:
        store = SessionStore(max_sessions=10)
        state = store.get_or_create("s1")
        assert state.mnemory_session_id is None
        assert state.turn_count == 0

    def test_get_or_create_existing(self) -> None:
        store = SessionStore(max_sessions=10)
        state1 = store.get_or_create("s1")
        state1.turn_count = 5
        state2 = store.get_or_create("s1")
        assert state2.turn_count == 5
        assert state1 is state2

    def test_remove(self) -> None:
        store = SessionStore(max_sessions=10)
        state1 = store.get_or_create("s1")
        state1.turn_count = 5
        store.remove("s1")
        state2 = store.get_or_create("s1")
        assert state2.turn_count == 0

    def test_remove_nonexistent(self) -> None:
        store = SessionStore(max_sessions=10)
        store.remove("nonexistent")  # Should not raise

    def test_fifo_eviction(self) -> None:
        store = SessionStore(max_sessions=3)
        store.get_or_create("s1").turn_count = 1
        store.get_or_create("s2").turn_count = 2
        store.get_or_create("s3").turn_count = 3
        # Adding s4 should evict s1 (oldest)
        store.get_or_create("s4").turn_count = 4
        # s3 and s4 should still exist
        assert store.get_or_create("s3").turn_count == 3
        assert store.get_or_create("s4").turn_count == 4

    def test_fifo_eviction_order(self) -> None:
        store = SessionStore(max_sessions=2)
        store.get_or_create("s1").turn_count = 1
        store.get_or_create("s2").turn_count = 2
        # Adding s3 evicts s1
        store.get_or_create("s3").turn_count = 3
        # s1 was evicted, should be fresh
        assert store.get_or_create("s1").turn_count == 0


class TestExtractLastExchange:
    """Tests for ``extract_last_exchange()``."""

    def test_basic_exchange(self) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = extract_last_exchange(history, after_index=0, include_assistant=True)
        assert result is not None
        assert result["user"] == "Hello"
        assert result["assistant"] == "Hi there!"
        assert result["new_count"] == 2

    def test_user_only(self) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = extract_last_exchange(history, after_index=0, include_assistant=False)
        assert result is not None
        assert result["user"] == "Hello"
        assert result["assistant"] is None

    def test_no_new_messages(self) -> None:
        history = [{"role": "user", "content": "Hello"}]
        result = extract_last_exchange(history, after_index=1, include_assistant=True)
        assert result is None

    def test_empty_history(self) -> None:
        result = extract_last_exchange([], after_index=0, include_assistant=True)
        assert result is None

    def test_after_index_slicing(self) -> None:
        history = [
            {"role": "user", "content": "Old message"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "user", "content": "New message"},
            {"role": "assistant", "content": "New reply"},
        ]
        result = extract_last_exchange(history, after_index=2, include_assistant=True)
        assert result is not None
        assert result["user"] == "New message"
        assert result["assistant"] == "New reply"

    def test_takes_last_user_message(self) -> None:
        history = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = extract_last_exchange(history, after_index=0, include_assistant=True)
        assert result is not None
        assert result["user"] == "Second"

    def test_list_content_blocks(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First block"},
                    {"type": "text", "text": "Second block"},
                ],
            },
        ]
        result = extract_last_exchange(history, after_index=0, include_assistant=True)
        assert result is not None
        # Takes the last text block
        assert result["user"] == "Second block"

    def test_skips_empty_content(self) -> None:
        history = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real message"},
        ]
        result = extract_last_exchange(history, after_index=0, include_assistant=True)
        assert result is not None
        assert result["user"] == "Real message"

    def test_no_user_message(self) -> None:
        history = [{"role": "assistant", "content": "I said something"}]
        result = extract_last_exchange(history, after_index=0, include_assistant=True)
        assert result is None
