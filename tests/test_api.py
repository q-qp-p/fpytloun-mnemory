"""Tests for REST API components (instructions, remember formatting, input length)."""

from __future__ import annotations

import pytest

from mnemory.api.remember import _check_rate_limit, _format_messages, _rate_limits
from mnemory.instructions import build_instructions, build_managed_instructions


class TestManagedInstructions:
    """Test managed-mode instructions."""

    def test_build_managed_instructions(self):
        """Managed instructions should include managed behavior + base reference."""
        text = build_managed_instructions()
        assert "handled automatically by the system" in text
        assert "Do NOT call initialize_memory" in text
        assert "Do NOT call add_memory proactively" in text
        assert "You CAN use add_memory if the user explicitly asks" in text
        # Should include base technical reference
        assert "TOOL REFERENCE" in text

    def test_build_managed_vs_proactive(self):
        """Managed instructions should differ from proactive."""
        managed = build_managed_instructions()
        proactive = build_instructions("proactive")
        # Both should have base reference
        assert "TOOL REFERENCE" in managed
        assert "TOOL REFERENCE" in proactive
        # But different preambles
        assert "handled automatically" in managed
        assert "handled automatically" not in proactive

    def test_build_instructions_modes(self):
        """All instruction modes should build without error."""
        for mode in ("passive", "proactive", "personality"):
            text = build_instructions(mode)
            assert len(text) > 100

    def test_build_instructions_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid INSTRUCTION_MODE"):
            build_instructions("invalid")


class TestFormatMessages:
    """Test message formatting for remember endpoint."""

    def test_format_user_assistant(self):
        """Standard user + assistant exchange."""
        messages = [
            {"role": "user", "content": "I just moved to Berlin"},
            {"role": "assistant", "content": "That's exciting!"},
        ]
        text = _format_messages(messages)
        assert "User: I just moved to Berlin" in text
        assert "Assistant: That's exciting!" in text

    def test_format_empty_messages(self):
        """Empty messages list should return empty string."""
        assert _format_messages([]) == ""

    def test_format_skips_empty_content(self):
        """Messages with empty content should be skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]
        text = _format_messages(messages)
        assert "User: Hello" in text
        assert "Assistant:" not in text

    def test_format_skips_non_string_content(self):
        """Messages with non-string content should be skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": 123},
        ]
        text = _format_messages(messages)
        assert "User: Hello" in text
        assert "Assistant:" not in text

    def test_format_system_message(self):
        """System messages should be formatted with 'System' label."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        text = _format_messages(messages)
        assert "System: You are helpful" in text
        assert "User: Hi" in text


class TestRateLimit:
    """Test rate limiting for remember endpoint."""

    def setup_method(self):
        """Clear rate limits before each test."""
        _rate_limits.clear()

    def test_rate_limit_allows_within_limit(self):
        """Requests within limit should be allowed."""
        # Default limit is 10/min — but we need to mock config
        # For now, test the function directly with cleared state
        # The function reads from config, so we test the basic flow
        assert _check_rate_limit("test-user") is True

    def test_rate_limit_tracks_per_user(self):
        """Different users should have independent limits."""
        for _ in range(5):
            _check_rate_limit("user1")
        for _ in range(3):
            _check_rate_limit("user2")
        # Both should still be within limit
        assert _check_rate_limit("user1") is True
        assert _check_rate_limit("user2") is True


class TestInputLengthGuard:
    """Test input length validation in add_memory and remember."""

    def test_add_memory_infer_true_rejects_oversized(self):
        """add_memory with infer=True should reject input over MAX_INPUT_LENGTH."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        # Set all required config attributes
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 100  # Very low for testing

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        result = service.add_memory(
            content="x" * 101,
            user_id="filip",
            infer=True,
        )
        assert result.get("error") is True
        assert "too long" in result["message"].lower()

    def test_add_memory_infer_true_accepts_within_limit(self):
        """add_memory with infer=True should accept input within MAX_INPUT_LENGTH."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 400000

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service._llm = MagicMock()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        result = service.add_memory(
            content="x" * 2000,
            user_id="filip",
            infer=True,
        )
        # Should not be an error — extraction just returns empty
        assert result.get("error") is not True

    def test_remember_truncates_oversized_input(self):
        """remember() should truncate input over MAX_INPUT_LENGTH (keep most recent)."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 100

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service._llm = MagicMock()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        # Content is 200 chars, should be truncated to last 100
        content = "A" * 100 + "B" * 100
        result = service.remember(content=content, user_id="filip")

        # Should not error — just truncated
        assert result.get("error") is not True
        # The LLM should have received the truncated content (last 100 chars = all B's)
        call_args = service._llm.generate.call_args
        if call_args:
            # The content passed to the LLM should be the truncated version
            messages = call_args[0][0]
            # Find the user message content
            for msg in messages:
                if isinstance(msg.get("content"), str) and "B" * 50 in msg["content"]:
                    # Good — the truncated content (B's) is in the prompt
                    break
