"""Unit tests for mnemory Hermes plugin configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from config import load_config


class TestLoadConfig:
    """Tests for ``load_config()``."""

    def test_minimal_config(self) -> None:
        # Use clear=True to isolate from real env vars (e.g. MNEMORY_API_KEY)
        with mock.patch.dict(
            os.environ, {"MNEMORY_URL": "http://localhost:8050"}, clear=True
        ):
            cfg = load_config()
        assert cfg.url == "http://localhost:8050"
        assert cfg.api_key == ""
        assert cfg.user_id == ""
        assert cfg.agent_prefix == "hermes"
        assert cfg.auto_recall is True
        assert cfg.auto_capture is True
        assert cfg.recall_find_first is True
        assert cfg.recall_search_mode == "search"
        assert cfg.score_threshold == 0.5
        assert cfg.include_assistant is True
        assert cfg.managed is True
        assert cfg.timeout == 60.0

    def test_url_required(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MNEMORY_URL"):
                load_config()

    def test_url_trailing_slash_stripped(self) -> None:
        with mock.patch.dict(
            os.environ, {"MNEMORY_URL": "http://localhost:8050///"}, clear=False
        ):
            cfg = load_config()
        assert cfg.url == "http://localhost:8050"

    def test_all_env_vars(self) -> None:
        env = {
            "MNEMORY_URL": "https://mem.example.com",
            "MNEMORY_API_KEY": "sk-test",
            "MNEMORY_USER_ID": "alice",
            "MNEMORY_AGENT_PREFIX": "mybot",
            "MNEMORY_AUTO_RECALL": "false",
            "MNEMORY_AUTO_CAPTURE": "0",
            "MNEMORY_RECALL_FIND_FIRST": "no",
            "MNEMORY_RECALL_SEARCH_MODE": "find",
            "MNEMORY_SCORE_THRESHOLD": "0.7",
            "MNEMORY_INCLUDE_ASSISTANT": "false",
            "MNEMORY_MANAGED": "false",
            "MNEMORY_TIMEOUT": "30",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = load_config()
        assert cfg.url == "https://mem.example.com"
        assert cfg.api_key == "sk-test"
        assert cfg.user_id == "alice"
        assert cfg.agent_prefix == "mybot"
        assert cfg.auto_recall is False
        assert cfg.auto_capture is False
        assert cfg.recall_find_first is False
        assert cfg.recall_search_mode == "find"
        assert cfg.score_threshold == 0.7
        assert cfg.include_assistant is False
        assert cfg.managed is False
        assert cfg.timeout == 30.0

    def test_invalid_search_mode(self) -> None:
        env = {
            "MNEMORY_URL": "http://localhost:8050",
            "MNEMORY_RECALL_SEARCH_MODE": "invalid",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="MNEMORY_RECALL_SEARCH_MODE"):
                load_config()

    def test_score_threshold_out_of_range(self) -> None:
        env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_SCORE_THRESHOLD": "1.5"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="MNEMORY_SCORE_THRESHOLD"):
                load_config()

    def test_score_threshold_not_a_number(self) -> None:
        env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_SCORE_THRESHOLD": "abc"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="MNEMORY_SCORE_THRESHOLD"):
                load_config()

    def test_timeout_too_low(self) -> None:
        env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_TIMEOUT": "0.5"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="MNEMORY_TIMEOUT"):
                load_config()

    def test_timeout_not_a_number(self) -> None:
        env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_TIMEOUT": "slow"}
        with mock.patch.dict(os.environ, env, clear=False):
            with pytest.raises(ValueError, match="MNEMORY_TIMEOUT"):
                load_config()

    def test_bool_env_truthy_values(self) -> None:
        for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
            env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_AUTO_RECALL": val}
            with mock.patch.dict(os.environ, env, clear=False):
                cfg = load_config()
            assert cfg.auto_recall is True, f"Expected True for '{val}'"

    def test_bool_env_falsy_values(self) -> None:
        for val in ("false", "False", "0", "no", "anything"):
            env = {"MNEMORY_URL": "http://localhost:8050", "MNEMORY_AUTO_RECALL": val}
            with mock.patch.dict(os.environ, env, clear=False):
                cfg = load_config()
            assert cfg.auto_recall is False, f"Expected False for '{val}'"
