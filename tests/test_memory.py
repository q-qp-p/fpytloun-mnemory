"""Tests for mnemory.memory module — pure logic that doesn't require backends."""

import pytest

from mnemory.memory import MemoryService, _validate_id

# ── _validate_id ──────────────────────────────────────────────────────


class TestValidateId:
    def test_valid_id(self):
        assert _validate_id("filip", "user_id") == "filip"

    def test_strips_whitespace(self):
        assert _validate_id("  filip  ", "user_id") == "filip"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="user_id must not be empty"):
            _validate_id("", "user_id")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="user_id must not be empty"):
            _validate_id("   ", "user_id")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="user_id too long"):
            _validate_id("a" * 257, "user_id")

    def test_max_length_ok(self):
        result = _validate_id("a" * 256, "user_id")
        assert len(result) == 256


# ── _rerank_by_importance ─────────────────────────────────────────────


class TestRerankByImportance:
    """Test the reranking logic without needing a full MemoryService."""

    @staticmethod
    def _rerank(memories):
        """Call the static-like rerank method via the class."""
        # Access the unbound method directly
        return MemoryService._rerank_by_importance(None, memories)

    def test_critical_boosted_over_normal(self):
        memories = [
            {"score": 0.8, "metadata": {"importance": "normal"}},
            {"score": 0.7, "metadata": {"importance": "critical"}},
        ]
        result = self._rerank(memories)
        # critical: 0.7*0.7 + 1.0*0.3 = 0.79
        # normal:   0.8*0.7 + 0.4*0.3 = 0.68
        assert result[0]["metadata"]["importance"] == "critical"

    def test_high_similarity_wins_over_low_importance(self):
        memories = [
            {"score": 0.3, "metadata": {"importance": "critical"}},
            {"score": 0.95, "metadata": {"importance": "normal"}},
        ]
        result = self._rerank(memories)
        # critical: 0.3*0.7 + 1.0*0.3 = 0.51
        # normal:   0.95*0.7 + 0.4*0.3 = 0.785
        assert result[0]["metadata"]["importance"] == "normal"

    def test_combined_score_not_in_output(self):
        memories = [{"score": 0.5, "metadata": {"importance": "normal"}}]
        result = self._rerank(memories)
        assert "_combined_score" not in result[0]

    def test_missing_importance_defaults_to_normal(self):
        memories = [
            {"score": 0.5, "metadata": {}},
            {"score": 0.5, "metadata": {"importance": "high"}},
        ]
        result = self._rerank(memories)
        # high: 0.5*0.7 + 0.7*0.3 = 0.56
        # default (normal): 0.5*0.7 + 0.4*0.3 = 0.47
        assert result[0]["metadata"]["importance"] == "high"

    def test_empty_list(self):
        assert self._rerank([]) == []

    def test_all_same_score_sorted_by_importance(self):
        memories = [
            {"score": 0.5, "metadata": {"importance": "low"}},
            {"score": 0.5, "metadata": {"importance": "critical"}},
            {"score": 0.5, "metadata": {"importance": "normal"}},
            {"score": 0.5, "metadata": {"importance": "high"}},
        ]
        result = self._rerank(memories)
        importances = [m["metadata"]["importance"] for m in result]
        assert importances == ["critical", "high", "normal", "low"]
